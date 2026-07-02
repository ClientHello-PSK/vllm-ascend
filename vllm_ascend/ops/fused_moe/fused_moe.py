#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group, tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner  # type: ignore

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_utils import init_eplb_config
from vllm_ascend.flash_common3_context import get_flash_common3_context, set_flash_common3_context
from vllm_ascend.ops.fused_moe.experts_selector import select_experts, zero_experts_compute
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, FusedExpertsResult, setup_moe_comm_method
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.methods.base import get_moe_num_logical_experts
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_NZ,
    enable_sp,
    maybe_trans_nz,
    npu_stream_switch,
    shared_expert_dp_enabled,
    shared_experts_calculation_stream,
)


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    global_indices = torch.where(expert_map != -1)[0]
    local_indices = expert_map[global_indices]
    return ", ".join(
        f"{local_index.item()}->{global_index.item()}"
        for local_index, global_index in zip(local_indices, global_indices)
    )


@dataclass
class FusedMoEResult:
    routed_out: torch.Tensor
    before_dispatch_evt: torch.npu.Event | None = None
    before_gmm2_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    swiglu_limit: float = 0.0


@dataclass
class FusedMoEEvents:
    before_routed_experts: torch.npu.Event
    after_routed_experts: torch.npu.Event | None = field(default=None)
    before_dispatch: torch.npu.Event | None = field(default=None)
    before_gmm2: torch.npu.Event | None = field(default=None)
    before_combine: torch.npu.Event | None = field(default=None)
    swiglu_limit: float = 0.0


def mock_false():
    return False


def mock_true():
    return True


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig = None, tid2eid=None):
        super().__init__(moe=moe)
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb
        self.tid2eid = tid2eid

    @property
    def is_monolithic(self) -> bool:
        return False

    def maybe_make_prepare_finalize(self, routing_tables=None):
        # Ascend uses its own MoE communication and forward_impl path.
        # Do not let upstream modular-kernel initialization replace it.
        return None

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)

        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

        # TODO: Current dispatch_ffn_combine fusion operator ONLY supports NZ format.
        # Therefore, we must cast weights to NZ when fusion is enabled.
        # Once the underlying dispatch_ffn_combine operator is updated to support
        # ND format (or other formats), remove this specific 'if' check and the forced
        # npu_format_cast. At that point, the operator should be able to handle weights
        # in their native format without explicit casting here.
        if get_ascend_config().enable_fused_mc2:
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
        else:
            layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
            layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: torch.Tensor | None = None,
        mc2_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)
        input_ids = getattr(get_forward_context(), "input_ids", None)
        num_shared_experts = getattr(layer, "n_shared_experts", 0)
        if num_shared_experts is None:
            num_shared_experts = 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_logical_experts,
            tid2eid=self.tid2eid,
            input_ids=input_ids,
        )
        if layer.vllm_config.model_config is not None and layer.vllm_config.model_config.enable_return_routed_experts:
            capturer = getattr(layer, "_ascend_routed_experts_capturer", None)
            if capturer is not None:
                capturer.capture(layer_id=layer.layer_id, topk_ids=topk_ids)

        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=num_logical_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=x,
            )

        topk_weights = topk_weights.to(x.dtype)
        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        # NOTE: In the MoECommType.FUSED_MC2 branch, we wrap weights (w1, w2) into lists
        # and provide dummy scales (w1_scale, w2_scale). This is required because:
        # The underlying Ascend fused operator (e.g., dispatch_ffn_combine) expects
        # inputs in a list format.
        # TODO: Passing an empty tensor as scale for float (BF16) cases is semantically
        # incorrect. The ideal solution is to pass None. However, if the underlying
        # dispatch_ffn_combine C++ operator does not support None for the scale argument
        # (due to signature constraints), we are forced to use a placeholder empty tensor.
        # This TODO tracks the requirement to update the C++ operator to accept Optional[Tensor]
        # or None for scales in non-quantized scenarios.
        if _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2:
            w1 = [layer.w13_weight]
            w1_scale = [torch.tensor([], dtype=torch.int64)]
            w2 = [layer.w2_weight]
            w2_scale = [torch.tensor([], dtype=torch.int64)]
            w1_scale_bias = [torch.tensor([], dtype=torch.float32)]
            w2_scale_bias = [torch.tensor([], dtype=torch.float32)]
        else:
            w1 = layer.w13_weight
            w1_scale = None
            w2 = layer.w2_weight
            w2_scale = None
            w1_scale_bias = None
            w2_scale_bias = None

        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                w1_bias=layer.w13_bias if self.moe.has_bias else None,
                w2_bias=layer.w2_bias if self.moe.has_bias else None,
                quant_type=QuantType.NONE,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                swiglu_limit=layer.swiglu_limit,
            )
        )
        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result
        return final_hidden_states


class AscendMoERunner(MoERunner):
    @property
    def use_dp_chunking(self) -> bool:
        """Ascend uses its own forward_impl path, not the FlashInfer Cutlass
        chunked path. Always return False to stay on forward_impl."""
        return False

    @property
    def _fused_output_is_reduced(self) -> bool:
        # For MC2/ALLTOALL/FUSED_MC2 comm types, finalize() already includes
        # TP all-reduce for the routed output, and _forward_shared_experts
        # handles it for the shared output. Signal this to the upstream
        # MoERunner.forward() so _maybe_reduce_final_output does not apply a
        # second TP all-reduce (which would double-count the contributions).
        moe_comm_type = _EXTRA_CTX.moe_comm_type
        return moe_comm_type in {
            MoECommType.ALLTOALL,
            MoECommType.MC2,
            MoECommType.FUSED_MC2,
        } or (moe_comm_type == MoECommType.ALLGATHER and _EXTRA_CTX.flash_comm_v1_enabled)

    def _maybe_reduce_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # _forward_shared_experts already handles shared expert TP all-reduce
        # for MC2/ALLTOALL/FUSED_MC2. For AllGather the reduction is done
        # via _maybe_reduce_final_output on the combined (shared + routed)
        # output. Skip any additional reduction here.
        return shared_output

    def _maybe_reduce_final_output(
        self,
        states: torch.Tensor,
        trunc_size: int,
    ) -> torch.Tensor:
        states = torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(states)
        return states[..., :trunc_size]

    # TODO: Remove this after drop v0.19.1 support
    def forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Override the default forward_impl to use Ascend-specific implementation.
        This delegates to the layer's forward_impl method which contains the
        Ascend-specific MoE computation logic.
        """
        if self.shared_experts is None:
            result = layer.forward_impl(hidden_states, router_logits)
            # If the layer has shared experts, forward_impl returns a tuple (shared_out, routed_out)
            # Otherwise, it returns just routed_out
            # The torch op expects the same return type based on whether it's moe_forward or moe_forward_shared
        else:
            result = layer.shared_forward_impl(hidden_states, router_logits)
        return result

    def _forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        with self._sequence_parallel_context():
            return self.forward_impl(
                layer,
                hidden_states,
                router_logits,
                shared_experts_input,
            )


class AscendFusedMoE(FusedMoE):
    moe_counter = -1
    gate_stream: torch.npu.Stream | None = None

    def __init__(self, *args, **kwargs):
        # Save original routed_scaling_factor before super().__init__ modifies it.
        # When apply_routed_scale_to_output=True, vLLM sets self.routed_scaling_factor
        # to 1.0 and expects the runner to apply scaling to output. But vllm-ascend
        # uses its own forward path, so we need the original value.
        _ = kwargs.pop("hash") if "hash" in kwargs else None
        tid2eid = kwargs.pop("tid2eid") if "tid2eid" in kwargs else None

        self._original_routed_scaling_factor = kwargs.get("routed_scaling_factor", 1.0)
        super().__init__(*args, **kwargs)
        self.use_overlapped = True
        self._routed_input_transform = kwargs.get("routed_input_transform")
        self._shared_experts = kwargs.get("shared_experts")
        self.shared_expert_stream = None
        has_shared_experts = self._shared_experts is not None
        num_experts = kwargs["num_experts"]
        intermediate_size = kwargs["intermediate_size"]
        num_shared_experts = kwargs.get("n_shared_experts", 0)

        AscendFusedMoE.moe_counter += 1
        self.moe_instance_id = AscendFusedMoE.moe_counter

        self._expert_map = None
        self.log2phy = None

        self.tid2eid = tid2eid

        if self.quant_config is None:
            self.quant_method = AscendUnquantizedFusedMoEMethod(self.moe_config, tid2eid=self.tid2eid)
        else:
            self.quant_method = self.quant_config.get_quant_method(self, self.layer_name, tid2eid=self.tid2eid)

        assert self.quant_method is not None
        # Keep base_quant_method in sync with the swapped-in Ascend method,
        # otherwise FusedMoE.maybe_init_modular_kernel (called via the V2
        # model runner's prepare_communication_buffer_for_model) would dispatch
        # to the upstream UnquantizedFusedMoEMethod.maybe_make_prepare_finalize,
        # which raises by design.
        self.base_quant_method = self.quant_method

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        if self.moe_config.ep_size > 1:
            self.moe_config.ep_group = get_ep_group()
            self.moe_config.mc2_group = get_mc2_group()
        self.moe_config.supports_eplb = self.quant_method.supports_eplb
        ascend_config = get_ascend_config()
        self.multistream_overlap_shared_expert = ascend_config.multistream_overlap_shared_expert and has_shared_experts
        self.shared_multistream_overlap_gate = ascend_config.multistream_overlap_gate and has_shared_experts
        if self.multistream_overlap_shared_expert:
            logger.info_once("[fused_moe/layer] Multistream overlap shared expert is enabled.")
        if enable_sp() and has_shared_experts:
            logger.info_once(
                "[fused_moe/layer] Sequence parallelism is enabled, shared experts are replicated for best performance."
            )

        # flashcommon3 gate stream
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
        if self.multistream_overlap_gate and AscendFusedMoE.gate_stream is None:
            AscendFusedMoE.gate_stream = torch.npu.Stream()
        if self.multistream_overlap_gate:
            logger.info_once("[fused_moe/layer] Multistream overlap gate is enabled.")
        vllm_config = get_current_vllm_config()
        if (
            self.custom_routing_function is None
            and self.e_score_correction_bias is not None
            and not vllm_config.model_config.is_deepseek_mla
        ):
            self.e_score_correction_bias.data = self.e_score_correction_bias.data.to(
                dtype=vllm_config.model_config.dtype
            )
        self._gate = kwargs.get("gate")

        # init moe
        eplb_config = ascend_config.eplb_config
        self.mix_placement = getattr(ascend_config, "mix_placement", False)
        self.n_shared_experts = num_shared_experts
        num_experts += num_shared_experts if self.mix_placement else 0
        self.moe_config.num_experts = num_experts
        self.global_expert_map, self._expert_map, self.log2phy, self.global_redundant_expert_num = init_eplb_config(
            eplb_config,
            self.moe_instance_id,
            self.moe_config,
            self.mix_placement,
            num_shared_experts,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
        )
        self.global_num_experts = num_experts + self.global_redundant_expert_num
        self.dynamic_eplb = eplb_config.dynamic_eplb and (self.log2phy is not None)
        self.local_num_experts = self.global_num_experts // self.ep_size
        self.expert_map_manager._local_num_experts = self.local_num_experts
        self.expert_map_manager._expert_map = self._expert_map
        if self._expert_map is not None:
            logger.info_once(
                "[fused_moe/layer] Expert parallelism is enabled."
                " ep_rank=%s/%s, local_num_experts=%s, global_num_experts=%s,"
                " expert_map=%s",
                self.ep_rank,
                self.ep_size,
                self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self._expert_map),
            )
        if self.dynamic_eplb:
            self.multi_stage = False
            self.moe_load = torch.zeros(self.local_num_experts, dtype=torch.int64).npu()
            if eplb_config.eplb_policy_type == 3:
                self.multi_stage = True
                self.load_counter = torch.tensor(0, dtype=torch.int32, device="npu")
                self.num_iter = eplb_config.expert_heat_collection_interval
                self.moe_load = torch.zeros((self.num_iter, self.local_num_experts), dtype=torch.int32, device="npu")

        self.moe_config.num_experts = self.global_num_experts
        self.moe_config.num_local_experts = self.local_num_experts
        self.moe_config.global_redundant_expert_num = self.global_redundant_expert_num
        self.swiglu_limit = getattr(self.vllm_config.model_config.hf_config, "swiglu_limit", 0)

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": self.hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": self.params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if self.quant_method.__class__.__name__ in ("GPTQMarlinMoEMethod", "CompressedTensorsWNA16MoEMethod"):
            moe_quant_params["intermediate_size_full"] = intermediate_size
        self.quant_method.create_weights(layer=self, **moe_quant_params)

        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        self.enable_npugraph_ex_static_kernel = ascend_config.ascend_compilation_config.enable_static_kernel

        setup_moe_comm_method(self.moe_config)
        self.quant_type = self._get_quant_type()

        self.runner = AscendMoERunner(
            self.layer_name,
            self.moe_config,
            self.router,
            self._routed_input_transform,
            kwargs.pop("gate", None),
            kwargs.pop("shared_experts", None),
            self.quant_method,
            self.vllm_config.parallel_config.enable_dbo,
        )

        if self.multistream_overlap_shared_expert:
            # Wrap the quant_method's process_weights_after_loading to validate that
            # splitting shared expert computation (gate_up projection + activation,
            # then down projection) yields identical results to integrated
            # computation after weight loading.
            original_process_weights = self.quant_method.process_weights_after_loading

            @wraps(original_process_weights)
            def wrapped_process_weights(*args, **kwargs):
                result = original_process_weights(*args, **kwargs)
                self._validate_shared_expert_consistency()
                return result

            self.quant_method.process_weights_after_loading = wrapped_process_weights  # type: ignore

        # Register this MoE layer with EPLB for PP compatibility.
        # PPMissingLayer (nn.Identity) never calls AscendFusedMoE.__init__,
        # so only real MoE layers on this rank are registered.
        VllmEplbAdaptor.register_layer(self)

    def _validate_shared_expert_consistency(self):
        """Validate that split shared expert computation matches integrated
        computation."""
        test_input = (
            torch.rand(10, self.hidden_size, device="npu", dtype=self.moe_config.in_dtype) * 2 - 1
        )  # Random input for testing, scoped to [-1, 1]

        assert self._shared_experts is not None
        integrated_out = self._shared_experts(test_input)
        part1_out = self._shared_experts_part1(test_input)
        split_out = self._shared_experts_part2(test_input, part1_out)

        if not torch.allclose(integrated_out, split_out):
            diff = (integrated_out - split_out).abs()
            logger.error(
                "[fused_moe/layer] Shared expert split computation validation failed."
                " The split-path computation does not match the integrated-path result."
                " max_abs_diff=%s, integrated_sum=%s, integrated_norm=%s,"
                " split_sum=%s, split_norm=%s, hidden_size=%s, dtype=%s.",
                diff.max().item(),
                integrated_out.sum().item(),
                integrated_out.norm().item(),
                split_out.sum().item(),
                split_out.norm().item(),
                self.hidden_size,
                self.moe_config.in_dtype,
            )
            raise ValueError("FusedMoE shared experts split computation does not match the integrated computation.")
        logger.info_once(
            "[fused_moe/layer] Shared expert split computation validation passed."
            " Integrated and split-path results are consistent."
        )

    def _shared_experts_part1(self, hidden_states: torch.Tensor):
        shared_gate_up, _ = self._shared_experts.gate_up_proj(hidden_states)  # type: ignore
        return shared_gate_up

    def _shared_experts_part2(self, hidden_states: torch.Tensor, shared_gate_up: torch.Tensor):
        shared_act = self._shared_experts.act_fn(shared_gate_up)  # type: ignore
        shared_out, _ = self._shared_experts.down_proj(shared_act)  # type: ignore

        # Qwen3-Next specific gating mechanism
        assert self._shared_experts is not None
        if hasattr(self._shared_experts, "expert_gate") and self._shared_experts.expert_gate is not None:
            gate_out, _ = self._shared_experts.expert_gate(hidden_states)  # type: ignore
            shared_out = F.sigmoid(gate_out) * shared_out
        return shared_out

    def _get_quant_type(self) -> QuantType:
        quant_type = QuantType.NONE
        method = getattr(self.quant_method, "quant_method", None)

        if method is not None:
            quant_type = getattr(method, "quant_type", QuantType.NONE)

        return quant_type

    def update_expert_map(self, new_expert_map):
        self._expert_map = new_expert_map

    def get_log2phy_map(self):
        return self.log2phy

    def clear_moe_load(self):
        if self.moe_load is not None:
            self.moe_load.zero_()
        if self.multi_stage:
            self.load_counter.zero_()

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """NOTE(Yizhou): This is to override the parent class method. In `mc2commimpl`,
        and `alltoallcommimpl`, we do not need to all-reduce the final outputs since
        the outputs are already aggregated across tensor parallel ranks in the
        `finalize` function. In `allgathercommimpl`, we still need to all-reduce the
        outputs since each rank only has partial outputs.
        """
        return torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(final_hidden_states)

    @property
    def gate(self) -> torch.nn.Module | None:
        return self._gate if self.use_overlapped else None

    @property
    def is_internal_router(self) -> bool:
        gate = self.gate
        return gate is not None and hasattr(gate, "weight_fp32")

    @property
    def use_dp_chunking(self) -> bool:
        """This func routes to the chunked forward path using the FlashInfer Cutlass kernel
        only when data parallelism (DP) is enabled. Thus just returning False in vllm-ascend
        """
        return False

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self.ensure_moe_quant_config_init()
        return self.runner.forward(
            hidden_states,
            router_logits,
        )

    def forward_impl(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor, return_with_event: bool = False
    ) -> torch.Tensor | FusedMoEResult:
        # ============================================================
        # Ascend MoE 层的核心实现。被 AscendMoERunner.forward_impl 委托调用。
        # 完整职责：路由打分 → 选 top-k 专家 → 分组 GEMM(专家FFN) → 通信 combine/reduce。
        # 整体分为几个阶段：① 层索引/负载均衡预处理 ② gate stream 重叠(路由+共享专家)
        #                  ③ 通信 prepare(dispatch前准备) ④ 核心 GEMM(专家FFN)
        #                  ⑤ EPLB 负载热度收集 ⑥ 通信 finalize(combine+reduce) ⑦ 返回
        # ============================================================
        assert self.quant_method is not None

        forward_context = get_forward_context()
        # When static kernels are enabled, the forward pass runs twice (compilation + capture),
        # causing moe_layer_index to overflow. Wrap the index to prevent out-of-bounds errors.
        # —— 静态 kernel(NPU graph)模式下，forward 会跑两遍(编译期+捕获期)，
        #    moe_layer_index 会一直自增导致越界；这里对层数取模回绕，防止下标越界。
        if self.enable_npugraph_ex_static_kernel and forward_context.all_moe_layers:
            moe_layer_index = forward_context.moe_layer_index % (len(forward_context.all_moe_layers))
            forward_context.moe_layer_index = moe_layer_index

        # Load balancing for token distribution among experts in dummy_run
        # TODO: The community only considers load balancing when DP > 1.
        # This approach may overlook some extreme scenarios.
        # —— 是否强制做"专家间 token 均匀分布"。仅在 profile/dummy_run 阶段开启，
        #    目的是让编译器在各专家负载均衡的假设下编译/捕获图，避免极端不均导致形状失配。
        enable_force_load_balance = _EXTRA_CTX.in_profile_run

        forward_context = get_forward_context()
        # —— ② gate stream 重叠：把"路由打分 + 共享专家计算"放到独立的 gate_stream 上，
        #    与主 stream 后续的分组 GEMM 并行重叠，隐藏 gate/共享专家的延迟。
        #    gate_stream 上算出的 shared_out 与 topk 通过全局 flash_common3_context 交给主 stream。
        if self.multistream_overlap_gate:
            # gate_stream：AscendFusedMoE 类级共享的一条独立 NPU 流，专门跑 gate/共享专家。
            assert AscendFusedMoE.gate_stream is not None
            # fc3_context：全局跨 stream 中转对象，承载 shared_out / topk / shared_experts 模块。
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            # gate_stream 先等主 stream 把前置依赖(hidden_states 等)算完，避免读到未就绪数据。
            AscendFusedMoE.gate_stream.wait_stream(torch.npu.current_stream())
            # 切到 gate_stream 上执行下面这段：路由打分 + 共享专家，与主 stream 的 GEMM 并行。
            with npu_stream_switch(AscendFusedMoE.gate_stream, enabled=self.multistream_overlap_gate):
                # —— 共享专家计算：所有 token 都走的那个 FFN(DeepseekV2MLP)，不参与路由。
                # share_expert
                assert fc3_context.shared_experts is not None
                shared_out = fc3_context.shared_experts(hidden_states)
                # NOTE: This is exactly the opposite of `maybe_all_reduce_tensor_model_parallel`
                # —— MC2/AlltoAll/FUSED_MC2 通信模式下，共享专家输出在多卡上是分片的，
                #    需要一次 all_reduce 拼成完整结果；AllGather 模式则由后面 finalize 统一 reduce。
                #    shared_expert_dp_enabled() 为真时(共享专家走 DP)各卡独立，不 reduce。
                moe_comm_type = _EXTRA_CTX.moe_comm_type
                if (
                    moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
                    and not shared_expert_dp_enabled()
                ):
                    shared_out = tensor_model_parallel_all_reduce(shared_out)
                # 把算好的 shared_out 存进中转上下文，主 stream 取来与 routed_out 相加。
                set_flash_common3_context(shared_out=shared_out)
                # —— 路由打分：根据 router_logits 选出每个 token 的 top-k 个专家。
                input_ids = getattr(get_forward_context(), "input_ids", None)
                topk_weights, topk_ids = select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    top_k=self.top_k,                          # 每个 token 选几个专家(如6)
                    use_grouped_topk=self.use_grouped_topk,    # 是否启用分组topk(DeepSeek的grouped专家选择)
                    renormalize=self.renormalize,              # 是否对topk权重做归一化(norm_topk_prob)
                    topk_group=self.topk_group,                # 分组选择时每组选几个
                    num_expert_group=self.num_expert_group,    # 专家分多少组
                    custom_routing_function=self.custom_routing_function,  # 自定义路由函数(可空)
                    scoring_func=self.scoring_func,            # 打分函数(softmax/sigmoid/sqrtsoftplus等)
                    routed_scaling_factor=self._original_routed_scaling_factor,  # 路由缩放因子(Flash=1.5/Pro=2.5)
                    e_score_correction_bias=self.e_score_correction_bias,  # noaux_tc 偏置修正项
                    num_experts=self.moe_config.num_experts,   # 路由专家总数(Flash=256/Pro=384)
                    input_ids=input_ids,                       # 用于 hash 路由(前几层 hash 层用)
                    tid2eid=self.tid2eid,                      # token-id→专家 映射(hash路由用)
                )
                # 返回：topk_weights (num_tokens, top_k) 归一化后的权重；topk_ids (num_tokens, top_k) 选中的专家编号。

                # AllGather 通信模式：topk 是按 TP 分片算的，需要跨卡 gather 成完整结果。
                if isinstance(_EXTRA_CTX.moe_comm_method, AllGatherCommImpl):
                    topk_weights = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(topk_weights, True, True)
                    topk_ids = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(topk_ids, True, True)

                # 把 topk 也存进中转上下文，主 stream 的 GEMM 靠它分发 token 到对应专家。
                set_flash_common3_context(topk_weights=topk_weights, topk_ids=topk_ids)

        # —— ③ 通信 prepare：分发(dispatch)前的准备工作。
        #    moe_comm_method 按 MoE 通信方式(AllGather / MC2 / AlltoAll)不同有不同实现，
        #    负责对 hidden_states/router_logits 做按专家分发、padding 对齐、生成量化所需掩码等。
        prepare_output = _EXTRA_CTX.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=_EXTRA_CTX.flash_comm_v1_enabled,   # flash comm v1：用 all_gather 替代 all_reduce
            enable_shared_expert_dp=self.enable_shared_expert_dp,  # 共享专家是否走数据并行
            quant_type=self.quant_type,                            # 量化类型(fp8/int8/w8a8等)，影响缩放生成
        )
        hidden_states = prepare_output.hidden_states                      # 处理后的输入(可能已分发/pad)
        router_logits = prepare_output.router_logits                      # 处理后的路由打分
        mc2_mask = prepare_output.mc2_mask                                # MC2 通信掩码：标记哪些 token 要参与通信
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape  # pad 后的形状，finalize 时用于裁回
        pertoken_scale = prepare_output.pertoken_scale                    # per-token 量化缩放(量化推理用)

        # Make sure the default stream waits for the gate stream to finish.
        # —— ④ 主 stream 等待 gate_stream：GEMM 要用到 gate_stream 上算出的 topk，必须先等它完成。
        if self.multistream_overlap_gate:
            torch.npu.current_stream().wait_stream(AscendFusedMoE.gate_stream)

        # —— ⑤ 核心 GEMM：分组矩阵乘，即"专家 FFN"的真正执行。
        #    按 topk_ids 把 token 分发到对应专家，跑 gate_up_proj → 激活 → down_proj(SwiGLU)，
        #    再按 topk_weights 加权合并。量化方法(quant_method)决定具体 kernel(fp8/int8/bf16)。
        # Matrix multiply.
        fused_experts_results: FusedExpertsResult = self.quant_method.apply(
            layer=self,                                            # FusedMoE 层自身(携带 w1/w2 权重)
            x=hidden_states,                                       # 输入 token 特征
            router_logits=router_logits,                           # 路由打分(部分实现内部再算topk)
            pertoken_scale=pertoken_scale,                         # per-token 量化缩放
            top_k=self.top_k,                                      # 每token选专家数
            renormalize=self.renormalize,                          # topk权重是否归一化
            use_grouped_topk=self.use_grouped_topk,                # 分组topk开关
            num_experts=self.moe_config.num_experts,               # 路由专家总数
            expert_map=self._expert_map,                           # EPLB: 逻辑专家→物理卡重映射表
            topk_group=self.topk_group,                            # 分组每组选几个
            num_expert_group=self.num_expert_group,                # 专家分组数
            custom_routing_function=self.custom_routing_function,  # 自定义路由(可空)
            scoring_func=self.scoring_func,                        # 打分函数
            routed_scaling_factor=self._original_routed_scaling_factor,  # 路由缩放因子
            e_score_correction_bias=self.e_score_correction_bias,  # noaux_tc 偏置修正
            activation=self.activation,                            # 激活函数(默认silu)
            apply_router_weight_on_input=self.apply_router_weight_on_input,  # 是否把路由权重提前乘到输入(省一次乘)
            enable_force_load_balance=enable_force_load_balance,   # 强制负载均衡(编译/profile期)
            log2phy=self.log2phy,                                  # EPLB: 逻辑专家→物理专家索引
            global_redundant_expert_num=self.global_redundant_expert_num,  # EPLB: 冗余专家数
            mc2_mask=mc2_mask,                                     # MC2 通信掩码
        )
        # 返回 FusedExpertsResult：含 routed_out(路由专家输出)、event(各阶段同步事件)、
        # expert_tokens/group_list_type(EPLB 热度统计用)、swiglu_limit 等。

        # —— ⑥ EPLB(Expert-Level Load Balance) 负载热度收集。
        #    动态专家负载均衡：统计每个专家分到的 token 数(负载热度)，攒到 moe_load 表里，
        #    后续按热度把"热专家"重映射复制到更多卡上，消除 MoE 的负载不均。
        if self.dynamic_eplb and _EXTRA_CTX.eplb_heat_collection_status:
            expert_tokens = fused_experts_results.expert_tokens       # 每个专家的 token 数统计
            group_list_type = fused_experts_results.group_list_type   # 统计值的组织形式
            assert expert_tokens is not None and group_list_type is not None, (
                "expert_tokens and group_list_type should not be None when dynamic_eplb is enabled."
            )
            # group_list_type==1：直接是每专家 token 数；否则是前缀和(cumsum)形式，需差分还原成增量。
            local_load = (
                expert_tokens
                if group_list_type == 1
                else torch.cat([expert_tokens[:1], expert_tokens[1:] - expert_tokens[:-1]])
            )
            if self.multi_stage:
                # 多阶段统计：按迭代轮转写入 moe_load 的不同行(滑动窗口)，取多步平均更稳。
                cur_iter = torch.remainder(self.load_counter, self.num_iter)
                self.moe_load.index_add_(
                    dim=0, index=cur_iter, source=local_load.to(torch.int32, non_blocking=True).view(1, -1)
                )
                self.load_counter.add_(1)
            else:
                self.moe_load.add_(local_load)

        # —— ⑦ 通信 finalize：combine(把分到各专家的 token 结果汇合) + reduce。
        #    AllGather 模式需要在此 reduce；MC2/AlltoAll 模式内部已 reduce(reduce_results=False)。
        routed_out = _EXTRA_CTX.moe_comm_method.finalize(
            hidden_states=fused_experts_results.routed_out,
            reduce_results=isinstance(_EXTRA_CTX.moe_comm_method, AllGatherCommImpl),
            padded_hidden_states_shape=padded_hidden_states_shape,   # 按 prepare 时记的 pad 形状裁回原长
        )

        # —— ⑧ 返回：是否带同步事件(event)。
        #    带 event：返回 FusedMoEResult，含 GMM 各阶段(dispatch/gmm2/combine)的 event，
        #              供上层做多 stream 调度/重叠时精确同步。
        if return_with_event:
            return FusedMoEResult(
                routed_out=routed_out,
                before_dispatch_evt=fused_experts_results.before_dispatch_evt,
                before_gmm2_evt=fused_experts_results.before_gmm2_evt,
                before_combine_evt=fused_experts_results.before_combine_evt,
                swiglu_limit=fused_experts_results.swiglu_limit,
            )
        else:
            # The vLLM FusedMoE forward_impl does not return events.
            # —— 不带 event：直接返回 routed_out，与上游 vLLM forward_impl 返回类型一致。
            return routed_out

    def _forward_shared_experts(self, hidden_states: torch.Tensor, fused_moe_evts: FusedMoEEvents):
        if self._shared_experts is None:
            return None

        def maybe_wait_event(evt: torch.npu.Event | None):
            if evt is not None:
                torch.npu.current_stream().wait_event(evt)

        with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap_shared_expert):
            # Only used for int quantization
            has_quantized_shared = hasattr(self._shared_experts.gate_up_proj, "weight_scale") and hasattr(
                self._shared_experts.down_proj, "weight_scale"
            )
            if has_quantized_shared and self.quant_type in (QuantType.W8A8, QuantType.W4A8):
                original_dtype = hidden_states.dtype
                # Execute dynamic quant concurrently with MoE gate.
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.after_routed_experts)
                hidden_states = torch_npu.npu_quant_matmul(
                    quantized_x,
                    self._shared_experts.gate_up_proj.weight,
                    self._shared_experts.gate_up_proj.weight_scale,
                    pertoken_scale=None,
                    bias=None,
                    output_dtype=torch.int32,
                )
                # Execute activation concurrently with gmm2.

                maybe_wait_event(fused_moe_evts.before_gmm2)
                quantized_x, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(
                    x=hidden_states,
                    weight_scale=self._shared_experts.gate_up_proj.weight_scale_fp32,
                    activation_scale=pertoken_scale,
                    bias=None,
                    quant_scale=None,
                    quant_offset=None,
                    group_index=None,
                    activate_left=True,
                    quant_mode=1,
                    swiglu_mode=1,
                    clamp_limit=fused_moe_evts.swiglu_limit,
                )
                # Execute the down projection concurrently with the combine
                # communication.
                maybe_wait_event(fused_moe_evts.before_combine)
                shared_out = torch_npu.npu_quant_matmul(
                    quantized_x,
                    self._shared_experts.down_proj.weight,
                    self._shared_experts.down_proj.weight_scale,
                    pertoken_scale=swiglu_out_scale,
                    bias=None,
                    output_dtype=original_dtype,
                )
            elif has_quantized_shared and self.quant_type == QuantType.W4A8MXFP:
                original_dtype = hidden_states.dtype
                # Execute dynamic quant concurrently with MoE gate.
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                    hidden_states, dst_type=torch.float8_e4m3fn
                )
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.before_dispatch)
                hidden_states = self._shared_experts.gate_up_proj((quantized_x, pertoken_scale))[0]
                # Execute activation concurrently with gmm2.
                maybe_wait_event(fused_moe_evts.before_gmm2)
                quantized_x, swiglu_out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                    hidden_states,
                    topk_weight=None,
                    group_index=None,
                    dst_type=torch.float8_e4m3fn,
                    quant_mode=2,
                    clamp_value=fused_moe_evts.swiglu_limit,
                )
                # Execute the down projection concurrently with the combine
                # communication.
                maybe_wait_event(fused_moe_evts.before_combine)
                shared_out = self._shared_experts.down_proj((quantized_x, swiglu_out_scale))[0]
            else:
                # Ensure the shared experts wait for hidden_states to be ready.
                torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.before_dispatch)
                part1_out = self._shared_experts_part1(hidden_states)
                # Execute the down projection concurrently with the combine
                # communication.
                maybe_wait_event(fused_moe_evts.before_combine)
                shared_out = self._shared_experts_part2(hidden_states, part1_out)

        # Make sure the default stream waits for the shared experts stream to
        # finish.
        if self.multistream_overlap_shared_expert:
            torch.npu.current_stream().wait_stream(shared_experts_calculation_stream())

        # NOTE: This is exactly the opposite of
        # `maybe_all_reduce_tensor_model_parallel`
        moe_comm_type = _EXTRA_CTX.moe_comm_type
        if (
            moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
            and not shared_expert_dp_enabled()
        ):
            shared_out = tensor_model_parallel_all_reduce(shared_out)
        return shared_out

    def shared_forward_impl(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ):
        # ============================================================
        # 带"共享专家"的 MoE 前向。被 AscendMoERunner.forward_impl 委托调用
        # (当本层有共享专家时)。职责：算出 路由专家输出 routed_out + 共享专家输出 shared_out，
        # 以元组 (shared_out, routed_out) 返回 —— 对应 deepseek_v4.py 里 fused_moe_out_is_tuple=True 分支。
        # 整体阶段：① 注册共享专家到全局上下文 ② gate 路由打分 ③ 路由专家 GEMM→routed_out
        #          ④ 无共享专家提前返回 ⑤ 算共享专家 shared_out ⑥ 返回 (shared_out, routed_out)
        # ============================================================
        # —— ① 若开启"共享专家与 gate 的多 stream 重叠"模式，先把共享专家模块注册进
        #    全局 flash_common3_context，供 forward_impl 内的 gate_stream 取用(在那里并行算共享专家)。
        if self.shared_multistream_overlap_gate:
            set_flash_common3_context(shared_experts=self._shared_experts)

        # —— ② gate 路由打分：决定每个 token 走哪些路由专家。
        #    is_internal_router=True(V4 走此路径)：gate 计算收进本类内部，用 fp32 权重算 router_logits。
        #    is_internal_router=False：外层(如 deepseek_v4.py 的 else 分支)已算好 router_logits 传入，这里直接用。
        if self.is_internal_router:
            gate = self.gate
            assert gate is not None
            # NOTE(Angazenn): To make this cast explicitly, the hbm usage might
            # increase with extra hidden states. We also assume that all gate
            # linear is unquantized so that we the weight is pre-casted in
            # process_weights_after_loading of AscendUnquantizedLinearMethod.
            # —— 转 fp32 做高精度路由：代价是多一份 fp32 拷贝增加 HBM 占用；
            #    前提是 gate 未量化、其权重在加载后已预先 cast 成 fp32(gate.weight_fp32)。
            hidden_states_fp32 = hidden_states.float()
            # 在当前 stream 打两个时序 event，夹住 gate 计算，供后面共享专家做多 stream 重叠同步。
            before_routed_experts = torch.npu.current_stream().record_event()
            # 核心路由打分：(num_tokens,hidden)@(hidden,n_experts) → (num_tokens,n_experts)，
            # 每个 token 对每个专家的偏好分(内积)。
            router_logits = F.linear(hidden_states_fp32, gate.weight_fp32)
            after_routed_experts = torch.npu.current_stream().record_event()
        else:
            # 外层已算好 router_logits：这里只在进入路由专家前打一个起始 event，无需计时(after=None)。
            before_routed_experts = torch.npu.current_stream().record_event()
            after_routed_experts = None

        # —— ③ 路由专家 GEMM：分组矩阵乘(专家 FFN)。return_with_event=True 带回各阶段时序 event，
        #    供后面共享专家做多 stream 重叠同步。详见 forward_impl 内部流程。
        fused_moe_results = self.forward_impl(
            hidden_states=hidden_states,
            router_logits=router_logits,
            return_with_event=True,
        )
        routed_out = fused_moe_results.routed_out   # 路由专家的加权融合输出

        # —— ④ 无共享专家：直接返回 routed_out(单个 tensor，非元组)。
        if self._shared_experts is None:
            return routed_out

        # —— ⑤ 算共享专家输出 shared_out。两条路径：
        #    A) shared_multistream_overlap_gate=True：共享专家已在 forward_impl 的 gate_stream 上
        #       并行算好(见 forward_impl ② 阶段)，直接从全局上下文取现成的 shared_out。
        #    B) 否则：同步调用 _forward_shared_experts 计算，并把路由专家的时序 event 传入，
        #       支撑"共享专家 ↔ 路由专家"的多 stream 重叠同步(详见 _forward_shared_experts)。
        if self.shared_multistream_overlap_gate:
            # 路径 A：取 gate_stream 上已算好的共享专家输出。
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            shared_out = fc3_context.shared_out
        else:
            # 路径 B：同步算共享专家。FusedMoEEvents 打包路由专家各阶段 event：
            #   before/after_routed_experts — gate 路由阶段的起止点；
            #   before_dispatch/before_gmm2/before_combine — 路由专家 GMM 各阶段(dispatch/down_proj/合并)之前；
            #   swiglu_limit — swiglu 限幅阈值(精度保护，防激活值爆炸)。
            #   共享专家流靠这些 event 与路由专家流做精确时序对齐，实现并行重叠。
            shared_out = self._forward_shared_experts(
                hidden_states,
                FusedMoEEvents(
                    after_routed_experts=after_routed_experts,
                    before_routed_experts=before_routed_experts,
                    before_dispatch=fused_moe_results.before_dispatch_evt,
                    before_gmm2=fused_moe_results.before_gmm2_evt,
                    before_combine=fused_moe_results.before_combine_evt,
                    swiglu_limit=fused_moe_results.swiglu_limit,
                ),
            )
        # —— ⑥ 返回元组 (shared_out, routed_out)。上层(deepseek_v4.py)会做
        #    final = routed_out * routed_scaling_factor + shared_out 合并。
        return shared_out, routed_out
