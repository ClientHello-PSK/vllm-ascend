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

import torch
import torch.nn.functional as F
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.utils import split_tensor_along_first_dim
from vllm_ascend.utils import get_weight_prefetch_method


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor=1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    indices_type: torch.dtype | None = None,
    mix_placement: bool = False,
    num_logical_experts: int = -1,
    num_shared_experts: int = 0,
    num_experts: int = -1,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
):
    """
    Fused experts with select experts.

    Args:
        router_logits: router logits of shape (num_tokens, hidden_size).
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        top_k: number of top k experts.
        use_grouped_topk: Whether to group experts before selecting top-k.
        renormalize: Whether to renormalize the routing weights.
        topk_group: Number of expert groups to select from.
        num_expert_group: Number of experts in each group.
        custom_routing_function: Custom routing function.
        scoring_func: Scoring function to use.
        e_score_correction_bias: Correction bias to apply to expert scores.
        indices_type: dtype of indices
        num_experts: Number of experts.

    Returns:
        topk_weights: router weights of shape (num_tokens, top_k).
        topk_ids: selected expert IDs of shape (num_tokens, top_k).
    """
    # prefetch w1_w3_proj.weight preprocess
    weight_prefetch_method = get_weight_prefetch_method()
    if weight_prefetch_method:
        weight_prefetch_method.maybe_prefetch_moe_weight_preprocess(hidden_states, "gate_up")
    is_support_npu_moe_gating_top_k = check_npu_moe_gating_top_k(
        hidden_states=hidden_states,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        scoring_func=scoring_func,
        custom_routing_function=custom_routing_function,
    )

    if is_support_npu_moe_gating_top_k:
        topk_weights, topk_ids = _select_experts_with_fusion_ops(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            tid2eid=tid2eid,
            input_ids=input_ids,
        )
    else:
        topk_weights, topk_ids = _native_select_experts(
            hidden_states=hidden_states,
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
            tid2eid=None,
            input_ids=None,
        )
        # Apply routed scaling factor to weights
        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * routed_scaling_factor
    if mix_placement:
        shared_expert_routing_factor = 1.0 if is_support_npu_moe_gating_top_k else (1 / routed_scaling_factor)
        batch_size = topk_ids.shape[0]
        pad_shared_expert_ids = torch.arange(
            num_logical_experts, num_logical_experts + num_shared_experts, dtype=topk_ids.dtype, device=topk_ids.device
        ).repeat(batch_size, 1)

        pad_shared_expert_weights = torch.full(
            (topk_weights.shape[0], num_shared_experts),
            shared_expert_routing_factor,
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )

        topk_ids = torch.cat([topk_ids, pad_shared_expert_ids], dim=1)
        topk_weights = torch.cat([topk_weights, pad_shared_expert_weights], dim=1)

    return topk_weights, topk_ids


def check_npu_moe_gating_top_k(
    hidden_states: torch.Tensor,
    top_k: int,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    scoring_func: str = "softmax",
    custom_routing_function: Callable | None = None,
):
    if scoring_func == "sigmoid" and not renormalize:  # sigmoid + renorm=0 is not supported in current branch
        return False
    if custom_routing_function is not None:
        return False
    if scoring_func != "softmax" and scoring_func != "sigmoid" and scoring_func != "sqrtsoftplus":
        return False
    topk_group = topk_group if topk_group is not None else 1
    num_expert_group = num_expert_group if num_expert_group is not None else 1
    if not (
        num_expert_group > 0
        and hidden_states.shape[-1] % num_expert_group == 0
        and hidden_states.shape[-1] // num_expert_group > 2
    ):
        return False
    if topk_group < 1 or topk_group > num_expert_group:
        return False
    if top_k < 1 or top_k > (hidden_states.shape[-1] / (num_expert_group * topk_group)):
        return False
    if topk_group * hidden_states.shape[-1] / num_expert_group < top_k:  # noqa: SIM103
        return False
    return True


def _native_grouped_topk(
    topk_weights: torch.Tensor,
    num_expert_group: int | None,
    topk_group: int | None,
):
    """分组筛选(grouped topk)第一阶段: 选"最强组",淘汰非选中组的专家。

    DeepSeek grouped expert 的两阶段选择:
      阶段1(本函数): 把 N_e 个专家分成 num_expert_group 组,选出最强的
                     topk_group 个组,非选中组的专家分数置 0。
      阶段2(由调用方做 torch.topk): 在幸存的候选里取 top_k 个专家。

    为什么要分组: 强制 token 先从"几个最强组"里选,防止所有 token 扎堆到
                  同一小撮专家,起到均衡负载的作用。

    输入: topk_weights [num_token, num_experts] —— 已打分(可能已加 bias)
    输出: topk_weights [num_token, num_experts] —— 非选中组置 0,选中组保留原分
    """
    topk_group = 0 if topk_group is None else topk_group
    num_expert_group = 0 if num_expert_group is None else num_expert_group

    num_token = topk_weights.shape[0]
    # ① 把 [num_token, num_experts] 重排为 [num_token, 组数, 每组专家数],
    #   组内取 max 作为"组代表分"——用每组最强专家的实力代表整组
    grouped_weights = topk_weights.view(num_token, num_expert_group, -1).max(dim=-1).values
    # ② 在组代表分上做 topk,选出最强的 topk_group 个组(返回组下标)
    topk_group_indices = torch.topk(grouped_weights.to(torch.float32), k=topk_group, dim=-1, sorted=False)[1]
    # ③ 构造组级 mask: 被选中的组位置标 1,其余 0
    topk_group_mask = torch.zeros_like(grouped_weights)
    topk_group_mask.scatter_(1, topk_group_indices, 1)
    # ④ 把组级 mask 扩展回专家级: 每个组位扩展成"该组所有专家"的位
    topk_weight_mask = (
        topk_group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, topk_weights.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )
    # ⑤ 淘汰制: 非选中组的专家分数全部置 0,后续 topk 时它们绝不可能入选
    topk_weights = topk_weights.masked_fill(~topk_weight_mask.bool(), 0.0)

    return topk_weights


def _renormalize_topk_weights(
    topk_weights: torch.Tensor,
    renormalize: bool,
):
    """对选中的 top-k 权重做归一化(norm_topk_prob)。

    作用: 让每个 token 选中的 k 个专家权重之和 = 1。
          即 topk_weights /= sum(topk_weights, dim=-1)。
    何时触发: 当配置 norm_topk_prob=True(即 renormalize=True)时执行。
    位置: 在 topk 选出专家之后、乘 routed_scaling_factor 之前。
    """
    if renormalize:
        # 沿专家维(dim=-1,即 k 个专家)求和并保持维度,再做逐元素除法
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights


def _select_expert_use_group_topk(
    topk_weights: torch.Tensor,
    topk_group: int | None,
    renormalize: bool,
    top_k: int,
    num_expert_group: int | None,
    e_score_correction_bias: torch.Tensor | None,
):
    """分组 topk 选择 + noaux_tc 权重还原。

    本函数浓缩 DeepSeek MoE 路由两大精髓:
      ① grouped topk(分组选择): 先选最强组,再选专家,防扎堆。
      ② noaux_tc(无辅助损失负载均衡): 用 bias 调"选谁",但不污染"权重多大"。

    noaux_tc 双轨机制(核心):
      · "选谁"(选组 + 选专家) 用 s + bias  ← bias 在此生效,引导负载均衡
      · "权重多大"(最终权重) 用 s(原始分)  ← gather 回不加 bias 的分数
      bias 像幕后调度员: 左右选择,但不出现在最终权重数字里。

    输入: topk_weights [num_token, num_experts] —— 已打分但未选专家
    输出: topk_weights [num_token, top_k] —— 选中的 k 个专家权重
          topk_ids    [num_token, top_k] —— 选中的 k 个专家 id(int32)
    注意: 本函数不乘 routed_scaling_factor,由调用方收尾时乘。
    """
    assert topk_group is not None
    assert num_expert_group is not None

    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        # —— noaux_tc: 备份原始分(用于最后算权重),加 bias 后的分仅用于"选"
        original_weights = topk_weights
        topk_weights = topk_weights + e_score_correction_bias.unsqueeze(0)

    # TODO: Change to npu_group_topk when the latest CANN and NNAL is available
    # >>> torch_npu._npu_group_topk(topk_weights, group_num=num_expert_group, k=topk_group)
    # 阶段1: 分组筛选 —— 选 topk_group 个最强组,非选中组专家置 0
    topk_weights = _native_grouped_topk(topk_weights, num_expert_group, topk_group)
    # TODO bfloat16 is not supported in torch.topk with ge graph.
    # 阶段2: topk —— 在幸存候选里取 top_k 个专家
    if e_score_correction_bias is not None:
        # 有 bias: 用加 bias 的分选专家(biased),用原始分当权重(unbiased)
        topk_ids = torch.topk(topk_weights.to(torch.float32), k=top_k, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        # —— noaux_tc 精髓: 权重 gather 不加 bias 的原始分
        topk_weights = original_weights.gather(1, topk_ids)
    else:
        # 无 bias: 直接 topk,分数同时当权重
        topk_weights, topk_ids = torch.topk(topk_weights.to(torch.float32), k=top_k, dim=-1, sorted=False)
    # 后续 npu_moe_init_routing 算子要求专家 id 为 int32
    topk_ids = topk_ids.to(torch.int32)
    # 可选归一化: 让选中 k 个权重和为 1(norm_topk_prob)
    topk_weights = _renormalize_topk_weights(topk_weights, renormalize)
    return topk_weights, topk_ids


def _select_experts_with_fusion_ops(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor | None,
    topk_group: int | None,
    num_expert_group: int | None,
    scoring_func: str = "softmax",
    routed_scaling_factor=1.0,
    tid2eid=None,
    input_ids=None,
):
    topk_group = topk_group if topk_group is not None else 1
    num_expert_group = num_expert_group if num_expert_group is not None else 1
    renorm = int(renormalize)
    if scoring_func == "sqrtsoftplus":
        if tid2eid is not None:
            forward_context = get_forward_context()
            input_ids = forward_context.input_ids.to(torch.int64)
            # tid2eid_ones = torch.ones(tid2eid.shape[0],tid2eid.shape[1],device=router_logits.device,dtype=torch.int32)
            tid2eid_ones = tid2eid.to(torch.int32)
            if forward_context.moe_comm_type == MoECommType.ALLGATHER:
                prepare_finalize = forward_context.moe_comm_method.prepare_finalize
                input_ids = prepare_finalize.all_gather_input_id_with_dp_group(input_ids)
            else:
                input_ids = forward_context.moe_comm_method.pad_and_split_input_ids(input_ids)

            if forward_context.flash_comm_v1_enabled and forward_context.moe_comm_type != MoECommType.ALLGATHER:
                # Process for Flash Comm V1
                tp_size = get_tp_group().world_size
                tp_rank = get_tp_group().rank_in_group
                splitted_input = split_tensor_along_first_dim(input_ids, num_partitions=tp_size)
                input_ids = splitted_input[tp_rank].contiguous()
            input_ids = torch.where(input_ids == -1, 0, input_ids)
        else:
            input_ids = None
            tid2eid_ones = None
        topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
            x=router_logits,
            k=top_k,
            bias=e_score_correction_bias,
            input_ids=input_ids,
            tid2eid=tid2eid_ones,
            k_group=topk_group,
            group_count=num_expert_group,
            routed_scaling_factor=routed_scaling_factor,
            eps=1e-20,
            group_select_mode=1,
            # The hash custom op currently rejects renorm != 0. Apply
            # norm_topk_prob in Python below before returning to MoE compute.
            renorm=0,
            norm_type=2,
            out_flag=False,
        )
        return topk_weights, topk_ids
    norm_type = 0 if scoring_func == "softmax" else 1
    if e_score_correction_bias is not None and e_score_correction_bias.dtype != router_logits.dtype:
        e_score_correction_bias = e_score_correction_bias.to(router_logits.dtype)
    topk_weights, topk_ids, _ = DeviceOperator.moe_gating_top_k(
        router_logits,
        k=top_k,
        k_group=topk_group,
        group_count=num_expert_group,
        group_select_mode=1,
        renorm=renorm,
        norm_type=norm_type,  # 0: softmax; 1: sigmoid
        out_flag=False,
        routed_scaling_factor=routed_scaling_factor,
        eps=1e-20,
        bias_opt=e_score_correction_bias,
    )

    return topk_weights, topk_ids


def _native_select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    use_hash: bool = False,
    tid2eid: dict[int, int] | None = None,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """纯 PyTorch 的 MoE 路由参考实现(NPU 融合算子不可用时的 fallback)。

    本函数逻辑完全透明,是理解"MoE 选专家到底在算什么"的最佳入口。
    V4 正常配置走融合算子(_select_experts_with_fusion_ops),不走这里;
    但融合算子做的事与本函数一一对应。

    选专家的本质四步:
      ① 打分 scoring    : router_logits → 每个专家的分数(全部 N_e 个)
      ② 筛选(可选) group : 分组场景先选"最强组",缩小候选
      ③ topk           : 取分数最高的 k 个 → topk_ids
      ④ 后处理         : 归一化(renorm) + 路由缩放(scaling) → topk_weights

    Args:
        hidden_states: shape (num_tokens, hidden_size);本函数仅用其 dtype
        router_logits: shape (num_tokens, num_experts);gate 已算好的原始偏好分
        top_k: 每个 token 选几个专家
        use_grouped_topk: 是否启用 DeepSeek 分组选择
        renormalize: 是否对 top-k 权重归一化(norm_topk_prob)
        topk_group: 分组选择时选几个组
        num_expert_group: 专家分多少组
        custom_routing_function: 自定义路由函数(可空)
        scoring_func: 打分函数 softmax/sigmoid/sqrtsoftplus(V4 用 sqrtsoftplus)
        routed_scaling_factor: 路由缩放因子
        e_score_correction_bias: noaux_tc 偏置(可空)
        use_hash/tid2eid/input_ids: hash 路由相关;本函数不支持(融合算子才支持)

    Returns:
        topk_weights: shape (num_tokens, top_k);路由权重
        topk_ids: shape (num_tokens, top_k);选中的专家 id(int32)

    Raises:
        ValueError: If an unsupported scoring function is provided.
    """

    # ① 打分: 对全部 N_e 个专家应用打分函数,得到可比、可作权重的分数
    #   注: 此时 topk_weights 形状仍为 [num_tokens, num_experts],还没开始选
    if scoring_func == "softmax":
        # softmax: 沿专家维归一化为概率分布(和为1),指数放大差距
        topk_weights = router_logits.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        # sigmoid: 逐元素压到 (0,1),各专家独立,互不影响
        topk_weights = router_logits.sigmoid()
    elif scoring_func == "sqrtsoftplus":
        # sqrtsoftplus(V4): sqrt(ln(1+e^x)),平滑、恒正、压缩范围,不剧烈放大差距
        topk_weights = F.softplus(router_logits).sqrt()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    # ②③ 分支选择专家
    if use_grouped_topk:
        # 分支 A: DeepSeek 分组选择(V4 走这条)
        # 内部完成 noaux_tc + 分组筛选 + topk + gather 还原权重 + renorm
        topk_weights, topk_ids = _select_expert_use_group_topk(
            topk_weights=topk_weights,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            e_score_correction_bias=e_score_correction_bias,
        )
        # 乘路由缩放因子(补偿"只激活 k 个专家"带来的幅度下降)
        return topk_weights * routed_scaling_factor, topk_ids

    # 非 grouped 分支: 先加 bias(如有),用于选专家
    if e_score_correction_bias is not None:
        topk_weights = topk_weights + e_score_correction_bias

    if custom_routing_function is not None:
        # 分支 B: 自定义路由 —— 调用户函数,自己负责选专家和权重(不乘 scaling)
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )
        # Required by npu_moe_init_routing
        topk_ids = topk_ids.to(torch.int32)
        return topk_weights, topk_ids

    # 分支 C: 普通 topk(无分组、无自定义)
    # 在(可能加 bias 的)分数上取 top_k 个专家
    # 注: 此分支权重含 bias(未像分支A那样 gather 还原为 unbiased),
    #     但本分支通常不与 noaux_tc 组合使用
    topk_weights, topk_ids = topk_weights.topk(top_k, dim=-1)
    topk_weights = topk_weights.to(hidden_states.dtype)

    # Required by npu_moe_init_routing
    topk_ids = topk_ids.to(torch.int32)
    # ④ 后处理: 归一化 + 路由缩放
    topk_weights = _renormalize_topk_weights(topk_weights, renormalize)
    topk_weights = topk_weights * routed_scaling_factor

    return topk_weights, topk_ids


def zero_experts_compute(
    expert_indices: torch.Tensor,
    expert_scales: torch.Tensor,
    num_experts: int,
    zero_expert_type: str,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if zero_expert_type == "identity":
        zero_expert_mask = expert_indices < num_experts
        zero_expert_scales = expert_scales.clone()
        zero_expert_scales = torch.where(zero_expert_mask, 0.0, zero_expert_scales)

        hidden_states = hidden_states.unsqueeze(1)
        zero_expert_scales = zero_expert_scales.unsqueeze(2)
        result = hidden_states * zero_expert_scales
        result = result.sum(dim=1)

    normal_expert_mask = expert_indices >= num_experts
    expert_indices = torch.where(normal_expert_mask, 0, expert_indices)
    expert_scales = torch.where(normal_expert_mask, 0.0, expert_scales)

    return expert_indices, expert_scales, result
