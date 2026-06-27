# private/vllm-ascend-0627 PP+MTP 实现方案

## 一、概述

private/vllm-ascend-0627 在 vllm-ascend 侧**完全自主实现** PP（流水线并行）+ MTP（多 Token 预测）功能，不依赖 vLLM 主仓是否合入 #39704。所有 PP+MTP 相关逻辑通过 vllm-ascend 自有的 override + patch 机制提供，无论主仓状态如何均能独立工作。

### 设计原则

| 原则 | 含义 |
|------|------|
| 完全自主实现 | vllm-ascend 侧自行提供 PP+MTP 全部逻辑，不假设主仓提供基础功能 |
| override 优先 | 对 vllm-ascend 自有的类（NPUModelRunner / RecomputeScheduler）用 override |
| 必要时 patch | 对主仓的类（ModelRunnerOutput / EngineCore / Request）无法 override，用 patch 注入字段/包装方法 |
| 防御性守卫 | patch 函数用 `hasattr` / `if "..." not in fields` 检测，主仓已有对应字段时自动跳过，不冲突 |

### 调度器选择

private-0627 主力调度器为 **RecomputeScheduler**（RCS），所有调度层改动集中在 `recompute_scheduler.py`。

---

## 二、修改清单总表

按文件分组，共 18 项修改。

### patch_pp_mtp.py（4 项）

| 项 | 函数 | 作用 |
|----|------|------|
| P1 | `_patch_model_runner_output` | 给 ModelRunnerOutput 注入 `spec_token_ids` 字段 |
| P2 | `_patch_engine_core` | PP batch_queue + sync + spec_decode 时跳过 `post_step` |
| P3 | `_patch_request_attributes` | 给 Request 注入 `num_spec_tokens_in_flight` 字段 |
| P4 | `_patch_model_config_validation` | 本地 Eagle/MTP drafter 验证为 PP=1 |

### platform.py（1 项）

| 项 | 修改 | 作用 |
|----|------|------|
| P0 | 删除 `_validate_pd_pp_mtp_config` / `_is_mtp_speculative_config` 及调用点 | 移除 PD 场景硬编码限制 |

### recompute_scheduler.py（5 项）

| 项 | 修改 | 作用 |
|----|------|------|
| S1 | `is_mtp_kv_consumer` / `is_kv_producer` 标志 | 识别 KV 消费/生产角色 |
| S2 | `add_request` placeholder tokens 填充 | KV 消费端 prefill 时携带占位 draft tokens |
| S3 | `schedule()` spec_token_ids 调度逻辑 | 调度 draft tokens 到 `scheduled_spec_decode_tokens` |
| S4 | `num_new_tokens` 公式补偿 `num_spec_tokens_in_flight` | 避免在途 spec token 重复计入 |
| S5 | `_update_after_schedule` override | sync 模式下追踪在途 spec token 数 |
| S6 | `_preempt_request` override | 抢占时清零 in_flight 计数 |
| S7 | `update_from_output` override + spec_token_ids 回填 | RCS 独立实现，不调 super，需手动回填 draft tokens |

### model_runner_v1.py（7 项）

| 项 | 修改 | 作用 |
|----|------|------|
| W1 | `propose_draft_token_ids` 开头置 `_draft_token_ids = None` | 防 stale draft data 污染下一步 |
| W2 | `output_spec_token_ids` 提取并传 ModelRunnerOutput | 将 drafter 输出传递给调度器 |
| W3 | `initialize_kv_cache` drafter attn group `is_last_rank` 守卫 | 非 last rank 不初始化 drafter attn |
| W4 | `_check_and_update_cudagraph_mode` drafter cudagraph `is_last_rank` 守卫 | 非 last rank 不初始化 drafter cudagraph |
| W5 | `_collect_pp_mtp_readded_token` + `_pp_mtp_update_req_spec_token_ids` | 非 last PP rank re-added 请求的 token_ids_cpu 修正 |
| W6 | sync/async 对称 broadcast/receive | sync 模式也走 PP broadcast 传递 sampled token |
| W7 | NPU dtype override（receive int32 / broadcast int64） | 解决 NPU scatter EZ1001 / HCCL EI0005 |

### llm_base_proposer.py（1 项）

| 项 | 修改 | 作用 |
|----|------|------|
| D1 | `load_model` `is_last_rank` assert | drafter 只在 last PP rank 加载 |

---

## 三、分项详述

### 3.1 patch_pp_mtp.py

文件位置：`vllm_ascend/patch/platform/patch_pp_mtp.py`

通过 `patch/platform/__init__.py` 注册，在 vllm-ascend 启动时自动执行。

#### P1. `_patch_model_runner_output`

给主仓 `ModelRunnerOutput` dataclass 注入 `spec_token_ids` 字段。

**原因**：drafter 产生的 draft tokens 属于当前 model output，需要携带在 ModelRunnerOutput 上，让调度器从"正在消费的 output"而非"可能已反映更新 schedule 步骤的 live request 状态"更新 `request.spec_token_ids`。

**实现**：
- 检测 `__dataclass_fields__` 是否含 `spec_token_ids`，无则注入类属性 + 包装 `__init__` 接受 `spec_token_ids` 参数
- 给 `EMPTY_MODEL_RUNNER_OUTPUT` 补 `spec_token_ids = None`

#### P2. `_patch_engine_core`

包装主仓 `EngineCore.post_step`。

**原因**：PP batch_queue 模式下 EngineCore 在消费旧 output 前就调度新 batch。`post_step` 从 live request 状态更新全局 `request.spec_token_ids`，会把 draft tokens 挂到错误的 schedule 步骤。

**实现**：PP batch_queue + sync + spec_decode + model_executed 时直接 return，跳过全局 draft token 更新，让调度器的 `update_from_output`（由 `ModelRunnerOutput.spec_token_ids` 驱动）作为唯一来源。

#### P3. `_patch_request_attributes`

给主仓 `Request` 注入 `num_spec_tokens_in_flight` 类属性。

**原因**：追踪当前步调度但尚未被模型消费的 spec token 数。调度器用它避免 `num_new_tokens` 预算公式重复计入在途 spec token，并在抢占时清零。

**实现**：`hasattr` 检测无则包装 `Request.__init__`，初始化 `self.num_spec_tokens_in_flight = 0`。

#### P4. `_patch_model_config_validation`

包装主仓 `ModelConfig.verify_with_parallel_config`。

**原因**：本地 Eagle/MTP drafter 加载在 last PP stage，不随 PP 分片。主仓对 drafter 模型按 PP>1 校验会报错。

**实现**：检测 `runner == "draft"` + Eagle/MTP 架构 + PP>1 时，拷贝 parallel_config 设 `pipeline_parallel_size=1` 后调原始 verify。

### 3.2 platform.py

文件位置：`vllm_ascend/platform.py`

#### P0. 删除 PD 场景硬编码限制

删除 `_validate_pd_pp_mtp_config` + `_is_mtp_speculative_config` 两个方法及 `check_and_update_config` 中的调用点。

**原因**：vllm-ascend 自加的临时保护，硬编码 `raise ValueError` 拒止所有非 PD 分离 P 节点的 PP+MTP 场景。完整实现后此保护无存在意义。

**效果**：PD 分离 P 节点 / PD 分离 D 节点 / PD 混部 / 纯 PP+MTP 等所有场景一视同仁。

### 3.3 recompute_scheduler.py

文件位置：`vllm_ascend/core/recompute_scheduler.py`

#### S1. `is_mtp_kv_consumer` / `is_kv_producer` 标志

`__init__` 中从 `vllm_config.speculative_config` + `kv_transfer_config` 推导角色标志。

#### S2. `add_request` placeholder tokens 填充

KV 消费端新请求入队时，若 `max_model_len` 允许，填 `PLACEHOLDER_TOKEN_ID * num_spec_tokens` 到 `request.spec_token_ids`，保证 full graph 兼容。

#### S3. `schedule()` spec_token_ids 调度逻辑

在 running / waiting 队列调度时，将 `request.spec_token_ids` 按 `num_scheduled_spec_tokens` 截断后放入 `scheduled_spec_decode_tokens`，并清空 `request.spec_token_ids`（新 spec tokens 由 `update_draft_token_ids` 在下一步前设置）。

#### S4. `num_new_tokens` 公式补偿 in_flight

running 队列公式从：
```
num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens
```
改为：
```
num_new_tokens = num_tokens_with_spec + num_output_placeholders
                 + getattr(request, "num_spec_tokens_in_flight", 0)
                 - num_computed_tokens
```

**原因**：sync 模式下上一步调度的 spec tokens 尚未被模型消费（在途），若不补偿会导致预算少算，spec token 无法正确放置。

#### S5. `_update_after_schedule` override

```python
def _update_after_schedule(self, scheduler_output):
    super()._update_after_schedule(scheduler_output)
    if not self.scheduler_config.async_scheduling:
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            if request is not None:
                request.num_spec_tokens_in_flight = len(
                    spec_decode_tokens.get(req_id, ())
                )
```

**原因**：sync 模式下记录本步调度的 spec token 数，供下一步 `num_new_tokens` 公式（S4）使用。

#### S6. `_preempt_request` override

```python
def _preempt_request(self, request, timestamp):
    request.num_spec_tokens_in_flight = 0
    return super()._preempt_request(request, timestamp)
```

**原因**：`_preempt_request` 内部会清空 `request.spec_token_ids`，in_flight 计数需同步清零，否则下一步预算会残留 stale 计数。

#### S7. `update_from_output` override + spec_token_ids 回填

RCS 完全 override `update_from_output`（不调 super），在末尾回填：
```python
spec_token_ids = getattr(model_runner_output, "spec_token_ids", None)
if spec_token_ids is not None:
    from vllm.v1.outputs import DraftTokenIds
    self.update_draft_token_ids(DraftTokenIds(
        req_ids=list(model_runner_output.req_id_to_index.keys()),
        draft_token_ids=spec_token_ids,
    ))
```

**原因**：RCS 不调 super，spec decode 链路会被屏蔽。需手动从 `ModelRunnerOutput.spec_token_ids` 回填到 `update_draft_token_ids`，让 `request.spec_token_ids` 在下一步 schedule 前被正确设置。

**备注**：当前为最小回填，未含 prefill-chunk 跳过 / grammar validate / `num_spec_tokens_in_flight` 清零等边界处理，待 RCS + spec decode 端到端验证后补齐。

### 3.4 model_runner_v1.py

文件位置：`vllm_ascend/worker/model_runner_v1.py`

#### W1. `propose_draft_token_ids` 开头置空

函数体开头加 `self._draft_token_ids = None` / `self._draft_token_req_ids = None`。

**原因**：若某步跳过 proposal（`input_fits_in_drafter=False`），上一步的 draft tokens 会残留，导致 stale `spec_token_ids` 被 attach 到 ModelRunnerOutput。

#### W2. `output_spec_token_ids` 提取传参

`execute_model` 构造 ModelRunnerOutput 前，从 `self._draft_token_ids` 提取 draft tokens，按 `req_ids_output_copy` 顺序对齐，传入 `spec_token_ids=output_spec_token_ids`。

**NPU 特殊处理**：用同步 `.cpu().tolist()` 拷贝，避免 NPU 异步 stream/event 同步问题（`_get_draft_token_ids_cpu` 的 `event.synchronize()` 在 NPU 上可能无法正确等待异步拷贝，导致 stale data）。

#### W3. `initialize_kv_cache` drafter attn group 守卫

drafter attention backend 初始化条件加 `get_pp_group().is_last_rank`。

**原因**：drafter 只在 last PP rank 加载，非 last rank 调用 `self.drafter.initialize_attn_backend` 会因 drafter 为 None 或未初始化而失败。

#### W4. `_check_and_update_cudagraph_mode` drafter cudagraph 守卫

drafter cudagraph keys 初始化条件加 `get_pp_group().is_last_rank`。

**原因**：同 W3，非 last rank 不初始化 drafter cudagraph。

#### W5. re-added token 修正

新增两个方法：
- `_collect_pp_mtp_readded_token`：收集非 last PP rank re-added 请求的 `new_token_ids` 和 `num_computed_tokens`
- `_pp_mtp_update_req_spec_token_ids`（contextmanager）：包装 `_update_states`，在父类 `update_req_spec_token_ids` 放置 spec tokens 前，back-fill `token_ids_cpu` 的缺失 sampled tokens

`execute_model` 中用 `with self._pp_mtp_update_req_spec_token_ids(scheduler_output):` 包装 `self._update_states(scheduler_output)`。

**原因**：非 last PP rank 的请求被移出 persistent batch 后重新加入时，`token_ids_cpu` 可能缺失之前步的 sampled tokens。父类 `update_req_spec_token_ids` 在 `num_tokens_no_spec` 偏移处放置 draft tokens，若 `token_ids_cpu` 有缺失，会导致 spec token 放置位置错误。

#### W6. sync/async 对称 broadcast/receive

`sample_tokens` 中：
- receive 端：移除 `use_async_scheduling` 守卫，sync 模式也调用 `_pp_receive_prev_sampled_token_ids_to_input_batch`
- broadcast 端：sync/async 对称调用 `_pp_broadcast_prev_sampled_token_ids`，按 `use_async_scheduling` 分支取 `sampled` tensor

**原因**：上游简化 `_update_states` 后，非 last rank 不再写 `token_ids_cpu`，sync 路径必须走 broadcast/receive 才能拿到上一步 sampled token。否则 PP0 的 input_ids 丢失，输出乱码。

#### W7. NPU dtype override

- `_pp_receive_prev_sampled_token_ids_to_input_batch`：调 super 后将 `prev_sampled_token_ids` 从 int64 cast 为 int32（匹配 `input_ids.gpu` dtype，解决 `EZ1001 aclnnInplaceScatter`）
- `_pp_broadcast_prev_sampled_token_ids`：调 super 前将 `sampled_token_ids` / `draft_token_ids` cast 为 int64（与接收端 `recv` dtype 对齐，解决 `EI0005 HcomBroadcast`）

**原因**：NPU 对 dtype 一致性比 CUDA 更严格。`scatter_` 不做隐式转换，HCCL 要求所有 ranks dataType 一致。

### 3.5 llm_base_proposer.py

文件位置：`vllm_ascend/spec_decode/llm_base_proposer.py`

#### D1. `load_model` is_last_rank assert

```python
def load_model(self, model):
    assert get_pp_group().is_last_rank, (
        f"{self.method} drafter must be loaded on the last pipeline stage."
    )
    ...
```

**原因**：drafter 作为独立模型加载在 last PP stage，不随 PP 分片。assert 防止误在非 last rank 加载。

### 3.6 模型适配范围说明

private-0627 的工作清单（§二、§三）不包含模型特定适配条目。原因：相关适配在 vllm-ascend 仓内**已存在**，是既有能力，不属于 PP+MTP 新增工作。

#### 仓内既有模型适配文件

| 文件 | 类 | 用途 |
|------|----|------|
| `vllm_ascend/patch/worker/patch_deepseek_mtp.py` | `AscendDeepSeekMTP` / `AscendDeepSeekMultiTokenPredictorLayer` / `AscendGlmMoeDsaForCausalLM` | GLM-5/5.1 (`glm_moe_dsa`→`deepseek_mtp`) + DeepSeek V3 的 MTP monkey-patch |
| `vllm_ascend/models/deepseek_v4_mtp.py` | `DeepSeekV4MTP` | DeepSeek V4 自有 MTP 实现（已声明 `SupportsPP`） |
| `vllm_ascend/models/deepseek_v4.py` | `AscendDeepseekV4ForCausalLM` | DeepSeek V4 目标模型（已声明 `SupportsPP`） |

#### GLM-5 / GLM-5.1 事实陈述

1. HF `model_type` = `glm_moe_dsa`，经 `vllm/config/speculative.py` 的 `hf_config_override` 映射为 `deepseek_mtp`，落入 `MTPModelTypes` → P4（`_patch_model_config_validation`）的 PP=1 校验自动生效。
2. MTP 模型走主仓 `DeepSeekMTP` 实现，但已被 monkey-patch 覆盖：`AscendDeepSeekMultiTokenPredictorLayer` 针对 `target_model_type == "glm_moe_dsa"` 的 `rot` 层做了适配；`AscendDeepSeekMTP` 重写 `_rewrite_spec_layer_name` 处理 `rot.weight` 命名。
3. MTP 作为独立 drafter 在 last PP stage 加载（PP=1），D1（`llm_base_proposer.py` 的 `is_last_rank` assert）自动生效。
4. **`DeepSeekMTP` / `AscendDeepSeekMTP` 未声明 `SupportsPP`**。但因 drafter 走 P4 的 PP=1 校验路径，实际不触发 PP 分片相关校验。若未来 vLLM 主仓加载期对 MTP 强校验 `SupportsPP`，再按 #39704 思路给 `AscendDeepSeekMTP` 补一行即可——当前不需要。

#### DeepSeek V4 事实陈述

1. MTP 类是 vllm-ascend 自有的 `DeepSeekV4MTP`，通过 `models/__init__.py` 的 `ModelRegistry.register_model("DeepSeekV4MTPModel", ...)` 覆盖主仓默认指向。
2. **`DeepSeekV4MTP` 已声明 `SupportsPP`**（`deepseek_v4_mtp.py:201`），无需再补。
3. MTP 作为独立 drafter 加载，**不需要**类似 `patch_qwen3_5.py` 的 forward PP 分流补丁（其 MTP 是独立模型，不随 PP 分片）。

#### 不在本工作清单内的理由

- 上述适配文件在 vllm-ascend 仓内已存在且工作正常，不随 PP+MTP 调度/worker 层修改而变化
- private-0627 的 18 项新修改聚焦于 scheduler / worker / patch_pp_mtp 的协调逻辑，与模型架构无关
- 唯一需要新代码的 Qwen3.5（MTP 嵌入主模型随 PP 分布，需 `patch_qwen3_5.py` forward 分流）已在 §5 P1 中显式声明"P1 不做"

---

## 四、与 10051.patch 的差异

### 4.1 调度器路线不同

| | private-0627 | 10051.patch |
|---|---|---|
| 主力调度器 | RecomputeScheduler | ProfilingChunkScheduler |
| 调度逻辑位置 | `recompute_scheduler.py` | `scheduler_profiling_chunk.py` |
| `update_from_output` | RCS 最小回填 | ProfilingChunkScheduler 完整实现（prefill skip + grammar validate + in_flight reset） |

### 4.2 private-0627 多出的功能

| 功能 | 说明 |
|------|------|
| `_patch_request_attributes` | 注入 `num_spec_tokens_in_flight`，10051.patch 无此 patch |
| `_update_after_schedule` override | RCS 追踪 in_flight，10051.patch 无 |
| `_preempt_request` override | RCS 抢占清零 in_flight，10051.patch 文档提到但未实现 |
| `num_new_tokens` 公式补偿 in_flight | 10051.patch 无 |
| NPU dtype override（N1/N2） | 10051.patch 无（CUDA 无此问题） |
| sync/async 对称 broadcast/receive | 10051.patch 无 |
| platform.py PD 限制删除 | 10051.patch 无 |
| 删除 `new_token_ids` 兜底块 | 10051.patch 无 |

### 4.3 10051.patch 有但 private-0627 未做的

| 功能 | 原因 |
|------|------|
| ProfilingChunkScheduler 完整 `update_from_output` | private-0627 用 RCS，RCS 是最小回填 |
| `max_num_running_reqs` / `max_num_per_batch` batch 上限 | RCS 有自己的调度逻辑 |
| 非 last PP rank accepted counts 推断 | P1 不做（mamba/hybrid attn 场景，GLM-5.1/DeepSeek V4 是 attention-only MoE 不需要） |
| `_prepare_non_last_pp_mtp_state_update` | P1 不做（mamba postprocess 时序对齐） |
| `patch_qwen3_5.py` MTP forward PP 分流 | P1 不做（GLM-5.1/DeepSeek V4 MTP 是独立模型，不需要） |
| `_check_and_update_cudagraph_mode` skip_parent_drafter_init + `_maybe_initialize_drafter_cudagraph_keys` | 用 `is_last_rank` 守卫替代 |
| 4 个 UT 文件 | 未写测试 |

### 4.4 实现方式不同

| 项 | private-0627 | 10051.patch |
|---|---|---|
| patch_pp_mtp.py 函数数 | 4 个 | 3 个 |
| `_check_and_update_cudagraph_mode` | `is_last_rank` 守卫 | skip_parent_drafter_init（临时置 None）+ 兜底方法 |
| `output_spec_token_ids` 提取 | `.cpu().tolist()` 同步拷贝 | `isinstance` + 局部变量 |

---

## 五、未做项（P1）

以下功能按"最小方案"暂未实现，待核心场景验证后再补：

| 项 | 场景 | 说明 |
|----|------|------|
| 非 last PP rank accepted counts 推断 | mamba / hybrid attn + PP + MTP | `_is_non_last_pp_mtp` / `_compute_non_last_pp_mtp_accepted_counts` / `_sync_num_accepted_tokens_to_gpu` |
| mamba postprocess 时序对齐 | 同上 | `_prepare_non_last_pp_mtp_state_update` |
| Qwen3.5 MTP forward PP 分流 | Qwen3.5 + PP + MTP | `patch_qwen3_5.py` 的 `qwen3_5_mtp_forward`（非 last rank 返回 IntermediateTensors） |
| RCS `update_from_output` 完整实现 | RCS + spec decode 端到端 | 补 prefill-chunk 跳过 / grammar validate / `num_spec_tokens_in_flight` 清零 |
| UT | 全场景 | 补 4 个测试文件 |

---

## 六、验证矩阵

| 调度模式 | GLM-5.1 | DeepSeek V4 |
|---------|---------|-------------|
| sync + PP + MTP | 18 项全修完即可运行 | 同左 |
| async + PP + MTP | W6（sync/async 对称 broadcast/receive）覆盖 async 路径 | 同左 |
| RCS + sync + PP + MTP | S7（spec_token_ids 回填）必修 | 同左 |
| PD 混部 + sync + PP + MTP | P0（platform.py 限制删除）+ W6/W7 必修 | 同左 |
| mamba + PP + MTP | 需补 P1 accepted counts 推断 | 同左 |
| Qwen3.5 + PP + MTP | 需补 P1 MTP forward 分流 | N/A |

---

## 七、文件修改索引

| 文件 | 修改项 |
|------|--------|
| `vllm_ascend/patch/platform/patch_pp_mtp.py` | P1-P4 |
| `vllm_ascend/patch/platform/__init__.py` | 注册 patch_pp_mtp |
| `vllm_ascend/platform.py` | P0 |
| `vllm_ascend/core/recompute_scheduler.py` | S1-S7 |
| `vllm_ascend/worker/model_runner_v1.py` | W1-W7 |
| `vllm_ascend/spec_decode/llm_base_proposer.py` | D1 |
