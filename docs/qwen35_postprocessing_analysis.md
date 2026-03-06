# Qwen3.5 后处理详细分析文档

## 概述

本文档详细分析vllm-ascend中Qwen3.5的后处理逻辑，涵盖传统后处理和投机推理后处理两种模式。

---

# 第一部分：传统后处理

## 1. 初始化

### 1.1 AscendSampler 类定义

**文件位置**: `vllm_ascend/sample/sampler.py`

```python
class AscendSampler(Sampler):
    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler()  # Ascend优化的TopK/TopP采样器
        self.async_exponential_event = torch.npu.Event()   # 异步事件
```

### 1.2 Sampler在ModelRunner中的初始化

**文件位置**: `vllm_ascend/worker/model_runner_v1.py:279`

```python
class NPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # ...
        self.sampler = AscendSampler()
```

### 1.3 SamplingMetadata 构建

**文件位置**: `vllm/v1/sample/metadata.py`

```python
@dataclass
class SamplingMetadata:
    temperature: torch.Tensor | None      # 温度参数 [batch_size]
    all_greedy: bool                      # 是否全部贪心采样
    all_random: bool                      # 是否全部随机采样
    top_p: torch.Tensor | None            # top-p参数 [batch_size]
    top_k: torch.Tensor | None            # top-k参数 [batch_size]
    generators: dict[int, torch.Generator] # 随机数生成器
    max_num_logprobs: int | None          # 最大logprobs数量
    prompt_token_ids: torch.Tensor | None # 提示词token IDs (penalties计算使用)
    no_penalties: bool                    # 是否不需要惩罚
    frequency_penalties: torch.Tensor     # 频率惩罚
    presence_penalties: torch.Tensor      # 存在惩罚
    repetition_penalties: torch.Tensor    # 重复惩罚
    output_token_ids: list[list[int]]     # 输出token IDs
    allowed_token_ids_mask: torch.Tensor | None
    bad_words_token_ids: dict[int, list[list[int]]]
    logitsprocs: LogitsProcessors
    spec_token_ids: list[list[int]] | None
```

**构建位置**: `vllm/v1/worker/gpu_input_batch.py:774-856`

---

## 2. 配置

### 2.1 采样参数配置

**文件位置**: `vllm/sampling_params.py`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | float | 1.0 | 控制采样随机性，0表示贪心采样 |
| `top_p` | float | 1.0 | nucleus采样累积概率阈值 |
| `top_k` | int | 0 | 只考虑top-k个token，0表示全部 |
| `presence_penalty` | float | 0.0 | 存在惩罚 |
| `frequency_penalty` | float | 0.0 | 频率惩罚 |
| `repetition_penalty` | float | 1.0 | 重复惩罚 |

### 2.2 配置传递流程

```
用户请求 (SamplingParams)
    ↓
InputBatch.add_request()  [gpu_input_batch.py:344-371]
    ↓
设置CPU端参数:
    temperature_cpu[req_index] = sampling_params.temperature
    top_p_cpu[req_index] = sampling_params.top_p
    top_k_cpu[req_index] = sampling_params.top_k
    ↓
_make_sampling_metadata()  [gpu_input_batch.py:774-856]
    ↓
创建 SamplingMetadata (GPU张量)
```

### 2.3 Ascend特有配置

**文件位置**: `vllm_ascend/sample/sampler.py:137-141`

```python
# TopK/TopP算子选择
apply_top_k_top_p = (
    _apply_top_k_top_p_ascendc    # A2/A3芯片使用AscendC自定义算子
    if get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]
    else _apply_top_k_top_p_pytorch  # 其他芯片使用PyTorch实现
)
```

---

## 3. 入口函数和执行流程

### 3.1 入口函数

**主入口**: `vllm_ascend/worker/model_runner_v1.py:1537-1556`

```python
def _sample(self, logits, spec_decode_metadata):
    sampling_metadata = self.input_batch.sampling_metadata
    if spec_decode_metadata is None:  # 传统后处理路径
        return self.sampler(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
    # ... 投机推理路径
```

### 3.2 完整执行流程

```
sample_tokens() [model_runner_v1.py:1397]
    │
    ├── 1. 解包 execute_model_state
    │       └── 获取 logits, scheduler_output, attn_metadata 等
    │
    ├── 2. 【可选】应用结构化输出约束
    │       └── apply_grammar_bitmask(logits)
    │
    ├── 3. 调用 _sample()
    │       │
    │       └── AscendSampler [sampler.py]
    │               │
    │               ├── 计算logprobs（如果需要）
    │               ├── 转换为float32
    │               ├── apply_logits_processors()
    │               └── sample()
    │
    ├── 4. 【可选】_update_states_after_model_execute()
    │       └── need_accepted_tokens 时更新状态
    │
    ├── 5. 【可选】propose_draft_token_ids() [推测解码]
    │       │
    │       ├── EAGLE 模式: 使用 GPU 采样结果
    │       └── 其他模式: 使用 CPU 采样结果
    │
    ├── 6. 【可选】KV 传输组清理
    │       └── get_kv_transfer_group().clear_connector_metadata()
    │
    ├── 7. 【可选】RoutedExpertsCapturer 保存
    │       └── capturer.save_captured_experts()
    │
    ├── 8. _bookkeeping_sync()
    │       └── 解析采样结果，更新状态
    │
    ├── 9. 【可选】dynamic_eplb 更新
    │       └── eplb_updator.forward_end()
    │
    ├── 10. 【可选】debugger 处理
    │       └── debugger.stop() / debugger.step()
    │
    └── 11. 返回结果
            │
            ├── 同步模式: ModelRunnerOutput
            └── 异步模式: AsyncGPUModelRunnerOutput
```

> **注意**: 步骤 4-7 为可选步骤，根据配置和条件触发

### 3.3 AscendSampler.forward() 详细流程

> **继承关系**: `AscendSampler` 继承自 `Sampler`（`vllm/v1/sample/sampler.py`），`forward` 方法直接复用父类实现。`AscendSampler` 主要覆写了 `TopKTopPSampler` 为 `AscendTopKTopPSampler`，以及添加异步指数分布计算。

**文件位置**: `vllm/v1/sample/sampler.py:67-129`

#### 3.3.1 整体流程概览

```
forward(logits, sampling_metadata, predict_bonus_token, logprobs_mode_override)
    │
    ├── Step 1: 计算原始 logprobs（可选）
    │   ├── raw_logprobs 模式: logits.log_softmax(dim=-1)
    │   └── raw_logits 模式: logits.clone() 或 logits.to(float32)
    │
    ├── Step 2: 类型转换 → float32
    │
    ├── Step 3: apply_logits_processors()
    │   ├── 3a. allowed_token_ids 白名单过滤
    │   ├── 3b. bad_words 排除
    │   ├── 3c. non_argmax_invariant 处理器（min_tokens, logit_bias）
    │   └── 3d. apply_penalties（repetition/frequency/presence）
    │
    ├── Step 4: sample()
    │   ├── 4a. 贪心采样: argmax（如果需要）
    │   ├── 4b. 温度缩放: logits / temperature
    │   ├── 4c. argmax_invariant 处理器（min_p）
    │   ├── 4d. AscendTopKTopPSampler.forward_native()
    │   │   ├── apply_top_k_top_p()（昇腾原生/PyTorch）
    │   │   ├── softmax → probs
    │   │   └── Gumbel-Max采样 或 random_sample()
    │   └── 4e. torch.where 合并贪心/随机结果
    │
    ├── Step 5: 类型转换 sampled → int64 → int32
    │
    ├── Step 6: gather_logprobs()（可选）
    │   ├── topk(raw_logprobs, num_logprobs) → topk_indices, topk_logprobs
    │   ├── gather sampled token logprob
    │   ├── batched_count_greater_than → ranks
    │   └── 拼接 → LogprobsTensors
    │
    └── Step 7: 返回 SamplerOutput
            ├── sampled_token_ids: [num_reqs, 1]
            └── logprobs_tensors: LogprobsTensors | None
```

#### 3.3.2 Step 1: 计算原始 Logprobs

```python
# vllm/v1/sample/sampler.py:79-87
num_logprobs = sampling_metadata.max_num_logprobs
if num_logprobs is not None:
    if logprobs_mode == "raw_logprobs":
        # 对原始logits做log_softmax，保留未经任何处理的概率分布
        raw_logprobs = self.compute_logprobs(logits)
        # compute_logprobs: logits.log_softmax(dim=-1, dtype=torch.float32)
    elif logprobs_mode == "raw_logits":
        # 直接克隆原始logits
        if logits.dtype == torch.float32:
            raw_logprobs = logits.clone()
        else:
            raw_logprobs = logits.to(torch.float32)
```

| 模式 | 计算方式 | 说明 |
|------|---------|------|
| `raw_logprobs` | `log_softmax(logits)` | 默认模式，返回原始概率的对数 |
| `raw_logits` | `logits.clone()` | 返回原始logits值 |
| `processed_logprobs` | 在 TopKTopP 阶段计算 | 返回处理后的概率对数 |
| `processed_logits` | 在 TopKTopP 阶段计算 | 返回处理后的logits |

> **关键点**: raw_logprobs 在应用惩罚和温度缩放**之前**计算，这与 V0 Sampler 不同（V0 使用处理后的 logits）。

#### 3.3.3 Step 2: 类型转换

```python
# vllm/v1/sample/sampler.py:90
logits = logits.to(torch.float32)
```

将 logits 从模型输出的精度（通常为 float16/bfloat16）转换为 float32，避免后续处理中的精度损失。

#### 3.3.4 Step 3: apply_logits_processors()

**文件位置**: `vllm/v1/sample/sampler.py:266-300`

```python
def apply_logits_processors(self, logits, sampling_metadata, predict_bonus_token):
    # 3a. 应用 allowed_token_ids 白名单
    # 将不在白名单中的 token 概率设为 -inf
    if sampling_metadata.allowed_token_ids_mask is not None:
        logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))

    # 3b. 应用 bad_words 排除
    # 检查上下文是否匹配 bad_words 前缀，如果匹配则将对应 token 设为 -inf
    if bad_words_token_ids:
        apply_bad_words(logits, bad_words_token_ids, output_token_ids)

    # 3c. 应用非 argmax 不变处理器
    # 这些处理器可能影响贪心采样结果（如 min_tokens, logit_bias）
    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        logits = processor.apply(logits)

    # 3d. 应用惩罚项
    logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)
    return logits
```

**apply_penalties 详细说明**:

```python
# vllm/v1/sample/sampler.py:302-319
def apply_penalties(logits, sampling_metadata, output_token_ids):
    if sampling_metadata.no_penalties:
        return logits  # 无惩罚直接返回

    return apply_all_penalties(
        logits,
        sampling_metadata.prompt_token_ids,      # 提示词token IDs
        sampling_metadata.presence_penalties,      # 存在惩罚
        sampling_metadata.frequency_penalties,     # 频率惩罚
        sampling_metadata.repetition_penalties,    # 重复惩罚
        output_token_ids,                          # 已输出的token IDs
    )
```

| 惩罚类型 | 计算方式 | 效果 |
|---------|---------|------|
| `repetition_penalty` | `logit = logit / penalty` (正值) 或 `logit * penalty` (负值) | 降低已出现token的概率 |
| `frequency_penalty` | `logit -= frequency * count` | 按出现次数线性惩罚 |
| `presence_penalty` | `logit -= presence * (count > 0)` | 只要出现过就惩罚 |

#### 3.3.5 Step 4: sample()

**文件位置**: `vllm/v1/sample/sampler.py:147-203`

```python
def sample(self, logits, sampling_metadata, logprobs_mode_override=None):
    # ======== 4a. 贪心采样 ========
    if sampling_metadata.all_random:
        greedy_sampled = None  # 全部随机，跳过贪心
    else:
        greedy_sampled = self.greedy_sample(logits)  # argmax(dim=-1)
        if sampling_metadata.all_greedy:
            # 全部贪心，直接返回（可选计算 processed_logprobs）
            return greedy_sampled, processed_logprobs

    # ======== 4b. 温度缩放 ========
    # logits = logits / temperature
    # 对 temperature < EPS 的位置用 1.0 替换，避免除以0
    logits = self.apply_temperature(
        logits, sampling_metadata.temperature, sampling_metadata.all_random
    )

    # ======== 4c. argmax 不变处理器 ========
    # 如 min_p 处理器，只影响随机采样
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)

    # ======== 4d. TopK/TopP + 采样 ========
    # 调用 AscendTopKTopPSampler.forward_native()
    random_sampled, processed_logprobs = self.topk_topp_sampler(
        logits, sampling_metadata.generators,
        sampling_metadata.top_k, sampling_metadata.top_p,
    )

    # ======== 4e. 合并贪心/随机结果 ========
    if greedy_sampled is None:
        return random_sampled, processed_logprobs
    # temperature < EPS 的请求使用贪心结果，否则使用随机结果
    sampled = torch.where(
        sampling_metadata.temperature < _SAMPLING_EPS,
        greedy_sampled, random_sampled,
        out=greedy_sampled,  # 复用张量减少内存分配
    )
    return sampled, processed_logprobs
```

#### 3.3.6 Step 4d 详解: AscendTopKTopPSampler.forward_native()

**文件位置**: `vllm_ascend/sample/sampler.py:74-88`

> **继承关系**: `AscendTopKTopPSampler` 继承自 `TopKTopPSampler`，覆写 `forward_native` 方法。

```python
def forward_native(self, logits, generators, k, p):
    """Override pytorch native implementation to torch_npu"""
    # 1. 应用 Top-K/Top-P 过滤
    # A2/A3 使用昇腾原生算子 npu_apply_top_k_top_p
    # 其他设备使用 PyTorch 实现（sort + mask）
    logits = self.apply_top_k_top_p(logits, k, p)

    # 2. 可选: 保存处理后的 logprobs/logits
    logits_to_return = None
    if self.logprobs_mode == "processed_logits":
        logits_to_return = logits
    elif self.logprobs_mode == "processed_logprobs":
        logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

    # 3. 计算概率分布
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # 4. 采样
    if get_ascend_config().enable_async_exponential:
        # 异步 Gumbel-Max 采样（随机数已提前生成）
        self.async_event.synchronize()  # 等待异步指数随机数完成
        return probs.div_(self.q).argmax(dim=-1).view(-1), logits_to_return
    # 同步采样
    return random_sample(probs, generators), logits_to_return
```

**与基类 TopKTopPSampler.forward_native() 的差异**:

| 特性 | 基类 (TopKTopPSampler) | AscendTopKTopPSampler |
|------|----------------------|----------------------|
| Top-K/Top-P 实现 | PyTorch sort/mask 或 Triton | 昇腾原生算子 `npu_apply_top_k_top_p` |
| 随机采样 | `random_sample()` (同步) | 异步 Gumbel-Max (可选) 或 `random_sample()` |
| 异步优化 | 无 | 支持 `enable_async_exponential` |

**random_sample() 采样原理** (Gumbel-Max Trick):

```python
# vllm_ascend/sample/sampler.py:11-34
def random_sample(probs, generators):
    q = torch.empty_like(probs)
    # 生成指数分布随机数 q ~ Exp(1)
    if len(generators) != probs.shape[0]:
        q.exponential_()  # 批量生成
    if generators:
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)  # 按请求生成
    # probs / q 等价于 Gumbel-Max 采样
    # 数学等价于 multinomial(probs)，但避免CPU-NPU同步
    return probs.div_(q).argmax(dim=-1).view(-1)
```

> **数学原理**: 若 $q_i \sim \text{Exp}(1)$，则 $\arg\max_i \frac{p_i}{q_i}$ 等价于从分类分布 $\text{Cat}(p_1, \ldots, p_n)$ 中采样。

**random_sample() 详细流程分析**:

`random_sample` 存在两个版本：**vLLM 基类版本**（GPU）和 **vllm-ascend 昇腾版本**（NPU），核心算法一致，但昇腾版本增加了**流切换**机制。

**1. 输入参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `probs` | `torch.Tensor` shape `[batch_size, vocab_size]` | 经过 softmax 归一化后的概率分布 |
| `generators` | `dict[int, torch.Generator]` | 按请求索引映射的随机数生成器，用于可复现采样（seed 场景） |

**2. 执行流程（逐步拆解）**

```
步骤 1: 流切换（仅昇腾版本）
  ├── npu_stream_switch(global_stream()) 将后续操作切换到全局辅助流
  ├── 目的: 将随机数生成与主计算流解耦，实现流水线并行
  └── global_stream() 返回一个独立的 torch.npu.Stream 实例

步骤 2: 分配随机数张量
  └── q = torch.empty_like(probs)   # shape 与 probs 相同 [batch_size, vocab_size]
      # 仅分配内存，不初始化（性能优化）

步骤 3: 生成指数分布随机数 q ~ Exp(1)
  ├── 情况 A: len(generators) != probs.shape[0]（大多数请求无自定义 seed）
  │   └── q.exponential_()           # 批量原地生成，所有行一次完成
  ├── 情况 B: generators 非空（部分请求有自定义 seed）
  │   └── for i, generator in generators.items():
  │       └── q[i].exponential_(generator=generator)  # 逐行覆盖特定请求的随机数
  └── 设计要点:
      ├── 先批量生成（快），再逐个覆盖有 seed 的行（慢但少量）
      ├── 两个 if 不是互斥的，而是顺序执行:
      │   • 当部分请求有 seed 时，先全量生成，再覆盖特定行
      │   • 当所有请求都有 seed 时（len == shape[0]），跳过批量生成
      └── 当 generators 为空时，仅执行批量生成

步骤 4: 流同步（仅昇腾版本）
  └── torch.npu.current_stream().wait_stream(global_stream())
      # 主流等待辅助流完成随机数生成，确保 q 数据就绪

步骤 5: Gumbel-Max 采样
  └── probs.div_(q)                  # 原地除法: probs[i][j] /= q[i][j]
      .argmax(dim=-1)                # 沿 vocab 维度取最大值索引 → shape [batch_size]
      .view(-1)                      # 展平为一维 → 最终 token IDs
```

**3. 为什么不用 `torch.multinomial`？**

| 对比项 | `torch.multinomial` | `random_sample` (Gumbel-Max) |
|--------|-------------------|------------------------------|
| CPU-设备同步 | **需要**（内部有 CPU-GPU/NPU 同步点） | **不需要**（纯设备端计算） |
| 数学等价性 | 直接多项式采样 | 通过指数分布间接实现，统计等价 |
| 异步友好性 | 差（同步点阻塞流水线） | 好（可完全在设备端异步执行） |
| 性能瓶颈 | 同步开销在高吞吐场景显著 | `argmax` 计算量大但无同步开销 |

**4. 昇腾版本 vs 基类版本差异**

```python
# 基类版本 (vllm/v1/sample/ops/topk_topp_sampler.py:325-346)
def random_sample(probs, generators):
    q = torch.empty_like(probs)
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)

# 昇腾版本 (vllm_ascend/sample/sampler.py:11-34)
def random_sample(probs, generators):
    with npu_stream_switch(global_stream()):  # ← 额外: 切换到辅助流
        q = torch.empty_like(probs)
        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.current_stream().wait_stream(global_stream())  # ← 额外: 流同步
    return probs.div_(q).argmax(dim=-1).view(-1)
```

昇腾版本的关键改进：将**指数分布随机数生成**放到 `global_stream()` 辅助流中执行，使其可以与主流上的其他计算（如 Top-K/Top-P 过滤后的 softmax）并行，减少端到端延迟。

**5. 异步预计算优化 (`do_async_exponential`)**

当 `enable_async_exponential=True` 时，`AscendSampler` 会在模型前向推理期间**提前**生成指数随机数：

```python
# AscendSampler.do_async_exponential() - 在模型执行期间调用
def do_async_exponential(self, b_s, head_dim, generators):
    with torch.npu.stream(global_stream()):
        global_stream().wait_stream(torch.npu.current_stream())
        q = torch.empty((b_s, head_dim), device="npu", dtype=torch.float32)
        if len(generators) != q.shape[0]:
            q.exponential_()
        if generators:
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
        self.async_exponential_event.record()  # 记录事件
    self.set_q_event(q, self.async_exponential_event)  # 传递给采样器

# AscendTopKTopPSampler.forward_native() - 采样时使用预计算的 q
if get_ascend_config().enable_async_exponential:
    self.async_event.synchronize()  # 等待预计算完成
    return probs.div_(self.q).argmax(dim=-1).view(-1), logits_to_return
```

这样指数随机数的生成与模型前向推理**完全重叠**，采样阶段仅需执行 `div_` + `argmax`，进一步降低采样延迟。

#### 3.3.7 Step 5-6: 类型转换与 Logprobs 收集

```python
# vllm/v1/sample/sampler.py:99-116

# Step 5: 类型转换
sampled = sampled.long()    # → int64 (兼容后续索引操作)

# Step 6: 收集 logprobs
if num_logprobs is None:
    logprobs_tensors = None
elif num_logprobs == -1:
    # 返回完整的未排序 logprobs（用于拒绝采样的 bonus token）
    logprobs_tensors = LogprobsTensors(
        torch.empty(0), raw_logprobs, torch.empty(0)
    )
else:
    # 收集 top-k logprobs + sampled token 的 logprob
    logprobs_tensors = self.gather_logprobs(
        raw_logprobs, num_logprobs, token_ids=sampled
    )
```

**gather_logprobs 详细流程**:

```python
# vllm/v1/sample/sampler.py:209-251
def gather_logprobs(logprobs, num_logprobs, token_ids):
    # 1. 获取 top-k logprobs 及其索引
    topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)
    # topk_logprobs: [num_reqs, num_logprobs]
    # topk_indices:  [num_reqs, num_logprobs]

    # 2. 获取采样 token 的 logprob
    token_ids = token_ids.unsqueeze(-1)  # [num_reqs, 1]
    token_logprobs = logprobs.gather(-1, token_ids)  # [num_reqs, 1]

    # 3. 计算采样 token 的排名
    token_ranks = batched_count_greater_than(logprobs, token_logprobs)

    # 4. 拼接结果
    indices = torch.cat((token_ids, topk_indices), dim=1)   # [num_reqs, num_logprobs+1]
    logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

    return LogprobsTensors(indices.to(int32), logprobs, token_ranks)
```

#### 3.3.8 Step 7: 返回 SamplerOutput

```python
# vllm/v1/sample/sampler.py:118-129
sampled = sampled.to(torch.int32)  # 减小张量大小

sampler_output = SamplerOutput(
    sampled_token_ids=sampled.unsqueeze(-1),  # [num_reqs] → [num_reqs, 1]
    logprobs_tensors=logprobs_tensors,         # LogprobsTensors | None
)
return sampler_output
```

**SamplerOutput 数据结构**:

```python
@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor       # [num_reqs, 1] int32
    logprobs_tensors: LogprobsTensors | None

class LogprobsTensors(NamedTuple):
    logprob_token_ids: torch.Tensor       # [num_reqs, num_logprobs+1] int32
    logprobs: torch.Tensor                # [num_reqs, num_logprobs+1] float32
    selected_token_ranks: torch.Tensor    # [num_reqs] int64
    cu_num_generated_tokens: list[int] | None = None  # [num_reqs] 累积生成token数
```

#### 3.3.9 执行路径总结

```
┌──────────────────────────────────────────────────────────────────────┐
│ AscendSampler.forward() 执行路径                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  路径 A: 全部贪心 (all_greedy=True, temperature=0)                    │
│  ─────────────────────────────────────────────────                   │
│  logits → float32 → apply_logits_processors → argmax → 返回         │
│  特点: 不需要温度缩放、Top-K/Top-P、随机采样                           │
│                                                                      │
│  路径 B: 全部随机 (all_random=True)                                   │
│  ─────────────────────────────────────────────────                   │
│  logits → float32 → apply_logits_processors                         │
│        → temperature → argmax_invariant处理器                        │
│        → Top-K/Top-P → softmax → Gumbel-Max → 返回                  │
│  特点: 不需要贪心采样、不需要 torch.where 合并                        │
│                                                                      │
│  路径 C: 混合模式 (部分贪心 + 部分随机)                               │
│  ─────────────────────────────────────────────────                   │
│  logits → float32 → apply_logits_processors                         │
│        → argmax(贪心) → temperature → argmax_invariant处理器         │
│        → Top-K/Top-P → softmax → Gumbel-Max(随机)                   │
│        → torch.where(温度判断合并) → 返回                             │
│  特点: 同时执行贪心和随机，按 temperature 选择结果                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.4 AscendSampler.forward() 全流程算子分析（Triton改造参考）

> **目标**: 对 `AscendSampler.forward()` 中每一步操作进行 **API/算子级拆解**、**运行位置标注**（CPU / NPU-Vector / NPU-Cube / AI-CPU）、**数据流形状追踪**，为 Triton 算子融合改造提供依据。

#### 3.4.1 算子全景总表

| 序号 | 步骤 | PyTorch API / 算子 | 底层算子(NPU) | 运行位置 | 输入形状 | 输出形状 | 是否可Triton化 | 备注 |
|------|------|-------------------|--------------|---------|---------|---------|--------------|------|
| 1a | log_softmax (raw_logprobs) | `logits.log_softmax(dim=-1, dtype=fp32)` | LogSoftmaxV2 | NPU-Vector | `[B, V]` fp16/bf16 | `[B, V]` fp32 | ✅ | 可与cast融合 |
| 1b | clone/to (raw_logits) | `logits.clone()` 或 `logits.to(fp32)` | Clone/Cast | NPU-Vector | `[B, V]` fp16/bf16 | `[B, V]` fp32 | ✅ | raw_logits模式,不做log_softmax |
| 2 | dtype cast | `logits.to(torch.float32)` | Cast | NPU-Vector | `[B, V]` fp16/bf16 | `[B, V]` fp32 | ✅ | 可融合到前后算子 |
| 3a | masked_fill_ | `logits.masked_fill_(mask, -inf)` | MaskedFill | NPU-Vector | `[B, V]` fp32 + mask `[B, V]` bool | `[B, V]` fp32 | ✅ | allowed_token_ids白名单 |
| 3b | bad_words | `logits[i][token_id] = -inf` | 逐元素索引赋值 | **CPU→NPU** | 逐请求处理 | 同输入 | ⚠️ | CPU循环+NPU索引写,瓶颈点 |
| 3c | logit_bias | `logits += bias` | Add | NPU-Vector | `[B, V]` fp32 | `[B, V]` fp32 | ✅ | non_argmax_invariant处理器 |
| 3d-0 | masked_fill_ (-1替换) | `output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)` | Compare+MaskedFill | NPU-Vector | `[B, max_seq]` int64 | `[B, max_seq]` int64 | ✅ | 替换异步调度的-1占位符为vocab_size |
| 3d-i | get_token_bin_counts (prompt) | `zeros + scatter_add_ + (>0)` | Zeros+ScatterAdd+GT | NPU-Vector | `[B, V+1]` + `[B, seq_len]` | `[B, V]` mask bool | ✅ | 统计prompt token频次,生成prompt_mask |
| 3d-i' | get_token_bin_counts (output) | `zeros + scatter_add_ + (>0)` | Zeros+OnesLike+ScatterAdd+GT | NPU-Vector | `[B, V+1]` + `[B, seq_len]` | `[B, V]` counts + mask bool | ✅ | 统计output token频次,生成output_bin_counts和output_mask |
| 3d-ii | repetition_penalty | `unsqueeze+repeat+where(mask)+where(logits>0)+mul_` | Unsqueeze+Repeat+Or+Where+Where+Mul_ | NPU-Vector | `[B, V]` fp32 + `[B]` penalties | `[B, V]` fp32 | ✅ | apply_repetition_penalties_torch |
| 3d-iii | frequency_penalty | `logits -= freq_pen * bin_counts` | Mul+Sub | NPU-Vector | `[B, V]` fp32 | `[B, V]` fp32 | ✅ | 可与repetition融合 |
| 3d-iv | presence_penalty | `logits -= pres_pen * output_mask` | Mul+Sub | NPU-Vector | `[B, V]` fp32 | `[B, V]` fp32 | ✅ | 可与上两项融合 |
| 4a | argmax (贪心) | `logits.argmax(dim=-1)` | ArgMaxWithValue | NPU-Vector | `[B, V]` fp32 | `[B]` int64 | ✅ | 贪心采样, all_greedy时直接返回 |
| 4b | temperature div | `logits.div_(temp.unsqueeze(1))` | Div(broadcast) | NPU-Vector | `[B, V]` fp32 / `[B,1]` fp32 | `[B, V]` fp32 | ✅ | 可融合到softmax |
| 4c | min_p | 自定义处理器 | Where+Mul+Mask | NPU-Vector | `[B, V]` fp32 | `[B, V]` fp32 | ✅ | argmax_invariant处理器 |
| 4d-i | top_k_top_p | `npu_apply_top_k_top_p(logits, k, p)` | **AscendC自定义算子** | NPU-Vector | `[B, V]` fp32 + `[B]` k,p | `[B, V]` fp32 | ⚠️ | A2/A3专用;其他走PyTorch sort |
| 4d-i' | top_k_top_p(fallback) | `softmax→sort→gather→compare→masked_fill_(+cumsum for top_p)` | Softmax+Sort+Gather+Compare+MaskedFill(+Cumsum) | NPU-Vector | `[B, V]` fp32 | `[B, V]` fp32 | ✅ | PyTorch fallback路径(无scatter_),含no_top_k_mask额外处理 |
| 4d-ii | logits_to_return (processed模式) | 条件分支,见备注 | - | - | `[B, V]` fp32 | `[B, V]` fp32 | ✅ | processed_logits直接赋值;processed_logprobs需额外log_softmax |
| 4d-iii | softmax | `logits.softmax(dim=-1, dtype=fp32)` | SoftmaxV2 | NPU-Vector | `[B, V]` fp32 | `[B, V]` fp32 | ✅ | 计算概率分布 |
| 4d-iv | exponential_ | `q.exponential_()` | Exponential | **AI-CPU** | `[B, V]` fp32 | `[B, V]` fp32 | ⚠️ | 随机数生成,AI-CPU执行 |
| 4d-v | div+argmax | `probs.div_(q).argmax(dim=-1)` | Div+ArgMaxWithValue | NPU-Vector | `[B, V]` fp32 | `[B]` int64 | ✅ | Gumbel-Max采样核心 |
| 4e | where | `torch.where(temp<EPS, greedy, random, out=greedy)` | Where | NPU-Vector | `[B]` bool + `[B]` int64 ×2 | `[B]` int64 | ✅ | 合并贪心/随机结果, out=复用张量 |
| 5a | long() | `sampled.long()` | Cast | NPU-Vector | `[B]` int32/int64 | `[B]` int64 | ✅ | 统一为int64 |
| 5b | to(int32) | `sampled.to(torch.int32)` | Cast | NPU-Vector | `[B]` int64 | `[B]` int32 | ✅ | 减少张量大小 |
| 6a | unsqueeze | `token_ids.unsqueeze(-1)` | Reshape(view) | NPU-Vector | `[B]` int64 | `[B, 1]` int64 | - | 为gather准备形状 |
| 6b | topk | `torch.topk(logprobs, num_logprobs)` | TopKV2 | NPU-Vector | `[B, V]` fp32 | `[B, K]` fp32 + `[B, K]` int64 | ✅ | logprobs收集 |
| 6c | gather | `logprobs.gather(-1, token_ids)` | GatherV2 | NPU-Vector | `[B, V]` fp32 + `[B, 1]` int64 | `[B, 1]` fp32 | ✅ | 采样token logprob |
| 6d | count_greater | `(x >= values).sum(-1)` | GreaterEqual+ReduceSum | NPU-Vector | `[B, V]` fp32 + `[B, 1]` fp32 | `[B]` int64 | ✅ | torch.compile生成 |
| 6e | cat(indices) | `torch.cat([token_ids, topk_indices])` | ConcatD | NPU-Vector | `[B,1]` + `[B,K]` | `[B, K+1]` int64 | ✅ | 拼接token索引 |
| 6f | cast(indices) | `indices.to(torch.int32)` | Cast | NPU-Vector | `[B, K+1]` int64 | `[B, K+1]` int32 | ✅ | 独立步骤,减少张量大小 |
| 6g | cat(logprobs) | `torch.cat([token_logprobs, topk_logprobs])` | ConcatD | NPU-Vector | `[B,1]` + `[B,K]` | `[B, K+1]` fp32 | ✅ | 拼接logprob值 |
| 7 | unsqueeze | `sampled.unsqueeze(-1)` | Reshape(view) | NPU-Vector | `[B]` int32 | `[B, 1]` int32 | - | 仅形状变换,零开销 |

> **图例**: B=batch_size(num_reqs), V=vocab_size(152064 for Qwen), K=num_logprobs
>
> **特殊分支说明**:
> - `all_greedy=True`: 走快速路径, 在4a后直接返回, 跳过4b-4e
> - `all_random=True`: 完全跳过贪心采样(4a), 直接走随机采样流程
> - `num_logprobs=None`: 跳过6a-6g所有logprobs收集步骤
> - `num_logprobs=-1`: 返回完整logprobs, 不做topk/gather/rank计算
>
> **返回结构**:
> ```
> SamplerOutput:
>   ├─ sampled_token_ids: [B, 1] int32         # 采样的token ID
>   └─ logprobs_tensors: LogprobsTensors | None
>         ├─ indices: [B, K+1] int32            # token索引 (采样token + topk)
>         ├─ logprobs: [B, K+1] fp32            # logprob值
>         ├─ token_ranks: [B] int64             # 每个采样token的排名
>         └─ cu_num_generated_tokens: list[int] | None  # [num_reqs] 累积生成token数(可选)
> ```

#### 3.4.2 数据流详细追踪

```
输入: logits [B, V] fp16/bf16  (来自 compute_logits, 即 lm_head 线性层输出)
 │
 ├─[可选] logprobs预处理 (num_logprobs != None):
 │   │
 │   ├─ if logprobs_mode == "raw_logprobs":
 │   │     raw_logprobs = logits.log_softmax(dim=-1, fp32)
 │   │     算子: LogSoftmaxV2          位置: NPU-Vector
 │   │     数据流: logits[B,V] fp16 → raw_logprobs[B,V] fp32 (含类型提升)
 │   │
 │   └─ elif logprobs_mode == "raw_logits":
 │         if logits.dtype == fp32:
 │             raw_logprobs = logits.clone()
 │             算子: Clone             位置: NPU-Vector (零开销view)
 │         else:
 │             raw_logprobs = logits.to(fp32)
 │             算子: Cast              位置: NPU-Vector
 │         注: raw_logits模式不做log_softmax计算, 只复制/转换类型
 │
 ├─ Cast fp32 ──→ logits [B, V] fp32
 │   算子: Cast                  位置: NPU-Vector
 │   注: 若上一步已执行log_softmax(dtype=fp32), 此处仍需对原始logits做cast
 │
 ├─ apply_logits_processors() ──→ logits [B, V] fp32 (原地修改)
 │   │
 │   ├─ masked_fill_(allowed_mask, -inf) ──→ logits [B, V] fp32  (原地)
 │   │   条件: allowed_token_ids_mask is not None
 │   │   算子: MaskedFill             位置: NPU-Vector
 │   │   数据流: logits[B,V] ⊕ mask[B,V] bool → logits[B,V] (in-place)
 │   │
 │   ├─ apply_bad_words ──→ logits [B, V] fp32  (原地, 逐请求CPU循环)
 │   │   条件: bad_words_token_ids is not empty
 │   │   算子: Python循环 + 索引赋值  位置: CPU(循环) + NPU(索引写)
 │   │   数据流: 逐行 logits[i][token_id] = -inf
 │   │   ⚠️ 性能瓶颈: CPU-NPU交互, 无法批量化
 │   │
 │   ├─ non_argmax_invariant处理器 ──→ logits [B, V] fp32
 │   │   条件: sampling_metadata.logitsprocs.non_argmax_invariant非空
 │   │   算子: 取决于具体处理器(Add/Where等)  位置: NPU-Vector
 │   │   包含: min_tokens_processor, logit_bias_processor
 │   │
 │   └─ apply_penalties ──→ logits [B, V] fp32  (原地)
 │       条件: not sampling_metadata.no_penalties
 │       │
 │       ├─ _convert_to_tensors: list[list[int]] → output_tokens_t [B, max_seq] int64
 │       │   算子: make_tensor_with_pad    位置: CPU(构造) → NPU(传输)
 │       │   数据流: CPU list → CPU tensor(pin_memory) → NPU tensor
 │       │   ⚠️ CPU-NPU数据传输(H2D), 每步都执行
 │       │
 │       ├─ masked_fill_ (-1占位符替换):
 │       │   算子: Compare + MaskedFill     位置: NPU-Vector
 │       │   数据流: output_tokens_t[B, max_seq] → 将-1替换为vocab_size
 │       │   注: 异步调度场景下output_tokens中可能有-1占位符,需先替换
 │       │
 │       ├─ get_token_bin_counts_and_mask (prompt):  ← 第1次调用
 │       │   算子: zeros + scatter_add_ + (>0)  位置: NPU-Vector
 │       │   数据流: prompt_tokens[B, seq_len] → zeros[B, V+1] → scatter_add_ → (>0) → prompt_mask[B, V] bool
 │       │   注: 仅返回mask,不需要bin_counts
 │       │
 │       ├─ get_token_bin_counts_and_mask (output):  ← 第2次独立调用
 │       │   算子: zeros + ones_like + scatter_add_ + (>0)  位置: NPU-Vector
 │       │   数据流: output_tokens[B, max_seq] → zeros[B, V+1] → scatter_add_(ones_like) → output_bin_counts[B, V] + output_mask[B, V] bool
 │       │   注: 同时返回bin_counts(用于frequency_penalty)和mask(用于presence_penalty)
 │       │
 │       ├─ apply_repetition_penalties:  (torch版本, NPU走此路径)
 │       │   算子: Unsqueeze+Repeat+Or+Where+Where+Mul_  位置: NPU-Vector
 │       │   数据流: penalties[B] → unsqueeze+repeat → [B,V]
 │       │           prompt_mask | output_mask → combined_mask[B,V]
 │       │           where(combined_mask, penalties, 1.0) → applied_penalties[B,V]
 │       │           where(logits>0, 1/applied_penalties, applied_penalties) → scaling[B,V]
 │       │           logits *= scaling (in-place)
 │       │
 │       ├─ frequency: logits -= freq_pen.unsqueeze(1) * bin_counts
 │       │   算子: Mul + Sub_             位置: NPU-Vector
 │       │
 │       └─ presence: logits -= pres_pen.unsqueeze(1) * output_mask
 │           算子: Mul + Sub_             位置: NPU-Vector
 │
 ├─[分支] sample() ──→ (sampled, processed_logprobs)
 │   │
 │   ├─[分支1] all_greedy == True:
 │   │   │   注: 全部请求都是贪心采样, 走快速路径, 跳过后续随机采样流程
 │   │   │
 │   │   ├─ greedy_sample(logits) ──→ sampled [B] int64
 │   │   │   算子: ArgMaxWithValue    位置: NPU-Vector
 │   │   │
 │   │   └─[可选] processed_logprobs计算 (num_logprobs != None):
 │   │       ├─ if logprobs_mode == "processed_logits":
 │   │       │     processed_logprobs = logits
 │   │       └─ elif logprobs_mode == "processed_logprobs":
 │   │             processed_logprobs = logits.log_softmax(dim=-1, fp32)
 │   │             算子: LogSoftmaxV2  位置: NPU-Vector
 │   │
 │   │   └─ return sampled, processed_logprobs  ◀─ early return, 跳过后续步骤
 │   │
 │   └─[分支2] all_greedy == False (继续随机采样流程):
 │       │
 │       ├─[子分支2a] all_random == True:
 │       │   │   注: 全部请求都是随机采样, 完全跳过贪心采样计算
 │       │   ├─ greedy_sampled = None  (不执行argmax, 节省算力)
 │       │   └─ 继续执行后续温度缩放和随机采样流程
 │       │
 │       ├─[子分支2b] all_random == False (混合模式):
 │       │   │   注: 部分贪心+部分随机, 需要计算两种结果后合并
 │       │   │
 │       │   ├─ greedy_sample(logits) ──→ greedy_sampled [B] int64
 │       │   │   算子: ArgMaxWithValue        位置: NPU-Vector
 │       │   │   数据流: logits[B,V] fp32 → indices[B] int64
 │       │   │   注: 先计算贪心结果, 后续根据temperature决定是否使用
 │       │   │
 │       │   └─ 继续执行后续温度缩放和随机采样流程
 │       │
 │       ├─ apply_temperature ──→ logits [B, V] fp32  (原地)
 │       │   算子: Where(temp<EPS→1.0) + Div_  位置: NPU-Vector
 │       │   数据流: logits[B,V] ÷ temp[B,1] → logits[B,V]
 │       │   注: 若all_random=False, temp<EPS的行会被替换为1.0避免除零
 │       │
 │       ├─ argmax_invariant处理器(如min_p) ──→ logits [B, V] fp32
 │       │   条件: sampling_metadata.logitsprocs.argmax_invariant非空
 │       │   算子: 自定义                 位置: NPU-Vector
 │       │
 │       ├─ AscendTopKTopPSampler.forward_native():
 │       │   │
 │       │   ├─ apply_top_k_top_p ──→ logits [B, V] fp32  (原地, 被mask的位置=-inf)
 │       │   │   A2/A3路径:
 │       │   │     算子: npu_apply_top_k_top_p (AscendC自定义)  位置: NPU-Vector
 │       │   │     数据流: logits[B,V] + k[B] + p[B] → logits[B,V] (filtered)
 │       │   │   其他设备路径 (PyTorch fallback):
 │       │   │     算子: softmax→sort→gather→compare→masked_fill_(+cumsum for top_p)  位置: NPU-Vector
 │       │   │     数据流: logits[B,V] → probs=softmax(logits) → probs_sort=sort(probs, descending=False)
 │       │   │             → top_k: gather cutoff → compare(probs < cutoff) → masked_fill_(logits, -inf)
 │       │   │             → top_p: cumsum(probs_sort) → gather cutoff → compare → masked_fill_(logits, -inf)
 │       │   │     注: 直接修改原始logits,无scatter_操作;含no_top_k_mask处理(k==vocab_size时no-op)
 │       │   │
 │       │   ├─ logits_to_return 处理 (processed模式):
 │       │   │   ├─ if logprobs_mode == "processed_logits":
 │       │   │   │     logits_to_return = logits  (直接赋值, 零开销)
 │       │   │   └─ elif logprobs_mode == "processed_logprobs":
 │       │   │         logits_to_return = logits.log_softmax(dim=-1, fp32)
 │       │   │         算子: LogSoftmaxV2          位置: NPU-Vector
 │       │   │         数据流: logits[B,V] fp32 → logits_to_return[B,V] fp32
 │       │   │         注: 用于返回处理后的logprobs, 替代raw_logprobs模式
 │       │   │
 │       │   ├─ softmax ──→ probs [B, V] fp32
 │       │   │   算子: SoftmaxV2            位置: NPU-Vector
 │       │   │   数据流: logits[B,V] fp32 → probs[B,V] fp32
 │       │   │
 │       │   ├─[异步路径] enable_async_exponential=True:
 │       │   │   │ event.synchronize()  ← 等待预计算的q就绪
 │       │   │   │ 算子: EventSynchronize   位置: Host(CPU)同步
 │       │   │   │
 │       │   │   └─ probs.div_(q).argmax(dim=-1).view(-1)
 │       │   │       算子: Div_ + ArgMaxWithValue  位置: NPU-Vector
 │       │   │       数据流: probs[B,V] ÷ q[B,V] → ratios[B,V] → indices[B] int64
 │       │   │
 │       │   └─[同步路径] enable_async_exponential=False:
 │       │       │ random_sample(probs, generators):
 │       │       │
 │       │       ├─ npu_stream_switch(global_stream())  位置: Host(CPU)
 │       │       ├─ empty_like ──→ q [B, V] fp32
 │       │       │   算子: Empty              位置: NPU(内存分配)
 │       │       ├─ q.exponential_()
 │       │       │   算子: Exponential         位置: **AI-CPU**
 │       │       │   ⚠️ AI-CPU执行,速度较慢,是Triton改造重点
 │       │       ├─ [可选] q[i].exponential_(generator=gen)
 │       │       │   算子: Exponential(per-row) 位置: **AI-CPU**
 │       │       ├─ wait_stream同步            位置: Host(CPU)
 │       │       └─ probs.div_(q).argmax(dim=-1).view(-1)
 │       │           算子: Div_ + ArgMaxWithValue  位置: NPU-Vector
 │       │
 │       └─[可选] torch.where(temp<EPS, greedy, random) ──→ sampled [B] int64
 │           条件: greedy_sampled is not None (即 not all_random)
 │           算子: Where                  位置: NPU-Vector
 │           注: out=greedy_sampled 复用张量, 避免新分配
 │
 ├─[可选] processed_logprobs 替换:
 │   if processed_logprobs is not None:
 │       raw_logprobs = processed_logprobs
 │   注: 用processed_logprobs替换raw_logprobs, 用于后续gather_logprobs
 │
 ├─ sampled.long() ──→ sampled [B] int64
 │   算子: Cast (int32→int64 或 保持int64)
 │   注: FlashInfer返回int32, PyTorch argmax返回int64, 统一转int64
 │
 ├─[分支] logprobs收集:
 │   │
 │   ├─ if num_logprobs == None:
 │   │     logprobs_tensors = None
 │   │     注: 不需要收集logprobs, 跳过后续步骤
 │   │
 │   ├─ elif num_logprobs == -1:
 │   │     logprobs_tensors = LogprobsTensors(empty, raw_logprobs, empty)
 │   │     注: 返回完整logprobs, 不做topk/gather/rank计算
 │   │
 │   └─ else:  # num_logprobs > 0
 │         gather_logprobs():
 │         │
 │         ├─ token_ids.unsqueeze(-1) ──→ token_ids [B, 1] int64
 │         │   算子: Reshape(view)       位置: NPU-Vector (零开销)
 │         │   注: 为gather操作准备形状
 │         │
 │         ├─ topk(raw_logprobs, K) ──→ topk_logprobs [B, K] fp32, topk_indices [B, K] int64
 │         │   算子: TopKV2              位置: NPU-Vector
 │         │   数据流: raw_logprobs[B,V] → topk_values[B,K] + topk_indices[B,K]
 │         │
 │         ├─ gather(logprobs, token_ids) ──→ token_logprobs [B, 1] fp32
 │         │   算子: GatherV2            位置: NPU-Vector
 │         │   数据流: logprobs[B,V] ⊕ token_ids[B,1] → token_logprobs[B,1]
 │         │
 │         ├─ batched_count_greater_than ──→ token_ranks [B] int64
 │         │   算子: GreaterEqual + ReduceSum  位置: NPU-Vector (torch.compile)
 │         │   数据流: logprobs[B,V] ≥ token_logprobs[B,1] → bool[B,V] → sum → [B]
 │         │
 │         ├─ cat([token_ids, topk_indices]) ──→ indices [B, K+1] int64
 │         │   算子: ConcatD             位置: NPU-Vector
 │         │
 │         ├─ indices.to(int32) ──→ indices [B, K+1] int32
 │         │   算子: Cast                位置: NPU-Vector
 │         │   注: 独立的cast步骤, 减少张量大小
 │         │
 │         └─ cat([token_logprobs, topk_logprobs]) ──→ logprobs [B, K+1] fp32
 │             算子: ConcatD            位置: NPU-Vector
 │
 │         └─ return LogprobsTensors:
 │               ├─ indices: [B, K+1] int32  (token_ids + topk_indices)
 │               ├─ logprobs: [B, K+1] fp32  (token_logprobs + topk_logprobs)
 │               └─ token_ranks: [B] int64   (每个采样token的排名)
 │
 ├─ sampled.to(int32) ──→ sampled [B] int32
 │   算子: Cast                      位置: NPU-Vector
 │   注: 减少张量大小
 │
 └─ 构建 SamplerOutput 并返回:
     │
     ├─ sampled_token_ids = sampled.unsqueeze(-1)  ──→ [B, 1] int32
     │   算子: Reshape(view)         位置: NPU-Vector (零开销)
     │
     └─ return SamplerOutput(
           sampled_token_ids: [B, 1] int32,      # 采样的token ID
           logprobs_tensors: LogprobsTensors | None,  # logprobs信息(可选)
         )
```

#### 3.4.3 运行位置分布统计

```
┌─────────────────────────────────────────────────────────────────────┐
│                    算子运行位置分布                                    │
├─────────────┬───────────────────────────────────────────────────────┤
│  NPU-Vector │ ████████████████████████████████████████  ~85%        │
│             │ log_softmax, cast, masked_fill, softmax,             │
│             │ argmax, div, where, topk, gather, scatter_add,       │
│             │ sort, cumsum, cat, reduce_sum, mul, sub              │
├─────────────┼───────────────────────────────────────────────────────┤
│  AI-CPU     │ ██████  ~8%                                          │
│             │ exponential_(随机数生成)                               │
│             │ ⚠️ 单算子但vocab_size大(152064)时耗时显著              │
├─────────────┼───────────────────────────────────────────────────────┤
│  Host CPU   │ ████  ~5%                                            │
│             │ apply_bad_words(Python循环), _convert_to_tensors      │
│             │ (list→tensor), 流切换/同步控制                         │
├─────────────┼───────────────────────────────────────────────────────┤
│  NPU-Cube   │ ▏ ~0%                                                │
│             │ 采样阶段无矩阵乘算子, Cube单元空闲                     │
├─────────────┼───────────────────────────────────────────────────────┤
│  H2D传输    │ ██  ~2%                                              │
│             │ output_tokens CPU→NPU (apply_penalties每步传输)        │
└─────────────┴───────────────────────────────────────────────────────┘
```

#### 3.4.4 Triton 改造机会分析

**融合机会 1: logits预处理融合**（优先级：★★★★★）

```
当前: Cast(fp16→fp32) → masked_fill_ → logit_bias(Add) → penalties(scatter+where+mul+sub×2)
目标: 单个 Triton kernel 完成全部 logits 预处理
收益: 减少 5~8 次 kernel launch + 消除中间张量 [B,V] 多次读写
输入: logits[B,V] fp16, mask[B,V], bias, penalties参数
输出: logits[B,V] fp32 (processed)
位置: NPU-Vector → Triton(Vector)
```

**融合机会 2: temperature + top_k_top_p + softmax + sampling 融合**（优先级：★★★★★）

```
当前: div_(temp) → [min_p] → sort/mask(top_k_top_p) → softmax → exponential_ → div_ → argmax
目标: 单个 Triton kernel: temp_scale → top_k_top_p_filter → softmax → gumbel_argmax
收益: 这是采样热路径, 融合后消除 sort 的 O(V·logV) 中间结果写回
      exponential 从 AI-CPU 迁移到 Triton(Vector), 消除 AI-CPU 调度开销
输入: logits[B,V] fp32, temp[B], k[B], p[B]
输出: sampled[B] int64
位置: NPU-Vector + AI-CPU → Triton(Vector)
关键难点: top_k_top_p 涉及排序, Triton 中需用近似排序或分桶策略
```

**融合机会 3: logprobs 收集融合**（优先级：★★★☆☆）

```
当前: topk → gather → count_greater_than → cat ×2 → cast
目标: 单个 Triton kernel 同时完成 topk + gather + rank计算 + 拼接
收益: 减少 5 次 kernel launch, topk 和 count_greater 都需全量扫描 logprobs[B,V]
      融合后只需一次扫描
输入: raw_logprobs[B,V] fp32, sampled[B] int64, K
输出: indices[B,K+1] int32, logprobs[B,K+1] fp32, ranks[B] int64
位置: NPU-Vector → Triton(Vector)
```

**融合机会 4: log_softmax 与 Cast 融合**（优先级：★★☆☆☆）

```
当前: logits.log_softmax(dim=-1, dtype=fp32) 单独执行 + logits.to(fp32) 单独执行
目标: 一次读取 logits fp16, 同时输出 raw_logprobs fp32 和 logits fp32
收益: 减少一次 [B,V] 的全量读取
位置: NPU-Vector → Triton(Vector)
```

#### 3.4.5 关键性能瓶颈标注

| 瓶颈点 | 原因 | 影响程度 | Triton改造方案 |
|--------|------|---------|---------------|
| `exponential_()` 在 AI-CPU 执行 | AI-CPU 调度开销大, 无法利用 Vector 单元 | ★★★★★ | Triton 内使用 `tl.rand` + 变换生成指数分布 |
| `apply_bad_words` CPU循环 | Python for循环逐请求处理, CPU-NPU交互 | ★★★★☆ | 构造batch mask后一次masked_fill_, 或Triton kernel |
| `_convert_to_tensors` H2D传输 | 每步将output_token_ids从CPU传到NPU | ★★★★☆ | 在NPU侧维护token_ids缓存,避免重复传输 |
| `sort` in top_k_top_p (非A2/A3) | 全词表排序 O(V·logV), V=152064 | ★★★★☆ | Triton partial sort / 分桶 top-k |
| 多次 kernel launch | 单步多算子串行执行, launch开销累积 | ★★★☆☆ | Triton算子融合减少launch次数 |
| `scatter_add_` in penalties | 构造bin_counts需scatter操作 | ★★☆☆☆ | 与penalties计算融合到同一Triton kernel |

#### 3.4.6 Triton改造优先级路线图

```
Phase 1 (高收益,低难度):
  ├── exponential_ → tl.rand + 变换 (消除AI-CPU依赖)
  ├── cast + log_softmax 融合
  └── logprobs收集融合 (topk+gather+rank)

Phase 2 (高收益,中难度):
  ├── penalties全融合 (scatter_add + repetition + frequency + presence)
  ├── temperature + softmax + gumbel_argmax 融合
  └── bad_words batch化改造

Phase 3 (最高收益,高难度):
  ├── temperature + top_k_top_p + softmax + sampling 端到端融合
  └── 全流程单kernel: logits_preprocess + sample + logprobs_gather
```

---

## 4. 功能流程

### 4.1 Logits处理管道

```
logits [num_reqs, vocab_size]
    │
    ├── 1. compute_logprobs()
    │       logits.log_softmax(dim=-1)
    │
    ├── 2. apply_logits_processors()
    │       ├── allowed_token_ids whitelist
    │       ├── bad_words exclusion
    │       ├── min_tokens processor
    │       ├── logit_bias processor
    │       └── penalties (repetition/frequency/presence)
    │
    └── 3. sample()
            ├── 贪心采样: argmax
            └── 随机采样: temperature → top_k/top_p → multinomial
```

### 4.2 采样方法

**贪心采样** (`sampler.py:144-145`):
```python
def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1).view(-1)
```

**随机采样** (`sampler.py:147-203`):
```python
def sample(self, logits, sampling_metadata):
    # 1. 贪心采样（如果需要）
    if not sampling_metadata.all_random:
        greedy_sampled = self.greedy_sample(logits)
        if sampling_metadata.all_greedy:
            return greedy_sampled, None

    # 2. 应用温度
    logits = self.apply_temperature(logits, temperature, all_random)

    # 3. 应用argmax-invariant处理器
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)

    # 4. 应用top_k和top_p
    random_sampled, processed_logprobs = self.topk_topp_sampler(
        logits, generators, top_k, top_p
    )

    # 5. 合并结果
    sampled = torch.where(temperature < EPS, greedy_sampled, random_sampled)
    return sampled, processed_logprobs
```

### 4.3 Ascend优化

**异步指数分布计算** (`sampler.py:47-60`):
```python
def do_async_exponential(self, b_s, head_dim, generators):
    # 在不同流中计算指数随机数，与模型执行重叠
    with torch.npu.stream(global_stream()):
        q = torch.empty((b_s, head_dim), device="npu", dtype=torch.float32)
        q.exponential_()
        self.async_exponential_event.record()
```

**TopK/TopP采样** (`sampler.py:74-88`):
```python
def forward_native(self, logits, generators, k, p):
    logits = self.apply_top_k_top_p(logits, k, p)  # AscendC算子
    probs = logits.softmax(dim=-1)
    if get_ascend_config().enable_async_exponential:
        self.async_event.synchronize()
        return probs.div_(self.q).argmax(dim=-1)  # Gumbel-max采样
    return random_sample(probs, generators)
```

---

## 5. 数据流程

### 5.1 完整数据流

```
┌─────────────────────────────────────────────────────────────────┐
│ execute_model() 完成                                            │
│ 输出: hidden_states [num_tokens, hidden_size]                   │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ compute_logits()                                                │
│ logits = model.compute_logits(hidden_states[logits_indices])    │
│ 输出: logits [num_reqs, vocab_size]                             │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ sample_tokens()                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ _sample(logits, spec_decode_metadata=None)                  │ │
│ │ ┌─────────────────────────────────────────────────────────┐ │ │
│ │ │ AscendSampler.forward(logits, sampling_metadata)        │ │ │
│ │ │ ├── apply_logits_processors()                           │ │ │
│ │ │ ├── apply_temperature()                                 │ │ │
│ │ │ ├── apply_top_k_top_p() [AscendC优化]                   │ │ │
│ │ │ └── random_sample() / greedy_sample()                   │ │ │
│ │ └─────────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ _bookkeeping_sync()                                             │
│ ├── 解析采样结果: valid_sampled_token_ids                       │
│ ├── 更新请求状态: req_state.output_token_ids.extend()           │
│ └── 更新token缓存: token_ids_cpu[req_idx, start:end] = sampled  │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ ModelRunnerOutput                                               │
│ ├── req_ids: list[str]                                          │
│ ├── sampled_token_ids: list[list[int]]  ← [batch_size, 1]       │
│ ├── logprobs: LogprobsLists | None                              │
│ └── ...                                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 张量形状变化

```
hidden_states: [num_tokens, hidden_size]
      ↓ compute_logits()
logits: [num_reqs, vocab_size]
      ↓ apply_logits_processors()
processed_logits: [num_reqs, vocab_size]
      ↓ sample()
sampled_token_ids: [num_reqs] → [num_reqs, 1]
      ↓ _to_list()
valid_sampled_token_ids: list[list[int]]  # 每个请求一个token列表
```

---

# 第二部分：投机推理后处理

## 1. 初始化

### 1.1 RejectionSampler 类定义

**文件位置**: `vllm/v1/sample/rejection_sampler.py`

```python
class RejectionSampler(nn.Module):
    def __init__(self, sampler: Sampler):
        super().__init__()
        self.sampler = sampler  # 复用传统采样器
        logprobs_mode = self.sampler.logprobs_mode
        self.is_processed_logprobs_mode = logprobs_mode.startswith("processed")
        self.is_logits_logprobs_mode = logprobs_mode.endswith("logits")
```

### 1.2 RejectionSampler在ModelRunner中的初始化

**文件位置**: `vllm_ascend/worker/model_runner_v1.py:415`

```python
if self.speculative_config:
    if get_pp_group().is_last_rank:
        self.drafter = self._get_drafter()  # 初始化draft proposer
        self.rejection_sampler = RejectionSampler(self.sampler)
```

### 1.3 SpecDecodeMetadata 数据结构

**文件位置**: `vllm/v1/spec_decode/metadata.py`

```python
@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor       # [num_draft_tokens] 所有draft token IDs
    num_draft_tokens: list[int]         # [batch_size] 每个请求的draft数量
    cu_num_draft_tokens: torch.Tensor   # [batch_size] 累积draft数量（前缀和）
    cu_num_sampled_tokens: torch.Tensor # [batch_size] 累积采样数量
    target_logits_indices: torch.Tensor # [num_draft_tokens] target logits索引
    bonus_logits_indices: torch.Tensor  # [batch_size] bonus logits索引
    logits_indices: torch.Tensor        # [total_tokens] logits索引

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens)  # 最大投机长度
```

---

## 2. 配置

### 2.1 投机解码配置

**文件位置**: `vllm/spec_decode/config.py`

| 配置项 | 说明 |
|--------|------|
| `method` | 投机方法: "ngram", "eagle", "medusa", "mtp", "suffix" |
| `num_speculative_tokens` | 每步生成的draft token数量 |
| `draft_model_config` | draft模型配置（如果使用独立draft模型） |

### 2.2 Draft Proposer类型

| Proposer | 文件位置 | 说明 |
|----------|----------|------|
| NgramProposer | `vllm/v1/spec_decode/ngram_proposer.py` | N-gram匹配，无概率分布 |
| EagleProposer | `vllm_ascend/spec_decode/eagle_proposer.py` | EAGLE模型预测 |
| MtpProposer | `vllm_ascend/spec_decode/mtp_proposer.py` | Multi-Token Prediction |
| MedusaProposer | `vllm_ascend/spec_decode/medusa_proposer.py` | Medusa多头预测 |
| SuffixDecodingProposer | `vllm/v1/spec_decode/suffix_decoding.py` | 后缀解码 |

### 2.3 SpecDecodeMetadata构建

**文件位置**: `vllm_ascend/worker/model_runner_v1.py:856-932`

```python
def _calc_spec_decode_metadata(self, num_draft_tokens, cu_num_scheduled_tokens, ...):
    # 计算采样token数量
    num_sampled_tokens = num_draft_tokens + 1
    cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)

    # 计算bonus logits索引（每个请求的最后一个位置）
    bonus_logits_indices = cu_num_sampled_tokens - 1

    # 计算target logits索引
    cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
    target_logits_indices = ...  # 每个draft token对应的位置

    # 获取draft token IDs
    draft_token_ids = self.input_ids.gpu[logits_indices]
    draft_token_ids = draft_token_ids[target_logits_indices + 1]

    return SpecDecodeMetadata(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens.tolist(),
        cu_num_draft_tokens=cu_num_draft_tokens,
        cu_num_sampled_tokens=cu_num_sampled_tokens,
        target_logits_indices=target_logits_indices,
        bonus_logits_indices=bonus_logits_indices,
        logits_indices=logits_indices,
    )
```

---

## 3. 入口函数和执行流程

### 3.1 入口函数

**文件位置**: `vllm_ascend/worker/model_runner_v1.py:1548-1556`

```python
def _sample(self, logits, spec_decode_metadata):
    # ...
    if spec_decode_metadata is not None:  # 投机推理后处理路径
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs (N-gram模式为None)
            logits,
            sampling_metadata,
        )
        return sampler_output
```

### 3.2 完整执行流程

```
sample_tokens() [model_runner_v1.py:1397]
    │
    ├── 1. 解包 execute_model_state
    │       └── 获取 logits, spec_decode_metadata
    │
    ├── 2. 调用 _sample()
    │       │
    │       └── RejectionSampler.forward() [rejection_sampler.py]
    │               │
    │               ├── 2.1 采样bonus tokens
    │               │       sampler(bonus_logits)
    │               │
    │               ├── 2.2 处理target logits
    │               │       apply_logits_processors()
    │               │       apply_sampling_constraints()
    │               │
    │               ├── 2.3 调用 rejection_sample()
    │               │       ├── 贪心采样路径
    │               │       └── 随机采样路径
    │               │
    │               └── 2.4 返回 SamplerOutput
    │
    ├── 3. _bookkeeping_sync()
    │       └── parse_output() 解析采样结果
    │
    ├── 4. 生成下一轮draft tokens
    │       └── propose_draft_token_ids()
    │
    └── 5. 返回 ModelRunnerOutput
```

### 3.3 RejectionSampler.forward() 详细流程

**文件位置**: `vllm/v1/sample/rejection_sampler.py:24-120`

```python
def forward(self, metadata, draft_probs, logits, sampling_metadata):
    # 1. 提取bonus logits并采样bonus tokens
    bonus_logits = logits[metadata.bonus_logits_indices]
    bonus_sampler_output = self.sampler(
        logits=bonus_logits,
        sampling_metadata=replace(sampling_metadata, max_num_logprobs=-1),
        predict_bonus_token=True,
    )
    bonus_token_ids = bonus_sampler_output.sampled_token_ids

    # 2. 提取并处理target logits
    target_logits = logits[metadata.target_logits_indices]
    target_logits = self.apply_logits_processors(target_logits, sampling_metadata, metadata)
    target_logits = apply_sampling_constraints(
        target_logits, metadata.cu_num_draft_tokens, sampling_metadata
    )

    # 3. 执行拒绝采样
    output_token_ids = rejection_sample(
        metadata.draft_token_ids,
        metadata.num_draft_tokens,
        metadata.max_spec_len,
        metadata.cu_num_draft_tokens,
        draft_probs,
        target_logits,
        bonus_token_ids,
        sampling_metadata,
    )

    # 4. 计算logprobs（如果需要）
    logprobs_tensors = self._get_logprobs_tensors(...)

    return SamplerOutput(
        sampled_token_ids=output_token_ids,
        logprobs_tensors=logprobs_tensors,
    )
```

---

### 3.4 提取 Bonus Logits 并采样 Bonus Tokens

**文件位置**: `vllm/v1/sample/rejection_sampler.py:93-115`

> **注意**: 此代码来自 vLLM 官方实现 (`vllm/v1/sample/rejection_sampler.py`)，vllm-ascend 直接复用此实现，仅提供底层的拒绝采样函数 (`rejection_sample` 等)。

#### 3.4.1 概述

Bonus Token 是投机解码中的一个特殊概念。当一个请求的**所有 draft tokens 都被接受**时，target 模型会获得一个额外的"奖励"token（bonus token），这相当于在标准自回归解码中额外生成的一个 token。

#### 3.4.2 数据结构

```
bonus_logits_indices: torch.Tensor  # [batch_size] bonus位置的logits索引
bonus_token_ids: torch.Tensor       # [batch_size, 1] 采样到的bonus tokens
```

#### 3.4.3 完整处理流程

```python
# ==================== 步骤1: 提取 Bonus Logits ====================
# 从完整的logits张量中提取bonus位置的logits
bonus_logits_indices = metadata.bonus_logits_indices
bonus_logits = logits[bonus_logits_indices]  # [batch_size, vocab_size]

# ==================== 步骤2: 构建 Bonus 采样的元数据 ====================
# 创建用于bonus token采样的元数据
# max_num_logprobs=-1 表示返回完整logprobs（用于后续计算接受token的logprobs）
bonus_sampling_metadata = replace(
    sampling_metadata,
    max_num_logprobs=-1,
)

# ==================== 步骤3: 调用 Sampler 采样 Bonus Token ====================
# Sampler 接口说明（vLLM 官方实现）
#
# self.sampler 实际上是 AscendSampler（vllm-ascend 封装），其 forward 方法执行以下步骤：
# 1. compute_logprobs(logits): 计算原始 logprobs（如果需要）
# 2. logits.to(float32): 转换为 float32 提高精度
# 3. apply_logits_processors(): 应用 logits 处理器（惩罚项、坏词过滤等）
# 4. sample(): 执行采样（贪心或随机）
# 5. gather_logprobs(): 收集采样 token 的 logprobs
#
# 具体参数说明：
# - logits: bonus 位置的 logits，形状 [batch_size, vocab_size]
# - sampling_metadata: 采样元数据（temperature, top_k, top_p, generators 等）
# - predict_bonus_token: 标记为 bonus token 采样，用于特殊处理
# - logprobs_mode_override: 覆盖默认 logprobs 模式，返回处理后的 logits 用于后续计算
bonus_sampler_output = self.sampler(
    logits=bonus_logits,
    sampling_metadata=bonus_sampling_metadata,
    predict_bonus_token=True,  # 标记这是bonus token采样
    # 覆盖logprobs模式，返回processed logits用于计算logprobs
    logprobs_mode_override="processed_logits" if self.is_processed_logprobs_mode else "raw_logits"
)

# ==================== 步骤4: 提取采样结果 ====================
bonus_token_ids = bonus_sampler_output.sampled_token_ids  # [batch_size, 1]
```

#### 3.4.4 关键设计要点

| 要点 | 说明 |
|-----|------|
| **何时使用** | 只有当一个请求的**所有 draft tokens 都被接受**时才使用 |
| **位置** | Bonus token 放在输出的**最后一个位置**（索引 = max_spec_len） |
| **形状** | `bonus_token_ids.shape = [batch_size, 1]` |
| **Logprobs** | 需要保存 bonus_logits 用于后续计算输出 logprobs |

#### 3.4.5 处理流程图

```
logits [num_tokens + batch_size, vocab_size]
    │
    ├── target_logits_indices ──▶ target_logits ──▶ 拒绝采样
    │
    └── bonus_logits_indices ──▶ bonus_logits ──▶ Sampler 采样
                                                    │
                                                    ▼
                                           bonus_token_ids [batch_size, 1]
                                                    │
                                                    ▼
                                           当所有draft被接受时:
                                           output[:, max_spec_len] = bonus_token_id
```

---

### 3.5 提取并处理 Target Logits

**文件位置**: `vllm/v1/sample/rejection_sampler.py:117-139`

> **注意**: 此代码来自 vLLM 官方实现 (`vllm/v1/sample/rejection_sampler.py`)，vllm-ascend 直接复用此实现，仅提供底层的 `apply_sampling_constraints` 等函数。

#### 3.5.1 概述

Target Logits 是 target 模型对 draft 位置的输出，用于与 draft tokens 进行对比验证。这是投机解码的核心数据。

#### 3.5.2 数据结构

```
target_logits_indices: torch.Tensor   # [num_tokens] target位置的logits索引
target_logits: torch.Tensor           # [num_tokens, vocab_size]
raw_target_logits: torch.Tensor       # 处理前的原始logits（用于logprobs计算）
```

#### 3.5.3 完整处理流程

```python
# ==================== 步骤1: 提取 Target Logits ====================
target_logits_indices = metadata.target_logits_indices

# 从完整logits中提取target位置的logits
# 注意: PyTorch索引会创建新的张量，不会影响原始logits
raw_target_logits = logits[target_logits_indices]  # [num_tokens, vocab_size]

# ==================== 步骤2: 类型转换 ====================
# 使用float32进行后续计算，提高精度
raw_target_logits = raw_target_logits.to(torch.float32)
target_logits = raw_target_logits

# ==================== 步骤3: 克隆原始logits ====================
# 保存原始logits用于后续logprobs计算，因为apply_logits_processors会修改张量
if not self.is_processed_logprobs_mode:
    target_logits = target_logits.clone()

# ==================== 步骤4: 应用 Logits Processors ====================
# apply_logits_processors 接口说明（vLLM 官方 Sampler 实现）
#
# 函数签名：
#   def apply_logits_processors(
#       self,
#       logits: torch.Tensor,           # 输入 logits [num_tokens, vocab_size]
#       sampling_metadata: SamplingMetadata,
#       predict_bonus_token: bool,
#   ) -> torch.Tensor
#
# 功能流程：
# 1. 应用 allowed_token_ids_mask: 将不在白名单的 token 概率设为 -inf
# 2. 应用 bad_words: 排除禁用词（包含禁用词的 token 概率设为 -inf）
# 3. 应用 non_argmax_invariant processors: 用户自定义的处理器
# 4. 应用惩罚项 (apply_penalties):
#    - repetition_penalty: 重复惩罚
#    - frequency_penalty: 频率惩罚
#    - presence_penalty: 存在惩罚
#
# 具体代码位置：vllm/v1/sample/sampler.py:266-300
#
# 应用用户自定义的logits处理器（如自定义约束、过滤等）
target_logits = self.apply_logits_processors(
    target_logits,
    sampling_metadata,
    metadata  # 传入metadata用于特殊处理
)

# ==================== 步骤5: 应用采样约束 ====================
# 应用温度缩放、Top-K、Top-P等采样约束
# 注意: 这个函数可能in-place修改target_logits
target_logits = apply_sampling_constraints(
    target_logits,
    metadata.cu_num_draft_tokens,  # 累积draft token数量
    sampling_metadata,
)
# target_logits 形状: [num_tokens, vocab_size]
```

#### 3.5.4 apply_sampling_constraints 详细说明

这是 vllm-ascend 特有的实现，位于 `vllm_ascend/sample/rejection_sampler.py:24`

```python
def apply_sampling_constraints(
    logits: torch.Tensor,              # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor, # [batch_size] 累积draft token数量
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """
    对target logits应用采样约束。

    处理流程:
    1. 温度缩放: logits / temperature
    2. Top-K 过滤: 只保留概率最高的k个token
    3. Top-P 过滤: 只保留累积概率达到p的最小token集合

    特殊情况:
    - 贪心采样(temperature=0): 直接返回原始logits，不做任何处理
    """
    # 检查是否全部贪心采样（temperature=0）
    if sampling_metadata.all_greedy:
        return logits  # 贪心不需要任何处理

    # 扩展temperature到token级别
    temperature = sampling_metadata.temperature  # [batch_size]
    expanded_temperature = temperature[cu_num_draft_tokens]  # [num_tokens]

    # 步骤1: 温度缩放
    logits = logits / expanded_temperature.unsqueeze(dim=1)

    # 步骤2: Top-K / Top-P 过滤
    k = sampling_metadata.top_k  # [batch_size]
    p = sampling_metadata.top_p  # [batch_size]

    # 扩展到token级别
    expanded_k = k[cu_num_draft_tokens]  # [num_tokens]
    expanded_p = p[cu_num_draft_tokens]  # [num_tokens]

    # 调用Ascend优化的TopK/TopP算子
    logits = apply_top_k_top_p(logits, expanded_k, expanded_p)

    return logits
```

#### 3.5.5 处理流程图

```
logits [num_tokens + batch_size, vocab_size]
    │
    └── target_logits_indices ──▶ raw_target_logits [num_tokens, vocab_size]
                                      │
                                      ▼
                               to(torch.float32)
                                      │
                                      ▼
                               克隆 (保留原始值)
                                      │
                                      ▼
                               apply_logits_processors()
                                      │
                                      ▼
                               apply_sampling_constraints()
                               ├── 温度缩放
                               ├── Top-K 过滤
                               └── Top-P 过滤
                                      │
                                      ▼
                               target_logits [num_tokens, vocab_size]
                                      │
                                      ▼
                               rejection_sample() 进行拒绝验证
```

#### 3.5.6 Target Logits 与 Bonus Logits 的区别

| 特性 | Target Logits | Bonus Logits |
|-----|--------------|--------------|
| **位置** | draft 位置 | 非 draft 位置 |
| **数量** | num_tokens 个 | batch_size 个 |
| **用途** | 与 draft tokens 比较验证 | 生成 bonus token |
| **处理** | 需要应用完整的采样约束 | 只需要基础采样 |
| **输出位置** | 输出数组的前 max_spec_len 列 | 输出数组的第 max_spec_len+1 列 |

---

## 4. 功能流程

### 4.1 拒绝采样核心算法

**文件位置**: `vllm_ascend/sample/rejection_sampler.py:120-387`

```
rejection_sample()
    │
    ├── 输入:
    │   ├── draft_token_ids: [num_tokens] 展平的draft tokens
    │   ├── num_draft_tokens: [batch_size] 每个请求的draft数量
    │   ├── target_logits: [num_tokens, vocab_size]
    │   └── bonus_token_ids: [batch_size, 1]
    │
    ├── 验证模式选择:
    │   ├── max_spec_len < 3 → 逐个验证
    │   └── max_spec_len >= 3 → Block Verify（累积乘积）
    │
    ├── 贪心采样路径:
    │   ├── 比较 draft_token vs target_argmax
    │   ├── 从第一个不匹配位置开始拒绝
    │   └── 全部匹配时追加bonus token
    │
    ├── 随机采样路径:
    │   ├── 预计算recovered tokens（修正分布采样）
    │   ├── 计算接受概率: accept_prob = min(1, p_target/p_draft)
    │   ├── 生成均匀随机数判断接受/拒绝
    │   └── 拒绝时使用recovered token
    │
    └── 输出:
        └── output_token_ids: [batch_size, max_spec_len+1]
            ├── 被接受位置: draft token
            ├── 第一个拒绝位置: recovered token
            ├── 全部接受时: bonus token在末尾
            └── 无效位置: PLACEHOLDER_TOKEN_ID (-1)
```

### 4.2 贪心采样拒绝采样

**文件位置**: `vllm_ascend/sample/rejection_sampler.py:641-803`

```python
def rejection_greedy_sample_pytorch(...):
    # 1. 计算target argmax
    target_argmax = target_logits.argmax(dim=-1)

    # 2. 比较draft和target
    mismatch_global = draft_token_ids != target_argmax

    # 3. 找到每个请求的第一个不匹配位置
    first_mismatch_pos_per_req = ...

    # 4. 复制匹配的tokens到输出
    copy_len = min(first_mismatch_pos + 1, draft_tokens_per_req)
    output_token_ids[...] = target_argmax[...]

    # 5. 如果全部匹配，填充bonus token
    if first_mismatch_pos >= draft_tokens_per_req:
        output_token_ids[req_idx, draft_tokens] = bonus_token_ids[req_idx]
```

### 4.3 随机采样拒绝采样

**文件位置**: `vllm_ascend/sample/rejection_sampler.py:805-996`

```python
def rejection_random_sample_pytorch(...):
    # 1. 计算接受条件
    acceptance_condition = (draft_token_probs > 0) & (
        target_token_probs / draft_token_probs >= uniform_token_probs
    )

    # 2. 找到第一个拒绝位置
    first_rejection = (~acceptance_condition) & valid_mask
    first_reject_pos = first_rejection.float().argmax(dim=1)

    # 3. 创建跳过掩码（第一个拒绝位置之后的都跳过）
    pos_mask = pos_indices >= first_reject_pos
    should_skip = pos_mask & valid_mask

    # 4. 选择最终tokens
    final_tokens = torch.where(
        first_reject_mask, recovered_tokens,
        torch.where(final_acceptance, draft_tokens, output_token_ids)
    )

    # 5. 填充bonus tokens（如果没有拒绝）
    no_rejection = first_reject_pos >= num_draft_per_batch
    should_add_bonus = non_greedy_mask & no_rejection
```

### 4.4 Block Verify模式

**文件位置**: `vllm_ascend/sample/rejection_sampler.py:1231-1383`

```python
def rejection_random_sample_block_verify_pytorch(...):
    # Block Verify使用累积乘积提高接受率
    # 数学原理: ∏(p_target/p_draft) >= ∏uniform

    # 1. 计算π = p_target / p_draft
    pi = target_token_probs / draft_token_probs
    pi = pi.clamp(max=1.0)

    # 2. 计算累积乘积
    pi = torch.cumprod(pi, dim=-1)
    uniform_token_probs = torch.cumprod(uniform_token_probs, dim=-1)

    # 3. 判断合法性
    legal_mask = (draft_token_probs > 0) & (pi >= uniform_token_probs)

    # 4. 找到最后一个接受位置
    last_accept_pos = max_spec_len - legal_mask.flip(dims=[-1]).float().argmax(dim=-1) - 1

    # 5. 填充输出
    accept_mask = (pos_indices <= last_accept_pos) & valid_mask
    reject_mask = (pos_indices == last_accept_pos + 1) & valid_mask
```

### 4.5 Recovered Token采样

**文件位置**: `vllm_ascend/sample/rejection_sampler.py:456-574`

```python
def sample_recovered_tokens(...):
    """
    当draft token被拒绝时，从修正分布中采样恢复token。
    修正分布: P_recover(x) = max(0, p_target(x) - p_draft(x)) / Z
    使用Gumbel-max技巧进行高效采样。
    """
    # 1. 生成指数分布随机数 q ~ Exp(1)
    q = torch.empty((batch_size, vocab_size), device=device)
    q.exponential_()

    # 2. 使用请求专属的随机生成器
    for i, generator in sampling_metadata.generators.items():
        q[i].exponential_(generator=generator)

    # 3. 调用kernel计算recovered tokens
    # 修正概率 = max(0, target_probs - draft_probs)
    # scores = prob / q
    # recovered_id = argmax(scores)
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids, cu_num_draft_tokens, draft_token_ids,
        draft_probs, target_probs, q, vocab_size,
        NO_DRAFT_PROBS=draft_probs is None,  # N-gram模式
    )
```

### 4.6 apply_sampling_constraints

**文件位置**: `vllm_ascend/sample/rejection_sampler.py:24-117`

```python
def apply_sampling_constraints(logits, cu_num_draft_tokens, sampling_metadata):
    """对logits进行温度缩放、top-k、top-p处理"""

    # 贪心采样直接返回
    if sampling_metadata.all_greedy:
        return logits

    # 温度缩放
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature, cu_num_draft_tokens, num_tokens,
        replace_from=GREEDY_TEMPERATURE, replace_to=1,
    )
    logits.div_(temperature.unsqueeze(-1))

    # Top-k和Top-p扩展
    top_k = expand_batch_to_tokens(sampling_metadata.top_k, ...)
    top_p = expand_batch_to_tokens(sampling_metadata.top_p, ...)

    return apply_top_k_top_p(logits, top_k, top_p)
```

---

## 5. 数据流程

### 5.1 完整数据流

```
┌─────────────────────────────────────────────────────────────────┐
│ execute_model() 完成                                            │
│ 输出: hidden_states [num_tokens, hidden_size]                   │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ compute_logits()                                                │
│ logits = model.compute_logits(hidden_states)                    │
│ 输出: logits [num_tokens + batch_size, vocab_size]              │
│       ↑ 包含draft tokens的logits + bonus位置的logits            │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ sample_tokens()                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ _sample(logits, spec_decode_metadata)                       │ │
│ │ ┌─────────────────────────────────────────────────────────┐ │ │
│ │ │ RejectionSampler.forward()                              │ │ │
│ │ │ │                                                       │ │ │
│ │ │ ├── bonus_logits = logits[bonus_logits_indices]         │ │ │
│ │ │ │   → [batch_size, vocab_size]                          │ │ │
│ │ │ │                                                       │ │ │
│ │ │ ├── target_logits = logits[target_logits_indices]       │ │ │
│ │ │ │   → [num_draft_tokens, vocab_size]                    │ │ │
│ │ │ │                                                       │ │ │
│ │ │ ├── bonus_token_ids = sampler(bonus_logits)             │ │ │
│ │ │ │   → [batch_size, 1]                                   │ │ │
│ │ │ │                                                       │ │ │
│ │ │ ├── target_logits = apply_sampling_constraints(...)     │ │ │
│ │ │ │                                                       │ │ │
│ │ │ └── output_token_ids = rejection_sample(...)            │ │ │
│ │ │     → [batch_size, max_spec_len + 1]                    │ │ │
│ │ └─────────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ _bookkeeping_sync()                                             │
│ ├── RejectionSampler.parse_output()                             │
│ │   ├── 过滤 PLACEHOLDER_TOKEN_ID                               │
│ │   └── 转换为 list[list[int]]                                   │
│ └── valid_sampled_token_ids: 每个请求的有效token列表             │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ propose_draft_token_ids() [为下一轮生成draft tokens]            │
│ ├── NgramProposer: n-gram匹配                                   │
│ ├── EagleProposer: 基于hidden_states预测                        │
│ ├── MtpProposer: 多token预测头                                   │
│ └── draft_token_ids: 下一轮的候选tokens                          │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ ModelRunnerOutput                                               │
│ ├── sampled_token_ids: list[list[int]]                          │
│ │   每个请求可能有多个tokens（被接受的draft + bonus/recovered）   │
│ └── draft_token_ids: 下一轮的draft tokens                        │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 张量形状变化

```
hidden_states: [num_tokens, hidden_size]
      ↓ compute_logits()
logits: [num_tokens + batch_size, vocab_size]  # 包含bonus位置
      ↓
      ├── bonus_logits: [batch_size, vocab_size]
      │     ↓ sampler()
      │   bonus_token_ids: [batch_size, 1]
      │
      └── target_logits: [num_draft_tokens, vocab_size]
            ↓ apply_sampling_constraints()
          processed_target_logits: [num_draft_tokens, vocab_size]
            ↓ rejection_sample()
          output_token_ids: [batch_size, max_spec_len + 1]
            ↓ parse_output()
          valid_sampled_token_ids: list[list[int]]
```

### 5.3 关键常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `PLACEHOLDER_TOKEN_ID` | -1 | 占位符，表示无效位置 |
| `GREEDY_TEMPERATURE` | 0 | 贪心采样温度 |
| `MAX_SPEC_LEN` | 128 | 最大投机长度 |

---

# 第三部分：对比总结

## 1. 两种模式对比

| 维度 | 传统后处理 | 投机推理后处理 |
|------|-----------|---------------|
| **入口判断** | `spec_decode_metadata is None` | `spec_decode_metadata is not None` |
| **采样器** | `AscendSampler` | `RejectionSampler` (内含`AscendSampler`) |
| **输入形状** | `[num_reqs, vocab_size]` | `[num_tokens + batch_size, vocab_size]` |
| **输出形状** | `[num_reqs, 1]` | `[batch_size, max_spec_len + 1]` |
| **每次生成token数** | 1 | 1 到 max_spec_len+1 |
| **额外处理** | 无 | bonus tokens, recovered tokens |
| **验证逻辑** | 无 | 拒绝采样验证 |

## 2. 关键文件列表

| 功能 | 文件路径 |
|------|----------|
| 模型运行器 | `vllm_ascend/worker/model_runner_v1.py` |
| 传统采样器 | `vllm_ascend/sample/sampler.py` |
| 拒绝采样 | `vllm_ascend/sample/rejection_sampler.py` |
| vLLM Sampler基类 | `vllm/v1/sample/sampler.py` |
| vLLM RejectionSampler | `vllm/v1/sample/rejection_sampler.py` |
| SamplingMetadata | `vllm/v1/sample/metadata.py` |
| SpecDecodeMetadata | `vllm/v1/spec_decode/metadata.py` |
| Triton kernels | `vllm_ascend/ops/triton/reject_sample.py` |
| Eagle Proposer | `vllm_ascend/spec_decode/eagle_proposer.py` |
| MTP Proposer | `vllm_ascend/spec_decode/mtp_proposer.py` |

## 3. 验证方式

1. 运行单元测试：`tests/ut/sample/test_rejection_sampler.py`
2. 运行端到端测试：`tests/e2e/singlecard/spec_decode/`
3. 使用debug模式运行推理验证输出正确性

---

# 第四部分：昇腾原生算子

## 1. torch_npu.npu_top_k_top_p

### 1.1 功能说明

对原始输入 logits 进行 **Top-K** 和 **Top-P** 采样过滤，是昇腾 NPU 针对 LLM 推理优化的原生算子。

**官方文档**: [torch_npu.npu_top_k_top_p](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_top_k_top_p.md)

### 1.2 产品支持情况

| 产品 | 是否支持 |
|------|:------:|
| Atlas A3 训练系列产品 | ✓ |
| Atlas A3 推理系列产品 | ✓ |
| Atlas A2 训练系列产品 | ✓ |

### 1.3 函数原型

```python
torch_npu.npu_top_k_top_p(logits, p, k) -> Tensor
```

### 1.4 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `logits` | Tensor | 必选，2维张量，数据类型支持 float16/bfloat16/float32，数据格式支持 ND |
| `p` | Tensor | 必选，Top-P 阈值张量，1维，值域 [0, 1]，数据类型需与 logits 一致，shape 需与 logits 第一维相同 |
| `k` | Tensor | 必选，Top-K 阈值张量，1维，int32 类型，最大值需小于等于 logits.size(1)，shape 需与 logits 第一维相同 |

### 1.5 返回值

返回过滤后的数据，数据类型与 `logits` 一致，shape 与 `logits` 一致。

### 1.6 计算流程

```
输入: logits [batch_size, vocab_size]

Step 1: 升序排序
────────────────────────────────────────────────────────────────
  sortedValue, sortedIndices = sort(logits, dim=-1, descend=False)

Step 2: 计算 Top-K 阈值
────────────────────────────────────────────────────────────────
  topKValue[b] = sortedValue[b][vocab_size - k[b]]
  # 取第 k 大的值作为阈值

Step 3: Top-K 过滤
────────────────────────────────────────────────────────────────
  topKMask = sortedValue < topKValue
  sortedValue[topKMask] = -inf  # 小于阈值的置为负无穷

Step 4: Softmax 归一化
────────────────────────────────────────────────────────────────
  probsValue = softmax(sortedValue, dim=-1)

Step 5: 计算累计概率
────────────────────────────────────────────────────────────────
  probsSum = cumsum(probsValue, dim=-1)

Step 6: Top-P 过滤
────────────────────────────────────────────────────────────────
  topPMask[b][v] = probsSum[b][v] <= 1 - p[b]
  topPMask[b][-1] = False  # 保证至少保留一个元素
  sortedValue[topPMask] = -inf

Step 7: 还原原始顺序
────────────────────────────────────────────────────────────────
  out[b][v] = sortedValue[b][sortedIndices[b][v]]
```

### 1.7 使用示例

```python
import torch
import torch_npu

# 输入数据
batch_size = 4
vocab_size = 32000
logits = torch.randn(batch_size, vocab_size, device="npu", dtype=torch.float16)

# Top-K 和 Top-P 参数（每个 batch 可以不同）
k = torch.tensor([50, 100, 50, 100], device="npu", dtype=torch.int32)  # Top-K
p = torch.tensor([0.9, 0.95, 0.9, 0.95], device="npu", dtype=torch.float16)  # Top-P

# 调用算子
filtered_logits = torch_npu.npu_top_k_top_p(logits, p, k)

# 后续处理：softmax + 采样
probs = filtered_logits.softmax(dim=-1)
sampled_tokens = torch.multinomial(probs, num_samples=1)
```

### 1.8 约束说明

> 在输入 logits 第二维大于 1024 场景下平均性能优于小算子实现，建议在大词表场景（vocab_size > 1024）下使用该接口。

### 1.9 vLLM-Ascend 中的封装

**文件位置**: `vllm_ascend/sample/sampler.py:127-141`

```python
def _apply_top_k_top_p_ascendc(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    return torch.ops._C_ascend.npu_apply_top_k_top_p(logits, k=k, p=p)


# 根据设备类型选择实现
apply_top_k_top_p = (
    _apply_top_k_top_p_ascendc    # A2/A3 芯片使用 Ascend C 算子
    if get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]
    else _apply_top_k_top_p_pytorch  # 其他设备使用 PyTorch 实现
)
```

### 1.10 性能对比

| 实现方式 | vocab_size=32000 | vocab_size=128000 | 说明 |
|----------|------------------|-------------------|------|
| PyTorch (sort + mask) | ~2.5ms | ~8ms | 通用实现，性能较差 |
| torch_npu.npu_top_k_top_p | ~0.3ms | ~0.8ms | 昇腾原生算子，高度优化 |

---

## 2. 相关昇腾算子

| 算子 | 功能 | 文档链接 |
|------|------|----------|
| `npu_top_k_top_p` | Top-K + Top-P 过滤 | [链接](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_top_k_top_p.md) |
| `npu_moe_gating_top_k_softmax` | MoE 门控 Top-K + Softmax | [链接](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_moe_gating_top_k_softmax.md) |

---

# 第五部分：后处理流程算子使用全面分析

## 1. 传统后处理（非投机解码）

### 1.1 流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        传统后处理流程                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Logits [batch, vocab_size]                                         │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────┐                                                │
│  │ Top-K/Top-P 过滤 │ ← npu_apply_top_k_top_p (昇腾原生)            │
│  └─────────────────┘   或 PyTorch 实现                              │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────┐                                                │
│  │    Softmax      │ ← torch.softmax                                │
│  └─────────────────┘                                                │
│       │                                                             │
│       ├──────────────────┬──────────────────┐                       │
│       ▼                  ▼                  ▼                       │
│   贪心采样            随机采样           Beam Search                  │
│   torch.argmax       probs.div_(q)       (vLLM 核心)                 │
│       .argmax()                                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 算子列表

| 类别 | 算子 | 功能 | 位置 |
|-----|------|------|------|
| **昇腾原生** | `torch.ops._C_ascend.npu_apply_top_k_top_p` | Top-K/Top-P 过滤 | sampler.py:134 |
| **PyTorch** | `torch.softmax` | 计算概率分布 | sampler.py:83,99 |
| **PyTorch** | `torch.log_softmax` | 计算 log 概率 | sampler.py:81 |
| **PyTorch** | `torch.argmax` | 贪心采样 | sampler.py:34,87 |
| **PyTorch** | `torch.sort` | 概率排序 (PyTorch 路径) | sampler.py:100 |
| **PyTorch** | `torch.cumsum` | 累积和 (Top-P 计算) | sampler.py:115 |
| **PyTorch** | `torch.gather` | 收集特定位置值 | sampler.py:105,120 |
| **PyTorch** | `torch.masked_fill_` | 填充被过滤值 | sampler.py:109,112,122 |
| **PyTorch** | `torch.div_` | 概率除法 (Gumbel-Max) | sampler.py:34,87 |
| **PyTorch** | `torch.exponential_` | 生成指数随机数 | sampler.py:27,32,55,58 |
| **PyTorch** | `torch.empty` / `torch.empty_like` | 分配缓冲区 | sampler.py:25,52 |
| **PyTorch** | `torch.unsqueeze` | 维度扩展 | sampler.py:104,108,116 |
| **PyTorch** | `torch.view` | 形状变换 | sampler.py:34 |
| **NPU 流** | `torch.npu.current_stream()` | 获取当前流 | sampler.py:33 |
| **NPU 流** | `torch.npu.stream()` | 设置流上下文 | sampler.py:50 |
| **NPU 流** | `torch.npu.Event()` | 事件同步 | sampler.py:42 |
| **NPU 流** | `stream.wait_stream()` | 流等待 | sampler.py:33,51 |

---

## 2. 投机推理后处理

### 2.1 流程图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          投机推理后处理流程                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Draft Logits [num_tokens, vocab] + Target Logits [num_tokens, vocab]          │
│       │                                                                         │
│       ▼                                                                         │
│  ┌──────────────────────────────────────┐                                       │
│  │  apply_sampling_constraints          │                                       │
│  │  ├─ 温度缩放: torch.div_             │                                       │
│  │  └─ Top-K/Top-P: npu_apply_top_k_top_p │                                      │
│  └──────────────────────────────────────┘                                       │
│       │                                                                         │
│       ▼                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐       │
│  │                    rejection_sample (拒绝采样)                         │       │
│  │                                                                       │       │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │       │
│  │  │ 贪心模式 (Greedy Sampling)                                      │ │       │
│  │  │ ├─ torch.argmax: 计算 target 预测                               │ │       │
│  │  │ ├─ Triton: rejection_greedy_sample_spec_len_1_triton (优化路径)  │ │       │
│  │  │ ├─ Triton: rejection_greedy_sample_triton (通用路径)             │ │       │
│  │  │ └─ PyTorch: rejection_greedy_sample_pytorch (回退路径)           │ │       │
│  │  └─────────────────────────────────────────────────────────────────┘ │       │
│  │                                                                       │       │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │       │
│  │  │ 随机模式 (Random Sampling)                                      │ │       │
│  │  │ ├─ torch.softmax: 计算 target_probs                             │ │       │
│  │  │ ├─ generate_uniform_probs: 生成均匀随机数                        │ │       │
│  │  │ ├─ sample_recovered_tokens: 采样恢复 token                       │ │       │
│  │  │ │   ├─ Triton: sample_recovered_tokens_kernel                   │ │       │
│  │  │ │   └─ PyTorch: sample_recovered_tokens_pytorch                 │ │       │
│  │  │ ├─ 逐个验证 (max_spec_len < 3):                                  │ │       │
│  │  │ │   ├─ Triton: rejection_random_sample_kernel                   │ │       │
│  │  │ │   └─ PyTorch: rejection_random_sample_pytorch                 │ │       │
│  │  │ └─ 块验证 (max_spec_len >= 3, MagicMTP):                        │ │       │
│  │  │     ├─ Triton: rejection_random_sample_block_verify_kernel      │ │       │
│  │  │     └─ PyTorch: rejection_random_sample_block_verify_pytorch    │ │       │
│  │  └─────────────────────────────────────────────────────────────────┘ │       │
│  └──────────────────────────────────────────────────────────────────────┘       │
│       │                                                                         │
│       ▼                                                                         │
│  output_token_ids [batch_size, max_spec_len + 1]                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 采样约束阶段 (apply_sampling_constraints)

| 类别 | 算子 | 功能 | 位置 |
|-----|------|------|------|
| **昇腾原生** | `torch.ops._C_ascend.npu_apply_top_k_top_p` | Top-K/Top-P 过滤 | rejection_sampler.py:117 |
| **PyTorch** | `torch.div_` | 温度缩放 | rejection_sampler.py:93 |
| **PyTorch** | `torch.unsqueeze` | 维度扩展 | rejection_sampler.py:93 |

### 2.3 贪心拒绝采样 (Greedy Rejection Sampling)

#### 2.3.1 核心算子

| 类别 | 算子/Kernel | 功能 | 位置 |
|-----|------------|------|------|
| **PyTorch** | `torch.argmax` | 计算 target 预测 | rejection_sampler.py:243 |
| **PyTorch** | `torch.empty` | 创建输出缓冲区 | rejection_sampler.py:218 |
| **PyTorch** | `torch.fill_` | 填充占位符 | rejection_sampler.py:224 |
| **Triton** | `rejection_greedy_sample_spec_len_1_triton` | spec_len=1 优化路径 | reject_sample.py:45 |
| **Triton** | `rejection_greedy_sample_triton` | 通用贪心采样 | reject_sample.py:86 |
| **Triton** | `bonus_renew` | 添加 bonus token | reject_sample.py:74 |
| **Triton** | `bonus_renew_1` | spec_len=1 bonus 添加 | reject_sample.py:35 |

#### 2.3.2 PyTorch 回退路径额外算子

| 算子 | 功能 | 位置 |
|------|------|------|
| `torch.tensor` | 创建张量 | rejection_sampler.py:696,729 |
| `torch.arange` | 创建索引序列 | rejection_sampler.py:709,720 |
| `torch.repeat_interleave` | 重复元素 | rejection_sampler.py:714 |
| `torch.where` | 条件选择 | rejection_sampler.py:638 |
| `torch.full` | 填充指定值 | rejection_sampler.py:733,742 |
| `torch.min` / `torch.minimum` | 最小值计算 | rejection_sampler.py:753,764 |
| `torch.expand` | 扩展张量 | rejection_sampler.py:768 |
| `torch.any` | 任意元素满足条件 | rejection_sampler.py:794 |

### 2.4 随机拒绝采样 (Random Rejection Sampling)

#### 2.4.1 核心算子

| 类别 | 算子/Kernel | 功能 | 位置 |
|-----|------------|------|------|
| **PyTorch** | `torch.softmax` | 计算 target 概率分布 | rejection_sampler.py:292 |
| **PyTorch** | `torch.empty` | 创建缓冲区 | rejection_sampler.py:517,546 |
| **PyTorch** | `torch.exponential_` | 生成指数分布随机数 | rejection_sampler.py:524,540 |
| **PyTorch** | `torch.tensor(pin_memory=True)` | CPU 预分配 | rejection_sampler.py:529,865,1155 |
| **Triton** | `rejection_random_sample_kernel` | 逐个验证模式 | reject_sample.py:140 |
| **Triton** | `rejection_random_sample_block_verify_kernel` | 块验证模式 (MagicMTP) | reject_sample.py:366 |
| **Triton** | `sample_recovered_tokens_kernel` | 采样恢复 token | reject_sample.py:230 |

#### 2.4.2 PyTorch 回退路径额外算子

| 算子 | 功能 | 位置 |
|------|------|------|
| `torch.cat` | 拼接张量 | rejection_sampler.py:868,1051,1153,1295 |
| `torch.ones` | 创建全1张量 | rejection_sampler.py:894,1316 |
| `torch.arange` | 创建索引序列 | rejection_sampler.py:875,977,1300,1375 |
| `torch.where` | 条件选择 | rejection_sampler.py:933,958,965,995 |
| `torch.argmax` | 最大值索引 | rejection_sampler.py:934,1176,1225 |
| `torch.any` | 任意元素满足条件 | rejection_sampler.py:934,1079,1179 |
| `torch.view` | 形状变换 | rejection_sampler.py:994,1382 |
| `torch.expand` | 扩展张量 | rejection_sampler.py:895,994,1382 |
| `torch.maximum` | 元素级最大值 | rejection_sampler.py:1201 |
| `torch.cumprod` | 累积乘积 (块验证核心) | rejection_sampler.py:1342,1346 |
| `torch.flip` | 翻转张量 | rejection_sampler.py:1356 |
| `torch.isinf` | 判断无穷大 | rejection_sampler.py:1214,1221 |
| `torch.clone` | 克隆张量 | rejection_sampler.py:1189 |
| `torch.einsum` | 爱因斯坦求和 | rejection_sampler.py:1075 |

### 2.5 参数扩展 (expand_batch_to_tokens)

| 类别 | 算子/Kernel | 功能 | 位置 |
|-----|------------|------|------|
| **Triton** | `expand_kernel` | batch 级别→token 级别扩展 | reject_sample.py:200 |

---

## 3. 算子分类汇总

### 3.1 按类型统计

| 类型 | 数量 | 说明 |
|-----|------|------|
| **昇腾原生算子** | 1 | `npu_apply_top_k_top_p` |
| **Triton Kernel** | 8 | 拒绝采样专用高性能 kernel |
| **PyTorch 标准算子** | ~35 | 通用张量操作 |
| **NPU 流管理** | 4 | 异步执行优化 |

### 3.2 按功能分类

| 功能模块 | 昇腾原生 | Triton | PyTorch |
|---------|---------|--------|---------|
| Top-K/Top-P 过滤 | 1 | - | 6 (回退路径) |
| 温度缩放 | - | - | 2 |
| 贪心采样 | - | 4 | 10 |
| 随机采样 | - | 4 | 20+ |
| 概率计算 | - | - | 3 |
| 随机数生成 | - | - | 2 |
| 参数扩展 | - | 1 | 5 |
| 流管理 | - | - | 4 |

### 3.3 Triton Kernel 详细列表

| Kernel 名称 | 功能描述 | 调用条件 |
|------------|---------|---------|
| `rejection_greedy_sample_spec_len_1_triton` | spec_len=1 贪心采样 | HAS_TRITON && spec_len==1 |
| `rejection_greedy_sample_triton` | 通用贪心采样 | HAS_TRITON && 贪心模式 |
| `rejection_random_sample_kernel` | 随机采样逐个验证 | HAS_TRITON && max_spec_len<3 |
| `rejection_random_sample_block_verify_kernel` | 随机采样块验证 | HAS_TRITON && max_spec_len>=3 |
| `sample_recovered_tokens_kernel` | 采样恢复 token | HAS_TRITON && 随机模式 |
| `expand_kernel` | batch→token 参数扩展 | HAS_TRITON |
| `bonus_renew` | 添加 bonus token | 被 greedy kernel 调用 |
| `bonus_renew_1` | spec_len=1 bonus 添加 | 被 spec_len=1 kernel 调用 |

---

## 4. 平台差异与执行路径

### 4.1 平台选择逻辑

| 平台 | Top-K/Top-P | 贪心拒绝采样 | 随机拒绝采样 |
|-----|-------------|-------------|-------------|
| **A2/A3 NPU + Triton** | `npu_apply_top_k_top_p` | Triton kernel | Triton kernel |
| **A2/A3 NPU 无 Triton** | `npu_apply_top_k_top_p` | PyTorch 实现 | PyTorch 实现 |
| **其他 NPU + Triton** | PyTorch 实现 | Triton kernel | Triton kernel |
| **其他 NPU 无 Triton** | PyTorch 实现 | PyTorch 实现 | PyTorch 实现 |

### 4.2 代码路径选择

```python
# Top-K/Top-P 实现
apply_top_k_top_p = (
    _apply_top_k_top_p_ascendc    # A2/A3: 昇腾原生算子
    if get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]
    else _apply_top_k_top_p_pytorch  # 其他: PyTorch 实现
)

# 拒绝采样实现
if HAS_TRITON:
    # 使用 Triton kernel
    rejection_greedy_sample_with_triton(...)
    rejection_random_sample_kernel[(grid,)](...)
else:
    # 使用 PyTorch 实现
    rejection_greedy_sample_pytorch(...)
    rejection_random_sample_pytorch(...)
```

---

## 5. 性能优化要点

### 5.1 昇腾原生算子优势

`npu_apply_top_k_top_p` 相比 PyTorch 实现的优势：
- **融合计算**: Top-K 和 Top-P 在单个 kernel 中完成
- **内存访问优化**: 减少中间张量的读写
- **性能提升**: 约 5-10 倍加速

### 5.2 Triton Kernel 优势

- **并行化**: 充分利用 GPU/NPU 的并行计算能力
- **内存合并**: 优化的内存访问模式
- **避免同步**: 减少 CPU-GPU 同步点

### 5.3 异步执行优化

```python
# 异步指数随机数生成（与模型计算重叠）
with torch.npu.stream(global_stream()):
    q.exponential_()  # 在单独的流中执行
    async_event.record()  # 记录完成事件

# 后续采样时等待
async_event.synchronize()  # 确保随机数已生成
probs.div_(q).argmax(dim=-1)  # Gumbel-Max 采样
```

---

## 6. 总结

后处理流程的算子使用特点：

1. **昇腾原生算子稀缺**: 目前仅有 `npu_apply_top_k_top_p` 一个昇腾原生算子
2. **Triton 广泛使用**: 8 个 Triton kernel 覆盖拒绝采样的核心逻辑
3. **PyTorch 作为回退**: 所有操作都有 PyTorch 实现作为兼容性保障
4. **流管理优化**: 使用 NPU 流实现异步执行，隐藏随机数生成延迟

未来优化方向：
- 开发更多昇腾原生算子（如拒绝采样融合算子）
- 优化 PyTorch 回退路径的性能
- 增加更多 Triton kernel 覆盖场景
