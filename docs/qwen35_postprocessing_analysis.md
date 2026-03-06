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
| 3d-H2D | _convert_to_tensors | `make_tensor_with_pad(output_token_ids) + .to(device)` | PadAndStack(CPU)+H2DTransfer | **CPU→NPU** | `list[list[int]]` | `[B, max_seq]` int64 | ⚠️ | CPU构造pin_memory张量→NPU传输(non_blocking); 每步执行,3.4.5性能瓶颈之一 |
| 3d-0 | masked_fill_ (-1替换) | `output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)` | Compare+MaskedFill | NPU-Vector | `[B, max_seq]` int64 | `[B, max_seq]` int64 | ✅ | 替换异步调度的-1占位符为vocab_size |
| 3d-i | get_token_bin_counts (prompt) | `zeros + ones_like + scatter_add_ + slice + (>0)` | Zeros+OnesLike+ScatterAdd+Slice+GT | NPU-Vector | `[B, V+1]` + `[B, seq_len]` | `[B, V]` mask bool | ✅ | 统计prompt token频次,生成prompt_mask; slice为`[:,:vocab_size]`截断 |
| 3d-i' | get_token_bin_counts (output) | `zeros + ones_like + scatter_add_ + slice + (>0)` | Zeros+OnesLike+ScatterAdd+Slice+GT | NPU-Vector | `[B, V+1]` + `[B, seq_len]` | `[B, V]` counts + mask bool | ✅ | 统计output token频次,生成output_bin_counts和output_mask; slice为`[:,:vocab_size]`截断 |
| 3d-ii | repetition_penalty | `unsqueeze+repeat+where(mask)+where(logits>0,1/pen,pen)+mul_` | Unsqueeze+Repeat+Or+Where+Reciprocal+Where+Mul_ | NPU-Vector | `[B, V]` fp32 + `[B]` penalties | `[B, V]` fp32 | ✅ | apply_repetition_penalties_torch; `1.0/penalties`产生Reciprocal |
| 3d-iii | frequency_penalty | `logits -= freq_pen.unsqueeze(1) * bin_counts` | Unsqueeze+Mul+Sub_ | NPU-Vector | `[B, V]` fp32 + `[B]` penalties | `[B, V]` fp32 | ✅ | 可与repetition融合; unsqueeze将`[B]→[B,1]`广播 |
| 3d-iv | presence_penalty | `logits -= pres_pen.unsqueeze(1) * output_mask` | Unsqueeze+Mul+Sub_ | NPU-Vector | `[B, V]` fp32 + `[B]` penalties | `[B, V]` fp32 | ✅ | 可与上两项融合; unsqueeze将`[B]→[B,1]`广播 |
| 4a | argmax (贪心) | `logits.argmax(dim=-1)` | ArgMaxWithValue | NPU-Vector | `[B, V]` fp32 | `[B]` int64 | ✅ | 贪心采样, all_greedy时直接返回 |
| 4b | temperature div | `where(temp<EPS,1.0,temp) + logits.div_(temp.unsqueeze(1))` | Compare(LT)+Where+Unsqueeze+Div_ | NPU-Vector | `[B, V]` fp32 / `[B]` fp32 | `[B, V]` fp32 | ✅ | all_random=False时先Compare+Where避免除零,再Unsqueeze+Div_; 可融合到softmax |
| 4c | min_p | 自定义处理器 | Where+Mul+Mask | NPU-Vector | `[B, V]` fp32 | `[B, V]` fp32 | ✅ | argmax_invariant处理器 |
| 4d-i | top_k_top_p | `npu_apply_top_k_top_p(logits, k, p)` | **AscendC自定义算子** | NPU-Vector | `[B, V]` fp32 + `[B]` k,p | `[B, V]` fp32 | ⚠️ | A2/A3专用;其他走PyTorch sort |
| 4d-i' | top_k_top_p(fallback) | `softmax→sort→sub+cast+unsqueeze+gather→compare(EQ)+unsqueeze+masked_fill_→compare(LT)+masked_fill_(+cumsum+sum+unsqueeze+gather for top_p)` | Softmax+Sort+Sub+Cast+Unsqueeze+Gather+Compare(EQ)+Unsqueeze+MaskedFill_+Compare(LT)+MaskedFill_(+Cumsum+Unsqueeze+ReduceSum+Unsqueeze+Gather+Compare(LT)+MaskedFill_) | NPU-Vector | `[B, V]` fp32 | `[B, V]` fp32 | ✅ | PyTorch fallback路径; top_k含Sub+Cast(int64)+no_top_k_mask处理; top_p含Cumsum+ReduceSum |
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
| 6d | count_greater | `(x >= values).sum(-1)` | GreaterEqual+ReduceSum | NPU-Vector | `[B, V]` fp32 + `[B, 1]` fp32 | `[B]` int64 | ✅ | `@torch.compile(backend=simple_compile_backend)`生成; NPU上backend可能非默认inductor |
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
 │       │   算子: Zeros+OnesLike+ScatterAdd+Slice+GT  位置: NPU-Vector
 │       │   数据流: prompt_tokens[B, seq_len] → zeros[B, V+1] → scatter_add_(ones_like(tokens)) → slice[:,:V] → [B,V] → (>0) → prompt_mask[B, V] bool
 │       │   注: ones_like生成与tokens同形状的全1张量; slice从[B,V+1]截断为[B,V]; 仅返回mask
 │       │
 │       ├─ get_token_bin_counts_and_mask (output):  ← 第2次独立调用
 │       │   算子: Zeros+OnesLike+ScatterAdd+Slice+GT  位置: NPU-Vector
 │       │   数据流: output_tokens[B, max_seq] → zeros[B, V+1] → scatter_add_(ones_like(tokens)) → slice[:,:V] → [B,V] → output_bin_counts[B, V] + (>0) → output_mask[B, V] bool
 │       │   注: 与prompt调用相同函数; 同时返回bin_counts(用于frequency_penalty)和mask(用于presence_penalty)
 │       │
 │       ├─ apply_repetition_penalties:  (torch版本, NPU走此路径)
 │       │   算子: Unsqueeze+Repeat+Or+Where+Reciprocal+Where+Mul_  位置: NPU-Vector
 │       │   数据流: penalties[B] → unsqueeze+repeat → [B,V]
 │       │           prompt_mask | output_mask → combined_mask[B,V]
 │       │           where(combined_mask, penalties, 1.0) → applied_penalties[B,V]
 │       │           1.0/applied_penalties → reciprocal[B,V] (Reciprocal算子)
 │       │           where(logits>0, reciprocal, applied_penalties) → scaling[B,V]
 │       │           logits *= scaling (in-place)
 │       │
 │       ├─ frequency: logits -= freq_pen.unsqueeze(1) * bin_counts
 │       │   算子: Unsqueeze + Mul + Sub_  位置: NPU-Vector
 │       │   数据流: freq_pen[B] → unsqueeze → [B,1] → broadcast mul → sub_
 │       │
 │       └─ presence: logits -= pres_pen.unsqueeze(1) * output_mask
 │           算子: Unsqueeze + Mul + Sub_  位置: NPU-Vector
 │           数据流: pres_pen[B] → unsqueeze → [B,1] → broadcast mul → sub_
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
 │       │   算子: Compare(LT) + Where + Unsqueeze + Div_  位置: NPU-Vector
 │       │   数据流: temp[B] → Compare(temp<EPS) → Where(mask,1.0,temp) → temp[B]
 │       │           → Unsqueeze → temp[B,1] → logits[B,V] ÷ temp[B,1] → logits[B,V]
 │       │   注: 若all_random=True则跳过Compare+Where; Unsqueeze将[B]→[B,1]用于广播除法
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
 │       │   │     算子: Softmax+Sort+Sub+Cast(int64)+Unsqueeze+Gather+Compare(EQ)+Unsqueeze+MaskedFill_
 │       │   │           +Compare(LT)+MaskedFill_(+Cumsum+Unsqueeze+ReduceSum+Unsqueeze+Gather+Compare(LT)+MaskedFill_)
 │       │   │     位置: NPU-Vector
 │       │   │     数据流: logits[B,V] → probs=softmax(logits) → probs_sort=sort(probs, descending=False)
 │       │   │       top_k路径:
 │       │   │         → top_k_count = probs_sort.size(1) - k.to(int64)  (Sub+Cast)
 │       │   │         → top_k_count.unsqueeze(1) (Unsqueeze)
 │       │   │         → probs_sort.gather(-1, top_k_count) → top_k_cutoff (Gather)
 │       │   │         → no_top_k_mask = (k==V).unsqueeze(1) (Compare(EQ)+Unsqueeze)
 │       │   │         → top_k_cutoff.masked_fill_(no_top_k_mask, -inf) (MaskedFill_)
 │       │   │         → elements_to_discard = probs < top_k_cutoff (Compare(LT))
 │       │   │         → logits.masked_fill_(elements_to_discard, -inf) (MaskedFill_)
 │       │   │       top_p路径:
 │       │   │         → cumprob = cumsum(probs_sort) (Cumsum)
 │       │   │         → top_p_mask = cumprob <= 1-p.unsqueeze(1) (Unsqueeze+Compare)
 │       │   │         → top_p_count = top_p_mask.sum(-1).unsqueeze(1) (ReduceSum+Unsqueeze)
 │       │   │         → top_p_cutoff = probs_sort.gather(-1, top_p_count) (Gather)
 │       │   │         → elements_to_discard = probs < top_p_cutoff (Compare(LT))
 │       │   │         → logits.masked_fill_(elements_to_discard, -inf) (MaskedFill_)
 │       │   │     注: 直接修改原始logits,无scatter_操作
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

> **与社区GPU版本差异**: vllm-ascend版本额外支持 `num_pcp_pads` 参数（PCP并行计算填充），当 `pcp_size > 1` 时需修正 `logits_indices` 以跳过padding位置。GPU版本（`vllm/v1/worker/gpu_model_runner.py:2209-2286`）无此逻辑，且使用 `_get_cumsum_and_arange` 辅助函数替代手动计算。

#### 2.3.1 整体流程概览

```
_calc_spec_decode_metadata(num_draft_tokens, cu_num_scheduled_tokens, num_pcp_pads)
    │
    ├── Step 1: 计算 num_sampled_tokens 和 cu_num_sampled_tokens
    │   └── num_sampled_tokens = num_draft_tokens + 1
    │   └── cu_num_sampled_tokens = cumsum(num_sampled_tokens)
    │
    ├── Step 2: 构建 logits_indices（CPU numpy计算）
    │   ├── cumsums_offsets = repeat(cu_num_sampled - num_sampled, num_sampled)
    │   ├── arange = arange_np[:total] - cumsums_offsets
    │   ├── logits_indices = repeat(cu_num_scheduled - num_sampled, num_sampled)
    │   └── logits_indices += arange
    │
    ├── Step 3: 【PCP分支】修正 logits_indices（pcp_size > 1）
    │   └── cu_num_scheduled = cu_num_scheduled * pcp_size - num_pcp_pads
    │   └── 重算 logits_indices_pcp
    │
    ├── Step 4: 计算 bonus_logits_indices
    │   └── bonus_logits_indices = cu_num_sampled_tokens - 1
    │
    ├── Step 5: 构建 target_logits_indices（CPU numpy计算）
    │   ├── cu_num_draft_tokens = cumsum(num_draft_tokens)
    │   ├── cumsums_offsets = repeat(cu_num_draft - num_draft, num_draft)
    │   ├── arange = arange_np[:total_draft] - cumsums_offsets
    │   ├── target_logits_indices = repeat(cu_num_sampled - num_sampled, num_draft)
    │   └── target_logits_indices += arange
    │
    ├── Step 6: CPU→NPU数据传输（5个张量 pin_memory + to(device)）
    │
    ├── Step 7: 计算 draft_token_ids（NPU上执行）
    │   ├── draft_token_ids = input_ids.gpu[logits_indices]
    │   └── draft_token_ids = draft_token_ids[target_logits_indices + 1]
    │
    └── Step 8: 构建并返回 SpecDecodeMetadata
```

#### 2.3.2 数值示例详解

以5个请求为例，展示完整的索引计算过程：

```
输入:
  cu_num_scheduled_tokens: [  4, 104, 107, 207, 209]  ← 累积调度token数
  num_draft_tokens:        [  3,   0,   2,   0,   1]  ← 每个请求的draft数

Step 1: 计算采样token数量
  num_sampled_tokens     = num_draft_tokens + 1
                         = [  4,   1,   3,   1,   2]
  cu_num_sampled_tokens  = cumsum([4, 1, 3, 1, 2])
                         = [  4,   5,   8,   9,  11]
  total_num_sampled_tokens = 11

Step 2: 构建 logits_indices（将连续索引映射到实际模型输出位置）
  2a. cumsums_offsets = repeat(cu_num_sampled - num_sampled, num_sampled)
      cu_num_sampled - num_sampled = [0, 4, 5, 8, 9]  ← 每组的起始偏移
      repeat展开:                  = [0, 0, 0, 0, 4, 5, 5, 5, 8, 9, 9]
  2b. arange = arange_np[:11] - cumsums_offsets
      arange_np[:11]               = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      arange                       = [0, 1, 2, 3, 0, 0, 1, 2, 0, 0,  1]
                                      ← 每组内的局部索引
  2c. logits_indices = repeat(cu_num_scheduled - num_sampled, num_sampled)
      cu_num_scheduled - num_sampled = [0, 103, 104, 206, 207]  ← 每组在模型输出中的起始位置
      repeat展开:                    = [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
  2d. logits_indices += arange
      logits_indices                 = [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
                                        ← 最终的模型输出位置索引

Step 4: bonus_logits_indices（每个请求最后一个采样位置）
  bonus_logits_indices = cu_num_sampled_tokens - 1
                       = [  3,   4,   7,   8,  10]

Step 5: 构建 target_logits_indices（draft token在采样空间中的位置）
  5a. cu_num_draft_tokens = cumsum([3, 0, 2, 0, 1]) = [3, 3, 5, 5, 6]
      total_num_draft_tokens = 6
  5b. cumsums_offsets = repeat(cu_num_draft - num_draft, num_draft)
      cu_num_draft - num_draft = [0, 3, 3, 5, 5]
      repeat展开(按num_draft): = [0, 0, 0, 3, 3, 5]
                                  (req0有3个draft, req1有0个, req2有2个, req3有0个, req4有1个)
  5c. arange = arange_np[:6] - cumsums_offsets
      arange_np[:6]            = [0, 1, 2, 3, 4, 5]
      arange                   = [0, 1, 2, 0, 1, 0]  ← 每组内的局部索引
  5d. target_logits_indices = repeat(cu_num_sampled - num_sampled, num_draft)
      cu_num_sampled - num_sampled = [0, 4, 5, 8, 9]
      repeat展开(按num_draft):     = [0, 0, 0, 5, 5, 9]
  5e. target_logits_indices += arange
      target_logits_indices        = [0, 1, 2, 5, 6, 9]

Step 7: 获取 draft_token_ids（NPU）
  draft_token_ids = input_ids.gpu[logits_indices]
    ← 从GPU上的input_ids按logits_indices索引取值，得到 [11] 个token
  draft_token_ids = draft_token_ids[target_logits_indices + 1]
    ← 按 [1, 2, 3, 6, 7, 10] 索引取值，得到 [6] 个draft token ID
```

#### 2.3.3 源码逐段分析

**Step 1-2: 计算 logits_indices**

```python
# vllm_ascend/worker/model_runner_v1.py:874-885
num_sampled_tokens = num_draft_tokens + 1                    # numpy向量加法
cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)  # 前缀和
total_num_sampled_tokens = cu_num_sampled_tokens[-1]

# 构建每组内的局部索引
cumsums_offsets = np.repeat(cu_num_sampled_tokens - num_sampled_tokens, num_sampled_tokens)
arange = self.arange_np[:total_num_sampled_tokens] - cumsums_offsets

# 构建模型输出中的起始位置，加上局部索引得到最终位置
logits_indices = np.repeat(cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
logits_indices += arange
```

**Step 3: PCP修正（pcp_size > 1时执行）**

```python
# vllm_ascend/worker/model_runner_v1.py:889-893
if self.pcp_size > 1:
    cu_num_scheduled_tokens = cu_num_scheduled_tokens * self.pcp_size - num_pcp_pads
    logits_indices_pcp = np.repeat(cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
    logits_indices_pcp += arange
    logits_indices_pcp = torch.from_numpy(logits_indices_pcp).pin_memory().to(self.device, non_blocking=True)
```

> **PCP说明**: PCP(Parallel Context Processing)在多卡并行时，all-gather后可能引入padding。此处用原始 `logits_indices` 获取 `draft_token_ids`（padding前的正确位置），再用修正后的 `logits_indices_pcp` 替换作为最终返回值。

**Step 4-5: 计算 bonus 和 target logits索引**

```python
# vllm_ascend/worker/model_runner_v1.py:896-909
bonus_logits_indices = cu_num_sampled_tokens - 1  # 每个请求最后一个位置

cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
total_num_draft_tokens = cu_num_draft_tokens[-1]
cumsums_offsets = np.repeat(cu_num_draft_tokens - num_draft_tokens, num_draft_tokens)
arange = self.arange_np[:total_num_draft_tokens] - cumsums_offsets
target_logits_indices = np.repeat(cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
target_logits_indices += arange
```

**Step 6: CPU→NPU传输**

```python
# vllm_ascend/worker/model_runner_v1.py:912-916
cu_num_draft_tokens    = torch.from_numpy(cu_num_draft_tokens).pin_memory().to(self.device, non_blocking=True)
cu_num_sampled_tokens  = torch.from_numpy(cu_num_sampled_tokens).pin_memory().to(self.device, non_blocking=True)
logits_indices         = torch.from_numpy(logits_indices).pin_memory().to(self.device, non_blocking=True)
target_logits_indices  = torch.from_numpy(target_logits_indices).pin_memory().to(self.device, non_blocking=True)
bonus_logits_indices   = torch.from_numpy(bonus_logits_indices).pin_memory().to(self.device, non_blocking=True)
```

> **与GPU版本差异**: vllm-ascend使用 `pin_memory()` + `to(device, non_blocking=True)` 两步传输，GPU版本直接 `to(device, non_blocking=True)`（CUDA张量无需显式pin_memory）。

**Step 7: 获取draft token IDs（NPU上执行）**

```python
# vllm_ascend/worker/model_runner_v1.py:920-923
draft_token_ids = self.input_ids.gpu[logits_indices]          # 高级索引取值
draft_token_ids = draft_token_ids[target_logits_indices + 1]  # 二次索引取draft位置
if self.pcp_size > 1:
    logits_indices = logits_indices_pcp  # PCP场景替换为修正后的索引
```

#### 2.3.4 算子全景总表

| 序号 | 步骤 | API / 算子 | 运行位置 | 输入形状 | 输出形状 | 备注 |
|------|------|-----------|---------|---------|---------|------|
| 1 | num_sampled_tokens | `num_draft_tokens + 1` (numpy) | **CPU** | `[B]` int32 | `[B]` int32 | numpy向量加法 |
| 2 | cu_num_sampled_tokens | `np.cumsum(num_sampled_tokens)` | **CPU** | `[B]` int32 | `[B]` int32 | numpy前缀和 |
| 3 | cumsums_offsets(sampled) | `np.repeat(arr, num_sampled_tokens)` | **CPU** | `[B]` int32 + `[B]` int32 | `[T_s]` int32 | T_s=total_num_sampled_tokens |
| 4 | arange(sampled) | `arange_np[:T_s] - cumsums_offsets` | **CPU** | `[T_s]` int64 - `[T_s]` int32 | `[T_s]` int64 | 预分配arange_np切片减法 |
| 5 | logits_indices(base) | `np.repeat(arr, num_sampled_tokens)` | **CPU** | `[B]` int32 + `[B]` int32 | `[T_s]` int32 | 起始位置展开 |
| 6 | logits_indices(final) | `logits_indices += arange` | **CPU** | `[T_s]` int32 + `[T_s]` int64 | `[T_s]` int64 | numpy原地加法 |
| 7 | 【PCP】cu_num_scheduled修正 | `cu * pcp_size - pads` (numpy) | **CPU** | `[B]` int32 | `[B]` int32 | 仅pcp_size>1 |
| 8 | 【PCP】logits_indices_pcp | `repeat + arange` (numpy) | **CPU** | `[B]` int32 | `[T_s]` int64 | 仅pcp_size>1 |
| 9 | 【PCP】H2D传输 | `from_numpy().pin_memory().to(device)` | **CPU→NPU** | `[T_s]` int64 | `[T_s]` int64 | non_blocking=True |
| 10 | bonus_logits_indices | `cu_num_sampled_tokens - 1` (numpy) | **CPU** | `[B]` int32 | `[B]` int32 | numpy向量减法 |
| 11 | cu_num_draft_tokens | `np.cumsum(num_draft_tokens)` | **CPU** | `[B]` int32 | `[B]` int32 | numpy前缀和 |
| 12 | cumsums_offsets(draft) | `np.repeat(arr, num_draft_tokens)` | **CPU** | `[B]` int32 + `[B]` int32 | `[T_d]` int32 | T_d=total_num_draft_tokens |
| 13 | arange(draft) | `arange_np[:T_d] - cumsums_offsets` | **CPU** | `[T_d]` int64 - `[T_d]` int32 | `[T_d]` int64 | 预分配arange_np切片减法 |
| 14 | target_logits_indices(base) | `np.repeat(arr, num_draft_tokens)` | **CPU** | `[B]` int32 + `[B]` int32 | `[T_d]` int32 | 起始位置展开 |
| 15 | target_logits_indices(final) | `target_logits_indices += arange` | **CPU** | `[T_d]` int32 + `[T_d]` int64 | `[T_d]` int64 | numpy原地加法 |
| 16a | H2D: cu_num_draft_tokens | `from_numpy().pin_memory().to(device)` | **CPU→NPU** | `[B]` int32 | `[B]` int32 NPU | non_blocking |
| 16b | H2D: cu_num_sampled_tokens | `from_numpy().pin_memory().to(device)` | **CPU→NPU** | `[B]` int32 | `[B]` int32 NPU | non_blocking |
| 16c | H2D: logits_indices | `from_numpy().pin_memory().to(device)` | **CPU→NPU** | `[T_s]` int64 | `[T_s]` int64 NPU | non_blocking |
| 16d | H2D: target_logits_indices | `from_numpy().pin_memory().to(device)` | **CPU→NPU** | `[T_d]` int64 | `[T_d]` int64 NPU | non_blocking |
| 16e | H2D: bonus_logits_indices | `from_numpy().pin_memory().to(device)` | **CPU→NPU** | `[B]` int32 | `[B]` int32 NPU | non_blocking |
| 17 | draft_token_ids(索引1) | `input_ids.gpu[logits_indices]` | **NPU** IndexSelect | `[max_tokens]` int64 + `[T_s]` int64 | `[T_s]` int64 | 高级索引(fancy indexing) |
| 18 | target+1 | `target_logits_indices + 1` | **NPU** Add | `[T_d]` int64 | `[T_d]` int64 | 张量标量加法 |
| 19 | draft_token_ids(索引2) | `draft_token_ids[target_logits_indices + 1]` | **NPU** IndexSelect | `[T_s]` int64 + `[T_d]` int64 | `[T_d]` int64 | 高级索引(fancy indexing) |
| 20 | 【PCP】logits_indices替换 | `logits_indices = logits_indices_pcp` | - | - | - | Python引用赋值,零开销 |

> **图例**: B=batch_size(num_reqs), T_s=total_num_sampled_tokens, T_d=total_num_draft_tokens

#### 2.3.5 数据流详细追踪

```
输入: num_draft_tokens [B] np.int32  (来自scheduler的每请求draft token数)
      cu_num_scheduled_tokens [B] np.int32  (累积调度token数,来自_prepare_inputs)
      num_pcp_pads [B] np.int32 | None  (PCP填充数,仅pcp_size>1)
 │
 ├─ Step 1: 计算采样token数量 (CPU numpy)
 │   ├─ num_sampled_tokens = num_draft_tokens + 1  ──→ [B] np.int32
 │   │   运算: numpy向量加标量             位置: CPU
 │   │   语义: 每个请求的采样数 = draft数 + 1(验证token)
 │   │
 │   ├─ cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)  ──→ [B] np.int32
 │   │   运算: numpy前缀和                 位置: CPU
 │   │
 │   └─ total_num_sampled_tokens = cu_num_sampled_tokens[-1]  ──→ scalar
 │
 ├─ Step 2: 构建 logits_indices (CPU numpy, 核心索引映射)
 │   │   目标: 将扁平化的采样位置映射到模型输出中的实际位置
 │   │
 │   ├─ cumsums_offsets = np.repeat(cu_num_sampled - num_sampled, num_sampled)  ──→ [T_s] np.int32
 │   │   运算: numpy repeat               位置: CPU
 │   │   语义: 每组的起始偏移,按组大小展开
 │   │
 │   ├─ arange = arange_np[:T_s] - cumsums_offsets  ──→ [T_s] np.int64
 │   │   运算: numpy切片 + 向量减法         位置: CPU
 │   │   语义: 每组内的局部递增索引 (0,1,2,...,n_i-1)
 │   │
 │   ├─ logits_indices = np.repeat(cu_num_scheduled - num_sampled, num_sampled)  ──→ [T_s] np.int32
 │   │   运算: numpy repeat               位置: CPU
 │   │   语义: 每组在模型输出中的起始位置,按组大小展开
 │   │
 │   └─ logits_indices += arange  ──→ [T_s] np.int64  (in-place)
 │       运算: numpy原地加法               位置: CPU
 │       语义: 起始位置 + 局部索引 = 最终模型输出位置
 │
 ├─ 【PCP分支】Step 3: 修正 logits_indices (pcp_size > 1)
 │   ├─ cu_num_scheduled = cu_num_scheduled * pcp_size - num_pcp_pads  ──→ [B] np.int32
 │   │   运算: numpy乘法 + 减法            位置: CPU
 │   │   语义: 扣除all-gather引入的padding后的实际位置
 │   │
 │   ├─ logits_indices_pcp = np.repeat(...) + arange  ──→ [T_s] np.int64
 │   │   运算: 同Step 2                    位置: CPU
 │   │
 │   └─ logits_indices_pcp = torch.from_numpy().pin_memory().to(device)  ──→ [T_s] NPU tensor
 │       运算: H2D传输                     位置: CPU→NPU (non_blocking)
 │
 ├─ Step 4: bonus_logits_indices (CPU numpy)
 │   └─ bonus_logits_indices = cu_num_sampled_tokens - 1  ──→ [B] np.int32
 │       运算: numpy向量减标量              位置: CPU
 │       语义: 每个请求的最后一个采样位置(用于bonus token logits)
 │
 ├─ Step 5: 构建 target_logits_indices (CPU numpy)
 │   │   目标: 将draft token位置映射到采样空间中的位置
 │   │
 │   ├─ cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)  ──→ [B] np.int32
 │   │   运算: numpy前缀和                 位置: CPU
 │   │
 │   ├─ total_num_draft_tokens = cu_num_draft_tokens[-1]  ──→ scalar
 │   │
 │   ├─ cumsums_offsets = np.repeat(cu_num_draft - num_draft, num_draft)  ──→ [T_d] np.int32
 │   │   运算: numpy repeat               位置: CPU
 │   │
 │   ├─ arange = arange_np[:T_d] - cumsums_offsets  ──→ [T_d] np.int64
 │   │   运算: numpy切片 + 向量减法         位置: CPU
 │   │
 │   ├─ target_logits_indices = np.repeat(cu_num_sampled - num_sampled, num_draft)  ──→ [T_d] np.int32
 │   │   运算: numpy repeat               位置: CPU
 │   │   注意: 这里按 num_draft_tokens 展开(非 num_sampled_tokens)
 │   │
 │   └─ target_logits_indices += arange  ──→ [T_d] np.int64  (in-place)
 │       运算: numpy原地加法               位置: CPU
 │
 ├─ Step 6: CPU→NPU批量传输 (5个张量)
 │   ├─ cu_num_draft_tokens:    [B] int32     ──pin_memory()──→ NPU
 │   ├─ cu_num_sampled_tokens:  [B] int32     ──pin_memory()──→ NPU
 │   ├─ logits_indices:         [T_s] int64   ──pin_memory()──→ NPU
 │   ├─ target_logits_indices:  [T_d] int64   ──pin_memory()──→ NPU
 │   └─ bonus_logits_indices:   [B] int32     ──pin_memory()──→ NPU
 │   全部使用 non_blocking=True, 与后续NPU计算重叠
 │
 ├─ Step 7: 计算 draft_token_ids (NPU)
 │   ├─ draft_token_ids = input_ids.gpu[logits_indices]  ──→ [T_s] int64
 │   │   算子: IndexSelect (fancy indexing) 位置: NPU
 │   │   数据流: input_ids[max_tokens] ⊕ logits_indices[T_s] → [T_s]
 │   │   语义: 按logits_indices从GPU端input_ids中取出对应token
 │   │
 │   └─ draft_token_ids = draft_token_ids[target_logits_indices + 1]  ──→ [T_d] int64
 │       算子: Add(标量) + IndexSelect     位置: NPU
 │       数据流: target_logits_indices[T_d] + 1 → indices[T_d]
 │               draft_token_ids[T_s][indices] → [T_d]
 │       语义: target位置+1即为对应draft token位置, 取出draft token IDs
 │
 ├─ 【PCP分支】替换 logits_indices
 │   └─ logits_indices = logits_indices_pcp  (Python引用赋值)
 │
 └─ 构建 SpecDecodeMetadata 并返回:
     │
     └─ return SpecDecodeMetadata(
           draft_token_ids:        [T_d] int64 NPU,     # draft token IDs
           num_draft_tokens:       list[int] len=B CPU,  # 每请求draft数(list)
           cu_num_draft_tokens:    [B] int32 NPU,        # 累积draft数
           cu_num_sampled_tokens:  [B] int32 NPU,        # 累积采样数
           target_logits_indices:  [T_d] int64 NPU,      # target logits位置
           bonus_logits_indices:   [B] int32 NPU,        # bonus logits位置
           logits_indices:         [T_s] int64 NPU,      # 全部logits位置
         )
```

#### 2.3.6 运行位置分布统计

```
┌─────────────────────────────────────────────────────────────────────┐
│              _calc_spec_decode_metadata 算子运行位置分布               │
├─────────────┬───────────────────────────────────────────────────────┤
│  Host CPU   │ ████████████████████████████████████████  ~80%        │
│  (numpy)    │ cumsum, repeat, arange切片, 向量加减法                 │
│             │ 全部索引构建逻辑在CPU端完成                              │
├─────────────┼───────────────────────────────────────────────────────┤
│  H2D传输    │ ██████████  ~12%                                      │
│             │ 5个张量 pin_memory() → to(device, non_blocking)       │
│             │ 传输数据量: 2×[B]×4B + [T_s]×8B + [T_d]×8B + [B]×4B  │
├─────────────┼───────────────────────────────────────────────────────┤
│  NPU        │ ██████  ~8%                                           │
│             │ IndexSelect×2 + Add(标量)                              │
│             │ 仅draft_token_ids获取在NPU上执行                       │
├─────────────┼───────────────────────────────────────────────────────┤
│  NPU-Cube   │ ▏ ~0%                                                │
│             │ 无矩阵乘算子                                           │
└─────────────┴───────────────────────────────────────────────────────┘
```

#### 2.3.7 关键性能瓶颈与优化分析

| 瓶颈点 | 原因 | 影响程度 | 优化方向 |
|--------|------|---------|---------|
| `np.repeat` 多次调用 | 4次repeat操作,每次分配新numpy数组 | ★★★☆☆ | 预分配buffer复用,或迁移到NPU |
| `pin_memory()` + H2D传输 | 5个张量逐个pin_memory+传输,增加CPU开销 | ★★★★☆ | 合并为单个buffer一次传输;或将索引计算迁移到NPU |
| CPU端索引计算 | 所有cumsum/repeat/arange在CPU执行,无法利用NPU并行 | ★★★☆☆ | 将索引计算迁移到NPU端(torch实现替代numpy) |
| `input_ids.gpu[logits_indices]` fancy indexing | 非连续内存访问,NPU端随机读取效率较低 | ★★☆☆☆ | 如果input_ids已排布好,可用连续slice替代 |
| `num_draft_tokens.tolist()` | numpy→Python list转换,GIL+内存分配 | ★☆☆☆☆ | 数据量小(B级别),影响有限 |

#### 2.3.8 与GPU社区版本差异对比

| 特性 | vllm-ascend版本 | GPU社区版本 |
|------|----------------|------------|
| 文件位置 | `vllm_ascend/worker/model_runner_v1.py:856-932` | `vllm/v1/worker/gpu_model_runner.py:2209-2286` |
| PCP支持 | ✅ `num_pcp_pads` 参数,修正logits_indices | ❌ 无PCP逻辑 |
| cumsum+arange实现 | 手动计算(repeat+arange_np切片) | `_get_cumsum_and_arange` 辅助函数 |
| H2D传输 | `from_numpy().pin_memory().to(device)` | `from_numpy().to(device)` |
| draft_token_ids获取 | 相同逻辑 | 相同逻辑 |

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

## 4. 功能流程与算子分析（HAS_TRITON=true 路径）

> **说明**: 本节参照第一部分 3.3/3.4 的文档结构，对投机推理后处理 `RejectionSampler.forward()` 及核心函数 `rejection_sample()` 在 **HAS_TRITON=true** 路径下进行逐步拆解和算子级分析。

### 4.1 RejectionSampler.forward() 详细流程（HAS_TRITON=true）

**文件位置**: `vllm/v1/sample/rejection_sampler.py:24-120`（vLLM 官方）+ `vllm_ascend/sample/rejection_sampler.py`（昇腾 patch）

#### 4.1.1 整体流程概览

```
forward(metadata, draft_probs, logits, sampling_metadata)
    │
    ├── Step 1: 提取 Bonus Logits 并采样 Bonus Token
    │   ├── bonus_logits = logits[metadata.bonus_logits_indices]        # 索引提取
    │   ├── replace(sampling_metadata, max_num_logprobs=-1)             # 元数据替换
    │   └── bonus_token_ids = self.sampler(bonus_logits, ...)           # 完整 AscendSampler.forward()
    │
    ├── Step 2: 提取 Target Logits
    │   ├── raw_target_logits = logits[metadata.target_logits_indices]  # 索引提取
    │   ├── raw_target_logits = raw_target_logits.to(float32)           # 类型转换
    │   └── target_logits = raw_target_logits.clone()                   # 克隆（保留原始值用于logprobs）
    │
    ├── Step 3: 应用 Logits Processors
    │   ├── allowed_token_ids_mask → masked_fill_(-inf)
    │   ├── bad_words → 逐请求设为 -inf
    │   ├── non_argmax_invariant 处理器
    │   └── apply_penalties (repetition/frequency/presence)
    │
    ├── Step 4: apply_sampling_constraints()  [昇腾 patch]
    │   ├── expand_batch_to_tokens(temperature) → expand_kernel (Triton)
    │   ├── logits.div_(temperature.unsqueeze(-1))
    │   ├── expand_batch_to_tokens(top_k) → expand_kernel (Triton)
    │   ├── expand_batch_to_tokens(top_p) → expand_kernel (Triton)
    │   └── apply_top_k_top_p(logits, top_k, top_p) → npu_apply_top_k_top_p (A2/A3)
    │
    ├── Step 5: rejection_sample()  [昇腾 patch, 核心拒绝采样]
    │   ├── 5a. 创建输出缓冲区 [B, S+1], fill_(PLACEHOLDER)
    │   ├── 5b. cal_grid_and_block_size(batch_size) → (grid, block_size)
    │   ├── 5c. 贪心路径: argmax → rejection_greedy_sample_with_triton()
    │   │   ├── spec_len=1 且 all_greedy: rejection_greedy_sample_spec_len_1_triton
    │   │   └── 通用: rejection_greedy_sample_triton (含 bonus_renew)
    │   ├── 5d. target_probs = target_logits.softmax(dim=-1, fp32)
    │   ├── 5e. uniform_probs = generate_uniform_probs()
    │   ├── 5f. recovered_token_ids = sample_recovered_tokens()
    │   │   ├── q.exponential_() (AI-CPU)
    │   │   └── sample_recovered_tokens_kernel[(B, S)] (Triton)
    │   └── 5g. 随机路径:
    │       ├── max_spec_len < 3: rejection_random_sample_kernel (Triton)
    │       └── max_spec_len >= 3: rejection_random_sample_block_verify_kernel (Triton)
    │
    ├── Step 6: 计算 logprobs（可选）
    │   └── _get_logprobs_tensors(...)
    │
    └── Step 7: 返回 SamplerOutput
            ├── sampled_token_ids: [B, S+1] int32
            └── logprobs_tensors: LogprobsTensors | None
```

> **符号说明**: B=batch_size, N=num_draft_tokens(展平总数), S=max_spec_len, V=vocab_size

#### 4.1.2 Step 1: Bonus Token 采样

```python
# vllm/v1/sample/rejection_sampler.py:93-102
bonus_logits = logits[metadata.bonus_logits_indices]  # [B, V]
bonus_sampler_output = self.sampler(
    logits=bonus_logits,
    sampling_metadata=replace(sampling_metadata, max_num_logprobs=-1),
    predict_bonus_token=True,
    logprobs_mode_override="processed_logits" if self.is_processed_logprobs_mode else "raw_logits"
)
bonus_token_ids = bonus_sampler_output.sampled_token_ids  # [B, 1]
```

> **关键**: 此步调用完整的 `AscendSampler.forward()` 流程（详见第一部分 3.3），包含 logits 预处理、温度缩放、Top-K/Top-P、Gumbel-Max 采样等全部步骤。

#### 4.1.3 Step 2-3: Target Logits 提取与 Logits Processors

```python
# vllm/v1/sample/rejection_sampler.py:104-127
raw_target_logits = logits[metadata.target_logits_indices]  # [N, V]
raw_target_logits = raw_target_logits.to(torch.float32)
if not self.is_processed_logprobs_mode:
    target_logits = raw_target_logits.clone()  # 保留原始值

# 应用 logits processors（同传统路径 Step 3）
target_logits = self.apply_logits_processors(target_logits, sampling_metadata, metadata)
```

#### 4.1.4 Step 4: apply_sampling_constraints()（昇腾 Triton 路径）

**文件位置**: `vllm_ascend/sample/rejection_sampler.py:24-117`

```python
def apply_sampling_constraints(logits, cu_num_draft_tokens, sampling_metadata):
    if sampling_metadata.all_greedy:
        return logits  # 贪心快速路径

    # Triton expand_kernel 将 batch 参数扩展到 token 级别
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature, cu_num_draft_tokens, num_tokens,
        replace_from=GREEDY_TEMPERATURE, replace_to=1,
    )
    logits.div_(temperature.unsqueeze(-1))

    top_k = expand_batch_to_tokens(sampling_metadata.top_k, cu_num_draft_tokens, num_tokens)
    top_p = expand_batch_to_tokens(sampling_metadata.top_p, cu_num_draft_tokens, num_tokens)
    return apply_top_k_top_p(logits, top_k, top_p)  # A2/A3: npu_apply_top_k_top_p
```

**expand_batch_to_tokens 的 Triton 路径**:

```python
# vllm_ascend/sample/rejection_sampler.py:439-441
if HAS_TRITON:
    expand_triton(batch_size, expanded_x, x, cu_num_tokens, replace_from, replace_to,
                  max_num_tokens=MAX_SPEC_LEN)
```

```
expand_kernel (Triton JIT):
  输入: x[B], cu_num_tokens[B]
  输出: expanded_x[N]
  逻辑: 根据累积计数将 batch 级值复制到对应 token 位置
  grid: cal_grid_and_block_size(batch_size)
```

#### 4.1.5 Step 5: rejection_sample() 核心流程（HAS_TRITON=true）

**文件位置**: `vllm_ascend/sample/rejection_sampler.py:120-385`

```
rejection_sample(draft_token_ids, num_draft_tokens, max_spec_len,
                 cu_num_draft_tokens, draft_probs, target_logits,
                 bonus_token_ids, sampling_metadata)
    │
    ├── 5a. 创建输出缓冲区
    │   output_token_ids = torch.empty([B, S+1], dtype=int32)
    │   output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)  # 填充 -1
    │
    ├── 5b. 确定验证模式与 grid 配置
    │   using_block_verify = max_spec_len >= 3
    │   grid, block_size = cal_grid_and_block_size(batch_size)
    │   │
    │   └── cal_grid_and_block_size:
    │       ├── vectorcore_num = get_vectorcore_num()
    │       ├── batch_size <= vectorcore_num: grid=batch_size, block=1
    │       └── batch_size > vectorcore_num: grid=vectorcore_num, block=next_power_of_2(⌈B/grid⌉)
    │
    ├── 5c. 贪心采样路径 (if not all_random)
    │   │
    │   ├── target_argmax = target_logits.argmax(dim=-1)  # [N] int64
    │   │
    │   └── rejection_greedy_sample_with_triton():
    │       │
    │       ├── [条件A] spec_len=1 且 all_greedy:
    │       │   rejection_greedy_sample_spec_len_1_triton[(grid,)](
    │       │       output_token_ids, draft_token_ids, target_argmax,
    │       │       bonus_token_ids, vec_len, BLOCK_SIZE=block_size)
    │       │   逻辑: 向量化比较 draft vs target, 匹配时调用 bonus_renew_1
    │       │
    │       └── [条件B] 通用贪心:
    │           rejection_greedy_sample_triton[(grid,)](
    │               output_token_ids, cu_num_draft_tokens, draft_token_ids,
    │               target_argmax, bonus_token_ids, is_greedy,
    │               vec_len, max_spec_len, BLOCK_SIZE=block_size)
    │           逻辑: 逐请求遍历 draft tokens, 首次不匹配即拒绝,
    │                  全部匹配时调用 bonus_renew
    │
    │   └── if all_greedy: return output_token_ids  ◀─ 快速返回
    │
    ├── 5d. 计算 target 概率分布
    │   target_probs = target_logits.softmax(dim=-1, dtype=fp32)  # [N, V]
    │
    ├── 5e. 生成均匀随机数
    │   uniform_probs = generate_uniform_probs(num_tokens, num_draft_tokens,
    │       generators, device)  # [N]
    │
    ├── 5f. 预计算恢复 tokens
    │   sample_recovered_tokens():
    │   │
    │   ├── q = torch.empty([B, V], fp32).exponential_()   # AI-CPU
    │   ├── for i, gen in generators.items():
    │   │       q[i].exponential_(generator=gen)            # AI-CPU (per-request)
    │   └── sample_recovered_tokens_kernel[(B, S)](
    │           recovered_token_ids, cu_num_draft_tokens,
    │           draft_token_ids, draft_probs, target_probs, q,
    │           vocab_size, PADDED_VOCAB_SIZE, NO_DRAFT_PROBS, SUB_BLOCK=4096)
    │       逻辑:
    │         ├── N-gram模式: target_probs[draft_token_id]置0 → prob/q → argmax
    │         └── 有draft_probs: max(0, target-draft)/q → argmax
    │
    └── 5g. 随机采样拒绝采样
        │
        ├── [max_spec_len < 3] 逐个验证:
        │   rejection_random_sample_kernel[(grid,)](
        │       output_token_ids, cu_num_draft_tokens, draft_token_ids,
        │       draft_probs, target_probs, bonus_token_ids,
        │       recovered_token_ids, uniform_probs.to(fp32), is_greedy,
        │       max_spec_len, vocab_size, batch_size,
        │       NO_DRAFT_PROBS, BLOCK_SIZE=block_size)
        │   逻辑: 逐token判断 target_prob/draft_prob >= uniform_prob
        │         接受→写入draft_token, 拒绝→写入recovered_token
        │         全部接受→追加bonus_token
        │
        └── [max_spec_len >= 3] Block Verify:
            rejection_random_sample_block_verify_kernel[(grid,)](
                ...同上参数...)
            逻辑: π = ∏min(target/draft, 1.0), u = ∏uniform
                  π >= u → 接受, 找到 last_accepted_pos
                  接受位置写入draft_token, 拒绝位置写入recovered_token
                  全部接受→追加bonus_token
```

#### 4.1.6 执行路径总结

```
┌──────────────────────────────────────────────────────────────────────────┐
│ RejectionSampler.forward() + rejection_sample() 执行路径               │
│ (HAS_TRITON=true)                                                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  路径 A: 全部贪心 (all_greedy=True)                                      │
│  ─────────────────────────────────────                                   │
│  bonus: AscendSampler.forward(bonus_logits)                              │
│  target: logits[indices] → fp32 → apply_logits_processors               │
│  rejection: argmax → rejection_greedy_sample_*_triton → 返回             │
│  特点: 跳过 softmax/uniform/recovered, 无随机采样路径                     │
│                                                                          │
│  路径 B: 全部随机 (all_random=True)                                      │
│  ─────────────────────────────────────                                   │
│  bonus: AscendSampler.forward(bonus_logits)                              │
│  target: logits[indices] → fp32 → apply_logits_processors               │
│          → apply_sampling_constraints (Triton expand + TopKTopP)         │
│  rejection: softmax → uniform → recovered (Triton kernel)                │
│          → rejection_random_sample_*_kernel → 返回                       │
│  特点: 跳过贪心 argmax 路径                                              │
│                                                                          │
│  路径 C: 混合模式 (部分贪心 + 部分随机)                                   │
│  ─────────────────────────────────────                                   │
│  bonus: AscendSampler.forward(bonus_logits)                              │
│  target: logits[indices] → fp32 → apply_logits_processors               │
│          → apply_sampling_constraints (Triton expand + TopKTopP)         │
│  rejection:                                                              │
│    贪心部分: argmax → rejection_greedy_sample_triton(is_greedy mask)     │
│    随机部分: softmax → uniform → recovered (Triton kernel)               │
│          → rejection_random_sample_*_kernel(is_greedy mask) → 返回      │
│  特点: Triton kernel 内部通过 is_greedy 掩码区分贪心/随机请求            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 全流程算子分析（HAS_TRITON=true, Triton 改造参考）

> **目标**: 对 `RejectionSampler.forward()` → `rejection_sample()` 在 HAS_TRITON=true 路径下，每一步操作进行 **API/算子级拆解**、**运行位置标注**、**数据流形状追踪**。

#### 4.2.1 算子全景总表

| 序号 | 步骤 | PyTorch API / Triton Kernel | 底层算子(NPU) | 运行位置 | 输入形状 | 输出形状 | 备注 |
|------|------|---------------------------|--------------|---------|---------|---------|------|
| 1a | 提取 bonus logits | `logits[bonus_logits_indices]` | IndexSelect/GatherV2 | NPU-Vector | `[N+B, V]` + `[B]` | `[B, V]` | 高级索引 |
| 1b | Bonus 采样 | `AscendSampler.forward()` | (见第一部分 3.4) | NPU-Vector | `[B, V]` fp16/bf16 | `[B, 1]` int32 | 完整传统采样流程 |
| 2a | 提取 target logits | `logits[target_logits_indices]` | IndexSelect/GatherV2 | NPU-Vector | `[N+B, V]` + `[N]` | `[N, V]` | 高级索引 |
| 2b | Cast fp32 | `raw_target_logits.to(fp32)` | Cast | NPU-Vector | `[N, V]` fp16/bf16 | `[N, V]` fp32 | |
| 2c | Clone | `raw_target_logits.clone()` | Clone | NPU-Vector | `[N, V]` fp32 | `[N, V]` fp32 | 保留原始值给logprobs |
| 3a | masked_fill_ | `logits.masked_fill_(mask, -inf)` | MaskedFill | NPU-Vector | `[N, V]` fp32 + `[N, V]` bool | `[N, V]` fp32 | allowed_token_ids白名单 |
| 3b | bad_words | `logits[i][token_id] = -inf` | 索引赋值 | **CPU→NPU** | 逐请求 | 同输入 | CPU循环, ⚠️瓶颈 |
| 3c | logit_bias | `logits += bias` | Add | NPU-Vector | `[N, V]` fp32 | `[N, V]` fp32 | non_argmax_invariant |
| 3d | apply_penalties | (scatter_add + repetition + freq + pres) | 多算子 | NPU-Vector+CPU | `[N, V]` fp32 | `[N, V]` fp32 | 同传统路径 3.4 的 3d |
| 4a | **expand temperature** | **`expand_kernel`** | **Triton JIT** | **NPU-Vector** | `[B]` fp32 + `[B]` int32 | `[N]` fp32 | batch→token 扩展 |
| 4b | temp div | `logits.div_(temp.unsqueeze(-1))` | Unsqueeze + Div_ | NPU-Vector | `[N, V]` / `[N, 1]` | `[N, V]` fp32 | 原地温度缩放 |
| 4c | **expand top_k** | **`expand_kernel`** | **Triton JIT** | **NPU-Vector** | `[B]` int32 + `[B]` int32 | `[N]` int32 | batch→token 扩展 |
| 4d | **expand top_p** | **`expand_kernel`** | **Triton JIT** | **NPU-Vector** | `[B]` fp32 + `[B]` int32 | `[N]` fp32 | batch→token 扩展 |
| 4e | top_k_top_p | `npu_apply_top_k_top_p` | **AscendC 自定义** | NPU-Vector | `[N, V]` + `[N]` + `[N]` | `[N, V]` fp32 | A2/A3; 其他走PyTorch sort |
| 5a | 创建输出缓冲 | `torch.empty + fill_` | Empty + Fill_ | NPU-Vector | - | `[B, S+1]` int32 | PLACEHOLDER=-1 |
| 5b | cal_grid_and_block | `get_vectorcore_num()` | Host 计算 | **Host CPU** | batch_size | (grid, block) | Triton launch 配置 |
| 5c-i | argmax | `target_logits.argmax(dim=-1)` | ArgMaxWithValue | NPU-Vector | `[N, V]` fp32 | `[N]` int64 | 贪心路径 |
| 5c-ii | **贪心拒绝(spec_len=1)** | **`rejection_greedy_sample_spec_len_1_triton`** | **Triton JIT** | **NPU-Vector** | `[N]` + `[N]` + `[B]` | `[B, 2]` int32 | 向量化比较 + bonus_renew_1 |
| 5c-iii | **贪心拒绝(通用)** | **`rejection_greedy_sample_triton`** | **Triton JIT** | **NPU-Vector** | `[N]` + `[N]` + `[B]` + `[B]` | `[B, S+1]` int32 | 逐请求遍历 + bonus_renew |
| 5d | softmax | `target_logits.softmax(dim=-1, fp32)` | SoftmaxV2 | NPU-Vector | `[N, V]` fp32 | `[N, V]` fp32 | 计算 target 概率 |
| 5e-i | uniform_probs 分配 | `torch.empty + rand_()` | Empty + Rand | **AI-CPU** | - | `[N]` fp32 | 均匀随机数 |
| 5e-ii | uniform per-gen | `q[i].uniform_(generator=gen)` | Rand(per-row) | **AI-CPU** | 逐请求 | `[N]` fp32 | 可复现 seed |
| 5f-i | q 分配+exponential | `torch.empty + exponential_()` | Empty + Exponential | **AI-CPU** | - | `[B, V]` fp32 | ⚠️ AI-CPU, vocab大时慢 |
| 5f-ii | q per-gen | `q[i].exponential_(generator=gen)` | Exponential(per-row) | **AI-CPU** | 逐请求 | `[B, V]` fp32 | 可复现 seed |
| 5f-iii | **恢复token采样** | **`sample_recovered_tokens_kernel`** | **Triton JIT** | **NPU-Vector** | `[N,V]` + `[B,V]` + `[N]` | `[N]` int32 | grid=(B,S), SUB_BLOCK=4096 |
| 5g-i | uniform cast | `uniform_probs.to(fp32)` | Cast | NPU-Vector | `[N]` | `[N]` fp32 | 确保 fp32 |
| 5g-ii | **随机拒绝(逐个)** | **`rejection_random_sample_kernel`** | **Triton JIT** | **NPU-Vector** | 多输入 | `[B, S+1]` int32 | max_spec_len < 3 |
| 5g-iii | **随机拒绝(块验证)** | **`rejection_random_sample_block_verify_kernel`** | **Triton JIT** | **NPU-Vector** | 多输入 | `[B, S+1]` int32 | max_spec_len >= 3 |

> **图例**: B=batch_size, N=num_draft_tokens(展平总数), S=max_spec_len, V=vocab_size(152064 for Qwen)
>
> **加粗标识 Triton kernel**; 步骤 1b (Bonus采样) 内部算子详见第一部分 3.4
>
> **特殊分支说明**:
> - `all_greedy=True`: 执行 5a→5c, 跳过 5d-5g
> - `all_random=True`: 跳过 5c, 直接走 5d→5g
> - `mixed`: 5c 和 5g 都执行, Triton kernel 通过 `is_greedy` 掩码区分请求

#### 4.2.2 数据流详细追踪

```
输入: logits [N+B, V] fp16/bf16  (来自 compute_logits)
      metadata.bonus_logits_indices [B], metadata.target_logits_indices [N]
 │
 ├─ 提取 Bonus Logits:
 │   logits[bonus_logits_indices] → bonus_logits [B, V] fp16
 │   算子: IndexSelect/GatherV2    位置: NPU-Vector
 │   │
 │   └─ AscendSampler.forward(bonus_logits) → bonus_token_ids [B, 1] int32
 │       (完整传统采样流程, 详见第一部分 3.4)
 │
 ├─ 提取 Target Logits:
 │   logits[target_logits_indices] → raw_target_logits [N, V] fp16
 │   算子: IndexSelect/GatherV2    位置: NPU-Vector
 │   │
 │   ├─ .to(fp32) → [N, V] fp32
 │   │   算子: Cast                 位置: NPU-Vector
 │   │
 │   └─ .clone() → target_logits [N, V] fp32 (新张量)
 │       算子: Clone                位置: NPU-Vector
 │
 ├─ apply_logits_processors() → target_logits [N, V] fp32 (原地修改)
 │   (同传统路径: masked_fill_ → bad_words → non_argmax → penalties)
 │
 ├─ apply_sampling_constraints():
 │   │
 │   ├─[贪心快速路径] all_greedy=True: 直接返回, 不处理
 │   │
 │   └─[需要约束处理]:
 │       ├─ expand_kernel (Triton):
 │       │   temperature[B] → expanded_temperature[N]
 │       │   位置: NPU-Vector (Triton JIT)
 │       │   数据流: x[B] + cu_num_draft_tokens[B] → expanded_x[N]
 │       │   注: replace_from=0(GREEDY_TEMP) → replace_to=1
 │       │
 │       ├─ Div_: logits[N,V] / temperature[N,1] → logits[N,V] (原地)
 │       │   算子: Unsqueeze + Div_   位置: NPU-Vector
 │       │
 │       ├─ expand_kernel (Triton): top_k[B] → expanded_top_k[N]
 │       ├─ expand_kernel (Triton): top_p[B] → expanded_top_p[N]
 │       │
 │       └─ apply_top_k_top_p(logits[N,V], top_k[N], top_p[N])
 │           A2/A3: npu_apply_top_k_top_p (AscendC)  位置: NPU-Vector
 │           其他: PyTorch sort+mask 路径
 │           数据流: logits[N,V] → filtered_logits[N,V] (被mask位=-inf)
 │
 ├─ rejection_sample():
 │   │
 │   ├─ Empty + Fill_: output_token_ids [B, S+1] int32 = PLACEHOLDER(-1)
 │   │   算子: Empty + Fill_          位置: NPU-Vector
 │   │
 │   ├─ cal_grid_and_block_size(B): 计算 Triton launch 参数
 │   │   位置: Host CPU
 │   │
 │   ├─[贪心路径] (not all_random):
 │   │   │
 │   │   ├─ ArgMax: target_logits[N,V] → target_argmax[N] int64
 │   │   │   算子: ArgMaxWithValue     位置: NPU-Vector
 │   │   │
 │   │   ├─[spec_len=1 且 all_greedy]:
 │   │   │   rejection_greedy_sample_spec_len_1_triton[(grid,)]:
 │   │   │     向量化: store(target_argmax) → 逐元素比较 → bonus_renew_1
 │   │   │     位置: NPU-Vector (Triton)
 │   │   │     输出: output_token_ids[B, 2] 部分填充
 │   │   │
 │   │   └─[通用贪心]:
 │   │       rejection_greedy_sample_triton[(grid,)]:
 │   │         逐请求: load draft → load target_argmax → store target_argmax
 │   │         → compare → reject时停止 → 不reject时 bonus_renew
 │   │         位置: NPU-Vector (Triton)
 │   │         输出: output_token_ids[B, S+1] 部分填充
 │   │
 │   │   └─ if all_greedy: return output_token_ids  ◀─ 快速返回
 │   │
 │   ├─ Softmax: target_logits[N,V] → target_probs[N,V] fp32
 │   │   算子: SoftmaxV2              位置: NPU-Vector
 │   │
 │   ├─ generate_uniform_probs: → uniform_probs[N] fp32
 │   │   算子: Empty + Rand_(uniform)  位置: **AI-CPU**
 │   │   + 逐请求 generator 覆盖
 │   │
 │   ├─ sample_recovered_tokens:
 │   │   │
 │   │   ├─ q = Empty[B,V].exponential_()
 │   │   │   算子: Empty + Exponential  位置: **AI-CPU**
 │   │   │   ⚠️ AI-CPU 执行, V=152064 时耗时显著
 │   │   │
 │   │   ├─ 逐请求 generator 覆盖: q[i].exponential_(gen)
 │   │   │   算子: Exponential(per-row) 位置: **AI-CPU**
 │   │   │
 │   │   └─ sample_recovered_tokens_kernel[(B, S)]:
 │   │       位置: NPU-Vector (Triton JIT)
 │   │       ├─ N-gram模式:
 │   │       │   target_probs[pos][draft_token]=0 → 分块遍历vocab
 │   │       │   → prob/q → argmax → 恢复原始值
 │   │       └─ 有draft_probs:
 │   │           max(0, target_prob - draft_prob) / q → argmax
 │   │       输出: recovered_token_ids[N] int32
 │   │
 │   └─[随机路径]:
 │       ├─ Cast: uniform_probs.to(fp32)
 │       │   算子: Cast                位置: NPU-Vector
 │       │
 │       ├─[max_spec_len < 3] rejection_random_sample_kernel[(grid,)]:
 │       │   位置: NPU-Vector (Triton JIT)
 │       │   逻辑: 逐请求逐token:
 │       │     load draft_token_id → load draft_prob, target_prob, uniform
 │       │     → if draft_prob>0 && target_prob/draft_prob >= uniform: accept
 │       │     → else: reject, 使用 recovered_token
 │       │     → 全部accept: 追加 bonus_token
 │       │   输出: output_token_ids[B, S+1] 完成填充
 │       │
 │       └─[max_spec_len >= 3] rejection_random_sample_block_verify_kernel[(grid,)]:
 │           位置: NPU-Vector (Triton JIT)
 │           逻辑: 逐请求:
 │             π=1.0, u=1.0
 │             for pos in range(num_draft):
 │               π = min(π * target/draft, 1.0)
 │               u = u * uniform[pos]
 │               if draft_prob>0 && π>=u: last_accepted=pos
 │             接受位置: store draft_token
 │             拒绝位置: store recovered_token
 │             全部接受: store bonus_token
 │           输出: output_token_ids[B, S+1] 完成填充
 │
 └─ 返回 SamplerOutput:
     ├─ sampled_token_ids: output_token_ids [B, S+1] int32
     └─ logprobs_tensors: LogprobsTensors | None
```

#### 4.2.3 运行位置分布统计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                投机推理后处理算子运行位置分布 (HAS_TRITON=true)                 │
├──────────────┬──────────────────────────────────────────────────────────────┤
│  NPU-Vector  │ ████████████████████████████████████████████  ~70%           │
│  (PyTorch)   │ IndexSelect, Cast, Clone, MaskedFill, Div_, SoftmaxV2,     │
│              │ ArgMaxWithValue, Fill_, Unsqueeze, penalties系列              │
├──────────────┼──────────────────────────────────────────────────────────────┤
│  NPU-Vector  │ ██████████████  ~18%                                        │
│  (Triton)    │ expand_kernel ×3, rejection_greedy_sample_*_triton,         │
│              │ sample_recovered_tokens_kernel,                              │
│              │ rejection_random_sample_*_kernel                             │
├──────────────┼──────────────────────────────────────────────────────────────┤
│  AI-CPU      │ ██████  ~8%                                                 │
│              │ exponential_(恢复token随机数), rand_(均匀随机数)               │
│              │ ⚠️ B×V=152064 时单次 exponential_ 耗时显著                   │
├──────────────┼──────────────────────────────────────────────────────────────┤
│  Host CPU    │ ██  ~3%                                                     │
│              │ cal_grid_and_block_size, bad_words循环,                      │
│              │ _convert_to_tensors(H2D), pin_memory tensor构造             │
├──────────────┼──────────────────────────────────────────────────────────────┤
│  NPU-Cube    │ ▏ ~0%                                                       │
│              │ 无矩阵乘算子, Cube单元空闲                                    │
├──────────────┼──────────────────────────────────────────────────────────────┤
│  AscendC     │ █  ~1%                                                      │
│              │ npu_apply_top_k_top_p (A2/A3, apply_sampling_constraints)   │
└──────────────┴──────────────────────────────────────────────────────────────┘
```

#### 4.2.4 Triton Kernel 详细说明

| Kernel 名称 | grid 配置 | 功能 | 调用条件 | 核心逻辑 |
|------------|----------|------|---------|---------|
| `expand_kernel` | `(grid,)` | batch→token 参数扩展 | 温度/top_k/top_p 扩展 | 根据 cu_num_tokens 将 x[B] 复制到 expanded[N], 支持 replace_from→replace_to |
| `rejection_greedy_sample_spec_len_1_triton` | `(grid,)` | spec_len=1 贪心采样 | all_greedy 且 min/max(num_draft)=1 | 向量化: store target_argmax → 逐元素比较 → 匹配时 bonus_renew_1 |
| `rejection_greedy_sample_triton` | `(grid,)` | 通用贪心采样 | not all_random | 逐请求: 遍历 draft tokens, 首次不匹配停止, 全匹配时 bonus_renew |
| `bonus_renew` / `bonus_renew_1` | (被调用) | 写入 bonus token | 贪心全匹配 | store bonus_token 到 output 的对应位置 |
| `sample_recovered_tokens_kernel` | `(B, S)` | 恢复 token 采样 | 随机路径 | 分块(SUB_BLOCK=4096)遍历vocab, max(0,target-draft)/q → argmax |
| `rejection_random_sample_kernel` | `(grid,)` | 随机逐个验证 | max_spec_len < 3 | 逐token: target/draft >= uniform → accept/reject |
| `rejection_random_sample_block_verify_kernel` | `(grid,)` | 随机块验证 | max_spec_len >= 3 | 累积乘积: π=∏min(t/d,1), u=∏uniform, π>=u → accept |

#### 4.2.5 关键性能瓶颈标注

| 瓶颈点 | 原因 | 影响程度 | 优化方向 |
|--------|------|---------|---------|
| `exponential_()` 在 AI-CPU 执行 | [B, V] 尺寸大, AI-CPU 调度慢 | ★★★★★ | Triton 内 `tl.rand` + 变换, 或异步预计算 |
| `generate_uniform_probs` 在 AI-CPU 执行 | rand_() 走 AI-CPU | ★★★☆☆ | Triton 内生成, 或与 rejection kernel 融合 |
| `apply_logits_processors` 中 penalties H2D | 每步 CPU→NPU 传输 output_token_ids | ★★★★☆ | NPU 侧维护 token_ids 缓存 |
| `apply_bad_words` CPU 循环 | Python for 循环逐请求 | ★★★★☆ | batch mask + masked_fill_ |
| `sample_recovered_tokens_kernel` vocab 遍历 | V=152064, SUB_BLOCK=4096, 需 ~37 轮循环 | ★★★☆☆ | 增大 SUB_BLOCK 或多级 argmax |
| Bonus 采样走完整 AscendSampler 流程 | 包含 penalties/logprobs 等不一定需要的步骤 | ★★☆☆☆ | 简化 bonus 采样专用路径 |

#### 4.2.6 与传统后处理算子对比

| 维度 | 传统后处理 (3.4) | 投机推理后处理 (4.2) |
|------|-----------------|---------------------|
| **主要算子类型** | PyTorch 原生 + 1 个 AscendC | PyTorch + AscendC + **8 个 Triton kernel** |
| **随机数生成** | 1 次 `exponential_()` [B, V] | 1 次 `exponential_()` [B, V] + 1 次 `rand_()` [N] |
| **Top-K/Top-P** | 1 次 (batch_size 行) | 1 次 (num_tokens 行, 通常 > batch_size) |
| **索引操作** | 无 | 2 次高级索引 (bonus + target 提取) |
| **batch→token 扩展** | 无 | 3 次 Triton expand_kernel |
| **核心采样** | argmax + Gumbel-Max | argmax + Triton 拒绝采样 kernel |
| **AI-CPU 依赖** | `exponential_()` | `exponential_()` + `rand_()` |
| **输出形状** | `[B, 1]` | `[B, S+1]` (多 token 输出) |

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

---

# 第六部分：GPU 投机推理后处理流程（v1 架构）

> **说明**: 本部分整理 vLLM 社区 GPU 版本（v1 架构）中投机推理后处理的完整流程，结构参照第二部分（Ascend NPU 版本），便于对比。

## 1. 初始化

### 1.1 RejectionSampler 类定义

**文件位置**: `vllm/v1/sample/rejection_sampler.py`

```python
class RejectionSampler(nn.Module):
    def __init__(self, sampler: Sampler):
        super().__init__()
        self.sampler = sampler  # 复用传统采样器（GPU版Sampler）
        logprobs_mode = self.sampler.logprobs_mode
        self.is_processed_logprobs_mode = logprobs_mode.startswith("processed")
        self.is_logits_logprobs_mode = logprobs_mode.endswith("logits")
```

### 1.2 RejectionSampler 在 GPUModelRunner 中的初始化

**文件位置**: `vllm/v1/worker/gpu_model_runner.py:491-526`

```python
if self.speculative_config and get_pp_group().is_last_rank:
    if self.speculative_config.method == "ngram":
        self.drafter = NgramProposer(self.vllm_config)
    elif self.speculative_config.uses_draft_model():
        self.drafter = DraftModelProposer(vllm_config=self.vllm_config, device=self.device, runner=self)
    elif self.speculative_config.method == "suffix":
        self.drafter = SuffixDecodingProposer(self.vllm_config)
    elif self.speculative_config.use_eagle():
        self.drafter = EagleProposer(self.vllm_config, self.device, self)
    elif self.speculative_config.method == "medusa":
        self.drafter = MedusaProposer(vllm_config=self.vllm_config, device=self.device)
    self.rejection_sampler = RejectionSampler(self.sampler)
```

### 1.3 SpecDecodeMetadata 数据结构

**文件位置**: `vllm/v1/spec_decode/metadata.py`（GPU/Ascend 共用）

```python
@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor       # [num_draft_tokens]
    num_draft_tokens: list[int]         # [batch_size]
    cu_num_draft_tokens: torch.Tensor   # [batch_size]
    cu_num_sampled_tokens: torch.Tensor # [batch_size]
    target_logits_indices: torch.Tensor # [num_draft_tokens]
    bonus_logits_indices: torch.Tensor  # [batch_size]
    logits_indices: torch.Tensor        # [total_tokens]

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens)
```

---

## 2. 配置

### 2.1 Draft Proposer 类型（GPU）

| Proposer | 文件位置 | 说明 |
|----------|----------|------|
| NgramProposer | `vllm/v1/spec_decode/ngram_proposer.py` | N-gram 匹配，无概率分布 |
| EagleProposer | `vllm/v1/spec_decode/eagle.py` | EAGLE/EAGLE3 模型预测 |
| MedusaProposer | `vllm/v1/spec_decode/medusa.py` | Medusa 多头预测 |
| DraftModelProposer | `vllm/v1/spec_decode/draft_model.py` | 独立 Draft 模型 |
| SuffixDecodingProposer | `vllm/v1/spec_decode/suffix_decoding.py` | 后缀解码 |

### 2.2 SpecDecodeMetadata 构建（GPU 版本）

**文件位置**: `vllm/v1/worker/gpu_model_runner.py:2209-2286`

> **与 Ascend 版本关键差异**:
> 1. 使用 `_get_cumsum_and_arange` 辅助函数替代手动 repeat+arange 计算
> 2. 无 PCP 相关逻辑
> 3. H2D 传输直接 `from_numpy().to(device)`，无需 `pin_memory()`

#### 2.2.1 整体流程概览

```
_calc_spec_decode_metadata(num_draft_tokens, cu_num_scheduled_tokens)
    │
    ├── Step 1: 计算 num_sampled_tokens 和 cu_num_sampled_tokens + arange
    │   └── num_sampled_tokens = num_draft_tokens + 1
    │   └── cu_num_sampled_tokens, arange = _get_cumsum_and_arange(num_sampled_tokens)
    │
    ├── Step 2: 构建 logits_indices（CPU numpy 计算）
    │   ├── logits_indices = repeat(cu_num_scheduled - num_sampled, num_sampled)
    │   └── logits_indices += arange
    │
    ├── Step 3: 计算 bonus_logits_indices
    │   └── bonus_logits_indices = cu_num_sampled_tokens - 1
    │
    ├── Step 4: 构建 target_logits_indices（CPU numpy 计算）
    │   ├── cu_num_draft_tokens, arange = _get_cumsum_and_arange(num_draft_tokens)
    │   ├── target_logits_indices = repeat(cu_num_sampled - num_sampled, num_draft)
    │   └── target_logits_indices += arange
    │
    ├── Step 5: CPU→GPU 数据传输（5个张量 from_numpy + to(device)）
    │
    ├── Step 6: 计算 draft_token_ids（GPU 上执行）
    │   ├── draft_token_ids = input_ids.gpu[logits_indices]
    │   └── draft_token_ids = draft_token_ids[target_logits_indices + 1]
    │
    └── Step 7: 构建并返回 SpecDecodeMetadata
```

#### 2.2.2 `_get_cumsum_and_arange` 辅助函数

**文件位置**: `vllm/v1/worker/gpu_model_runner.py:1329-1347`

```python
def _get_cumsum_and_arange(self, num_tokens, cumsum_dtype=None):
    """
    例: [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
    """
    cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
    total_num_tokens = cu_num_tokens[-1]
    cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
    arange = self.arange_np[:total_num_tokens] - cumsums_offsets
    return cu_num_tokens, arange
```

> **对比 Ascend**: Ascend 版本在 `_calc_spec_decode_metadata` 中手动执行 repeat+arange 逻辑，GPU 版本将其封装为可复用的 `_get_cumsum_and_arange` 方法。两者数学计算完全一致。

#### 2.2.3 数值示例

与 Ascend 版本完全一致（参见 2.3.2），此处省略。

#### 2.2.4 源码逐段分析

**Step 1-2: 计算 logits_indices**

```python
# vllm/v1/worker/gpu_model_runner.py:2226-2238
num_sampled_tokens = num_draft_tokens + 1
cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
    num_sampled_tokens, cumsum_dtype=np.int32
)
logits_indices = np.repeat(
    cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
)
logits_indices += arange
```

**Step 3-4: 计算 bonus 和 target logits 索引**

```python
# vllm/v1/worker/gpu_model_runner.py:2241-2254
bonus_logits_indices = cu_num_sampled_tokens - 1

cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
    num_draft_tokens, cumsum_dtype=np.int32
)
target_logits_indices = np.repeat(
    cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
)
target_logits_indices += arange
```

**Step 5: CPU→GPU 传输**

```python
# vllm/v1/worker/gpu_model_runner.py:2257-2271
cu_num_draft_tokens   = torch.from_numpy(cu_num_draft_tokens).to(self.device, non_blocking=True)
cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(self.device, non_blocking=True)
logits_indices        = torch.from_numpy(logits_indices).to(self.device, non_blocking=True)
target_logits_indices = torch.from_numpy(target_logits_indices).to(self.device, non_blocking=True)
bonus_logits_indices  = torch.from_numpy(bonus_logits_indices).to(self.device, non_blocking=True)
```

> **与 Ascend 差异**: GPU 版本直接 `from_numpy().to(device)`，CUDA 自动使用 pinned memory 路径进行异步传输。Ascend 版本需要显式 `pin_memory()` + `to(device)`。

**Step 6: 获取 draft token IDs（GPU 上执行）**

```python
# vllm/v1/worker/gpu_model_runner.py:2275-2276
draft_token_ids = self.input_ids.gpu[logits_indices]
draft_token_ids = draft_token_ids[target_logits_indices + 1]
```

#### 2.2.5 算子全景总表

| 序号 | 步骤 | API / 算子 | 运行位置 | 输入形状 | 输出形状 | 备注 |
|------|------|-----------|---------|---------|---------|------|
| 1 | num_sampled_tokens | `num_draft_tokens + 1` (numpy) | **CPU** | `[B]` int32 | `[B]` int32 | |
| 2 | cu_num_sampled + arange | `_get_cumsum_and_arange()` | **CPU** | `[B]` int32 | `[B]` + `[T_s]` | 封装 cumsum+repeat+arange |
| 3 | logits_indices(base) | `np.repeat(arr, num_sampled)` | **CPU** | `[B]` + `[B]` | `[T_s]` int32 | |
| 4 | logits_indices(final) | `logits_indices += arange` | **CPU** | `[T_s]` + `[T_s]` | `[T_s]` int64 | |
| 5 | bonus_logits_indices | `cu_num_sampled - 1` | **CPU** | `[B]` | `[B]` int32 | |
| 6 | cu_num_draft + arange | `_get_cumsum_and_arange()` | **CPU** | `[B]` int32 | `[B]` + `[T_d]` | |
| 7 | target_logits_indices | `repeat + arange` | **CPU** | `[B]` + `[T_d]` | `[T_d]` int64 | |
| 8a-e | H2D 传输 ×5 | `from_numpy().to(device)` | **CPU→GPU** | 各不同 | 各不同 GPU tensor | non_blocking |
| 9 | draft_token_ids(索引1) | `input_ids.gpu[logits_indices]` | **GPU** | `[max_tokens]` + `[T_s]` | `[T_s]` | IndexSelect |
| 10 | draft_token_ids(索引2) | `[target_logits_indices + 1]` | **GPU** | `[T_s]` + `[T_d]` | `[T_d]` | Add + IndexSelect |

> **图例**: B=batch_size, T_s=total_num_sampled_tokens, T_d=total_num_draft_tokens

---

## 3. 入口函数和执行流程

### 3.1 入口函数

**文件位置**: `vllm/v1/worker/gpu_model_runner.py:2927-2955`

```python
def _sample(self, logits, spec_decode_metadata):
    sampling_metadata = self.input_batch.sampling_metadata
    # 异步调度时用上轮采样结果更新 output_token_ids（供惩罚项计算使用）
    self.input_batch.update_async_output_token_ids()
    if spec_decode_metadata is None:
        return self.sampler(logits=logits, sampling_metadata=sampling_metadata)

    # 异步调度时用上轮 draft token IDs 更新 spec_token_ids
    # 仅在需要 output_token_ids 时执行（penalties 或 bad_words 在使用中）
    if self.use_async_scheduling and self._draft_token_req_ids is not None:
        draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
        self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)

    # 投机推理后处理路径
    sampler_output = self.rejection_sampler(
        spec_decode_metadata,
        None,  # draft_probs（当前 GPU 实现中始终为 None，所有 proposer 类型均不传递 draft 概率）
        logits,
        sampling_metadata,
    )
    return sampler_output
```

### 3.2 完整执行流程

```
sample_tokens() [gpu_model_runner.py:3693]
    │
    ├── 0. kv_connector_output 处理
    │       ├── 非最后PP rank 时提前返回
    │       └── PP+KV transfer 场景处理
    │
    ├── 1. 解包 execute_model_state（10个字段）
    │       └── scheduler_output, logits, spec_decode_metadata,
    │           spec_decode_common_attn_metadata, hidden_states,
    │           sample_hidden_states, aux_hidden_states,
    │           ec_connector_output, cudagraph_stats, slot_mappings
    │
    ├── 2. 应用 grammar bitmask（如有结构化输出）
    │
    ├── 3. 调用 _sample(logits, spec_decode_metadata)
    │       │
    │       ├── 3.0a update_async_output_token_ids()
    │       │       └── 异步调度时用上轮采样结果更新 output_token_ids
    │       │
    │       ├── 3.0b update_async_spec_token_ids()
    │       │       └── 异步调度时用上轮 draft token IDs 更新 spec_token_ids
    │       │           （供惩罚项/坏词计算使用）
    │       │
    │       └── RejectionSampler.forward()
    │               ├── 3.1 采样 bonus tokens
    │               ├── 3.2 处理 target logits
    │               ├── 3.3 rejection_sample()
    │               └── 3.4 返回 SamplerOutput
    │
    ├── 4. _update_states_after_model_execute()
    │       └── Hybrid模型（Mamba）状态更新，计算accepted tokens数
    │
    ├── 5. PP 异步广播 sampled_token_ids
    │       └── _pp_broadcast_prev_sampled_token_ids()
    │
    ├── 6. 清理 draft token 状态
    │       └── _draft_token_ids = None, prev_sampled_token_ids = None
    │
    ├── 7. Draft 提案策略分支
    │       ├── 路径A (EAGLE/DraftModel + fit):
    │       │       └── 在bookkeeping前直接用GPU tokens提案
    │       ├── 路径B (EAGLE/DraftModel + 不fit):
    │       │       ├── prepare_next_token_ids_padded()
    │       │       └── zeros fallback + _copy_draft_token_ids_to_cpu()
    │       └── 路径C (Ngram/Suffix):
    │               └── 标记 propose_drafts_after_bookkeeping = True
    │
    ├── 8. _bookkeeping_sync()
    │       ├── 8.1 NaN 检测 _get_nans_in_logits(logits)
    │       ├── 8.2 丢弃请求 generator offset 回退
    │       ├── 8.3 拷贝 req_ids（防异步修改）
    │       ├── 8.4 同步路径: parse_output() → valid_sampled_token_ids
    │       ├── 8.5 异步路径: 缓存 GPU tokens，延迟拷贝
    │       ├── 8.6 更新 token_ids_cpu + num_tokens_no_spec + req_state
    │       └── 8.7 计算 prompt_logprobs
    │
    ├── 9. 延迟 draft 提案（路径C: ngram等）
    │       └── propose_draft_token_ids(valid_sampled_token_ids)
    │
    ├── 10. clear_kv_connector_metadata()
    │       └── 延迟到 draft model 运行后再清理 KV 元数据
    │
    ├── 11. eplb_step()
    │       └── Expert Load Balancing 步进
    │
    ├── 12. 构建 ModelRunnerOutput（含完整字段）
    │       └── req_ids, req_id_to_index, sampled_token_ids,
    │           logprobs, prompt_logprobs_dict, kv_connector_output,
    │           ec_connector_output, num_nans_in_logits, cudagraph_stats
    │
    └── 13. 异步调度返回路径
            ├── 同步: 直接返回 ModelRunnerOutput
            └── 异步: AsyncGPUModelRunnerOutput 构建
                    └── set_async_sampled_token_ids()（保存异步拷贝引用）
```

### 3.3 RejectionSampler.forward() 详细流程

**文件位置**: `vllm/v1/sample/rejection_sampler.py:60-166`（GPU/Ascend 共用）

```python
def forward(self, metadata, draft_probs, logits, sampling_metadata):
    # 1. 提取 bonus logits 并采样 bonus tokens
    bonus_logits = logits[metadata.bonus_logits_indices]
    bonus_sampler_output = self.sampler(
        logits=bonus_logits,
        sampling_metadata=replace(sampling_metadata, max_num_logprobs=-1),
        predict_bonus_token=True,
        # 覆盖 logprobs 模式，后续需要 logits 来计算 accepted token logprobs
        logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode else "raw_logits",
    )
    bonus_token_ids = bonus_sampler_output.sampled_token_ids

    # 2. 提取并处理 target logits
    raw_target_logits = logits[metadata.target_logits_indices]
    raw_target_logits = raw_target_logits.to(torch.float32)
    target_logits = raw_target_logits
    if not self.is_processed_logprobs_mode:
        # 仅在非 processed logprobs 模式下 clone，保留原始 raw_target_logits
        # 用于后续 logprobs 计算（因为 apply_logits_processors 会原地修改）
        target_logits = target_logits.clone()
    target_logits = self.apply_logits_processors(target_logits, sampling_metadata, metadata)
    target_logits = apply_sampling_constraints(
        target_logits, metadata.cu_num_draft_tokens, sampling_metadata
    )

    # 3. 执行拒绝采样
    output_token_ids = rejection_sample(...)

    # 4. 计算 logprobs（如果需要）
    if sampling_metadata.max_num_logprobs is not None:
        logprobs_tensors = self._get_logprobs_tensors(
            ...,
            # 关键：根据 logprobs 模式选择传入 processed 或 raw target logits
            target_logits if self.is_processed_logprobs_mode else raw_target_logits,
            bonus_sampler_output.logprobs_tensors.logprobs,
            output_token_ids,
        )

    return SamplerOutput(sampled_token_ids=output_token_ids, logprobs_tensors=logprobs_tensors)
```

---

## 4. 功能流程与算子分析

### 4.1 RejectionSampler.forward() + rejection_sample() 详细流程

#### 4.1.1 整体流程概览

```
forward(metadata, draft_probs, logits, sampling_metadata)
    │
    ├── Step 1: 提取 Bonus Logits 并采样 Bonus Token
    │   ├── bonus_logits = logits[metadata.bonus_logits_indices]        # IndexSelect
    │   ├── replace(sampling_metadata, max_num_logprobs=-1)
    │   ├── logprobs_mode_override="processed_logits"/"raw_logits"     # 覆盖logprobs模式
    │   └── bonus_token_ids = self.sampler(bonus_logits, ...)           # 完整 Sampler.forward()
    │
    ├── Step 2: 提取 Target Logits
    │   ├── raw_target_logits = logits[metadata.target_logits_indices]  # IndexSelect
    │   ├── raw_target_logits = raw_target_logits.to(float32)           # Cast
    │   ├── target_logits = raw_target_logits                           # 默认共享引用
    │   └── if not is_processed_logprobs_mode:                          # 条件 Clone
    │       └── target_logits = target_logits.clone()                   # 保留raw用于logprobs
    │
    ├── Step 3: 应用 Logits Processors (apply_logits_processors)
    │   ├── 3a. _combine_outputs_with_spec_tokens()                    # 合并历史+draft tokens
    │   ├── 3b. 计算 repeat_indices (batch→token 索引映射)
    │   │   └── original_indices.repeat_interleave(num_draft_tokens)   # CPU→GPU
    │   ├── 3c. apply_penalties(repeat_indices) → apply_all_penalties  # repetition/freq/pres
    │   ├── 3d. allowed_token_ids_mask[repeat_indices] → masked_fill_(-inf)
    │   ├── 3e. apply_bad_words_with_drafts()
    │   └── 3f. non_argmax_invariant: MinTokensLogitsProcessor.apply_with_spec_decode()
    │
    ├── Step 4: apply_sampling_constraints()
    │   ├── expand_batch_to_tokens(temperature) → expand_kernel (Triton)
    │   ├── logits.div_(temperature.unsqueeze(-1))
    │   ├── expand_batch_to_tokens(top_k) → expand_kernel (Triton)
    │   ├── expand_batch_to_tokens(top_p) → expand_kernel (Triton)
    │   └── apply_top_k_top_p(logits, top_k, top_p)
    │       ├── batch >= 8: apply_top_k_top_p_triton (Triton kernel)
    │       └── batch < 8: apply_top_k_top_p_pytorch (sort+mask)
    │
    ├── Step 5: rejection_sample()  [核心拒绝采样]
    │   ├── 5a. 创建输出缓冲区 [B, S+1], fill_(PLACEHOLDER)
    │   │
    │   ├── 5b. 贪心路径 (if not all_random):
    │   │   ├── target_argmax = target_logits.argmax(dim=-1)
    │   │   └── rejection_greedy_sample_kernel[(batch_size,)]  (Triton)
    │   │       逻辑: 逐请求遍历 draft tokens, 首次不匹配即拒绝
    │   │              全部匹配时追加 bonus token
    │   │   └── if all_greedy: return  ◀─ 快速返回
    │   │
    │   ├── 5c. target_probs = target_logits.softmax(dim=-1, fp32)
    │   │
    │   ├── 5d. uniform_probs = generate_uniform_probs()
    │   │   └── torch.rand([N], dtype=float64, device=cuda)
    │   │
    │   ├── 5e. recovered_token_ids = sample_recovered_tokens()
    │   │   ├── q = torch.empty([B, V], fp32).exponential_()
    │   │   ├── inv_q = q.reciprocal()  ◀─ GPU特有：预计算倒数
    │   │   └── sample_recovered_tokens_kernel[(B, S)]  (Triton)
    │   │       逻辑: 分块遍历 vocab, prob * inv_q → argmax
    │   │
    │   └── 5f. rejection_random_sample_kernel[(batch_size,)]  (Triton)
    │       逻辑: 逐请求逐 token:
    │         target_prob / draft_prob >= uniform → accept
    │         else → reject, 使用 recovered_token
    │         全部 accept → 追加 bonus_token
    │
    ├── Step 6: 计算 logprobs（可选）
    │   └── _get_logprobs_tensors(...)
    │
    └── Step 7: 返回 SamplerOutput
            ├── sampled_token_ids: [B, S+1] int32
            └── logprobs_tensors: LogprobsTensors | None
```

> **符号说明**: B=batch_size, N=num_draft_tokens(展平总数), S=max_spec_len, V=vocab_size

#### 4.1.2 Step 1: Bonus Token 采样

```python
# vllm/v1/sample/rejection_sampler.py:93-115
bonus_logits = logits[metadata.bonus_logits_indices]  # [B, V]
bonus_sampler_output = self.sampler(
    logits=bonus_logits,
    sampling_metadata=replace(sampling_metadata, max_num_logprobs=-1),
    predict_bonus_token=True,
    logprobs_mode_override="processed_logits" if ... else "raw_logits"
)
bonus_token_ids = bonus_sampler_output.sampled_token_ids  # [B, 1]
```

> **GPU Sampler.forward() 完整流程**:
> 1. compute_logprobs(logits) — 如需 logprobs
> 2. logits.to(float32) — 精度转换
> 3. apply_logits_processors() — 白名单/坏词/惩罚项
> 4. sample() — 贪心 argmax 或温度+Top-K/Top-P+Gumbel-Max
> 5. gather_logprobs() — 收集 top-k logprobs

#### 4.1.3 Step 4: apply_sampling_constraints()（GPU 版本）

**文件位置**: `vllm/v1/sample/rejection_sampler.py:451-506`

```python
def apply_sampling_constraints(logits, cu_num_draft_tokens, sampling_metadata):
    if sampling_metadata.all_greedy:
        return logits  # 贪心快速路径

    # Triton expand_kernel 将 batch 参数扩展到 token 级别
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature, cu_num_draft_tokens, num_tokens,
        replace_from=GREEDY_TEMPERATURE, replace_to=1,
    )
    logits.div_(temperature.unsqueeze(-1))  # 原地温度缩放

    top_k = expand_batch_to_tokens(sampling_metadata.top_k, ...) if top_k else None
    top_p = expand_batch_to_tokens(sampling_metadata.top_p, ...) if top_p else None

    return apply_top_k_top_p(logits, top_k, top_p)
```

**apply_top_k_top_p 的 GPU 路径选择**:

```python
# vllm/v1/sample/ops/topk_topp_sampler.py:245-255
def apply_top_k_top_p(logits, k, p):
    if p is None and k is None:
        return logits
    if HAS_TRITON and logits.shape[0] >= 8:
        return apply_top_k_top_p_triton(logits, k, p)  # Triton 高性能路径
    return apply_top_k_top_p_pytorch(logits, k, p)      # PyTorch sort+mask 回退
```

> **与 Ascend 差异**: GPU 版 `apply_top_k_top_p` 在 batch≥8 时使用 Triton kernel (`topk_topp_triton.py`)，小 batch 使用 PyTorch `sort+mask`。Ascend 版在 A2/A3 上使用 `npu_apply_top_k_top_p` 昇腾原生算子。

#### 4.1.4 Step 5: rejection_sample() 核心流程

**文件位置**: `vllm/v1/sample/rejection_sampler.py:350-448`

```python
def rejection_sample(draft_token_ids, num_draft_tokens, max_spec_len,
                     cu_num_draft_tokens, draft_probs, target_logits,
                     bonus_token_ids, sampling_metadata):
    # 5a. 创建输出缓冲区
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32, device=device)

    # 5b. 贪心路径
    if not sampling_metadata.all_random:
        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids, cu_num_draft_tokens, draft_token_ids,
            target_argmax, bonus_token_ids, is_greedy, max_spec_len)
        if sampling_metadata.all_greedy:
            return output_token_ids  # 快速返回

    # 5c. 计算 target 概率分布
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)

    # 5d. 生成均匀随机数
    uniform_probs = generate_uniform_probs(...)  # [N] float64

    # 5e. 预计算恢复 tokens
    recovered_token_ids = sample_recovered_tokens(...)

    # 5f. 随机拒绝采样
    rejection_random_sample_kernel[(batch_size,)](
        output_token_ids, cu_num_draft_tokens, draft_token_ids,
        draft_probs, target_probs, bonus_token_ids,
        recovered_token_ids, uniform_probs, is_greedy,
        max_spec_len, vocab_size, NO_DRAFT_PROBS=draft_probs is None)
    return output_token_ids
```

#### 4.1.5 sample_recovered_tokens()（GPU 版本）

**文件位置**: `vllm/v1/sample/rejection_sampler.py:604-648`

```python
def sample_recovered_tokens(max_spec_len, num_draft_tokens, cu_num_draft_tokens,
                            draft_token_ids, draft_probs, target_probs,
                            sampling_metadata, device):
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty((batch_size, vocab_size), dtype=torch.float32, device=device)
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    inv_q = q.reciprocal()  # ◀─ GPU特有：预计算倒数，避免kernel内除法

    recovered_token_ids = torch.empty_like(draft_token_ids)
    BLOCK_SIZE = 8192
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids, cu_num_draft_tokens, draft_token_ids,
        draft_probs, target_probs, inv_q,
        vocab_size, BLOCK_SIZE, NO_DRAFT_PROBS=draft_probs is None)
    return recovered_token_ids
```

> **与 Ascend 差异**:
> - GPU 使用 `q.reciprocal()` 预计算 `inv_q`，kernel 内执行 `prob * inv_q`（乘法）
> - Ascend 传入原始 `q`，kernel 内执行 `prob / q`（除法）
> - GPU `BLOCK_SIZE=8192`，Ascend `SUB_BLOCK=4096`
> - GPU 的 `exponential_()` 在 CUDA 上高效执行；Ascend 走 AI-CPU，是性能瓶颈

#### 4.1.6 执行路径总结

```
┌──────────────────────────────────────────────────────────────────────────┐
│ RejectionSampler.forward() + rejection_sample() 执行路径 (GPU v1)       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  路径 A: 全部贪心 (all_greedy=True)                                      │
│  ─────────────────────────────────────                                   │
│  bonus: Sampler.forward(bonus_logits)                                    │
│  target: logits[indices] → fp32 → apply_logits_processors               │
│  rejection: argmax → rejection_greedy_sample_kernel → 返回               │
│  特点: 跳过 softmax/uniform/recovered, 无随机采样路径                     │
│                                                                          │
│  路径 B: 全部随机 (all_random=True)                                      │
│  ─────────────────────────────────────                                   │
│  bonus: Sampler.forward(bonus_logits)                                    │
│  target: logits[indices] → fp32 → apply_logits_processors               │
│          → apply_sampling_constraints (Triton expand + TopK/TopP)        │
│  rejection: softmax → uniform → recovered (Triton kernel)                │
│          → rejection_random_sample_kernel → 返回                         │
│  特点: 跳过贪心 argmax 路径                                              │
│                                                                          │
│  路径 C: 混合模式 (部分贪心 + 部分随机)                                   │
│  ─────────────────────────────────────                                   │
│  bonus: Sampler.forward(bonus_logits)                                    │
│  target: logits[indices] → fp32 → apply_logits_processors               │
│          → apply_sampling_constraints (Triton expand + TopK/TopP)        │
│  rejection:                                                              │
│    贪心部分: argmax → rejection_greedy_sample_kernel(is_greedy mask)     │
│    随机部分: softmax → uniform → recovered (Triton kernel)               │
│          → rejection_random_sample_kernel(is_greedy mask) → 返回        │
│  特点: Triton kernel 内部通过 is_greedy 掩码区分贪心/随机请求            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 全流程算子分析

#### 4.2.1 算子全景总表

| 序号 | 步骤 | PyTorch API / Triton Kernel | 运行位置 | 输入形状 | 输出形状 | 备注 |
|------|------|---------------------------|---------|---------|---------|------|
| 1a | 提取 bonus logits | `logits[bonus_logits_indices]` | **GPU** | `[N+B, V]` + `[B]` | `[B, V]` | IndexSelect |
| 1b | Bonus 采样 | `Sampler.forward()` | **GPU** | `[B, V]` | `[B, 1]` int32 | 完整传统采样流程 |
| 2a | 提取 target logits | `logits[target_logits_indices]` | **GPU** | `[N+B, V]` + `[N]` | `[N, V]` | IndexSelect |
| 2b | Cast fp32 | `.to(fp32)` | **GPU** | `[N, V]` fp16/bf16 | `[N, V]` fp32 | |
| 2c | 条件 Clone | `.clone()` (仅 `not is_processed_logprobs_mode`) | **GPU** | `[N, V]` fp32 | `[N, V]` fp32 | 保留 raw logits 用于 logprobs |
| 3a | _combine_outputs | `_combine_outputs_with_spec_tokens()` | **CPU** | `list[list[int]]` ×2 | `list[list[int]]` | 合并历史+draft tokens |
| 3b | repeat_indices | `repeat_interleave(num_draft_tokens)` | **CPU→GPU** | `[B]` int64 | `[N]` int64 | batch→token 索引映射 |
| 3c | apply_penalties | `apply_all_penalties(repeat_indices)` | **GPU** | `[N, V]` fp32 | `[N, V]` fp32 | repetition/freq/pres |
| 3d | masked_fill_ | `mask[repeat_indices] → masked_fill_(-inf)` | **GPU** | `[N, V]` fp32 | `[N, V]` fp32 | allowed_token_ids |
| 3e | bad_words | `apply_bad_words_with_drafts()` | **GPU** | 逐请求 | 同输入 | |
| 3f | non_argmax_invariant | `MinTokensLogitsProcessor.apply_with_spec_decode()` | **GPU** | `[N, V]` fp32 | `[N, V]` fp32 | spec_decode 专用接口 |
| 4a | **expand temperature** | **`expand_kernel`** | **GPU (Triton)** | `[B]` fp32 + `[B]` int32 | `[N]` fp32 | batch→token |
| 4b | temp div | `logits.div_(temp.unsqueeze(-1))` | **GPU** | `[N, V]` / `[N, 1]` | `[N, V]` fp32 | 原地 |
| 4c | **expand top_k** | **`expand_kernel`** | **GPU (Triton)** | `[B]` + `[B]` | `[N]` | 可选 |
| 4d | **expand top_p** | **`expand_kernel`** | **GPU (Triton)** | `[B]` + `[B]` | `[N]` | 可选 |
| 4e | top_k_top_p | `apply_top_k_top_p_triton` 或 `_pytorch` | **GPU (Triton/CUDA)** | `[N, V]` + `[N]` + `[N]` | `[N, V]` fp32 | batch≥8用Triton |
| 5a | 创建输出缓冲 | `torch.full + fill_` | **GPU** | - | `[B, S+1]` int32 | PLACEHOLDER=-1 |
| 5b-i | argmax | `target_logits.argmax(dim=-1)` | **GPU** | `[N, V]` fp32 | `[N]` int64 | 贪心路径 |
| 5b-ii | **贪心拒绝** | **`rejection_greedy_sample_kernel`** | **GPU (Triton)** | `[N]` + `[N]` + `[B]` | `[B, S+1]` int32 | 逐请求遍历 |
| 5c | softmax | `.softmax(dim=-1, fp32)` | **GPU** | `[N, V]` fp32 | `[N, V]` fp32 | 随机路径 |
| 5d | uniform_probs | `torch.rand([N], fp64)` | **GPU (CUDA)** | - | `[N]` fp64 | + per-gen 覆盖 |
| 5e-i | q + exponential | `torch.empty + exponential_()` | **GPU (CUDA)** | - | `[B, V]` fp32 | CUDA 原生高效 |
| 5e-ii | inv_q | `q.reciprocal()` | **GPU** | `[B, V]` fp32 | `[B, V]` fp32 | 预计算倒数 |
| 5e-iii | **恢复token采样** | **`sample_recovered_tokens_kernel`** | **GPU (Triton)** | `[N,V]` + `[B,V]` | `[N]` int32 | grid=(B,S), BLOCK=8192 |
| 5f | **随机拒绝** | **`rejection_random_sample_kernel`** | **GPU (Triton)** | 多输入 | `[B, S+1]` int32 | 逐token验证 |

> **图例**: B=batch_size, N=num_draft_tokens(展平总数), S=max_spec_len, V=vocab_size
>
> **特殊分支**:
> - `all_greedy=True`: 执行 5a→5b, 跳过 5c-5f
> - `all_random=True`: 跳过 5b, 直接走 5c→5f
> - `mixed`: 5b 和 5f 都执行, Triton kernel 通过 `is_greedy` 掩码区分

#### 4.2.2 数据流详细追踪

```
输入: logits [N+B, V] fp16/bf16  (来自 compute_logits)
      metadata.bonus_logits_indices [B], metadata.target_logits_indices [N]
 │
 ├─ 提取 Bonus Logits:
 │   logits[bonus_logits_indices] → bonus_logits [B, V]
 │   算子: IndexSelect          位置: GPU
 │   │
 │   └─ Sampler.forward(bonus_logits) → bonus_token_ids [B, 1] int32
 │       (完整传统采样流程: fp32→logits_processors→sample→logprobs)
 │
 ├─ 提取 Target Logits:
 │   logits[target_logits_indices] → raw_target_logits [N, V]
 │   算子: IndexSelect          位置: GPU
 │   │
 │   ├─ .to(fp32) → [N, V] fp32
 │   │   算子: Cast              位置: GPU
 │   │
 │   └─ .clone() → target_logits [N, V] fp32
 │       算子: Clone             位置: GPU
 │
 ├─ apply_logits_processors() → target_logits [N, V] fp32 (原地修改)
 │   ├─ allowed_token_ids_mask → masked_fill_(-inf)
 │   ├─ apply_bad_words_with_drafts()
 │   ├─ non_argmax_invariant 处理器 (MinTokensLogitsProcessor等)
 │   └─ apply_all_penalties (repetition/frequency/presence)
 │
 ├─ apply_sampling_constraints():
 │   │
 │   ├─[贪心快速路径] all_greedy=True: 直接返回
 │   │
 │   └─[需要约束处理]:
 │       ├─ expand_kernel (Triton): temperature[B] → [N]
 │       │   replace_from=0(GREEDY_TEMP) → replace_to=1
 │       │
 │       ├─ Div_: logits[N,V] / temperature[N,1] → logits[N,V] (原地)
 │       │
 │       ├─ expand_kernel (Triton): top_k[B] → [N]  (如有)
 │       ├─ expand_kernel (Triton): top_p[B] → [N]  (如有)
 │       │
 │       └─ apply_top_k_top_p(logits[N,V], top_k[N], top_p[N])
 │           ├─ batch≥8: Triton kernel (sort-free)
 │           └─ batch<8: PyTorch sort+mask
 │
 ├─ rejection_sample():
 │   │
 │   ├─ torch.full: output_token_ids [B, S+1] int32 = PLACEHOLDER(-1)
 │   │
 │   ├─[贪心路径] (not all_random):
 │   │   ├─ ArgMax: target_logits[N,V] → target_argmax[N] int64
 │   │   │
 │   │   └─ rejection_greedy_sample_kernel[(batch_size,)]:
 │   │       位置: GPU (Triton JIT)
 │   │       逐请求: 遍历 draft tokens
 │   │         load draft_token → load target_argmax → store target_argmax
 │   │         → 比较 → 不匹配时停止 → 全匹配追加 bonus_token
 │   │       输出: output_token_ids[B, S+1] 部分填充
 │   │
 │   │   └─ if all_greedy: return output_token_ids  ◀─ 快速返回
 │   │
 │   ├─ Softmax: target_logits[N,V] → target_probs[N,V] fp32
 │   │
 │   ├─ generate_uniform_probs: → uniform_probs[N] fp64
 │   │   算子: torch.rand(fp64)     位置: GPU (CUDA)
 │   │   + 逐请求 generator 覆盖
 │   │
 │   ├─ sample_recovered_tokens:
 │   │   ├─ q = Empty[B,V].exponential_()
 │   │   │   算子: Exponential       位置: GPU (CUDA, 高效)
 │   │   ├─ inv_q = q.reciprocal()
 │   │   │   算子: Reciprocal        位置: GPU
 │   │   └─ sample_recovered_tokens_kernel[(B, S)]:
 │   │       位置: GPU (Triton JIT)
 │   │       分块遍历 vocab (BLOCK=8192):
 │   │         N-gram: target_probs[draft_token]=0 → prob * inv_q → argmax
 │   │         有draft_probs: max(0, target-draft) * inv_q → argmax
 │   │       输出: recovered_token_ids[N] int32
 │   │
 │   └─[随机路径]:
 │       rejection_random_sample_kernel[(batch_size,)]:
 │         位置: GPU (Triton JIT)
 │         逐请求逐token:
 │           draft_prob>0 && target_prob/draft_prob >= uniform → accept
 │           else → reject, 使用 recovered_token
 │           全部 accept → 追加 bonus_token
 │         输出: output_token_ids[B, S+1] 完成填充
 │
 └─ 返回 SamplerOutput:
     ├─ sampled_token_ids: output_token_ids [B, S+1] int32
     └─ logprobs_tensors: LogprobsTensors | None
```

#### 4.2.3 运行位置分布统计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  GPU 投机推理后处理算子运行位置分布                             │
├──────────────┬──────────────────────────────────────────────────────────────┤
│  GPU-CUDA    │ ████████████████████████████████████████████████  ~75%       │
│  (PyTorch)   │ IndexSelect, Cast, Clone, MaskedFill, Div_, SoftmaxV2,     │
│              │ ArgMax, Fill_, Exponential_, Reciprocal, penalties系列        │
│              │ ✅ exponential_() 在 CUDA 上高效执行                          │
├──────────────┼──────────────────────────────────────────────────────────────┤
│  GPU-CUDA    │ ██████████████████  ~22%                                    │
│  (Triton)    │ expand_kernel ×3, apply_top_k_top_p_triton,                │
│              │ rejection_greedy_sample_kernel,                              │
│              │ sample_recovered_tokens_kernel,                              │
│              │ rejection_random_sample_kernel                               │
├──────────────┼──────────────────────────────────────────────────────────────┤
│  Host CPU    │ █  ~2%                                                      │
│              │ _get_cumsum_and_arange (numpy), bad_words处理,              │
│              │ H2D传输 (from_numpy → to(device))                           │
├──────────────┼──────────────────────────────────────────────────────────────┤
│  GPU-Tensor  │ ▏ ~1% (batch<8时 apply_top_k_top_p_pytorch sort路径)       │
│  Core        │ sort + mask + scatter                                       │
└──────────────┴──────────────────────────────────────────────────────────────┘
```

#### 4.2.4 Triton Kernel 详细列表

| Kernel 名称 | grid 配置 | 功能 | 核心逻辑 |
|------------|----------|------|---------|
| `expand_kernel` | `(batch_size,)` | batch→token 参数扩展 | 根据 cu_num_tokens 将 x[B] 复制到 expanded[N], 支持 replace_from→replace_to |
| `rejection_greedy_sample_kernel` | `(batch_size,)` | 贪心拒绝采样 | 逐请求: 遍历 draft tokens, 首次不匹配停止, 全匹配时追加 bonus |
| `sample_recovered_tokens_kernel` | `(B, max_spec_len)` | 恢复 token 采样 | 分块(BLOCK=8192)遍历vocab, max(0,target-draft)*inv_q → argmax |
| `rejection_random_sample_kernel` | `(batch_size,)` | 随机拒绝采样 | 逐token: target/draft >= uniform → accept/reject |
| `apply_top_k_top_p_triton` | (内部计算) | Top-K + Top-P 过滤 | Triton 实现的 sort+mask+filter (batch≥8时启用) |

> **与 Ascend Triton Kernel 对比**:
>
> | 特性 | GPU | Ascend |
> |------|-----|--------|
> | 贪心 kernel 数量 | 1 个通用 | 2 个 (spec_len=1 优化 + 通用) |
> | 随机 kernel 数量 | 1 个通用 | 2 个 (逐个验证 + block verify) |
> | block verify | ❌ | ✅ max_spec_len≥3 时使用 |
> | grid 配置 | `(batch_size,)` | `cal_grid_and_block_size()` 动态计算 |
> | recovered kernel BLOCK | 8192 | 4096 (SUB_BLOCK) |
> | inv_q 预计算 | ✅ `reciprocal()` | ❌ kernel 内除法 |

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
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ sample_tokens()                                                 │
│                                                                 │
│ 0. kv_connector_output 处理                                     │
│    ├── PP 非最后 rank: 提前返回 None / EMPTY_MODEL_RUNNER_OUTPUT │
│    └── PP+KV transfer: 封装 kv_connector_output 返回            │
│                                                                 │
│ 1. 解包 execute_model_state（10个字段）                          │
│    scheduler_output, logits, spec_decode_metadata,              │
│    spec_decode_common_attn_metadata, hidden_states,             │
│    sample_hidden_states, aux_hidden_states,                     │
│    ec_connector_output, cudagraph_stats, slot_mappings          │
│                                                                 │
│ 2. 应用 grammar bitmask（结构化输出）                            │
│                                                                 │
│ 3. _sample(logits, spec_decode_metadata)                        │
│    ├── update_async_output_token_ids()  [异步调度]              │
│    ├── update_async_spec_token_ids()    [异步调度+draft tokens]  │
│    └── RejectionSampler.forward()                               │
│        ├── bonus_logits → Sampler → bonus_token_ids [B, 1]     │
│        ├── target_logits → processors → constraints             │
│        └── rejection_sample() → output_token_ids [B, S+1]      │
│                                                                 │
│ 4. _update_states_after_model_execute()  [Hybrid模型状态更新]   │
│                                                                 │
│ 5. PP 异步广播 _pp_broadcast_prev_sampled_token_ids()           │
│                                                                 │
│ 6. 清理 _draft_token_ids, prev_sampled_token_ids = None        │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. Draft 提案策略分支                                           │
│ ┌───────────────────────────────────────────────────────────┐   │
│ │ 路径A: EAGLE/DraftModel + input fits                      │   │
│ │ → bookkeeping 前直接 propose_draft_token_ids(GPU tokens)  │   │
│ ├───────────────────────────────────────────────────────────┤   │
│ │ 路径B: EAGLE/DraftModel + input 不 fit                    │   │
│ │ → prepare_next_token_ids_padded() + zeros fallback        │   │
│ │ → _copy_draft_token_ids_to_cpu(zeros_only=True)           │   │
│ ├───────────────────────────────────────────────────────────┤   │
│ │ 路径C: Ngram/Suffix 等                                    │   │
│ │ → propose_drafts_after_bookkeeping = True (延迟提案)      │   │
│ └───────────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8. _bookkeeping_sync()                                          │
│ ├── NaN 检测: _get_nans_in_logits(logits)                       │
│ ├── 丢弃请求: generator offset 回退                             │
│ ├── 拷贝 req_ids（防异步修改）                                   │
│ ├── 同步路径: parse_output() → valid_sampled_token_ids          │
│ │   └── 过滤 PLACEHOLDER(-1) 和越界 token → list[list[int]]     │
│ ├── 异步路径: 缓存 GPU tokens，延迟拷贝                         │
│ ├── 更新 token_ids_cpu + num_tokens_no_spec + req_state         │
│ └── 计算 prompt_logprobs: _get_prompt_logprobs_dict()           │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ 9. 延迟 draft 提案（路径C: ngram等，用 CPU tokens）              │
│    propose_draft_token_ids(valid_sampled_token_ids)              │
│                                                                 │
│ 10. clear_kv_connector_metadata()                               │
│     延迟到 draft model 运行后再清理 KV 元数据                    │
│                                                                 │
│ 11. eplb_step()  Expert Load Balancing 步进                      │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│ 12. 构建 ModelRunnerOutput                                      │
│ ├── req_ids, req_id_to_index                                    │
│ ├── sampled_token_ids: list[list[int]]                          │
│ ├── logprobs, prompt_logprobs_dict                              │
│ ├── kv_connector_output, ec_connector_output                    │
│ ├── num_nans_in_logits, cudagraph_stats                         │
│ └── draft_token_ids: 下一轮的 draft tokens                      │
├─────────────────────────────────────────────────────────────────┤
│ 13. 异步调度返回路径                                             │
│ ├── 同步: 直接返回 ModelRunnerOutput                             │
│ └── 异步: AsyncGPUModelRunnerOutput                              │
│     ├── 封装 sampled_token_ids(GPU), logprobs_tensors           │
│     ├── invalid_req_indices, async_output_copy_stream           │
│     └── set_async_sampled_token_ids() 保存异步拷贝引用          │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 张量形状变化

```
hidden_states: [num_tokens, hidden_size]
      ↓ compute_logits()
logits: [num_tokens + batch_size, vocab_size]
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

---

## 6. GPU vs Ascend 投机推理后处理对比总结

### 6.1 SpecDecodeMetadata 构建差异

| 特性 | GPU | Ascend |
|------|-----|--------|
| 文件位置 | `gpu_model_runner.py:2209-2286` | `model_runner_v1.py:856-932` |
| cumsum+arange | `_get_cumsum_and_arange` 封装方法 | 手动内联计算 |
| PCP 支持 | ❌ | ✅ `num_pcp_pads` 参数修正 |
| H2D 传输 | `from_numpy().to(device)` | `from_numpy().pin_memory().to(device)` |

### 6.2 apply_sampling_constraints 差异

| 特性 | GPU | Ascend |
|------|-----|--------|
| temperature 扩展 | Triton `expand_kernel` | Triton `expand_kernel` (HAS_TRITON) |
| Top-K/Top-P | `apply_top_k_top_p_triton` (batch≥8) 或 PyTorch sort | `npu_apply_top_k_top_p` 昇腾原生算子 (A2/A3) |
| 回退路径 | PyTorch sort+mask+scatter | PyTorch sort 实现 |

### 6.3 rejection_sample 差异

| 特性 | GPU | Ascend |
|------|-----|--------|
| 贪心 Triton kernel | 1 个通用 kernel | 2 个 (spec_len=1 优化 + 通用 + bonus_renew) |
| 随机 Triton kernel | 1 个通用 kernel | 2 个 (逐个验证 + block verify) |
| block verify 优化 | ❌ | ✅ max_spec_len≥3 时累积乘积验证 |
| grid 配置 | `(batch_size,)` 固定 | `cal_grid_and_block_size()` 基于 vectorcore 数量动态计算 |
| exponential_() | CUDA 上高效执行 | AI-CPU 执行，性能瓶颈 |
| inv_q 预计算 | ✅ `reciprocal()` + kernel 内乘法 | ❌ kernel 内除法 |
| uniform_probs dtype | float64 | float32 (cast后) |
| BLOCK_SIZE | 8192 | 4096 (SUB_BLOCK) |
| PyTorch 回退路径 | ❌ 无 (Triton 必须可用) | ✅ 完整 PyTorch 实现 |

### 6.4 Bonus 采样差异

| 特性 | GPU | Ascend |
|------|-----|--------|
| 采样器 | `Sampler` (含 TopKTopPSampler) | `AscendSampler` (含异步流优化) |
| FlashInfer 支持 | ✅ (VLLM_USE_FLASHINFER_SAMPLER=1) | ❌ |
| 异步流优化 | ❌ | ✅ NPU 多流重叠 |

### 6.5 关键文件列表（GPU）

| 功能 | 文件路径 |
|------|----------|
| 模型运行器 | `vllm/v1/worker/gpu_model_runner.py` |
| 传统采样器 | `vllm/v1/sample/sampler.py` |
| 拒绝采样 | `vllm/v1/sample/rejection_sampler.py` |
| TopK/TopP 采样 | `vllm/v1/sample/ops/topk_topp_sampler.py` |
| TopK/TopP Triton | `vllm/v1/sample/ops/topk_topp_triton.py` |
| SamplingMetadata | `vllm/v1/sample/metadata.py` |
| SpecDecodeMetadata | `vllm/v1/spec_decode/metadata.py` |
| Eagle Proposer | `vllm/v1/spec_decode/eagle.py` |
| Medusa Proposer | `vllm/v1/spec_decode/medusa.py` |
| DraftModel Proposer | `vllm/v1/spec_decode/draft_model.py` |
| Ngram Proposer | `vllm/v1/spec_decode/ngram_proposer.py` |
