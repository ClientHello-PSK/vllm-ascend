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
    all_greedy: bool                      # 是否全部贪婪采样
    all_random: bool                      # 是否全部随机采样
    top_p: torch.Tensor | None            # top-p参数 [batch_size]
    top_k: torch.Tensor | None            # top-k参数 [batch_size]
    generators: dict[int, torch.Generator] # 随机数生成器
    max_num_logprobs: int | None          # 最大logprobs数量
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
| `temperature` | float | 1.0 | 控制采样随机性，0表示贪婪采样 |
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
    │       └── 获取 logits
    │
    ├── 2. 【可选】应用结构化输出约束
    │       └── apply_grammar_bitmask(logits)
    │
    ├── 3. 调用 _sample()
    │       │
    │       └── AscendSampler.forward() [sampler.py]
    │               │
    │               ├── 计算logprobs（如果需要）
    │               ├── 转换为float32
    │               ├── apply_logits_processors()
    │               └── sample()
    │
    ├── 4. _bookkeeping_sync()
    │       └── 解析采样结果，更新状态
    │
    └── 5. 返回 ModelRunnerOutput
```

### 3.3 Sampler.forward() 详细流程

**文件位置**: `vllm/v1/sample/sampler.py:67-129`

```python
def forward(self, logits, sampling_metadata, ...):
    # 1. 计算原始logprobs（如果需要）
    if num_logprobs is not None:
        raw_logprobs = self.compute_logprobs(logits)

    # 2. 转换为float32
    logits = logits.to(torch.float32)

    # 3. 应用logits处理器
    logits = self.apply_logits_processors(logits, sampling_metadata)

    # 4. 采样
    sampled, processed_logprobs = self.sample(logits, sampling_metadata)

    # 5. 收集logprobs
    logprobs_tensors = self.gather_logprobs(raw_logprobs, num_logprobs, sampled)

    # 6. 返回结果
    return SamplerOutput(
        sampled_token_ids=sampled.unsqueeze(-1),
        logprobs_tensors=logprobs_tensors,
    )
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
            ├── 贪婪采样: argmax
            └── 随机采样: temperature → top_k/top_p → multinomial
```

### 4.2 采样方法

**贪婪采样** (`sampler.py:144-145`):
```python
def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1).view(-1)
```

**随机采样** (`sampler.py:147-203`):
```python
def sample(self, logits, sampling_metadata):
    # 1. 贪婪采样（如果需要）
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
    │               │       ├── 贪婪采样路径
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

**文件位置**: `vllm/v1/sample/rejection_sampler.py:60-166`

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
    ├── 贪婪采样路径:
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

### 4.2 贪婪采样拒绝采样

**文件位置**: `vllm_ascend/sample/rejection_sampler.py:637-799`

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

**文件位置**: `vllm_ascend/sample/rejection_sampler.py:801-992`

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

**文件位置**: `vllm_ascend/sample/rejection_sampler.py:1227-1380`

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

    # 贪婪采样直接返回
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
| `GREEDY_TEMPERATURE` | 0 | 贪婪采样温度 |
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
