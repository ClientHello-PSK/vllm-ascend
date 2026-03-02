# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.triton_utils import HAS_TRITON, triton
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (
    GREEDY_TEMPERATURE,
    MAX_SPEC_LEN,
    PLACEHOLDER_TOKEN_ID,
    generate_uniform_probs,
)

from vllm_ascend.ops.triton.reject_sample import (
    cal_grid_and_block_size,
    expand_triton,
    rejection_greedy_sample_with_triton,
    rejection_random_sample_block_verify_kernel,
    rejection_random_sample_kernel,
    sample_recovered_tokens_kernel,
)
from vllm_ascend.sample.sampler import apply_top_k_top_p


def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size] 每个token的logits向量
    cu_num_draft_tokens: torch.Tensor,  # [batch_size] 累积draft token数量
    sampling_metadata: SamplingMetadata,  # 采样元数据，包含temperature、top_k、top_p等
) -> torch.Tensor:
    """
    根据采样元数据对logits进行处理（温度缩放、top-k、top-p）。

    核心功能：
    =========
    1. 温度缩放 (Temperature Scaling): logits / temperature
       - temperature > 1: 使分布更平滑，增加多样性
       - temperature < 1: 使分布更尖锐，更确定性
       - temperature = 1: 不改变分布

    2. Top-k 过滤: 只保留概率最高的k个token
    3. Top-p (Nucleus) 过滤: 只保留累积概率达到p的最小token集合

    特殊情况：
    =========
    - 贪婪采样 (greedy decoding): 直接返回原始logits，不做任何处理
      因为贪婪采样只需要argmax，不需要概率分布

    举例说明：
    =========
    假设 batch_size=2, num_tokens=5, vocab_size=100

    请求0: 3个draft tokens, temperature=0.8, top_k=50, top_p=0.9
    请求1: 2个draft tokens, temperature=1.0 (贪婪)

    处理流程：
    1. 检查all_greedy -> 不是，继续处理
    2. 扩展temperature: [0.8, 0.8, 0.8, 1.0, 1.0]
    3. 温度缩放: logits[i] /= temperature[i]
    4. 扩展top_k: [50, 50, 50, None, None]
    5. 扩展top_p: [0.9, 0.9, 0.9, None, None]
    6. 应用top-k和top-p过滤

    返回：
    =====
    处理后的logits，可用于后续softmax计算概率
    """
    # ==================== 步骤1: 参数校验 ====================
    assert logits.ndim == 2  # 必须是2D张量 [num_tokens, vocab_size]
    assert cu_num_draft_tokens.ndim == 1  # 必须是1D张量 [batch_size]

    # ==================== 步骤2: 贪婪采样快速路径 ====================
    # 如果所有请求都是贪婪采样，直接返回原始logits
    # 贪婪采样只需要argmax，不需要温度缩放和top-k/top-p
    if sampling_metadata.all_greedy:
        return logits

    # ==================== 步骤3: 温度缩放 ====================
    num_tokens = logits.shape[0]

    # 将batch级别的temperature扩展到token级别
    # replace_from=GREEDY_TEMPERATURE, replace_to=1 表示：
    #   如果temperature是贪婪温度（通常是0），替换为1（即不缩放）
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )

    # 原地除法，避免分配新张量，节省内存
    # logits.div_(temperature.unsqueeze(-1)) 等价于 logits /= temperature[:, None]
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # ==================== 步骤4: Top-k 和 Top-p 扩展 ====================
    # 将batch级别的top_k和top_p扩展到token级别
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )

    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )

    # ==================== 步骤5: 应用 Top-k 和 Top-p 过滤 ====================
    # apply_top_k_top_p 会将不符合条件的logits设为负无穷
    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    return apply_top_k_top_p(logits, top_k, top_p)


def rejection_sample(
    # [num_tokens] draft模型生成的所有token ids（展平为一维）
    draft_token_ids: torch.Tensor,
    # [batch_size] 每个请求的draft token数量列表
    num_draft_tokens: list[int],
    # 最大投机长度（draft模型最多生成的token数量）
    max_spec_len: int,
    # [batch_size] 累积draft token数量（前缀和）
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size] draft模型的概率分布，N-gram模式时为None
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size] target模型的logits
    target_logits: torch.Tensor,
    # [batch_size, 1] 每个请求的bonus token（当所有draft tokens都被接受时使用）
    bonus_token_ids: torch.Tensor,
    # 采样元数据，包含temperature、随机生成器等
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """
    投机解码的核心拒绝采样函数。

    这是推测解码(Speculative Decoding)的主入口函数，负责：
    1. 验证draft模型生成的tokens
    2. 决定接受或拒绝每个draft token
    3. 生成最终输出的token序列

    算法概述：
    =========
    投机解码的核心思想是用一个小而快的draft模型生成候选tokens，
    然后用大而慢的target模型并行验证这些tokens。

    对于贪婪采样：
    - 比较draft token和target token是否相同
    - 从第一个不匹配的位置开始拒绝

    对于随机采样：
    - 使用接受概率: accept_prob = min(1, p_target / p_draft)
    - 生成均匀随机数 u，如果 u < accept_prob 则接受
    - 拒绝时从修正分布中采样恢复token

    两种验证模式：
    =========
    1. 逐个验证 (max_spec_len < 3): 传统方式，逐token验证
    2. 块验证 (max_spec_len >= 3): MagicMTP方式，使用累积乘积提高接受率

    举例说明：
    =========
    假设 batch_size=2, max_spec_len=4

    请求0: draft生成3个tokens [10, 20, 30], temperature=0 (贪婪)
    请求1: draft生成2个tokens [40, 50], temperature=0.8 (随机)

    draft_token_ids: [10, 20, 30, 40, 50]
    num_draft_tokens: [3, 2]
    cu_num_draft_tokens: [3, 5]

    处理流程：
    1. 创建输出缓冲区 [2, 5]，填充占位符
    2. 请求0使用贪婪采样路径
    3. 请求1使用随机采样路径
    4. 返回output_token_ids

    返回：
    =====
    output_token_ids: [batch_size, max_spec_len + 1]
    - 每行是一个请求的输出tokens
    - 被接受的位置填入draft token（或target token）
    - 被拒绝的位置填入恢复token
    - 最后一个位置可能填入bonus token
    """

    # ==================== 步骤1: 参数校验 ====================
    assert draft_token_ids.ndim == 1  # 必须是1D张量
    assert draft_probs is None or draft_probs.ndim == 2  # None或2D张量
    assert cu_num_draft_tokens.ndim == 1  # 必须是1D张量
    assert target_logits.ndim == 2  # 必须是2D张量

    # 获取基本信息
    batch_size = len(num_draft_tokens)  # 批次大小
    num_tokens = draft_token_ids.shape[0]  # 总token数
    vocab_size = target_logits.shape[-1]  # 词表大小
    device = target_logits.device  # 计算设备

    # 确保所有张量是连续的，这对GPU性能很重要
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_logits.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)

    # ==================== 步骤2: 确定验证模式 ====================
    # 当 max_spec_len >= 3 时，使用块验证(Block Verify)模式
    # 块验证使用累积乘积来提高接受率（MagicMTP技术）
    using_block_verify = max_spec_len >= 3

    # ==================== 步骤3: 创建输出缓冲区 ====================
    # 输出形状为 [batch_size, max_spec_len + 1]
    # +1 是为了存储可能需要追加的bonus token
    output_token_ids = torch.empty(
        (batch_size, max_spec_len + 1),
        dtype=torch.int32,  # 使用int32与SamplerOutput保持一致
        device=device,
    )
    # 用占位符填充，未被填充的位置保持为占位符
    output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

    # ==================== 步骤4: 确定采样类型 ====================
    # is_greedy[i] = True 表示请求i使用贪婪采样
    if sampling_metadata.all_greedy:
        # 所有请求都是贪婪采样
        is_greedy = None
    else:
        # 根据temperature判断：GREEDY_TEMPERATURE(通常是0)表示贪婪
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE

    # 如果有Triton支持，计算grid和block大小
    if HAS_TRITON:
        grid, block_size = cal_grid_and_block_size(batch_size)

    # ==================== 步骤5: 贪婪采样拒绝采样 ====================
    # 如果不是所有请求都是随机采样，则处理贪婪采样的请求
    if not sampling_metadata.all_random:
        # 计算target模型对每个位置的argmax（贪婪选择）
        target_argmax = target_logits.argmax(dim=-1)

        if HAS_TRITON:
            # 使用Triton GPU kernel加速
            rejection_greedy_sample_with_triton(
                output_token_ids,
                num_draft_tokens,
                cu_num_draft_tokens,
                draft_token_ids,
                target_argmax,
                bonus_token_ids,
                is_greedy,
                max_spec_len,
                grid,
                block_size,
            )
        else:
            # 使用PyTorch实现
            if min(num_draft_tokens) == 1 and max(num_draft_tokens) == 1 and sampling_metadata.all_greedy:
                # 特殊优化：所有请求都只有1个draft token且都是贪婪采样
                rejection_greedy_sample_spec_len_1_pytorch(
                    output_token_ids,
                    draft_token_ids,
                    target_argmax,
                    bonus_token_ids,
                )
            else:
                # 通用贪婪采样拒绝采样
                rejection_greedy_sample_pytorch(
                    output_token_ids,
                    cu_num_draft_tokens,
                    draft_token_ids,
                    target_argmax,
                    bonus_token_ids,
                    num_draft_tokens,
                    max_spec_len,
                    is_greedy,
                )

        # 如果所有请求都是贪婪采样，直接返回结果
        if sampling_metadata.all_greedy:
            return output_token_ids

    # ==================== 步骤6: 计算target概率分布 ====================
    # 从logits计算softmax概率分布
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    assert target_probs.is_contiguous()

    # ==================== 步骤7: 生成均匀随机数 ====================
    # 用于随机采样的拒绝判断
    # uniform_probs[i] ~ Uniform(0, 1)
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # ==================== 步骤8: 预计算恢复tokens ====================
    # 当draft token被拒绝时，需要从修正分布中采样恢复token
    # 预先为每个位置计算一个恢复token
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # ==================== 步骤9: 随机采样拒绝采样 ====================
    if not using_block_verify:
        # 传统逐个验证模式
        if HAS_TRITON:
            rejection_random_sample_kernel[(grid,)](
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                bonus_token_ids,
                recovered_token_ids,
                uniform_probs.to(torch.float32),
                is_greedy,
                max_spec_len,
                vocab_size,
                batch_size,
                NO_DRAFT_PROBS=draft_probs is None,
                BLOCK_SIZE=block_size,
            )
        else:
            rejection_random_sample_pytorch(
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                bonus_token_ids,
                recovered_token_ids,
                uniform_probs,
                is_greedy,
                max_spec_len,
                vocab_size,
                IS_NGRAM=draft_probs is None,
            )
    else:
        # MagicMTP块验证模式：使用累积乘积提高接受率
        if HAS_TRITON:
            rejection_random_sample_block_verify_kernel[(grid,)](
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                bonus_token_ids,
                recovered_token_ids,
                uniform_probs.to(torch.float32),
                is_greedy,
                max_spec_len,
                vocab_size,
                batch_size,
                NO_DRAFT_PROBS=draft_probs is None,
                BLOCK_SIZE=block_size,
            )
        else:
            rejection_random_sample_block_verify_pytorch(
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                bonus_token_ids,
                recovered_token_ids,
                uniform_probs,
                is_greedy,
                max_spec_len,
                vocab_size,
                IS_NGRAM=draft_probs is None,
            )

    return output_token_ids


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size] 需要扩展的batch级别张量
    cu_num_tokens: torch.Tensor,  # [batch_size] 累积token数量（前缀和）
    num_tokens: int,  # 总token数量
    replace_from: int = 0,  # 需要被替换的值
    replace_to: int = 0,  # 替换后的值
) -> torch.Tensor:
    """
    将batch级别的张量扩展为token级别的张量。

    核心功能：
    =========
    根据 cu_num_tokens 中的累积token计数，将 [batch_size] 的张量
    扩展为 [num_tokens] 的张量。类似于 repeat_interleave 操作。

    举例说明：
    =========
    假设 batch_size=3, num_tokens=6

    x = [10, 20, 30]  (每个batch的值)
    cu_num_tokens = [2, 5, 6]  (累积token数量)

    计算：
    - batch 0: 2个tokens，值为10
    - batch 1: 3个tokens (5-2)，值为20
    - batch 2: 1个token (6-5)，值为30

    扩展结果: expanded_x = [10, 10, 20, 20, 20, 30]

    参数说明：
    =========
    replace_from 和 replace_to 用于条件替换：
    - 如果 x[i] == replace_from，则替换为 replace_to
    - 例如在temperature扩展中，将GREEDY_TEMPERATURE(0)替换为1

    Args:
        x: [batch_size] 需要扩展的张量
        cu_num_tokens: [batch_size] 累积token数量，每个元素表示到该batch为止的token总数
        num_tokens: 总token数量
        replace_from: 需要被替换的值
        replace_to: 替换后的值

    Returns:
        expanded_x: [num_tokens] 扩展后的张量
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size

    # 创建输出张量
    expanded_x = x.new_empty(num_tokens)

    if HAS_TRITON:
        # 使用Triton GPU kernel加速
        expand_triton(batch_size, expanded_x, x, cu_num_tokens, replace_from, replace_to, max_num_tokens=MAX_SPEC_LEN)
    else:
        # 使用PyTorch实现（fallback）
        expand_pytorch(
            expanded_x,
            x,
            cu_num_tokens,
            replace_from,
            replace_to,
            MAX_NUM_TOKENS=MAX_SPEC_LEN,  # 固定大小避免重编译
        )

    return expanded_x


def sample_recovered_tokens(
    max_spec_len: int,  # 最大投机长度（draft模型最多生成的token数量）
    num_draft_tokens: list[int],  # 每个请求的draft token数量列表
    cu_num_draft_tokens: torch.Tensor,  # [batch_size] 累积draft token数量（前缀和）
    draft_token_ids: torch.Tensor,  # [num_tokens] draft模型生成的所有token ids（展平）
    draft_probs: torch.Tensor | None,  # [num_tokens, vocab_size] draft模型的概率分布，N-gram模式时为None
    target_probs: torch.Tensor,  # [num_tokens, vocab_size] target模型的概率分布
    sampling_metadata: SamplingMetadata,  # 采样元数据，包含每个请求的随机数生成器
    device: torch.device,  # 计算设备（CPU/GPU）
) -> torch.Tensor:
    """
    推测解码中的恢复token采样函数。

    核心目的：
    =========
    当draft token被拒绝时，需要从修正后的分布中采样一个"恢复token"。
    这个修正分布是基于 target_probs - draft_probs 计算的（取正部分），
    确保恢复的token既符合target模型的分布，又不会重复draft已经生成过的token。

    数学原理：
    =========
    恢复分布: P_recover(x) = max(0, p_target(x) - p_draft(x)) / Z
    使用Gumbel-max技巧的变体进行采样:
    1. 生成指数分布随机数 q
    2. 计算 prob / q
    3. 取 argmax 作为采样结果

    这等价于从修正后的分布中进行加权随机采样，但可以高效地向量化实现。

    举例说明：
    =========
    假设 batch_size=2, vocab_size=5

    请求0: draft生成2个tokens [10, 20]
    请求1: draft生成1个token  [30]

    draft_token_ids: [10, 20, 30] (展平)
    num_draft_tokens: [2, 1]
    cu_num_draft_tokens: [2, 3]

    对于每个token位置，计算修正概率后采样恢复token：
    - 位置0 (请求0): 从 max(0, target_probs[0] - draft_probs[0]) 采样
    - 位置1 (请求0): 从 max(0, target_probs[1] - draft_probs[1]) 采样
    - 位置2 (请求1): 从 max(0, target_probs[2] - draft_probs[2]) 采样

    返回:
    =====
    recovered_token_ids: [num_tokens] 每个位置对应的恢复token id
    """

    # ==================== 步骤1: 获取基本信息 ====================
    batch_size = len(num_draft_tokens)  # 批次大小
    vocab_size = target_probs.shape[-1]  # 词表大小

    # ==================== 步骤2: 初始化随机数张量 q ====================
    # q 用于Gumbel-max采样技巧，形状为 [batch_size, vocab_size]
    # 每个请求有独立的随机数序列，用于从修正分布中采样
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    # 使用指数分布填充 q（Gumbel-max技巧的一部分）
    # 指数分布的采样: q ~ Exp(1)
    q.exponential_()

    # ==================== 步骤3: 确定哪些请求有draft tokens ====================
    # 将 num_draft_tokens 从 list 转为 tensor
    # pin_memory=True 用于优化 CPU->GPU 数据传输
    num_draft_tensor = torch.tensor(num_draft_tokens, pin_memory=True).to(device, non_blocking=True)
    # has_draft_mask[i] = True 表示请求 i 有至少一个 draft token
    has_draft_mask = num_draft_tensor > 0

    # ==================== 步骤4: 使用请求级随机生成器重新生成 q ====================
    # 为了保证采样的可复现性，每个请求使用自己独立的随机数生成器
    # sampling_metadata.generators 是一个字典: {请求索引: 随机数生成器}
    for i, generator in sampling_metadata.generators.items():
        # 为当前请求创建临时张量
        temp_q = torch.empty_like(q[i])
        # 使用请求专属的随机生成器填充指数分布
        temp_q.exponential_(generator=generator)
        # 只有当该请求有 draft tokens 时才替换，否则保留默认值
        q[i] = torch.where(has_draft_mask[i], temp_q, q[i])

    # ==================== 步骤5: 调用核心kernel计算恢复token ====================
    # 创建输出张量，用于存储每个位置的恢复token
    recovered_token_ids = torch.empty_like(draft_token_ids)

    if HAS_TRITON:
        # 使用 Triton GPU kernel 加速计算
        # grid大小为 (batch_size, max_spec_len)，每个位置并行处理
        sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
            recovered_token_ids,  # 输出: 恢复token ids
            cu_num_draft_tokens,  # 累积draft token数量，用于索引映射
            draft_token_ids,  # draft token ids，用于N-gram模式时置零
            draft_probs,  # draft概率分布，用于计算修正概率
            target_probs,  # target概率分布
            q,  # 随机数张量，用于采样
            vocab_size,  # 词表大小
            triton.next_power_of_2(vocab_size),  # 词表大小的下一个2的幂（用于优化）
            NO_DRAFT_PROBS=draft_probs is None,  # 是否为N-gram模式（无draft概率）
            SUB_BLOCK=4 * 1024,  # 子块大小（用于分块优化）
            # TODO: enable multibuffer when accuracy problem is solved.
            multibuffer=False,  # 多缓冲优化（目前禁用以保证精度）
        )
    else:
        # 使用 PyTorch 实现（fallback方案）
        sample_recovered_tokens_pytorch(
            recovered_token_ids,  # 输出: 恢复token ids
            cu_num_draft_tokens,  # 累积draft token数量
            draft_token_ids,  # draft token ids
            draft_probs,  # draft概率分布
            target_probs,  # target概率分布
            q,  # 随机数张量
            vocab_size,  # 词表大小
            IS_NGRAM=draft_probs is None,  # 是否为N-gram模式
        )

    return recovered_token_ids


def rejection_greedy_sample_spec_len_1_pytorch(
    output_token_ids,  # [batch_size, 2] 输出token矩阵，每行2列（draft + bonus）
    draft_token_ids,  # [num_tokens] draft模型生成的token ids
    target_argmax,  # [num_tokens] target模型的argmax结果
    bonus_token_ids,  # [batch_size] bonus token ids
):
    """
    贪婪采样拒绝采样的特殊优化版本（spec_len=1）。

    核心目的：
    =========
    这是一个针对 max_spec_len=1 情况的优化实现。
    当每个请求只有1个draft token时，可以大幅简化逻辑。

    优化点：
    =======
    1. 不需要计算起始索引和位置映射（每个请求只有1个token）
    2. 不需要创建位置矩阵和不匹配矩阵
    3. 直接比较draft和target即可

    举例说明：
    =========
    假设 batch_size=3, max_spec_len=1

    draft_token_ids: [10, 20, 30]
    target_argmax:   [10, 25, 30]
                      ↑匹配  ↑不匹配  ↑匹配

    处理结果：
    - 请求0: draft=10, target=10 → 匹配，接受draft，bonus放到位置1
      output: [10, bonus0]
    - 请求1: draft=20, target=25 → 不匹配，接受target
      output: [25, -1]
    - 请求2: draft=30, target=30 → 匹配，接受draft，bonus放到位置1
      output: [30, bonus2]
    """
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)

    # 特殊优化场景：batch_size == num_tokens（每个请求只有1个draft token）
    assert batch_size == num_tokens

    # 判断每个请求的draft token是否与target匹配
    # accept_req_mask[i] = True 表示请求i的draft token被接受
    accept_req_mask = draft_token_ids == target_argmax

    # 位置0总是填入target的argmax结果
    # 如果匹配：填入draft token（与target相同）
    # 如果不匹配：填入target token（相当于拒绝draft）
    output_token_ids[:, 0] = target_argmax

    # 去掉bonus_token_ids的最后一维
    bonus_token_ids = bonus_token_ids.squeeze(1)

    # 位置1：只有在draft被接受时才填入bonus token
    # - 如果accept_req_mask[i]=True：填入bonus_token_ids[i]
    # - 如果accept_req_mask[i]=False：保持原值（占位符）
    output_token_ids[:, 1] = torch.where(accept_req_mask, bonus_token_ids, output_token_ids[:, 1])


def rejection_greedy_sample_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1] 输出token矩阵，每行存储一个请求的输出tokens
    cu_num_draft_tokens,  # [batch_size] 累积draft token数量，前缀和形式
    draft_token_ids,  # [num_tokens] draft模型生成的所有token ids（展平为一维）
    target_argmax,  # [num_tokens] target模型对每个位置的argmax结果（展平为一维）
    bonus_token_ids,  # [batch_size] 每个请求的bonus token（当所有draft tokens都被接受时使用）
    draft_tokens_per_req,  # [batch_size], list 每个请求的draft token数量
    max_spec_len,  # 最大投机长度（即draft模型最多生成多少个token）
    is_greedy=None,  # [batch_size] or None 标记每个请求是否使用贪婪采样
):
    """
    投机解码(Speculative Decoding)中的贪婪采样拒绝采样函数。

    核心思想：
    - Draft模型快速生成多个候选tokens
    - Target模型验证这些tokens是否与自己生成的相同
    - 从第一个不匹配的位置开始，后续draft tokens被拒绝

    举例说明：
    ==========
    假设 batch_size=2, max_spec_len=4

    请求0: draft生成3个tokens  [10, 20, 30]
    请求1: draft生成2个tokens  [40, 50]

    draft_token_ids (展平): [10, 20, 30, 40, 50]
    draft_tokens_per_req: [3, 2]
    cu_num_draft_tokens: [3, 5]  (累积和)

    target_argmax (展平): [10, 20, 35, 40, 55]
                          ↑匹配  ↑匹配  ↑不匹配  ↑匹配  ↑不匹配

    处理过程：
    - 请求0: 第2个位置不匹配(30 vs 35)，所以接受[10,20]，拒绝[30]，bonus token放到位置3
    - 请求1: 第1个位置不匹配(50 vs 55)，所以接受[40]，拒绝[50]，bonus token放到位置2

    最终 output_token_ids:
    请求0: [10, 20, bonus0, -1]  (位置0,1接受，位置2放bonus)
    请求1: [40, bonus1, -1, -1]  (位置0接受，位置1放bonus)
    """

    # ==================== 步骤1: 初始化基本信息 ====================
    batch_size = output_token_ids.size(0)  # 批次大小
    num_tokens = draft_token_ids.size(0)  # 总token数（所有请求的draft tokens之和）
    device = output_token_ids.device  # 设备信息

    # 将draft_tokens_per_req从list转为tensor，异步传输到GPU以提高效率
    draft_tokens_per_req = torch.tensor(draft_tokens_per_req).to(device, non_blocking=True)

    # 如果is_greedy为None，说明所有请求都是贪婪采样，创建全True的mask
    if is_greedy is None:
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)

    # ==================== 步骤2: 计算索引映射 ====================
    # 计算每个请求在draft_token_ids中的起始索引
    # 例如: cu_num_draft_tokens=[3,5], draft_tokens_per_req=[3,2]
    #       start_indices = [3-3, 5-2] = [0, 3]
    start_indices = cu_num_draft_tokens - draft_tokens_per_req

    # 创建请求ID序列 [0, 1, 2, ..., batch_size-1]
    req_ids = torch.arange(batch_size, device=device)

    # 将请求ID扩展到每个token，表示每个token属于哪个请求
    # 例如: req_ids=[0,1], draft_tokens_per_req=[3,2]
    #       token_req_ids = [0,0,0,1,1] (前3个token属于请求0，后2个属于请求1)
    token_req_ids = torch.repeat_interleave(req_ids, draft_tokens_per_req)

    # 计算每个token在其所属请求中的位置（从0开始）
    # token_positions[i] = token i 在其请求中的相对位置
    # 例如: token_req_ids=[0,0,0,1,1], start_indices=[0,3]
    #       token_positions = [0-0, 1-0, 2-0, 3-3, 4-3] = [0,1,2,0,1]
    token_positions = torch.arange(num_tokens, device=device) - start_indices[token_req_ids]

    # ==================== 步骤3: 找到每个请求的第一个不匹配位置 ====================
    # 比较draft token和target token是否相同
    # mismatch_global[i] = True 表示第i个token不匹配
    mismatch_global = draft_token_ids != target_argmax

    if max_spec_len == 0:
        # 边界情况：没有投机tokens
        first_mismatch_pos_per_req = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        # 创建位置矩阵 [batch_size, max_spec_len]，初始填充-1
        # 用于记录每个请求在每个位置的实际位置值
        pos_matrix = torch.full((batch_size, max_spec_len), -1, dtype=torch.long, device=device)

        # 将每个token的位置填入对应的位置矩阵
        # 例如: token_req_ids=[0,0,0,1,1], token_positions=[0,1,2,0,1]
        #       pos_matrix[0, 0]=0, pos_matrix[0, 1]=1, pos_matrix[0, 2]=2
        #       pos_matrix[1, 0]=0, pos_matrix[1, 1]=1
        pos_matrix[token_req_ids, token_positions] = token_positions

        # 创建不匹配矩阵 [batch_size, max_spec_len]，初始全False
        mismatch_matrix = torch.full((batch_size, max_spec_len), False, dtype=torch.bool, device=device)

        # 将不匹配信息填入矩阵
        # mismatch_matrix[i, j] = True 表示请求i在位置j处不匹配
        mismatch_matrix[token_req_ids, token_positions] = mismatch_global

        # 对于不匹配的位置保留位置值，对于匹配的位置设为一个很大的值(max_spec_len * 2)
        # 这样在求min时，匹配位置会被忽略
        mismatch_positions = torch.where(mismatch_matrix, pos_matrix, max_spec_len * 2)

        # 对每行求最小值，得到每个请求的第一个不匹配位置
        first_mismatch_pos_per_req, _ = torch.min(mismatch_positions, dim=1)

        # 如果所有位置都匹配，first_mismatch_pos会是max_spec_len*2
        # 此时应该设为draft_tokens_per_req（表示没有不匹配，所有draft tokens都被接受）
        no_mismatch_mask = first_mismatch_pos_per_req == max_spec_len * 2
        first_mismatch_pos_per_req[no_mismatch_mask] = draft_tokens_per_req[no_mismatch_mask]

    # ==================== 步骤4: 复制匹配的tokens到输出 ====================
    # 计算每个请求需要复制的长度
    # copy_len = min(第一个不匹配位置+1, draft_tokens数量)
    # +1是因为第一个不匹配位置本身也要被替换为target的token
    copy_len = torch.minimum(first_mismatch_pos_per_req + 1, draft_tokens_per_req)

    # 创建复制索引矩阵 [batch_size, max_spec_len+1]
    # copy_indices[i, j] = j (每个位置的列索引)
    copy_indices = torch.arange(max_spec_len + 1, device=device).expand(batch_size, -1)

    # 创建复制mask，标记哪些位置需要复制
    # copy_mask[i, j] = True 表示请求i的位置j需要复制
    copy_mask = copy_indices < copy_len.unsqueeze(1)

    # 扩展greedy mask以便与copy_mask进行广播
    greedy_mask = is_greedy.unsqueeze(1)

    # 最终复制mask：需要复制且是贪婪采样的请求
    final_copy_mask = copy_mask & greedy_mask

    # 计算每个输出位置对应的draft_token_ids中的全局索引
    # global_idx[i, j] = start_indices[i] + j
    global_idx = start_indices.unsqueeze(1) + copy_indices

    # 执行复制：将target_argmax中的值复制到output_token_ids
    # 注意：复制的是target的结果，而不是draft的结果
    output_token_ids[final_copy_mask] = target_argmax[global_idx[final_copy_mask]].to(output_token_ids.dtype)

    # ==================== 步骤5: 填充bonus token ====================
    # bonus token的条件：
    # 1. 是贪婪采样 (is_greedy)
    # 2. 所有draft tokens都被接受 (第一个不匹配位置 >= draft_tokens数量)
    needs_bonus = is_greedy & (first_mismatch_pos_per_req >= draft_tokens_per_req)

    if torch.any(needs_bonus):
        # 找出需要bonus token的请求索引
        bonus_rows = torch.where(needs_bonus)[0]
        # 获取每个需要bonus的请求的draft tokens数量，这决定了bonus token放在哪一列
        bonus_cols = draft_tokens_per_req[bonus_rows]
        # 去掉bonus_token_ids的最后一维（squeeze）
        bonus_token_ids = bonus_token_ids.squeeze(1)
        # 将bonus token填入对应位置
        output_token_ids[bonus_rows, bonus_cols] = bonus_token_ids[bonus_rows]


def rejection_random_sample_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1] 输出token矩阵
    cu_num_draft_tokens,  # [batch_size] 累积draft token数量
    draft_token_ids,  # [num_tokens] draft模型生成的token ids（展平）
    draft_probs,  # [num_tokens, vocab_size] draft概率分布，N-gram模式时为None
    target_probs,  # [num_tokens, vocab_size] target概率分布
    bonus_token_ids,  # [batch_size] bonus token ids
    recovered_token_ids,  # [num_tokens] 每个位置的恢复token
    uniform_probs,  # [num_tokens] 均匀随机数，用于拒绝判断
    is_greedy,  # [batch_size] 是否贪婪采样的标记
    max_spec_len,  # 最大投机长度
    vocab_size,  # 词表大小
    IS_NGRAM=False,  # 是否为N-gram模式（无draft概率）
):
    """
    投机解码的随机采样拒绝采样函数（PyTorch实现）。

    核心功能：
    =========
    实现推测解码的拒绝采样步骤，使用完全向量化的方法，
    避免了逐请求逐token循环的高开销。

    算法步骤：
    =========
    1. **索引映射**: 将展平的1D token数组转换为2D [batch_size, max_draft_len] 网格
    2. **并行验证**: 对所有draft token并行计算接受条件
       accept_prob = target_prob / draft_prob
       如果 uniform_sample <= accept_prob 则接受
    3. **短路模拟**: 找到第一个被拒绝的位置，后续位置都需要跳过
    4. **Token选择**: 使用torch.where选择最终输出
       - 接受的位置：draft token
       - 第一个拒绝的位置：recovered token
       - 全部接受：bonus token
    5. **掩码处理**: 确保只对非贪婪请求和有效序列长度进行操作

    举例说明：
    =========
    假设 batch_size=2, max_spec_len=3

    请求0: draft生成2个tokens [10, 20], temperature=0.8 (随机)
    请求1: draft生成3个tokens [30, 40, 50], temperature=0.8 (随机)

    接受概率计算：
    - 请求0位置0: p_target/p_draft = 0.9, uniform=0.5 → 接受
    - 请求0位置1: p_target/p_draft = 0.7, uniform=0.8 → 拒绝
    - 请求1位置0: p_target/p_draft = 0.95, uniform=0.3 → 接受
    - 请求1位置1: p_target/p_draft = 0.85, uniform=0.4 → 接受
    - 请求1位置2: p_target/p_draft = 0.9, uniform=0.2 → 接受

    结果：
    - 请求0: [10, recovered0, -1] (位置0接受，位置1拒绝)
    - 请求1: [30, 40, 50, bonus1] (全部接受，追加bonus)
    """

    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    # ==================== 步骤1: 计算每个请求的起始和结束索引 ====================
    # cu_start[i] = 请求i在展平数组中的起始索引
    # cu_end[i] = 请求i在展平数组中的结束索引
    zero_cpu = torch.tensor([0], pin_memory=True)
    zero_device = zero_cpu.to(device, non_blocking=True)

    cu_start = torch.cat([zero_device, cu_num_draft_tokens[:-1]])
    cu_end = cu_num_draft_tokens
    num_draft_per_batch = cu_end - cu_start  # 每个请求的draft token数量

    # ==================== 步骤2: 创建位置索引网格 ====================
    # pos_indices: [1, max_draft_len] = [[0, 1, 2, ..., max_draft_len-1]]
    max_draft_len = max_spec_len
    pos_indices_cpu = torch.arange(max_draft_len, pin_memory=True)
    pos_indices = pos_indices_cpu.to(device, non_blocking=True)[None, :]

    # valid_mask: [batch_size, max_draft_len]
    # valid_mask[i, j] = True 表示请求i在位置j有有效的draft token
    valid_mask = pos_indices < num_draft_per_batch[:, None]

    # global_token_indices: [batch_size, max_draft_len]
    # 将2D位置映射到1D展平数组的索引
    global_token_indices = cu_start[:, None] + pos_indices
    # 防止越界访问
    global_token_indices = global_token_indices.clamp(0, draft_token_ids.shape[0] - 1)

    # 获取每个位置的draft token
    draft_tokens = draft_token_ids[global_token_indices]  # [batch_size, max_draft_len]

    # ==================== 步骤3: 获取每个位置的概率 ====================
    if IS_NGRAM:
        # N-gram模式：draft没有概率分布，假设概率为1（总是接受）
        ones_cpu = torch.ones(1, pin_memory=True, dtype=torch.float32)
        draft_token_probs = ones_cpu.to(device, non_blocking=True).expand_as(draft_tokens)
    else:
        # 获取draft token在draft分布中的概率
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = draft_tokens.flatten()
        flat_draft_probs = draft_probs[flat_indices, flat_draft_tokens]
        draft_token_probs = flat_draft_probs.view(batch_size, max_draft_len)

    # 获取draft token在target分布中的概率
    flat_indices = global_token_indices.flatten()
    flat_draft_tokens = draft_tokens.flatten()
    flat_target_probs = target_probs[flat_indices, flat_draft_tokens]
    target_token_probs = flat_target_probs.view(batch_size, max_draft_len)

    # 获取均匀随机数和恢复tokens
    uniform_token_probs = uniform_probs[global_token_indices]
    recovered_tokens = recovered_token_ids[global_token_indices]

    # ==================== 步骤4: 计算接受条件 ====================
    # 接受条件: draft_prob > 0 且 target_prob / draft_prob >= uniform_prob
    # 即: p_target >= p_draft * uniform
    zero_threshold_cpu = torch.tensor([0.0], pin_memory=True, dtype=torch.float32)
    zero_threshold = zero_threshold_cpu.to(device, non_blocking=True)

    acceptance_condition = (draft_token_probs > zero_threshold) & (
        target_token_probs / draft_token_probs >= uniform_token_probs
    )

    # ==================== 步骤5: 找到第一个拒绝位置 ====================
    # first_rejection: [batch_size, max_draft_len]
    # True表示该位置是第一个被拒绝的位置
    first_rejection = (~acceptance_condition) & valid_mask

    # 默认位置设为max_draft_len（表示没有拒绝）
    default_pos_cpu = torch.full([batch_size, 1], max_draft_len, pin_memory=True)
    default_pos = default_pos_cpu.to(device, non_blocking=True)

    # 找到每行第一个True的位置（第一个拒绝位置）
    first_reject_pos = torch.where(
        first_rejection.any(dim=1, keepdim=True), first_rejection.float().argmax(dim=1, keepdim=True), default_pos
    )

    # ==================== 步骤6: 创建跳过掩码 ====================
    # pos_mask: 位置 >= 第一个拒绝位置
    pos_mask = pos_indices >= first_reject_pos
    # should_skip: 需要跳过的位置（第一个拒绝位置之后的有效位置）
    should_skip = pos_mask & valid_mask

    # ==================== 步骤7: 确定最终接受和更新掩码 ====================
    # 最终接受：原本接受且不在跳过范围内
    final_acceptance = acceptance_condition & (~should_skip)

    # 非贪婪掩码
    non_greedy_mask = ~is_greedy

    # 需要更新的掩码：非贪婪 + 有效 + 不跳过
    update_mask = non_greedy_mask[:, None] & valid_mask & (~should_skip)

    # 第一个拒绝位置也需要更新（填入recovered token）
    first_reject_mask = (pos_indices == first_reject_pos) & valid_mask & non_greedy_mask[:, None]
    final_update_mask = update_mask | first_reject_mask

    # ==================== 步骤8: 选择最终tokens ====================
    final_tokens = torch.where(
        first_reject_mask,  # 第一个拒绝位置
        recovered_tokens,  # 使用recovered token
        torch.where(final_acceptance, draft_tokens, output_token_ids[:, :max_draft_len]),  # 接受的位置使用draft
    )

    # 更新输出
    output_token_ids[:, :max_draft_len] = torch.where(
        final_update_mask, final_tokens, output_token_ids[:, :max_draft_len]
    )

    # ==================== 步骤9: 填充bonus tokens ====================
    # 判断哪些请求没有拒绝（全部接受）
    no_rejection = first_reject_pos.squeeze(1) >= num_draft_per_batch
    should_add_bonus = non_greedy_mask & no_rejection

    bonus_positions = num_draft_per_batch  # [batch_size]

    seq_len = output_token_ids.shape[1]
    all_positions_cpu = torch.arange(seq_len, pin_memory=True)
    all_positions = all_positions_cpu.to(device, non_blocking=True)[None, :]  # [1, seq_len]

    batch_bonus_positions = bonus_positions[:, None]  # [batch_size, 1]

    max_spec_len_cpu = torch.tensor([max_spec_len], pin_memory=True)
    max_spec_len_device = max_spec_len_cpu.to(device, non_blocking=True)

    # bonus位置必须在有效范围内
    valid_bonus_pos = bonus_positions < (max_spec_len_device + 1)
    final_bonus_mask = should_add_bonus & valid_bonus_pos

    # 找到bonus应该填入的位置
    bonus_pos_match = all_positions == batch_bonus_positions
    bonus_pos_mask = bonus_pos_match & final_bonus_mask[:, None]

    # 填入bonus tokens
    bonus_values_expanded = bonus_token_ids.view(-1, 1).expand(-1, seq_len)
    output_token_ids[:] = torch.where(bonus_pos_mask, bonus_values_expanded, output_token_ids)


def expand_pytorch(
    output_ptr,  # [num_tokens] 输出张量，存储扩展后的token级别值
    input_ptr,  # [batch_size] 输入张量，batch级别的值
    cu_num_tokens_ptr,  # [batch_size] 累积token数量（前缀和）
    replace_from,  # 需要被替换的值
    replace_to,  # 替换后的值
    MAX_NUM_TOKENS,  # 最大token数量（用于避免重编译）
):
    """
    将batch级别的值扩展到token级别（PyTorch实现）。

    核心功能：
    =========
    将 [batch_size] 的张量扩展为 [num_tokens] 的张量，
    类似于 "scatter" 或 "repeat_interleave" 操作，但带有自定义逻辑。

    算法步骤：
    =========
    1. **范围广播**: 创建布尔矩阵 [num_tokens, batch_size]
       标识每个token属于哪个batch
    2. **条件替换**: 在扩展前替换特定值（如填充值或特殊标记）
    3. **矩阵映射**: 使用 torch.einsum 执行加权和，
       一次性为所有token位置选择正确的batch值

    举例说明：
    =========
    假设 batch_size=3, num_tokens=6

    input_ptr: [10, 20, 30]  (每个batch的值)
    cu_num_tokens_ptr: [2, 5, 6]  (累积token数量)
    replace_from=0, replace_to=1

    计算：
    cu_start: [0, 2, 5]
    cu_end: [2, 5, 6]

    token 0,1 属于 batch 0 (0 <= idx < 2)
    token 2,3,4 属于 batch 1 (2 <= idx < 5)
    token 5 属于 batch 2 (5 <= idx < 6)

    输出: [10, 10, 20, 20, 20, 30]
    """
    device = cu_num_tokens_ptr.device
    batch_size = input_ptr.shape[0]
    num_tokens = output_ptr.shape[0]

    # 边界情况：空批次或空tokens
    if batch_size == 0 or num_tokens == 0:
        return

    # ==================== 步骤1: 计算每个batch的起始和结束索引 ====================
    # cu_start[i] = batch i 的起始token索引
    # cu_end[i] = batch i 的结束token索引（不包含）
    cu_start = torch.cat([torch.tensor([0], pin_memory=True).to(device, non_blocking=True), cu_num_tokens_ptr[:-1]])
    cu_end = cu_num_tokens_ptr

    # ==================== 步骤2: 创建范围掩码 ====================
    # token_indices: [num_tokens, 1] = [[0], [1], ..., [num_tokens-1]]
    token_indices = torch.arange(num_tokens, device=device)[:, None]
    # 扩展cu_start和cu_end以便广播比较
    cu_start_exp = cu_start[None, :]  # [1, batch_size]
    cu_end_exp = cu_end[None, :]  # [1, batch_size]

    # in_range[i, j] = True 表示token i 属于 batch j
    # 即 cu_start[j] <= token_indices[i] < cu_end[j]
    in_range = (token_indices >= cu_start_exp) & (token_indices < cu_end_exp)

    # ==================== 步骤3: 条件替换 ====================
    # 将input_ptr中等于replace_from的值替换为replace_to
    # 转换为float以便进行einsum计算
    replaced_input = torch.where(input_ptr == replace_from, replace_to, input_ptr).float()

    # ==================== 步骤4: 使用einsum计算扩展值 ====================
    # in_range.float(): [num_tokens, batch_size] - 每个token所属的batch
    # replaced_input: [batch_size] - 每个batch的值
    # token_values: [num_tokens] - 每个token对应的值
    # einsum("tb,b->t") 等价于矩阵乘法：对于每个token，选择其所属batch的值
    token_values = torch.einsum("tb,b->t", in_range.float(), replaced_input)

    # ==================== 步骤5: 更新输出 ====================
    # 只有属于某个batch的token才需要更新
    needs_update = in_range.any(dim=1)

    output_ptr[:] = torch.where(needs_update, token_values, output_ptr)


def sample_recovered_tokens_pytorch(
    output_token_ids,  # [num_tokens] 输出张量，存储每个位置的恢复token id
    cu_num_draft_tokens,  # [batch_size] 累积draft token数量（前缀和）
    draft_token_ids,  # [num_tokens] draft模型生成的所有token ids（展平）
    draft_probs,  # [num_tokens, vocab_size] draft概率分布，N-gram模式时为None
    target_probs,  # [num_tokens, vocab_size] target模型的概率分布
    q,  # [batch_size, vocab_size] 指数分布随机数，用于Gumbel-max采样
    vocab_size,  # 词表大小
    IS_NGRAM=False,  # 是否为N-gram模式（无draft概率）
):
    """
    推测解码中的恢复token采样函数（PyTorch实现）。

    核心目的：
    =========
    当draft token被拒绝时，需要从修正后的分布中采样一个"恢复token"。
    这是 sample_recovered_tokens 的 PyTorch fallback 实现。

    算法步骤：
    =========
    1. **Token-to-Batch映射**: 使用累积draft token计数，确定每个token属于哪个请求。
       这是必要的，因为 'q' 是按请求存储的 [batch_size, vocab_size]。

    2. **概率修正**:
       - N-GRAM模式: 将draft token在target分布中的概率置零
       - 概率模式: 计算 max(0, target_probs - draft_probs)，标准推测解码算法

    3. **归一化与采样**: 将修正概率除以归一化分布 'q'，使用向量化操作。

    4. **Argmax选择**: 使用 torch.argmax 在一次pass中为所有位置选择恢复token。

    数学原理：
    =========
    使用 Gumbel-max 技巧的变体:
    - 生成指数分布随机数 q ~ Exp(1)
    - 计算 scores = prob / q
    - argmax(scores) 等价于从 prob 分布中采样

    举例说明：
    =========
    假设 batch_size=2, num_tokens=5, vocab_size=10000

    请求0: draft生成3个tokens
    请求1: draft生成2个tokens

    cu_num_draft_tokens: [3, 5]
    cu_start: [0, 3]
    cu_end: [3, 5]

    token_to_batch: [0, 0, 0, 1, 1]
      - token 0,1,2 属于请求0
      - token 3,4 属于请求1

    对于每个token位置，计算:
      prob = max(0, target_probs[i] - draft_probs[i])
      scores = prob / q[token_to_batch[i]]
      recovered_ids[i] = argmax(scores)
    """
    device = output_token_ids.device
    num_tokens = output_token_ids.shape[0]

    # 边界情况：没有token需要处理
    if num_tokens == 0:
        return

    # ==================== 步骤1: 计算每个请求的起始和结束索引 ====================
    # cu_start[i] = 请求i在draft_token_ids中的起始索引
    # cu_end[i] = 请求i在draft_token_ids中的结束索引（不包含）
    # 例如: cu_num_draft_tokens=[3,5] => cu_start=[0,3], cu_end=[3,5]
    cu_start = torch.cat(
        [
            torch.tensor([0], pin_memory=True).to(device, non_blocking=True),
            cu_num_draft_tokens[:-1],
        ]
    )
    cu_end = cu_num_draft_tokens

    # ==================== 步骤2: 建立token到batch的映射 ====================
    # 创建token索引 [0, 1, 2, ..., num_tokens-1]
    token_indices = torch.arange(num_tokens, device=device)  # [num_tokens]

    # 扩展维度以便进行广播比较
    token_indices_expanded = token_indices[:, None]  # [num_tokens, 1]
    cu_start_expanded = cu_start[None, :]  # [1, batch_size]
    cu_end_expanded = cu_end[None, :]  # [1, batch_size]

    # 创建范围mask: in_range_mask[i, j] = True 表示token i 属于请求 j
    # 即 cu_start[j] <= token_indices[i] < cu_end[j]
    in_range_mask = (token_indices_expanded >= cu_start_expanded) & (token_indices_expanded < cu_end_expanded)

    # 使用argmax找到每个token所属的batch索引
    # argmax返回第一个True的位置，即所属的batch
    token_to_batch = torch.argmax(in_range_mask.int(), dim=1)

    # 处理边界情况：如果某个token不属于任何batch（理论上不应发生）
    has_match = in_range_mask.any(dim=1)
    token_to_batch = torch.where(has_match, token_to_batch, 0)

    # ==================== 步骤3: 计算修正后的概率分布 ====================
    if IS_NGRAM:
        # N-gram模式：draft没有概率分布
        # 修正方法：将draft token在target分布中的概率置零
        # 这样恢复时不会选择与draft相同的token
        token_indices = torch.arange(num_tokens, device=device)

        modified_target_probs = target_probs.clone()
        # 将每个位置的draft token对应的概率置零
        # modified_target_probs[i, draft_token_ids[i]] = 0
        modified_target_probs[token_indices, draft_token_ids] = 0
        prob = modified_target_probs

    else:
        # 概率模式：标准推测解码算法
        # 修正概率 = max(0, target_probs - draft_probs)
        # 这确保:
        # 1. 只保留target比draft概率高的部分
        # 2. 恢复分布不会重复draft已经高概率生成的token
        prob = torch.maximum(
            target_probs - draft_probs,
            torch.tensor(0.0, pin_memory=True).to(device, non_blocking=True),
        )

    # ==================== 步骤4: 使用Gumbel-max技巧采样 ====================
    # 将batch级别的q扩展到token级别
    # q_values[i] = q[token_to_batch[i]]
    q_values = q[token_to_batch]  # [num_tokens, vocab_size]

    # 数值稳定性处理：避免除以0或无穷大
    epsilon = 1e-10
    q_values_safe = torch.where(q_values == 0, epsilon, q_values)
    q_values_safe = torch.where(torch.isinf(q_values), epsilon, q_values_safe)

    # 计算采样分数: prob / q
    # 这等价于Gumbel-max采样
    prob_over_q = prob / q_values_safe

    # 对于q=0或inf的位置，将分数设为很小的值，确保不会被选中
    prob_over_q = torch.where((q_values == 0) | torch.isinf(q_values), -1e10, prob_over_q)

    # ==================== 步骤5: 选择恢复token ====================
    # 对每个token位置，选择分数最高的token作为恢复token
    recovered_ids = torch.argmax(prob_over_q, dim=1)

    # 将结果写入输出张量
    output_token_ids[:] = recovered_ids


def rejection_random_sample_block_verify_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1] 输出token矩阵
    cu_num_draft_tokens,  # [batch_size] 累积draft token数量
    draft_token_ids,  # [num_tokens] draft模型生成的token ids（展平）
    draft_probs,  # [num_tokens, vocab_size] draft概率分布，N-gram模式时为None
    target_probs,  # [num_tokens, vocab_size] target概率分布
    bonus_token_ids,  # [batch_size] bonus token ids
    recovered_token_ids,  # [num_tokens] 每个位置的恢复token
    uniform_probs,  # [num_tokens] 均匀随机数
    is_greedy,  # [batch_size] 是否贪婪采样
    max_spec_len,  # 最大投机长度
    vocab_size,  # 词表大小
    IS_NGRAM=False,  # 是否为N-gram模式
):
    """
    使用Block Verify的随机采样拒绝采样（PyTorch实现）。

    核心功能：
    =========
    这是MagicMTP论文中提出的Block Verify方法的实现。
    与传统的逐个验证不同，Block Verify使用累积乘积来一次性验证整个序列。

    算法原理：
    =========
    传统方法：逐个检查 accept_prob = p_target / p_draft >= uniform
    Block Verify：检查累积乘积 ∏(p_target / p_draft) >= ∏uniform

    数学推导：
    - 设 π_k = min(1, p_target(x_k) / p_draft(x_k))
    - 接受前k个token的条件：∏(i=0 to k-1) π_i >= ∏(i=0 to k-1) u_i
    - 其中 u_i 是独立同分布的均匀随机数

    优势：
    =====
    Block Verify可以提高接受率，特别是在draft模型质量较好的情况下。
    这是因为它考虑了整个序列的联合概率，而不是单独验证每个token。

    举例说明：
    =========
    假设 batch_size=1, max_spec_len=3

    draft tokens: [10, 20, 30]
    p_target/p_draft: [0.9, 0.8, 0.95]
    uniform_probs: [0.5, 0.7, 0.6]

    传统方法：
    - 位置0: 0.9 >= 0.5 → 接受
    - 位置1: 0.8 >= 0.7 → 接受
    - 位置2: 0.95 >= 0.6 → 接受

    Block Verify (累积乘积)：
    - 累积π: [0.9, 0.72, 0.684]
    - 累积u: [0.5, 0.35, 0.21]
    - 比较: 0.9>=0.5, 0.72>=0.35, 0.684>=0.21 → 全部接受

    结果：全部接受，追加bonus token
    """
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    # ==================== 步骤1: 计算起始和结束索引 ====================
    zero_cpu = torch.tensor([0], pin_memory=True)
    zero_device = zero_cpu.to(device, non_blocking=True)

    cu_start = torch.cat([zero_device, cu_num_draft_tokens[:-1]])
    cu_end = cu_num_draft_tokens
    num_draft_per_batch = (cu_end - cu_start)[:, None]  # [batch_size, 1]

    # ==================== 步骤2: 创建位置索引 ====================
    pos_indices_cpu = torch.arange(max_spec_len, pin_memory=True)
    pos_indices = pos_indices_cpu.to(device, non_blocking=True)[None, :]  # [1, max_spec_len]

    # 有效位置掩码
    valid_mask = pos_indices < num_draft_per_batch

    # 全局token索引
    global_token_indices = cu_start[:, None] + pos_indices
    global_token_indices = global_token_indices.clamp(0, draft_token_ids.shape[0] - 1)

    # 获取draft tokens
    draft_tokens = draft_token_ids[global_token_indices]

    # ==================== 步骤3: 获取概率 ====================
    if IS_NGRAM:
        # N-gram模式：draft概率为1
        ones_cpu = torch.ones(1, pin_memory=True, dtype=torch.float32)
        draft_token_probs = ones_cpu.to(device, non_blocking=True).expand_as(draft_tokens)
    else:
        # 获取draft token的draft概率
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = draft_tokens.flatten()
        flat_draft_probs = draft_probs[flat_indices, flat_draft_tokens]
        draft_token_probs = flat_draft_probs.view(batch_size, max_spec_len)

    # 获取draft token的target概率
    flat_indices = global_token_indices.flatten()
    flat_draft_tokens = draft_tokens.flatten()
    flat_target_probs = target_probs[flat_indices, flat_draft_tokens]
    target_token_probs = flat_target_probs.view(batch_size, max_spec_len)

    # 获取均匀随机数和恢复tokens
    uniform_token_probs = uniform_probs[global_token_indices]
    recovered_tokens = recovered_token_ids[global_token_indices]

    # ==================== 步骤4: 计算Block Verify接受条件 ====================
    # π = p_target / p_draft，限制最大为1
    pi = target_token_probs / draft_token_probs
    pi = pi.clamp(max=1.0)

    # 计算累积乘积
    # cumprod_pi[k] = ∏(i=0 to k) π_i
    pi = torch.cumprod(pi, dim=-1)

    # 计算均匀随机数的累积乘积
    # cumprod_uniform[k] = ∏(i=0 to k) u_i
    uniform_token_probs = torch.cumprod(uniform_token_probs, dim=-1)

    # 判断是否合法：draft_prob > 0 且 累积π >= 累积uniform
    legal_mask = (draft_token_probs > 0) & (pi >= uniform_token_probs)
    legal_mask = legal_mask & valid_mask

    # ==================== 步骤5: 找到最后一个接受位置 ====================
    # 使用flip + argmax技巧找到最后一个True的位置
    last_accept_pos = torch.where(
        legal_mask.any(dim=-1, keepdim=True),
        (max_spec_len - legal_mask.flip(dims=[-1]).float().argmax(dim=-1, keepdim=True) - 1),
        -1,  # 如果没有接受的位置，设为-1
    )

    # ==================== 步骤6: 创建接受和拒绝掩码 ====================
    non_greedy_mask = (~is_greedy)[:, None]

    # 接受掩码：位置 <= 最后接受位置
    accept_mask = (pos_indices <= last_accept_pos) & valid_mask & non_greedy_mask
    output_token_ids[:, :max_spec_len] = torch.where(accept_mask, draft_tokens, output_token_ids[:, :max_spec_len])

    # 拒绝掩码：位置 = 最后接受位置 + 1（第一个被拒绝的位置）
    reject_mask = (pos_indices == last_accept_pos + 1) & valid_mask & non_greedy_mask
    output_token_ids[:, :max_spec_len] = torch.where(reject_mask, recovered_tokens, output_token_ids[:, :max_spec_len])

    # ==================== 步骤7: 填充bonus tokens ====================
    # 判断是否全部接受：最后接受位置 + 1 >= draft tokens数量
    bonus_mask = (last_accept_pos + 1 >= num_draft_per_batch) & non_greedy_mask

    all_positions_cpu = torch.arange(max_spec_len + 1, pin_memory=True)
    all_positions = all_positions_cpu.to(device, non_blocking=True)[None, :]

    # bonus应该填入的位置 = draft tokens数量
    bonus_pos_match = all_positions == num_draft_per_batch
    bonus_mask = bonus_mask & bonus_pos_match

    bonus_values_expanded = bonus_token_ids.view(-1, 1).expand(-1, max_spec_len + 1)
    output_token_ids[:] = torch.where(bonus_mask, bonus_values_expanded, output_token_ids)
