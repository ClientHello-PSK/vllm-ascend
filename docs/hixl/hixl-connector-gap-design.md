# HIXL Connector 能力缺口清单

> 对照 `MooncakeConnectorV1`（[`mooncake_connector.py`](../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py)）与 `HIXLConnector`（[`hixl_connector.py`](../../vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py)）的能力对比。
> 配合 [`hixl-connector-design.md`](./hixl-connector-design.md)、[`hixl-connector-phase3-plan.md`](./hixl-connector-phase3-plan.md) 使用。
> 代码引用 `文件:行`（相对仓库根）；mooncake 行号仅作 fork 锚点，实施前需复核。

---

## 1. 背景

本文件记录当前 `mooncake_connector.py` 支持、而 `hixl_connector.py` 不支持或未正确实现的能力。HIXL 用 `llm_datadist` 的 `pull_blocks`（block 级寻址）替代了 Mooncake 的字节级 `batch_transfer` / `kv_caches_base_addr` / `_append_mamba_transfer_meta` 字节算术——这是寻址方式差异，**非能力缺口**，不在本清单内。本清单只列寻址无关的能力缺口与潜在缺陷。

- **对比时间**：2026-08-05
- **对照仓库**：`M:\code\hixl\vllm-ascend-v0.23.0`
- **最后提交**：`9234a8f9`（2026-08-03 11:15:49 +0800）
- **对照文件**：
  - Mooncake：[`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`](../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py)
  - HIXL：[`vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py`](../../vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py)

> 注：多数项基于静态代码分析，未在 NPU 上实测；标注「潜在缺陷」者为代码层面发现的 bug，触发条件见各节。工作量估算单位为**人日**，为粗估，实施前需复核。

---

## 2. 缺口总览

| # | 能力 | 类别 | mooncake 锚点 | HIXL 现状 | 影响 | 优先级 |
|---|---|---|---|---|---|---|
| 1 | SWA 滑窗裁剪 | 功能缺口 | `SlidingWindowSpec` import `:53`；`blocks_per_window` `:142,1695`；`_get_swa_transfer_block_ids` `:1735-1749`，调用 `:1904` | `GroupTransferInfo` 无 `blocks_per_window`；无 `_get_swa_transfer_block_ids` | 滑窗模型（Mistral 等）D 侧多收/错收 block | 中 |
| 2 | sparse attention（`use_sparse`/`index_topk`） | 功能缺口 | 检测 `:2335`；HMA sparse 特例 `:2862,3559,3585` | 硬编码 `use_sparse=False`（hixl:1497/1532） | sparse/MoE-attn 模型 CP rank 映射错误 | 中 |
| 3 | SFA DCP replicate-K（含 `enable_sfa_dcp_replicated_indexer` 子项） | 功能缺口 | `_get_sfa_replicate_k_block_ids` `:3301-3372`；`ReqMeta.local_full_block_ids` `:127`；`enable_sfa_dcp_replicated_indexer` `:516,916` | 无任何 SFA 字段/方法/spec import | SFA + DCP + prefix cache K 块位置错 | 中 |
| 4 | Eagle3 投机层层索引 | 功能缺口 | `:2188-2192`（layer id ≥ total_layers 识别 eagle 层并重分配 id） | grep `eagle` 0 命中；`_build_kv_group2layeridx` 只认 `"mtp"`（hixl:1705） | Eagle3 模型草稿层被错索引到目标层段 | 中 |
| 5 | PP 接收侧层范围过滤 | 功能缺口 | 接收线程 `pp_layer_indices(layer_indices, prefill_pp_rank)` `:820-830`，末段扩展 MTP 草稿层 | `prefill_pp_rank` 在 `GroupPull`（hixl:211）但接收侧用 `range(num_layers)`（hixl:844-845,894-895），字段未用 | PP>1 时 D 侧拉错层范围；多 PP rank 层重叠/缺失 | 中 |
| 6 | HMA hybrid-linear 就地重排路径 | 功能缺口 | `reformat_kv_cache_hybrid_linear_torch` `:1083-1109`；`is_hma_required` 分支 `:1032-1049` | `_apply_kv_cache_reformat`（hixl:993-1020）无 `is_hma_required` 分支，全走 staging 路径 `_reformat_staging_to_local` | HMA 多 attention group 共享合并 linear KV 张量时，D 侧就地铁头重组缺失（HIXL 仅 staging 间接覆盖） | 中 |
| 7 | Mamba `prefill_tp>decode_tp`（TP>1 head-shard） | 硬限制 | `_append_mamba_transfer_meta` 字节级切 conv/ssm per-head `:1052-1135` | `assert tp_n==1`（hixl:788/811）；block API 无法 sub-block 头切分 | Mamba PD 分离下 P/D TP 不等的场景不支持 | 中（限制） |
| 8 | `r_blk>1`（Bd>Bp，无 MLA） | 功能缺口（受限） | `:2553-2566` | `assert r_blk==1 or use_mla`（hixl:2220）；`scale==1`+`Bd>Bp` 时 `Bp%Bd≠0` fail fast | P 细 block / D 粗 block 异构 | 低 |
| 9 | Hybrid KV-cache-manager 合并注册 | 功能缺口 | `use_hybrid` `:1666,2015`；`_get_registered_kv_tensor_buffers_hybrid` `:2300`；`collect_storage_merged_register_regions` `:63,2392` | grep `use_hybrid`/合并注册 0 命中；HIXL 走 block API 逐组 `register_blocks_cache` | hybrid（非全 attention）存储模型注册区不合并 | 低 |
| 10 | 按 KV spec 拆分 transfer group | 功能缺口 | `_get_kv_transfer_spec_key` `:2150`；`_serialize_kv_group_spec` `:2087`；`spec_groups` `:2215-2216` | 0 命中；HIXL 一对一不拆（hixl:1678-1734） | 同 manager group 内异质 head 数的层无法分别传输 | 低 |
| 11 | GQA reshard 融合算子 | 性能缺口 | `reformat_kv_cache_with_fused_op` `:1218-1240`，调 `torch.ops._C_ascend.transpose_kv_cache_by_block`，门控 `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` `:1063` | 0 命中；只有纯 torch `_reformat_staging_to_local` | TP>1 reshard 慢（非功能缺口） | 低 |
| 12 | P 侧 rank 感知 delayed-free | 性能缺口 | `if tp_rank in _prefill_get_remote_rank: add_delayed_request else add_not_transfer_request` `:3467-3470` | 无 `_prefill_get_remote_rank`；`start_load_kv` 全量无条件 `add_delayed_request` | 未选中传输的 rank 不及时释放 → 显存占用偏高 | 低 |
| 13 | NZ + GQA 联合（TP>1+NZ） | 功能缺口 | `reformat_kv_cache(..., need_cat_cache, need_nz_cache, ...)` `:1244-1319` 一次过 | HIXL NZ 仅 `tp_n==1` 可用（hixl:896）；TP>1+NZ 显式不支持 | GQA + NZ 排布 + TP>1 模型 | 低 |
| 14 | 多节点 meta_mapping 缺 `handshake_port` 字段 | 潜在缺陷 | `multi_nodes_meta_mapping` 每项含 `"handshake_port": kv_port+port_offset` `:1953-1957` | 只填 `host`/`engine_id`（hixl:1609-1612）；端口解析回退 `kv_port+int(key)`（hixl:2383） | worker 实际 handshake_port 偏离公式时静默用错端口 | 低 |
| 15 | `KVCacheRecvingThread` 未初始化 `self.num_layers` | 潜在缺陷 | RecvingThread `__init__` 设 `self.num_layers = num_hidden_layers` `:525`，`_get_group_kv_caches` MTP 守卫引用 `:1205` | RecvingThread `__init__`（hixl:427-533）未设 `self.num_layers`，但 `_get_group_kv_caches`（hixl:965）引用之 | TP>1 reformat 路径遇 `"mtp"` 层名时 `AttributeError`（tp_n==1 不触发，故 Qwen3.6 实测未暴露） | 低 |
| 16 | `_get_group_pulls_metadata` CP 检测不一致 | 潜在缺陷 | `cp_transfer = remote_pcp*remote_dcp*self.pcp*self.dcp > 1` `:3143`（含本地 CP） | `cp_transfer = remote_pcp*remote_dcp > 1`（hixl:2193，仅远端） | D 侧有 CP、P 侧无 CP 的非对称场景走非 CP 分支，`GroupPull`/`prefill_pp_rank` 描述与 CP shard 布局不一致 | 低 |

**依赖关系**：#3 含 `enable_sfa_dcp_replicated_indexer` 子项；#8 依赖 MLA/compress（`scale>1` 才能使 `kernel_size=Bd/scale` 整除 Bp，V3.2 MLA+CP+Bd>Bp 仍不支持）；#13 仅 GQA+NZ 模型需要（MLA NZ 单 head 不触发）。

---

## 3. 中优先级

### 3.1 #1 SWA 滑窗裁剪

**mooncake 锚点**：`SlidingWindowSpec`（import `:53`）、`GroupTransferInfo.blocks_per_window`（`:142,1695`）、`_get_swa_transfer_block_ids`（`:1735-1749`，裁窗口尾 + 丢占位 block 0）、调用 `:1904`。

**HIXL 现状**：`GroupTransferInfo` 无 `blocks_per_window`；无 `_get_swa_transfer_block_ids`。

**HIXL 实现方式**：
1. `GroupTransferInfo` 加 `blocks_per_window` 字段（hixl:216 dataclass）。
2. `_get_group_transfer_info` 读 `SlidingWindowSpec.sliding_window`，算 `blocks_per_window = cdiv(sliding_window, block_size)+1`。
3. `import SlidingWindowSpec`；fork `_get_swa_transfer_block_ids`（`:1735-1749`，寻址无关可直接 fork）。
4. `request_finished` 调用链：`_get_transfer_block_ids` → `_get_swa_transfer_block_ids`（fork `:1904` 顺序）。

**工作量**：小（1–2 人日）。寻址无关，纯 fork + 字段补充。

**风险**：低。已确认：block 0 是 vLLM 保留的 null/占位 block（非首个真实 block），`_get_swa_transfer_block_ids`（mooncake `:1735-1749`，docstring 明述 "drop placeholder block 0"）仅对 SWA 组做尾部裁剪并 `filter != 0`，Mamba/非 SWA 组原样穿过；`blocks_per_window = cdiv(sliding_window, block_size)+1`（`+1` 覆盖窗口 token 跨度）。寻址无关可直接 fork。剩余仅需 SWA 模型（Mistral 等）NPU 实测。

**验收**：SWA 模型 P/D，D 侧只收窗口内尾部 block，与 mooncake 一致。

### 3.2 #2 sparse attention

**mooncake 锚点**：`use_sparse` 检测 `index_topk`（`:2335`）；HMA 分支 `:2862,3559,3585` 的 sparse 特例（退化为单 head group，类似 MLA）。

**HIXL 现状**：硬编码 `use_sparse=False`（hixl:1497/1532）。

**HIXL 实现方式**：
1. `use_sparse` 改为 `_model_uses_sparse()`（检测 `hf_text_config.index_topk`）。
2. HMA 分支（`_get_remote_ranks_for_req`、`_get_cp_shard_pulls` 等）加 `use_sparse` 判定，sparse 下 rank 选择退化为单组（fork mooncake `:2728` 的 `if use_mla or use_sparse`）。

**工作量**：中（2–3 人日）。需理解 sparse HMA 退化为单 head group 的 rank 选择逻辑。

**风险**：低-中。已确认：sparse 在所有分支与 MLA 行为**完全一致**——`use_sparse` 在每处都与 `use_mla` 成对出现（`if use_mla or use_sparse`，mooncake `:2722,2728,2862,3559`），且 `:3585` 对 sparse 强制 `num_key_value_heads=1`（同 MLA），`_get_cp_shard_pulls`（`:3060-3109`）无 sparse/MLA 分支，故 CP 下 sparse 退化与 MLA 完全相同，无 sparse-specific 处理。fork 时按 MLA 路径同构即可。剩余仅需 sparse 模型实测。

**验收**：sparse attention 模型 P/D，rank 选择与 mooncake 一致。

### 3.3 #3 SFA DCP replicate-K（含 `enable_sfa_dcp_replicated_indexer`）

**mooncake 锚点**：`_get_sfa_replicate_k_block_ids`（`:3301-3372`）、`ReqMeta.local_full_block_ids`（`:127`）、`enable_sfa_dcp_replicated_indexer`（`:516,916`）。

**HIXL 现状**：无任何 SFA 字段/方法/spec import。

**HIXL 实现方式**：
1. `ReqMeta` 加 `local_full_block_ids` 字段（hixl:184 dataclass）。
2. fork `_get_sfa_replicate_k_block_ids`（`:3301`，寻址无关）。
3. `enable_sfa_dcp_replicated_indexer` 检测（fork `:516`）。
4. CP 分支（`_get_kv_split_metadata_cp`）SFA 下按 global block 重建 K 副本。

**工作量**：中-大（4–6 人日）。SFA indexer 语义复杂，CP 下 K 副本重建逻辑较重，需贯通 ReqMeta/发送/接收三侧。

**风险**：中-高。需 SFA + DCP + prefix cache 模型实测；replicate-K 块位置正确性关键，错位会导致 attention 计算错且不一定 fail fast；与多组唯一 `model_id` 约束（Bug 8 教训）交互需复核。

**验收**：SFA + DCP + prefix cache，D 侧 K 块位置与 mooncake 一致。

### 3.4 #4 Eagle3 投机层层索引

**mooncake 锚点**：`:2188-2192`（layer id ≥ total_layers 识别 eagle 层并重分配 id）。

**HIXL 现状**：`_build_kv_group2layeridx` 只认 `"mtp"`（hixl:1705），grep `eagle` 0 命中。

**HIXL 实现方式**：
1. `_build_kv_group2layeridx`（hixl:1678-1734）fork mooncake `:2188-2192` 的 eagle 判定：layer id ≥ total_layers 视为 eagle 层，从 total_layers 起重新编号。
2. 与 MTP 草稿层共用 draft-layer 分组路径（`num_draft_layers` 已有）。

**工作量**：小（1–2 人日）。

**风险**：低。已确认：eagle/draft 层经 `:2200-2202` 冲突检测重分配到 id `≥ total_layers`，且仅末段 PP rank 的 `end_layer_index += num_draft_layers`（mooncake `:822-823`），故 eagle 层**结构性地**只落入末段 PP rank 的范围（非偶然）；`num_draft_layers` 等于 eagle 草稿层 `num_hidden_layers`，扩展宽度与重分配 id 范围对齐。fork 该逻辑即可。剩余仅需 Eagle3 模型实测。

**验收**：Eagle3 模型 P/D，草稿层 KV 与目标层不混淆，逐位与 mooncake 对齐。

### 3.5 #5 PP 接收侧层范围过滤

**mooncake 锚点**：接收线程 `pp_layer_indices(layer_indices, prefill_pp_rank)`（`:820-830`），末段扩展 MTP 草稿层。

**HIXL 现状**：`prefill_pp_rank` 在 `GroupPull`（hixl:211），`self.pp_layer_indices` 也建了（hixl:1692），但接收侧 `_transfer_kv_cache_all_groups` 直接 `range(num_layers)`（hixl:844-845,894-895），字段未使用。注释 hixl:885-892 明示"故意不过滤"，依赖"D 侧每个 PP rank 的注册 cache 只含本段层"——**该依赖仅在 D-PP == P-PP（`self.pp_size == _prefill_pp_size`）时成立**；D-PP==1（PD 分离常见）时 D 侧 cache 含全部层，`range(num_layers)` 会把 P-rank-r 段写到 D 的 `[0, seg)` 而非 `[pp_start, pp_end)`，r>0 即错位。

**关键确认（layer_range 语义）**：`pull_blocks` 的 `src/dst_layer_range` 是**注册 cache 张量列表的位置索引（0-based positional），不是绝对层 id**（`num_tensors == len(layer_indices)*2`，hixl:2962-2983；`num_layers = num_tensors//2`，hixl:893）。mooncake `:820-830` 的 `pp_layer_indices` 返回**绝对层 id**（字节寻址下按 `local_kv_caches_base_addrs[layer_idx]` 直接索引，mooncake `:879-880,909-911`）。**故不能把 mooncake 的绝对范围直接塞给 `pull_blocks` 的两个 layer_range。**

**HIXL 实现方式**（需非对称范围，非单一 range）：
1. 接收侧 fork mooncake `:820-830` 的 `pp_layer_indices` 逻辑得到绝对 `[pp_start, pp_end)`（末段 PP rank 且 `speculative_config` 非空时 `pp_end += num_draft_layers`）。
2. `pull_blocks` 传**非对称** layer_range：
   - `src_layer_range = range(0, pp_end - pp_start)`（P-rank 段 cache 是 positional `[0, seg)`）；
   - `dst_layer_range = range(pp_start, pp_end)`（D-PP==1 时 D cache 含全层，positional==absolute）。
3. D-PP==P-PP 对称场景（每 D rank cache 段化）下 src/dst 都用 `range(0, seg)`，与现行代码一致，无回归。

**工作量**：中（3–4 人日）。原估 2–3 偏低，因需处理 src/dst 非对称 + D-PP 与 P-PP 两场景。

**风险**：中。src/dst 非对称范围易写反；末段 PP rank 的 MTP 扩展边界（`num_draft_layers`）在 src/dst 两侧需一致；D-PP==1 vs D-PP==P-PP 两路径都需覆盖。

**验收**：PP>1 配置 P/D，每个 D rank 只拉自己 PP 段的层，与 mooncake 一致；MTP 草稿层归属末段 PP rank。

### 3.6 #6 HMA hybrid-linear 就地重排路径

**mooncake 锚点**：`reformat_kv_cache_hybrid_linear_torch`（`:1083-1109`），`is_hma_required` 分支（`:1032-1049`），就地在 D cache 上 `[block, split, token, head_per_split, dim] -> [block, token, split, ...]` 转置，无需 staging。

**HIXL 现状**：`_apply_kv_cache_reformat`（hixl:993-1020）无 `is_hma_required` 分支，所有组走 staging 路径 `_reformat_staging_to_local`（注释 hixl:999 自述为 mooncake hybrid-linear 的 adaptation，但用 staging 而非就地）。HIXL 的 HMA rank 选择（`_is_hma_required`、`_get_hybrid_remote_rank_group_pulls`）已具备，缺接收侧就地重排。

**关键确认（staging 不覆盖 HMA）**：staging 是**逐 attention group**分配的（`_init_staging_caches` hixl:2766-2822，每 `kv_cache_group_id` 一套 staging 张量），无"多 group 共享一个合并 linear 张量"的概念。合并 linear 张量场景下不崩溃只是**巧合**：共享层的 `kv_caches[layer_name]` 是合并张量，`ref_k.shape[-2]` 取到合并头数，使 `index_copy_` 形状恰好匹配，但每个共享组各自分配全合并尺寸 staging、各自 pull 同一 shard、各自对同一 D 张量做 N 次幂等覆写（N× 显存、N× pull）。**故 staging 路径不真正覆盖 HMA，是实打实的功能缺口**（作者自己也标为中优先级缺口）。

**HIXL 实现方式**：fork `reformat_kv_cache_hybrid_linear_torch`（mooncake `:1083-1109`），在 `_apply_kv_cache_reformat` 加 `is_hma_required` 分支，就地在 D cache 上做 `[block, split, token, head_per_split, dim] -> [block, token, split, ...]` 转置（`reshape(tp_num_need_pulls,...).transpose(1,2).reshape_as`），不经 staging。

**工作量**：中-大（4–6 人日）。

**风险**：中-高。就地写 D cache 需与 `pull_blocks` 落 staging 的时序协调（HMA 路径绕过 staging，pull 须直落 D cache 或 staging 后就地）；HMA 多 group 共享张量与唯一 `model_id` 约束交互（Bug 8 教训）；多共享组须共享同一 `tp_n` 否则幂等覆写冲突。

**验收**：HMA 多 attention group 共享 linear KV 模型 P/D，D 侧铁头重组与 mooncake 逐位一致。

### 3.7 #7 Mamba `prefill_tp>decode_tp`（TP>1 head-shard）— 硬限制

**mooncake 锚点**：`_append_mamba_transfer_meta`（`:1052-1135`）按 `remote_tp_offset` 切 conv 的 per-head 子段（key/value）与 ssm 的 TP 均分，发出 flat `(src, dst, length)` 三元组。

**HIXL 现状**：block API（`pull_blocks`）以整 block 为最小单位，无法在 block 内做 head 维度 sub-block 切分；mamba 分支 `assert tp_n==1`（hixl:788/811）。

**为何不支持**：block 级寻址表达不了 per-head 字节范围。Qwen3.6 实测为 `prefill_tp==decode_tp`（tp_n==1），此分支未覆盖。

**HIXL 实现方式**（二选一）：
- **方案 A（connector 字节切分）**：在 connector 层自算 conv/ssm per-head 字节偏移（近似 mooncake `_append_mamba_transfer_meta`），走字节级传输通道。需 HIXL 引入与 block API 并存的字节传输路径。
- **方案 B（上游扩展）**：等 `llm_datadist` block API 扩展 sub-block / per-head 切分能力。

**工作量**：方案 A 大（7–10 人日，含字节传输通道引入与纯 block 寻址一致性的破坏评估）；方案 B 0（依赖上游，不可控）。

**风险**：高。方案 A 破坏 HIXL 纯 block 寻址一致性，引入字节路径后 conv/ssm 张量布局/as_strided 切片行为需实测（R2 类风险）；方案 B 时间不可控。建议先评估业务是否真需 `prefill_tp>decode_tp` 的 Mamba PD，再决定是否投入。

**验收**：Mamba PD 分离 `prefill_tp>decode_tp`，D 侧 conv/ssm 字节与 mooncake 逐位一致（需先选定方案并实现）。

---

## 4. 低优先级 / 边缘

### 4.1 #8 `r_blk>1`（Bd>Bp，无 MLA）

**mooncake 锚点**：`:2553-2566`。

**HIXL 现状**：`assert r_blk==1 or use_mla`（hixl:2220）；`_get_group_kernel_params` 的 `assert remote_block_size % kernel_size == 0`（hixl:2467）在 `scale==1`+`Bd>Bp` 时 `Bp%Bd≠0` fail fast。

**HIXL 实现方式**：不单独实现，随 MLA/compress；DeepseekV4 `scale>1` 时 `kernel_size=Bd/scale` 整除 Bp 自然支持。

**工作量**：0（受限，等 DSV4；当前 fail fast 保护）。

**风险**：低。fail fast 保证不错位；DSV4 落地后需实测 `scale>1`+`Bd>Bp`+CP 联合。

**验收**：随 MLA，DSV4 + Bd>Bp 配置下 KV 逐位对齐。

### 4.2 #9 Hybrid KV-cache-manager 合并注册

**mooncake 锚点**：`use_hybrid`（`:1666,2015`）、`_get_registered_kv_tensor_buffers_hybrid`（`:2300`）、`collect_storage_merged_register_regions`（`:63,2392`）。

**HIXL 现状**：grep `use_hybrid`/合并注册 0 命中；HIXL 走 block API 逐组 `register_blocks_cache`。

**HIXL 实现方式**：block 寻址本身逐组注册，合并注册的字节技巧不直接适用；评估 hybrid 存储模型在 block API 下是否必须合并，或可逐组注册（参考多组唯一 `model_id` 方案）。

**工作量**：中（3–5 人日，主要在评估 + 实测）。

**风险**：中。需 hybrid 存储模型实测；逐组 `register_blocks_cache` 可能与 native last-wins 冲突（Bug 8 教训），每组唯一 `model_id` 是前置约束。

**验收**：hybrid 存储模型 P/D，注册区与 mooncake 等价（逐组或合并）。

### 4.3 #10 按 KV spec 拆分 transfer group

**mooncake 锚点**：`_get_kv_transfer_spec_key`（`:2150`）、`_serialize_kv_group_spec`（`:2087`）、`spec_groups`（`:2215-2216`），按 `(spec_type, num_kv_heads, total_num_kv_heads)` 拆组。

**HIXL 现状**：0 命中；`_build_kv_group2layeridx`（hixl:1678-1734）一对一不拆。

**HIXL 实现方式**：fork 上述方法，在 `_build_kv_group2layeridx` 按 spec_key 拆组；每组需唯一 `model_id`。

**工作量**：中（3–4 人日）。

**风险**：低。vLLM 的 uniform-type 分组（`UniformTypeKVCacheSpecs.from_specs`/`is_uniform_with_collection`，`kv_cache_interface.py:861-871,159-172`）已保证组内 `spec_type` 与 `num_kv_heads` 恒定；`total_num_kv_heads` 仅在「同一 kv_cache_group 同时含 base 层与 draft(MTP/Eagle) 层、且 draft_model 的 total_num_kv_heads 与 base 不同」时才不同（`_get_spec_total_num_kv_heads` mooncake `:2164-2177`，draft 层用 `draft_model_config.get_total_num_kv_heads()`）。现有模型不构成此条件（MTP/Eagle 草稿层通常与 base 同头数且多单独成组），故拆分产出 1 组，**当前不可达**，无实测场景；未来出现 base 与 draft 异头数且同组的模型时此拆分才可达。

> 注：spec_key 为 `(type(spec).__name__, num_kv_heads, total_num_kv_heads)`（mooncake `:2150-2162`），**不含 compress_ratio**；compress_ratio 一致性是组内另一条不变量（MLA 族 `merge` assert `:432-439`），与本拆分无关。

**验收**：异质 spec group 模型 P/D，各组层分别传输，与 mooncake 对齐。

### 4.4 #11 GQA reshard 融合算子

**mooncake 锚点**：`reformat_kv_cache_with_fused_op`（`:1218-1240`）调 `torch.ops._C_ascend.transpose_kv_cache_by_block`，门控 `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK`（`:1063`）。

**HIXL 现状**：0 命中；只有纯 torch `_reformat_staging_to_local`。

**HIXL 实现方式**：fork `reformat_kv_cache_with_fused_op`，在 `_apply_kv_cache_reformat` 加融合算子分支（同门控）。

**工作量**：小（1–2 人日）。寻址无关 fork。

**风险**：低。已确认算子在 v0.23.0 **已实现并注册**：`csrc/torch_binding.cpp:2512-2515`（schema+impl，`torch::kPrivateUse1`）、`:688-702`（C++ impl 调 `aclnnTransposeKvCacheByBlock`）、`csrc/moe/transpose_kv_cache_by_block/op_kernel/transpose_kv_cache_by_block.cpp`（AscendC kernel）、门控 `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK`（`ascend_config.py:203`/`envs.py:106`）、E2E 测试 `tests/e2e/.../test_transpose_kv_cache_by_block.py`。HIXL 0 调用，纯性能缺口。剩余仅需融合路径与纯 torch 路径逐位一致性验证。

**验收**：TP>1 P/D，融合算子路径与纯 torch 路径逐位一致，耗时下降。

### 4.5 #12 P 侧 rank 感知 delayed-free

**mooncake 锚点**：`if tp_rank in _prefill_get_remote_rank(req_id): add_delayed_request else add_not_transfer_request`（`:3467-3470`）。

**HIXL 现状**：无 `_prefill_get_remote_rank`；`start_load_kv` 对 `requests_to_send` 全量无条件 `add_delayed_request`。

**HIXL 实现方式**：fork `_prefill_get_remote_rank`/`_get_prefill_ranks_for_group`，在 `start_load_kv`（hixl:2962-2964）按 rank 过滤；未选中 `add_not_transfer_request`。

**工作量**：小（1–2 人日）。

**风险**：低。属显存优化非正确性；rank 选择一致性需与发送侧 `_get_remote_ranks_for_req` 对齐，避免误释放正在传输的 rank。

**验收**：多 TP rank 场景，未选中 rank 的 KV 及时释放，显存占用与 mooncake 一致。

### 4.6 #13 NZ + GQA 联合（TP>1+NZ）

**mooncake 锚点**：`reformat_kv_cache(..., need_cat_cache, need_nz_cache, ...)`（`:1244-1319`）一次过。

**HIXL 现状**：NZ reformat 仅 `tp_n==1` 触发（hixl:896）；TP>1+NZ 显式不支持（staging NZ scatter 分支缺失）。MLA NZ `num_kv_heads==1`→`tp_n==1` 不触发，仅 GQA+NZ 模型需要。

**HIXL 实现方式**：fork mooncake `reformat_kv_cache` 的 `need_cat_cache`+`need_nz_cache` 联合分支，在 staging-to-local 路径叠加 NZ scatter（`npu_scatter_pa_kv_cache`）。

**工作量**：中（3–5 人日）。

**风险**：中。已确认 `npu_paged_cache_load`（mooncake `:1286-1294` vs hixl:1118-1126）与 `npu_scatter_pa_kv_cache`（mooncake `:1365` vs hixl:1160-1162）**调用签名完全一致**（位置参数与 kwargs 同序同义），NZ reshape（`[-1, head_dim*num_kv_heads//16, block_size, 16]`，`nz_fmt_last_dim=16`）两侧也一致；唯一结构差异是 HIXL 的 NZ 触发门控为 `tp_n==1`（hixl:927），mooncake 可 `need_cat_cache+need_nz_cache` 联合跑 TP>1。剩余仅需 GQA+NZ+TP>1 模型实测 + staging 张量物理布局与 scatter 输入匹配验证。

**验收**：GQA + NZ 排布 + TP>1 P/D，D 侧布局与 mooncake 一致。

### 4.7 #14 多节点 meta_mapping 缺 `handshake_port`（缺陷）

**mooncake 锚点**：`multi_nodes_meta_mapping` 每项含 `"handshake_port": kv_port + port_offset`（`:1953-1957`）。

**HIXL 现状**：`multi_nodes_meta_mapping` 只填 `host`/`engine_id`（hixl:1609-1612）；`_get_remote_host_info_by_port`/`get_remote_port_send_num` 回退 `kv_port + int(key)`（hixl:2383）。

**HIXL 实现方式**：fork mooncake `:1953-1957`，在 meta_mapping 每项加 `handshake_port`；端口解析优先用该字段，回退保留公式。

**工作量**：小（0.5–1 人日）。

**风险**：低。需多节点自定义端口偏移配置实测；回退路径保留避免回归。

**验收**：自定义端口偏移的多节点 P/D，端口解析与 mooncake 一致。

### 4.8 #15 `KVCacheRecvingThread` 未初始化 `self.num_layers`（缺陷）

**mooncake 锚点**：RecvingThread `__init__` 设 `self.num_layers = hf_text_config.num_hidden_layers`（`:525`），`_get_group_kv_caches` MTP 守卫 `if layer_idx >= self.num_layers`（`:1205`）。

**HIXL 现状**：RecvingThread `__init__`（hixl:427-533）未设 `self.num_layers`，但 `_get_group_kv_caches`（hixl:965）的 MTP 守卫 `return any(layer_idx >= self.num_layers ...)` 仍引用之。

**为何未暴露**：`_get_group_kv_caches` 仅在 `_apply_kv_cache_reformat`（hixl:1017）的 `num_group_pulls>1` 循环内被调；tp_n==1（Qwen3.6 实测）时 `gqa_reformat_groups` 为空，循环不执行，故未触发。TP>1 + MTP 层进 reformat 路径会 `AttributeError`。

**HIXL 实现方式**：RecvingThread `__init__` 加 `self.num_layers = vllm_config.model_config.hf_text_config.num_hidden_layers`（或经构造参数传入，与 `HIXLConnectorWorker.num_layers` hixl:1647 一致来源）。

**工作量**：极小（0.5 人日）。

**风险**：低。修复后需 TP>1+MTP reformat 路径验证；同时复核 `total_layers`（hixl:1641）与 `num_layers` 在 MTP/Eagle 守卫中的语义是否需区分。

**验收**：TP>1 + MTP 模型 P/D，reformat 路径不报 `AttributeError`，结果与 mooncake 一致。

### 4.9 #16 `_get_group_pulls_metadata` CP 检测不一致（缺陷）

**mooncake 锚点**：`cp_transfer = remote_pcp_size * remote_dcp_size * self.pcp_size * self.dcp_size > 1`（`:3143`，含本地 CP）。

**HIXL 现状**：`cp_transfer = remote_pcp_size * remote_dcp_size > 1`（hixl:2193，仅远端）。注：`_get_kv_split_metadata`（hixl:2262）用完整乘积，split metadata 走 CP 路径；但 `_get_group_pulls_metadata` 单独调用，窄检测在 D 侧有 CP、P 侧无 CP（`self.pcp_size>1` 而 `remote_pcp_size==1`）时判为非 CP。

**HIXL 实现方式**：`cp_transfer` 改为 `remote_pcp_size * remote_dcp_size * self.pcp_size * self.dcp_size > 1`，与 mooncake/`_get_kv_split_metadata` 对齐。

**工作量**：极小（0.5 人日）。

**风险**：低。非对称 CP（D 有 CP、P 无 CP）场景实测；改后两处 CP 判定一致，避免 split metadata 与 group pulls 走不同分支。

**验收**：D 侧有 CP、P 侧无 CP 的非对称配置 P/D，`GroupPull` 描述与 mooncake 一致。

---

## 5. 优先级与落地序

| 批次 | 子项 | 类别 | 依赖 | 工作量 | 说明 |
|---|---|---|---|---|---|
| 1 | #1 SWA | 功能 | 独立 | 1–2 | 滑窗模型，寻址无关可直接 fork |
| 2 | #2 sparse | 功能 | 独立 | 2–3 | sparse 模型，rank 选择退化 |
| 3 | #4 Eagle3 | 功能 | 独立 | 1–2 | Eagle3 草稿层索引 |
| 4 | #5 PP 接收侧层过滤 | 功能 | 独立 | 3–4 | PP>1 正确性（src/dst 非对称 range） |
| 5 | #6 HMA hybrid-linear 就地重排 | 功能 | 独立 | 4–6 | HMA 多 group 共享 linear KV |
| 6 | #3 SFA（含子项） | 功能 | 独立 | 4–6 | SFA + DCP + prefix cache |
| 7 | #7 Mamba TP>1 | 限制 | — | 7–10（方案 A） | block API 限制，需评估字节切分方案 |
| 8 | #8 r_blk>1 无 MLA | 功能 | MLA | 0 | 仍受 `scale==1`+`Bd>Bp` 限制，需 DSV4 |
| 9 | #9 hybrid 注册 | 功能 | 独立 | 3–5 | block API 下需评估 |
| 10 | #10 按 spec 拆组 | 功能 | 独立 | 3–4 | 当前不可达，未来异质 spec |
| 11 | #11 融合算子 | 性能 | 独立 | 1–2 | 性能优化 |
| 12 | #12 rank 感知 delayed-free | 性能 | 独立 | 1–2 | 显存优化 |
| 13 | #13 TP>1+NZ | 功能 | 独立 | 3–5 | GQA + NZ 模型 |
| 14 | #14 handshake_port 字段 | 缺陷 | 独立 | 0.5–1 | 多节点端口解析 |
| 15 | #15 RecvingThread num_layers | 缺陷 | 独立 | 0.5 | TP>1+MTP 触发 |
| 16 | #16 CP 检测一致性 | 缺陷 | 独立 | 0.5 | 非对称 CP |

**建议**：#1/#2/#4/#5/#6/#7 为正确性或硬限制，优先；#8–#13 为性能/边缘/未来场景；#14–#16 为潜在缺陷，可与对应功能项一起修（#15 随 #5 TP>1 路径暴露时修，#16 随 CP 场景修）。

---

## 6. 验收

每子项与 `MooncakeConnectorV1` 同 P/D 几何**逐位对齐**：
- #1：SWA 模型，D 侧只收窗口内 block。
- #2：sparse 模型，rank 选择一致。
- #3：SFA + DCP + prefix cache，K 块位置一致。
- #4：Eagle3 模型，草稿层与目标层不混淆。
- #5：PP>1，每 D rank 只拉自己 PP 段层；MTP 草稿层归末段。
- #6：HMA 多 group 共享 linear KV，D 侧铁头重组一致。
- #7：Mamba `prefill_tp>decode_tp`，conv/ssm 字节一致（需先选定实现方案）。
- #8：随 MLA（仍受 `scale==1`+`Bd>Bp` 限制）。
- #9：hybrid 存储模型，注册区等价。
- #10：异质 spec group，各组层分别传输（当前不可达）。
- #11：TP>1，融合算子与纯 torch 逐位一致。
- #12：多 TP rank，未选中 rank 及时释放。
- #13：GQA + NZ + TP>1，D 侧布局一致。
- #14：自定义端口偏移多节点，端口解析一致。
- #15：TP>1 + MTP，reformat 路径不报 `AttributeError`。
- #16：非对称 CP，`GroupPull` 描述一致。

> 注：本清单基于 mooncake 代码静态分析与 HIXL block 寻址语义推导，多数项未在 NPU 上实测。行号为 `vllm-ascend-v0.23.0` 当前状态（commit `9234a8f9`），实施前需复核。工作量与风险为粗估，实施前需结合实测复核。
