# HIXL Connector 缺失能力设计与实现流程

> 对照 `MooncakeConnectorV1`（[`mooncake_connector.py`](../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py)）的寻址无关能力缺口。
> 配合 [`hixl-connector-design.md`](./hixl-connector-design.md)、[`hixl-connector-phase3-plan.md`](./hixl-connector-phase3-plan.md) 使用。
> 生成日期：2026-07-28。代码引用 `文件:行`（相对仓库根）；mooncake 行号仅作 fork 锚点，实施前需复核。
> `hixl_connector.py` = [`vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py`](../../vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py)。

> **实施状态（2026-07-31）**：G1（MLA/compress 主体）+ G8（CP+MLA 联合 kernel）+ G2（conv_padding，含重构 mamba 注册）+ G3（NZ layout，静态验证）已实现并逐字对齐 mooncake；G7 经评估无需 import `MLAAttentionSpec`（`compress_ratio` 已经 `msgspec.to_builtins` 握手到 D 端，纯 MLA 够用）。详见各节「实施状态」段。hixl 行号已复核至 2026-07-30 实施态（与 2026-07-28 文档原始行号有偏移，因 Phase 3 其他子项推进）。剩余未实现：G4（SWA）/G5（sparse）/G6+G10（SFA）/G9（r_blk>1 受限）。
>
> **2026-07-31 端到端实测**：Qwen3.6-27B（attention + mamba + MTP）PD 分离跑通——D 侧 `pull_blocks` 成功（attention + mamba conv/ssm 全组），MTP `Mean acceptance length: 2.60`、`External prefix cache hit rate: 66.7%`（= (N-1)/N，mamba 末位重算设计值）、请求 200 OK。**G2 的 base mamba P/D 首次端到端验证通过**（R1 关闭）。实测中暴露并修复 hybrid 多组 `BlocksCacheKey` 冲突（见 §3.2「2026-07-31 实测修正」与 [`bugfix-log.md`](../../tests/hixl/bugfix-log.md) Bug 8）——这是任何 hybrid 多组模型的前置条件，文档原未记。

---

## 1. 背景

HIXL Phase 3 已完成 A(MTP)/B(PP)/C(Mamba state)/D(PCP/DCP)，约束 `block_size_scale==1` 且 `r_blk==1`（P/D 同 block_size、无 MLA/compress）。相对 mooncake，HIXL 用 `pull_blocks` 替代了字节寻址（`batch_transfer_sync_read`/`kv_caches_base_addr`/`_append_mamba_transfer_meta` 字节算术等），这部分是寻址方式差异，**非缺口**。本文聚焦 mooncake 有、HIXL 无的**寻址无关能力缺口**（10 项），给出 block 寻址下的设计与实现流程。

---

## 2. 缺口总览

| # | 能力 | mooncake 锚点 | HIXL 状态 | 影响模型/场景 | 优先级 |
|---|---|---|---|---|---|
| G1 | MLA/compress `block_size_scale>1` | `:2374,2585,2618` | ✅ 已实现（放开 `assert scale>=1`，hixl:2562；No-CP `_get_kernel_block_ids` hixl:2394 已对齐 MC） | DeepSeek-V3 MLA、DSV4 compress | 高 |
| G2 | `conv_padding`（Mamba+MTP conv state） | `:2267,2278-2293` | ✅ 已实现（mamba 注册重构：conv/ssm 拆两 Cache `MambaCacheBundle`，hixl:158/2652；`tensor_num_per_layer=1`） | Mamba+MTP（Qwen3.6 潜在） | 高 |
| G3 | NZ layout（`enable_kv_nz`） | `:1059,1244-1366` | ✅ 已实现（fork `_reformat_kv_cache_nz`+`_nz_kv_cache` hixl:1028；TP=1+NZ 触发 hixl:896；静态验证） | NPU NZ 排布 KV cache | 高 |
| G4 | SWA 滑窗裁剪 | `:1735,142,53` | ❌ 全缺 | 滑窗注意力（Mistral 等） | 中 |
| G5 | sparse attention（`use_sparse`） | `:2335,2862` | ⚠️ 硬编码 `False`（hixl:1532） | sparse/MoE-attn（`index_topk`） | 中 |
| G6 | SFA DCP replicate-K | `:3301,127,516` | ❌ 全缺 | SFA + DCP + prefix cache | 中 |
| G7 | `MLAAttentionSpec` 远端握手 | `:52,2166` | ✅ 无需单独实现（`compress_ratio` 经 `msgspec.to_builtins` 进 `group_spec`，`_group_compress_ratio` hixl:2379 能读；纯 MLA 够用） | MLA group spec 误判 | 中（G1 子项） |
| G8 | CP + MLA 联合 kernel（`r_blk>1`+`scale>1`） | `:2505,2618` | ✅ 已实现（fork `_local_kernel_ids_for_shard` hixl:2474 + `_get_group_kernel_params` hixl:2456；CP attention 段 hixl:2328 对齐 MC:3016-3042） | CP + P/D block_size 不等 + MLA | 低（G1 子项） |
| G9 | `r_blk>1`（Bd>Bp，无 MLA） | `:2553-2566` | ❌ assert（hixl:2220） | P 细 block / D 粗 block 异构 | 低 |
| G10 | `enable_sfa_dcp_replicated_indexer` | `:516,916` | ❌ 缺 | SFA DCP | 低（G6 子项） |

**依赖关系**：G7、G8 是 G1 的子项；G10 是 G6 的子项。G2、G3 独立。G4、G5 独立。G9 依赖 G1（r_blk>1 需 scale>1 使 kernel_size 整除 Bp）。

---

## 3. 高优先级

### 3.1 G1：MLA / compress（`block_size_scale>1`）

**能力**：MLA（DeepSeek）把多个 logical block 压进一个 tensor block（`compress_ratio`），`block_size_scale = tensor_num_blocks / logical_num_blocks > 1`。传输需按 **kernel block**（`block_size/scale` 粒度）寻址，logical block 展开为 `scale` 个 kernel block。

**mooncake 锚点**：
- `block_size_scale[layer][cache]` 填充 `:2374-2379`
- `_expand_block_ids(bids, scale)` = `[bid*scale+off]`（`:2500`）
- `_get_kernel_block_ids`（`:2585-2616`）：`local_scale=block_size_scale[layer][0]`、`kernel_size=block_size/scale`、`remote_scale=remote_block_size/kernel_size`
- `_get_group_kernel_params`（`:2618-2633`）：返回 per-group `(local_scale, remote_scale, kernel_size)`
- `MLAAttentionSpec.storage_block_size = block_size//compress_ratio`（vllm `kv_cache_interface.py:402`）

**HIXL block 寻址适配**：
- HIXL `block_size_scale` 已是 **per-group `[scale]`**（`hixl:2504`），与 mooncake per-layer-per-cache `[layer][cache]` 结构不同。同 group 同 scale 时 per-group 足够；若同 group 内不同层 scale 不同（罕见），需升为 per-layer。
- `register_kv_caches`（`hixl:2499-2504`）：放开 `assert scale==1`，`scale = ref_shape[0] // self.num_blocks` 已算出，存 `block_size_scale.append([scale])`。
- `_get_kernel_block_ids`（`hixl:2348`）：`local_scale` 当前恒 1（`hixl:2387`），改为 `self.block_size_scale[group_idx][0]`；`_expand_block_ids` 已移植（`hixl:2338`），scale>1 时自动展开。
- `pull_blocks` 的 `src/dst_blocks` 用 kernel block id（展开后），layer_range 不变。

**实现步骤**：
1. ✅ 放开 `hixl:2562` 的 `assert scale==1`，改为 `assert scale >= 1`（行号较文档原始 2527 偏移，因 Phase 3 其他子项推进）。
2. ✅ `_get_kernel_block_ids`（`hixl:2394`）：`local_scale = self.block_size_scale[group_idx][0]`（替换原硬编码 1 的来源）；`kernel_size`、`remote_scale`、`remote_start_idx`（`:2451`）路径已对齐 mooncake `:2585-2616`，逐字一致。
3. ✅ `block_size_scale` 结构复核：保持 per-group `[[scale]]`（`hixl:2566`），同 group 强制同 shape（`hixl:2546-2550`）保证 group 内 scale 一致，无需升 per-layer。
4. ✅ G7：评估后**不 import `MLAAttentionSpec`**。`compress_ratio` 经 `_build_kv_group2layeridx`（`hixl:1629` `msgspec.to_builtins`）序列化进 `group_spec["kv_cache_spec"]` dict，`_group_compress_ratio`（`hixl:2379`）能读出；纯 MLA（同 group 同 compress_ratio）够用，spec_key 拆分属未来混合 group 场景。
5. ⏳ 验收：DeepSeek-V3 MLA P/D 配置，与 mooncake 同配置 KV 逐位对齐——**待 NPU 实测**（见下「未消化风险」）。

> **实施状态（2026-07-30）**：G1 主体已完成。No-CP 路径（`_get_kernel_block_ids` kernel 展开 + `pull_blocks` 用 kernel id + layer_range 不变）已逐字对齐 mooncake；CP 路径见 G8。staging/TP>1+scale>1 经核实为死代码（MLA `num_kv_heads==1` → `tp_num_need_pulls==1` → 不走 staging，`hixl:_init_staging_caches` `if tp_n<=1: continue`），本轮不动。

> **未消化风险**：No-CP `remote_start_idx`（`hixl:2450-2452`）在 `scale>1`（DeepseekV4 compress）+ D 端有 prefix cache（`num_computed_tokens>0`）时存在与 mooncake **共享**的潜在错位：`remote_kernel_token_size = kernel_size * compress_ratio = block_size`，`remote_start_idx = num_computed_tokens // block_size` = N（logical prefix block 数），但 `kernel_remote` 经 `_expand_block_ids(..., remote_scale=scale)` 展开为 kernel 粒度，跳 N 个 logical 应跳 `N*scale` 个 kernel，代码只跳 N。`scale==1`（V3.2 MLA）不受影响；无 prefix cache（`num_computed_tokens==0`）不触发。按"逐位对齐 mooncake"要求未擅改（mooncake `:2612-2614` 逐字相同）。**建议 NPU 实测 DeepseekV4 + D 端 prefix cache 对照 mooncake**：若 mooncake 也错位则上游一起修（正确改法 `remote_start_idx = num_computed_tokens // kernel_size`）；若 mooncake 正确则复盘 `compress_ratio` 取值。

**风险**：~~`block_size_scale` per-group vs per-layer 结构差异~~（已核实：per-group 足够，同 group 同 shape）；~~CP 分支（`_get_kv_split_metadata_cp`）当前直接用 block id，MLA+CP 联合需 `_local_kernel_ids_for_shard`（G8）~~ → **G8 已完成**（见 §5.2）。剩余风险为 No-CP `remote_start_idx`（见上「未消化风险」）。

### 3.2 G2：conv_padding（Mamba+MTP conv state）— ✅ 已完成

**能力**：MTP 草稿层若共享 Mamba conv state，mooncake 在 register 时 `base_addr -= conv_padding` 把 conv state buffer 纳入注册区一并传输（`:2278-2293`）。

**mooncake 锚点**：
- `_get_mamba_conv_padding`（`:2267-2272`）：`num_blocks * conv_shape.numel() * dtype_size`
- `_get_registered_kv_tensor_buffers`（`:2278-2293`）：`has_mtp → base_addr -= conv_padding`

**Qwen3.5 调查结论（2026-07-28，agent 核实）**：Qwen3.5 MTP 草稿层（[`qwen3_5_mtp.py:106-113`](../../vllm/vllm/model_executor/models/qwen3_5_mtp.py)）强制 `layer_type="full_attention"`，是**纯 attention 层**，无 GDN/Mamba 子模块，不共享 mamba conv_state。**故 Qwen3.6 不受 G2 影响**。

> **结论修正（2026-07-30 实施态）**：上版文档称"基础模型 GDN 的 conv_state+ssm_state 已作为 MambaSpec group 的 tensor 被 register_kv_caches 注册并传输"**不准确**——实测 `register_kv_caches` 的 uniform-shape assert（conv 2D vs ssm 3D，`:2648`）对任何 mamba group 必然崩溃，**base mamba P/D 在 hixl 从未跑通**，不止 G2 的 MTP 场景。G2 实现前需先修 mamba 注册路径本身（见下「实施状态」）。

**G2 实际影响场景**：仅当 MTP 草稿层本身是 mamba 层（共享基础模型 mamba conv state）时才需 conv_padding。当前 Qwen3.5 不属此场景，属罕见/未来模型（当前 vLLm 无此模型，G2 的 MTP-conv_padding **无可测场景**）。

**HIXL block 寻址适配**（若未来需补）：
- HIXL `register_blocks_cache` 按 `kv_caches[layer_name]` tensor 的 `data_ptr` 注册。conv state buffer 不在 `kv_caches` 字典里 → 不被注册。
- **block 寻址下不能搬 `base_addr -= conv_padding`**（无裸字节基址）。改为：把 conv state 作为**额外 tensor** 加入 Mamba group 的 `addrs` 列表，`num_tensors` 相应增加，`CacheDesc.shape` 兼容。

**实现步骤**：
1. ✅ ~~确认目标模型 MTP 草稿层是否共享 mamba conv state（Qwen3.5 已确认否）。~~
2. ✅ `register_kv_caches` 的 Mamba group 收集 conv state buffer addr，加入 `addrs`；调整 `num_tensors` assert（hixl:2491）。
3. ✅ `_get_group_kv_caches` Mamba 分支返回含 conv state 的 tensor 集。
4. ⏳ 验收：Mamba+MTP（mamba 草稿层）配置，D 侧草稿层 conv state 与 P 侧逐字节对齐——**待首个 mamba P/D 模型落地后实测**。

**风险**：~~conv state 的 shape/block 结构与 K/V 不同，`CacheDesc` 需兼容异质 tensor（可能需独立 group 注册）。~~ → **已采用独立 Cache 方案**（见下）。

> **实施状态（2026-07-30）**：已完成。方案为 conv/ssm 拆两个独立 Cache（`MambaCacheBundle` dataclass，hixl:158），各 `num_tensors=num_mamba_layers`、pull 传 `tensor_num_per_layer=1`（已验证 `pull_blocks` 支持、`register_blocks_cache` 仅校验长度）。改动：
> - `register_kv_caches` mamba 专支（hixl:2652）：conv/ssm 各建 `CacheDesc`，绕开 uniform-shape/`*2` assert。
> - `_transfer_kv_cache_all_groups` mamba 分支（hixl:811）：sub-cache 循环 pull，`tensor_num_per_layer=1`、`num_layers=num_tensors`（不再 `//2`），`assert tp_n==1`。
> - block id 无需改（conv+ssm 共享 block table）。
> - **MTP conv_padding 自动解决**：conv_state 已独立注册传输，mooncake 的 `base_addr -= conv_padding` 字节技巧不再需要（block 寻址副产品）。草稿 mamba 层若共享基础 conv_state（同 raw_tensor），重复 data_ptr——register 不拒（已验证），pull 重复写幂等。
>
> **2026-07-31 实测修正（Bug 8，hybrid 多组 `BlocksCacheKey` 冲突）**：上述 conv/ssm 拆两 Cache + attention Cache + 多 mamba 组，**全部**原先用同一个 `BlocksCacheKey(cluster_id, model_id=0)` 注册。native `cache_manager.cc:AddCacheIndices` 对 blocks cache 是 `cache_key_to_id_[key] = cache_id`（直接赋值，last-wins；且 `register_blocks_cache` 走 `RegisterCacheEntry` **不查重**），导致 key 最终只指向最后注册的 mamba-ssm cache（501 blocks、16 tensors），attention 的 pull（src block 636、34 个 tensor 索引）越界 → `LLM_FAILED`。
> **修复**：每个注册的 blocks cache 分配唯一 `model_id`（`_alloc_model_id` 顺序分配，P/D 同配置同序确定性一致）：attention 存 `_group_model_ids[gid]`、mamba 存 `MambaCacheBundle.conv_model_id`/`ssm_model_id`、staging 亦唯一；两处 pull 用对应 model_id。`KVCacheRecvingThread` 加 `group_model_ids` 参数透传。详见 [`bugfix-log.md`](../../tests/hixl/bugfix-log.md) Bug 8。
> **教训（设计约束）**：`BlocksCacheKey` 只有 `(cluster_id, model_id)` 两维、无 group 维度。**任何 hybrid 多组模型（形状各异无法合并进单 cache）都必须给每组（含 mamba 的 conv/ssm 子 cache）唯一 model_id**，否则 native 侧 last-wins 静默覆盖、pull 打错 cache。此约束跨 G1/G2/G3，文档原仅记 G2 conv/ssm 拆分，未记跨组 key 唯一性，实测补。
>
> **未消化风险**：
> - ~~**R1**：base mamba P/D 从未跑通，无 mamba 模型端到端验证；仅几何/契约自洽。~~ → **2026-07-31 关闭**：Qwen3.6-27B（attention+mamba+MTP）PD 跑通，pull 成功、MTP acceptance 2.60、external hit 66.7%。
> - ~~**R2**：conv/ssm dim0 连续、block id 与 dim0 索引一致...未实测 pull 对 as_strided 张量 block 切片行为。~~ → **2026-07-31 关闭**：Qwen3.6 实测 mamba conv/ssm pull 正确（MTP acceptance 正常即反证 KV 字节正确）。
> - **R3**：mamba TP>1（`prefill_tp>decode_tp`）不支持（block API 无法 sub-block 头切分，`assert tp_n==1`）。Qwen3.6 实测为 `prefill_tp==decode_tp`（tp_n==1），此分支仍未覆盖。
> - **R4**：MTP 共享 conv_state 重复 data_ptr 幂等性未实测（register 不拒已验证）。Qwen3.6 MTP 草稿层是纯 attention（非 mamba），此场景仍未覆盖。

### 3.3 G3：NZ layout（`enable_kv_nz`）— ✅ 已完成（静态验证）

**能力**：NPU 上 KV cache 用 NZ 排布（16-末维），拉完后需 NZ scatter 写 D cache。

**mooncake 锚点**：
- `enable_kv_nz` 检测（`:1059` 调用处，仅 MLA D-node，`ascend_config.py:269-278`）
- `reformat_kv_cache`（`:1244-1319`）：`need_cat`/`need_nz` 分支
- `_cat_kv_cache`/`_nz_kv_cache`（`:1321-1366`）：`npu_paged_cache_load` + `_npu_reshape_and_cache` / `npu_scatter_pa_kv_cache`
- `reformat_kv_cache_hybrid_linear_torch`（`:1083-1092`）

**HIXL block 寻址适配**：
- HIXL `_reformat_staging_to_local`（`hixl:914`）仅做 TP head 转置（`transpose(1,2)`），无 cat/NZ/paged-load。
- ~~NZ 下 D cache 是 NZ 排布，pull 落 staging 后需 `npu_scatter_pa_kv_cache` 写 NZ。~~ → 实测发现 MLA NZ `num_kv_heads==1`→`tp_n==1`，**不走 staging**（TP=1 直接落 D cache）；NZ reformat 对 D real cache 操作（fork mooncake，pull 落 D cache ND 后 reformat NZ 化），无需 staging。

**实现步骤**：
1. ✅ 加 `enable_kv_nz` 检测（读 ascend_config，hixl:511 `get_ascend_config().enable_kv_nz`）。
2. ✅ fork `_reformat_kv_cache_nz` + `_nz_kv_cache`（hixl:1028，对齐 MC:1244-1365）：`npu_paged_cache_load` 从 D cache load ND buffer，`npu_scatter_pa_kv_cache` scatter 回 D cache NZ view（`nz_fmt_last_dim=16`，对齐 MLA）。
3. ✅ TP=1+NZ 触发点（hixl:896）：pull 后对涉及 group 调 `_reformat_kv_cache_nz`。
4. ⏳ 验收：NZ 排布下 P/D，D 侧 cache 布局与 mooncake 一致——**待 NPU 实测**。

**风险**：NZ scatter 算子依赖 vllm-ascend CANN ops（`torch_npu.npu_scatter_pa_kv_cache`，attention 层已用）；~~staging Cache 的 tensor 布局需与 NZ scatter 输入匹配。~~ → TP=1 无 staging，D cache NZ view 直接喂 scatter。

> **实施状态（2026-07-30）**：已完成（静态验证，无 NPU）。MLA NZ `num_kv_heads==1`→`tp_n==1`，故 G3 只需 TP=1+NZ 路径（fork mooncake `reformat_kv_cache` NZ 分支，对 D real cache 操作）。
>
> **未消化风险（无法验证）**：
> - **R5**：NZ reformat 的 `slot_mapping = block_id*block_size`、`num_tokens = num_blocks*block_size` 假设 `scale==1`（fork mooncake 同）。MLA NZ + compress（DeepseekV4 `scale>1`）+ NZ 同时启用时，block_id 是 kernel id、每 block 实际 `block_size//scale` tokens，slot_mapping/num_tokens 会错位——mooncake 上游同此假设。建议 NPU 实测时若启用 compress+NZ 重点验证。
> - **TP>1+NZ** 不支持（staging NZ scatter 分支，MLA 单 head 不触发，留待）。
> - 无 NPU，`npu_paged_cache_load`/`npu_scatter_pa_kv_cache` 调用签名对齐 mooncake 但未实测。

---

## 4. 中优先级

### 4.1 G4：SWA 滑窗裁剪

**mooncake 锚点**：`SlidingWindowSpec`（import `:53`）、`GroupTransferInfo.blocks_per_window`（`:142,1695`）、`_get_swa_transfer_block_ids`（`:1735-1749`，裁窗口尾 + 丢占位 block 0）、调用 `:1904`。

**HIXL 适配**：
1. `GroupTransferInfo` 加 `blocks_per_window` 字段（hixl:309 dataclass）。
2. `_get_group_transfer_info`（hixl:1180）读 `SlidingWindowSpec.sliding_window`，算 `blocks_per_window = cdiv(sliding_window, block_size)+1`。
3. `import SlidingWindowSpec`；加 `_get_swa_transfer_block_ids`（fork `:1735-1749`，寻址无关可直接 fork）。
4. `request_finished`（hixl:1132）调用链：`_get_transfer_block_ids` → `_get_swa_transfer_block_ids`（fork `:1904` 顺序）。

**验收**：SWA 模型 P/D，D 侧只收窗口内尾部 block，与 mooncake 一致。

### 4.2 G5：sparse attention

**mooncake 锚点**：`use_sparse` 检测 `index_topk`（`:2335`）；HMA 分支 `:2862,3559,3585` 的 sparse 特例（退化为单 head group，类似 MLA）。

**HIXL 适配**：
1. `use_sparse` 从 `False`（hixl:1497）改为 `_model_uses_sparse()`（检测 `hf_text_config.index_topk`）。
2. HMA 分支（`_get_remote_ranks_for_req` hixl:1481、`_get_cp_shard_pulls` 等）加 `use_sparse` 判定，sparse 下 rank 选择退化为单组（fork mooncake `:2728` 的 `if use_mla or use_sparse`）。

**验收**：sparse attention 模型 P/D，rank 选择与 mooncake 一致。

### 4.3 G6：SFA DCP replicate-K

**mooncake 锚点**：`_get_sfa_replicate_k_block_ids`（`:3301-3372`）、`ReqMeta.local_full_block_ids`（`:127`）、`enable_sfa_dcp_replicated_indexer`（`:516,916`）。

**HIXL 适配**：
1. `ReqMeta` 加 `local_full_block_ids` 字段（hixl:130 dataclass）。
2. 加 `_get_sfa_replicate_k_block_ids`（fork `:3301`，寻址无关）。
3. `enable_sfa_dcp_replicated_indexer` 检测（fork `:516`）。
4. CP 分支（`_get_kv_split_metadata_cp`）SFA 下按 global block 重建 K 副本。

**验收**：SFA + DCP + prefix cache，D 侧 K 块位置与 mooncake 一致。

---

## 5. 低优先级 / 边缘

### 5.1 G7：`MLAAttentionSpec` 远端握手（G1 子项）— ✅ 无需实现

~~`import MLAAttentionSpec`；`_build_kv_group2layeridx` 加 `_get_spec_total_num_kv_heads`/`_get_spec_num_key_value_heads`/`_get_kv_transfer_spec_key`/`_serialize_kv_group_spec`（fork `:2142,2166`）。MLA spec 当 `num_kv_heads==1` 处理。~~

> **实施状态（2026-07-30）**：经评估无需实现。`_build_kv_group2layeridx`（`hixl:1629`）已用 `msgspec.to_builtins` 把 spec 序列化为 dict，`compress_ratio`/`num_kv_heads` 保留进 `group_spec["kv_cache_spec"]`；`_group_compress_ratio`（`hixl:2379`）能从 dict 取 `compress_ratio`，`_get_attention_group_num_key_value_heads`（`hixl:1699`）能取 `num_kv_heads`（MLA=1）。纯 MLA（同 group 同 compress_ratio）够用。spec_key 拆分 / `total_num_kv_heads` 仅对**混合 compress_ratio 的 kv_cache_group**需要——当前 vLLM 中 compress_ratio 仅存在于 MLA 族 spec（`MLAAttentionSpec`/`SlidingWindowMLASpec`/`HiddenStateCacheSpec`，均 `num_kv_heads==1`），同 group 必同 compress_ratio（`_merge_kv_cache_specs` assert，`kv_cache_interface.py:432-439`），故实际不可达。未来出现异质 spec group 时再补。

### 5.2 G8：CP + MLA 联合 kernel（G1 子项）— ✅ 已完成

~~`_get_group_kernel_params`（fork `:2618`）+ `_local_kernel_ids_for_shard`（fork `:2505`）的 block 级改写。`r_blk>1` + `scale>1` 联合。放开 `hixl:2185` 的 `assert r_blk==1 or use_mla`。随 G1 + D4 一起做。~~

> **实施状态（2026-07-30）**：已完成，逐字对齐 mooncake。
> 1. ✅ fork `_get_group_kernel_params`（hixl:2456，对齐 MC:2618-2633）：唯一差异是 `block_size_scale` 索引 `[group_idx]`（hixl per-group）vs MC `[layer_indices[0]]`（per-layer），hixl 同 group 同 shape 故 per-group 正确。
> 2. ✅ fork `_local_kernel_ids_for_shard`（hixl:2474，对齐 MC:2505-2567）：逐字移植，签名/循环体/边界保护不变。直接把 shard 拉的 P-block 映射到 D 侧 kernel block id；`scale==1`+`r_blk==1` 退化为原 logical 切片行为。
> 3. ✅ CP attention 段（hixl:2328，对齐 MC:3016-3042）：从原 logical 切片改为 `_expand_block_ids(remote_logical, remote_scale)` + `_local_kernel_ids_for_shard(...)`；MambaSpec 分支不变。
> 4. ✅ CP 分支接入 `group_kernel_params = self._get_group_kernel_params(remote_block_size)`（hixl:2214 后）；`assert r_blk==1 or use_mla`（hixl:2220）保留（`scale>1` 必 `use_mla`，已放行）。
>
> **附带修正**：原 hixl CP 分支用 `local_block_ids[first_d : first_d + len(remote_logical)]`，而 `meta.local_block_ids` 是 **external-only**（`hixl:2256` assert：`len == num_external_blocks/(pcp*dcp)`；`update_state_after_alloc` `hixl:1265` 取 `get_unhashed_block_ids_all_groups`），`first_d` 是全局 D block id，作 external-only 列表切片起点会**越界→返回空→`num_blocks=0`→不传 block**。即原 hixl 在 CP + D 端有 prefix cache 时静默传空（一直坏，无 prefix 时 `first_d==0` 退化正常故未暴露）。改用 `_local_kernel_ids_for_shard` 后 `d_block_local_idx` 是 external-only 列表内的相对偏移，正确对齐 mooncake。退化场景（无 prefix）下与原行为一致，无回归。

### 5.3 G9：`r_blk>1`（无 MLA）

依赖 G1（r_blk>1 需 scale>1 使 kernel_size 整除 Bp，mooncake `:2362` assert）。无 MLA 下 mooncake 也不支持（产出空）。**不单独实现**，随 G1。G1 已完成但 G9 仍受限：`_get_group_kernel_params` 的 `assert remote_block_size % kernel_size == 0`（hixl:2467）在 `scale==1`+`Bd>Bp` 时 `Bp%Bd≠0` 会 fail fast（与 mooncake `:2602` 一致），即 V3.2 MLA + CP + Bd>Bp 仍不支持，需 DeepseekV4 `scale>1` 才能使 `kernel_size=Bd/scale` 整除 Bp。

### 5.4 G10：`enable_sfa_dcp_replicated_indexer`（G6 子项）

随 G6。

---

## 6. 优先级与落地序

| 批次 | 子项 | 依赖 | 说明 |
|---|---|---|---|
| 1 | ✅ G2（conv_padding） | ~~需先确认 Qwen3.6 MTP 是否共享 mamba conv state~~ 已确认纯 attention | ~~Qwen3.6 潜在卡点~~ → 含重构 mamba 注册（base mamba P/D 原未跑通）；**2026-07-31 Qwen3.6-27B 端到端实测通过**（含 Bug 8 多组唯一 model_id 修复）；R3/R4 未覆盖 |
| 2 | ✅ G3（NZ） | 独立 | NPU NZ 排布，MLA D-node，静态验证（无 NPU） |
| 3 | ✅ G1 + G7（MLA/compress + 握手） | 独立 | DeepSeek-V3，最大缺口 — **已完成**（G7 评估无需 import） |
| 4 | G4（SWA） | 独立 | 滑窗模型 |
| 5 | G5（sparse） | 独立 | sparse 模型 |
| 6 | G6 + G10（SFA） | 独立 | SFA 场景 |
| 7 | ✅ G8 + G9（CP+MLA / r_blk>1） | 依赖 G1 | MLA + CP 联合 — **G8 已完成**；G9 仍受 `scale==1`+`Bd>Bp` 限制（见 §5.3） |

**建议**：G2 先做（Qwen3.6 潜在卡点，且需先查 Qwen3.6 MTP 架构确认）；G3 次之（NZ 是 NPU 常见）；G1 是最大工程（MLA 全链路）。

---

## 7. 验收

每子项与 `MooncakeConnectorV1` 同 P/D 几何**逐位对齐**：
- ✅ G1：DeepSeek-V3 MLA，`block_size_scale>1`，KV 逐位对齐 — **代码已对齐 mooncake，待 NPU 实测**（注意 No-CP `remote_start_idx` 在 DSV4+prefix cache 的潜在错位，见 §3.1「未消化风险」）
- ✅ G2：Mamba+MTP，D 侧草稿层 conv state 逐字节对齐 — **2026-07-31 端到端实测通过**（Qwen3.6-27B attention+mamba+MTP PD 跑通，pull 成功、MTP acceptance 2.60、external hit 66.7%）。含 Bug 8 修复：多组唯一 `model_id`（见 §3.2「2026-07-31 实测修正」）。R3（`prefill_tp>decode_tp`）/R4（mamba-MTP 共享 conv_state）仍待覆盖
- ✅ G3：NZ 排布，D 侧 cache 布局与 mooncake 一致 — **代码已完成**（fork `_reformat_kv_cache_nz`+`_nz_kv_cache`，TP=1+NZ 触发），静态验证，待 NPU 实测
- G4：SWA 模型，D 侧只收窗口内 block
- G5：sparse 模型，rank 选择一致
- G6：SFA + DCP + prefix cache，K 块位置一致
- ✅ G7：评估无需实现（`compress_ratio` 已握手够用）
- ✅ G8：CP+MLA+compress 联合 — **代码已对齐 mooncake**（含修正原 CP `first_d` 切片 bug，见 §5.2），待 NPU 实测
- G9：随 G1（仍受 `scale==1`+`Bd>Bp` 限制，见 §5.3）
- G10：随 G6

> 注：本设计基于 mooncake 代码静态分析与 HIXL block 寻址语义推导，未在 NPU 上实测。行号为 `vllm-ascend-v0.23.0` 当前状态，实施前需复核。MLA/compress（G1）是最大工程，建议单独出详细实施计划（类比 phase3-plan）。
