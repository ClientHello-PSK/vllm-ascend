# HIXL Connector 调试 Bug 记录

> 记录 HIXL connector 在 v0.23.0（及 dss v0.24.0）部署调试中遇到的 bug 及修复。
> 日期：2026-07-22 ~ 2026-07-23

---

## 运行时报错（已解决）

### Bug 1: `get_physical_gpu_ids_for_local_dp_rank` 缺失（dss v0.24.0 环境）

**错误**：
```
AttributeError: module 'vllm.v1.engine.utils' has no attribute 'get_physical_gpu_ids_for_local_dp_rank'
```

**根因**：vllm-ascend-dss（锁定 vllm v0.24.0）的 `patch/platform/patch_dp_device_ids.py` 要 monkey-patch `get_physical_gpu_ids_for_local_dp_rank`（vllm **v0.24.0+ 才有**的函数），但环境实际装的 vllm 是 v0.23.0（< 0.24.0，没有这个函数），patch 一加载就 AttributeError。

**解决**：把 HIXL connector 移植到 `vllm-ascend-v0.23.0`（对应 vllm v0.23.0）。v0.23.0 的 vllm-ascend **没有 `patch_dp_device_ids.py`**（不调这个函数），规避了该错误。

**影响**：促使从 dss 移植到 v0.23.0（3 个文件：`hixl_connector.py` + `hixl_datadist.py` + `__init__.py` 注册）。

---

### Bug 2: `data_parallel_external_lb` 校验失败

**错误**：
```
pydantic_core.ValidationError: 1 validation error for ParallelConfig
  Value error, data_parallel_external_lb can only be set when data_parallel_size > 1
```

**根因**：`hixl-p.sh` / `hixl-d.sh` 里的 `--data-parallel-rank 0` 让 vllm 进入**外置 DP 模式**（认为有外部 coordinator 管 DP rank），自动开 `data_parallel_external_lb`。但 `external_lb` 要求 `DP > 1`，而配置是 `--data-parallel-size 1`（DP=1），校验失败。

**解决**：去掉 `--data-parallel-rank 0`。DP=1 单进程不需要外置 DP rank —— PD 分离靠 **kv-transfer（KV 传输）+ router（请求转发）**，跟 data-parallel 外置无关。

**影响文件**：`hixl-p.sh` / `hixl-d.sh`（v0.23.0 + dss，各去掉 `--data-parallel-rank 0` 一行）。

---

### Bug 3: `torch.dtype` 序列化失败

**错误**：
```
TypeError: Encoding objects of type torch.dtype is unsupported
  File "hixl_connector.py", line 1099, in _build_kv_group2layeridx
    serialized_spec = msgspec.to_builtins(spec)
```

**根因**：`_build_kv_group2layeridx` 用 `msgspec.to_builtins(spec)` 直接序列化 `kv_cache_spec`，但 spec 对象含 `torch.dtype` 字段，**msgspec 不支持序列化 torch.dtype**。这是 Phase 1 简化时的疏忽——mooncake 原版有 `to_msgpackable` 包装，我漏了。

**解决**：在 `_build_kv_group2layeridx` 里加 `to_msgpackable` helper（fork mooncake 的 `_serialize_kv_group_spec`）：
- 递归处理 dict / list / 标量
- 遇 `TypeError`（如 torch.dtype）→ `repr(value)` 转字符串，不崩
- 用 `to_msgpackable(spec)` 替代裸 `msgspec.to_builtins(spec)`

**影响文件**：`hixl_connector.py`（v0.23.0 + dss），`_build_kv_group2layeridx` 方法（加 helper + 改 1 行调用）。

---

### Bug 4: HIXL `listen_port` 与 ZMQ `handshake_port` 同端口冲突

**错误**：
```
zmq.error.ZMQError: Address already in use (addr='tcp://7.246.80.223:21299')
  HIXL KVCacheSendingThread exception ... Error: Address already in use
```

**根因**：HIXL `LLMDataDist` 的 `listen_port` 和 ZMQ `KVCacheSendingThread` 的 `handshake_port` 都算成 `kv_port + offset`（**同一个端口**）。P 机上 HIXL 引擎先 listen 21299，ZMQ ROUTER 再 bind 21299 → 冲突。**换 kv_port 没用**（两者永远同号，只是换冲突的端口号）。

设计文档 §5 算式注释写"listen_port 可复用 handshake_port"是错的——两个不同服务（HIXL 引擎 listen vs ZMQ ROUTER bind）不能同端口。mooncake 没这问题（mooncake 无 listen_port，只有 ZMQ handshake_port + TransferEngine te_rpc_port 分开）。

**解决**：`listen_port_base` 默认改成 `kv_port + 10000`（`_extra_options` setdefault + `_compute_identity` default，2 处）。这样：
- ZMQ handshake_port = `kv_port`(21299) + offset = 21299
- HIXL listen_port = `kv_port + 10000`(31299) + offset = 31299

两者差 10000，不冲突。

**影响文件**：`hixl_connector.py`（v0.23.0 + dss），`_extra_options` + `_compute_identity`（2 处 `listen_port_base` default 改 `+ 10000`）。

---

### Bug 5: P/D KV cache shape 不一致 → `pull_blocks` 报 `LLM_PARAM_INVALID`

**错误**：
```
llm_datadist.status.LLMException: [pull_blocks] failed, error code is LLMStatusCode.LLM_PARAM_INVALID,
  src_cache_key = BlocksCacheKey(cluster_id=1000, model_id=0).
INFO:     ... "POST /v1/completions HTTP/1.1" 500 Internal Server Error
```

**根因**：P/D 各自按 `gpu_memory_utilization=0.9` 动态算 KV 显存，两卡初始 free 显存略有差异（P 48.73 / D 49.05 GiB），落到 num_blocks 不等（P 11087 / D 11161）。`register_blocks_cache` 注册的 tensor shape 第一维即 num_blocks（`[num_blocks, block_size, num_kv_heads, head_size]`），两端 cache 经 `BlocksCacheKey(cluster_id, model_id)` 关联，`pull_blocks` 要求两端注册 shape 完全一致——num_blocks 不等就 `LLM_PARAM_INVALID`。注意这是**池级 num_blocks**（KV 池总容量，启动时定死），与**请求级 block 数**（=cdiv(tokens, block_size)，按 token 天然一致、无需对齐）是两回事。

**解决**：两端启动都加 `--kv-cache-memory-bytes 52162245120`（48.58 GiB），强制 KV 显存一致 → num_blocks 都 = 11054。等价做法：`--num-gpu-blocks-override <min(两端)>`。

**影响**：`hixl-p.sh` / `hixl-d.sh`（加 `--kv-cache-memory-bytes`）。无代码改动。

**为何 Mooncake 无此问题**：Mooncake 注册的是原始内存段（ptr+len），按 block id 逐段 RDMA，只校验**单块字节数**（`kv_block_len`），与 num_blocks 无关；HIXL 注册的是带完整 shape 的 blocks_cache tensor，shape 必须两端一致。

---

### Bug 6: D 端 cache `remote_accessible=False` → `pull_blocks` 仍报 `LLM_PARAM_INVALID`（代码级）

**错误**：shape 对齐（11054==11054）后，D 端 `pull_blocks` 仍报同样的 `LLM_PARAM_INVALID`，三次 500。

**根因**（代码级）：`register_kv_caches` 把两端 cache 按 `remote_accessible=(kv_role=="kv_producer")` 注册，即 D(kv_consumer)→False。但 llm_datadist 的 pull 路径会走到 `PullCacheByGet`，该校验链要求**本地 dst cache 也必须 `remote_accessible=True`**：

1. `LlmDataDistImpl::Initialize` **无条件** `EnableRemoteCacheAccessible=1`（`llm_datadist_impl.cc:214`，无配置可关）→ `access_remote_cache_=true`。
2. `DataCacheEngine::PullCache` 命中 `if (access_remote_cache_)` → 走 `PullCacheByGet`（`data_cache_engine.cc:153-155`）。
3. `PullCacheByGet` 双向校验（`data_transfer_client.cc:230-234`）：远端 P cache（P 本就 True，过）+ **本地 D dst cache（False → 命中 `LLM_PARAM_INVALID "local cache is not remote accessible"`）**。

> 排除"找不到 P 远端 cache"：那种情况 `FindCacheEntry` 返回 `LLM_KV_CACHE_NOT_EXIST`（`cache_access_table.cc:294,303`），不是 `LLM_PARAM_INVALID`。日志报 PARAM_INVALID 且 P 是 True，故唯一命中点是本地 D 的 `remote_accessible=False`。

**解决**：`register_kv_caches` 里两端都按 `remote_accessible=True` 注册（D 也 True）。不违反 `register_blocks_cache` "link 之后不能再注册 remote_accessible=True" 的约束——D 是先注册后 link。

**验证**：修复后 D 端请求 200，`External prefix cache hit rate: 100.0%`（KV 成功从 P 拉到 D）。

**附带修复**：register 日志原来打印的是旧表达式 `(kv_role=="kv_producer")` 而非实际注册值，导致 D 实际已 True 但日志仍显示 False（一度误导排查）。改为局部变量 `remote_accessible=True`，注册与日志共用同一值。

**影响**：`hixl_connector.py`（v0.23.0）`register_kv_caches`（`remote_accessible=True` + 日志改打印真实值）。dss 副本未改（同 bug，按需同步）。

**与 Bug 5 的关系**：两 bug 独立叠加。shape 对齐后 `FindCacheEntry` 才能匹配上 P 的 cache，之后才轮到 `remote_accessible` 校验；两道都过，pull 才真正跑通。所以 Bug 5 修复后错误码不变（仍 PARAM_INVALID），一度误判 shape 修复无效，实则是暴露了第二道校验。

---

## 开发自查修正（编码时发现，未触发运行报错）

这些是写 `hixl_connector.py` 时自查发现并修复的，没等到运行报错：

| 修正点 | 说明 |
|---|---|
| `device_id` 计算 | `__init__` 的 device_id 稳健化（`_current_npu_device_id` helper，处理 `int` / `torch.device` 两种返回） |
| `shutdown` 方法 | 补 `shutdown`（调 `shutdown_datadist`，unlink + finalize 资源清理） |
| K/V shape 一致 assert | `register_kv_caches` 加 assert（同 group K/V tensor shape 一致，不一致早报错） |
| `block_size_scale==1` assert | Phase 1 约束（标准 FullAttention，非 compress/SWA） |
| imports 清理 | 去未用 import（`get_pp_indices` / `FullAttentionSpec` / `ascend_envs` / `enable_custom_op`） |

---

## 已知限制 / 后续路径（2026-07-23 复核状态）

Phase 1（TP=1 + 标准 FullAttention + PP=1）全链路已跑通。下表为历史"待验证"项的最终状态：

| 项 | 状态 | 说明 |
|---|---|---|
| `register_blocks_cache` 多 tensor 独立注册 | ✅ 已验证 | Bug 6 修复后 pull 成功（mooncake 走合并 region，HIXL 每 tensor 独立 addr，已验证可行） |
| `pull_blocks` 整块写入字节布局 | ✅ 已验证 | `External prefix cache hit rate: 100.0%`，输出正确（TP=1 + 非 NZ 下 no-op reformat 成立） |
| `ensure_linked` 时序 | ✅ 已验证 | D 单边 link 到 P 即可 pull，无需 P 反向 link |
| `llm_datadist` 库依赖 | ⚪ 非问题 | 仍懒加载 import（`hixl_connector.py:84/638/1155`），属部署前置条件，现环境（CANN HIXL）已满足，无需改 |
| TP>1 / PP>1 / compress(SWA) / NZ / Mamba | 🔴 设计性限制 | Phase 1 未覆盖，**有硬 assert 把关**，遇到会直接报错而非静默错传 |

### 仍存在的限制（第 5 项展开）

只要保持 TP=1 / PP=1 / 标准 FullAttention（Qwen2.5 这类模型）就不会碰到。以下场景需实现 Phase 2/3 才支持：

| 场景 | 触发点（assert） | 计划 |
|---|---|---|
| TP>1 | `hixl_connector.py:1016` `assert self.tp_size == 1`；`:645` `assert num_group_pulls == 1`（需 staging） | Phase 2 |
| PP>1 | `hixl_connector.py:1017` `assert self.pp_size == 1`（需 layer_range） | Phase 2 |
| compress / SWA 等 block_size_scale≠1 | `hixl_connector.py:1195` `assert scale == 1` | Phase 2 |
| GQA/NZ reformat、Mamba/NZ 路径 | `hixl_connector.py:688-690`（当前 no-op，Phase 3 做 Mamba/NZ） | Phase 3 |

**当前不阻塞使用。** 遇到新报错，按前文格式追加到本文档。
