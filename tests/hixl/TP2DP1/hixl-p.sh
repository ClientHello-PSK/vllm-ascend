#!/bin/bash
# ============================================================================
# HIXLConnectorV1 Phase 2 测试 - Prefill 节点（kv_producer），TP=2
# ----------------------------------------------------------------------------
# 相对 TP1DP1 的改动：
#   - ASCEND_RT_VISIBLE_DEVICES 由单卡改为 2 卡
#   - PREFILL/DECODE_TENSOR_PARALLEL_SIZE 1 -> 2
#   - kv_connector_extra_config.prefill/decode.tp_size 1 -> 2
# Phase 2 已放开 hixl_connector.py:1016 的 tp_size==1 断言；
# kv_port 不变（21299），HIXL 按 tp_rank 自动加偏移（hixl_connector.py:1322 附近）；
# cluster_id_base 不变（P=1000），多 rank cluster_id 由 base+rank 内部推导。
#
# 启动顺序：先启本脚本，等 P 端 log 出现
#   "HIXL KVCacheSendingThread listening on ..." 后，再启 hixl-d.sh。
#
# 验证点（P 端 log）：
#   1. "HixlDataDist initialized: role=PROMPT cluster_id=1000 ..."
#   2. "HIXL KVCacheSendingThread listening on tcp://<ip>:<handshake_port>"
#   3. 收到 D 的 GET_META / DONE 后正常回 ACK（每个 tp_rank 各一次）
# ============================================================================

# ======================== 需修改：环境相关 ========================
export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$ASCEND_CUSTOM_OPP_PATH

MASTER_IP_ADDRESS="x.x.x.223"   # TODO: 改成 P 节点 IP
IP_ADDRESS="x.x.x.223"          # TODO: 改成 P 节点 IP（本机）
SERVICE_PORT=8800

# [Phase2] TP=2 用 2 张卡
NETWORK_INTERFACE="eth0"
export ASCEND_RT_VISIBLE_DEVICES=6,7

# 模型：须与 D 端一致（同路径/同配置）
# Phase 2 若按 plan §5 降级路径（仅 FullAttention + TP>1，未做 HMA），用标准 attention 模型
MODEL_PATH="/data/models/Qwen2.5-3B-Instruct"
MODEL_NAME="Qwen2.5-3B"

# [Phase2] Phase 2 并行维度：TP 1 -> 2，PP 仍 =1（Phase 3 才放开）
PREFILL_DATA_PARALLEL_SIZE=1
PREFILL_TENSOR_PARALLEL_SIZE=2
DECODE_DATA_PARALLEL_SIZE=1
DECODE_TENSOR_PARALLEL_SIZE=2

LIBJEMALLOC_SO_PATH="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
GPU_MEMORY_USE=0.9
MAX_BATCHED_TOKENS=4096

# ======================== 通用环境变量（无 mooncake）========================
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=512
export HCCL_IF_IP="$IP_ADDRESS"
export GLOO_SOCKET_IFNAME="$NETWORK_INTERFACE"
export TP_SOCKET_IFNAME="$NETWORK_INTERFACE"
export HCCL_SOCKET_IFNAME="$NETWORK_INTERFACE"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=100
export OMP_PROC_BIND=false
export TASK_QUEUE_ENABLE=1
export LD_PRELOAD=$LIBJEMALLOC_SO_PATH:$LD_PRELOAD

# ======================== 启动命令 ========================
# 注意：去掉 --kv-cache-memory-bytes（TP1DP1 写死的总量在 TP=2 下会 OOM），
#       改由 --gpu_memory_utilization 自动划分每卡 KV。
vllm serve "$MODEL_PATH" \
  --host "$IP_ADDRESS" \
  --port "$SERVICE_PORT" \
  --data-parallel-size "$PREFILL_DATA_PARALLEL_SIZE" \
  --tensor-parallel-size "$PREFILL_TENSOR_PARALLEL_SIZE" \
  --served-model-name "$MODEL_NAME" \
  --max_model_len 32768 \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --max-num-seqs 16 \
  --trust-remote-code \
  --gpu_memory_utilization "$GPU_MEMORY_USE" \
  --enforce-eager \
  --kv-transfer-config \
'{
  "kv_connector": "HIXLConnectorV1",
  "kv_role": "kv_producer",
  "kv_port": "21299",
  "kv_connector_extra_config": {
    "hixl": {
      "cluster_id_base": 1000,
      "model_id": 0,
      "link_timeout_ms": 5000
    },
    "prefill": {"dp_size": 1, "tp_size": 2},
    "decode":   {"dp_size": 1, "tp_size": 2}
  }
}' \
  > /tmp/hixl-p.log 2>&1 &

echo "Prefill (kv_producer) started, log: /tmp/hixl-p.log, pid: $!"
