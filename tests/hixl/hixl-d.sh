#!/bin/bash
# ============================================================================
# HIXLConnectorV1 Phase 1 测试 - Decode 节点（kv_consumer）
# ----------------------------------------------------------------------------
# 最小配置：单 D、TP=1/PP=1/DP=1、单 HIXLConnectorV1（不组合）、
# 标准 FullAttention 模型。
#
# 启动顺序：确认 hixl-p.sh 已启动且 P 端 ZMQ ROUTER 已监听后再启本脚本。
#
# 验证点（D 端 log）：
#   1. "HixlDataDist initialized: role=DECODER cluster_id=2000 ..."
#   2. "HIXL linked remote_cluster_id=1000 (<P_IP>:<P_listen_port>)"  ← ensure_linked 成功
#   3. "HIXL pull ok. request=... group=... remote_cluster=1000 ..."  ← pull_blocks 成功
#   4. 输出 token 与同配置 MooncakeConnectorV1 一致（正确性验收）
# ============================================================================

# ======================== 需修改：环境相关 ========================
export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$ASCEND_CUSTOM_OPP_PATH

MASTER_IP_ADDRESS="x.x.x.225"   # TODO: 改成 D 节点 IP
IP_ADDRESS="x.x.x.225"          # TODO: 改成 D 节点 IP（本机）
SERVICE_PORT=8801

# 卡：TP=1 用单卡
NETWORK_INTERFACE="eth0"
export ASCEND_RT_VISIBLE_DEVICES=7

# 模型：须与 P 端一致（同路径/同配置）
MODEL_PATH="/data/models/Qwen2.5-3B-Instruct"
MODEL_NAME="Qwen2.5-3B"

# Phase 1 并行维度（硬约束：TP=1/PP=1，建议 DP=1）
PREFILL_DATA_PARALLEL_SIZE=1
PREFILL_TENSOR_PARALLEL_SIZE=1
DECODE_DATA_PARALLEL_SIZE=1
DECODE_TENSOR_PARALLEL_SIZE=1

LIBJEMALLOC_SO_PATH="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
GPU_MEMORY_USE=0.9
MAX_BATCHED_TOKENS=64   # decode 用小 batched tokens

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
vllm serve "$MODEL_PATH" \
  --host "$IP_ADDRESS" \
  --port "$SERVICE_PORT" \
  --data-parallel-size "$DECODE_DATA_PARALLEL_SIZE" \
  --tensor-parallel-size "$DECODE_TENSOR_PARALLEL_SIZE" \
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
  "kv_role": "kv_consumer",
  "kv_port": "21299",
  "kv_connector_extra_config": {
    "hixl": {
      "cluster_id_base": 2000,
      "model_id": 0,
      "link_timeout_ms": 5000
    },
    "prefill": {"dp_size": 1, "tp_size": 1},
    "decode":   {"dp_size": 1, "tp_size": 1}
  }
}' \
  > /tmp/hixl-d.log 2>&1 &

echo "Decode (kv_consumer) started, log: /tmp/hixl-d.log, pid: $!"
