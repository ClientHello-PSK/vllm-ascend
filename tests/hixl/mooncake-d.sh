#!/bin/bash
# ============================================================================
# MooncakeConnectorV1 基线测试 - Decode 节点（kv_consumer）
# ----------------------------------------------------------------------------
# 作为 HIXL 的对比基线：与 hixl-d.sh 严格对齐（同 Qwen2.5-3B / TP=1 / DP=1）。
#
# ⚠️ mooncake 必需（同 mooncake-p.sh）：MOONCAKE_CONFIG_PATH + mooncake 库 +
#    mooncake.json，缺一启动就崩。
#
# 启动顺序：确认 mooncake-p.sh 已启动且 P 端 ZMQ ROUTER 已监听后再启本脚本。
# 验证：输出 token 与 hixl-d.sh 一致（HIXL 正确性验收的基准）。
# ============================================================================

# ======================== 需修改：环境相关 ========================
export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$ASCEND_CUSTOM_OPP_PATH

MASTER_IP_ADDRESS="x.x.x.225"   # TODO: 改成 D 节点 IP
IP_ADDRESS="x.x.x.225"          # TODO: 改成 D 节点 IP（本机）
SERVICE_PORT=8801

# ---- mooncake 必需（HIXL 不需要这些，别漏）----
export MOONCAKE_CONFIG_PATH=./mooncake-d.json
# TODO: 改成你环境的 mooncake 库路径（参考原 d0-225.sh，cann 版本号按实际）
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/python/site-packages/mooncake:$LD_LIBRARY_PATH

NETWORK_INTERFACE="eth0"
export ASCEND_RT_VISIBLE_DEVICES=0

# 模型：须与 P 端一致（同 hixl-d.sh）
MODEL_PATH="/data/models/Qwen2.5-3B-Instruct"
MODEL_NAME="Qwen2.5-3B"

# 并行维度：与 hixl-d.sh 一致（TP=1/DP=1）
PREFILL_DATA_PARALLEL_SIZE=1
PREFILL_TENSOR_PARALLEL_SIZE=1
DECODE_DATA_PARALLEL_SIZE=1
DECODE_TENSOR_PARALLEL_SIZE=1

LIBJEMALLOC_SO_PATH="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
GPU_MEMORY_USE=0.9
MAX_BATCHED_TOKENS=64   # decode 用小 batched tokens

# ======================== 通用环境变量 ========================
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
  --data-parallel-rank 0 \
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
  "kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_consumer",
  "kv_port": "20102",
  "kv_connector_extra_config": {
    "use_ascend_direct": true,
    "prefill": {"dp_size": 1, "tp_size": 1},
    "decode":   {"dp_size": 1, "tp_size": 1}
  }
}' \
  > /tmp/mooncake-d.log 2>&1 &

echo "Decode (MooncakeConnectorV1 kv_consumer) started, log: /tmp/mooncake-d.log, pid: $!"
