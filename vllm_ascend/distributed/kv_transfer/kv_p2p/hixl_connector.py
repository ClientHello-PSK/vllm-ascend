# SPDX-License-Identifier: Apache-2.0
"""HIXLConnector: direct HIXL LLM-DataDist P2P KV connector (no Mooncake).

Phase 1 minimal implementation: TP=1, single FullAttention group, PP=1.
Forked from MooncakeConnectorV1 with the byte-addressed Mooncake transfer
(repeated register_memory + batch_transfer_sync_read) replaced by block-indexed
HIXL transfer (register_blocks_cache + pull_blocks). Control plane (handshake
/ metadata / done-notify) still uses ZMQ, identical to Mooncake.

See docs/vllm-ascend-hixl-connector-design.md for the full design. Phase 2/3
will add TP>1 staging, HMA multi-group, PP/PCP/DCP and Mamba.
"""
import contextlib
import hashlib
import logging
import math
import queue
import struct
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import msgspec
import numpy as np
import numpy.typing as npt
import torch
import torch_npu
import zmq
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import RequestStatus

from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.distributed.kv_transfer.utils.hixl_datadist import (
    get_datadist,
    shutdown_datadist,
)

# isort: off
if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
# isort: on

GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"

# A busy peer can otherwise keep a global executor worker forever when the
# number of peers is larger than max_workers. Yield after a small FIFO batch so
# other peers already waiting in the global executor queue can make progress.
MAX_REQUESTS_PER_PEER_HANDLER = 5


# ---------------------------------------------------------------------------
# torch dtype -> llm_datadist DataType
# ---------------------------------------------------------------------------
def _torch_dtype_to_llm_dtype(dtype: torch.dtype):
    from llm_datadist import DataType

    if dtype == torch.bfloat16:
        return DataType.DT_BF16
    if dtype == torch.float16:
        return DataType.DT_FLOAT16
    if dtype == torch.float32:
        return DataType.DT_FLOAT
    if dtype == torch.int8:
        return DataType.DT_INT8
    raise ValueError(f"Unsupported kv cache dtype for HIXLConnector: {dtype}")


# ---------------------------------------------------------------------------
# Metadata & dataclasses (engine-agnostic, forked from Mooncake)
# ---------------------------------------------------------------------------
class RemotePortInfo(TypedDict):
    num: int
    host: str


class HixlAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    """Replaces MooncakeAgentMetadata.

    Drops byte-addressed fields (te_rpc_port / kv_caches_base_addr /
    block_lens / block_strides); HIXL routes by cluster_id and addresses by
    block index, so no raw byte arithmetic is carried.
    """

    engine_id: str
    cluster_id: int  # local LLMDataDist cluster_id (D uses it as pull key)
    listen_ip: str  # llm listen ip (D->P link)
    listen_port: int  # llm listen port
    model_id: int = 0
    num_tensors_per_group: list[int] = []  # Cache.num_tensors per group (=layers*2)
    # Retained for block expansion / future reformat:
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]] = {}
    block_size: int = 0
    num_blocks: int = 0
    block_size_scale: list[list[int]] = []
    local_ip: str = ""
    handshake_port: int = 0


@dataclass
class ReqMeta:
    local_block_ids: BlockIds
    num_external_tokens: int
    num_computed_tokens: int
    remote_block_ids: BlockIds

    remote_host: str
    remote_port: int
    remote_engine_id: str
    remote_request_id: str
    remote_ptp_size: int
    num_prompt_blocks: int
    remote_block_size: int


@dataclass(frozen=True)
class GroupPull:
    group_id: int
    remote_tp_offset: int
    num_group_pulls: int
    is_group_transfer_end: bool = False


@dataclass
class SizedDict(OrderedDict):
    def __init__(self, max_size=16000, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            value: dict[int, list[int]] = {}
            self[key] = value
            return value


class KVCacheTaskTracker:
    """Tracks finished / delayed-free requests. Forked unchanged from Mooncake."""

    def __init__(self):
        super().__init__()
        self.done_task_lock = threading.Lock()
        self.finished_requests: set[str] = set()
        # Only used in prefill node. Tracks requests whose kv blocks freeing is
        # intentionally delayed. Force-freed after VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT.
        self.delayed_free_requests: OrderedDict[str, float] = OrderedDict()
        self.reqs_to_process: set[str] = set()

    def add_req_to_process(self, request_id: str):
        self.reqs_to_process.add(request_id)

    def add_not_transfer_request(self, request_id: str):
        with self.done_task_lock:
            self.finished_requests.add(request_id)
            self.reqs_to_process.discard(request_id)

    def update_done_task_count(self, request_id: str):
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.finished_requests.add(request_id)
                self.reqs_to_process.discard(request_id)
                self.delayed_free_requests.pop(request_id, None)
            else:
                logger.warning(
                    "HIXLConnector finish req not in reqs to process. request_id=%s. ",
                    request_id,
                )

    def get_and_clear_finished_requests(self) -> set[str]:
        with self.done_task_lock:
            finished_requests = self.finished_requests.copy()
            expired_requests = self._retrieve_expired_requests()
            finished_requests.update(expired_requests)
            self.finished_requests.clear()
        return finished_requests

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.delayed_free_requests[request_id] = delay_start_time

    def _retrieve_expired_requests(self):
        expired_requests: set[str] = set()
        current_time = time.time()
        while self.delayed_free_requests:
            request_id = next(iter(self.delayed_free_requests))
            delay_start_time = self.delayed_free_requests[request_id]
            if current_time - delay_start_time > envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
                self.delayed_free_requests.popitem(last=False)
                self.reqs_to_process.discard(request_id)
                expired_requests.add(request_id)
                logger.error(
                    "Force freed expired request: %s. Reason: exceeded timeout (%s seconds).",
                    request_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
            else:
                break
        return expired_requests


# ---------------------------------------------------------------------------
# KVCacheSendingThread (P side, only answers ZMQ handshake). Forked; only the
# metadata type changes (MooncakeAgentMetadata -> HixlAgentMetadata).
# ---------------------------------------------------------------------------
class KVCacheSendingThread(threading.Thread):
    def __init__(
        self,
        vllm_config: VllmConfig,
        tp_rank: int,
        prefill_tp_size: int,
        local_engine_id: str,
        side_channel_host: str,
        side_channel_port: int,
        metadata: HixlAgentMetadata,
        ready_event: threading.Event,
        kv_caches: dict[str, Any],
        pcp_rank: int,
    ):
        super().__init__(daemon=True, name="HIXLKCacheSendingThread")
        self.tp_rank = tp_rank
        self.prefill_tp_size = prefill_tp_size
        self.pp_rank = get_pp_group().rank_in_group
        self.pcp_size = 1  # Phase 1: no PCP
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.local_engine_id = local_engine_id
        self.side_channel_host = side_channel_host
        self.side_channel_port = side_channel_port
        self.metadata = metadata
        self.ready_event = ready_event
        self.kv_caches = kv_caches
        self.pcp_rank = pcp_rank
        self.port_send_num: dict[str, int] = {}
        self.task_tracker = KVCacheTaskTracker()

    def get_and_clear_finished_requests(self) -> set[str]:
        return self.task_tracker.get_and_clear_finished_requests()

    def add_not_transfer_request(self, request_id: str):
        self.task_tracker.add_not_transfer_request(request_id)

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        return self.task_tracker.add_delayed_request(request_id, delay_start_time)

    def run(self):
        try:
            device_index = (self.pp_rank * self.pcp_size + self.pcp_rank) * self.tp_size + self.tp_rank
            handshake_port = self.side_channel_port + device_index
            path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
            logger.info(
                "HIXL KVCacheSendingThread listening on %s. tp_rank=%d pp_rank=%d",
                path, self.tp_rank, self.pp_rank,
            )
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                self.ready_event.set()
                self.run_busy_loop(sock)
        except Exception as e:
            logger.exception(
                "HIXL KVCacheSendingThread exception. tp_rank=%d path=%s. Error: %s",
                self.tp_rank, path, e,
            )

    def run_busy_loop(self, sock: zmq.Socket):  # type: ignore
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(self.metadata)
        decoder = msgspec.msgpack.Decoder(type=tuple)
        while True:
            try:
                frames = sock.recv_multipart()
                if len(frames) < 2:
                    logger.error("Invalid message in KVCacheSendingThread: %d frames", len(frames))
                    continue
                identity = frames[0]
                payload = [f for f in frames[1:] if f != b""]
                if len(payload) != 1:
                    logger.error("Invalid payload in KVCacheSendingThread: %d frames", len(payload))
                    continue
                msg = decoder.decode(payload[0])
                if msg[0] == GET_META_MSG:
                    sock.send_multipart((identity, b"", encoded_data))
                elif msg[0] == DONE_RECVING_MSG:
                    request_id = msg[1]
                    remote_port_send_num = msg[2]
                    if remote_port_send_num:
                        if request_id not in self.port_send_num:
                            self.port_send_num[request_id] = 0
                        self.port_send_num[request_id] += 1
                        device_index = (self.pp_rank * self.pcp_size + self.pcp_rank) * self.tp_size + self.tp_rank
                        handshake_port = self.side_channel_port + device_index
                        if self.port_send_num[request_id] >= remote_port_send_num[handshake_port]["num"]:
                            self.task_tracker.update_done_task_count(request_id)
                            del self.port_send_num[request_id]
                    else:
                        self.task_tracker.update_done_task_count(request_id)
                    while True:
                        try:
                            sock.send_multipart((identity, b"", b"ACK"), flags=zmq.NOBLOCK)  # type: ignore
                            break
                        except zmq.Again:  # type: ignore
                            time.sleep(0.01)
                else:
                    logger.error("Unexpected message type in KVCacheSendingThread: %s", msg[0])
            except Exception as e:
                logger.error("KVCacheSendingThread handler exception: %s", e)


# ---------------------------------------------------------------------------
# KVCacheRecvingThread (D side). _get_remote_metadata and
# _transfer_kv_cache_all_groups are rewritten for HIXL; the request-queue /
# peer-fairness / done-signal / socket-pool skeleton is forked from Mooncake.
# ---------------------------------------------------------------------------
class KVCacheRecvingThread(threading.Thread):
    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        hixl,  # HixlDataDist
        model_id: int,
        local_engine_id: str,
        local_handshake_port: int,
        side_channel_port: int,
        vllm_config: VllmConfig,
        kv_caches: dict[str, Any],
        kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]],
        group_caches: dict[int, Any],  # kv_cache_group_id -> registered Cache
        block_size_scale: list[list[int]] | None = None,
        ready_event: threading.Event | None = None,
    ):
        super().__init__(daemon=True, name="HIXLKCacheRecvingThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.local_engine_id = local_engine_id
        self.local_handshake_port = local_handshake_port
        self.side_channel_port = side_channel_port
        self.hixl = hixl
        self.cache_manager = hixl.cache_manager
        self.model_id = model_id
        if ready_event is None:
            ready_event = threading.Event()
        self.ready_event = ready_event

        self.kv_caches = kv_caches
        self.kv_group2layeridx = kv_group2layeridx
        self.group_caches = group_caches  # kv_cache_group_id -> Cache
        self.block_size_scale = block_size_scale or []

        # remote_cluster_id[engine_id][handshake_port] = P cluster_id
        self.remote_cluster_id: dict[str, dict[int, int]] = SizedDict()
        self.remote_kv_group2layeridx: dict[str, dict[int, dict[int, tuple[dict[str, Any], list[int]]]]] = SizedDict()
        self.remote_metadata_lock = threading.Lock()

        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.peer_request_queues: defaultdict[tuple[str, int], deque[dict[str, Any]]] = defaultdict(deque)
        self.active_peer_request_handlers: set[tuple[str, int]] = set()
        self.peer_request_queues_lock = threading.Lock()
        self.request_task_counts: defaultdict[str, int] = defaultdict(int)
        self.finished_request_markers: set[str] = set()
        self.request_task_counts_lock = threading.Lock()

        self.task_tracker = KVCacheTaskTracker()

        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(HixlAgentMetadata)
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[str, deque[zmq.Socket]] = defaultdict(deque)  # type: ignore
        self.timeout = 1.0

        assert vllm_config is not None
        self.vllm_config: VllmConfig = vllm_config
        self.block_size = self.vllm_config.cache_config.block_size

        self.proc_not_transfer_request: dict[str, bool] = {}
        self.proc_not_transfer_request_lock = threading.Lock()
        self.failed_recv_requests: set[str] = set()
        self.invalid_block_ids: set[int] = set()
        self.failed_recv_requests_lock = threading.Lock()

    def add_request(
        self,
        request_id: str,
        remote_request_id: str,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        group_pulls: list[GroupPull],
        remote_engine_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_port_send_num: dict[int, RemotePortInfo] | None = None,
        num_computed_tokens: int = 0,
        all_task_done: bool = False,
    ):
        if remote_port_send_num is None:
            remote_port_send_num = {}
        trans_info = {
            "request_id": request_id,
            "local_block_ids": local_block_ids,
            "remote_block_ids": remote_block_ids,
            "group_pulls": group_pulls,
            "remote_engine_id": remote_engine_id,
            "remote_request_id": remote_request_id,
            "remote_host": remote_host,
            "remote_handshake_port": remote_handshake_port,
            "num_computed_tokens": num_computed_tokens,
            "remote_port_send_num": remote_port_send_num,
            "all_task_done": all_task_done,
        }
        self.request_queue.put(trans_info)

    def get_and_clear_finished_requests(self) -> set[str]:
        return self.task_tracker.get_and_clear_finished_requests()

    def get_and_clear_invalid_block_ids(self) -> set[int]:
        with self.failed_recv_requests_lock:
            invalid_block_ids = self.invalid_block_ids
            self.invalid_block_ids = set()
        return invalid_block_ids

    def _is_failed_recv_request(self, request_id: str) -> bool:
        with self.failed_recv_requests_lock:
            return request_id in self.failed_recv_requests

    def _mark_failed_recv_request(self, request_id: str, local_block_ids: BlockIds) -> None:
        with self.failed_recv_requests_lock:
            self.failed_recv_requests.add(request_id)
            self.invalid_block_ids.update(local_block_ids[0])

    def _clear_failed_recv_request(self, request_id: str) -> None:
        with self.failed_recv_requests_lock:
            self.failed_recv_requests.discard(request_id)

    def run(self):
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    self.request_queue.task_done()
                    continue
                self._submit_request(request_data)
            except Exception as e:
                logger.error("Error in HIXL KVCacheRecvingThread. error=%s. ", e)

    def _submit_request(self, request_data: dict[str, Any]) -> None:
        peer_key = (request_data["remote_host"], request_data["remote_handshake_port"])
        self._mark_request_task_submitted(request_data)
        should_start_worker = False
        with self.peer_request_queues_lock:
            self.peer_request_queues[peer_key].append(request_data)
            if peer_key not in self.active_peer_request_handlers:
                self.active_peer_request_handlers.add(peer_key)
                should_start_worker = True
        if should_start_worker:
            self.executor.submit(self._handle_peer_requests, peer_key)

    def _handle_peer_requests(self, peer_key: tuple[str, int]) -> None:
        requests_handled = 0
        while requests_handled < MAX_REQUESTS_PER_PEER_HANDLER:
            with self.peer_request_queues_lock:
                peer_queue = self.peer_request_queues.get(peer_key)
                if not peer_queue:
                    self.peer_request_queues.pop(peer_key, None)
                    self.active_peer_request_handlers.discard(peer_key)
                    return
                req_meta = peer_queue.popleft()
            requests_handled += 1
            try:
                self._handle_request(req_meta)
            except Exception:
                logger.exception("Error handling HIXL KV transfer for peer %s:%d.", peer_key[0], peer_key[1])

        should_resubmit = False
        with self.peer_request_queues_lock:
            peer_queue = self.peer_request_queues.get(peer_key)
            if peer_queue:
                should_resubmit = True
            else:
                self.peer_request_queues.pop(peer_key, None)
                self.active_peer_request_handlers.discard(peer_key)
        if should_resubmit:
            self.executor.submit(self._handle_peer_requests, peer_key)

    def _mark_request_task_submitted(self, req_meta: dict[str, Any]) -> None:
        request_id = req_meta["request_id"]
        with self.request_task_counts_lock:
            self.request_task_counts[request_id] += 1
            if req_meta["all_task_done"]:
                self.finished_request_markers.add(request_id)

    def _mark_request_task_done(self, request_id: str, all_task_done: bool) -> bool:
        with self.request_task_counts_lock:
            pending_count = self.request_task_counts.get(request_id)
            if pending_count is None:
                return all_task_done
            pending_count -= 1
            if pending_count > 0:
                self.request_task_counts[request_id] = pending_count
                return False
            self.request_task_counts.pop(request_id, None)
            has_finished_marker = request_id in self.finished_request_markers
            self.finished_request_markers.discard(request_id)
            return has_finished_marker

    def _handle_request(self, req_meta: dict[str, Any]):
        request_id = req_meta["request_id"]
        remote_request_id = req_meta["remote_request_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        remote_port_send_num = req_meta["remote_port_send_num"]
        all_task_done = req_meta["all_task_done"]
        transfer_failed = self._is_failed_recv_request(request_id)
        try:
            if transfer_failed:
                self._mark_failed_recv_request(request_id, req_meta["local_block_ids"])
                logger.warning("Skipping HIXL KV transfer for request. remote=%s. ", remote_request_id)
            else:
                try:
                    self._transfer_kv_cache_all_groups(req_meta)
                except Exception as e:
                    transfer_failed = True
                    self._mark_failed_recv_request(request_id, req_meta["local_block_ids"])
                    logger.exception("Failed HIXL KV transfer for request %s: %s", remote_request_id, e)
        finally:
            if self._mark_request_task_done(request_id, all_task_done):
                self.task_tracker.update_done_task_count(request_id)
                with self.proc_not_transfer_request_lock:
                    self.proc_not_transfer_request.pop(remote_request_id, None)
                self._clear_failed_recv_request(request_id)
            self.request_queue.task_done()
            self._send_done_recv_signal(remote_request_id, remote_host, remote_handshake_port, remote_port_send_num)

    def _get_remote_metadata(self, remote_host: str, remote_handshake_port: int) -> None:
        """Fetch HixlAgentMetadata over ZMQ and link the remote P cluster.

        Replaces Mooncake's byte-address bookkeeping: instead of caching
        kv_caches_base_addr / te_rpc_port, we cache the remote cluster_id and
        immediately ensure_linked to it (D3)."""
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(sock, self.encoder.encode((GET_META_MSG, "")), f"{remote_host}:{remote_handshake_port}")
            metadata_bytes = ensure_zmq_recv(sock, f"{remote_host}:{remote_handshake_port}")
            agent_meta = self.decoder.decode(metadata_bytes)
            logger.info(
                "HIXL DEBUG D got meta: cluster_id=%s listen=%s:%s num_tensors_per_group=%s",
                agent_meta.cluster_id, agent_meta.listen_ip, agent_meta.listen_port,
                agent_meta.num_tensors_per_group,
            )
            engine_id = agent_meta.engine_id
            assert engine_id != self.local_engine_id, (
                f"Conflict engine id {engine_id} with local {self.local_engine_id}."
            )
            if agent_meta.kv_group2layeridx != self.kv_group2layeridx:
                logger.warning(
                    "Remote kv_group2layeridx inconsistent. remote=%s local=%s",
                    agent_meta.kv_group2layeridx, self.kv_group2layeridx,
                )
            # D3: link the remote P cluster before any pull_blocks to it.
            self.hixl.ensure_linked(
                remote_cluster_id=agent_meta.cluster_id,
                remote_ip=agent_meta.listen_ip,
                remote_port=agent_meta.listen_port,
            )
            with self.remote_metadata_lock:
                self.remote_kv_group2layeridx[engine_id][remote_handshake_port] = agent_meta.kv_group2layeridx
                self.remote_cluster_id[engine_id][remote_handshake_port] = agent_meta.cluster_id
        except Exception:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
            raise
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)

    def _transfer_kv_cache_all_groups(self, req_meta: dict[str, Any]):
        """D-pull KV via HIXL pull_blocks (block-index addressed).

        Replaces Mooncake's src_list/dst_list/length_list byte arithmetic +
        batch_transfer_sync_read. Phase 1: TP=1 (num_group_pulls==1), single
        FullAttention group, PP=1 (no layer_range)."""
        remote_request_id = req_meta["remote_request_id"]
        local_block_ids: BlockIds = req_meta["local_block_ids"]
        remote_block_ids: BlockIds = req_meta["remote_block_ids"]
        group_pulls: list[GroupPull] = req_meta["group_pulls"]
        remote_engine_id = req_meta["remote_engine_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]

        num_local_blocks = sum(len(group_block_ids) for group_block_ids in local_block_ids)
        if num_local_blocks == 0:
            return

        with self.remote_metadata_lock:
            has_remote_metadata = (
                remote_engine_id in self.remote_cluster_id
                and remote_handshake_port in self.remote_cluster_id[remote_engine_id]
            )
        if not has_remote_metadata:
            self._get_remote_metadata(remote_host, remote_handshake_port)

        from llm_datadist import BlocksCacheKey, LLMException

        for group_pull in group_pulls:
            group_idx = group_pull.group_id
            group_spec, _ = self.kv_group2layeridx[group_idx]
            kv_cache_group_id = group_spec.get("kv_cache_group_id", group_idx)
            # Phase 1: TP=1 only.
            assert group_pull.num_group_pulls == 1, (
                "HIXLConnector Phase 1 supports TP=1 only (num_group_pulls>1 needs staging)."
            )
            with self.remote_metadata_lock:
                remote_cluster_id = self.remote_cluster_id[remote_engine_id][remote_handshake_port]
            dst_cache = self.group_caches.get(kv_cache_group_id)
            if dst_cache is None:
                logger.error("No registered HIXL cache for group %s; skip.", kv_cache_group_id)
                continue
            src_blocks = list(remote_block_ids[kv_cache_group_id])
            dst_blocks = list(local_block_ids[kv_cache_group_id])
            if not src_blocks or not dst_blocks:
                continue
            # Chunk by contiguous spans to reduce number of pull calls.
            grouped_remote, grouped_local = group_concurrent_contiguous(src_blocks, dst_blocks)
            logger.info(
                "HIXL DEBUG pull: remote_cluster=%s dst_cache_id=%s dst_is_blocks=%s "
                "dst_shape=%s dst_num_tensors=%s src_blocks=%s dst_blocks=%s",
                remote_cluster_id, dst_cache.cache_id, dst_cache.is_blocks_cache,
                dst_cache.cache_desc.shape, dst_cache.cache_desc.num_tensors,
                src_blocks, dst_blocks,
            )
            num_layers = dst_cache.cache_desc.num_tensors // 2
            for chunk_remote, chunk_local in zip(grouped_remote, grouped_local):
                try:
                    self.cache_manager.pull_blocks(
                        BlocksCacheKey(remote_cluster_id, self.model_id),
                        dst_cache,
                        src_blocks=chunk_remote,
                        dst_blocks=chunk_local,
                        src_layer_range=range(num_layers),
                        dst_layer_range=range(num_layers),
                    )
                except LLMException as e:
                    logger.error(
                        "HIXL pull_blocks failed for request %s group %s: %s",
                        remote_request_id, group_idx, e,
                    )
                    raise
            logger.debug(
                "HIXL pull ok. request=%s group=%s remote_cluster=%s src=%s dst=%s",
                remote_request_id, group_idx, remote_cluster_id, src_blocks, dst_blocks,
            )
        # Phase 1 (TP=1, non-NZ): post-transfer GQA/NZ reformat is a no-op,
        # so it is intentionally omitted here. Phase 2 will add TP>1 staging
        # reformat and Phase 3 the Mamba / NZ paths.

    def _send_done_recv_signal(
        self,
        request_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_port_send_num: dict[int, RemotePortInfo],
    ):
        logger.debug("Sending DONE for request %s to %s:%d", request_id, remote_host, remote_handshake_port)
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            data_bytes = self.encoder.encode((DONE_RECVING_MSG, request_id, remote_port_send_num))
            ensure_zmq_send(sock, data_bytes, f"{remote_host}:{remote_handshake_port}")
            resp = ensure_zmq_recv(sock, f"{remote_host}:{remote_handshake_port}")
            if resp != b"ACK":
                logger.error("Failed ACK for request %s from %s:%d", request_id, remote_host, remote_handshake_port)
                raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        except RuntimeError as e:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
                logger.warning("Socket error in DONE send. error=%s. ", e)
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()
            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(ctx=ctx, path=remote_path, socket_type=zmq.REQ, bind=False)  # type: ignore
            sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))  # type: ignore
            sock.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))  # type: ignore
            return sock

    def _return_remote_socket(self, sock: zmq.Socket, remote_host: str, remote_handshake_port: int) -> None:  # type: ignore
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)


# ---------------------------------------------------------------------------
# ConnectorMetadata + dispatcher (forked; class rename only)
# ---------------------------------------------------------------------------
class HIXLConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}
        self.requests_to_send: dict[str, float] = {}
        self.reqs_in_batch: set[str] = set()

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        num_external_tokens: int,
        kv_transfer_params: dict[str, Any],
    ):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            num_external_tokens=num_external_tokens,
            num_computed_tokens=kv_transfer_params.get("num_computed_tokens", 0),
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_request_id=kv_transfer_params["remote_request_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_ptp_size=kv_transfer_params.get("remote_ptp_size", 1),
            num_prompt_blocks=kv_transfer_params.get("num_prompt_blocks", 0),
            remote_block_size=kv_transfer_params.get("remote_block_size", 0),
        )


class HIXLConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = HIXLConnectorMetadata()
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: HIXLConnectorScheduler | None = HIXLConnectorScheduler(
                vllm_config, str(self.engine_id), kv_cache_config
            )
            self.connector_worker: HIXLConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = HIXLConnectorWorker(vllm_config, str(self.engine_id), kv_cache_config)

    # ---- Scheduler side ----
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    # ---- Worker side ----
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, HIXLConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        pass

    def wait_for_save(self):
        pass

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        assert self.connector_worker is not None
        return self.connector_worker.xfer_handshake_metadata

    def set_xfer_handshake_metadata(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata_from_workers(metadata)

    def shutdown(self):
        if self.connector_worker is not None:
            self.connector_worker.shutdown()


# ---------------------------------------------------------------------------
# Scheduler (Phase 1: single FullAttention group; Mamba/SWA/compress paths
# removed). Forked from MooncakeConnectorScheduler and trimmed.
# ---------------------------------------------------------------------------
class HIXLConnectorScheduler:
    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        init_ascend_config(vllm_config)
        self.ascend_config = get_ascend_config()
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id = engine_id
        self.side_channel_host = get_ip()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        logger.info("Initializing HIXL Scheduler %s", engine_id)

        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.pipeline_parallel_size
        )
        self._reqs_need_recv: dict[str, tuple[Request, BlockIds, int]] = {}
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_in_batch: set[str] = set()

        self.multi_nodes_meta_mapping: dict[str, dict[str, Any]] = {}
        self.kv_cache_groups = kv_cache_config.kv_cache_groups

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            token_ids = request.prompt_token_ids or []
            count = max(len(token_ids) - num_computed_tokens, 0)
            params["num_computed_tokens"] = num_computed_tokens
            if count > 0:
                return count, True
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        params = request.kv_transfer_params
        if params is not None and (params.get("do_remote_prefill", False) or params.get("do_remote_decode", False)):
            self._reqs_in_batch.add(request.request_id)
        if params is not None and params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_engine_id", "remote_host", "remote_port", "remote_request_id")):
                    local_block_ids = (
                        blocks.get_unhashed_block_ids_all_groups() if num_external_tokens > 0 else []
                    )
                    self._reqs_need_recv[request.request_id] = (request, local_block_ids, num_external_tokens)
                else:
                    logger.warning("Got invalid KVTransferParams. params=%s. ", params)
            else:
                assert num_external_tokens == 0
            params["do_remote_prefill"] = False

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = HIXLConnectorMetadata()
        for req_id, (req, block_ids, num_external_tokens) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                num_external_tokens=num_external_tokens,
                kv_transfer_params=req.kv_transfer_params,
            )
        self._reqs_need_recv.clear()
        meta.requests_to_send = self._reqs_need_send
        self._reqs_need_send = {}
        meta.reqs_in_batch = self._reqs_in_batch
        self._reqs_in_batch = set()
        return meta

    def request_finished(
        self, request: "Request", block_ids: BlockIds
    ) -> tuple[bool, dict[str, Any] | None]:
        params = request.kv_transfer_params
        if (
            params is None
            or not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)
        computed_block_ids = block_ids
        # Phase 1 single group: just take group 0 length.
        computed_block_lens = [len(bid_list) for bid_list in computed_block_ids]
        delay_free_blocks = sum(computed_block_lens) > 0
        if delay_free_blocks:
            logger.info("Delaying free of %d blocks for request %s", sum(computed_block_lens), request.request_id)
            self._reqs_need_send[request.request_id] = time.time()

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            remote_ptp_size=self.tp_size,
            last_token_id=request.output_token_ids[-1],
            remote_multi_nodes_meta_mapping=self.multi_nodes_meta_mapping,
            num_prompt_blocks=num_prompt_blocks,
            remote_block_size=self.block_size,
        )

    def _port_offset_from_handshake_metadata(
        self, rank_metadata: KVConnectorHandshakeMetadata, metadata_key: int | tuple[int, ...]
    ) -> int:
        kv_port = self.vllm_config.kv_transfer_config.kv_port
        handshake_port = getattr(rank_metadata, "handshake_port", 0)
        if handshake_port > 0:
            return handshake_port - kv_port
        if isinstance(metadata_key, int):
            return metadata_key
        raise ValueError(f"HIXL handshake metadata missing handshake_port for key {metadata_key}")

    def set_xfer_handshake_metadata_from_workers(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        if not metadata:
            return
        updated_mapping: dict[str, dict[str, Any]] = {}
        for metadata_key, rank_metadata in metadata.items():
            port_offset = self._port_offset_from_handshake_metadata(rank_metadata, metadata_key)
            updated_mapping[str(port_offset)] = {
                "host": rank_metadata.local_ip,
                "engine_id": rank_metadata.engine_id,
            }
        self.multi_nodes_meta_mapping.update(updated_mapping)
        logger.info(
            "HIXL set_xfer_handshake_metadata: worker_count=%d, mapping=%s",
            len(metadata), self.multi_nodes_meta_mapping,
        )

    def set_xfer_handshake_metadata(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        self.set_xfer_handshake_metadata_from_workers(metadata)


# ---------------------------------------------------------------------------
# Worker (Phase 1: TP=1, single FullAttention group, PP=1).
# __init__ uses get_datadist; register_kv_caches uses register_blocks_cache.
# ---------------------------------------------------------------------------
class HIXLConnectorWorker:
    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        self.vllm_config = vllm_config
        self.ascend_config = get_ascend_config()
        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.side_channel_host = get_ip()
        self.total_layers = vllm_config.model_config.get_total_num_hidden_layers()
        self.num_key_value_heads = vllm_config.model_config.hf_text_config.num_key_value_heads

        # Phase 1 constraints.
        assert self.tp_size == 1, "HIXLConnector Phase 1 supports TP=1 only."
        assert self.pp_size == 1, "HIXLConnector Phase 1 supports PP=1 only."

        self.kv_cache_config = kv_cache_config
        self.num_blocks: int = kv_cache_config.num_blocks
        self.kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]] = {}
        self._layer_specs = {
            layer: group.kv_cache_spec for group in kv_cache_config.kv_cache_groups for layer in group.layer_names
        }

        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.pipeline_parallel_size
        )
        device_index = self.tp_rank  # pp=1, pcp=1, tp=1 -> 0
        self.handshake_port = self.side_channel_port + device_index

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.block_size = vllm_config.cache_config.block_size

        # HIXL engine handle (replaces Mooncake global_te).
        cluster_id, listen_ip, listen_port = self._compute_identity()
        extra = self._extra_options()
        self.hixl = get_datadist(
            kv_role=self.kv_role,
            cluster_id=cluster_id,
            listen_ip=listen_ip,
            listen_port=listen_port,
            device_id=self._current_npu_device_id(),
            link_timeout_ms=extra.get("link_timeout_ms", 5000),
            extra_options=extra,
        )
        self.cache_manager = self.hixl.cache_manager
        self.cluster_id = cluster_id
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.model_id = extra.get("model_id", 0)
        self.group_caches: dict[int, Any] = {}  # kv_cache_group_id -> registered Cache

        self.kv_send_thread: KVCacheSendingThread | None = None
        self.kv_recv_thread: KVCacheRecvingThread | None = None
        self.xfer_handshake_metadata: HixlAgentMetadata | None = None

    def _extra_options(self) -> dict[str, Any]:
        kvtc = self.vllm_config.kv_transfer_config
        role_key = "prefill" if self.kv_role == "kv_producer" else "decode"
        role_cfg: dict[str, Any] = kvtc.get_from_extra_config(role_key, {})
        role_cfg["tp_size"] = role_cfg.get("tp_size", self.tp_size)
        hixl_cfg: dict[str, Any] = kvtc.get_from_extra_config("hixl", {})
        hixl_cfg.setdefault("prefill" if self.kv_role == "kv_producer" else "decode", role_cfg)
        hixl_cfg.setdefault("cluster_id_base", None)
        hixl_cfg.setdefault("listen_port_base", kvtc.kv_port + 1000)
        return hixl_cfg

    def _compute_identity(self) -> tuple[int, str, int]:
        """Per-rank unique cluster_id / listen_port (see design §5)."""
        extra = self._extra_options()
        cluster_id_base = extra["cluster_id_base"]
        assert cluster_id_base is not None, (
            "extra_config['hixl']['cluster_id_base'] is required (P/D must use disjoint bases)."
        )
        listen_port_base = extra.get("listen_port_base", self.vllm_config.kv_transfer_config.kv_port + 1000)
        device_index = (self.pp_rank * 1 + 0) * self.tp_size + self.tp_rank  # pcp=1, pcp_rank=0
        offset = self.dp_rank * (self.tp_size * self.pp_size) + device_index
        return int(cluster_id_base) + offset, self.side_channel_host, int(listen_port_base) + offset

    @staticmethod
    def _current_npu_device_id() -> int | None:
        try:
            cur = torch.npu.current_device()
            if isinstance(cur, int):
                return cur
            s = str(cur)
            return int(s.split(":")[-1]) if ":" in s else None
        except Exception:
            return None

    @staticmethod
    def _as_kv_cache_tuple(kv_cache_tuple: Any) -> list[torch.Tensor]:
        if isinstance(kv_cache_tuple, (list, tuple)):
            return list(kv_cache_tuple)
        return [kv_cache_tuple]

    def _build_kv_group2layeridx(self) -> dict[int, tuple[dict[str, Any], list[int]]]:
        from vllm.v1.worker.utils import extract_layer_index

        def to_msgpackable(value: Any) -> Any:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, dict):
                return {str(k): to_msgpackable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [to_msgpackable(item) for item in value]
            try:
                builtins_value = msgspec.to_builtins(value)
                if builtins_value is value:
                    return repr(value)
                return to_msgpackable(builtins_value)
            except TypeError:
                return repr(value)

        kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]] = {}
        num_attn_module = 1
        transfer_group_id = 0
        for kv_cache_group_id, group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
            layer_names = list(group_spec.layer_names)
            layer_indices = [extract_layer_index(name, num_attn_module) for name in layer_names]
            spec = group_spec.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = {n: spec.kv_cache_specs[n] for n in layer_names}
            serialized_spec = to_msgpackable(spec)
            if not isinstance(serialized_spec, dict):
                serialized_spec = {"repr": serialized_spec}
            num_kv_heads = getattr(group_spec.kv_cache_spec, "num_kv_heads", None)
            if isinstance(num_kv_heads, int):
                serialized_spec["num_kv_heads"] = num_kv_heads
            kv_group2layeridx[transfer_group_id] = (
                {
                    "layer_names": layer_names,
                    "kv_cache_spec_type": type(group_spec.kv_cache_spec).__name__,
                    "kv_cache_spec": serialized_spec,
                    "kv_cache_group_id": kv_cache_group_id,
                },
                layer_indices,
            )
            transfer_group_id += 1
        return kv_group2layeridx

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV via HIXL register_blocks_cache (one Cache per group).

        Replaces Mooncake's collect_storage_merged_register_regions +
        global_te.register_buffer. Both P and D register with
        remote_accessible=True: P so the decoder can find/resolve the src
        cache, D because llm_datadist's PullCacheByGet path (force-enabled by
        EnableRemoteCacheAccessible=1) requires the local dst cache to be
        remote_accessible too, else pull_blocks returns LLM_PARAM_INVALID."""
        from llm_datadist import CacheDesc, BlocksCacheKey, Placement

        self.kv_caches = kv_caches
        self.kv_group2layeridx = self._build_kv_group2layeridx()

        num_tensors_per_group: list[int] = []
        block_size_scale: list[list[int]] = []
        layer_name_to_idx = {
            name: idx
            for _, (spec, idxs) in self.kv_group2layeridx.items()
            for name, idx in zip(spec["layer_names"], idxs)
        }

        for group_id, (group_spec, layer_indices) in self.kv_group2layeridx.items():
            kv_cache_group_id = group_spec.get("kv_cache_group_id", group_id)
            layer_names = group_spec["layer_names"]
            addrs: list[int] = []
            ref_shape = None
            ref_dtype = None
            for layer_name in layer_names:
                for single_kv_cache in self._as_kv_cache_tuple(kv_caches[layer_name]):
                    addrs.append(int(single_kv_cache.data_ptr()))
                    if ref_shape is None:
                        ref_shape = tuple(single_kv_cache.shape)
                        ref_dtype = single_kv_cache.dtype
                    else:
                        assert tuple(single_kv_cache.shape) == ref_shape, (
                            f"HIXL register_blocks_cache requires all tensors in a group "
                            f"share one shape; {layer_name} "
                            f"{tuple(single_kv_cache.shape)} != {ref_shape}"
                        )
            assert ref_shape is not None, f"No KV tensors found for group {group_id}"
            num_tensors = len(addrs)
            assert num_tensors == len(layer_indices) * 2, (
                f"num_tensors {num_tensors} != layers*2 {len(layer_indices) * 2}"
            )
            num_tensors_per_group.append(num_tensors)
            # block_size_scale[group]: tensor num_blocks / logical num_blocks.
            scale = ref_shape[0] // self.num_blocks
            assert scale == 1, (
                f"HIXLConnector Phase 1 requires block_size_scale==1 (standard "
                f"FullAttention), got {scale}."
            )
            block_size_scale.append([scale])

            cache_desc = CacheDesc(
                num_tensors=num_tensors,
                shape=list(ref_shape),  # full single-tensor shape incl num_blocks
                data_type=_torch_dtype_to_llm_dtype(ref_dtype),
                placement=Placement.DEVICE,
            )
            # Both P and D register remote_accessible=True. P so the decoder
            # can resolve the src cache; D because llm_datadist's PullCacheByGet
            # path (force-enabled via EnableRemoteCacheAccessible=1) requires
            # the local dst cache to be remote_accessible too, else
            # pull_blocks returns LLM_PARAM_INVALID.
            remote_accessible = True
            cache = self.cache_manager.register_blocks_cache(
                cache_desc,
                addrs,
                BlocksCacheKey(self.cluster_id, self.model_id),
                remote_accessible=remote_accessible,
            )
            logger.info(
                "HIXL DEBUG register: cluster_id=%s model_id=%s shape=%s num_tensors=%s "
                "num_blocks=%s remote_accessible=%s kv_role=%s",
                self.cluster_id, self.model_id, cache_desc.shape, cache_desc.num_tensors,
                ref_shape[0], remote_accessible, self.kv_role,
            )
            self.group_caches[kv_cache_group_id] = cache

        self.block_size_scale = block_size_scale

        metadata = HixlAgentMetadata(
            engine_id=self.engine_id,
            cluster_id=self.cluster_id,
            listen_ip=self.listen_ip,
            listen_port=self.listen_port,
            model_id=self.model_id,
            num_tensors_per_group=num_tensors_per_group,
            kv_group2layeridx=self.kv_group2layeridx,
            block_size=self.block_size,
            num_blocks=self.num_blocks,
            block_size_scale=block_size_scale,
            local_ip=get_ip(),
            handshake_port=self.handshake_port,
        )
        logger.info(
            "HIXL DEBUG P metadata: cluster_id=%s listen=%s:%s num_tensors_per_group=%s",
            self.cluster_id, self.listen_ip, self.listen_port, num_tensors_per_group,
        )
        self.xfer_handshake_metadata = metadata

        ready_event = threading.Event()
        if self.kv_role == "kv_producer":
            self.kv_send_thread = KVCacheSendingThread(
                self.vllm_config,
                self.tp_rank,
                self.tp_size,  # prefill_tp_size == tp_size (Phase 1)
                self.engine_id,
                self.side_channel_host,
                self.side_channel_port,
                metadata,
                ready_event,
                self.kv_caches,
                0,  # pcp_rank
            )
            self.kv_send_thread.start()
        else:
            self.kv_recv_thread = KVCacheRecvingThread(
                self.tp_rank,
                self.tp_size,
                self.hixl,
                self.model_id,
                self.engine_id,
                self.handshake_port,
                self.side_channel_port,
                self.vllm_config,
                self.kv_caches,
                self.kv_group2layeridx,
                self.group_caches,
                block_size_scale,
                ready_event,
            )
            self.kv_recv_thread.start()

        start_wait_time = time.time()
        thread = self.kv_send_thread if self.kv_role == "kv_producer" else self.kv_recv_thread
        assert thread is not None
        while not ready_event.is_set():
            if not thread.is_alive():
                raise RuntimeError("HIXL KV Cache sending/receiving thread failed to start.")
            if time.time() - start_wait_time > 5 * 60:
                raise RuntimeError("Timeout waiting for HIXL KV Cache thread to be ready.")
            time.sleep(3)

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending = (
            self.kv_send_thread.get_and_clear_finished_requests()
            if self.kv_role == "kv_producer"
            else set()
        )
        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests()
            if self.kv_role == "kv_consumer"
            else set()
        )
        return done_sending, done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.kv_role == "kv_consumer" and self.kv_recv_thread is not None:
            return self.kv_recv_thread.get_and_clear_invalid_block_ids()
        return set()

    def start_load_kv(self, metadata: HIXLConnectorMetadata):
        """Phase 1 simplified shard metadata: TP=1 -> single P rank, single
        shard, one GroupPull per group (num_group_pulls=1). No CP/PCP/DCP
        port mapping (forked _get_kv_split_metadata CP branch dropped)."""
        for req_id in metadata.reqs_in_batch:
            if self.kv_send_thread is not None:
                self.kv_send_thread.task_tracker.add_req_to_process(req_id)
            if self.kv_recv_thread is not None:
                self.kv_recv_thread.task_tracker.add_req_to_process(req_id)

        for req_id, meta in metadata.requests.items():
            remote_req_id = meta.remote_request_id
            # Phase 1: TP=1, single P rank == remote_port + tp_rank(0).
            remote_handshake_port = meta.remote_port + self.tp_rank
            group_pulls = [
                GroupPull(
                    group_id=gid,
                    remote_tp_offset=0,
                    num_group_pulls=1,
                    is_group_transfer_end=True,
                )
                for gid, (spec, idxs) in self.kv_group2layeridx.items()
                if idxs
            ]
            assert self.kv_recv_thread is not None
            self.kv_recv_thread.add_request(
                request_id=req_id,
                remote_request_id=remote_req_id,
                local_block_ids=meta.local_block_ids,
                remote_block_ids=meta.remote_block_ids,
                group_pulls=group_pulls,
                remote_engine_id=meta.remote_engine_id,
                remote_host=meta.remote_host,
                remote_handshake_port=remote_handshake_port,
                num_computed_tokens=meta.num_computed_tokens,
                all_task_done=True,
            )

        if self.kv_send_thread is not None:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                self.kv_send_thread.add_delayed_request(req_id, delay_start_time)

    def shutdown(self):
        try:
            shutdown_datadist()
        except Exception as e:
            logger.warning("HIXL shutdown_datadist failed: %s", e)


# ---------------------------------------------------------------------------
# ZMQ helpers (forked from Mooncake, engine-agnostic)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:  # type: ignore
    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):  # type: ignore
        raise ValueError(f"Unexpected socket type: {socket_type}")
    ctx: zmq.Context | None = None  # type: ignore
    try:
        ctx = zmq.Context()  # type: ignore
        yield make_zmq_socket(ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER)  # type: ignore
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def group_concurrent_contiguous(
    src: list[int], dst: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Group block ids that are contiguous in both id space and memory."""
    src_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)
    dst_indices: npt.NDArray[np.int64] = np.array(dst, dtype=np.int64)
    if src_indices.size == 0:
        return [], []
    src_byte_contiguous = np.diff(src_indices) == 1
    dst_byte_contiguous = np.diff(dst_indices) == 1
    brk = np.where(~(src_byte_contiguous & dst_byte_contiguous))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)
    return [g.tolist() for g in src_groups], [g.tolist() for g in dst_groups]


def string_to_int64_hash(input_str):
    hashed_bytes = hashlib.sha256(input_str.encode("utf-8")).digest()
    trunked_bytes = hashed_bytes[:8]
    uint64_value = struct.unpack("<Q", trunked_bytes)[0]
    return uint64_value


def ensure_zmq_send(socket: zmq.Socket, data: bytes, path: str, max_retries: int = 3):  # type: ignore
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Send failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                raise RuntimeError(f"Failed to send data to {path} after {max_retries} retries: {e}")


def ensure_zmq_recv(socket: zmq.Socket, path: str, max_retries: int = 3) -> bytes:  # type: ignore
    retries_left = max_retries
    while True:
        try:
            return socket.recv()
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Receive failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                raise RuntimeError(f"Failed to receive data after {max_retries} retries: {e}")
