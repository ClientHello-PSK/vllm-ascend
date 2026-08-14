# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HIXLEngineConnector — address-level KV transfer via hixl::Hixl.

Data-plane counterpart of NIXL's pull connector (vllm/vllm/distributed/
kv_transfer/kv_connector/v1/nixl/). Where the existing HIXLConnector talks to
LLM-DataDist's block API (register_blocks_cache/pull_blocks) and therefore
needs staging + post-transpose for TP>1, this connector drives the
address-level hixl::Hixl API (RegisterMem/TransferAsync) directly. Head
split/gather for heterogeneous TP is expressed as remote_addr offsets in
TransferOpDesc, so KV lands in the D cache at its real layout — no staging,
no reformat, async transfer handle polling.

Control plane (ZMQ handshake / scheduler decisions / port allocation /
delayed free) is forked from NIXL (pull_scheduler / base_scheduler /
metadata) and self-contained in this module — it does NOT import
hixl_connector.py (which may be retired independently).

STATUS: data plane + control plane implemented. The connector drives the
hixl::Hixl address-level API (RegisterMem / TransferAsync / GetTransferStatus
/ SendNotify / GetNotifies / Connect) directly, with head split/gather for
heterogeneous TP expressed as remote_addr offsets in TransferOpDesc.
Control plane (HIXLEngineRemoteMeta / HIXLEngineReqMeta /
HIXLEngineConnectorMetadata / HIXLEngineConnectorScheduler / handshake payload
/ zmq_ctx / KVCacheTaskTracker) is forked from NIXL and self-contained in this
module. No RecvingThread — transfers are async and handles are polled in
get_finished / wait_for_layer_load on the worker main thread. Outstanding
TODOs: Mamba conv decomposition, heterogeneous block_size reshard
(desc decoupling), CP>1, and the items logged inline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import msgspec
import torch
import zmq
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineTransferInfo,
    TransferTopology,
    get_current_attn_backends,
)
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    MambaConvSplitInfo,
    derive_mamba_conv_split,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    MLAAttentionSpec,
    MambaSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend.distributed.kv_transfer.kv_p2p.tp_mapping import (
    TPMapping,
    compute_tp_mapping,
)
# isort: off
if TYPE_CHECKING:
    from hixl import TransferOpDesc
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request
# isort: on

# hixl_py (import hixl) is the address-level pybind11 binding built from
# src/python/hixl_py/hixl_py.cc. Loaded lazily so vllm-ascend stays importable
# without the hixl build artefact unless HIXLEngineConnector is selected.
_HIXL_MOD = None


def _load_hixl():
    global _HIXL_MOD
    if _HIXL_MOD is None:
        import hixl  # type: ignore[import-not-found]
        _HIXL_MOD = hixl
    return _HIXL_MOD

# ---------------------------------------------------------------------------
# Control-plane constants & helpers (forked so this module does not import
# hixl_connector — see Step R). done-notification runs over the hixl
# data-plane (hixl.send_notify / get_notifies), so there is no
# DONE_RECVING_MSG side-channel constant here.
# ---------------------------------------------------------------------------
GET_META_MSG = b"get_meta_msg"
# Error-reply marker for a malformed GET_META handshake. A normal reply's
# handshake_bytes is msgpack-encoded and never starts with this ASCII prefix.
HIXL_ERR_PREFIX = b"__HIXL_ERR__"

# EngineFactory selectors. Do not change engine_factory.cc; this connector
# injects/strips Initialize options so 910 defaults to HixlCS.
_HIXL_ENGINE_BACKEND_CS = "hixl_cs"
_HIXL_ENGINE_BACKEND_COMM = "comm"
_HIXL_ENGINE_BACKENDS = frozenset({
    _HIXL_ENGINE_BACKEND_CS,
    _HIXL_ENGINE_BACKEND_COMM,
})
_HIXL_CS_LOCAL_COMM_RES = '{"version":"1.3"}'
_HIXL_OPTION_LOCAL_COMM_RES = "LocalCommRes"
_HIXL_OPTION_GLOBAL_RESOURCE_CONFIG = "GlobalResourceConfig"
_HIXL_PROTOCOL_DESC_FLAT = "comm_resource_config.protocol_desc"


def _loads_json_object(raw: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(raw)
    except (TypeError, json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _protocol_desc_from_grc(obj: dict[str, Any]) -> Any:
    if _HIXL_PROTOCOL_DESC_FLAT in obj:
        return obj[_HIXL_PROTOCOL_DESC_FLAT]
    crc = obj.get("comm_resource_config")
    if isinstance(crc, dict):
        return crc.get("protocol_desc")
    return None


def _protocol_desc_nonempty(desc: Any) -> bool:
    if desc is None:
        return False
    if isinstance(desc, str):
        return bool(desc)
    if isinstance(desc, list):
        return any(bool(item) for item in desc)
    return True


def _local_comm_res_is_cs(raw: str) -> bool:
    obj = _loads_json_object(raw)
    return obj is not None and obj.get("version") == "1.3"


def _has_protocol_desc(options: dict[str, str]) -> bool:
    raw = options.get(_HIXL_OPTION_GLOBAL_RESOURCE_CONFIG)
    if not raw:
        return False
    obj = _loads_json_object(raw)
    if obj is None:
        return False
    return _protocol_desc_nonempty(_protocol_desc_from_grc(obj))


def _strip_protocol_desc(grc_raw: str) -> tuple[str | None, bool]:
    obj = _loads_json_object(grc_raw)
    if obj is None:
        return grc_raw, False
    stripped = False
    if _HIXL_PROTOCOL_DESC_FLAT in obj:
        desc = obj.pop(_HIXL_PROTOCOL_DESC_FLAT)
        stripped = stripped or _protocol_desc_nonempty(desc)
    crc = obj.get("comm_resource_config")
    if isinstance(crc, dict) and "protocol_desc" in crc:
        desc = crc.pop("protocol_desc")
        stripped = stripped or _protocol_desc_nonempty(desc)
        if not crc:
            obj.pop("comm_resource_config", None)
    if not obj:
        return None, stripped
    return json.dumps(obj, separators=(",", ":")), stripped


def _normalize_hixl_engine_options(raw: Any) -> dict[str, str]:
    """Coerce hixl_engine.options to map<string,string> for pybind Initialize."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            "hixl_engine.options must be a dict, "
            f"got {type(raw).__name__}."
        )
    out: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            out[name] = json.dumps(value, separators=(",", ":"))
        elif isinstance(value, str):
            out[name] = value
        else:
            out[name] = str(value)
    return out


def _apply_hixl_engine_backend_options(
    backend: str, options: dict[str, str]
) -> tuple[dict[str, str], str]:
    """Inject or strip EngineFactory CS selectors. Does not mutate `options`.

    Factory order: nonempty LocalCommRes version=="1.3" → hixl_cs; other
    nonempty LocalCommRes → comm (protocol_desc is never consulted); else
    nonempty protocol_desc → hixl_cs. So a leftover 1.2 LCR would pin comm
    even if protocol_desc is set — backend=hixl_cs replaces it.
    """
    out = dict(options)
    if backend == _HIXL_ENGINE_BACKEND_CS:
        lcr = out.get(_HIXL_OPTION_LOCAL_COMM_RES, "")
        if lcr and _local_comm_res_is_cs(lcr):
            return out, "none"
        if lcr:
            logger.warning(
                "hixl_engine.backend=hixl_cs but LocalCommRes is not version "
                "1.3; replacing it so EngineFactory selects hixl_cs. "
                "original=%s",
                lcr,
            )
            out[_HIXL_OPTION_LOCAL_COMM_RES] = _HIXL_CS_LOCAL_COMM_RES
            return out, "LocalCommRes:1.3"
        if _has_protocol_desc(out):
            return out, "none"
        out[_HIXL_OPTION_LOCAL_COMM_RES] = _HIXL_CS_LOCAL_COMM_RES
        return out, "LocalCommRes:1.3"

    if backend == _HIXL_ENGINE_BACKEND_COMM:
        stripped: list[str] = []
        lcr = out.get(_HIXL_OPTION_LOCAL_COMM_RES, "")
        if lcr and _local_comm_res_is_cs(lcr):
            out.pop(_HIXL_OPTION_LOCAL_COMM_RES)
            stripped.append("LocalCommRes:1.3")
        grc = out.get(_HIXL_OPTION_GLOBAL_RESOURCE_CONFIG)
        if grc:
            new_grc, did_strip = _strip_protocol_desc(grc)
            if did_strip:
                stripped.append("protocol_desc")
                if new_grc is None:
                    out.pop(_HIXL_OPTION_GLOBAL_RESOURCE_CONFIG)
                else:
                    out[_HIXL_OPTION_GLOBAL_RESOURCE_CONFIG] = new_grc
        if stripped:
            logger.warning(
                "hixl_engine.backend=comm is authoritative; stripped CS "
                "selector(s) %s from options.",
                ",".join(stripped),
            )
        injected = "none" if not stripped else "stripped:" + ",".join(stripped)
        return out, injected

    raise ValueError(
        "hixl_engine.backend must be 'hixl_cs' or 'comm', "
        f"got {backend!r}."
    )


@contextmanager
def zmq_ctx(socket_type: Any, addr: str):  # type: ignore[override]
    """NIXL-style ZMQ context manager (forked from hixl_connector.zmq_ctx).

    Accepts ROUTER / REQ / DEALER; ROUTER binds (scheduler-side listener),
    the others connect. A fresh per-call Context is destroyed with linger=0
    so a crashed peer does not hang shutdown.
    """
    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):  # type: ignore[attr-defined]
        raise ValueError(f"Unexpected socket type: {socket_type}")
    ctx = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(
            ctx=ctx, path=addr, socket_type=socket_type,
            bind=socket_type == zmq.ROUTER,  # type: ignore[attr-defined]
        )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)  # type: ignore[attr-defined]


class KVCacheTaskTracker:
    """Tracks finished / in-flight requests (forked & trimmed from
    hixl_connector.KVCacheTaskTracker).

    add_req_to_process (start_load_kv) registers a req as in-batch;
    discard_from_process drops aborted reqs (metadata.reqs_not_processed);
    update_done_task_count is called from _get_new_notifs once a req's
    consumer count reaches consumers_per_producer, moving it to finished;
    get_and_clear_finished_requests drains finished for get_finished.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._finished_requests: set[str] = set()
        self._reqs_to_process: set[str] = set()

    def add_req_to_process(self, request_id: str) -> None:
        with self._lock:
            self._reqs_to_process.add(request_id)

    def discard_from_process(self, request_id: str) -> None:
        with self._lock:
            self._reqs_to_process.discard(request_id)

    def __contains__(self, request_id: str) -> bool:
        # _get_new_notifs uses this to distinguish a premature DONE (D finished
        # reading before this P worker's request_finished moved the req into
        # _reqs_to_send) from a truly unknown/expired req; the former is a
        # debug log, not an error.
        with self._lock:
            return request_id in self._reqs_to_process

    def update_done_task_count(self, request_id: str) -> None:
        with self._lock:
            if request_id in self._reqs_to_process:
                self._finished_requests.add(request_id)
                self._reqs_to_process.discard(request_id)
            else:
                logger.warning(
                    "HIXLEngine finish req not in process: %s", request_id)

    def get_and_clear_finished_requests(self) -> set[str]:
        with self._lock:
            finished = self._finished_requests
            self._finished_requests = set()
        return finished

# ---------------------------------------------------------------------------
# Handshake metadata (address-level, mirrors NIXL NixlAgentMetadata /
# NixlHandshakePayload). Exchanged over the ZMQ side channel so the D-side
# worker learns the P-side KV base addresses / page sizes needed to build
# TransferOpDesc batches.
# ---------------------------------------------------------------------------


class HixlEngineAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    """Address-level agent metadata for one (pp_rank, tp_rank) engine.

    Mirrors NixlAgentMetadata (vllm/.../nixl/metadata.py:48-60) but swaps the
    opaque NIXL ``agent_metadata`` bytes for ``local_engine_endpoint`` (the
    hixl Initialize host:port string the D side Connect()s to). All address
    fields are populated in register_kv_caches and shipped over ZMQ.
    """

    engine_id: str
    local_engine_endpoint: str  # hixl Initialize host:port (D calls Connect)
    cluster_id: int
    listen_ip: str
    listen_port: int
    # Per-region (per layer K/V split) base virtual address and page stride.
    kv_caches_base_addr: list[int] = []
    block_lens: list[int] = []
    num_blocks: int = 0
    device_id: int = 0
    block_size: int = 0
    kv_cache_layout: str = ""  # "HND" / "NHD"
    ssm_sizes: tuple[int, int] = (0, 0)  # (conv_size, ssm_size)
    physical_blocks_per_logical_kv_block: int = 1
    model_id: int = 0
    # Compatibility factors also encoded in the handshake hash.
    num_kv_heads: int = 0
    head_size: int = 0
    attn_backend_name: str = ""


class HixlEngineHandshakePayload(msgspec.Struct, omit_defaults=True):
    """Two-stage handshake envelope (mirrors NixlHandshakePayload).

    ``compatibility_hash`` is verified before ``agent_metadata_bytes`` is
    decoded, so a version mismatch raises a clear RuntimeError instead of a
    confusing msgspec schema error.
    """

    compatibility_hash: str
    agent_metadata_bytes: bytes


# msgspec.Struct and ABC have incompatible metaclasses, so the payload
# cannot directly inherit KVConnectorHandshakeMetadata (the way NIXL's
# @dataclass NixlHandshakePayload does). Register it as a virtual subclass
# so isinstance(payload, KVConnectorHandshakeMetadata) holds for any framework
# type check, keeping the handshake contract intact.
KVConnectorHandshakeMetadata.register(HixlEngineHandshakePayload)


# ---------------------------------------------------------------------------
# Per-request metadata (forked from NIXL metadata.py:156-175 so the connector
# does not depend on hixl_connector.py's flat ReqMeta). ``remote`` is a nested
# RemoteMeta; ``tp_size`` lives at the top level (NIXL convention, not the HIXL
# ``remote_ptp_size`` flat field). ``block_ids`` stays a per-group list to
# match the existing data-plane (_read_blocks indexes [g]).
# ---------------------------------------------------------------------------


@dataclass
class HIXLEngineRemoteMeta:
    """Remote producer info for one request (mirrors NIXL RemoteMeta)."""

    block_ids: list[list[int]]
    host: str
    port: int
    engine_id: str
    request_id: str
    blocks_expiry_time: float | None = None


@dataclass
class HIXLEngineReqMeta:
    """Per-request metadata consumed by the D-side worker."""

    local_block_ids: list[list[int]]
    tp_size: int
    remote: HIXLEngineRemoteMeta | None = None


def compute_hixl_engine_compat_hash(
    *,
    vllm_version: str,
    model_name: str,
    dtype: str,
    num_kv_heads: int,
    head_size: int,
    num_hidden_layers: int,
    attn_backend_name: str,
    cache_dtype: str,
    is_hma_enabled: bool,
) -> str:
    """SHA-256 over the factors that must match for P/D byte compatibility.

    Mirrors NIXL compute_nixl_compatibility_hash (metadata.py:81-141) but
    without NIXL_CONNECTOR_VERSION (HIXLEngineConnector is new, versioned via
    the hash itself). Bump the prefix string when the on-wire metadata schema
    changes in a backward-incompatible way.
    """
    prefix = "hixl-engine-v1"
    payload = "|".join(
        [
            prefix,
            vllm_version,
            model_name,
            dtype,
            str(num_kv_heads),
            str(head_size),
            str(num_hidden_layers),
            attn_backend_name,
            cache_dtype,
            str(is_hma_enabled),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# Lease / heartbeat constants (mirrors NIXL base_scheduler.py:70-76).
_HIXL_ENGINE_LEASE_DURATION_S = 30
# D-side REQ receive timeout (NIXL uses 5s, base_worker.py:619).
_HIXL_ENGINE_REQ_TIMEOUT_S = 5.0
# P-side ROUTER poll timeout (how long recv blocks before checking stop).
_HIXL_ENGINE_LISTENER_POLL_MS = 1000
# How long to wait for the P-side ROUTER socket to bind before returning.
_HIXL_ENGINE_LISTENER_READY_TIMEOUT_S = 30.0


class HIXLEngineConnectorMetadata(KVConnectorMetadata):
    """Per-step connector metadata (forked from NIXL NixlConnectorMetadata).

    Carries HIXLEngineReqMeta entries from the scheduler-side
    request_finished/update_state_after_alloc decisions to the D-side
    worker. add_new_req_to_recv fills the nested HIXLEngineRemoteMeta
    from kv_transfer_params (P-side request_finished returns
    remote_host/remote_port/remote_engine_id/remote_request_id
    /tp_size/remote_block_ids/remote_blocks_expiry_time).
    """

    def __init__(self):
        self.reqs_to_recv: dict[str, HIXLEngineReqMeta] = {}
        self.reqs_to_send: dict[str, float] = {}
        self.reqs_in_batch: set[str] = set()
        self.reqs_not_processed: set[str] = set()
        # Heartbeat data grouped by remote engine, sent by D worker to P.
        self.heartbeat_by_engine: dict[str, Any] = {}

    def _add_new_req(
        self,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict[str, Any],
    ) -> HIXLEngineReqMeta:
        return HIXLEngineReqMeta(
            local_block_ids=local_block_ids,
            tp_size=kv_transfer_params.get("tp_size", 1),
        )

    def add_new_req_to_recv(
        self,
        request_id: str,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict[str, Any],
    ) -> None:
        req = self._add_new_req(local_block_ids, kv_transfer_params)
        req.remote = HIXLEngineRemoteMeta(
            block_ids=kv_transfer_params["remote_block_ids"],
            engine_id=kv_transfer_params["remote_engine_id"],
            request_id=kv_transfer_params["remote_request_id"],
            host=kv_transfer_params["remote_host"],
            port=kv_transfer_params["remote_port"],
            blocks_expiry_time=kv_transfer_params.get(
                "remote_blocks_expiry_time"),
        )
        self.reqs_to_recv[request_id] = req


@dataclass
class _HixlEngineHeartbeatInfo:
    """Per-remote-engine heartbeat bundle (fork of NIXL HeartbeatInfo)."""

    req_ids: set[str] = field(default_factory=set)
    host: str = ""
    port: int = 0
    tp_size: int = 1
    pp_size: int = 1


class HIXLEngineConnectorScheduler:
    """Scheduler-side decisions for the HIXLEngine pull connector.

    Forked from NIXL NixlPullConnectorScheduler (pull_scheduler.py:23-280)
    and NixlBaseConnectorScheduler (base_scheduler.py:51-167,402-437). Does
    NOT inherit NIXL: the single ROUTER handshake listener and the relaxed
    set_xfer signature (vllm-ascend CP>1 yields 3-tuple keys) live on
    HIXLEngineConnector itself. Only the four decision methods, the metadata
    builder, and heartbeat bookkeeping live here.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
        side_channel_port: int,
    ):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id = engine_id
        self.kv_cache_config = kv_cache_config
        self.side_channel_host = get_ip()
        self.side_channel_port = side_channel_port
        kvtc = vllm_config.kv_transfer_config
        assert kvtc is not None
        self._kv_lease_duration: int = kvtc.get_from_extra_config(
            "kv_lease_duration", 30)
        self._heartbeat_interval = self._kv_lease_duration // 6
        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.kv_cache_groups
            )
        )
        self._has_mamba = any(
            isinstance(g.kv_cache_spec, MambaSpec)
            for g in kv_cache_config.kv_cache_groups
        )
        # Compress-aware truncation. Mamba state groups already force N-1
        # truncation; models with compress_ratios (e.g. hybrid
        # attention/compress) need it too. Forked from hixl_connector L1316-1319.
        self._use_compress = self._model_uses_compress()
        self._need_truncate = self._use_compress or self._has_mamba
        self._reqs_need_recv: dict[str, tuple["Request", BlockIds]] = {}
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_in_batch: set[str] = set()
        self._reqs_not_processed: set[str] = set()
        self._heartbeat_by_engine: dict[str, _HixlEngineHeartbeatInfo] = {}
        self._heartbeat_req_engine: dict[str, tuple[str, str]] = {}
        self._last_heartbeat_time: float = 0.0
        sw_sizes: list[tuple[int, int]] = [
            (g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec)
            else (0, self.block_size)
            for g in kv_cache_config.kv_cache_groups
        ]
        self.blocks_per_sw = [
            cdiv(n, b) + 1 if n else 0 for n, b in sw_sizes
        ]
        self.kv_recompute_threshold = int(
            kvtc.get_from_extra_config("kv_recompute_threshold", 64)
        )
        self.is_bidirectional_kv_xfer_enabled = (
            kvtc.get_from_extra_config("bidirectional_kv_xfer", False)
        )
        self.decoder_kv_blocks_ttl = kvtc.get_from_extra_config(
            "decoder_kv_blocks_ttl", 480
        )
        logger.info("Initializing HIXLEngine scheduler %s", engine_id)

    # -- heartbeat bookkeeping (fork base_scheduler.py:175-219) ----------
    def on_new_request(self, request: "Request") -> None:
        params = request.kv_transfer_params
        if params is None or not params.get("do_remote_prefill"):
            return
        remote_engine_id = params.get("remote_engine_id")
        remote_request_id = params.get("remote_request_id")
        host = params.get("remote_host")
        port = params.get("remote_port")
        tp_size = params.get("tp_size")
        pp_size = params.get("pp_size", 1)
        if None in (remote_engine_id, remote_request_id,
                   host, port, tp_size):
            return
        if remote_engine_id not in self._heartbeat_by_engine:
            self._heartbeat_by_engine[remote_engine_id] = (
                _HixlEngineHeartbeatInfo(
                    host=host, port=port, tp_size=tp_size, pp_size=pp_size)
            )
        self._heartbeat_by_engine[remote_engine_id].req_ids.add(
            remote_request_id)
        self._heartbeat_req_engine[request.request_id] = (
            remote_engine_id, remote_request_id)

    def _stop_heartbeat(self, req_id: str) -> None:
        if key := self._heartbeat_req_engine.pop(req_id, None):
            engine_id, remote_id = key
            if info := self._heartbeat_by_engine.get(engine_id):
                info.req_ids.discard(remote_id)
                if not info.req_ids:
                    del self._heartbeat_by_engine[engine_id]

    # -- SWA clipping (fork base_scheduler.py:221-246) -------------------
    def get_sw_clipped_blocks(self, block_ids: BlockIds) -> BlockIds:
        if len(block_ids) == 0 or not self._is_hma_required:
            return block_ids
        assert len(block_ids) == len(self.blocks_per_sw)
        return tuple(
            blocks[-self.blocks_per_sw[i]:] if self.blocks_per_sw[i] > 0
            else blocks
            for i, blocks in enumerate(block_ids)
        )

    # -- truncate helpers (fork hixl_connector L1337-1369) ---------------
    # Covers both Mamba state groups and compress-ratio models.
    def _model_uses_compress(self) -> bool:
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        compress_ratios = getattr(hf_config, "compress_ratios", None)
        return isinstance(compress_ratios, (list, tuple, dict))

    def _get_remote_prefill_token_count(
        self, num_prompt_tokens: int
    ) -> int:
        # D-side: Mamba / compress models recompute the last token from
        # h(N-1), so only N-1 prompt tokens are pulled from the P side.
        if self._need_truncate and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_request_for_prefill(
        self, request: "Request"
    ) -> None:
        """P-side: drop the last prompt token so the prefiller computes
        h(N-1). The decoder recomputes the last token to derive h(N).
        Guarded by _p_side_truncated against preempt-reschedule repeats.
        Applies to Mamba state groups and compress-ratio models."""
        params = request.kv_transfer_params
        if (params is not None
                and not params.get("_p_side_truncated")
                and request.num_prompt_tokens > 1):
            if request.prompt_token_ids is not None:
                request.prompt_token_ids.pop()
            elif request.prompt_embeds is not None:
                request.prompt_embeds = request.prompt_embeds[:-1]
            else:
                return
            request._all_token_ids.pop()
            request.num_prompt_tokens -= 1
            request.max_tokens = 1
            params["_p_side_truncated"] = True

    # -- four decision methods (fork pull_scheduler.py:34-280) ----------
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            token_ids = request.prompt_token_ids or []
            actual = self._get_remote_prefill_token_count(len(token_ids))
            count = actual - num_computed_tokens
            if count > 0:
                return count, True
        if (params is not None and params.get("do_remote_decode")
                and self._need_truncate):
            self._truncate_request_for_prefill(request)
        if (params is not None and params.get("do_remote_decode")
                and params.get("remote_block_ids")
                and all(p in params for p in (
                    "remote_engine_id", "remote_request_id",
                    "remote_host", "remote_port"))):
            remote_num_tokens = params.get("remote_num_tokens") or 0
            count = (min(remote_num_tokens, request.num_prompt_tokens)
                     - num_computed_tokens)
            if count > 0:
                if (self.kv_recompute_threshold > 0
                        and count < self.kv_recompute_threshold):
                    logger.debug(
                        "Skipping remote pull for %s: %d < threshold %d",
                        request.request_id, count,
                        self.kv_recompute_threshold)
                    return 0, False
                return count, True
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        params = request.kv_transfer_params
        if not params:
            return
        if params.get("do_remote_decode") or (
            params.get("do_remote_prefill")
            and self.is_bidirectional_kv_xfer_enabled
        ):
            self._reqs_in_batch.add(request.request_id)
        # HIXLEngine transfers device memory directly (no host-buffer staging
        # path), so the NIXL use_host_buffer / _reqs_need_save branch is dropped.
        if params.get("do_remote_prefill") or (
            params.get("do_remote_decode")
            and self.is_bidirectional_kv_xfer_enabled
            and not params.get("_remote_blocks_processed")
        ):
            if params.get("remote_block_ids"):
                if all(p in params for p in (
                        "remote_engine_id", "remote_request_id",
                        "remote_host", "remote_port")):
                    unhashed = (
                        blocks.get_unhashed_block_ids_all_groups()
                        if num_external_tokens > 0 else ())
                    local_block_ids = self.get_sw_clipped_blocks(unhashed)
                    self._reqs_need_recv[request.request_id] = (
                        request, local_block_ids)
                else:
                    logger.error(
                        "Got invalid KVTransferParams: %s. This request "
                        "will not utilize KVTransfer", params)
            else:
                assert num_external_tokens == 0
            params["do_remote_prefill"] = False
            params["_remote_blocks_processed"] = True

    def request_finished(
        self, request: "Request", block_ids: BlockIds
    ) -> tuple[bool, dict[str, Any] | None]:
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        if not params:
            return False, None
        is_p_node = bool(params.get("do_remote_decode"))
        is_d_node = not is_p_node
        self._stop_heartbeat(request.request_id)
        if params.get("do_remote_prefill"):
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None
        if is_d_node and not self.is_bidirectional_kv_xfer_enabled:
            return False, None
        if request.status not in (RequestStatus.FINISHED_LENGTH_CAPPED,
                                  RequestStatus.FINISHED_STOPPED):
            self._reqs_not_processed.add(request.request_id)
            return False, None
        delay_free_blocks = any(len(group) > 0 for group in block_ids)
        remote_num_tokens = 0
        blocks_expiry_time = None
        if delay_free_blocks:
            request_kv_blocks_ttl = self._kv_lease_duration
            if is_d_node:
                request_kv_blocks_ttl = self.decoder_kv_blocks_ttl
            self._reqs_need_send[request.request_id] = (
                time.perf_counter() + request_kv_blocks_ttl)
            if is_d_node:
                blocks_expiry_time = self._reqs_need_send[request.request_id]
            block_ids = self.get_sw_clipped_blocks(block_ids)
            remote_num_tokens = request.num_computed_tokens
        logger.debug(
            "HIXLTRACE P-request_finished req=%s is_p=%d delay_free=%d "
            "remote_port=%d n_blocks=%s",
            request.request_id, is_p_node, delay_free_blocks,
            self.side_channel_port,
            [len(g) for g in block_ids] if block_ids else [],
        )
        return delay_free_blocks, dict(
            do_remote_prefill=is_p_node,
            do_remote_decode=is_d_node,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            remote_num_tokens=remote_num_tokens,
            remote_blocks_expiry_time=blocks_expiry_time,
        )

    # -- metadata builder (fork base_scheduler.py:402-437) ---------------
    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> "KVConnectorMetadata":
        meta = HIXLEngineConnectorMetadata()
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )
        meta.reqs_to_send = self._reqs_need_send
        meta.reqs_in_batch = self._reqs_in_batch
        meta.reqs_not_processed = self._reqs_not_processed
        if self._heartbeat_by_engine:
            now = time.perf_counter()
            if now - self._last_heartbeat_time >= self._heartbeat_interval:
                self._last_heartbeat_time = now
                meta.heartbeat_by_engine = self._heartbeat_by_engine
        self._reqs_need_recv.clear()
        self._reqs_in_batch = set()
        self._reqs_not_processed = set()
        self._reqs_need_send = {}
        return meta

    def update_connector_output(
        self, connector_output: "KVConnectorOutput"
    ) -> None:
        for req_id in connector_output.finished_recving or ():
            self._stop_heartbeat(req_id)

    def has_pending_push_work(self) -> bool:
        return False


class HIXLEngineConnector(KVConnectorBase_V1, SupportsHMA):
    """Pull-mode KV connector backed by hixl::Hixl address-level transfer.

    One instance per rank. Holds a hixl::Hixl instance, the
    registered mem handles for local KV layers, and a set of in-flight
    TransferAsync request handles to poll.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._vllm_config = vllm_config
        self._kv_cache_config = kv_cache_config
        self._parse_hixl_engine_config(vllm_config)

        # Parallel / model identity (mirrors NIXL base_worker.py:470-510).
        kvtc = vllm_config.kv_transfer_config
        self._engine_id = str(kvtc.engine_id)
        if role == KVConnectorRole.SCHEDULER:
            # SCHEDULER 在 EngineCore 主进程创建,主进程未初始化并行组,
            # 调 get_tensor_model_parallel_rank() 会触发 "_TP is not None"
            # 断言。SCHEDULER 回调委托 HIXLEngineConnectorScheduler,不消费
            # tp_rank 等字段,故设默认值跳过。_local_engine_endpoint 保持
            # base(SCHEDULER 不 bind,握手 payload 由 WORKER 填)。
            self._tp_rank = 0
            self._tp_size = vllm_config.parallel_config.tensor_parallel_size
            self._world_size = 1
            self._pp_rank = 0
            self._pp_size = vllm_config.parallel_config.pipeline_parallel_size
            self._pcp_size = 1
            self._pcp_rank = 0
            self._dcp_size = 1
            self._dcp_rank = 0
        else:
            self._tp_rank = get_tensor_model_parallel_rank()
            self._tp_size = vllm_config.parallel_config.tensor_parallel_size
            # Offset the hixl listen port by tp_rank so co-located TP ranks
            # each bind a distinct port (avoids the 503900 bind failure when
            # the second rank rebinds the same port). The peer reads the
            # actual port from local_engine_endpoint in the handshake
            # metadata, so P/D stay aligned without per-rank config. TP=1
            # offsets by 0 (no change).
            # TODO: extend the offset for DP>1 (currently DP=1, matching
            # side_channel_port which also offsets by data_parallel_index only).
            if self._local_engine_base_port > 0:
                self._local_engine_endpoint = (
                    f"{self._local_engine_host}:"
                    f"{self._local_engine_base_port + self._tp_rank}"
                )
            self._world_size = get_tensor_model_parallel_world_size()
            self._pp_rank = get_pp_group().rank_in_group
            self._pp_size = vllm_config.parallel_config.pipeline_parallel_size
            # CP parallel-state fields. Forked from hixl_connector L1564-1572.
            # get_pcp_group is already imported (L43); DCP helpers are imported
            # lazily to avoid a hard dependency when CP is unused. The
            # single-listener handshake model (one ROUTER per (engine_id,
            # dp_index), routing by (pp, tp)) means CP shards share the
            # listener endpoint, so the old per-rank multi-port offset is
            # eliminated by this architecture (TODO: per-pcp routing if
            # non-HMA CP is ever required).
            self._pcp_size = get_pcp_group().world_size
            self._pcp_rank = (
                get_pcp_group().rank_in_group if self._pcp_size > 1 else 0)
            from vllm.distributed import get_dcp_group
            _dcp_group = get_dcp_group()
            self._dcp_size = _dcp_group.world_size
            self._dcp_rank = (
                _dcp_group.rank_in_group if self._dcp_size > 1 else 0)
        assert not (self._pp_size > 1 and self._pcp_size > 1), (
            "HIXLEngineConnector: pp and pcp cannot be enabled at the "
            "same time."
        )
        # SCHEDULER role 在 EngineCore 主进程创建,无并行组与
        # set_current_vllm_config() context(worker 进程经
        # init_model_parallel 后才有)。下面 worker 初始化
        # (get_current_attn_backends / get_kv_cache_layout /
        # _sync_block_size_with_kernel / mamba conv decomp)依赖这些
        # context,主进程必崩。SCHEDULER 回调委托独立
        # HIXLEngineConnectorScheduler,不消费 worker 字段,故对齐
        # nixl __init__:SCHEDULER 只建 scheduler 与握手路由表后返回,
        # 跳过整个 worker 数据面初始化。
        # (pp_rank, tp_rank) -> encoded HixlEngineHandshakePayload. Filled on
        # the SCHEDULER side by set_xfer_handshake_metadata[_pp_aware] (L2843+)
        # from the payloads each worker produced in register_kv_caches; the
        # single listener routes GET_META requests by (pp, tp) (NIXL
        # base_scheduler.py:316-322).
        # P-side handshake ROUTER lifecycle + handshake serialization. Both
        # roles build these: SCHEDULER runs the listener thread and stores
        # handshake payloads; WORKER needs them present so shutdown() can join
        # a (never-started, None) listener and stop the executor without
        # AttributeError (the SCHEDULER early-return below skips the worker
        # data-plane init that originally created these, so they must precede it).
        self._handshake_initiation_executor = ThreadPoolExecutor(max_workers=1)
        self._handshake_lock = threading.RLock()
        self._handshake_stop_event = threading.Event()
        self._handshake_listener_thread: threading.Thread | None = None
        self._handshake_payloads: dict[tuple[int, int], bytes] = {}
        if role == KVConnectorRole.SCHEDULER:
            # NIXL-style single scheduler-side ROUTER listener. The base port
            # comes from hixl_engine.side_channel_port; data_parallel_index
            # separates DP groups (NIXL base_scheduler.py:64-68). The scheduler
            # is constructed with this port directly (no override hack), and
            # its request_finished writes it into kv_transfer_params.remote_port
            # so the D-side REQ connect lands on the same port the listener
            # binds.
            self._side_channel_port: int = (
                self._side_channel_port_base
                + vllm_config.parallel_config.data_parallel_index
            )
            self._scheduler: HIXLEngineConnectorScheduler | None = (
                HIXLEngineConnectorScheduler(
                    vllm_config, self._engine_id, kv_cache_config,
                    self._side_channel_port,
                )
            )
            return
        self._scheduler = None
        self._side_channel_port = 0
        self._model_config = vllm_config.model_config
        self._use_mla = self._model_config.is_deepseek_mla
        self._num_kv_heads = self._model_config.get_total_num_kv_heads()
        self._head_size = self._model_config.get_head_size()
        self._num_hidden_layers = self._model_config.get_total_num_hidden_layers()
        self._attn_backends = get_current_attn_backends(vllm_config)
        self._attn_backend_name = self._attn_backends[0].get_name()
        self._kv_cache_layout = get_kv_cache_layout()

        self._block_size = vllm_config.cache_config.block_size
        self._num_blocks = kv_cache_config.num_blocks if kv_cache_config else 0
        self._logical_num_blocks = self._num_blocks
        self._physical_blocks_per_logical_kv_block = 1
        self._sync_block_size_with_kernel()

        # Layer specs from kv_cache_groups (mirrors NIXL _layer_specs).
        self._layer_specs: dict[str, Any] = {}
        self._group_spec_types: tuple[type, ...] = ()
        self._layer_to_group: dict[str, int] = {}
        if kv_cache_config is not None:
            self._layer_specs = {
                layer: group.kv_cache_spec
                for group in kv_cache_config.kv_cache_groups
                for layer in group.layer_names
            }
            self._group_spec_types = tuple(
                self._representative_spec_type(group.kv_cache_spec)
                for group in kv_cache_config.kv_cache_groups
            )
            self._layer_to_group = {
                layer: group_idx
                for group_idx, group in enumerate(kv_cache_config.kv_cache_groups)
                for layer in group.layer_names
            }

        # Per-region bookkeeping (mirrors NIXL block_len_per_layer /
        # kv_caches_base_addr[engine_id][tp_rank]).
        self._block_len_per_layer: list[int] = []
        self._region_is_mla: list[bool] = []
        self._region_group_idx: list[int] = []
        self._kv_caches_base_addr: dict[str, dict[int, list[int]]] = {
            self._engine_id: {self._tp_rank: []}
        }
        self._device_id: int = 0
        # NZ layout reformat fallback. Filled from AscendConfig in
        # register_kv_caches. When HCCL cannot scatter-write NZ offsets the
        # D cache is ND after transfer and must be reformatted to NZ.
        self._enable_kv_nz: bool = False
        # layer_name -> KV tensor(s). Saved in register_kv_caches so
        # _apply_nz_reformat can operate on the D real cache post-transfer.
        self._kv_caches: dict[str, Any] = {}
        self._has_mamba: bool = any(
            self._is_ssm_spec(t) for t in self._group_spec_types
        )
        # Conv state sub-projection decomposition (None when no Mamba). Mirrors
        # NIXL base_worker.py:292-321. ssm_sizes is shipped in the handshake so
        # the peer can address conv/ssm regions; the per-region conv offsets are
        # consumed by _build_op_descs' SSM branch (TODO(conv-decomp): port the
        # remote_conv_offsets addressing once a Mamba model is available to
        # validate the DS-layout assumption on NPU).
        self._conv_decomp: MambaConvSplitInfo | None = None
        mamba_ssm_size: tuple[int, int] = (0, 0)
        if self._has_mamba:
            from vllm.model_executor.layers.mamba.mamba_utils import (
                is_conv_state_dim_first,
            )

            mamba_spec = next(
                spec
                for spec in self._layer_specs.values()
                if isinstance(spec, MambaSpec)
            )
            if is_conv_state_dim_first():
                # DS 布局:走 NIXL 3-read 子投影分解取 ssm_sizes。hixl 尚未
                # 移植子投影传输路径(_build_op_descs 走整块线性寻址,见
                # TODO),保留 decomp 备后续移植。
                self._conv_decomp = derive_mamba_conv_split(
                    mamba_spec, self._tp_size
                )
                mamba_ssm_size = self._conv_decomp.ssm_sizes
            else:
                # SD 布局:Ascend npu_causal_conv1d_custom 要求
                # convStates=(num_cache_lines, state_len, dim),与 DS 断言
                # 冲突。hixl 整块传输与 ssm_sizes 均布局无关,故 SD 下不
                # 强制 DS,不调 derive_mamba_conv_split(其内部断言要求 DS),
                # 仅按 numel*dtype_size 算 ssm_sizes。P_TP==D_TP 整块 memcpy
                # 安全;P_TP>D_TP reshard 下 slot*chunk 线性寻址假设 DS,SD
                # 未验证。
                conv_dt = torch.tensor(
                    [], dtype=mamba_spec.dtypes[0]
                ).element_size()
                ssm_dt = torch.tensor(
                    [], dtype=mamba_spec.dtypes[1]
                ).element_size()
                conv_state_bytes = (
                    torch.Size(mamba_spec.shapes[0]).numel() * conv_dt
                )
                ssm_state_bytes = (
                    torch.Size(mamba_spec.shapes[1]).numel() * ssm_dt
                )
                mamba_ssm_size = (conv_state_bytes, ssm_state_bytes)
                logger.warning(
                    "HIXLEngine running with SD conv state layout. Safe when "
                    "P_TP == D_TP (whole-block memcpy); P_TP > D_TP reshard "
                    "with SD is unverified."
                )
        self._mamba_ssm_size: tuple[int, int] = mamba_ssm_size
        # Local transfer topology; built lazily in register_kv_caches (mirrors
        # NIXL base_worker.py:1041-1054). Used by compute_tp_mapping.
        self._transfer_topo: TransferTopology | None = None
        # engine_id -> last-seen perf_counter time (for TTL eviction).
        self._remote_engine_last_seen: dict[str, float] = {}
        self._engine_ttl_s: float = float(
            vllm_config.kv_transfer_config.get_from_extra_config("engine_ttl", 3600.0)
        )

        self._hixl_mod = _load_hixl()
        self._hixl = self._hixl_mod.Hixl()
        self._hixl_initialized = False
        # hixl_py serializes calls internally (C++ mutex + GIL release), but
        # connect (handshake thread) and transfer_async / get_transfer_status
        # (main worker thread) share this Python-side lock so the call sequence
        # per engine stays ordered.
        self._hixl_lock = threading.Lock()
        # layer_name -> (mem_handle, base_addr, length)
        self._kv_mem_handles: dict[str, tuple[int, int, int]] = {}
        # request_id -> list[TransferAsync handle]
        self._recving_transfers: dict[str, list[int]] = {}
        # request_id -> submitted bytes / first-submit perf_counter.
        self._xfer_bytes: dict[str, int] = {}
        self._xfer_start: dict[str, float] = {}
        # Started lazily on first enqueue so the SCHEDULER role never spawns it.
        self._notify_queue: "queue.Queue[tuple[str, str, str] | None]" = (
            queue.Queue(maxsize=self._notify_queue_size))
        self._notify_thread: threading.Thread | None = None
        self._notify_thread_lock = threading.Lock()
        # request_id -> HIXLEngineReqMeta (for failure recovery / post-process)
        self._recving_metadata: dict[str, HIXLEngineReqMeta] = {}
        # Reqs whose handshake hadn't landed yet are parked on
        # _pending_handshake_reqs per engine; the handshake done_callback
        # releases them into _ready_requests, drained next step (NIXL
        # pull_worker.py:66-79).
        self._ready_requests: deque[tuple[str, "HIXLEngineReqMeta"]] = deque()
        self._pending_handshake_reqs: dict[
            str, list[tuple[str, "HIXLEngineReqMeta"]]
        ] = {}
        # Ranks already sent a prefix-hit DONE in _read_blocks; _notify_release
        # skips them to avoid a double-notify.
        self._notified_release_ranks: dict[str, set[int]] = {}
        # Ranks this request actually issued an async READ to; _notify_release
        # sends DONE to these real readers only.
        self._transferred_ranks: dict[str, set[int]] = {}
        # request_id -> failed flag (for get_block_ids_with_load_errors)
        self._failed_recv_reqs: set[str] = set()
        self._invalid_block_ids: set[int] = set()
        self._task_tracker = KVCacheTaskTracker()
        # Remote engine metadata, populated by ZMQ handshake.
        self._remote_metadata: dict[str, HixlEngineAgentMetadata] = {}
        self._tp_mappings: dict[str, TPMapping] = {}
        # engine_id -> (block_size_ratio, remote_physical_per_logical,
        # d_ssm_group_count). Handshake-invariant inputs to _build_op_descs,
        # memoized there on first use and dropped by _cleanup_remote_engine.
        self._plan_invariants: dict[str, tuple[int, int, int]] = {}
        # engine_id -> {(pp_rank, tp_rank): agent_name} (hixl Connect target).
        self._remote_agents: dict[str, dict[tuple[int, int], str]] = {}
        # engine_id -> perf_counter midpoint offset (mirrors NIXL).
        self._engine_clock_offset: dict[str, float] = {}
        # Handshake payload produced in register_kv_caches (P side) / consumed
        # by get_handshake_metadata. None until register_kv_caches runs.
        self._xfer_handshake_metadata: HixlEngineHandshakePayload | None = None
        self._compat_hash: str | None = None
        # ZMQ handshake futures. The executor and _handshake_lock are built
        # above (before the SCHEDULER early-return) so both roles own them;
        # only WORKER populates _handshake_futures (D-side REQ connect).
        self._handshake_futures: dict[str, Future] = {}
        # req_id -> perf_counter lease expiry (P-side delayed free).
        self._reqs_to_send: dict[str, float] = {}
        # Multi-consumer done-notification counting (mirrors NIXL).
        self._consumer_notification_counts_by_req: dict[str, int] = defaultdict(int)
        # P-side lease / heartbeat timing (mirrors NIXL base_worker). Used by
        # _get_new_notifs (lease expiry) and _handle_heartbeat (extension).
        self._kv_lease_duration: int = kvtc.get_from_extra_config(
            "kv_lease_duration", 30)
        # 2/3 factor (NIXL base_worker.py:268): heartbeats only extend the
        # lease when its remaining < _lease_extension, so the lease converges
        # to now+extension instead of growing unboundedly on each heartbeat.
        self._lease_extension: int = self._kv_lease_duration * 2 // 3
        # (P-side handshake ROUTER lifecycle — _handshake_stop_event and
        # _handshake_listener_thread — built above, before the early-return.)
        # (SCHEDULER early-return + _handshake_payloads + _scheduler/
        #  _side_channel_port defaults are set above, right after the
        #  pp/pcp assert; the worker role continues below.)

    # ------------------------------------------------------------------
    # Config & kernel-block-size derivation (mirrors NIXL base_worker.py
    # _sync_block_size_with_kernel and HIXLConnector._extra_options).
    # ------------------------------------------------------------------
    # ==================================================================
    # hixl_py adapter: thin inlined shim over the address-level ``hixl``
    # module (import hixl, built from src/python/hixl_py/hixl_py.cc). hixl_py
    # returns (Status, T) tuples instead of raising; _hixl_check turns non-
    # SUCCESS into RuntimeError so the connector's try/except control flow
    # (failure recovery / _handle_failed_transfer) is unchanged. Struct
    # construction (MemDesc / NotifyDesc) and op-enum lookup live here too.
    # ==================================================================
    def _hixl_check(self, status: int, ctx: str) -> None:
        if status != self._hixl_mod.SUCCESS:
            raise RuntimeError(f"HIXLEngine {ctx} failed, code={status}")

    def _hixl_op(self, op: str):
        if op not in ("READ", "WRITE"):
            raise ValueError(f"Unsupported TransferOp: {op!r}")
        return getattr(self._hixl_mod.TransferOp, op)

    def _hixl_initialize(self, local_engine: str, options: dict[str, str]) -> None:
        # EngineFactory 的 selected engine 走 CANN slog，不会进 vllm 日志。
        logger.info(
            "HIXLEngine Initialize backend=%s requested=%s injected=%s "
            "local_engine=%s options=%s",
            self._engine_backend,
            self._engine_backend_requested,
            self._engine_backend_injected,
            local_engine,
            options,
        )
        status = self._hixl.initialize(local_engine, options)
        self._hixl_check(status, "Initialize")
        self._hixl_initialized = True

    def _hixl_finalize(self) -> None:
        self._hixl.finalize()
        self._hixl_initialized = False

    def _hixl_register_mem(self, addr: int, length: int, is_device: bool = True) -> int:
        mod = self._hixl_mod
        mt = mod.MemType.MEM_DEVICE if is_device else mod.MemType.MEM_HOST
        status, handle = self._hixl.register_mem(mod.MemDesc(addr, length), mt)
        self._hixl_check(status, "RegisterMem")
        return handle

    def _hixl_deregister_mem(self, handle: int) -> None:
        status = self._hixl.deregister_mem(handle)
        self._hixl_check(status, "DeregisterMem")

    def _hixl_connect(self, remote_engine: str, timeout_ms: int = 1000) -> None:
        status = self._hixl.connect(remote_engine, timeout_ms)
        self._hixl_check(status, "Connect")

    def _hixl_disconnect(self, remote_engine: str, timeout_ms: int = 1000) -> None:
        status = self._hixl.disconnect(remote_engine, timeout_ms)
        self._hixl_check(status, "Disconnect")

    def _hixl_transfer_async(self, remote_engine: str, op: str, op_descs: list) -> int:
        status, req = self._hixl.transfer_async(
            remote_engine, self._hixl_op(op), op_descs)
        self._hixl_check(status, "TransferAsync")
        return req

    def _hixl_get_transfer_status(self, req: int):
        # "not-found" contract: hixl drops its record on ANY terminal status
        # (COMPLETED / FAILED / TIMEOUT) and on any channel error, after which
        # a re-query returns PARAM_INVALID (103900). None therefore means
        # "record gone", NOT "completed" — a caller that observed a failure
        # must record it before the handle is re-queried.
        #
        # Order matters: on the PARAM_INVALID path hixl also writes FAILED into
        # the status out-param, so testing `st` before `status` would misread
        # an already-consumed handle as a hard failure.
        status, st = self._hixl.get_transfer_status(req)
        if status == self._hixl_mod.PARAM_INVALID:
            return None
        self._hixl_check(status, "GetTransferStatus")
        return st

    def _hixl_send_notify(self, remote_engine: str, name: str, msg: str,
                         timeout_ms: int = 1000) -> None:
        nd = self._hixl_mod.NotifyDesc(name=name, notify_msg=msg)
        status = self._hixl.send_notify(remote_engine, nd, timeout_ms)
        self._hixl_check(status, "SendNotify")

    def _hixl_get_notifies(self) -> list[tuple[str, str]]:
        status, ns = self._hixl.get_notifies()
        self._hixl_check(status, "GetNotifies")
        return [(n.name, n.notify_msg) for n in ns]

    # ------------------------------------------------------------------
    def _parse_hixl_engine_config(self, vllm_config: VllmConfig) -> None:
        """Read kv_connector_extra_config.hixl_engine (plan §2.5).

        Fields: local_engine (host:port for hixl Initialize), backend
        (hixl_cs|comm, default hixl_cs), options (dict passed through to
        Initialize after backend injection), link_timeout_ms,
        transfer_timeout_ms, side_channel_port (base ZMQ handshake port;
        the scheduler-side single ROUTER listener binds base +
        data_parallel_index, mirroring NIXL base_scheduler.py:64-68 — the
        D side learns it via the remote_port field that P's
        request_finished writes into kv_transfer_params).
        """
        kvtc = vllm_config.kv_transfer_config
        cfg: dict[str, Any] = kvtc.get_from_extra_config("hixl_engine", {})
        raw_backend = cfg.get("backend")
        if raw_backend is None or (
                isinstance(raw_backend, str) and not raw_backend.strip()):
            self._engine_backend_requested = "default"
            backend = _HIXL_ENGINE_BACKEND_CS
        else:
            backend = str(raw_backend).strip()
            self._engine_backend_requested = backend
        if backend not in _HIXL_ENGINE_BACKENDS:
            raise ValueError(
                "hixl_engine.backend must be 'hixl_cs' or 'comm', "
                f"got {backend!r}."
            )
        options = _normalize_hixl_engine_options(cfg.get("options", {}))
        options, injected = _apply_hixl_engine_backend_options(backend, options)
        self._engine_backend = backend
        self._engine_backend_injected = injected
        self._engine_options: dict[str, str] = options
        self._link_timeout_ms: int = int(cfg.get("link_timeout_ms", 5000))
        self._transfer_timeout_ms: int = int(cfg.get("transfer_timeout_ms", 60_000))
        # Bounds how long one notify can freeze the data plane: SendNotify
        # holds hixl_py's process-wide C++ mutex until the peer ACKs. hixl's
        # own default (1000ms) is far too long for a call on the KV path;
        # DONE loss is covered by the P-side lease, a lost HB is re-sent.
        self._notify_timeout_ms: int = int(cfg.get("notify_timeout_ms", 100))
        self._notify_queue_size: int = int(cfg.get("notify_queue_size", 1024))
        # Opt-in only, to A/B the two behaviours without a rebuild; see
        # wait_for_layer_load for why blocking is not needed.
        self._blocking_layer_wait: bool = bool(
            cfg.get("blocking_layer_wait", False))
        self._side_channel_port_base: int = int(cfg.get("side_channel_port", 0))
        raw_local_engine: str = cfg.get("local_engine", "")
        if not raw_local_engine:
            raise ValueError(
                "HIXLEngineConnector requires kv_connector_extra_config."
                "hixl_engine.local_engine (host:port for hixl Initialize)."
            )
        # Split host:port now; the per-rank endpoint is resolved in __init__
        # once tp_rank is known (co-located TP ranks must not share one
        # listen port, or the second rank's Hixl Initialize bind fails with
        # 503900). The peer learns the actual port from the local_engine_endpoint
        # field in the handshake metadata, so P/D stay aligned without per-rank
        # config.
        if ":" in raw_local_engine:
            self._local_engine_host, port_str = raw_local_engine.rsplit(":", 1)
            self._local_engine_base_port: int = int(port_str)
        else:
            self._local_engine_host = raw_local_engine
            self._local_engine_base_port = 0
        self._local_engine_endpoint: str = raw_local_engine

    def _sync_block_size_with_kernel(self) -> None:
        """Align block_size to the kernel's physical block size.

        Mirrors NIXL base_worker.py:546-563. If the user block_size is larger
        than the kernel block size, one logical block spans multiple physical
        blocks; num_blocks is scaled up accordingly so addressing by
        physical block id stays correct in _build_op_descs.
        """
        kernel_block_size = select_common_block_size(
            self._block_size, self._attn_backends
        )
        self._logical_num_blocks = self._num_blocks
        if self._block_size != kernel_block_size:
            assert self._block_size > kernel_block_size, (
                f"block_size {self._block_size} < kernel {kernel_block_size}"
            )
            self._physical_blocks_per_logical_kv_block = (
                self._block_size // kernel_block_size
            )
            self._block_size = kernel_block_size
            self._num_blocks *= self._physical_blocks_per_logical_kv_block

    @staticmethod
    def _is_attention_spec(spec_type: type) -> bool:
        return issubclass(spec_type, AttentionSpec)

    @staticmethod
    def _is_ssm_spec(spec_type: type) -> bool:
        return issubclass(spec_type, MambaSpec)

    @staticmethod
    def _representative_spec_type(spec: Any) -> type:
        """Unwrap UniformTypeKVCacheSpecs to its representative sub-spec type.

        Mirrors NIXL get_representative_spec_type so _has_mamba still detects
        SSM when the group is a uniform-type bundle (e.g. MLA DSv32 Indexer).
        """
        if isinstance(spec, UniformTypeKVCacheSpecs):
            return type(next(iter(spec.kv_cache_specs.values())))
        return type(spec)

    # ==================================================================
    # ZMQ side-channel handshake (D-side REQ client). Mirrors NIXL
    # base_worker.py:565-711 _nixl_handshake. The P-side ROUTER is the
    # self._handshake_listener_thread started in set_xfer_handshake_metadata
    # (Step R fork); here we only implement the D-side fetch + two-stage
    # decode.
    # ==================================================================
    def _hixl_engine_handshake(
        self,
        remote_host: str,
        remote_handshake_port: int,
        remote_tp_size: int,
        expected_engine_id: str,
    ) -> tuple[list[tuple[int, int, HixlEngineAgentMetadata]], float]:
        """Fetch peer metadata for every remote TP rank this local rank reads.

        Heterogeneous TP needs multiple handshakes: when remote_tp_size >
        local tp_size (P_TP>D_TP gather), one D rank reads from several P
        ranks and must collect each rank's base addresses. NIXL does this in
        a single background job (base_worker.py:589-711); HIXLEngine mirrors
        it: loop handshake_target_ranks over one ZMQ REQ socket, keep the
        lowest-RTT clock-offset sample. Returns ([(pp_rank, tp_rank, meta)],
        offset). Homogeneous/split TP yields a single-element list.
        """
        assert self._transfer_topo is not None
        p_remote_ranks = self._transfer_topo.handshake_target_ranks(
            remote_tp_size
        )
        path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        agent_decoder = msgspec.msgpack.Decoder(HixlEngineAgentMetadata)
        payload_decoder = msgspec.msgpack.Decoder(HixlEngineHandshakePayload)
        results: list[tuple[int, int, HixlEngineAgentMetadata]] = []
        best_rtt = float("inf")
        best_offset: float | None = None

        with zmq_ctx(zmq.REQ, path) as sock:  # type: ignore[arg-type]
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, int(_HIXL_ENGINE_REQ_TIMEOUT_S * 1000))
            for remote_rank in p_remote_ranks:
                remote_pp_rank = 0  # pull mode: single pipeline stage
                req_msg = msgspec.msgpack.encode(
                    (GET_META_MSG, remote_pp_rank, remote_rank)
                )
                start = time.perf_counter()
                sock.send_multipart((req_msg,))
                reply = sock.recv_multipart()
                recv = time.perf_counter()
                if len(reply) < 2:
                    raise RuntimeError(
                        f"HIXLEngine handshake short reply: {len(reply)} frames"
                    )
                handshake_bytes = reply[0]
                if handshake_bytes.startswith(HIXL_ERR_PREFIX):
                    raise RuntimeError(
                        "HIXLEngine handshake rejected by remote: "
                        f"{handshake_bytes[len(HIXL_ERR_PREFIX):].decode(errors='replace')}"
                    )
                remote_perf = msgspec.msgpack.decode(reply[1])
                # perf_counter midpoint clock-offset estimate; keep the
                # lowest-RTT sample (NIXL base_worker.py:628-631).
                rtt = recv - start
                if rtt < best_rtt:
                    best_rtt = rtt
                    best_offset = remote_perf - (start + recv) / 2

                # Two-stage decode: hash first, then metadata.
                payload = payload_decoder.decode(handshake_bytes)
                if self._compat_hash is not None and payload.compatibility_hash != self._compat_hash:
                    raise RuntimeError(
                        "HIXLEngine handshake compat-hash mismatch: local="
                        f"{self._compat_hash} remote={payload.compatibility_hash}"
                    )
                peer_meta = agent_decoder.decode(payload.agent_metadata_bytes)
                if peer_meta.engine_id != expected_engine_id:
                    raise RuntimeError(
                        f"HIXLEngine handshake engine_id mismatch: expected="
                        f"{expected_engine_id} got={peer_meta.engine_id}"
                    )
                results.append((remote_pp_rank, remote_rank, peer_meta))

        assert best_offset is not None
        return results, best_offset

    def _ensure_handshake(
        self,
        remote_engine_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_tp_size: int,
    ) -> None:
        """Submit the ZMQ handshake on the single-worker executor and stash
        the result into _remote_metadata / _engine_clock_offset once done.
        On success, also Connect to every peer rank and compute the TP
        mapping (Step C). Mirrors NIXL base_worker.py:840-896.
        """
        # Evict engines past their TTL before adding a new one (mirrors
        # NIXL base_worker.py:858 — otherwise stale peers accumulate).
        self._evict_stale_engines()
        with self._handshake_lock:
            if remote_engine_id in self._remote_metadata:
                return
            if remote_engine_id in self._handshake_futures:
                return
            future = self._handshake_initiation_executor.submit(
                self._hixl_engine_handshake,
                remote_host,
                remote_handshake_port,
                remote_tp_size,
                remote_engine_id,
            )
            self._handshake_futures[remote_engine_id] = future

        def _done_callback(fut: Future) -> None:
            if fut.exception() is not None:
                logger.error(
                    "HIXLEngine handshake failed for %s: %s",
                    remote_engine_id, fut.exception(),
                )
                with self._handshake_lock:
                    self._handshake_futures.pop(remote_engine_id, None)
                    # Defer failure handling to the main thread: park on
                    # _ready_requests; start_load_kv finds plan is None and
                    # routes to _handle_failed_transfer there.
                    parked = self._pending_handshake_reqs.pop(
                        remote_engine_id, []
                    )
                    self._ready_requests.extend(parked)
                return
            meta_list, offset = fut.result()
            # Connect + plan BEFORE publishing metadata: if connect fails we
            # must not leave a half-registered engine in _remote_metadata (the
            # _ensure_handshake dedup keys on it, which would block retries).
            try:
                for remote_pp_rank, remote_rank, peer_meta in meta_list:
                    self._connect_and_plan(
                        remote_engine_id, peer_meta, remote_tp_size,
                        remote_pp_rank, remote_rank,
                    )
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "HIXLEngine connect/plan failed for %s: %s",
                    remote_engine_id, e,
                )
                self._cleanup_remote_engine(remote_engine_id)
                with self._handshake_lock:
                    self._handshake_futures.pop(remote_engine_id, None)
                    # Defer to the main thread via _ready_requests (same as
                    # the handshake-failure path above).
                    parked = self._pending_handshake_reqs.pop(
                        remote_engine_id, []
                    )
                    self._ready_requests.extend(parked)
                return
            with self._handshake_lock:
                # All ranks of an engine share block_lens/block_size/etc;
                # one copy is enough — _build_op_descs reads per-rank base
                # addresses from _kv_caches_base_addr, not from here.
                self._remote_metadata[remote_engine_id] = meta_list[0][2]
                self._engine_clock_offset[remote_engine_id] = offset
                self._remote_engine_last_seen[remote_engine_id] = time.perf_counter()
                self._handshake_futures.pop(remote_engine_id, None)
                # Release parked reqs into the ready queue under
                # _handshake_lock so the park in start_load_kv and this
                # release are mutually exclusive (NIXL pull_worker.py:78-79).
                parked = self._pending_handshake_reqs.pop(
                    remote_engine_id, []
                )
                self._ready_requests.extend(parked)
            logger.info(
                "HIXLEngine handshake ok. engine=%s ranks=%d endpoint=%s "
                "num_blocks=%d",
                remote_engine_id, len(meta_list),
                meta_list[0][2].local_engine_endpoint,
                meta_list[0][2].num_blocks,
            )

        future.add_done_callback(_done_callback)

    # ==================================================================
    # Remote engine connect + TP mapping (Step C). Replaces NIXL's
    # add_remote_agent / prep_xfer_dlist: hixl Connect takes the endpoint
    # string directly, and TransferAsync eats op_descs with no pre-built
    # xfer-side handles, so there is no dst_xfer_side_handles bookkeeping.
    # ==================================================================
    def _connect_and_plan(
        self,
        remote_engine_id: str,
        peer_meta: HixlEngineAgentMetadata,
        remote_tp_size: int,
        remote_pp_rank: int,
        remote_tp_rank: int,
    ) -> None:
        """Connect to the peer engine and compute the hetero-TP mapping.

        Mirrors NIXL base_worker.py:1565-1671 (add_remote_agent +
        compute_tp_mapping + _validate_remote_agent_handshake), minus the
        NIXL-only prep_xfer_dlist / dst_xfer_side_handles plumbing.
        """
        assert self._transfer_topo is not None, (
            "register_kv_caches must run before handshake completion"
        )
        # Pull mode addresses by region index assuming P/D region arrays line
        # up 1:1, which only holds for pipeline_parallel_size==1 (single
        # stage). NIXL slices remote base/block_lens by PP stage
        # (base_worker.py:1537-1550); that is not ported here, so fail closed
        # rather than land a confusing region-parity IndexError downstream.
        assert self._pp_size == 1, (
            "HIXLEngineConnector pull mode supports only "
            "pipeline_parallel_size==1; PP>1 remote region-window slicing is "
            "not implemented."
        )
        # 1. Establish the hixl connection (replaces nixl add_remote_agent).
        with self._hixl_lock:
            self._hixl_connect(
                peer_meta.local_engine_endpoint, self._link_timeout_ms
            )
        self._remote_agents.setdefault(remote_engine_id, {})[
            (remote_pp_rank, remote_tp_rank)
        ] = peer_meta.local_engine_endpoint

        # 2. Record the peer's per-rank base addresses (used by
        # _build_op_descs). Each rank has distinct base addresses; this runs
        # once per handshake target rank.
        self._kv_caches_base_addr.setdefault(
            remote_engine_id, {}
        )[remote_tp_rank] = peer_meta.kv_caches_base_addr

        # 3. Per-engine setup (block_lens/block_size/tp_ratio are rank-invariant
        # for a given engine, so do it once on the first rank handshaken).
        if remote_engine_id not in self._tp_mappings:
            self._tp_mappings[remote_engine_id] = compute_tp_mapping(
                transfer_topology=self._transfer_topo,
                remote_tp_size=remote_tp_size,
                group_spec_types=self._group_spec_types,
            )
            # Register the remote engine in the topology so is_kv_replicated /
            # get_engine_info are usable during validation (mirrors NIXL
            # base_worker.py:1683 assertion precondition).
            self._transfer_topo.register_remote_engine(
                remote_engine_id,
                EngineTransferInfo(
                    remote_tp_size=remote_tp_size,
                    remote_block_len=(
                        peer_meta.block_lens[0] if peer_meta.block_lens else 0
                    ),
                    remote_block_size=peer_meta.block_size,
                    remote_physical_blocks_per_logical=(
                        peer_meta.physical_blocks_per_logical_kv_block
                    ),
                ),
            )
            # Validate layout / block-size compatibility.
            self._validate_remote_agent_handshake(peer_meta, remote_tp_size)
        logger.info(
            "HIXLEngine connected to %s (rank pp=%d tp=%d, remote_tp_size=%d, "
            "rank_offset_factor=%d).",
            remote_engine_id, remote_pp_rank, remote_tp_rank,
            remote_tp_size,
            self._tp_mappings[remote_engine_id].rank_offset_factor,
        )

    def _validate_remote_agent_handshake(
        self, peer_meta: HixlEngineAgentMetadata, remote_tp_size: int,
    ) -> None:
        """Validate peer metadata invariants.

        Mirrors NIXL base_worker.py:1673-1713. HIXLEngine only checks the
        factors that affect address-level transfer correctness:
        block_size ratio, physical_blocks_per_logical_kv_block, and (for
        non-MLA / non-Mamba) the tp_ratio vs replication sanity.
        """
        assert self._transfer_topo is not None
        tp_ratio = self._transfer_topo.tp_ratio(remote_tp_size)
        block_size_ratio = self._transfer_topo.block_size_ratio(
            peer_meta.block_size
        )
        # num_kv_heads < tp_size with P_TP > D_TP is unsupported for plain FA.
        if not self._use_mla and not self._has_mamba:
            assert not (
                tp_ratio < 0
                and self._transfer_topo.is_kv_replicated(peer_meta.engine_id)
            ), (
                f"KV replication with P_TP>D_TP unsupported for {peer_meta.engine_id}"
            )
        remote_phys = peer_meta.physical_blocks_per_logical_kv_block
        if (
            self._has_mamba
            and remote_phys != self._physical_blocks_per_logical_kv_block
            and self._vllm_config.cache_config.enable_prefix_caching
        ):
            raise RuntimeError(
                "Prefix caching with heterogeneous "
                "physical_blocks_per_logical_kv_block is unsupported for "
                f"Mamba hybrids. local={self._physical_blocks_per_logical_kv_block}"
                f" remote={remote_phys}. Disable --enable-prefix-caching."
            )
        # Per-region block_len compatibility (mirrors NIXL
        # base_worker.py:1790-1818). Branch like NIXL: replicated /
        # tp_ratio>0 (D_TP>=P_TP) / tp_ratio<0 (P_TP>D_TP).
        remote_bl = peer_meta.block_lens
        local_bl = self._block_len_per_layer
        if remote_bl and local_bl:
            assert len(remote_bl) == len(local_bl), (
                f"region count mismatch for {peer_meta.engine_id}: "
                f"local={len(local_bl)} remote={len(remote_bl)}"
            )
            total_kv_heads = self._transfer_topo.total_num_kv_heads
            local_heads = self._transfer_topo.local_physical_heads
            remote_heads = max(1, total_kv_heads // remote_tp_size)
            model_replicated = (
                self._use_mla
                or self._transfer_topo.is_kv_replicated(peer_meta.engine_id)
            )
            for i, (lb, rb) in enumerate(zip(local_bl, remote_bl)):
                replicated = model_replicated or self._region_is_mla[i]
                if replicated:
                    assert lb // block_size_ratio == rb, (
                        f"block_len mismatch region {i} for "
                        f"{peer_meta.engine_id}: local//bsr="
                        f"{lb // block_size_ratio} remote={rb} (replicated)"
                    )
                elif tp_ratio > 0:
                    expected = (
                        lb * remote_heads // local_heads
                    ) // block_size_ratio
                    assert rb == expected, (
                        f"block_len mismatch region {i} for "
                        f"{peer_meta.engine_id}: remote={rb} expected="
                        f"{expected} (SPLIT D_TP>=P_TP: lb*"
                        f"{remote_heads}//{local_heads}//{block_size_ratio})"
                    )
                else:
                    assert block_size_ratio == 1, (
                        f"region {i} for {peer_meta.engine_id}: different "
                        "local/remote block sizes are not supported when "
                        "P_TP > D_TP."
                    )
                    expected = lb * remote_heads // local_heads
                    assert rb == expected, (
                        f"block_len mismatch region {i} for "
                        f"{peer_meta.engine_id}: remote={rb} expected="
                        f"{expected} (SPLIT P_TP>D_TP: lb*"
                        f"{remote_heads}//{local_heads})"
                    )

    # ------------------------------------------------------------------
    # TTL eviction (mirrors NIXL base_worker.py:2382-2441). Stale engines
    # that have not been heard from within engine_ttl are disconnected and
    # dropped so their mem mappings can be reclaimed.
    # ------------------------------------------------------------------
    def _evict_stale_engines(self) -> None:
        now = time.perf_counter()
        stale = [
            engine_id
            for engine_id, last_seen in self._remote_engine_last_seen.items()
            if now - last_seen > self._engine_ttl_s
        ]
        for engine_id in stale:
            self._cleanup_remote_engine(engine_id)

    def _cleanup_remote_engine(self, remote_engine_id: str) -> None:
        """Disconnect and forget a remote engine."""
        # Collect every endpoint we Connect()ed to. On the happy path these
        # live in _remote_metadata; on a validate/plan failure after a
        # successful connect (M2) _remote_metadata is still empty, so fall back
        # to _remote_agents which is populated right after connect() succeeds.
        endpoints: set[str] = set()
        peer_meta = self._remote_metadata.get(remote_engine_id)
        if peer_meta is not None:
            endpoints.add(peer_meta.local_engine_endpoint)
        for ep in self._remote_agents.get(remote_engine_id, {}).values():
            endpoints.add(ep)
        for endpoint in endpoints:
            try:
                with self._hixl_lock:
                    self._hixl_disconnect(
                        endpoint, self._link_timeout_ms
                    )
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "HIXLEngine disconnect failed for %s: %s",
                    remote_engine_id, e,
                )
        self._remote_metadata.pop(remote_engine_id, None)
        self._engine_clock_offset.pop(remote_engine_id, None)
        self._remote_engine_last_seen.pop(remote_engine_id, None)
        self._tp_mappings.pop(remote_engine_id, None)
        self._plan_invariants.pop(remote_engine_id, None)
        self._remote_agents.pop(remote_engine_id, None)
        self._kv_caches_base_addr.pop(remote_engine_id, None)
        if self._transfer_topo is not None:
            self._transfer_topo.unregister_remote_engine(remote_engine_id)
        logger.warning("HIXLEngine evicted stale remote engine %s.", remote_engine_id)

    # ==================================================================
    # P-side ZMQ ROUTER handshake listener. Mirrors NIXL
    # base_scheduler.py:291-332 _nixl_handshake_listener. Replies to
    # GET_META_MSG with [handshake_bytes, perf_counter_ts] so the D side can
    # estimate the clock offset. KVCacheSendingThread is NOT reused here:
    # its reply frame layout ([identity, b"", encoded_metadata], hixl_connector
    # :379) carries no perf ts and no handshake-payload envelope.
    # ==================================================================
    def start_handshake_listener(
        self, host: str, port: int, ready_event: threading.Event | None = None,
        stop_event: threading.Event | None = None,
    ) -> threading.Thread:
        if ready_event is None:
            ready_event = threading.Event()
        if stop_event is None:
            stop_event = self._handshake_stop_event
        thread = threading.Thread(
            target=self._handshake_listener_loop,
            args=(host, port, ready_event, stop_event),
            daemon=True,
            name="hixl_engine_handshake_listener",
        )
        self._handshake_listener_thread = thread
        thread.start()
        if not ready_event.wait(_HIXL_ENGINE_LISTENER_READY_TIMEOUT_S):
            logger.warning(
                "HIXLEngine handshake listener not ready within %.1fs on %s:%d",
                _HIXL_ENGINE_LISTENER_READY_TIMEOUT_S, host, port,
            )
        return thread

    def _ensure_handshake_listener(self) -> None:
        """Start the scheduler-side ROUTER once, idempotently.

        Bound to get_ip():_side_channel_port — the same port
        request_finished advertises via kv_transfer_params.remote_port, so
        the D side's _hixl_engine_handshake REQ lands here.
        """
        if (
            self._handshake_listener_thread is not None
            and self._handshake_listener_thread.is_alive()
        ):
            return
        if self._side_channel_port == 0:
            logger.error(
                "HIXLEngine cannot start listener: side_channel_port is 0 "
                "(set hixl_engine.side_channel_port)."
            )
            return
        self.start_handshake_listener(get_ip(), self._side_channel_port)

    def _handshake_listener_loop(
        self, host: str, port: int,
        ready_event: threading.Event, stop_event: threading.Event,
    ) -> None:
        path = make_zmq_path("tcp", host, port)
        logger.info("HIXLEngine handshake listener on %s.", path)
        encoder = msgspec.msgpack.Encoder()
        try:
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore[arg-type]
                sock.setsockopt(zmq.RCVTIMEO, _HIXL_ENGINE_LISTENER_POLL_MS)
                ready_event.set()
                while not stop_event.is_set():
                    try:
                        frames = sock.recv_multipart()
                    except zmq.Again:  # type: ignore[attr-defined]
                        continue
                    except zmq.ZMQError as e:  # type: ignore[attr-defined]
                        logger.warning("HIXLEngine listener recv error: %s", e)
                        continue
                    try:
                        self._handle_handshake_request(sock, frames, encoder)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("HIXLEngine listener handler error: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.exception("HIXLEngine handshake listener fatal: %s", e)

    def _handle_handshake_request(
        self, sock: zmq.Socket, frames: list[bytes],  # type: ignore[name-defined]
        encoder: msgspec.msgpack.Encoder,
    ) -> None:
        if len(frames) < 2:
            return
        identity = frames[0]
        payload = [f for f in frames[1:] if f != b""]
        if len(payload) != 1:
            self._reject_handshake(sock, identity, "expected exactly one payload frame")
            return
        try:
            msg = msgspec.msgpack.decode(payload[0])
        except Exception:
            self._reject_handshake(sock, identity, "unparseable GET_META payload")
            return
        if not isinstance(msg, (list, tuple)) or not msg or msg[0] != GET_META_MSG:
            self._reject_handshake(sock, identity, "not a GET_META message")
            return
        # NIXL single-listener routing (base_scheduler.py:316-322): the D side
        # addresses a specific (pp, tp) rank; serve that rank's pre-encoded
        # payload from the mapping set_xfer_handshake_metadata populated.
        if len(msg) < 3:
            logger.warning("HIXLEngine GET_META without (pp, tp): %s", msg)
            self._reject_handshake(sock, identity, "GET_META missing (pp, tp)")
            return
        pp_rank, tp_rank = msg[1], msg[2]
        # Snapshot under the lock: the scheduler thread may re-enter
        # set_xfer_handshake_metadata (resize) while this listener thread
        # iterates for the warning path, which would otherwise raise
        # RuntimeError: dictionary changed size during iteration.
        with self._handshake_lock:
            handshake_bytes = self._handshake_payloads.get((pp_rank, tp_rank))
            have = list(self._handshake_payloads)
        if handshake_bytes is None:
            logger.warning(
                "HIXLEngine GET_META for unknown (pp=%s, tp=%s); have %s",
                pp_rank, tp_rank, have,
            )
            self._reject_handshake(sock, identity, f"unknown (pp={pp_rank}, tp={tp_rank})")
            return
        perf_ts = msgspec.msgpack.encode(time.perf_counter())
        sock.send_multipart((identity, b"", handshake_bytes, perf_ts))

    @staticmethod
    def _reject_handshake(
        sock: zmq.Socket, identity: bytes, reason: str,  # type: ignore[name-defined]
    ) -> None:
        """Reply with an explicit error frame instead of silently dropping a
        malformed GET_META, so the D side fails fast on the real cause.
        Frame shape mirrors the successful reply (identity, b"",
        handshake_bytes, perf_ts); reply[0] becomes HIXL_ERR_PREFIX + reason.
        """
        try:
            sock.send_multipart(
                (identity, b"", HIXL_ERR_PREFIX + reason.encode(), b"")
            )
        except Exception:  # noqa: BLE001
            logger.error("HIXLEngine handshake reject failed: %s", reason)

    # ==================================================================
    # Scheduler-side decisions delegated to HIXLEngineConnectorScheduler
    # (forked from NIXL pull_scheduler / base_scheduler). The connector
    # process owns both roles; the scheduler instance is created in
    # __init__ for KVConnectorRole.SCHEDULER.
    # ==================================================================
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int  # noqa: F821
    ) -> tuple[int, bool]:
        return self._scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self, request: "Request",  # noqa: F821
        blocks: "KVCacheBlocks",  # noqa: F821
        num_external_tokens: int,
    ):
        self._scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return self._scheduler.build_connector_meta(scheduler_output)

    def request_finished_all_groups(
        self, request: "Request", block_ids: BlockIds
    ) -> tuple[bool, dict[str, Any] | None]:
        # SupportsHMA path: the framework passes the full multi-group
        # BlockIds tuple directly (vllm/v1/core/sched/scheduler.py:2501-2503
        # routes SupportsHMA connectors to request_finished_all_groups).
        # Forward it verbatim — the scheduler expects a BlockIds tuple.
        return self._scheduler.request_finished(request, block_ids)

    def request_finished(
        self, request: "Request", block_ids: list[int]  # noqa: F821
    ) -> tuple[bool, dict[str, Any] | None]:
        # Non-HMA fallback: the framework passes a single group's
        # list[int]; wrap into a 1-tuple so the scheduler's BlockIds
        # iteration matches (NIXL connector.py:194-200).
        return self._scheduler.request_finished(request, (block_ids,))

    def on_new_request(self, request: "Request") -> None:  # noqa: F821
        self._scheduler.on_new_request(request)

    def update_connector_output(
        self, connector_output: "KVConnectorOutput"  # noqa: F821
    ) -> None:
        self._scheduler.update_connector_output(connector_output)

    def has_pending_push_work(self) -> bool:
        return self._scheduler.has_pending_push_work()

    # ==================================================================
    # Worker-side: memory registration (replaces register_blocks_cache)
    # ==================================================================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register each layer's KV tensor with hixl::Hixl.

        Address-level: one mem handle per tensor region, MEM_DEVICE. No
        BlocksCacheKey/CacheDesc — heterogeneous shapes (conv 2D / ssm 3D /
        MLA latent) coexist as separate handles on the same engine. Mirrors
        NIXL base_worker.register_kv_caches (L1024-1275) but uses the
        address-level RegisterMem API and skips NIXL's prep_xfer_dlist /
        get_agent_metadata (hixl Connect uses the endpoint string directly).
        """
        # Read NZ switch + save layer tensors for post-transfer ND->NZ
        # reformat. Imported lazily so the module stays importable without
        # an NPU (ascend_config pulls in the NPU stack).
        from vllm_ascend.ascend_config import get_ascend_config  # noqa: PLC0415
        self._enable_kv_nz = bool(
            getattr(get_ascend_config(), "enable_kv_nz", False))
        self._kv_caches = kv_caches
        # ensure_linked: Initialize the hixl engine on first registration.
        # All wrapper calls hold _hixl_lock — hixl::Hixl is not thread-safe
        # and a handshake callback on _handshake_initiation_executor may be
        # racing connect() while we re-register here.
        with self._hixl_lock:
            if not self._hixl_initialized:
                self._hixl_initialize(
                    self._local_engine_endpoint, self._engine_options
                )

        # Idempotent re-registration: deregister previous handles first.
        with self._hixl_lock:
            for handle, _, _ in self._kv_mem_handles.values():
                try:
                    self._hixl_deregister_mem(handle)
                except Exception:  # noqa: BLE001
                    pass
        self._kv_mem_handles.clear()
        self._block_len_per_layer.clear()
        self._region_is_mla.clear()
        self._region_group_idx.clear()
        # Real per-logical-block bytes from each tensor's own shape, used by
        # SSM addressing. vllm-ascend pads MambaSpec.page_size_bytes to align
        # mamba/attn blocks for HMA pooling, which would inflate
        # block_len_per_layer past the real ssm per-block; FA regions keep
        # block_len_per_layer (FA page_size is not padded).
        self._per_block_per_layer = []
        seen_base_addresses: list[int] = []
        self._kv_caches_base_addr[self._engine_id][self._tp_rank] = (
            seen_base_addresses
        )

        for layer_name, cache_or_caches in kv_caches.items():
            layer_spec = self._layer_specs.get(layer_name)
            if layer_spec is None:
                # Layer shares another tensor's KV cache (hybrid allocator);
                # nothing to register for this name.
                continue
            if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                # MLA DSv32 Indexer: merge specs, pick this layer's own.
                layer_spec = layer_spec.kv_cache_specs[layer_name]
            tensors = (
                list(cache_or_caches)
                if isinstance(cache_or_caches, (list, tuple))
                else [cache_or_caches]
            )
            # physical_page_size mirrors NIXL base_worker.py:1117-1134.
            physical_page_size = layer_spec.page_size_bytes
            if not isinstance(layer_spec, MambaSpec):
                physical_page_size //= self._physical_blocks_per_logical_kv_block
            physical_page_size //= len(tensors)
            block_len = (
                physical_page_size // self._physical_blocks_per_logical_kv_block
                if isinstance(layer_spec, MambaSpec)
                else physical_page_size
            )
            is_mla_region = isinstance(
                layer_spec, (MLAAttentionSpec, SlidingWindowMLASpec)
            )
            for i, t in enumerate(tensors):
                base_addr = int(t.data_ptr())
                per_block = t[0].numel() * t.element_size()
                is_mamba_region = isinstance(layer_spec, MambaSpec)
                if base_addr in seen_base_addresses:
                    dup_idx = seen_base_addresses.index(base_addr)
                    dup_per_block = self._per_block_per_layer[dup_idx]
                    # SSM state shares the same base/length as an
                    # already-registered attn region (HMA pools attn K and
                    # mamba ssm_state onto one raw_tensor — same base, same
                    # length, different block view). The physical segment is
                    # already covered by attn's register_mem, and
                    # TransferAsync addresses purely by (addr, len) without
                    # referencing mem handles — so ssm needs neither
                    # register_mem nor a mem handle. But its logical region
                    # record MUST be appended: the is_ssm branch of
                    # _build_op_descs takes per_block from
                    # _per_block_per_layer[i], so without this entry ssm is
                    # never iterated and its state is never transferred. A
                    # conv duplicate (same per_block) is a true HMA shared_by
                    # re-reference and stays skipped.
                    if is_mamba_region and per_block != dup_per_block:
                        seen_base_addresses.append(base_addr)
                        self._block_len_per_layer.append(block_len)
                        self._per_block_per_layer.append(per_block)
                        self._region_is_mla.append(is_mla_region)
                        self._region_group_idx.append(
                            self._layer_to_group.get(layer_name, 0)
                        )
                        self._device_id = max(t.get_device(), 0)
                        logger.debug(
                            "HIXLTRACE reg_ssm_alias layer=%s sub=%d "
                            "base=0x%x length=%d per_block=%d group=%d "
                            "shape=%s dup_seen_idx=%d dup_per_block=%d "
                            "(ssm reuses attn registered segment; "
                            "no register_mem, no mem handle)",
                            layer_name, i, base_addr,
                            t.numel() * t.element_size(), per_block,
                            self._layer_to_group.get(layer_name, 0),
                            tuple(t.shape), dup_idx, dup_per_block,
                        )
                        continue
                    logger.debug(
                        "HIXLTRACE reg_dedup_skip layer=%s sub=%d base=0x%x "
                        "length=%d per_block=%d is_mamba=%d group=%d shape=%s "
                        "dup_seen_idx=%d",
                        layer_name, i, base_addr,
                        t.numel() * t.element_size(),
                        per_block,
                        is_mamba_region,
                        self._layer_to_group.get(layer_name, 0),
                        tuple(t.shape),
                        dup_idx,
                    )
                    # HMA memory pooling: same backing tensor shared across
                    # groups; register the region once.
                    continue
                length = t.numel() * t.element_size()
                with self._hixl_lock:
                    handle = self._hixl_register_mem(
                        base_addr, length, is_device=True
                    )
                self._kv_mem_handles[f"{layer_name}/{i}"] = (
                    handle, base_addr, length,
                )
                seen_base_addresses.append(base_addr)
                self._block_len_per_layer.append(block_len)
                # Real per-logical-block bytes from the tensor's own shape
                # (block-0 element count * dtype size), immune to the
                # vllm-ascend mamba page padding; used by SSM addressing.
                self._per_block_per_layer.append(per_block)
                self._region_is_mla.append(is_mla_region)
                self._region_group_idx.append(
                    self._layer_to_group.get(layer_name, 0)
                )
                # Torch uses -1 for CPU; hixl needs a non-negative device id.
                self._device_id = max(t.get_device(), 0)
                logger.debug(
                    "HIXLTRACE reg_region layer=%s sub=%d base=0x%x "
                    "length=%d per_block=%d is_mamba=%d group=%d shape=%s",
                    layer_name, i, base_addr, length,
                    self._per_block_per_layer[-1],
                    is_mamba_region,
                    self._layer_to_group.get(layer_name, 0),
                    tuple(t.shape),
                )

        self._build_transfer_topology(kv_caches)
        self._build_xfer_handshake_metadata()
        _pb_counts: dict[int, int] = {}
        for _pb in self._per_block_per_layer:
            _pb_counts[_pb] = _pb_counts.get(_pb, 0) + 1
        _grp_counts: dict[int, int] = {}
        for _g in self._region_group_idx:
            _grp_counts[_g] = _grp_counts.get(_g, 0) + 1
        logger.debug(
            "HIXLTRACE reg_summary n_regions=%d n_handles=%d "
            "per_block_dist=%s group_dist=%s",
            len(self._per_block_per_layer), len(self._kv_mem_handles),
            _pb_counts, _grp_counts,
        )
        logger.info(
            "HIXLEngineConnector registered %d KV regions on rank %s "
            "(num_blocks=%d, block_size=%d, layout=%s).",
            len(self._kv_mem_handles), self._tp_rank,
            self._num_blocks, self._block_size, self._kv_cache_layout,
        )

    def _build_transfer_topology(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Build the local TransferTopology (mirrors NIXL base_worker.py:1041-1054).

        compute_tp_mapping reads tp_rank / tp_size / total_num_kv_heads /
        is_mla from this; cross_layers_blocks is detected from the first
        tensor's shape vs the spec's single-layer shape.
        """
        first_tensor = next(iter(kv_caches.values()), None)
        tensor_shape = (
            None
            if self._has_mamba or first_tensor is None
            else first_tensor.shape
        )
        self._transfer_topo = TransferTopology(
            tp_rank=self._tp_rank,
            tp_size=self._tp_size,
            block_size=self._block_size,
            engine_id=self._engine_id,
            is_mla=self._use_mla,
            is_mamba=self._has_mamba,
            total_num_kv_heads=self._num_kv_heads,
            attn_backends=self._attn_backends,
            tensor_shape=tensor_shape,
        )

    def _build_xfer_handshake_metadata(self) -> None:
        """Construct HixlEngineAgentMetadata + compat-hash payload.

        Mirrors NIXL base_worker.py:1252-1275. The payload is what the P-side
        ROUTER hands to D-side REQ over ZMQ (see Step A handshake listener).
        """
        from vllm import __version__ as vllm_version

        agent_meta = HixlEngineAgentMetadata(
            engine_id=self._engine_id,
            local_engine_endpoint=self._local_engine_endpoint,
            cluster_id=0,  # hixl Connect routes by endpoint string, not cluster
            listen_ip=get_ip(),
            listen_port=0,  # filled by scheduler wiring (Step R); per-device
                            # port offset TODO
            kv_caches_base_addr=(
                self._kv_caches_base_addr[self._engine_id][self._tp_rank]
            ),
            block_lens=list(self._block_len_per_layer),
            num_blocks=self._num_blocks,
            device_id=self._device_id,
            block_size=self._block_size,
            kv_cache_layout=self._kv_cache_layout,
            ssm_sizes=self._mamba_ssm_size,
            physical_blocks_per_logical_kv_block=(
                self._physical_blocks_per_logical_kv_block
            ),
            model_id=0,
            num_kv_heads=self._num_kv_heads,
            head_size=self._head_size,
            attn_backend_name=self._attn_backend_name,
        )
        self._compat_hash = compute_hixl_engine_compat_hash(
            vllm_version=vllm_version,
            model_name=self._model_config.model,
            dtype=str(self._model_config.dtype),
            num_kv_heads=self._num_kv_heads,
            head_size=self._head_size,
            num_hidden_layers=self._num_hidden_layers,
            attn_backend_name=self._attn_backend_name,
            cache_dtype=str(self._vllm_config.cache_config.cache_dtype),
            is_hma_enabled=(
                not self._vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            ),
        )
        self._xfer_handshake_metadata = HixlEngineHandshakePayload(
            compatibility_hash=self._compat_hash,
            agent_metadata_bytes=msgspec.msgpack.encode(agent_meta),
        )

    # ==================================================================
    # Worker-side: address-level transfer (replaces pull_blocks + reformat)
    # ==================================================================
    def _build_op_descs(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        plan: TPMapping,
        remote_engine_id: str,
        group_idx: int,
        source_rank: int,
    ) -> tuple[list[TransferOpDesc], int]:
        """Build TransferOpDesc batch for one source rank, one group.

        Returns ``(descs, total_bytes)``; the byte count feeds the
        effective-bandwidth log in _pop_done_transfers and is accumulated
        during the build so it costs nothing extra.

        Ports NIXL _build_fa_remote (base_worker.py:1386-1429) for the remote
        side and _build_local_splits_from_plan (base_worker.py:160-209) for
        the per-source-rank local head offset. reshard is expressed purely as
        address offsets, no staging cache:

          stride       = block_len_per_layer[i] // block_size_ratio
          chunk        = stride // num_reads            (transfer length)
          rank_offset  = 0 if replicated else plan.rank_offset_factor * stride
          slot         = 0 if replicated
                       = plan.rank_to_attention_slot[source_rank]  (FA group)
                       = positional index in plan.all_source_ranks   (SSM)
          remote_addr  = remote_base_rank_i + rank_offset + remote_bid * page_size
          local_addr   = local_base_i + local_bid * stride + slot * chunk
          len          = chunk

        stride (full block, for addressing) and chunk (divided, for transfer
        length) are kept separate — NIXL _build_fa_local (base_worker.py
        :1381) uses block_len_per_layer//ratio as page_stride while
        _build_fa_remote (:1424) uses the same value //num_reads as length.

        ``remote_base`` is per-source-rank: each remote TP rank exposes its
        own KV base addresses, stored in _kv_caches_base_addr[engine_id][rank]
        during _connect_and_plan. Iterates only the regions of ``group_idx``
        so a Mamba hybrid model transfers each group's blocks to its regions.
        """
        assert self._transfer_topo is not None
        remote_bases_by_rank = self._kv_caches_base_addr[remote_engine_id]
        assert source_rank in remote_bases_by_rank, (
            f"remote rank {source_rank} not connected for engine "
            f"{remote_engine_id}; connected={list(remote_bases_by_rank)}"
        )
        remote_bases = remote_bases_by_rank[source_rank]
        remote_meta = self._remote_metadata[remote_engine_id]
        # block_size_ratio / remote_physical_per_logical / d_ssm_group_count are
        # fixed once the engine is handshaken, but this method runs per
        # (group, source_rank) per request. Memoize them per engine;
        # _cleanup_remote_engine drops the entry so a re-handshake recomputes.
        #
        # remote_physical_per_logical: non-Mamba regions address by physical
        # block id (_read_blocks expands them), so their stride/page_size are
        # per-physical-block spans. Mamba state regions keep logical-id
        # addressing, so theirs must span a full logical block (per-physical
        # span * physical_blocks_per_logical) — NIXL _build_mamba_remote
        # (base_worker.py:1345).
        #
        # d_ssm_group_count: P may split SSM into N kv_cache_groups (per mamba
        # spec) while D merges them into one; when D has exactly one SSM group,
        # route all plan SSM groups to D's SSM regions by spec type instead of
        # group_idx. FA always matches strictly.
        invariants = self._plan_invariants.get(remote_engine_id)
        if invariants is None:
            invariants = (
                self._transfer_topo.block_size_ratio(remote_meta.block_size),
                self._transfer_topo.get_engine_info(
                    remote_engine_id
                ).remote_physical_blocks_per_logical,
                self._count_d_ssm_groups(len(remote_bases)),
            )
            self._plan_invariants[remote_engine_id] = invariants
        block_size_ratio, remote_physical_per_logical, _d_ssm_group_count = invariants
        local_phys = self._physical_blocks_per_logical_kv_block
        # SPLIT regions read their head slice from this many remote ranks;
        # REPLICATE (MLA) regions read the whole block once.
        split_reads = len(plan.source_ranks_per_group[group_idx])
        local_bases = self._kv_caches_base_addr[self._engine_id][self._tp_rank]
        is_ssm_group = self._is_ssm_spec(self._group_spec_types[group_idx])
        route_ssm_by_spec = is_ssm_group and _d_ssm_group_count == 1
        # Per-source-rank local head slot (gather scenario). split/MLA => 0.
        if is_ssm_group:
            # TODO(conv-decomp): replace with conv decomposition
            # (NIXL _build_mamba_remote, base_worker.py:1311-1352).
            slot = plan.all_source_ranks.index(source_rank)
        else:
            slot = plan.rank_to_attention_slot.get(source_rank, 0)

        descs: list[TransferOpDesc] = []
        _TransferOpDesc = self._hixl_mod.TransferOpDesc
        _emit = descs.append
        # Open coalescing run; cur_len == 0 means there is none.
        cur_local = 0
        cur_remote = 0
        cur_len = 0
        n_matched_regions = 0
        # Accumulated here rather than summed off the finished descs, which
        # would cost a pybind attribute sweep.
        total_bytes = 0
        pairs = list(zip(local_block_ids, remote_block_ids))
        if is_ssm_group and pairs:
            logger.debug(
                "HIXLTRACE ssm_pairs group=%d source_rank=%d n_pairs=%d "
                "slot=%d local_phys=%d remote_phys=%d block_size_ratio=%d "
                "split_reads=%d local_bids=%s remote_bids=%s",
                group_idx, source_rank, len(pairs),
                slot, local_phys, remote_physical_per_logical,
                block_size_ratio, split_reads,
                local_block_ids, remote_block_ids,
            )
        # Region parity: a layout mismatch would otherwise land as a silent
        # IndexError / wrong-region address. PP>1 remote-window slicing (NIXL
        # base_worker.py:1537-1550) is not yet ported (pull mode assumes
        # single pipeline stage).
        n_regions = len(remote_bases)
        assert (
            len(local_bases) == len(self._block_len_per_layer)
            == len(self._region_group_idx) == len(self._region_is_mla)
            == n_regions == len(remote_meta.block_lens)
        ), (
            f"region parity violated: local_bases={len(local_bases)} "
            f"block_len={len(self._block_len_per_layer)} "
            f"group_idx={len(self._region_group_idx)} "
            f"is_mla={len(self._region_is_mla)} "
            f"remote_bases={n_regions} "
            f"remote_block_lens={len(remote_meta.block_lens)}"
        )
        for i, remote_base in enumerate(remote_bases):
            if route_ssm_by_spec:
                # D has one SSM group: route this plan SSM group to every D
                # SSM region.
                _ri = self._region_group_idx[i]
                if not (
                    _ri < len(self._group_spec_types)
                    and self._is_ssm_spec(self._group_spec_types[_ri])
                ):
                    continue
            elif self._region_group_idx[i] != group_idx:
                continue
            replicated = self._region_is_mla[i]
            if is_ssm_group:
                # Mamba state: logical-id addressing (expansion is skipped
                # for state groups). stride/page_size use the real
                # per-logical-block bytes from the tensor shape, not the
                # padded MambaSpec.page_size_bytes (see register_kv_caches).
                # P/D run the same model/spec, so local per_block is valid
                # for the remote side too.
                per_block = self._per_block_per_layer[i]
                stride = per_block // block_size_ratio
                page_size = per_block
            else:
                stride = self._block_len_per_layer[i] // block_size_ratio
                page_size = remote_meta.block_lens[i]
            num_reads = 1 if replicated else split_reads
            chunk = stride // num_reads
            rank_offset = (
                0 if replicated else plan.rank_offset_factor * stride
            )
            local_base = local_bases[i]
            n_matched_regions += 1
            for local_bid, remote_bid in pairs:
                remote_addr = remote_base + rank_offset + remote_bid * page_size
                local_addr = local_base + local_bid * stride + slot * chunk
                # Extend the open run only when this desc abuts it on BOTH
                # sides; a strided layout (chunk < stride) never satisfies the
                # test and falls through to one desc per pair.
                total_bytes += chunk
                if (cur_len
                        and local_addr == cur_local + cur_len
                        and remote_addr == cur_remote + cur_len):
                    cur_len += chunk
                    continue
                if cur_len:
                    _emit(_TransferOpDesc(cur_local, cur_remote, cur_len))
                cur_local = local_addr
                cur_remote = remote_addr
                cur_len = chunk
            if pairs and is_ssm_group and logger.isEnabledFor(logging.DEBUG):
                _max_lbid = max(p[0] for p in pairs)
                _max_rbid = max(p[1] for p in pairs)
                logger.debug(
                    "HIXLTRACE build_op region=%d group_idx=%d "
                    "region_group=%d is_ssm=%d remote_base=0x%x "
                    "local_base=0x%x stride=%d page=%d chunk=%d "
                    "rank_offset=%d slot=%d max_rbid=%d "
                    "max_remote_addr=0x%x max_lbid=%d "
                    "max_local_addr=0x%x",
                    i, group_idx, self._region_group_idx[i],
                    is_ssm_group, remote_base, local_base, stride,
                    page_size, chunk, rank_offset, slot, _max_rbid,
                    remote_base + rank_offset + _max_rbid * page_size,
                    _max_lbid, local_base + _max_lbid * stride + slot * chunk,
                )
        # Close the last open run.
        if cur_len:
            _emit(_TransferOpDesc(cur_local, cur_remote, cur_len))
        if logger.isEnabledFor(logging.DEBUG):
            # n_descs vs n_descs_unmerged is the coalescing hit rate.
            logger.debug(
                "HIXLTRACE build_op_summary group_idx=%d source_rank=%d "
                "is_ssm=%d route_by_spec=%d d_ssm_groups=%d "
                "n_matched_regions=%d n_pairs=%d n_regions_total=%d "
                "n_descs=%d n_descs_unmerged=%d region_groups=%s",
                group_idx, source_rank, is_ssm_group, route_ssm_by_spec,
                _d_ssm_group_count, n_matched_regions, len(pairs), n_regions,
                len(descs), n_matched_regions * len(pairs),
                list(self._region_group_idx),
            )
        return descs, total_bytes

    def _read_blocks(
        self, request_id: str, req_meta: HIXLEngineReqMeta, plan: TPMapping
    ) -> None:
        """Issue async READ batches, one per source remote rank.

        Port of NIXL pull_worker._read_blocks_for_req (pull_worker.py:
        116-204) + _read_blocks (pull_worker.py:215-342). Iterates source
        ranks outermost (NIXL ReadSpec is per-rank, carrying all groups'
        block_ids) so a full prefix hit across all groups for a rank sends a
        single release notify. Each non-empty rank's op_descs are submitted
        via TransferAsync(READ, endpoint); the handle is stashed under the
        *local* request_id for get_finished/wait_for_layer_load to poll.
        P-side release notify is sent from _pop_done_transfers once all of a
        req's handles resolve (replaces NIXL's transfer notif_msg
        auto-delivery, plan §2.4).

        transfer_async takes the peer endpoint string (the value passed to
        Connect), not the engine_id name. All wrapper calls that touch the
        hixl instance are serialized with _hixl_lock (hixl is not
        thread-safe; connect runs on the handshake thread, transfers here).
        """
        remote_engine_id = req_meta.remote.engine_id
        assert remote_engine_id in self._remote_metadata, (
            f"remote engine {remote_engine_id} not handshaken yet"
        )
        # Always store metadata for failure recovery (NIXL pull_worker.py:66).
        self._recving_metadata[request_id] = req_meta
        # Refresh last-seen so an engine with in-flight transfers is not
        # stale-evicted mid-read (NIXL pull_worker.py:121). _evict_stale_engines
        # runs on this same (main) thread, so no lock needed here.
        self._remote_engine_last_seen[remote_engine_id] = time.perf_counter()
        remote_agents = self._remote_agents[remote_engine_id]
        num_groups = len(req_meta.local_block_ids)
        notif_id = f"{req_meta.remote.request_id}:{self._world_size}"
        logger.debug(
            "HIXLTRACE D-read_blocks req=%s remote_engine=%s n_groups=%d "
            "source_ranks=%s local_blks=%s remote_blks=%s",
            request_id, remote_engine_id, num_groups,
            list(plan.all_source_ranks),
            [len(g) for g in req_meta.local_block_ids],
            [len(g) for g in (req_meta.remote.block_ids or [])],
        )

        assert self._transfer_topo is not None
        remote_info = self._transfer_topo.get_engine_info(remote_engine_id)
        remote_physical_per_logical = remote_info.remote_physical_blocks_per_logical

        # Pre-trim per group (SSM tail-trim to last block, FA tail-trim
        # when same phys / front-trim to min when heterogeneous phys).
        trimmed_local: list[list[int]] = []
        trimmed_remote: list[list[int]] = []
        for g in range(num_groups):
            lb = list(req_meta.local_block_ids[g])
            rb = (
                list(req_meta.remote.block_ids[g])
                if g < len(req_meta.remote.block_ids)
                else []
            )
            lb, rb = self._apply_prefix_caching(
                lb, rb, g, remote_physical_per_logical
            )
            trimmed_local.append(lb)
            trimmed_remote.append(rb)

        local_phys = self._physical_blocks_per_logical_kv_block
        for rank in plan.all_source_ranks:
            endpoint = remote_agents.get((0, rank))
            assert endpoint is not None, (
                f"no endpoint for engine {remote_engine_id} rank (0,{rank})"
            )
            # Gather this rank's descs across all groups it sources.
            group_descs: list[TransferOpDesc] = []
            rank_bytes = 0
            for g, source_ranks in enumerate(plan.source_ranks_per_group):
                if g >= num_groups or rank not in source_ranks:
                    continue
                lb = trimmed_local[g]
                rb = trimmed_remote[g]
                if not lb:
                    continue
                # Expand logical -> physical block ids for non-Mamba groups
                # so _build_op_descs (which addresses by physical id with
                # per-physical-block spans) lands on the right offset when
                # physical_blocks_per_logical>1. Mamba state regions keep the
                # TODO conv-decomposition path (phys==1 there today).
                if g < len(self._group_spec_types) and not self._is_ssm_spec(
                    self._group_spec_types[g]
                ):
                    lb = self._expand_physical(lb, local_phys)
                    rb = self._expand_physical(rb, remote_physical_per_logical)
                    # zip() in _build_op_descs pairs local/remote physical
                    # ids positionally; heterogeneous phys (different kernel
                    # block_size) makes the expanded lengths diverge. Fail
                    # closed — the address-offset path cannot express a
                    # per-side desc count without NIXL-style desc decoupling
                    # (not ported).
                    assert len(lb) == len(rb), (
                        f"group {g}: physical id count mismatch after expand: "
                        f"local={len(lb)} remote={len(rb)} (local_phys="
                        f"{local_phys} remote_phys="
                        f"{remote_physical_per_logical}); heterogeneous "
                        "block_size not supported by the zip-pair path."
                    )
                _g_descs, _g_bytes = self._build_op_descs(
                    list(lb), list(rb), plan, remote_engine_id, g, rank,
                )
                group_descs.extend(_g_descs)
                rank_bytes += _g_bytes
            # Full prefix hit across all groups for this rank: no transfer,
            # just notify P to release (NIXL pull_worker.py:277-294).
            if not group_descs:
                self._enqueue_notify(endpoint, "DONE", notif_id)
                # Mark this rank notified so _notify_release skips it (the
                # prefix-hit notify and the release notify share the same
                # notif_id).
                self._notified_release_ranks.setdefault(
                    request_id, set()
                ).add(rank)
                continue
            if group_descs and logger.isEnabledFor(logging.DEBUG):
                # Guarded: the args below sweep group_descs nine times and
                # Python evaluates them before the level check.
                _la = [d.local_addr for d in group_descs]
                _ra = [d.remote_addr for d in group_descs]
                _ln = [d.len for d in group_descs]
                logger.debug(
                    "HIXLTRACE D-transfer_async req=%s rank=%d endpoint=%s "
                    "n_descs=%d local=[0x%x..0x%x] remote=[0x%x..0x%x] "
                    "len=[%d..%d]",
                    request_id, rank, endpoint, len(group_descs),
                    min(_la), max(_la), min(_ra), max(_ra),
                    min(_ln), max(_ln),
                )
            try:
                with self._hixl_lock:
                    handle = self._hixl_transfer_async(
                        endpoint, "READ", group_descs
                    )
                self._recving_transfers.setdefault(
                    request_id, []
                ).append(handle)
                # Bytes accumulate across a request's per-rank transfers; the
                # clock starts at the first submit.
                self._xfer_bytes[request_id] = (
                    self._xfer_bytes.get(request_id, 0) + rank_bytes)
                self._xfer_start.setdefault(request_id, time.perf_counter())
                # Record this rank as a real reader so _notify_release sends
                # DONE to it only.
                self._transferred_ranks.setdefault(
                    request_id, set()
                ).add(rank)
            except Exception as e:
                _la = [d.local_addr for d in group_descs] if group_descs else []
                _ra = [d.remote_addr for d in group_descs] if group_descs else []
                logger.error(
                    "HIXLEngine transfer_async failed. req=%s rank=%s err=%s "
                    "endpoint=%s n_descs=%d local=[0x%x..0x%x] "
                    "remote=[0x%x..0x%x]",
                    request_id, rank, e, endpoint, len(group_descs),
                    min(_la) if _la else 0, max(_la) if _la else 0,
                    min(_ra) if _ra else 0, max(_ra) if _ra else 0,
                )
                # Do NOT pop already-issued handles. hixl has no transfer
                # cancel API, so popping leaks them — _pop_done_transfers
                # never sees them again and the RDMA handle is never freed.
                # Leave them in _recving_transfers so _pop_done_transfers
                # polls them to COMPLETED and drops them normally; the req is
                # already in _failed_recv_reqs (via _handle_failed_transfer
                # below), which gates _notify_release so no P-side release
                # fires for these blocks. The in-flight RDMA write may still
                # land on a block the scheduler will recompute; the recompute
                # retry overwrites.
                self._handle_failed_transfer(request_id, None)
                return

    # ==================================================================
    # Worker-side: load/save lifecycle (poll handles, no reformat)
    # ==================================================================
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:  # noqa: F821
        """Trigger async READ for requests scheduled this step.

        Mirrors hixl_connector.start_load_kv but issues TransferAsync instead
        of pull_blocks — no RecvingThread is needed since the transfer is non-
        blocking and handles are polled in get_finished / wait_for_layer_load
        (NIXL pull_worker.start_load_kv:44-103). A request whose remote engine
        has not been handshaken yet is parked on _pending_handshake_reqs and
        read on a later step once the handshake done_callback releases it into
        _ready_requests (NIXL pull_worker.py:66-79). Without that re-queue
        the req would be lost — the scheduler clears _reqs_need_recv /
        flips do_remote_prefill the same step.
        """
        # metadata comes from bind_connector_metadata (base.py:221), which the
        # model runner calls with scheduler_output.kv_connector_metadata just
        # before start_load_kv (kv_connector_model_runner_mixin.py:88-95).
        metadata = self._connector_metadata
        if metadata is None:
            return
        for req_id in metadata.reqs_in_batch:
            self._task_tracker.add_req_to_process(req_id)
        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = meta.remote.engine_id
            # Always store metadata (NIXL pull_worker.py:66) so the
            # ready-queue drain / failure paths recover req state after a
            # deferred handshake.
            self._recving_metadata[req_id] = meta
            # Check + park under _handshake_lock so the done_callback
            # (executor thread) cannot publish _remote_metadata and drain
            # _pending_handshake_reqs between our check and our park. RLock
            # so _ensure_handshake's own lock acquisition re-enters cleanly.
            with self._handshake_lock:
                already = remote_engine_id in self._remote_metadata
                if not already:
                    self._pending_handshake_reqs.setdefault(
                        remote_engine_id, []
                    ).append((req_id, meta))
            if not already:
                self._ensure_handshake(
                    remote_engine_id,
                    meta.remote.host,
                    meta.remote.port,
                    meta.tp_size,
                )
                continue
            plan = self._tp_mappings[remote_engine_id]
            self._read_blocks(req_id, meta, plan)

        # Drain reqs whose handshakes finished (this step or a prior one) —
        # NIXL pull_worker.py:78-79. Also drained from get_finished so a
        # handshake that lands during this step's forward issues its READ at
        # step end instead of waiting for the next step's start_load_kv
        # (saves one step of KV-read latency on the cold-start path).
        self._drain_ready_requests()

        # Drop aborted reqs from the in-process set (NIXL pull_worker:90-94).
        for req_id in metadata.reqs_not_processed:
            self._task_tracker.discard_from_process(req_id)
        # Track lease expiry for reqs awaiting remote read (P-side delayed
        # free). NIXL guards on _reqs_to_process membership to avoid
        # resurrecting an already-freed req; here _reqs_to_send is the
        # authoritative expiry table and _get_new_notifs / lease-expiry are
        # the only removers (NIXL pull_worker:96-99).
        for req_id, expiration_time in metadata.reqs_to_send.items():
            self._reqs_to_send[req_id] = expiration_time
        # D-side: extend P-side leases for reqs still WAITING in scheduler
        # (NIXL pull_worker:101-103).
        self._send_heartbeats(metadata)

    def _drain_ready_requests(self) -> None:
        """Issue READ for reqs released by handshake done_callbacks.

        Pulled out of start_load_kv so get_finished can also drain mid-step:
        if a handshake completes during this step's forward, its parked req
        is already in _ready_requests by step end. Draining here issues the
        READ one step earlier than waiting for the next start_load_kv,
        shaving KV-read latency off the cold-start path (TTFT). Same worker
        thread as start_load_kv, so _read_blocks / _hixl_lock semantics are
        unchanged. _ready_requests is extend()ed under _handshake_lock by the
        done_callback on the executor thread and popleft()ed here without
        the lock — same pattern as the original start_load_kv drain (deque
        append/popleft is atomic under CPython's GIL).
        """
        while self._ready_requests:
            rid, rmeta = self._ready_requests.popleft()
            plan = self._tp_mappings.get(rmeta.remote.engine_id)
            if plan is None:
                # Engine evicted while parked; surface as failure so the
                # scheduler recomputes the blocks.
                self._handle_failed_transfer(rid, None)
                continue
            self._read_blocks(rid, rmeta, plan)

    def _apply_prefix_caching(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        group_idx: int,
        remote_physical_per_logical: int,
    ) -> tuple[list[int], list[int]]:
        """Trim to the locally-uncached tail, per group (NIXL base_worker.py:
        2255-2315).

        Branch per group:

        - SSM state: only the last block holds the full in-place state;
          assert num_local==1 and tail-trim remote to it.
        - FA, same phys: tail-trim remote to len(local) (skip cached prefix).
        - FA, heterogeneous phys: front-trim both to min. This is the only
          pairing that keeps the physical-id arrays zip-aligned after
          _expand_physical; a true heterogeneous reshard needs NIXL-style
          desc decoupling (not done here) — _read_blocks asserts len(lb)
          ==len(rb) and fails closed on divergence.

        Trims logical block ids; physical-id expansion (when phys>1) is
        done by _read_blocks via _expand_physical.
        """
        num_local = len(local_block_ids)
        num_remote = len(remote_block_ids)
        assert num_local <= num_remote, (
            f"group {group_idx}: local {num_local} > remote {num_remote}; "
            "prefix trim invariant violated"
        )
        if num_local == 0:
            # Full prefix hit: caller (_read_blocks) short-circuits on empty
            # local, but guard [-0:] == [0:] (full list) anyway.
            return local_block_ids, []
        if num_local == num_remote:
            return local_block_ids, remote_block_ids
        is_ssm = (
            group_idx < len(self._group_spec_types)
            and self._is_ssm_spec(self._group_spec_types[group_idx])
        )
        if is_ssm:
            assert num_local == 1, (
                f"group {group_idx}: SSM state expects exactly one local "
                f"block, got {num_local}"
            )
            return local_block_ids, remote_block_ids[-num_local:]
        if self._physical_blocks_per_logical_kv_block == remote_physical_per_logical:
            return local_block_ids, remote_block_ids[-num_local:]
        max_padding = max(
            self._physical_blocks_per_logical_kv_block,
            remote_physical_per_logical,
        )
        assert abs(num_local - num_remote) < max_padding, (
            f"group {group_idx}: heterogeneous phys trim diverges "
            f"|{num_local}-{num_remote}| >= {max_padding}; heterogeneous "
            "block_size reshard is not supported by the zip-pair path."
        )
        n = min(num_local, num_remote)
        return local_block_ids[:n], remote_block_ids[:n]

    def _count_d_ssm_groups(self, n_regions: int) -> int:
        """Number of distinct D-side kv_cache groups backed by SSM regions."""
        if not self._has_mamba:
            return 0
        n_specs = len(self._group_spec_types)
        return len({
            self._region_group_idx[i]
            for i in range(n_regions)
            if self._region_group_idx[i] < n_specs
            and self._is_ssm_spec(
                self._group_spec_types[self._region_group_idx[i]])
        })

    @staticmethod
    def _expand_physical(block_ids: list[int], phys: int) -> list[int]:
        """Expand logical block ids to physical block ids.

        Mirrors NIXL _compute_desc_ids (pull_worker.py). phys==1 is a no-op;
        phys>1 maps logical id b to [b*phys, ..., b*phys+phys-1]. _build_op_descs
        addresses by physical id (page_size/stride are per-physical-block spans),
        so this expansion must happen before building op descs whenever
        physical_blocks_per_logical>1.
        """
        if phys <= 1:
            return list(block_ids)
        out: list[int] = []
        for b in block_ids:
            base = b * phys
            out.extend(range(base, base + phys))
        return out

    def _log_transfer_bandwidth(self, req_id: str, had_failure: bool) -> None:
        """Report effective bandwidth for one request's KV pull.

        Answers two questions: whether a single transfer_async saturates the
        link (hixl slices op_descs into 256-desc batches onto one stream, so it
        may not), and whether the submitted bytes match the model's theoretical
        KV size — a larger figure would mean ranges are transferred twice.
        """
        nbytes = self._xfer_bytes.pop(req_id, 0)
        started = self._xfer_start.pop(req_id, None)
        if started is None or nbytes <= 0:
            return
        elapsed = time.perf_counter() - started
        if elapsed <= 0.0:
            return
        logger.info(
            "HIXLEngine KV pull done. req=%s bytes=%d (%.1f MiB) "
            "elapsed=%.1fms effective=%.2f GB/s failed=%s",
            req_id, nbytes, nbytes / 1048576.0, elapsed * 1000.0,
            nbytes / elapsed / 1e9, had_failure,
        )

    def _transfer_status_name(self, status: Any) -> str:
        """Normalize a hixl TransferStatus enum value to a comparable name."""
        name = getattr(status, "name", None)
        if name is not None:
            return name
        return str(status).split(".")[-1]

    def _pop_done_transfers(
        self, transfers: dict[str, list[int]]
    ) -> set[str]:
        """Poll each handle; return req_ids whose transfers all resolved.

        Port of NIXL _pop_done_transfers (base_worker.py:2092-2137).
        COMPLETED drops the handle; WAITING stays in_progress; FAILED/
        TIMEOUT marks the req invalid via _handle_failed_transfer. A req is
        done only when every handle resolved. On clean (failure-free)
        completion, _notify_release tells the P side to release blocks
        (replaces NIXL's transfer notif_msg auto-delivery).
        """
        done_req_ids: set[str] = set()
        for req_id, handles in list(transfers.items()):
            in_progress: list[int] = []
            had_failure = False
            for handle in handles:
                try:
                    # Read-only; see wait_for_layer_load on the missing lock.
                    st = self._hixl_get_transfer_status(handle)
                    if st is None:
                        # Record gone: an earlier query consumed a terminal
                        # status or hit an error, indistinguishable here. If it
                        # was a failure the observer already put the req in
                        # _failed_recv_reqs, which gates _notify_release below.
                        continue
                    sname = self._transfer_status_name(st)
                    if sname == "COMPLETED":
                        # TODO: confirm whether hixl needs an explicit
                        # release for the req handle (no wrapper API yet).
                        continue
                    if sname == "WAITING":
                        in_progress.append(handle)
                        continue
                    # FAILED / TIMEOUT / unknown -> mark invalid.
                    logger.error(
                        "HIXLEngine transfer failed. req=%s status=%s",
                        req_id, sname,
                    )
                    had_failure = True
                    self._handle_failed_transfer(req_id, handle)
                except Exception as e:
                    logger.error(
                        "HIXLEngine get_transfer_status exception. req=%s err=%s",
                        req_id, e,
                    )
                    had_failure = True
                    self._handle_failed_transfer(req_id, handle)
            if not in_progress:
                done_req_ids.add(req_id)
                logger.debug(
                    "HIXLTRACE D-transfer_done req=%s handles=%d failed=%s",
                    req_id, len(handles), had_failure,
                )
                self._log_transfer_bandwidth(req_id, had_failure)
                del transfers[req_id]
                # Only notify P to release if the req finished cleanly AND no
                # earlier rank's transfer_async raised (_read_blocks
                # stashes req-level failure in _failed_recv_reqs on a partial
                # submit; notify there would release P blocks for ranks whose
                # transfers never issued).
                if not had_failure and req_id not in self._failed_recv_reqs:
                    # ND->NZ reformat on the D real cache before telling
                    # the P side to release (transfer is COMPLETED, so the ND
                    # bytes are fully written). No-op when enable_kv_nz is off
                    # or the group is a state (Mamba) group.
                    self._apply_nz_reformat(req_id)
                    self._notify_release(req_id)
            else:
                transfers[req_id] = in_progress
        return done_req_ids

    # -- NZ reformat fallback (fork hixl_connector L1000-1090) ---------
    # npu_paged_cache_load / npu_scatter_pa_kv_cache / sync / _nz_kv_cache.
    # Pure torch_npu ops on the post-transfer tensor; only needed when HCCL
    # cannot scatter-write NZ offsets so the D cache lands ND. The
    # num_group_pulls>1 staging transpose is NOT forked — the address-offset
    # path in _build_op_descs replaces it.
    def _apply_nz_reformat(self, request_id: str) -> None:
        """ND->NZ reformat of D real cache for each attention group of a req.

        Called from _pop_done_transfers once all of a req's handles resolve
        cleanly. State groups (Mamba conv/ssm) are skipped; NZ applies to
        attention KV only. TP=1 only (MLA NZ has num_kv_heads==1 -> tp_n==1);
        TP>1+NZ needs a staging NZ scatter branch (left unsupported, same as
        the old connector).
        """
        if not self._enable_kv_nz:
            return
        # NZ reformat assumes a single-rank full-cache view (the reshape in
        # _nz_kv_cache spans num_kv_heads//16 blocks). TP>1 each rank holds a
        # head slice, so the reshape/scatter would land on wrong positions. A
        # staging NZ-scatter branch for TP>1 is not implemented; skip NZ and
        # let the direct ND write stand (only correct when HCCL scatter-writes
        # NZ offsets, which is the non-NZ fast path anyway).
        if self._tp_size > 1:
            logger.warning(
                "HIXLEngine enable_kv_nz with TP=%d: NZ reformat is TP=1 only; "
                "skipping NZ reformat and relying on direct ND write.",
                self._tp_size,
            )
            return
        meta = self._recving_metadata.get(request_id)
        if meta is None:
            return
        local_phys = self._physical_blocks_per_logical_kv_block
        for g, block_ids in enumerate(meta.local_block_ids):
            if g >= len(self._group_spec_types):
                continue
            if self._is_ssm_spec(self._group_spec_types[g]):
                continue  # state group: no NZ reformat
            if not block_ids:
                continue
            # _reformat_kv_cache_nz indexes the D paged cache by physical
            # block id (block_table -> npu_paged_cache_load). With phys>1 the
            # logical ids from the scheduler must be expanded to physical ids
            # or the ND->NZ scatter lands on wrong slots. SSM groups keep
            # logical-id addressing (Mamba phys==1 in practice) and are skipped
            # above.
            phys_block_ids = block_ids
            if local_phys > 1:
                phys_block_ids = self._expand_physical(
                    list(block_ids), local_phys
                )
            group_kv = {
                name: t for name, t in self._kv_caches.items()
                if self._layer_to_group.get(name) == g
            }
            self._reformat_kv_cache_nz(group_kv, list(phys_block_ids))

    def _reformat_kv_cache_nz(
        self,
        group_kv: dict[str, Any],
        block_ids: list[int],
    ) -> None:
        """ND -> NZ reformat of D real cache after transfer (NZ branch).

        TransferAsync writes ND into the D cache; under enable_kv_nz the D
        cache is physically NZ-ordered (attention writes via
        npu_scatter_pa_kv_cache). Load each layer's ND block range out of the
        D cache, then scatter it back into the D cache's NZ view.
        """
        if not block_ids:
            return
        # Import lazily so the module stays importable without an NPU (the
        # NPU-free unit tests never reach this branch). Importing torch_npu
        # injects torch.npu + torch_npu.atb ops.
        import torch_npu  # noqa: F401, PLC0415
        # The head/dim indexing below assumes NHD (shape[-2]=num_kv_heads,
        # shape[-1]=head_dim). HND would put block_size at shape[-2] and
        # silently mis-size the NZ reshape/scatter — fail closed instead.
        if self._kv_cache_layout != "NHD":
            raise RuntimeError(
                f"HIXLEngine NZ reformat assumes NHD layout "
                f"(shape[-2]=num_kv_heads, shape[-1]=head_dim), got "
                f"{self._kv_cache_layout}; disable enable_kv_nz or switch to "
                "NHD."
            )
        first_cache = next(iter(group_kv.values()))
        if isinstance(first_cache, (list, tuple)):
            k_ref, v_ref = first_cache[0], first_cache[1]
        else:
            k_ref = v_ref = first_cache
        dtype = k_ref.dtype
        device = k_ref.device
        num_kv_heads = int(k_ref.shape[-2])
        k_head_dim = int(k_ref.shape[-1])
        v_head_dim = int(v_ref.shape[-1])

        num_blocks = len(block_ids)
        num_tokens = num_blocks * self._block_size
        block_ids_tensor = torch.tensor(
            block_ids, dtype=torch.int32, device=device)
        block_table = block_ids_tensor.view(1, -1)
        block_len_tensor = torch.tensor(
            [num_tokens], dtype=torch.int32, device=device)
        seq_start_tensor = torch.tensor(
            [0], dtype=torch.int32, device=device)
        # slot_mapping = intra-block offset + block_id * block_size.
        block_offsets = torch.arange(
            0, self._block_size, dtype=torch.int32, device=device)
        slot_mapping = (
            block_offsets.reshape((1, self._block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * self._block_size
        ).flatten()
        k_buffer = torch.empty(
            (num_tokens, num_kv_heads, k_head_dim), dtype=dtype, device=device)
        v_buffer = torch.empty(
            (num_tokens, num_kv_heads, v_head_dim), dtype=dtype, device=device)
        # Pull writes ND; synchronize before reformatting so the NPU sees
        # the fully written range.
        torch.npu.synchronize()
        for d_cache in group_kv.values():
            if isinstance(d_cache, (list, tuple)):
                k_cache_layer, v_cache_layer = d_cache[0], d_cache[1]
            else:
                k_cache_layer = v_cache_layer = d_cache
            torch_npu.atb.npu_paged_cache_load(  # ND cache -> buffer
                k_cache_layer,
                v_cache_layer,
                block_table,
                block_len_tensor,
                seq_starts=seq_start_tensor,
                key=k_buffer,
                value=v_buffer,
            )
            self._nz_kv_cache(
                k_cache_layer,
                v_cache_layer,
                k_buffer,
                v_buffer,
                slot_mapping,
                num_kv_heads,
                k_head_dim,
                v_head_dim,
            )

    def _nz_kv_cache(
        self,
        k_cache_layer,
        v_cache_layer,
        k_buffer,
        v_buffer,
        slot_mapping,
        num_kv_heads: int,
        k_head_dim: int,
        v_head_dim: int,
    ):
        # nz_fmt_last_dim=16 (MLA NZ layout). Scatter buffer -> NZ view.
        import torch_npu  # noqa: F401, PLC0415
        nz_fmt_last_dim = 16
        k_cache_layer = k_cache_layer.view(
            -1, k_head_dim * num_kv_heads // nz_fmt_last_dim,
            self._block_size, nz_fmt_last_dim,
        )
        v_cache_layer = v_cache_layer.view(
            -1, v_head_dim * num_kv_heads // nz_fmt_last_dim,
            self._block_size, nz_fmt_last_dim,
        )
        torch_npu.npu_scatter_pa_kv_cache(
            k_buffer, v_buffer, k_cache_layer, v_cache_layer, slot_mapping,
        )

    # ------------------------------------------------------------------
    # Async notify sender.
    #
    # SendNotify blocks until the peer ACKs (adxl_inner_engine.cc:611-621), and
    # hixl_py holds its one process-wide C++ mutex for the whole call
    # (hixl_py.cc:195-202) — so a slow ACK freezes transfer_async /
    # get_transfer_status / get_notifies too. Neither half of the fix is
    # sufficient alone: this thread keeps the forward thread off the ACK wait,
    # while a short _notify_timeout_ms bounds the freeze, which happens no
    # matter which thread issues the call.
    # ------------------------------------------------------------------
    def _ensure_notify_sender(self) -> None:
        if self._notify_thread is not None:
            return
        with self._notify_thread_lock:
            if self._notify_thread is not None:
                return
            t = threading.Thread(
                target=self._notify_sender_loop,
                name="hixl-notify-sender",
                daemon=True,
            )
            self._notify_thread = t
            t.start()

    def _notify_sender_loop(self) -> None:
        while True:
            item = self._notify_queue.get()
            if item is None:  # shutdown sentinel
                return
            endpoint, name, msg = item
            try:
                # No _hixl_lock: taking it would hand the ACK wait back to the
                # forward thread, which is what this thread exists to avoid.
                self._hixl_send_notify(
                    endpoint, name, msg, self._notify_timeout_ms)
            except Exception as e:  # noqa: BLE001
                # DONE loss falls back to the P-side lease; HB is re-sent next
                # round. Neither warrants a retry that could pile up behind a
                # dead peer.
                logger.warning(
                    "HIXLEngine notify send failed. name=%s endpoint=%s err=%s",
                    name, endpoint, e,
                )

    def _enqueue_notify(self, endpoint: str, name: str, msg: str) -> None:
        """Hand a notify to the sender thread; never blocks the caller."""
        self._ensure_notify_sender()
        try:
            self._notify_queue.put_nowait((endpoint, name, msg))
        except queue.Full:
            if name == "HB":
                logger.debug(
                    "HIXLEngine notify queue full, dropping HB to %s", endpoint)
            else:
                # Loud on purpose: a full queue means the peer is not ACKing.
                logger.warning(
                    "HIXLEngine notify queue full (%d), dropping %s to %s; "
                    "P-side lease expiry is the backstop.",
                    self._notify_queue_size, name, endpoint,
                )

    def _notify_release(self, req_id: str) -> None:
        """Tell every P-side source rank this req actually read from that its
        reads are done.

        Replaces NIXL's make_prepped_xfer(notif_msg=...) auto-delivery
        (pull_worker.py:335). The P-side get_notifies() consumes
        ``("DONE", "<remote_request_id>:<world_size>")`` and decrements its
        per-req consumer counter.

        Only notify ranks this request issued an async READ to
        (``_transferred_ranks``), never broadcast to ``plan.all_source_ranks``.
        Ranks covered by a prefix-hit DONE (no transfer) were already notified
        in _read_blocks and are skipped here to avoid a double-notify.
        """
        meta = self._recving_metadata.get(req_id)
        if meta is None:
            return
        remote_agents = self._remote_agents.get(meta.remote.engine_id, {})
        notif_id = f"{meta.remote.request_id}:{self._world_size}"
        # Prefix-hit ranks already got a DONE in _read_blocks.
        already_notified = self._notified_release_ranks.pop(req_id, set())
        # Real readers only (transfer_async issued in _read_blocks).
        to_notify = self._transferred_ranks.pop(req_id, set()) - already_notified
        for rank in to_notify:
            endpoint = remote_agents.get((0, rank))
            if endpoint is None:
                continue
            self._enqueue_notify(endpoint, "DONE", notif_id)

    def _send_heartbeats(self, metadata: "HIXLEngineConnectorMetadata") -> None:
        """D-side: extend P-side leases for reqs still WAITING in scheduler.

        Mirrors NIXL base_worker.py:2157-2187. For each remote engine in
        metadata.heartbeat_by_engine, send an "HB" notify whose msg is a
        comma-separated list of P-side request ids; the P side extends
        _reqs_to_send expiry on receipt (_handle_heartbeat). Skips engines
        not yet handshaken (the next heartbeat round picks them up).
        """
        for engine_id, hb_info in metadata.heartbeat_by_engine.items():
            if engine_id not in self._remote_agents:
                # Proactive handshake (NIXL base_worker.py:2162-2175). The
                # req behind this heartbeat may still be waiting for peer
                # metadata + Connect to land, so THIS heartbeat round has no
                # agent to send to. Kick off _ensure_handshake now so the
                # NEXT round (one step later) actually delivers the HB and
                # extends the P-side lease — otherwise a first-sight
                # engine's lease is never extended and the P blocks expire
                # before the read.
                self._ensure_handshake(
                    engine_id,
                    hb_info.host,
                    hb_info.port,
                    hb_info.tp_size,
                )
                continue
            req_ids = [rid for rid in hb_info.req_ids if rid]
            if not req_ids:
                continue
            hb_msg = ",".join(req_ids)
            for agent_endpoint in self._remote_agents[engine_id].values():
                self._enqueue_notify(agent_endpoint, "HB", hb_msg)

    def _handle_heartbeat(self, payload: str) -> None:
        """P-side: extend leases for reqs referenced in a heartbeat.

        Mirrors NIXL base_worker.py:2071-2090. payload is a comma-separated
        list of P-side request ids. Each referenced req's expiry is pushed to
        max(old, now + lease_extension) so a late heartbeat never shortens the
        lease.
        """
        new_expiry = time.perf_counter() + self._lease_extension
        for req_id in payload.split(","):
            if req_id in self._reqs_to_send:
                old = self._reqs_to_send[req_id]
                self._reqs_to_send[req_id] = max(old, new_expiry)

    def _get_new_notifs(self) -> set[str]:
        """P-side: drain DONE/HB notifies from the hixl data plane.

        Fork of NIXL pull_worker._get_new_notifs (pull_worker.py:355-408).
        - DONE msg "<remote_request_id>:<world_size>": increment the per-req
          consumer count; release (promote to finished) when it reaches
          consumers_per_producer. For homogeneous TP (D_TP==P_TP) this is 1;
          for D_TP>P_TP (one P rank serves multiple D ranks) it is
          D_TP//P_TP. D_TP<P_TP is 1 (each P rank expects the one D rank
          that read it). TODO(hetero-gather): complex GQA/MLA heterogeneous gather
          needs transfer_topo.tp_ratio for exact per-producer counts.
        - HB: extend lease via _handle_heartbeat.
        """
        notified_req_ids: set[str] = set()
        try:
            # Read-only drain; hixl_py serializes it internally.
            notifs = self._hixl_get_notifies()
        except Exception:
            logger.error("HIXLEngine get_notifies failed", exc_info=True)
            return notified_req_ids
        for name, msg in notifs:
            try:
                if name == "HB":
                    self._handle_heartbeat(msg)
                    continue
                if name != "DONE":
                    continue
                req_id, tp_size = msg.rsplit(":", 1)
                if req_id not in self._reqs_to_send:
                    # Distinguish a premature DONE (D finished reading
                    # before this P worker's request_finished moved the req
                    # into _reqs_to_send — NIXL guards on _reqs_to_process
                    # membership the same way) from a truly unknown/expired
                    # req. The premature case is benign: the lease-expiry
                    # sweep is the backstop that eventually frees the P
                    # blocks, and a later DONE (D re-reads on recompute) will
                    # find the req in _reqs_to_send.
                    if req_id in self._task_tracker:
                        logger.debug(
                            "HIXLEngine premature DONE for in-process req %s; "
                            "dropping (lease expiry is the backstop).", req_id,
                        )
                    else:
                        logger.warning(
                            "HIXLEngine DONE notify for unknown/expired request "
                            "%s; ignoring.", req_id,
                        )
                    continue
                n_consumers = int(tp_size)  # D-side world_size (= D_TP)
                # Mirror NIXL pull_worker.py:388-396: tp_ratio asserts TP
                # divisibility and yields the correct per-producer consumer
                # count for split (D_TP>P_TP => -tp_ratio) and 1 otherwise.
                assert self._transfer_topo is not None
                tp_ratio = self._transfer_topo.tp_ratio(n_consumers)
                consumers_per_producer = (
                    -tp_ratio if n_consumers > self._world_size else 1
                )
                self._consumer_notification_counts_by_req[req_id] += 1
                if (self._consumer_notification_counts_by_req[req_id]
                        >= consumers_per_producer):
                    notified_req_ids.add(req_id)
                    del self._consumer_notification_counts_by_req[req_id]
                    self._task_tracker.update_done_task_count(req_id)
                    self._reqs_to_send.pop(req_id, None)
            except Exception:
                logger.error(
                    "HIXLEngine notify handling failed for %s:%s",
                    name, msg, exc_info=True,
                )
        return notified_req_ids

    def _handle_failed_transfer(self, req_id: str, handle: int | None) -> None:
        """Mark a failed transfer's request and its attention blocks invalid.

        Port of NIXL _handle_failed_transfer (base_worker.py:2139-2155).
        Records the req in _failed_recv_reqs and surfaces its local block
        ids via _invalid_block_ids so the scheduler recomputes them. Only
        attention group blocks are invalidated — Mamba/SSM state groups
        (conv state) are not per-token recomputable and must not be fed into
        the recompute path (NIXL gates on group 0 + _is_hma_required;
        here we skip every state group via _is_ssm_spec).
        """
        self._failed_recv_reqs.add(req_id)
        meta = self._recving_metadata.get(req_id)
        if meta is not None:
            for g, group_blocks in enumerate(meta.local_block_ids):
                if g < len(self._group_spec_types) and self._is_ssm_spec(
                    self._group_spec_types[g]
                ):
                    continue  # state group: not a recompute unit
                self._invalid_block_ids.update(group_blocks)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No-op by default; see ``blocking_layer_wait`` to restore the wait.

        There are no per-layer handles here — TransferAsync is issued once per
        source rank covering every region — so the only thing this method could
        wait for is "all in-flight transfers", i.e. other requests' pulls. That
        is a whole-batch stall for the length of a KV pull.

        It is also unnecessary: get_num_new_matched_tokens returns
        ``(count, True)`` for remote-prefill requests, so the scheduler parks
        them in WAITING_FOR_REMOTE_KVS and keeps them out of the forward batch
        until finished_recving is reported
        (vllm/v1/core/sched/scheduler.py:2192-2203) — nothing in the current
        batch needs the KV that is in flight. Every other P/D pull connector
        (NIXL, Mooncake, MoRIIO, HIXLConnector) is a no-op for the same reason.

        Handles are polled once per step in _pop_done_transfers (get_finished),
        which also owns failure reporting.

        The blocking path below, when enabled: block until in-flight transfers
        covering this layer resolve.

        Replaces hixl_connector's pull_blocks sync + _reformat_staging_to_local.
        Here there is nothing to reformat — once GetTransferStatus returns
        COMPLETED the KV is already in the D real cache at the right layout.

        hixl's GetTransferStatus follows a "not-found" contract: the engine
        drops its record on ANY terminal status (COMPLETED / FAILED / TIMEOUT)
        and on any channel error, after which a re-query returns
        HIXL_PARAM_INVALID (103900). This method runs once per full-attention
        layer per forward, plus again for the MTP drafter forward, so a handle
        would otherwise be queried many times and every query after the first
        terminal one would hit 103900 (the incident root cause). Resolved
        handles are therefore dropped here by replacing each per-req handle
        list with its still-in-flight subset.

        Because the record is consumed by whichever query first sees a
        terminal state, a failure observed here CANNOT be handed over to
        _pop_done_transfers: by then the handle only yields "not found", which
        is indistinguishable from success and would silently release the P
        side's blocks. FAILED/TIMEOUT and hixl errors are therefore recorded
        via _handle_failed_transfer right here, at the point of observation.

        Replacing the handle list (not just polling) is safe because this
        method, _read_blocks (start_load_kv) and _pop_done_transfers
        (get_finished) all run in the same worker thread, never concurrently.
        An emptied per-req list is left in place (not del'd) so
        _pop_done_transfers still sees the req_id and reports it as finished;
        _apply_nz_reformat / _notify_release are gated on _failed_recv_reqs
        there, so a req failed here does not notify the P side. Bounded by
        transfer_timeout to avoid an infinite loop on a stuck handle.
        """
        if not self._blocking_layer_wait:
            # Default path: no-op, matching every other P/D pull connector
            # (NIXL, Mooncake, MoRIIO, HIXLConnector). See the docstring for
            # why blocking here is neither required nor desirable.
            return
        if not self._recving_transfers:
            return
        deadline = time.perf_counter() + self._transfer_timeout_ms / 1000.0
        backoff = 0.001
        while time.perf_counter() < deadline:
            any_waiting = False
            # No _hixl_lock: get_transfer_status is read-only and hixl_py
            # serializes every call on one C++ mutex (hixl_py.h:50-53), so the
            # Python lock adds neither safety nor concurrency.
            # _recving_transfers is worker-thread-only — _read_blocks,
            # _pop_done_transfers and this method never run concurrently, and
            # the handshake thread only extends _ready_requests.
            #
            # Snapshot the req ids: replacing a handle list below can't
            # trigger "dictionary changed size" while iterating.
            for req_id in list(self._recving_transfers.keys()):
                handles = self._recving_transfers[req_id]
                still_in_flight: list[int] = []
                for h in list(handles):
                    try:
                        st = self._hixl_get_transfer_status(h)
                    except Exception as e:
                        # hixl dropped its record as part of this very query,
                        # so _pop_done_transfers would later see only
                        # "not found" — record the failure here or it becomes
                        # a silent success.
                        logger.error(
                            "HIXLEngine wait_for_layer_load poll failed. "
                            "req=%s err=%s", req_id, e,
                        )
                        self._handle_failed_transfer(req_id, h)
                        continue
                    if st is None:
                        # Record gone; cause is unrecoverable from here, and
                        # whoever observed a failure already recorded it.
                        continue
                    sname = self._transfer_status_name(st)
                    if sname == "COMPLETED":
                        continue
                    if sname == "WAITING":
                        any_waiting = True
                        still_in_flight.append(h)
                        continue
                    # Terminal, so hixl released the record on this query —
                    # mark failed now rather than deferring.
                    logger.error(
                        "HIXLEngine transfer failed during layer wait. "
                        "req=%s status=%s", req_id, sname,
                    )
                    self._handle_failed_transfer(req_id, h)
                self._recving_transfers[req_id] = still_in_flight
            if not any_waiting:
                return
            # Exponential backoff (1ms -> 8ms cap) instead of a fixed 1ms
            # busy poll. All-COMPLETED returns above without sleeping.
            # deadline (_transfer_timeout_ms, default 60s) is unaffected.
            time.sleep(backoff)
            backoff = min(backoff * 2, 0.008)
        logger.error(
            "HIXLEngine wait_for_layer_load timed out after %sms for %s; "
            "handles still WAITING are left for get_finished to keep polling.",
            self._transfer_timeout_ms, layer_name,
        )

    def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any = None, **kwargs: Any) -> None:
        # Pull-mode connector does not save (P exposes registered mem for D
        # to READ). Kept for KVConnectorBase_V1 interface compliance.
        pass

    def wait_for_save(self, *args, **kwargs):
        pass

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Return (scheduler-finished req ids, transfer-finished req ids).

        First set (done_sending): P-side reqs whose consumer count reached
        consumers_per_producer via DONE notify, plus reqs whose lease expired
        — both promote via _task_tracker.update_done_task_count. D-side this
        stays empty (no delayed-free lease table).

        Second set (done_recving): req ids whose async READ batches all
        resolved this step, plus any reqs that failed setup/transfer — failed
        reqs are merged in so the scheduler can drive their recompute path
        even though their blocks are already marked invalid (NIXL
        base_worker.py:1958-1968 drains _failed_recv_reqs into done_recving
        for the same reason).
        """
        # P-side: drain DONE/HB notifies; a DONE whose consumer count reaches
        # consumers_per_producer promotes the req to finished via
        # update_done_task_count (NIXL pull_worker._get_new_notifs:355-408).
        self._get_new_notifs()
        done_recving = self._pop_done_transfers(self._recving_transfers)
        # Drop metadata for completed reqs (NIXL base_worker.py:1984). Without
        # this pop _recving_metadata grew monotonically — _pop_done_transfers
        # already del'd _recving_transfers[req_id] and ran failure recovery /
        # _apply_nz_reformat / _notify_release against the still-present meta,
        # so it is safe to drop now.
        for req_id in done_recving:
            self._recving_metadata.pop(req_id, None)
            self._notified_release_ranks.pop(req_id, None)
            self._transferred_ranks.pop(req_id, None)
        failed = set(self._failed_recv_reqs)
        self._failed_recv_reqs.clear()
        # Also drop metadata for reqs that failed without ever issuing a
        # transfer (e.g. handshake failure on a parked req) — they
        # never enter _recving_transfers so _pop_done_transfers won't surface
        # them, and would otherwise leak in _recving_metadata.
        for req_id in failed:
            self._recving_metadata.pop(req_id, None)
            self._notified_release_ranks.pop(req_id, None)
            self._transferred_ranks.pop(req_id, None)
            # Failed before any handle resolved: never reaches
            # _log_transfer_bandwidth, so drop the accounting here.
            self._xfer_bytes.pop(req_id, None)
            self._xfer_start.pop(req_id, None)
        # Drain reqs whose handshakes finished during this step's forward:
        # issue their READ at step end so next step's wait_for_layer_load
        # sees COMPLETED handles sooner (one less step of cold-start KV-read
        # latency). See _drain_ready_requests.
        self._drain_ready_requests()
        # Lease expiry: force-release reqs whose lease lapsed before every
        # consumer reported DONE (NIXL base_worker.py:2027-2044). Full scan
        # (not NIXL's sorted-dict early-exit) — _reqs_to_send is
        # insertion-ordered not expiry-ordered, and per-step counts are small.
        now = time.perf_counter()
        for req_id in [rid for rid, exp in self._reqs_to_send.items()
                       if now >= exp]:
            count = self._consumer_notification_counts_by_req.pop(req_id, 0)
            logger.info(
                "HIXLEngine releasing expired KV blocks for request %s "
                "retrieved by %d consumer(s) before lease expired.",
                req_id, count,
            )
            self._task_tracker.update_done_task_count(req_id)
            del self._reqs_to_send[req_id]
        done_sending = self._task_tracker.get_and_clear_finished_requests()
        return done_sending, done_recving | failed

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Local block ids whose transfer failed — scheduler recomputes them.

        Drains and clears so the same batch is not reported twice (NIXL
        base_worker.py:2373-2380 uses a get_nowait drain for the same effect).
        """
        result = set(self._invalid_block_ids)
        self._invalid_block_ids.clear()
        return result

    # ==================================================================
    # Handshake metadata (ZMQ side channel — address fields, not block keys)
    # ==================================================================
    def get_handshake_metadata(self):  # noqa: D401
        """Return this worker's handshake payload (built in
        register_kv_caches). The framework ships it to the scheduler, which
        fans it out to peers via set_xfer_handshake_metadata_pp_aware.
        """
        logger.debug(
            "HIXLTRACE P-get_handshake engine=%s has_payload=%d",
            self._engine_id, self._xfer_handshake_metadata is not None,
        )
        return self._xfer_handshake_metadata

    def set_xfer_handshake_metadata(
        self, metadata: Mapping[int | tuple[int, ...], Any]
    ) -> None:
        """Scheduler-side: aggregate per-worker handshake payloads.

        NIXL single-listener model (base_scheduler.py:248-289): encode each
        worker's HixlEngineHandshakePayload under its (pp, tp) key and start
        one scheduler-side ROUTER (self._handshake_listener_thread) that
        serves them all, routing by (pp, tp). The signature stays relaxed
        to ``Mapping[int | tuple[int, ...], Any]`` (and keys are folded via
        key[0], key[1]) because vllm-ascend worker.py:875 emits 3-tuple
        ``(pp, pcp, tp)`` keys when pcp_size > 1 — NIXL's strict 2-tuple
        dict signature would reject them.
        """
        self._store_handshake_payloads(metadata)

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: Mapping[int | tuple[int, ...], Any]
    ) -> None:
        """PP-aware variant: keys are (pp_rank, tp_rank). Same NIXL
        single-listener path as set_xfer_handshake_metadata.
        """
        self._store_handshake_payloads(metadata)

    def _store_handshake_payloads(
        self, metadata: Mapping[int | tuple[int, ...], Any]
    ) -> None:
        encoder = msgspec.msgpack.Encoder()
        # Serialize with the listener thread: it may iterate
        # self._handshake_payloads in the unknown-rank warning path while we
        # resize here.
        with self._handshake_lock:
            for key, payload in metadata.items():
                if isinstance(key, tuple):
                    pp_rank, tp_rank = key[0], key[1]
                else:
                    # Non-pp form: key is a flat rank; pull is single-stage so
                    # treat it as tp_rank under pp=0.
                    pp_rank, tp_rank = 0, key
                self._handshake_payloads[(pp_rank, tp_rank)] = encoder.encode(
                    payload
                )
            nonempty = bool(self._handshake_payloads)
        logger.debug(
            "HIXLTRACE P-store_handshake n_payloads=%d keys=%s side_port=%d",
            len(self._handshake_payloads), list(self._handshake_payloads),
            self._side_channel_port,
        )
        # Only the P side receives worker payloads, so starting the listener
        # here binds it exactly where request_finished advertised
        # remote_port = _side_channel_port.
        if nonempty:
            self._ensure_handshake_listener()

    def shutdown(self):
        # Stop the P-side handshake ROUTER listener (Step A). Both roles own
        # _handshake_stop_event / _handshake_listener_thread (built before the
        # SCHEDULER early-return), so this is safe for both.
        self._handshake_stop_event.set()
        if self._handshake_listener_thread is not None:
            self._handshake_listener_thread.join(timeout=2.0)
        # Cancel queued handshakes (executor shared by both roles).
        self._handshake_initiation_executor.shutdown(wait=True, cancel_futures=True)
        # SCHEDULER role has no data plane — _hixl / _hixl_lock /
        # _kv_mem_handles were never built (the early-return skipped worker
        # init). Only WORKER finalizes the hixl engine; without this guard
        # the SCHEDULER shutdown would AttributeError on _hixl_lock.
        if self._scheduler is not None:
            return
        # Drain before finalizing the engine below. Must sit after the
        # SCHEDULER early-return: _notify_queue / _notify_thread are built with
        # the worker data plane and do not exist on the scheduler side.
        # Bounded join so a peer that stopped ACKing cannot hang shutdown.
        if self._notify_thread is not None:
            try:
                self._notify_queue.put_nowait(None)
            except queue.Full:
                pass
            self._notify_thread.join(timeout=2.0)
        # Let in-flight connect() finish before we pull the engine out from
        # under it (connect runs on the executor above and takes _hixl_lock —
        # finalizing underneath it would be a use-after-free).
        with self._hixl_lock:
            for handle, _, _ in self._kv_mem_handles.values():
                try:
                    self._hixl_deregister_mem(handle)
                except Exception:  # noqa: BLE001
                    pass
            self._hixl_finalize()
