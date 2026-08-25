# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HIXLEngineConnector — address-level KV transfer via hixl::Hixl.

Data-plane counterpart of NIXL's pull connector. HIXLConnector talks to
LLM-DataDist's block API and needs staging + post-transpose for TP>1;
this connector drives RegisterMem / TransferAsync / GetTransferStatus /
Connect directly. Heterogeneous-TP head split/gather is remote_addr
offsets in TransferOpDesc, so KV lands in the D cache at its real layout.

Control plane (ZMQ handshake / scheduler decisions / delayed free) is
forked from NIXL and self-contained here — it does not import
hixl_connector.py. DONE/HB go over the ZMQ side channel, not HIXL
SendNotify. No RecvingThread: handles are polled in get_finished on the
worker main thread (wait_for_layer_load is a no-op by default).
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
# Control-plane constants. This module does not import hixl_connector.
# DONE/HB share the ZMQ ROUTER with GET_META so they do not share the
# HIXL CommEngine control TCP with Transfer BufferReq.
# ---------------------------------------------------------------------------
GET_META_MSG = b"get_meta_msg"
NOTIFY_MSG = b"notify_msg"
_NOTIFY_ACK = b"ACK"
_NOTIFY_ALL_RANKS = -1
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


def _flatten_grc_protocol_desc(options: dict[str, str]) -> bool:
    """Rewrite nested protocol_desc to the flat key HixlOptions::from_json reads.

    C++ only checks ``comm_resource_config.protocol_desc`` on the GRC root.
    Nested ``{"comm_resource_config":{"protocol_desc":[...]}}`` is valid JSON
    but ignored, so Factory falls through to CommEngine on 910.
    """
    raw = options.get(_HIXL_OPTION_GLOBAL_RESOURCE_CONFIG)
    if not raw:
        return False
    obj = _loads_json_object(raw)
    if obj is None:
        return False
    if _HIXL_PROTOCOL_DESC_FLAT in obj:
        return False
    crc = obj.get("comm_resource_config")
    if not isinstance(crc, dict) or "protocol_desc" not in crc:
        return False
    desc = crc.pop("protocol_desc")
    obj[_HIXL_PROTOCOL_DESC_FLAT] = desc
    if not crc:
        obj.pop("comm_resource_config", None)
    options[_HIXL_OPTION_GLOBAL_RESOURCE_CONFIG] = json.dumps(
        obj, separators=(",", ":")
    )
    return True


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
        # CS only: flatten before the LCR/protocol_desc checks so
        # _has_protocol_desc sees the form C++ reads. comm skips it —
        # _strip_protocol_desc handles both forms, and the warning
        # below is off-topic on the comm path.
        if _flatten_grc_protocol_desc(out):
            logger.warning(
                "hixl_engine.options GlobalResourceConfig used nested "
                "comm_resource_config.protocol_desc; flattened to the key "
                "HixlOptions parses. Nested form is ignored by C++ and would "
                "silently select CommEngine."
            )
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
    update_done_task_count is called from _count_done_notify once a req's
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
        # True iff this P worker has the req in-batch. _park_pending_done
        # parks only those; an id this worker never owned is discarded.
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

    Mirrors NixlAgentMetadata but swaps the
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
# Per-request metadata (forked from NIXL so the connector
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
    # P writes tokens, not full scheduler blocks. D uses these to drop the
    # unwritten hybrid-alignment tail after logical→kernel expand
    # (Mooncake _get_kernel_block_ids). 0 / missing → no kernel clip.
    num_external_tokens: int = 0
    num_computed_tokens: int = 0
    # Filled by _read_blocks after prefix trim + kernel clip. NZ reformat
    # uses these so it only touches pages that were actually written.
    local_kernel_ids: list[list[int]] | None = None


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

    Mirrors NIXL compute_nixl_compatibility_hash but
    without NIXL_CONNECTOR_VERSION (HIXLEngineConnector is new, versioned via
    the hash itself). Bump the prefix string when the on-wire metadata schema
    changes in a backward-incompatible way.
    """
    # v2: a logical region is keyed by (base, per_block, group), not base
    # alone, so kv_caches_base_addr / block_lens gained one entry per
    # sharing group. Bumping the prefix rejects a v1 peer at handshake
    # instead of failing the region-parity assert in _build_op_descs.
    prefix = "hixl-engine-v2"
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


# Lease / heartbeat constants (mirrors NIXL base_scheduler).
_HIXL_ENGINE_LEASE_DURATION_S = 30
# D-side REQ receive timeout (NIXL uses 5s).
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
        # P scheduler ROUTER → workers: (name, msg, target_tp). target_tp
        # is -1 for all ranks (HB); DONE is the P tp rank that was read.
        self.inbound_notifies: list[tuple[str, str, int]] = []

    def _add_new_req(
        self,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict[str, Any],
        num_external_tokens: int = 0,
    ) -> HIXLEngineReqMeta:
        return HIXLEngineReqMeta(
            local_block_ids=local_block_ids,
            tp_size=kv_transfer_params.get("tp_size", 1),
            num_external_tokens=num_external_tokens,
            num_computed_tokens=int(
                kv_transfer_params.get("num_computed_tokens", 0) or 0
            ),
        )

    def add_new_req_to_recv(
        self,
        request_id: str,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict[str, Any],
        num_external_tokens: int = 0,
    ) -> None:
        req = self._add_new_req(
            local_block_ids, kv_transfer_params, num_external_tokens,
        )
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

    Forked from NIXL NixlPullConnectorScheduler and
    NixlBaseConnectorScheduler. Does
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
        # truncation; models with compress_ratios need it too.
        self._use_compress = self._model_uses_compress()
        self._need_truncate = self._use_compress or self._has_mamba
        # (request, local_block_ids, num_external_tokens). Token count is
        # required on the worker for attn kernel-tail clip; block ids alone
        # cannot express a partial 1536-token scheduler block.
        self._reqs_need_recv: dict[str, tuple["Request", BlockIds, int]] = {}
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
        # Per-group prompt span for P-side MTP-extra clip (Mooncake
        # _get_transfer_block_ids). State groups keep the full table;
        # attention is cut to cdiv(prompt_len, tokens_per_block * cp).
        pc = vllm_config.parallel_config
        self._cp_size = max(
            1,
            int(getattr(pc, "prefill_context_parallel_size", 1) or 1)
            * int(getattr(pc, "decode_context_parallel_size", 1) or 1),
        )
        self._group_is_state: list[bool] = []
        self._group_tokens_per_block: list[int] = []
        for group in kv_cache_config.kv_cache_groups:
            specs = self._group_unique_specs(group)
            is_state = any(isinstance(spec, MambaSpec) for spec in specs)
            first = specs[0] if specs else group.kv_cache_spec
            block_size = getattr(
                group.kv_cache_spec, "block_size",
                getattr(first, "block_size", self.block_size),
            )
            compress = 1
            for spec in specs:
                ratio = getattr(spec, "compress_ratio", None)
                if ratio:
                    compress = int(ratio)
            self._group_is_state.append(is_state)
            self._group_tokens_per_block.append(
                int(block_size) * max(1, compress)
            )
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

    # -- heartbeat bookkeeping ------------------------------------------
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

    # -- SWA clipping ---------------------------------------------------
    def get_sw_clipped_blocks(self, block_ids: BlockIds) -> BlockIds:
        if len(block_ids) == 0 or not self._is_hma_required:
            return block_ids
        assert len(block_ids) == len(self.blocks_per_sw)
        return tuple(
            blocks[-self.blocks_per_sw[i]:] if self.blocks_per_sw[i] > 0
            else blocks
            for i, blocks in enumerate(block_ids)
        )

    @staticmethod
    def _group_unique_specs(group: Any) -> list[Any]:
        spec = group.kv_cache_spec
        if not isinstance(spec, UniformTypeKVCacheSpecs):
            return [spec]
        specs: list[Any] = []
        for layer_name in group.layer_names:
            layer_spec = spec.kv_cache_specs[layer_name]
            if layer_spec not in specs:
                specs.append(layer_spec)
        return specs

    def _get_transfer_block_ids(
        self, block_ids: BlockIds, prompt_len: int,
    ) -> BlockIds:
        """Keep prompt KV blocks; drop MTP extras on attention groups.

        Mirrors MooncakeConnectorScheduler._get_transfer_block_ids. State
        groups (Mamba) are not context-aligned with attention, so they
        pass through. Attention is cut to
        ``cdiv(prompt_len, tokens_per_block * cp_size)`` from the front
        — the prefix that holds prompt tokens, not the tail (which is
        where align / MTP extra slots land).
        """
        if len(block_ids) == 0:
            return block_ids
        assert len(block_ids) == len(self._group_is_state), (
            f"block groups {len(block_ids)} != "
            f"kv_cache_groups {len(self._group_is_state)}"
        )
        cp_size = max(1, self._cp_size)
        out: list[list[int]] = []
        for blocks, is_state, tokens_per_block in zip(
            block_ids, self._group_is_state, self._group_tokens_per_block,
        ):
            if is_state:
                out.append(list(blocks))
                continue
            span = max(1, int(tokens_per_block)) * cp_size
            n = cdiv(max(0, int(prompt_len)), span)
            out.append(list(blocks[:n]))
        return tuple(out)

    # -- truncate helpers (Mamba + compress-ratio) ----------------------
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

    # -- four decision methods ------------------------------------------
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            # Worker kernel-tail clip reads this off ReqMeta. Block ids
            # lose the intra-block token count (1 and 1536 both look
            # like 1 scheduler block).
            params["num_computed_tokens"] = num_computed_tokens
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
                        request, local_block_ids, num_external_tokens)
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
            self._reqs_need_recv[request.request_id] = (request, [], 0)
            params["do_remote_prefill"] = False
            return False, None
        if is_d_node and not self.is_bidirectional_kv_xfer_enabled:
            return False, None
        if request.status not in (RequestStatus.FINISHED_LENGTH_CAPPED,
                                  RequestStatus.FINISHED_STOPPED):
            self._reqs_not_processed.add(request.request_id)
            return False, None
        token_ids = request.prompt_token_ids or []
        prompt_len = (
            len(token_ids)
            if token_ids
            else int(getattr(request, "num_prompt_tokens", 0) or 0)
        )
        if block_ids:
            # P only: drop MTP extras from the front, then SWA tail.
            # Same order as Mooncake request_finished. D bidirectional
            # must not clip to prompt_len — those blocks include decode.
            if is_p_node:
                block_ids = self._get_transfer_block_ids(
                    block_ids, prompt_len)
            block_ids = self.get_sw_clipped_blocks(block_ids)
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
            remote_num_tokens = request.num_computed_tokens
        logger.debug(
            "HIXLTRACE P-request_finished req=%s is_p=%d delay_free=%d "
            "remote_port=%d prompt_len=%d n_blocks=%s",
            request.request_id, is_p_node, delay_free_blocks,
            self.side_channel_port, prompt_len,
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

    # -- metadata builder -----------------------------------------------
    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> "KVConnectorMetadata":
        meta = HIXLEngineConnectorMetadata()
        for req_id, (req, block_ids, num_external_tokens) in (
            self._reqs_need_recv.items()
        ):
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                num_external_tokens=num_external_tokens,
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
        # Align / MTP: SSM block table is 1 committed + N speculative slots.
        # _apply_prefix_caching picks remote[len - spec - 1] (Mooncake /
        # hixl_connector). 0 when speculative_config is absent.
        spec_cfg = getattr(vllm_config, "speculative_config", None)
        self._num_speculative_tokens = (
            int(getattr(spec_cfg, "num_speculative_tokens", 0) or 0)
            if spec_cfg is not None
            else 0
        )

        # Parallel / model identity (mirrors NIXL base_worker).
        kvtc = vllm_config.kv_transfer_config
        self._engine_id = str(kvtc.engine_id)
        if role == KVConnectorRole.SCHEDULER:
            # SCHEDULER 在 EngineCore 主进程创建，并行组尚未初始化。
            # 调 get_tensor_model_parallel_rank() 会断言失败。本角色
            # 只委托 HIXLEngineConnectorScheduler，不消费 tp_rank。
            # 握手 endpoint 由 WORKER 填写。
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
            # CP parallel-state. DCP helpers are imported lazily so unused
            # CP does not become a hard dependency. One ROUTER per
            # (engine_id, dp_index) routes by (pp, tp); CP shards share
            # that listener (TODO: per-pcp routing if non-HMA CP is required).
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
        # Worker 数据面依赖并行组与 vLLM config context，主进程没有。
        # SCHEDULER 只建 scheduler 与握手路由表后 return。
        # 握手字段两边都建：SCHEDULER 跑 ROUTER；WORKER 的 shutdown
        # 要能 join 未启动的 listener / 停 executor。
        # (pp, tp) -> encoded handshake payload，由
        # set_xfer_handshake_metadata[_pp_aware] 写入，ROUTER 按键路由。
        self._handshake_initiation_executor = ThreadPoolExecutor(max_workers=1)
        self._handshake_lock = threading.RLock()
        self._handshake_stop_event = threading.Event()
        self._handshake_listener_thread: threading.Thread | None = None
        self._handshake_payloads: dict[tuple[int, int], bytes] = {}
        # DONE/HB received on the scheduler ROUTER; copied into metadata
        # each step so P workers can apply lease/DONE counting.
        self._inbound_notifies: list[tuple[str, str, int]] = []
        self._inbound_notifies_lock = threading.Lock()
        if role == KVConnectorRole.SCHEDULER:
            # One scheduler-side ROUTER. Base port is
            # hixl_engine.side_channel_port; data_parallel_index separates
            # DP groups. request_finished writes the bound port into
            # kv_transfer_params.remote_port so D's REQ lands on it.
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
        self._attn_compress_ratio = 1
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
            for group in kv_cache_config.kv_cache_groups:
                spec = group.kv_cache_spec
                cr = getattr(spec, "compress_ratio", None)
                if not isinstance(cr, int):
                    inner = getattr(spec, "kv_cache_spec", None)
                    cr = getattr(inner, "compress_ratio", None)
                if isinstance(cr, int) and cr > 1:
                    self._attn_compress_ratio = cr
                    break

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
        # Conv-state split is used only to fill handshake ssm_sizes.
        # _build_op_descs still addresses SSM as one linear region; the
        # decomp object is not consumed on the transfer path.
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
                # DS：用 NIXL 3-read 分解取 ssm_sizes。子投影传输未移植，
                # _build_op_descs 仍整块寻址；decomp 对象不进传输路径。
                self._conv_decomp = derive_mamba_conv_split(
                    mamba_spec, self._tp_size
                )
                mamba_ssm_size = self._conv_decomp.ssm_sizes
            else:
                # SD：Ascend convStates=(num_cache_lines, state_len, dim)。
                # derive_mamba_conv_split 内部断言 DS，不能调用。
                # 按 numel*dtype 算 ssm_sizes（布局无关）。P_TP==D_TP
                # 整块 memcpy 安全；P_TP>D_TP 的 slot*chunk 假设 DS，未验证。
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
        # Local transfer topology; built lazily in register_kv_caches.
        # Used by compute_tp_mapping.
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
        # (host, port, name, msg, target_tp) or shutdown sentinel None.
        self._notify_queue: "queue.Queue[tuple[str, int, str, str, int] | None]" = (
            queue.Queue(maxsize=self._notify_queue_size))
        self._notify_thread: threading.Thread | None = None
        self._notify_thread_lock = threading.Lock()
        # endpoint -> perf_counter deadline; SendNotify failures trip this.
        self._notify_dead_until: dict[str, float] = {}
        # Handles abandoned after a wait-timeout so GetTransferStatus can
        # still drain them (no cancel API; dropping leaks RDMA reqs).
        self._orphan_xfer_handles: list[int] = []
        # request_id -> HIXLEngineReqMeta (for failure recovery / post-process)
        self._recving_metadata: dict[str, HIXLEngineReqMeta] = {}
        # Reqs whose handshake hasn't landed yet sit on
        # _pending_handshake_reqs; the done_callback releases them into
        # _ready_requests for the next drain.
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
        # D-side REQ connect futures. Executor / lock are built above so
        # both roles own them; only WORKER populates this map.
        self._handshake_futures: dict[str, Future] = {}
        # req_id -> perf_counter lease expiry (P-side delayed free).
        self._reqs_to_send: dict[str, float] = {}
        # DONE notifies that arrived before _reqs_to_send knew the req, parked
        # for replay: req_id -> (msg, count, first_seen). The count matters —
        # with D_TP > P_TP one req draws consumers_per_producer > 1 DONEs, and
        # collapsing them would leave the req short of its promote threshold.
        # first_seen is only written on insert so the TTL measures age, not
        # last touch.
        self._pending_dones: dict[str, tuple[str, int, float]] = {}
        # Multi-consumer done-notification counting (mirrors NIXL).
        self._consumer_notification_counts_by_req: dict[str, int] = defaultdict(int)
        # P-side lease / heartbeat timing (mirrors NIXL base_worker). Used by
        # get_finished (lease expiry) and _handle_heartbeat (extension).
        self._kv_lease_duration: int = kvtc.get_from_extra_config(
            "kv_lease_duration", 30)
        # Heartbeats only extend the lease when remaining < _lease_extension
        # (2/3 of duration), so it converges to now+extension instead of
        # growing unboundedly on each heartbeat.
        self._lease_extension: int = self._kv_lease_duration * 2 // 3
        # Parking a DONE longer than the lease is pointless: by then the
        # expiry sweep has already released the blocks the DONE would free.
        self._pending_done_ttl_s: float = float(self._kv_lease_duration)
        if self._xfer_wait_timeout_s <= 0.0:
            # Fail the req before P lease expiry so D does not keep READing
            # blocks P already freed (log: 44s wait vs 30s lease).
            self._xfer_wait_timeout_s = float(
                max(1, int(self._kv_lease_duration) - 5)
            )

    # ------------------------------------------------------------------
    # Config & kernel-block-size derivation.
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
        # Unused for DONE/HB (those go over ZMQ). Kept so a leftover
        # GetNotifies drain and debug scripts still have a send path.
        nd = self._hixl_mod.NotifyDesc(name=name, notify_msg=msg)
        status = self._hixl.send_notify(remote_engine, nd, timeout_ms)
        self._hixl_check(status, "SendNotify")

    def _hixl_get_notifies(self) -> list[tuple[str, str]]:
        status, ns = self._hixl.get_notifies()
        self._hixl_check(status, "GetNotifies")
        return [(n.name, n.notify_msg) for n in ns]

    # ------------------------------------------------------------------
    def _parse_hixl_engine_config(self, vllm_config: VllmConfig) -> None:
        """Read kv_connector_extra_config.hixl_engine.

        Fields: local_engine (host:port for hixl Initialize), backend
        (hixl_cs|comm, default hixl_cs), options (dict passed through to
        Initialize after backend injection), link_timeout_ms,
        transfer_timeout_ms, side_channel_port (base ZMQ handshake port;
        the scheduler ROUTER binds base + data_parallel_index; D learns
        it from remote_port written by P's request_finished).
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
        # ZMQ REQ timeout for DONE/HB. No longer bounds a HIXL C++ mutex
        # (notifies left the HIXL control socket); 1000ms is enough for
        # a side-channel RTT without stalling TransferAsync.
        self._notify_timeout_ms: int = int(cfg.get("notify_timeout_ms", 1000))
        self._notify_queue_size: int = int(cfg.get("notify_queue_size", 1024))
        # After a ZMQ DONE/HB send fails, stop hammering that scheduler
        # side-channel for cooldown seconds (key is host:port).
        self._notify_dead_cooldown_s: float = float(
            cfg.get("notify_dead_cooldown_s", 5.0)
        )
        # 0 = derive from kv_lease_duration after worker init (lease - 5s).
        self._xfer_wait_timeout_s: float = float(
            cfg.get("transfer_wait_timeout_s", 0.0)
        )
        # Opt-in; default is no-op (see wait_for_layer_load).
        self._blocking_layer_wait: bool = bool(
            cfg.get("blocking_layer_wait", False))
        # DONE/HB arrive over the ZMQ side channel, so GetNotifies returns
        # empty while still taking the global hixl_py mutex on every engine
        # step. Only a peer old enough to still SendNotify needs this drain.
        self._drain_hixl_notifies: bool = bool(
            cfg.get("drain_hixl_notifies", False))
        # Cap on DONE notifies parked for replay (see _pending_dones). A flood
        # of ids this worker never owned must not grow without bound.
        self._pending_done_max: int = int(cfg.get("pending_done_max", 1024))
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

        Mirrors NIXL _sync_block_size_with_kernel. If the user block_size is larger
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
    # ZMQ side-channel handshake (D-side REQ client). The P-side ROUTER
    # is _handshake_listener_thread, started in set_xfer_handshake_metadata.
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
        ranks and must collect each rank's base addresses. Loop
        handshake_target_ranks over one ZMQ REQ socket, keep the
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
                # lowest-RTT sample.
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
        mapping.
        """
        # Evict engines past their TTL before adding a new one.
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
                # Release parked reqs under _handshake_lock so the park
                # in start_load_kv and this release are mutually exclusive.
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
    # Remote engine connect + TP mapping. hixl Connect takes the endpoint
    # string directly; TransferAsync takes op_descs with no pre-built
    # xfer-side handles.
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

        Mirrors NIXL add_remote_agent + compute_tp_mapping +
        _validate_remote_agent_handshake, minus prep_xfer_dlist /
        dst_xfer_side_handles.
        """
        assert self._transfer_topo is not None, (
            "register_kv_caches must run before handshake completion"
        )
        # Pull mode addresses by region index assuming P/D region arrays line
        # up 1:1, which only holds for pipeline_parallel_size==1 (single
        # stage). PP-stage window slicing is not ported, so fail closed
        # rather than land a region-parity IndexError downstream.
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
            # get_engine_info are usable during validation.
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

        Mirrors NIXL _validate_remote_agent_handshake. HIXLEngine only checks the
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
        # Per-region block_len compatibility. Branch like NIXL: replicated /
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
    # TTL eviction. Stale engines
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
        # successful connect, _remote_metadata is still empty, so fall back
        # to _remote_agents which is populated right after connect() succeeds.
        endpoints: set[str] = set()
        peer_meta = self._remote_metadata.get(remote_engine_id)
        if peer_meta is not None:
            endpoints.add(peer_meta.local_engine_endpoint)
        for ep in self._remote_agents.get(remote_engine_id, {}).values():
            endpoints.add(ep)
        for endpoint in endpoints:
            self._notify_dead_until.pop(endpoint, None)
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
    # P-side ZMQ ROUTER. GET_META_MSG replies [handshake_bytes,
    # perf_counter_ts]. Also accepts NOTIFY_MSG (DONE/HB) so those never
    # share the HIXL control TCP.
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
        if not isinstance(msg, (list, tuple)) or not msg:
            self._reject_handshake(sock, identity, "empty side-channel payload")
            return
        if msg[0] == NOTIFY_MSG:
            self._handle_notify_request(sock, identity, msg)
            return
        if msg[0] != GET_META_MSG:
            self._reject_handshake(sock, identity, "not a GET_META message")
            return
        # Single-listener routing: the D side
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

    def _handle_notify_request(
        self, sock: zmq.Socket, identity: bytes, msg: Any,  # type: ignore[name-defined]
    ) -> None:
        """Accept DONE/HB from D. Frame: [NOTIFY_MSG, name, body, target_tp].

        target_tp is the P tp rank that should apply a DONE; -1 means all
        ranks (HB). Queued here and copied into connector metadata next
        step so each P worker applies locally.
        """
        if len(msg) < 4:
            self._reject_handshake(sock, identity, "NOTIFY missing fields")
            return
        name, body, target_tp = msg[1], msg[2], msg[3]
        if isinstance(name, bytes):
            name = name.decode()
        if isinstance(body, bytes):
            body = body.decode()
        try:
            target_tp = int(target_tp)
        except (TypeError, ValueError):
            self._reject_handshake(sock, identity, "NOTIFY target_tp not int")
            return
        with self._inbound_notifies_lock:
            self._inbound_notifies.append((str(name), str(body), target_tp))
        try:
            sock.send_multipart((identity, b"", _NOTIFY_ACK))
        except Exception:  # noqa: BLE001
            logger.error("HIXLEngine notify ACK failed: name=%s", name)

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
        meta = self._scheduler.build_connector_meta(scheduler_output)
        if isinstance(meta, HIXLEngineConnectorMetadata):
            with self._inbound_notifies_lock:
                if self._inbound_notifies:
                    meta.inbound_notifies = list(self._inbound_notifies)
                    self._inbound_notifies.clear()
        return meta

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

        Address-level, MEM_DEVICE. No BlocksCacheKey/CacheDesc —
        heterogeneous shapes (conv 2D / ssm 3D / MLA latent) coexist on the
        same engine. Uses address-level RegisterMem and skips NIXL's
        prep_xfer_dlist / get_agent_metadata (hixl Connect uses the
        endpoint string directly).

        Physical registration and logical regions are counted separately,
        so n_regions >= n_handles. HMA pools several layers onto one
        tensor; every logical view needs its own region record (own
        per_block stride and group block ids) while the segment is
        registered once.
        """
        # Read NZ switch + save layer tensors for post-transfer ND->NZ
        # reformat. Imported lazily so the module stays importable without
        # an NPU (ascend_config pulls in the NPU stack).
        from vllm_ascend.ascend_config import get_ascend_config  # noqa: PLC0415
        self._enable_kv_nz = bool(
            getattr(get_ascend_config(), "enable_kv_nz", False))
        self._kv_caches = kv_caches
        self._warn_if_kv_zeroing_disabled()
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
        # Real per-page bytes from each tensor's own shape. vllm-ascend
        # pads spec.page_size_bytes so attn and mamba share an HMA page
        # (attn + conv). That inflates FA to 66816 (65536+1280) and SSM
        # past the real state. Addressing always uses this tensor size.
        self._per_block_per_layer = []
        seen_base_addresses: list[int] = []
        self._kv_caches_base_addr[self._engine_id][self._tp_rank] = (
            seen_base_addresses
        )

        # A logical region is identified by (base, per_block, group) — NOT by
        # base alone. HMA pools one layer from every kv_cache_group onto a
        # single KVCacheTensor; each shared layer receives the same tensor
        # object, so a base-only key collapses regions that must be addressed
        # with different per-group block ids (and would drop later mamba
        # groups' conv state). register_mem still keys on base alone:
        # TransferAsync addresses by (addr, len) and never dereferences a
        # mem handle, so one registration per physical segment covers every
        # logical view of it.
        registered_bases: set[int] = set()
        seen_logical: set[tuple[int, int, int]] = set()

        for layer_name, cache_or_caches in kv_caches.items():
            layer_spec = self._layer_specs.get(layer_name)
            if layer_spec is None:
                # No kv_cache_group owns this layer, so no block table
                # addresses it and nothing can be transferred for it. Warn
                # instead of dropping silently: a spec-lookup miss on a real
                # layer loses that layer's whole KV with no other symptom.
                logger.warning(
                    "HIXLEngine layer %s has no kv_cache_group spec; its KV "
                    "will NOT be registered or transferred.", layer_name,
                )
                continue
            if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                # MLA DSv32 Indexer: merge specs, pick this layer's own.
                layer_spec = layer_spec.kv_cache_specs[layer_name]
            tensors = (
                list(cache_or_caches)
                if isinstance(cache_or_caches, (list, tuple))
                else [cache_or_caches]
            )
            # Per-tensor page stride after K/V split.
            is_mla_region = isinstance(
                layer_spec, (MLAAttentionSpec, SlidingWindowMLASpec)
            )
            group_idx = self._layer_to_group.get(layer_name, 0)
            for i, t in enumerate(tensors):
                base_addr = int(t.data_ptr())
                per_block = t[0].numel() * t.element_size()
                is_mamba_region = isinstance(layer_spec, MambaSpec)
                length = t.numel() * t.element_size()
                logical_key = (base_addr, per_block, group_idx)
                if logical_key in seen_logical:
                    # Same segment, same block view, same group: a genuine
                    # duplicate. The hybrid attn-mamba path aliases k and v
                    # onto one tensor (model_runner_v1.py:4442-4444), which
                    # would otherwise transfer the same bytes twice.
                    logger.debug(
                        "HIXLTRACE reg_dedup_skip layer=%s sub=%d base=0x%x "
                        "length=%d per_block=%d is_mamba=%d group=%d shape=%s",
                        layer_name, i, base_addr, length, per_block,
                        is_mamba_region, group_idx, tuple(t.shape),
                    )
                    continue
                seen_logical.add(logical_key)
                if base_addr in registered_bases:
                    # Another group's layer (or another block view of this
                    # layer) already registered this segment. Record the
                    # logical region, skip register_mem.
                    kind = "reg_alias"
                else:
                    with self._hixl_lock:
                        handle = self._hixl_register_mem(
                            base_addr, length, is_device=True
                        )
                    self._kv_mem_handles[f"{layer_name}/{i}"] = (
                        handle, base_addr, length,
                    )
                    registered_bases.add(base_addr)
                    kind = "reg_region"
                seen_base_addresses.append(base_addr)
                # Handshake block_lens and FA/SSM addressing both use the
                # tensor page. spec.page_size_bytes is HMA-padded
                # (66816 = 65536 + conv/12/2) and must not be shipped.
                self._block_len_per_layer.append(per_block)
                self._per_block_per_layer.append(per_block)
                self._region_is_mla.append(is_mla_region)
                self._region_group_idx.append(group_idx)
                # Torch uses -1 for CPU; hixl needs a non-negative device id.
                self._device_id = max(t.get_device(), 0)
                logger.debug(
                    "HIXLTRACE %s layer=%s sub=%d base=0x%x length=%d "
                    "per_block=%d is_mamba=%d group=%d shape=%s",
                    kind, layer_name, i, base_addr, length, per_block,
                    is_mamba_region, group_idx, tuple(t.shape),
                )

        self._build_transfer_topology(kv_caches)
        self._build_xfer_handshake_metadata()
        _pb_counts: dict[int, int] = {}
        for _pb in self._per_block_per_layer:
            _pb_counts[_pb] = _pb_counts.get(_pb, 0) + 1
        _grp_counts: dict[int, int] = {}
        for _g in self._region_group_idx:
            _grp_counts[_g] = _grp_counts.get(_g, 0) + 1
        # INFO: group_dist must be symmetric per spec type — an
        # asymmetry means a group's state is not being transferred.
        # Once per worker at startup, not a hot path.
        logger.info(
            "HIXLTRACE reg_summary n_regions=%d n_handles=%d "
            "per_block_dist=%s group_dist=%s phys=%d logical_blocks=%d",
            len(self._per_block_per_layer), len(self._kv_mem_handles),
            _pb_counts, _grp_counts,
            self._physical_blocks_per_logical_kv_block,
            self._logical_num_blocks,
        )
        logger.info(
            "HIXLEngineConnector registered %d KV regions on rank %s "
            "(num_blocks=%d, block_size=%d, layout=%s).",
            len(self._kv_mem_handles), self._tp_rank,
            self._num_blocks, self._block_size, self._kv_cache_layout,
        )

    def _warn_if_kv_zeroing_disabled(self) -> None:
        """Surface the platform's silently-skipped mamba block zeroing.

        vllm_ascend/worker/worker.py gates _init_kv_zero_meta() on
        speculative method == "eagle3"; upstream vLLM (gpu_worker.py) gates
        only on needs_kv_cache_zeroing, which is True for any model with
        mamba layers. When the gate misses, _kv_block_zeroer is never built
        and gpu_model_runner._zero_block_ids() is a hasattr-guarded no-op,
        so every new_block_ids_to_zero the scheduler produces is dropped
        without a word. The worker comment at that gate states the
        consequence: stale mamba state reused across requests degrades MTP
        acceptance. Re-derived here rather than probed, because the
        connector has no handle on the model runner — keep in sync with
        the worker.py _init_kv_zero_meta gate.
        """
        spec_cfg = getattr(self._vllm_config, "speculative_config", None)
        if spec_cfg is None or not self._has_mamba:
            return
        method = getattr(spec_cfg, "method", None)
        num_spec = getattr(spec_cfg, "num_speculative_tokens", 0)
        if method == "eagle3" or num_spec <= 1:
            return
        logger.warning(
            "HIXLEngine: mamba KV block zeroing is DISABLED. worker.py gates "
            "_init_kv_zero_meta on method=='eagle3', but this deployment has "
            "method=%s num_speculative_tokens=%d with mamba layers. Stale "
            "mamba state will be reused across requests and MTP acceptance "
            "will be degraded.", method, num_spec,
        )

    def _build_transfer_topology(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Build the local TransferTopology (mirrors NIXL base_worker).

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

        The payload is what the P-side ROUTER hands to D-side REQ over ZMQ.
        """
        from vllm import __version__ as vllm_version

        agent_meta = HixlEngineAgentMetadata(
            engine_id=self._engine_id,
            local_engine_endpoint=self._local_engine_endpoint,
            cluster_id=0,  # hixl Connect routes by endpoint string, not cluster
            listen_ip=get_ip(),
            listen_port=0,  # unused; peer uses local_engine_endpoint
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

        Ports NIXL _build_fa_remote for the remote side and
        _build_local_splits_from_plan for the per-source-rank local head
        offset. reshard is expressed purely as
        address offsets, no staging cache:

          stride       = per_block_per_layer[i] // block_size_ratio
          chunk        = stride // num_reads            (transfer length)
          rank_offset  = 0 if replicated else plan.rank_offset_factor * stride
          slot         = 0 if replicated
                       = plan.rank_to_attention_slot[source_rank]  (FA group)
                       = positional index in plan.all_source_ranks   (SSM)
          remote_addr  = remote_base_rank_i + rank_offset + remote_bid * page_size
          local_addr   = local_base_i + local_bid * stride + slot * chunk
          len          = chunk

        stride (full block, for addressing) and chunk (divided, for transfer
        length) are kept separate — NIXL _build_fa_local uses
        per_block//ratio as page_stride while _build_fa_remote
        uses the same value //num_reads as length.

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
        # span * physical_blocks_per_logical).
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
            # TODO(conv-decomp): replace with NIXL-style conv decomposition.
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
        # IndexError / wrong-region address. PP>1 remote-window slicing is
        # not ported (pull mode assumes a single pipeline stage).
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
            # Tensor page, not padded spec.page_size_bytes (E1 SSM, E9 FA).
            # 0958: spec gave FA 66816, tensor 65536; Mooncake uses
            # stride(0)*element_size. P/D same model, so local per_block
            # is valid for the remote side too (same as SSM).
            per_block = self._per_block_per_layer[i]
            stride = per_block // block_size_ratio
            page_size = per_block
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

        Port of NIXL pull_worker._read_blocks_for_req + _read_blocks.
        Iterates source ranks outermost so a full prefix hit across all
        groups for a rank sends a single release notify. Each non-empty
        rank's op_descs are submitted via TransferAsync(READ, endpoint);
        the handle is stashed under the local request_id for get_finished
        to poll. P-side release notify is sent from _pop_done_transfers
        once all of a req's handles resolve.

        transfer_async takes the peer endpoint string (the value passed to
        Connect), not the engine_id name. All wrapper calls that touch the
        hixl instance are serialized with _hixl_lock (hixl is not
        thread-safe; connect runs on the handshake thread, transfers here).
        """
        remote = req_meta.remote
        assert remote is not None
        remote_engine_id = remote.engine_id
        assert remote_engine_id in self._remote_metadata, (
            f"remote engine {remote_engine_id} not handshaken yet"
        )
        # Always store metadata for failure recovery.
        self._recving_metadata[request_id] = req_meta
        # Refresh last-seen so an engine with in-flight transfers is not
        # stale-evicted mid-read. _evict_stale_engines
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

        # Pre-trim per group (SSM: one committed slot; FA tail-trim when
        # same phys / front-trim to min when heterogeneous phys). Then
        # expand FA logical ids to kernel pages and clip the unwritten
        # hybrid-alignment tail (1 token must not pull 12 pages).
        trimmed_local: list[list[int]] = []
        trimmed_remote: list[list[int]] = []
        pre_remote_lens: list[int] = []
        # Diagnostic: pairs with the read_blocks timing line on req=.
        # SSM lists printed whole; FA adds k{expanded}->{clipped}.
        trim_probe: list[str] = []
        same_phys = (
            self._physical_blocks_per_logical_kv_block
            == remote_physical_per_logical
        )
        for g in range(num_groups):
            lb = list(req_meta.local_block_ids[g])
            rb = (
                list(req_meta.remote.block_ids[g])
                if g < len(req_meta.remote.block_ids)
                else []
            )
            pre_local, pre_remote = len(lb), len(rb)
            pre_remote_lens.append(pre_remote)
            is_ssm = (
                g < len(self._group_spec_types)
                and self._is_ssm_spec(self._group_spec_types[g])
            )
            lb, rb = self._apply_prefix_caching(
                lb, rb, g, remote_physical_per_logical
            )
            # Reconstruct the trim outcome from lengths so
            # _apply_prefix_caching stays a pure transform.
            if pre_local == 0:
                branch = "empty"
            elif is_ssm:
                branch = "ssm_align"
            elif pre_local == pre_remote:
                branch = "passthrough"
            elif same_phys:
                branch = "fa_tail"
            else:
                branch = "fa_front"
            probe = (
                f"g{g}{'/ssm' if is_ssm else ''}="
                f"{pre_local}->{len(lb)}/{pre_remote}->{len(rb)}:{branch}"
            )
            if lb or rb:
                # Logical ids after prefix trim, before kernel expand.
                # 21k / g0=28 needs FA ids, not just lengths.
                probe += f" L{lb} R{rb}"
            trim_probe.append(probe)
            trimmed_local.append(lb)
            trimmed_remote.append(rb)

        local_phys = self._physical_blocks_per_logical_kv_block
        num_computed = int(getattr(req_meta, "num_computed_tokens", 0) or 0)
        num_external = int(getattr(req_meta, "num_external_tokens", 0) or 0)
        for g in range(num_groups):
            lb, rb = trimmed_local[g], trimmed_remote[g]
            is_ssm = (
                g < len(self._group_spec_types)
                and self._is_ssm_spec(self._group_spec_types[g])
            )
            if is_ssm or not lb:
                continue
            # Expand once (not per rank). _build_op_descs addresses by
            # physical id; SSM stays logical (phys==1 today).
            lb = self._expand_physical(lb, local_phys)
            rb = self._expand_physical(rb, remote_physical_per_logical)
            assert len(lb) == len(rb), (
                f"group {g}: physical id count mismatch after expand: "
                f"local={len(lb)} remote={len(rb)} (local_phys="
                f"{local_phys} remote_phys="
                f"{remote_physical_per_logical}); heterogeneous "
                "block_size not supported by the zip-pair path."
            )
            dropped = (
                (pre_remote_lens[g] - len(trimmed_remote[g]))
                * remote_physical_per_logical
            )
            pre_k = len(lb)
            lb, rb = self._clip_attn_kernel_pages(
                lb, rb,
                num_computed_tokens=num_computed,
                num_external_tokens=num_external,
                dropped_kernels=dropped,
            )
            trim_probe[g] += f" k{pre_k}->{len(lb)}"
            trimmed_local[g] = lb
            trimmed_remote[g] = rb
        req_meta.local_kernel_ids = [list(x) for x in trimmed_local]
        logger.info(
            "HIXLEngine trim probe. req=%s ext=%d computed=%d %s",
            request_id, num_external, num_computed, " ".join(trim_probe),
        )

        # Split model-thread cost: desc build vs TransferAsync submit.
        # n_descs travels with them so a coalescing regression shows here
        # before it shows as bandwidth.
        t_desc = 0.0
        t_submit = 0.0
        n_descs_total = 0
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
                _t0 = time.perf_counter()
                _g_descs, _g_bytes = self._build_op_descs(
                    list(lb), list(rb), plan, remote_engine_id, g, rank,
                )
                t_desc += time.perf_counter() - _t0
                group_descs.extend(_g_descs)
                rank_bytes += _g_bytes
            # Full prefix hit across all groups for this rank: no transfer,
            # just notify P to release.
            if not group_descs:
                self._enqueue_notify(
                    remote.host,
                    remote.port,
                    "DONE",
                    notif_id,
                    rank,
                )
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
            n_descs_total += len(group_descs)
            try:
                _t0 = time.perf_counter()
                with self._hixl_lock:
                    handle = self._hixl_transfer_async(
                        endpoint, "READ", group_descs
                    )
                t_submit += time.perf_counter() - _t0
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
                self._log_read_blocks_timing(
                    request_id, plan, n_descs_total, t_desc, t_submit,
                )
                return
        self._log_read_blocks_timing(
            request_id, plan, n_descs_total, t_desc, t_submit,
        )

    def _log_read_blocks_timing(
        self,
        request_id: str,
        plan: TPMapping,
        n_descs: int,
        t_desc: float,
        t_submit: float,
    ) -> None:
        """Report the model-thread cost of one request's READ submission.

        INFO rather than DEBUG on purpose: DEBUG also turns on the per-region
        HIXLTRACE logs inside the same loops, which would dominate the very
        numbers this is measuring. One line per request, same volume as the
        bandwidth log.
        """
        logger.info(
            "HIXLEngine read_blocks timing. req=%s n_ranks=%d n_descs=%d "
            "build=%.2fms submit=%.2fms",
            request_id, len(plan.all_source_ranks), n_descs,
            t_desc * 1000.0, t_submit * 1000.0,
        )

    # ==================================================================
    # Worker-side: load/save lifecycle (poll handles, no reformat)
    # ==================================================================
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:  # noqa: F821
        """Trigger async READ for requests scheduled this step.

        Issues TransferAsync — no RecvingThread; handles are polled in
        get_finished. A request whose remote engine has not been
        handshaken yet is parked on _pending_handshake_reqs and read on a
        later step once the handshake done_callback releases it into
        _ready_requests. Without that re-queue the req would be lost —
        the scheduler clears _reqs_need_recv / flips do_remote_prefill
        the same step.
        """
        # metadata comes from bind_connector_metadata, which the model
        # runner calls with scheduler_output.kv_connector_metadata just
        # before start_load_kv.
        metadata = self._connector_metadata
        if metadata is None:
            return
        for req_id in metadata.reqs_in_batch:
            self._task_tracker.add_req_to_process(req_id)
        # Track lease expiry for reqs awaiting remote read (P-side delayed
        # free). _reqs_to_send is the expiry table; DONE notifies and
        # lease-expiry are the only removers.
        #
        # Must precede the inbound drain: a DONE in the same metadata
        # packet as its reqs_to_send entry would otherwise look premature.
        # add_req_to_process stays ahead of both because _apply_one_notify
        # only parks a DONE whose req is still in-batch; an unknown id is
        # discarded.
        for req_id, expiration_time in metadata.reqs_to_send.items():
            self._reqs_to_send[req_id] = expiration_time
        self._replay_pending_dones()
        # P workers: apply DONE/HB that the scheduler ROUTER received last
        # step. D workers see an empty list.
        self._drain_inbound_notifies(metadata)
        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = meta.remote.engine_id
            # Always store metadata so the ready-queue drain / failure
            # paths recover req state after a deferred handshake.
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

        # Drain reqs whose handshakes finished (this step or a prior one).
        # Also drained from get_finished so a handshake that lands during
        # this step's forward issues its READ at step end instead of
        # waiting for the next start_load_kv.
        self._drain_ready_requests()

        # Drop aborted reqs from the in-process set.
        for req_id in metadata.reqs_not_processed:
            self._task_tracker.discard_from_process(req_id)
        # D-side: extend P-side leases for reqs still WAITING in scheduler.
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
        """Trim to the locally-uncached tail, per group.

        Branch per group:

        - SSM state: align / MTP block table is 1 committed + N speculative
          slots. Only the committed slot is live (Mooncake /
          hixl_connector). Pick local[0] and remote[len - spec - 1]
          (clamp idx < 0 to 0). Zip-pairing all slots writes P's draft
          slots into D and kills MTP acceptance.
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
        is_ssm = (
            group_idx < len(self._group_spec_types)
            and self._is_ssm_spec(self._group_spec_types[group_idx])
        )
        if is_ssm:
            if num_local == 0 or num_remote == 0:
                return local_block_ids, []
            spec = int(getattr(self, "_num_speculative_tokens", 0) or 0)
            idx = num_remote - spec - 1
            if idx < 0:
                idx = 0
            return [local_block_ids[0]], [remote_block_ids[idx]]
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

    def _clip_attn_kernel_pages(
        self,
        kernel_local: list[int],
        kernel_remote: list[int],
        *,
        num_computed_tokens: int,
        num_external_tokens: int,
        dropped_kernels: int = 0,
    ) -> tuple[list[int], list[int]]:
        """Drop unwritten hybrid-alignment tail pages after expand.

        Scheduler attention blocks are sized to share a page with mamba
        (1536 tokens here); the kernel page is 128 tokens. Expanding a
        partial block yields empty tail pages (1 token → 11 unused / 12).
        Mooncake `_get_kernel_block_ids` keeps ``cdiv(t, kernel)`` pages.
        ``dropped_kernels`` is how many prefix kernel pages
        ``_apply_prefix_caching`` already removed, so the start index
        stays aligned with Mooncake's full-list expand. ``num_external_tokens
        == 0`` leaves the lists untouched (no pull in that case).
        """
        if num_external_tokens <= 0 or not kernel_local or not kernel_remote:
            return kernel_local, kernel_remote
        cr = int(getattr(self, "_attn_compress_ratio", 1) or 1)
        kernel_token_size = max(1, int(self._block_size)) * max(1, cr)
        remote_start_idx = num_computed_tokens // kernel_token_size
        tokens_to_cover = num_computed_tokens + num_external_tokens
        needed = cdiv(tokens_to_cover, kernel_token_size) - remote_start_idx
        if needed <= 0:
            return [], []
        skip = max(0, remote_start_idx - max(0, dropped_kernels))
        end = skip + needed
        return kernel_local[skip:end], kernel_remote[skip:end]

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

    def _xfer_wait_timed_out(self, req_id: str, now: float) -> bool:
        started = self._xfer_start.get(req_id)
        if started is None:
            return False
        return (now - started) >= self._xfer_wait_timeout_s

    def _poll_orphan_xfer_handles(self) -> None:
        """Drain abandoned handles so a wait-timeout does not leak RDMA reqs."""
        if not self._orphan_xfer_handles:
            return
        still: list[int] = []
        for handle in self._orphan_xfer_handles:
            try:
                st = self._hixl_get_transfer_status(handle)
                if st is None:
                    continue
                if self._transfer_status_name(st) == "WAITING":
                    still.append(handle)
            except Exception:  # noqa: BLE001
                continue
        self._orphan_xfer_handles = still

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

        Port of NIXL _pop_done_transfers.
        COMPLETED drops the handle; WAITING stays in_progress; FAILED/
        TIMEOUT marks the req invalid via _handle_failed_transfer. A req is
        done only when every handle resolved. On clean (failure-free)
        completion, _notify_release tells the P side to release blocks
        (replaces NIXL's transfer notif_msg auto-delivery).
        """
        done_req_ids: set[str] = set()
        self._poll_orphan_xfer_handles()
        now = time.perf_counter()
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
            if in_progress and self._xfer_wait_timed_out(req_id, now):
                started = self._xfer_start.get(req_id)
                elapsed_s = (now - started) if started is not None else -1.0
                logger.error(
                    "HIXLEngine transfer wait timeout. req=%s "
                    "elapsed=%.1fs limit=%.1fs handles=%d",
                    req_id, elapsed_s, self._xfer_wait_timeout_s,
                    len(in_progress),
                )
                had_failure = True
                self._handle_failed_transfer(req_id, None)
                self._orphan_xfer_handles.extend(in_progress)
                in_progress = []
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

    # -- NZ reformat fallback -------------------------------------------
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
        kernel_groups = getattr(meta, "local_kernel_ids", None)
        for g, block_ids in enumerate(meta.local_block_ids):
            if g >= len(self._group_spec_types):
                continue
            if self._is_ssm_spec(self._group_spec_types[g]):
                continue  # state group: no NZ reformat
            # Prefer the kernel ids _read_blocks already clipped; otherwise
            # expand the scheduler logical ids (tests / NZ-before-read).
            if kernel_groups is not None and g < len(kernel_groups):
                phys_block_ids = list(kernel_groups[g])
            else:
                if not block_ids:
                    continue
                phys_block_ids = block_ids
                if local_phys > 1:
                    phys_block_ids = self._expand_physical(
                        list(block_ids), local_phys
                    )
            if not phys_block_ids:
                continue
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
    # Async notify sender (ZMQ REQ → P scheduler ROUTER).
    #
    # DONE/HB must not share the HIXL CommEngine control TCP with
    # Transfer BufferReq — that mix desynced the notify stream (P
    # parse_error / D SendNotify 103901). The sender thread keeps the
    # forward path off the side-channel RTT.
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

    def _notify_endpoint_blocked(self, endpoint: str) -> bool:
        return time.perf_counter() < self._notify_dead_until.get(endpoint, 0.0)

    def _trip_notify_circuit(self, endpoint: str, err: object) -> None:
        self._notify_dead_until[endpoint] = (
            time.perf_counter() + self._notify_dead_cooldown_s
        )
        logger.warning(
            "HIXLEngine notify circuit-open. endpoint=%s cooldown=%.1fs err=%s",
            endpoint, self._notify_dead_cooldown_s, err,
        )

    def _notify_sender_loop(self) -> None:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        sockets: dict[str, Any] = {}
        encoder = msgspec.msgpack.Encoder()
        try:
            while True:
                item = self._notify_queue.get()
                if item is None:  # shutdown sentinel
                    return
                host, port, name, msg, target_tp = item
                dest = f"{host}:{port}"
                if self._notify_endpoint_blocked(dest):
                    logger.debug(
                        "HIXLEngine notify skipped (circuit-open). name=%s "
                        "dest=%s",
                        name, dest,
                    )
                    continue
                try:
                    self._zmq_send_notify(
                        ctx, sockets, encoder, host, port, name, msg, target_tp)
                except Exception as e:  # noqa: BLE001
                    # DONE loss falls back to the P-side lease; HB is re-sent
                    # next round. A failed REQ must be dropped — the socket
                    # is stuck until a reply arrives.
                    logger.warning(
                        "HIXLEngine notify send failed. name=%s dest=%s err=%s",
                        name, dest, e,
                    )
                    self._close_notify_socket(sockets, dest)
                    self._trip_notify_circuit(dest, e)
        finally:
            for dest in list(sockets):
                self._close_notify_socket(sockets, dest)
            ctx.destroy(linger=0)  # type: ignore[attr-defined]

    def _zmq_send_notify(
        self,
        ctx: Any,
        sockets: dict[str, Any],
        encoder: msgspec.msgpack.Encoder,
        host: str,
        port: int,
        name: str,
        msg: str,
        target_tp: int,
    ) -> None:
        dest = f"{host}:{port}"
        sock = sockets.get(dest)
        if sock is None:
            path = make_zmq_path("tcp", host, port)
            sock = make_zmq_socket(
                ctx=ctx, path=path, socket_type=zmq.REQ, bind=False,
            )
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, self._notify_timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self._notify_timeout_ms)
            sockets[dest] = sock
        payload = encoder.encode((NOTIFY_MSG, name, msg, target_tp))
        sock.send_multipart((payload,))
        reply = sock.recv_multipart()
        if not reply or reply[0] != _NOTIFY_ACK:
            raise RuntimeError(
                f"unexpected notify ACK from {dest}: {reply!r}"
            )

    @staticmethod
    def _close_notify_socket(sockets: dict[str, Any], dest: str) -> None:
        sock = sockets.pop(dest, None)
        if sock is None:
            return
        try:
            sock.close(linger=0)
        except Exception:  # noqa: BLE001
            pass

    def _enqueue_notify(
        self, host: str, port: int, name: str, msg: str, target_tp: int,
    ) -> None:
        """Hand a notify to the sender thread; never blocks the caller."""
        dest = f"{host}:{port}"
        if self._notify_endpoint_blocked(dest):
            if name == "HB":
                logger.debug(
                    "HIXLEngine notify circuit-open, dropping HB to %s",
                    dest,
                )
            else:
                logger.warning(
                    "HIXLEngine notify circuit-open, dropping %s to %s; "
                    "P-side lease expiry is the backstop.",
                    name, dest,
                )
            return
        self._ensure_notify_sender()
        try:
            self._notify_queue.put_nowait((host, port, name, msg, target_tp))
        except queue.Full:
            if name == "HB":
                logger.debug(
                    "HIXLEngine notify queue full, dropping HB to %s", dest)
            else:
                logger.warning(
                    "HIXLEngine notify queue full (%d), dropping %s to %s; "
                    "P-side lease expiry is the backstop.",
                    self._notify_queue_size, name, dest,
                )

    def _notify_release(self, req_id: str) -> None:
        """Tell every P-side source rank this req actually read from that its
        reads are done.

        DONE goes to the P scheduler ROUTER with target_tp=rank so only
        the P worker that was read decrements its consumer count.

        Only notify ranks this request issued an async READ to
        (``_transferred_ranks``), never broadcast to ``plan.all_source_ranks``.
        Ranks covered by a prefix-hit DONE (no transfer) were already notified
        in _read_blocks and are skipped here to avoid a double-notify.
        """
        meta = self._recving_metadata.get(req_id)
        if meta is None or meta.remote is None:
            return
        notif_id = f"{meta.remote.request_id}:{self._world_size}"
        already_notified = self._notified_release_ranks.pop(req_id, set())
        to_notify = self._transferred_ranks.pop(req_id, set()) - already_notified
        for rank in to_notify:
            self._enqueue_notify(
                meta.remote.host, meta.remote.port, "DONE", notif_id, rank,
            )

    def _send_heartbeats(self, metadata: "HIXLEngineConnectorMetadata") -> None:
        """D-side: extend P-side leases for reqs still WAITING in scheduler.

        One HB to the P scheduler ROUTER (target_tp=-1); every P rank
        extends its own lease table. Handshake is still kicked if this
        engine is unseen, but HB no longer waits on HIXL Connect.
        """
        for engine_id, hb_info in metadata.heartbeat_by_engine.items():
            if engine_id not in self._remote_agents:
                self._ensure_handshake(
                    engine_id,
                    hb_info.host,
                    hb_info.port,
                    hb_info.tp_size,
                )
            req_ids = [rid for rid in hb_info.req_ids if rid]
            if not req_ids or not hb_info.host or not hb_info.port:
                continue
            self._enqueue_notify(
                hb_info.host,
                hb_info.port,
                "HB",
                ",".join(req_ids),
                _NOTIFY_ALL_RANKS,
            )

    def _handle_heartbeat(self, payload: str) -> None:
        """P-side: extend leases for reqs referenced in a heartbeat.

        payload is a comma-separated
        list of P-side request ids. Each referenced req's expiry is pushed to
        max(old, now + lease_extension) so a late heartbeat never shortens the
        lease.
        """
        new_expiry = time.perf_counter() + self._lease_extension
        for req_id in payload.split(","):
            if req_id in self._reqs_to_send:
                old = self._reqs_to_send[req_id]
                self._reqs_to_send[req_id] = max(old, new_expiry)

    def _drain_inbound_notifies(
        self, metadata: "HIXLEngineConnectorMetadata"
    ) -> set[str]:
        """Apply ZMQ DONE/HB copied into this step's metadata.

        start_load_kv is the only caller; it applies metadata.reqs_to_send
        first so a DONE riding in the same packet as its lease entry is not
        seen as premature. Clears the list once applied.
        """
        inbound = getattr(metadata, "inbound_notifies", None)
        if not inbound:
            return self._replay_pending_dones()
        notified = self._apply_inbound_notifies(inbound)
        inbound.clear()
        # A DONE parked earlier in this very drain may already be replayable
        # (its lease entry was written before the drain started).
        return notified | self._replay_pending_dones()

    def _apply_inbound_notifies(
        self, inbound: list[tuple[str, str, int]]
    ) -> set[str]:
        notified_req_ids: set[str] = set()
        for name, msg, target_tp in inbound:
            if target_tp >= 0 and target_tp != self._tp_rank:
                continue
            notified_req_ids |= self._apply_one_notify(name, msg)
        return notified_req_ids

    def _apply_one_notify(self, name: str, msg: str) -> set[str]:
        """Apply one DONE/HB. Returns req ids released this call."""
        notified_req_ids: set[str] = set()
        try:
            if name == "HB":
                self._handle_heartbeat(msg)
                return notified_req_ids
            if name != "DONE":
                return notified_req_ids
            req_id, tp_size = msg.rsplit(":", 1)
            if req_id not in self._reqs_to_send:
                if req_id in self._task_tracker:
                    # Still in batch, so request_finished has not published
                    # its lease entry yet. start_load_kv applies
                    # metadata.reqs_to_send before draining inbound, so
                    # same-packet ordering is already covered; a DONE can
                    # still land a step ahead of its own request_finished.
                    # Park it — dropping costs the P blocks a full lease.
                    self._park_pending_done(req_id, msg)
                else:
                    # Never owned here, or already released by an earlier
                    # DONE / the expiry sweep. Nothing to wait for.
                    logger.warning(
                        "HIXLEngine DONE notify for unknown/expired request "
                        "%s; ignoring.", req_id,
                    )
                return notified_req_ids
            if self._count_done_notify(req_id, tp_size):
                notified_req_ids.add(req_id)
        except Exception:
            logger.error(
                "HIXLEngine notify handling failed for %s:%s",
                name, msg, exc_info=True,
            )
        return notified_req_ids

    def _count_done_notify(self, req_id: str, tp_size: str) -> bool:
        """Count one DONE for a req known to _reqs_to_send.

        Returns True when the req reached consumers_per_producer and was
        promoted. Split out of _apply_one_notify so the replay path in
        _replay_pending_dones counts through the same tp_ratio logic instead
        of duplicating it.
        """
        n_consumers = int(tp_size)  # D-side world_size (= D_TP)
        # tp_ratio asserts TP
        # divisibility and yields the correct per-producer consumer
        # count for split (D_TP>P_TP => -tp_ratio) and 1 otherwise.
        assert self._transfer_topo is not None
        tp_ratio = self._transfer_topo.tp_ratio(n_consumers)
        consumers_per_producer = (
            -tp_ratio if n_consumers > self._world_size else 1
        )
        self._consumer_notification_counts_by_req[req_id] += 1
        if (self._consumer_notification_counts_by_req[req_id]
                < consumers_per_producer):
            return False
        del self._consumer_notification_counts_by_req[req_id]
        self._task_tracker.update_done_task_count(req_id)
        self._reqs_to_send.pop(req_id, None)
        return True

    def _park_pending_done(self, req_id: str, msg: str) -> None:
        """Hold a DONE whose req is not in _reqs_to_send yet, for replay."""
        prev = self._pending_dones.get(req_id)
        if prev is not None:
            self._pending_dones[req_id] = (msg, prev[1] + 1, prev[2])
            return
        if len(self._pending_dones) >= self._pending_done_max:
            # Bound the table even if reqs somehow never reach the lease
            # table; the oldest entry is the one closest to its TTL anyway.
            oldest = min(
                self._pending_dones, key=lambda r: self._pending_dones[r][2]
            )
            del self._pending_dones[oldest]
            logger.warning(
                "HIXLEngine pending DONE table full (%d); dropped oldest "
                "entry %s to park %s.",
                self._pending_done_max, oldest, req_id,
            )
        self._pending_dones[req_id] = (msg, 1, time.perf_counter())
        logger.debug(
            "HIXLEngine DONE for in-process req %s arrived before its lease "
            "entry; parked for replay.", req_id,
        )

    def _replay_pending_dones(self) -> set[str]:
        """Apply parked DONEs whose reqs have since entered _reqs_to_send.

        Called right after every write to _reqs_to_send and at the end of
        _drain_inbound_notifies. Entries older than the lease are discarded —
        past that point the expiry sweep has already freed the blocks.
        """
        notified_req_ids: set[str] = set()
        if not self._pending_dones:
            return notified_req_ids
        now = time.perf_counter()
        for req_id in list(self._pending_dones):
            msg, count, first_seen = self._pending_dones[req_id]
            if req_id in self._reqs_to_send:
                del self._pending_dones[req_id]
                try:
                    _, tp_size = msg.rsplit(":", 1)
                    for _ in range(count):
                        if req_id not in self._reqs_to_send:
                            # Promoted on an earlier iteration; the remaining
                            # parked DONEs are duplicates (a recompute re-read
                            # sends its own). Counting them again would warn
                            # from update_done_task_count.
                            break
                        if self._count_done_notify(req_id, tp_size):
                            notified_req_ids.add(req_id)
                except Exception:
                    logger.error(
                        "HIXLEngine pending DONE replay failed for %s",
                        msg, exc_info=True,
                    )
                continue
            if now - first_seen >= self._pending_done_ttl_s:
                del self._pending_dones[req_id]
                logger.warning(
                    "HIXLEngine dropping parked DONE for %s after %.1fs; it "
                    "never entered the lease table.", req_id, now - first_seen,
                )
        return notified_req_ids

    def _get_new_notifs(self) -> set[str]:
        """P-side: drain leftover HIXL GetNotifies (should be empty).

        DONE/HB now arrive via the ZMQ side channel and are applied from
        metadata.inbound_notifies, so get_finished only calls this when
        hixl_engine.drain_hixl_notifies is set — every call takes the global
        hixl_py mutex, which the engine step cannot afford to pay for an
        always-empty result. Kept for a peer old enough to still SendNotify.
        """
        notified_req_ids: set[str] = set()
        try:
            notifs = self._hixl_get_notifies()
        except Exception:
            logger.error("HIXLEngine get_notifies failed", exc_info=True)
            return notified_req_ids
        for name, msg in notifs:
            notified_req_ids |= self._apply_one_notify(name, msg)
        return notified_req_ids

    def _handle_failed_transfer(self, req_id: str, handle: int | None) -> None:
        """Mark a failed transfer's request and its attention blocks invalid.

        Port of NIXL _handle_failed_transfer.
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
        """No-op unless ``blocking_layer_wait`` is set.

        TransferAsync is per source rank, not per layer, so a wait here
        would stall the whole batch on other requests' pulls. The
        scheduler already parks remote-prefill reqs in
        WAITING_FOR_REMOTE_KVS until finished_recving; get_finished
        polls handles. Matches NIXL / Mooncake / HIXLConnector.

        The opt-in blocking path waits for in-flight transfers.
        GetTransferStatus drops a handle on any terminal status, so that
        path must record FAILED/TIMEOUT here — a later poll only sees
        "not found".
        """
        if not self._blocking_layer_wait:
            return
        if not self._recving_transfers:
            return
        deadline = time.perf_counter() + self._transfer_timeout_ms / 1000.0
        backoff = 0.001
        while time.perf_counter() < deadline:
            any_waiting = False
            # No _hixl_lock: get_transfer_status is read-only and hixl_py
            # serializes every call on one C++ mutex, so the Python lock
            # adds neither safety nor concurrency.
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
        even though their blocks are already marked invalid.
        """
        # Inbound DONE/HB are applied only in start_load_kv so the lease
        # table is written before DONEs are applied. GetNotifies is off
        # unless drain_hixl_notifies (DONE/HB already moved to ZMQ).
        if self._drain_hixl_notifies:
            self._get_new_notifs()
        done_recving = self._pop_done_transfers(self._recving_transfers)
        # Drop metadata for completed reqs. Without
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
        # consumer reported DONE. Full scan — _reqs_to_send is
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

        Drains and clears so the same batch is not reported twice.
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

        Encode each worker's HixlEngineHandshakePayload under its (pp, tp)
        key and start one scheduler-side ROUTER that serves them all.
        The signature stays relaxed to
        ``Mapping[int | tuple[int, ...], Any]`` (keys folded via
        key[0], key[1]) because vllm-ascend worker emits 3-tuple
        ``(pp, pcp, tp)`` keys when pcp_size > 1.
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
        # Stop the P-side handshake ROUTER. Both roles own the stop event
        # and listener thread, so this is safe for both.
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
        # Drain the worker notify thread before finalizing the engine.
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
