# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
"""Thin Python wrapper around the ``hixl_wrapper`` pybind11 module.

Mirrors the role of NIXL's ``NixlWrapper`` (vllm/distributed/nixl_utils.py):
a lazily-loaded handle over the C++ address-level transfer API. No KV/block
semantics live here — those belong to the connector. This module only:
  * loads ``hixl_wrapper.<abi>.so`` built from src/python/llm_wrapper/hixl_wrapper.cc
  * turns the Status return codes into ``HixlError`` exceptions (already done
    in the binding, re-raised here for convenience)
  * exposes TransferOpDesc as a small dataclass-shaped helper so the connector
    can build op_desc lists without touching the C++ types directly.

The wrapper is imported lazily so that vllm-ascend does not require the hixl
build artefact unless HIXLEngineConnector is actually selected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_HIXL_WRAPPER_MOD = None


def _load_hixl_wrapper():
    """Lazily import the compiled pybind11 module."""
    global _HIXL_WRAPPER_MOD
    if _HIXL_WRAPPER_MOD is None:
        try:
            import hixl_wrapper  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "hixl_wrapper pybind11 module not found. Build it from "
                "hixl/src/python/llm_wrapper/hixl_wrapper.cc and install "
                "alongside llm_datadist."
            ) from e
        _HIXL_WRAPPER_MOD = hixl_wrapper
    return _HIXL_WRAPPER_MOD


class HixlError(RuntimeError):
    """Raised when a hixl::Hixl call returns a non-SUCCESS status."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


@dataclass
class TransferOpDesc:
    """Address-level transfer descriptor: copy ``len`` bytes from
    ``remote_addr`` to ``local_addr`` (READ) or vice versa (WRITE).

    Mirrors hixl::TransferOpDesc (hixl_types.h:69-73). head-shard reshard is
    expressed purely via these addresses — no staging cache needed.
    """
    local_addr: int
    remote_addr: int
    len: int

    def to_native(self, mod):
        return mod.TransferOpDesc(self.local_addr, self.remote_addr, self.len)


class HixlEngineWrapper:
    """Holds one hixl::Hixl instance and forwards the address-level API.

    The connector creates one wrapper per rank. Memory registered here becomes
    remotely accessible to the peer after Connect/link; TransferAsync issues
    RDMA READ/WRITE batches and returns an opaque handle to poll.
    """

    def __init__(self):
        self._mod = _load_hixl_wrapper()
        self._hixl = self._mod.Hixl()
        self._initialized = False

    # -- lifecycle ---------------------------------------------------------
    def initialize(self, local_engine: str, options: dict[str, str]) -> None:
        self._hixl.initialize(local_engine, options)
        self._initialized = True

    def finalize(self) -> None:
        if self._initialized:
            self._hixl.finalize()
            self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # -- memory registration ----------------------------------------------
    def register_mem(self, addr: int, length: int, is_device: bool = True) -> int:
        """Register a memory region. Returns an opaque mem handle (uintptr_t)."""
        mem_type = self._mod.MemType.MEM_DEVICE if is_device else self._mod.MemType.MEM_HOST
        return self._hixl.register_mem(addr, length, mem_type)

    def deregister_mem(self, handle: int) -> None:
        self._hixl.deregister_mem(handle)

    # -- connection --------------------------------------------------------
    def connect(self, remote_engine: str, timeout_ms: int = 1000) -> None:
        self._hixl.connect(remote_engine, timeout_ms)

    def disconnect(self, remote_engine: str, timeout_ms: int = 1000) -> None:
        self._hixl.disconnect(remote_engine, timeout_ms)

    def connect_async(self, remote_engine: str, timeout_ms: int = 1000) -> None:
        self._hixl.connect_async(remote_engine, timeout_ms)

    def disconnect_async(self, remote_engine: str, timeout_ms: int = 1000) -> None:
        self._hixl.disconnect_async(remote_engine, timeout_ms)

    def get_async_connect_status(self, remote_engine: str):
        return self._hixl.get_async_connect_status(remote_engine)

    def get_async_connect_status_all(self) -> dict[str, Any]:
        return self._hixl.get_async_connect_status_all()

    # -- transfer ----------------------------------------------------------
    def transfer_sync(self, remote_engine: str, op: str,
                      op_descs: list[TransferOpDesc],
                      timeout_ms: int = 1000) -> None:
        native_op = self._op(op)
        native_descs = [d.to_native(self._mod) for d in op_descs]
        self._hixl.transfer_sync(remote_engine, native_op, native_descs, timeout_ms)

    def transfer_async(self, remote_engine: str, op: str,
                       op_descs: list[TransferOpDesc]) -> int:
        """Issue an async RDMA batch. Returns an opaque request handle to poll
        with get_transfer_status()."""
        native_op = self._op(op)
        native_descs = [d.to_native(self._mod) for d in op_descs]
        return self._hixl.transfer_async(remote_engine, native_op, native_descs)

    def get_transfer_status(self, req: int):
        return self._hixl.get_transfer_status(req)

    def get_transfer_status_batch(self, max_query_count: int | None = None,
                                  skip_waiting: bool = False):
        args = self._mod.GetTransferStatusArgs()
        if max_query_count is not None:
            args.max_query_count = max_query_count
        args.skip_waiting = skip_waiting
        return self._hixl.get_transfer_status_batch(args)

    # -- notify (replaces ZMQ DONE side-channel, optional) -----------------
    def send_notify(self, remote_engine: str, name: str, msg: str,
                    timeout_ms: int = 1000) -> None:
        self._hixl.send_notify(remote_engine, name, msg, timeout_ms)

    def get_notifies(self) -> list[tuple[str, str]]:
        return self._hixl.get_notifies()

    # -- capability --------------------------------------------------------
    def get_capability(self, feature_type: str) -> int:
        ft = getattr(self._mod.FeatureType, feature_type)
        return self._mod.get_capability(ft)

    # -- helpers -----------------------------------------------------------
    def _op(self, op: str):
        if op not in ("READ", "WRITE"):
            raise ValueError(f"Unsupported TransferOp: {op!r}")
        return getattr(self._mod.TransferOp, op)
