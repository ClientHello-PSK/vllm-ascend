# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""TP mapping computation for HIXLEngineConnector.

Address-level port of NIXL's tp_mapping (vllm/vllm/distributed/kv_transfer/
kv_connector/v1/nixl/tp_mapping.py). The head split/gather for heterogeneous
TP is expressed purely as remote_addr offsets here (see hixl_engine_connector
_build_op_descs), so no staging cache is needed — the transfer layer carries
the reshard.

Three scenarios, identical to NIXL:
  * D_TP >= P_TP (or MLA): each local rank reads its head slice from one
    remote rank at a per-rank head offset (rank_offset_factor).
  * P_TP > D_TP: one local rank gathers from multiple remote ranks; GQA dedup
    via np.unique avoids redundant reads of the same KV head.
  * MLA: cache replicated across TP, rank_offset_factor=0, single read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    TransferTopology,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, MambaSpec


@dataclass(frozen=True)
class ReadSpec:
    """Specification for a single remote block read operation.

    One of these is produced per source remote TP rank; the connector turns
    each into a batch of TransferOpDesc (local_addr, remote_addr, len).
    """
    remote_rank: int
    local_block_ids: BlockIds
    remote_block_ids: BlockIds


def _is_attention_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, AttentionSpec)


def _is_ssm_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, MambaSpec)


@dataclass(frozen=True)
class TPMapping:
    """Complete local-to-remote TP mapping for one remote engine.

    Generated once per remote engine during handshake.
    """
    # Remote TP ranks that this local rank reads from, per group.
    source_ranks_per_group: tuple[tuple[int, ...], ...]
    # Superset of all source ranks (union of all groups).
    all_source_ranks: tuple[int, ...]
    # Maps each source rank to its FA head slot index.
    rank_to_attention_slot: dict[int, int]
    # FA head offset factor for hetero-TP (D_TP > P_TP); multiplied by
    # remote_kv_block_len to form remote_addr offset in _build_op_descs.
    rank_offset_factor: int


def compute_tp_mapping(
    transfer_topology: TransferTopology,
    remote_tp_size: int,
    group_spec_types: tuple[type[KVCacheSpec], ...],
) -> TPMapping:
    """Build the complete local-to-remote TP mapping.

    Computes source ranks, head slot assignments, and the rank offset
    factor in a single pass.
    """
    tp_rank = transfer_topology.tp_rank
    tp_size = transfer_topology.tp_size
    total_num_kv_heads = transfer_topology.total_num_kv_heads

    # --- Attention source ranks ---
    if transfer_topology.is_mla or tp_size >= remote_tp_size:
        # D (local TP) > P (remote TP): multiple local ranks read different
        # chunks from *one* remote rank, corresponding to different kv heads.
        # For MLA, cache is duplicated; one remote read suffices.
        attn_ranks = [tp_rank * remote_tp_size // tp_size]
    else:
        # P (remote TP) > D (local TP): one local rank reads from multiple
        # remote ranks. GQA dedup: when K < remote_tp_size, several remote
        # ranks hold the same KV head — np.unique keeps only the first rank
        # per unique head so we don't issue redundant reads.
        abs_tp = remote_tp_size // tp_size
        start = tp_rank * abs_tp
        heads = np.arange(start, start + abs_tp) * total_num_kv_heads // remote_tp_size
        _, unique_idx = np.unique(heads, return_index=True)
        attn_ranks = (start + np.sort(unique_idx)).tolist()

    # --- SSM source ranks ---
    has_ssm = any(_is_ssm_spec(t) for t in group_spec_types)
    if has_ssm:
        if tp_size < remote_tp_size:
            abs_tp = remote_tp_size // tp_size
            ssm_ranks = list(range(tp_rank * abs_tp, (tp_rank + 1) * abs_tp))
        else:
            ssm_ranks = list(attn_ranks)
    else:
        ssm_ranks = []

    all_ranks = sorted(set(attn_ranks) | set(ssm_ranks))

    # --- Per-group ordered source ranks ---
    source_ranks_per_group = tuple(
        tuple(ssm_ranks) if _is_ssm_spec(t) else tuple(attn_ranks)
        for t in group_spec_types
    )

    # --- Attention head slots ---
    head_to_slot: dict[int, int] = {}
    for i, r in enumerate(attn_ranks):
        head_to_slot[r * total_num_kv_heads // remote_tp_size] = i
    rank_to_attention_slot = {
        r: head_to_slot.get(r * total_num_kv_heads // remote_tp_size, 0)
        for r in all_ranks
    }

    # --- Rank offset factor ---
    if transfer_topology.is_mla or tp_size <= remote_tp_size:
        # We don't index into remote for reading, no offset needed.
        rank_offset_factor = 0
    elif tp_size > total_num_kv_heads:
        local_head = tp_rank * total_num_kv_heads // tp_size
        p_start = attn_ranks[0] * total_num_kv_heads // remote_tp_size
        rank_offset_factor = local_head - p_start
    else:
        # D TP > P TP: index into remote to read different heads depending on rank.
        rank_offset_factor = tp_rank % (tp_size // remote_tp_size)

    return TPMapping(
        source_ranks_per_group=source_ranks_per_group,
        all_source_ranks=tuple(all_ranks),
        rank_to_attention_slot=rank_to_attention_slot,
        rank_offset_factor=rank_offset_factor,
    )
