# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX Expert Parallelism (EP) API."""

import atexit
import ctypes
from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.multihost_utils as jmu
import numpy as np

import transformer_engine_jax
import transformer_engine.jax.cpp_extensions as tex
from transformer_engine.jax.cpp_extensions.ep import _ep_outer_axis
from transformer_engine.jax.cpp_extensions.misc import jax_dtype_to_te_dtype
from transformer_engine.jax.sharding import (
    _get_mesh,
    get_num_devices_in_mesh,
    global_mesh_resource,
    get_mesh_axis_size,
    with_sharding_constraint,
)

ep_prepare = tex.ep_prepare
EpLayerConfig = tex.EpLayerConfig
ep_handle_mem_size = tex.ep_handle_mem_size

__all__ = [
    "EpLayerConfig",
    "ep_bootstrap",
    "ep_finalize",
    "ep_handle_mem_size",
    "ep_prepare",
    "ep_dispatch",
    "ep_combine",
]

_atexit_registered = False


def _normalize_axes(axes):
    if axes is None:
        return ()
    return (axes,) if isinstance(axes, str) else tuple(axes)


def _allgather_uid(uid_arr, world_size, uid_size):
    """Allgather UID bytes across all processes.

    Tries ``jax.experimental.multihost_utils.process_allgather`` first;
    falls back to an XLA collective (process-local sharded global array
    replicated via ``jax.jit``) when the multihost helper returns a
    short buffer, which has been observed under some launchers.
    """
    try:
        gathered = jmu.process_allgather(uid_arr, tiled=True)
        if gathered.size == world_size * uid_size:
            return np.asarray(gathered).reshape(world_size, uid_size)
    except Exception:  # pylint: disable=broad-except
        pass
    devices = np.asarray(jax.devices())
    if devices.size != world_size:
        raise RuntimeError(
            f"_allgather_uid fallback expected {world_size} global devices, got {devices.size}."
        )
    mesh = jax.sharding.Mesh(devices, ("_uid_all",))
    sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("_uid_all", None))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    local = np.asarray(uid_arr).reshape(1, uid_size)
    g_in = jax.make_array_from_process_local_data(sharded, local, (world_size, uid_size))
    with jax.set_mesh(mesh):
        g_out = jax.jit(lambda x: x, out_shardings=replicated)(g_in)
    return np.asarray(g_out).reshape(world_size, uid_size)


def _publish_and_fetch_domain_uid(uid_bytes, is_color_root, root_rank, uid_size):
    """Resolve one EP domain's NCCL UID via a point-to-point KV rendezvous.

    Only `all_uids[root_rank]` is ever read out of `_allgather_uid`'s result
    (see call site below) -- every other row is gathered and discarded. This
    does the same job (the domain's root publishes its UID, every member of
    the domain reads it back) without requiring a whole-job gather, by using
    JAX's distributed coordination-service key-value store directly, keyed
    by `root_rank` (globally unique per domain -- see `_ep_domain_for_rank`).

    This is the same put/get rendezvous pattern JaxPP uses for its own
    inter-stage NCCL communicators (`jaxpp.dime2.get_nccl_id`), generalized
    from a single leader->followers broadcast (jaxpp's case: exactly one
    process per side of a stage boundary) to root->all-domain-members. It
    doesn't require every process to be mutually reachable via a whole-job
    collective the way `_allgather_uid`'s `process_allgather`/`jax.devices()`
    paths do -- only the processes in this one EP domain need to agree on
    the key. That's what makes this safe to use when the ambient mesh (and
    therefore `jax.devices()`) has been narrowed to a subset of the job's
    devices, e.g. one JaxPP pipeline stage's local process group: those
    other paths hard-require `world_size`/`mesh` device count to equal the
    *entire* distributed job (see `_allgather_uid`'s size-mismatch checks),
    which is never true for a narrowed, per-stage mesh.

    Rank numbering (`root_rank`, `rank_within_group`, `EpConfig.rank`/
    `world_size`) is untouched by this function on purpose: it stays exactly
    whatever `_ep_domain_for_rank`/the caller already computed, so this is a
    transport-only change, not a semantic one.

    Uses a static, `root_rank`-only key (no call counter): `ep_bootstrap` is
    normally called exactly once per process (guarded by the
    `_TE_EP_STATE is not None` check in maxtext's te_ep_init.py), so there's
    no staleness risk in the common case. A counter would need to be
    identical across the root and every follower in the domain to produce
    matching keys, but a plain per-process Python global can't guarantee
    that (different processes may have made a different number of prior
    calls) -- so it would risk mismatched keys instead of preventing
    staleness. `reset_te_ep_state_for_test` already documents that
    re-bootstrapping doesn't tear down the underlying NCCL state, i.e. test
    re-init isn't fully clean today regardless of this function.
    """
    from jax._src.distributed import global_state  # pylint: disable=import-outside-toplevel

    client = global_state.client
    if client is None:
        raise RuntimeError(
            "_publish_and_fetch_domain_uid requires a distributed JAX runtime "
            "(jax.distributed.initialize() must have been called)."
        )
    key = f"te_ep_domain_uid:{root_rank}"
    if is_color_root:
        client.key_value_set_bytes(key, uid_bytes)
        result = uid_bytes
    else:
        TIMEOUT = 120_000  # ms; matches jaxpp's own get_nccl_id default order of magnitude
        result = bytes(client.blocking_key_value_get_bytes(key, TIMEOUT))
    if len(result) != uid_size:
        raise RuntimeError(
            f"_publish_and_fetch_domain_uid: expected {uid_size} bytes, got {len(result)} "
            f"(key={key!r})."
        )
    return result


# ── Bootstrap ────────────────────────────────────────────────────────────────


def _ep_domain_for_rank(mesh, ep_resource, rank, device_to_rank=None):
    """Resolve the EP domain (NCCL comm) for ``rank`` from the mesh layout.

    One domain groups ranks sharing all non-ep coordinates, so any orthogonal
    axis (tp, pp, cp, ...) yields its own domains. Returns
    ``(root_rank, rank_within_group, num_domains)``; ``root_rank`` (ep
    coordinate 0) posts the domain's NCCL unique id.
    """
    if device_to_rank is None:

        def device_to_rank(d):
            return d.process_index

    ep_axes = _normalize_axes(ep_resource)
    if not ep_axes or len(set(ep_axes)) != len(ep_axes):
        raise ValueError(f"ep_bootstrap: invalid ep_resource={ep_resource!r}.")
    missing = tuple(axis for axis in ep_axes if axis not in mesh.axis_names)
    if missing:
        raise ValueError(
            f"ep_bootstrap: EP axes {missing} are absent from mesh axes {mesh.axis_names}."
        )
    non_ep_axes = tuple(axis for axis in mesh.axis_names if axis not in ep_axes)
    transpose_axes = tuple(mesh.axis_names.index(axis) for axis in (*non_ep_axes, *ep_axes))
    ep_size = int(np.prod([mesh.shape[axis] for axis in ep_axes]))
    ranks = np.vectorize(device_to_rank, otypes=[np.int64])(mesh.devices)
    # Flatten every EP axis into one communicator dimension. Each row fixes all
    # non-EP coordinates and follows ep_resource tuple order within the domain.
    grid = np.transpose(ranks, transpose_axes).reshape(-1, ep_size)
    loc = np.argwhere(grid == rank)
    if loc.shape[0] != 1:
        raise ValueError(
            f"ep_bootstrap: rank {rank} must appear exactly once in the mesh device"
            f" grid; found {loc.shape[0]} occurrences."
        )
    row, col = int(loc[0, 0]), int(loc[0, 1])
    return int(grid[row, 0]), col, int(grid.shape[0])


def _num_ep_output_groups(mesh_resource, axis_size_fn=get_mesh_axis_size):
    """Count outer DP/FSDP output slabs not consumed by compound EP."""
    ep_axes = set(_normalize_axes(getattr(mesh_resource, "ep_resource", None)))
    distinct_token_axes = tuple(
        dict.fromkeys(
            axis
            for resource in (mesh_resource.dp_resource, mesh_resource.fsdp_resource)
            for axis in _normalize_axes(resource)
            if axis not in ep_axes
        )
    )
    num_groups = 1
    for axis in distinct_token_axes:
        num_groups *= int(axis_size_fn(axis))
    return num_groups


def ep_bootstrap(
    world_size,
    rank,
    num_experts,
    max_tokens_per_rank,
    recv_capacity_per_rank,
    hidden_dim,
    max_token_dtype=jnp.bfloat16,
    max_num_sms=0,
    drop_on_overflow=False,
    scope_uid_exchange_to_mesh=False,
):
    """Initialize the EP communicator. Call once per process before any EP op.

    Must run inside the active JAX Mesh and a global_shard_guard; ep_size and
    num_ep_groups are read from the mesh axes named by MeshResource.ep_resource
    and MeshResource.dp_resource/fsdp_resource. Axes orthogonal to EP (tp, pp,
    cp, ...) are supported and replicated across EP tensors.

    Args:
        world_size: Total number of processes (product of all mesh axes).
        rank: Global rank of the calling process.
        num_experts: Total experts across the EP group.
        max_tokens_per_rank: Max tokens one rank dispatches per step (sizes send buffers).
        recv_capacity_per_rank: Max tokens one rank receives per step; set to
            at least ep_size * max_tokens_per_rank * top_k to avoid drops.
        hidden_dim: Feature dimension of token tensors passed to ep_dispatch.
        max_token_dtype: Widest dtype the group will dispatch (only bfloat16 supported).
        max_num_sms: SM budget for EP kernels; 0 = auto.
        drop_on_overflow: Drop tokens exceeding recv_capacity_per_rank instead of
            trapping on overflow. Dropped tokens are still counted in
            total_recv_tokens, so callers can detect overflow from it.
        scope_uid_exchange_to_mesh: NOTE(jaxpp) When False (default, matches
            all existing behavior exactly), the NCCL unique-ID exchange uses
            `_allgather_uid`, which requires `world_size`/the active mesh's
            device count to equal the *entire* distributed JAX job -- true
            for ordinary flat-SPMD runs, never true when the active mesh has
            been narrowed to one JaxPP pipeline stage's local process group.
            When True, uses `_publish_and_fetch_domain_uid` instead: a
            point-to-point KV-store rendezvous scoped to just this EP
            domain's processes (see that function's docstring), which works
            correctly for a narrowed `world_size`/mesh. Rank numbering is
            identical either way; this only changes the UID transport.
    """
    if jnp.dtype(max_token_dtype) != jnp.bfloat16:
        raise NotImplementedError(
            "ep_bootstrap: only max_token_dtype=jnp.bfloat16 is supported today, got"
            f" {jnp.dtype(max_token_dtype)}."
        )
    if world_size < 2:
        raise ValueError(
            f"ep_bootstrap requires world_size >= 2 (got {world_size}); NCCL EP needs"
            " at least 2 ranks to form a group."
        )
    if jax.local_device_count() != 1:
        raise ValueError(
            "ep_bootstrap requires one local device per process (got"
            f" jax.local_device_count() = {jax.local_device_count()}); NCCL EP does not"
            " support single-process multi-device setups."
        )

    gsr = global_mesh_resource()
    ep_resource = gsr.ep_resource
    if ep_resource is None:
        raise ValueError(
            "ep_bootstrap requires MeshResource.ep_resource to be set; enter a"
            " global_shard_guard(MeshResource(..., ep_resource=<axis name>)) before bootstrap."
        )
    mesh = _get_mesh()
    if mesh.empty:
        raise ValueError(
            "ep_bootstrap must run inside an active jax.sharding.Mesh; enter"
            " `with mesh:` (or jax.set_mesh(mesh)) before calling it."
        )
    if get_num_devices_in_mesh(mesh) != world_size:
        raise ValueError(
            f"ep_bootstrap: mesh device count ({get_num_devices_in_mesh(mesh)}) must equal"
            f" world_size ({world_size})."
        )
    ep_size = get_mesh_axis_size(ep_resource)
    if world_size % ep_size != 0:
        raise ValueError(
            f"ep_bootstrap: world_size ({world_size}) must be divisible by ep_size ({ep_size})."
        )
    # Communicators cover every fixed non-EP coordinate, including TP replicas.
    expected_num_domains = world_size // ep_size
    # EP outputs only distinguish token-owning DP/FSDP coordinates. TP is a
    # replicated communicator dimension unless deliberately named as DP, as in
    # MaxText's compound ETP1 view (dp=tensor, fsdp=fsdp).
    num_ep_groups = _num_ep_output_groups(gsr)
    if num_experts % ep_size != 0:
        raise ValueError(f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size}).")

    UID_SIZE = 128
    root_rank, rank_within_group, num_domains = _ep_domain_for_rank(mesh, ep_resource, rank)
    if num_domains != expected_num_domains:
        raise ValueError(
            f"ep_bootstrap: mesh has {num_domains} EP domains, expected"
            f" world_size // ep_size = {expected_num_domains}."
        )
    is_color_root = rank_within_group == 0
    if is_color_root:
        libnccl = ctypes.CDLL("libnccl.so.2", use_errno=True)
        uid_arr = (ctypes.c_uint8 * UID_SIZE)()
        ret = libnccl.ncclGetUniqueId(ctypes.cast(uid_arr, ctypes.c_void_p))
        assert ret == 0, f"ncclGetUniqueId failed with code {ret}"
        uid_bytes = bytes(uid_arr)
    else:
        uid_bytes = bytes(UID_SIZE)

    if scope_uid_exchange_to_mesh:
        uid_bytes = _publish_and_fetch_domain_uid(uid_bytes, is_color_root, root_rank, UID_SIZE)
    else:
        uid_arr = jnp.frombuffer(uid_bytes, dtype=jnp.uint8)
        all_uids = _allgather_uid(uid_arr, world_size, UID_SIZE)
        uid_bytes = bytes(np.asarray(all_uids[root_rank]).tolist())

    # Eager NCCL init while ranks are barrier-synced by the UID broadcast above.
    transformer_engine_jax.set_ep_bootstrap_params(
        uid_bytes,
        ep_size,
        rank_within_group,
        num_experts,
        max_tokens_per_rank,
        recv_capacity_per_rank,
        hidden_dim,
        max_num_sms=int(max_num_sms),
        max_token_dtype=int(jax_dtype_to_te_dtype(max_token_dtype)),
        drop_on_overflow=bool(drop_on_overflow),
    )

    # Release the C++ anchor at interpreter shutdown so RAII can tear down NCCL.
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(transformer_engine_jax.release_ep_resources)
        _atexit_registered = True

    tex.ep.set_ep_config(
        tex.ep.EpConfig(
            world_size=world_size,
            rank=rank,
            ep_size=ep_size,
            num_ep_groups=num_ep_groups,
            num_experts=num_experts,
            num_local_experts=num_experts // ep_size,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=recv_capacity_per_rank,
            hidden_dim=hidden_dim,
        )
    )


def ep_finalize():
    """Tear down the EP communicator so ``ep_bootstrap`` can run again.

    Only for killing and re-bootstrapping EP mid-program (e.g. tests sweeping
    configs); a normal run bootstraps once and lets atexit clean up. Calls the
    process-global ``jax.clear_caches()`` so every cached executable releases
    the NCCL comm it pins, then frees the EP resources. Call outside any active
    EP computation.
    """
    jax.clear_caches()
    transformer_engine_jax.release_ep_resources()
    tex.ep.reset_ep_config()


def _default_out_partition_spec():
    """Leading-axis default: ``(("dp","ep"),)`` if DP/FSDP is set, else ``("ep",)``."""
    gsr = global_mesh_resource()
    if gsr.ep_resource is None:
        raise ValueError(
            "ep_resource is not set on the active MeshResource; pass out_sharding=... explicitly."
        )
    leading_axes = (*_normalize_axes(_ep_outer_axis()), *_normalize_axes(gsr.ep_resource))
    leading = leading_axes[0] if len(leading_axes) == 1 else leading_axes
    return (leading,)


# ── ep_dispatch (custom_vjp) ─────────────────────────────────────────────────


@partial(jax.custom_vjp, nondiff_argnums=(0, 4, 5))
def ep_dispatch(
    cfg,
    topk_idx,
    tokens,
    topk_weights,
    recv_capacity_per_rank,
    out_sharding=None,
):
    """Scatter tokens and weights to expert ranks.

    ``cfg`` is a per-layer ``EpLayerConfig``; distinct layers may share a
    ``cfg`` (the pointer-keyed C++ cache keys on handle_mem, not on cfg).
    Inputs are ``[..., H]`` with only the leading dim sharded as ``ep`` or
    ``(dp, ep)``. Returns
    ``(recv_tokens, recv_topk_weights, handle_mem, token_counts, total_recv_tokens)``;
    pass ``handle_mem`` and ``token_counts`` to the matching ``ep_combine``.

    ``total_recv_tokens`` is the per-rank pre-drop recv-slot count (a
    ``[num_procs, 1]`` sharded array); it counts dropped tokens too when
    ``drop_on_overflow`` is set. When ``recv_capacity_per_rank`` is not sized for
    the worst case, detect overflow by ``process_allgather``-ing it, then
    ``max(...) > recv_capacity_per_rank`` flags an overflowing step.
    """
    return _dispatch_fwd(
        cfg,
        topk_idx,
        tokens,
        topk_weights,
        recv_capacity_per_rank,
        out_sharding,
    )[0]


def _dispatch_fwd(
    cfg,
    topk_idx,
    tokens,
    topk_weights,
    recv_capacity_per_rank,
    out_sharding=None,
):
    if not jnp.issubdtype(topk_weights.dtype, jnp.floating):
        raise TypeError(
            f"ep_dispatch: topk_weights must be a floating dtype; got {topk_weights.dtype}."
        )
    token_counts, total_recv_tokens, handle_mem = tex.ep_prepare(cfg, topk_idx)
    recv_tokens, recv_topk_weights = tex.ep_dispatch_fwd(
        cfg, handle_mem, topk_idx, tokens, topk_weights, recv_capacity_per_rank
    )
    out_leading = tuple(tokens.shape[:-1])
    primal = (recv_tokens, recv_topk_weights, handle_mem, token_counts, total_recv_tokens)
    return primal, (handle_mem, out_leading)


def _dispatch_bwd(cfg, recv_capacity_per_rank, out_sharding, res, g_outputs):
    del recv_capacity_per_rank
    handle_mem, out_leading = res
    # Re-pin cotangent: XLA transpose can drop the EP axis and feed the FFI a global tensor.
    out_spec = _default_out_partition_spec() if out_sharding is None else out_sharding
    spec = jax.sharding.PartitionSpec(*out_spec)
    g_recv_tokens = with_sharding_constraint(g_outputs[0], spec)
    g_recv_topk_weights = with_sharding_constraint(g_outputs[1], spec)
    grad_tokens, grad_topk_weights = tex.ep_dispatch_bwd(
        cfg,
        handle_mem,
        g_recv_tokens,
        g_recv_topk_weights,
        out_leading,
        out_partition_spec=out_spec,
    )
    return (None, grad_tokens, grad_topk_weights)


ep_dispatch.defvjp(_dispatch_fwd, _dispatch_bwd)


# ── ep_combine (custom_vjp) ──────────────────────────────────────────────────


@partial(jax.custom_vjp, nondiff_argnums=(0, 4, 5))
def ep_combine(
    cfg,
    handle_mem,
    token_counts,
    expert_out,
    num_local_tokens,
    out_sharding=None,
):
    """Scatter-sum expert outputs back to source ranks. **Unweighted.**

    Caller must pre-multiply ``expert_out`` by ``recv_topk_weights`` (and
    zero padded slots); gradients w.r.t. weights flow through that hadamard,
    not through this op. ``num_local_tokens`` is STATIC: int -> ``[T, H]``,
    tuple -> ``[*tuple, H]``. ``out_sharding`` defaults via
    ``_default_out_partition_spec``; only the leading dim may be sharded.
    """
    return _combine_fwd(
        cfg,
        handle_mem,
        token_counts,
        expert_out,
        num_local_tokens,
        out_sharding,
    )[0]


def _combine_fwd(
    cfg,
    handle_mem,
    token_counts,
    expert_out,
    num_local_tokens,
    out_sharding,
):
    del token_counts
    if out_sharding is None:
        out_sharding = _default_out_partition_spec()
    result = tex.ep_combine_fwd(
        cfg, handle_mem, expert_out, num_local_tokens, out_partition_spec=out_sharding
    )
    return result, (handle_mem, expert_out.shape[-2])


def _combine_bwd(cfg, _num_local_tokens, _out_sharding, res, g_result):
    handle_mem, recv_capacity_per_rank = res
    # Re-pin cotangent (same XLA-transpose workaround as _dispatch_bwd).
    if _out_sharding is None:
        _out_sharding = _default_out_partition_spec()
    spec = jax.sharding.PartitionSpec(*_out_sharding)
    g_result = with_sharding_constraint(g_result, spec)
    grad_expert_out = tex.ep_combine_bwd(cfg, handle_mem, g_result, recv_capacity_per_rank)
    return (None, None, grad_expert_out)


ep_combine.defvjp(_combine_fwd, _combine_bwd)
