# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TransformerEngine NCCL EP bootstrap helpers for MaxText MoE.

Process-singleton bootstrap of TE NCCL EP, mirroring HybridEP's pattern.
Designed so the module imports cleanly even when transformer_engine is not
installed; TE imports happen lazily inside :func:`init_te_ep_for_maxtext`.

Lessons baked in (see plans/jax_hybridep/te_ep_maxtext_v2_todo.md and
plans/jax_hybridep/te_ep_recv_capacity_overflow.md):
  * MeshResource preserves optional ICI TP for TE EP bootstrap and MoE
    dispatch/combine, while the outer train context can keep TP unset.
  * ``dispatch_alignment`` passed to TE EP is the *small* alignment
    ``moe_permutation_group_align_size`` (default 128). This minimizes per-expert
    padding overhead, matching how HybridEP/DeepEP uses ``pad_multiple``. Earlier
    versions forced ``dispatch_alignment = slots_per_expert`` (~4096) to get a
    uniform per-expert reshape, but the resulting padding overhead under routing
    skew always overflowed (each "hot" expert wastes one ~4096-slot block).
  * ``recv_capacity_per_rank = (T_per_ep_group * top_k * overconc * factor) +
    num_local_experts * dispatch_alignment``. The headroom term covers the
    worst-case per-expert padding: every expert may pad up to one align unit (=
    `dispatch_alignment - 1` slots wasted), so the global overhead is bounded
    by ``num_local_experts * dispatch_alignment``.
  * The MoE GMM consumer no longer assumes uniform per-expert blocks; it
    computes ``padded_token_counts`` from the returned ``token_counts`` and uses
    those as ``group_sizes`` (mirroring HybridEP's pattern in moe.py).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any

import jax
from jax.sharding import PartitionSpec

from maxtext.utils import max_logging


_TE_EP_AXIS = "expert"
_TE_EP_COMPOUND_AXES = ("tensor", "expert")
_TE_EP_STATE: "TeEpState | None" = None


@dataclass(frozen=True)
class TeEpState:
  """Process-local TE EP bootstrap state."""

  mesh: Any
  mesh_resource: Any
  ep_axis: str | tuple[str, ...]
  ep_axes: tuple[str, ...]
  uses_compound_ep: bool
  outer_axes: tuple[str, ...]
  outer_size: int
  ep_size: int
  tensor_axis: str | None
  tensor_size: int
  dense_tensor_size: int
  uses_etp1_view: bool
  expected_world_size: int
  num_experts: int
  num_local_experts: int
  max_tokens_per_rank: int
  recv_capacity_per_rank: int
  drop_on_overflow: bool
  dispatch_alignment: int
  hidden_dim: int
  max_num_sms: int
  em_unfused_num_sms: int
  needs_v1_tail_absorb: bool
  top_k: int          # config.num_experts_per_tok
  num_moe_layers: int # num_decoder_layers - first_num_dense_layers
  routing_spec_2d: PartitionSpec
  input_spec_2d: PartitionSpec
  input_spec_3d: PartitionSpec
  ep_spec_2d: PartitionSpec
  ep_spec_3d: PartitionSpec
  config_key: tuple[Any, ...]

  @property
  def outer_axis(self) -> str | tuple[str, ...] | None:
    """Compatibility view of the outer axes used by older callers."""
    if not self.outer_axes:
      return None
    if len(self.outer_axes) == 1:
      return self.outer_axes[0]
    return self.outer_axes


def _mesh_axis_size(mesh: jax.sharding.Mesh, axis: str | tuple[str, ...]) -> int:
  axes = (axis,) if isinstance(axis, str) else tuple(axis)
  missing = tuple(name for name in axes if name not in mesh.shape)
  if missing:
    raise ValueError(
        f"TE EP requires mesh axes {missing}. Available axes: {tuple(mesh.shape.keys())}."
    )
  return math.prod(int(mesh.shape[name]) for name in axes)


def _active_mesh_axes(mesh: jax.sharding.Mesh) -> dict[str, int]:
  return {axis: int(size) for axis, size in mesh.shape.items() if int(size) > 1}


def _normalize_axes(axes: str | tuple[str, ...] | None) -> tuple[str, ...]:
  if axes is None:
    return ()
  return (axes,) if isinstance(axes, str) else tuple(axes)


def te_ep_expert_partition_axis(config: Any) -> str | tuple[str, ...]:
  """Physical axis resource that partitions complete expert parameters."""
  if bool(getattr(config, "te_ep_compound_tensor_expert", False)):
    return _TE_EP_COMPOUND_AXES
  return _TE_EP_AXIS


def select_te_ep_outer_axis(mesh: jax.sharding.Mesh) -> str | None:
  """Compatibility selector for callers that can represent only one outer axis.

  New TE EP state construction uses :func:`select_te_ep_outer_axes` and supports
  active ``data`` and ``fsdp`` together. This helper preserves the historical
  preference ``fsdp`` > ``data`` for older callers.
  """
  if "fsdp" in mesh.shape:
    return "fsdp"
  if "data" in mesh.shape:
    return "data"
  return None


def select_te_ep_outer_axes(mesh: jax.sharding.Mesh) -> tuple[str, ...]:
  """Return active data/FSDP axes in deterministic operand-leading order."""
  return tuple(axis for axis in ("data", "fsdp") if int(mesh.shape.get(axis, 1)) > 1)


def _validate_v1_mesh(
    mesh: jax.sharding.Mesh,
    outer_axes: tuple[str, ...],
    tensor_axis: str | None,
    ep_axes: tuple[str, ...] = (_TE_EP_AXIS,),
) -> None:
  """Validate EP, optional ICI TP, and any active data/FSDP outer axes."""
  active_axes = _active_mesh_axes(mesh)
  allowed = set(ep_axes)
  allowed.update(outer_axes)
  if tensor_axis is not None:
    allowed.add(tensor_axis)
  unsupported = {axis: size for axis, size in active_axes.items() if axis not in allowed}
  if unsupported:
    raise ValueError(
        "use_te_ep=True v1 supports only the expert axis, optional ICI tensor axis, "
        "and outer data/FSDP axes. "
        f"Unsupported active mesh axes: {unsupported}."
    )


def _build_mesh_resource(
    outer_axes: tuple[str, ...], ep_axis: str | tuple[str, ...], tensor_axis: str | None
) -> Any:
  """Build a MeshResource for TE EP bootstrap.

  Sets ``fsdp_resource`` + ``ep_resource`` and, when active, ``tp_resource``.
  Leaves ``cp_resource`` unset. TE's ``_validate_mesh_resource_configuration``
  calls ``get_mesh_axis_size`` on every set resource, which asserts when the
  named axis is missing from the active JAX mesh (e.g. inside ``jax.eval_shape``).
  """
  from transformer_engine.jax.sharding import MeshResource  # pylint: disable=import-outside-toplevel

  kwargs: dict[str, Any] = {"ep_resource": ep_axis}
  for outer_axis in outer_axes:
    if outer_axis in ("data", "tensor"):
      if "dp_resource" in kwargs:
        raise ValueError(f"TE EP has multiple data resources: {outer_axes}.")
      kwargs["dp_resource"] = outer_axis
    elif outer_axis == "fsdp":
      kwargs["fsdp_resource"] = outer_axis
    else:
      raise ValueError(f"Unsupported TE EP outer mesh axis: {outer_axis}.")
  if tensor_axis is not None:
    kwargs["tp_resource"] = tensor_axis
  return MeshResource(**kwargs)


def calculate_te_ep_capacity(
    *,
    max_tokens_per_rank: int,
    ep_size: int,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    recv_capacity_factor: float,
    dispatch_alignment: int,
) -> int:
  """Worst-case TE EP receive-buffer size per rank.

  Formula::

      T_per_ep_group = max_tokens_per_rank * ep_size
      worst_case     = max(T_per_ep_group * top_k, 16) * overconc
      target_tokens  = ceil(worst_case * recv_capacity_factor)
      recv_capacity  = ceil_to(target_tokens + NLE * dispatch_alignment, dispatch_alignment)

  ``overconc`` covers the degenerate case where ``num_experts`` exceeds the
  total routing pool ``T_per_ep_group * top_k``. ``dispatch_alignment`` is the
  per-expert padding granularity used by TE EP (mirrors HybridEP's
  ``pad_multiple``); it should be a POW2 (TE EP's ``ncclEpInitHandle``
  asserts ``dispatch_output_per_expert_alignment`` is power-of-two), and
  defaults to ``moe_permutation_group_align_size`` (=128).

  The ``NLE * dispatch_alignment`` headroom term covers the worst-case
  per-expert padding overhead. TE EP allocates each expert's recv block as
  ``ceil(actual_count / dispatch_alignment) * dispatch_alignment`` rows; under
  routing skew, every expert can waste up to ``dispatch_alignment - 1`` rows,
  so total padding overhead ≤ ``NLE * (dispatch_alignment - 1)``. Using
  ``NLE * dispatch_alignment`` gives one full align-block of safety margin.

  With ``dispatch_alignment == 0`` (unaligned mode for tests/diagnostics),
  the result equals ``target_tokens``.
  """
  tokens_per_ep_group = max_tokens_per_rank * ep_size
  active_experts = min(num_experts, tokens_per_ep_group * top_k)
  overconc = max(1, math.ceil(num_experts / max(1, active_experts)))
  worst_case = max(tokens_per_ep_group * top_k, 16) * overconc
  target = max(1, math.ceil(worst_case * recv_capacity_factor))

  if dispatch_alignment > 0:
    # Worst-case per-expert padding: each of NLE experts pads up to one
    # dispatch_alignment block; the +1 margin keeps a full block of headroom.
    recv_capacity = target + num_local_experts * dispatch_alignment
    # Round recv_capacity up to a clean dispatch_alignment multiple.
    recv_capacity = math.ceil(recv_capacity / dispatch_alignment) * dispatch_alignment
    return recv_capacity
  return target


def calculate_te_ep_padded_capacity_bound(
    *,
    max_tokens_per_rank: int,
    ep_size: int,
    top_k: int,
    num_local_experts: int,
    dispatch_alignment: int,
) -> int:
  """Worst-case ``sum(ceil(token_count / align) * align)`` for one recv rank."""
  routed_token_bound = max(0, int(max_tokens_per_rank) * int(ep_size) * int(top_k))
  if dispatch_alignment <= 0:
    return routed_token_bound

  align = int(dispatch_alignment)
  active_local_experts = min(int(num_local_experts), routed_token_bound)
  padded_bound = routed_token_bound + active_local_experts * (align - 1)
  return math.ceil(padded_bound / align) * align


def _max_tokens_per_rank(config: Any, leading_axis_size: int) -> int:
  global_tokens = int(config.micro_batch_size_to_train_on * config.max_target_length)
  return max(1, math.ceil(global_tokens / max(1, leading_axis_size)))


def _hidden_dim(config: Any) -> int:
  return int(config.moe_expert_input_dim if config.moe_expert_input_dim > 0 else config.emb_dim)


def _needs_v1_tail_absorb() -> bool:
  """True when the MXFP8 GroupedQuantize path requires sum(group_sizes) == recv_capacity.

  Both V1 and V2 MXFP8 grouped quantize/GEMM paths require
  ``sum(group_sizes) == recv_capacity``:
  - V1 (pre-sm_100, or old containers): explicit NVTE_CHECK assertion.
  - V2 (sm_100+, TE >= commit 70af7305): no explicit assertion, but the kernel
    reads exactly ``recv_capacity`` rows from the buffer; if sum(group_sizes) <
    recv_capacity the remaining rows are read but not assigned to any expert →
    wrong outputs.

  Our path-C1 variable-block layout always has sum(padded) <= recv_capacity
  (padded per-expert counts don't fill the buffer under typical load). We must
  absorb the unused tail (recv_capacity - sum(padded)) into the last expert's
  group to satisfy the invariant. The ~17% overhead from extra padded rows is
  acceptable vs silent wrong results or an FFI crash.

  Historical note: the original sm < 100 gating was based on the incorrect
  assumption that the V2 path doesn't enforce this invariant. Container 0529
  (TE >= 70af7305) introduced V2 MXFP8 GroupedQuantize and revealed the issue.
  """
  return True


def build_te_ep_state(config: Any, mesh: jax.sharding.Mesh) -> TeEpState:
  """Build the TE EP state without mutating the process singleton.

  Pure function; tests can call this without triggering ``ep_bootstrap``.
  """
  dense_tensor_size = int(mesh.shape.get("tensor", 1))
  requested_etp = int(getattr(config, "te_ep_expert_tensor_parallelism", 0))
  uses_compound_ep = bool(getattr(config, "te_ep_compound_tensor_expert", False))
  if requested_etp not in (0, 1):
    raise ValueError(
        "te_ep_expert_tensor_parallelism must be 0 (inherit dense TP) or 1; "
        f"got {requested_etp}."
    )

  dense_outer_axes = select_te_ep_outer_axes(mesh)
  dense_tensor_axis = "tensor" if dense_tensor_size > 1 else None

  uses_etp1_view = requested_etp == 1
  if uses_compound_ep and not uses_etp1_view:
    raise ValueError(
        "te_ep_compound_tensor_expert=True requires "
        "te_ep_expert_tensor_parallelism=1."
    )
  ep_axes = _TE_EP_COMPOUND_AXES if uses_compound_ep else (_TE_EP_AXIS,)
  ep_axis: str | tuple[str, ...] = ep_axes if uses_compound_ep else _TE_EP_AXIS
  _validate_v1_mesh(
      mesh,
      dense_outer_axes,
      None if uses_compound_ep else dense_tensor_axis,
      ep_axes,
  )
  te_mesh = mesh
  ep_size = _mesh_axis_size(mesh, ep_axis)
  if uses_compound_ep:
    outer_axes = dense_outer_axes
    outer_size = math.prod(_mesh_axis_size(mesh, axis) for axis in outer_axes)
    tensor_axis = None
    tensor_size = 1
  elif uses_etp1_view:
    outer_axes = tuple(
        axis
        for axis in (dense_tensor_axis, *dense_outer_axes)
        if axis is not None and _mesh_axis_size(mesh, axis) > 1
    )
    outer_size = math.prod(_mesh_axis_size(mesh, axis) for axis in outer_axes)
    tensor_axis = None
    tensor_size = 1
  else:
    outer_axes = dense_outer_axes
    outer_size = math.prod(_mesh_axis_size(mesh, axis) for axis in outer_axes)
    tensor_axis = dense_tensor_axis
    tensor_size = dense_tensor_size

  if int(config.num_experts) % ep_size != 0:
    raise ValueError(
        f"num_experts ({config.num_experts}) must be divisible by TE EP size ({ep_size})."
    )

  num_local_experts = int(config.num_experts) // ep_size
  min_dispatch_alignment = int(config.moe_permutation_group_align_size)
  token_partition_size = outer_size * ep_size
  expected_world_size = token_partition_size * tensor_size
  max_tokens_per_rank = _max_tokens_per_rank(config, token_partition_size)
  derived_recv_capacity = calculate_te_ep_capacity(
      max_tokens_per_rank=max_tokens_per_rank,
      ep_size=ep_size,
      num_experts=int(config.num_experts),
      top_k=int(config.num_experts_per_tok),
      num_local_experts=num_local_experts,
      recv_capacity_factor=float(config.te_ep_recv_capacity_factor),
      dispatch_alignment=min_dispatch_alignment,
  )
  # Mirrors HybridEP's JAX_DEEP_EP_MAX_PERMUTED_TOKENS env-var override. Set to
  # a tighter (or larger) static recv-buffer size without editing the yaml — the
  # rest of the formula (per-expert alignment, mesh) stays the same. Useful for
  # quick A/B benchmarks of dispatch/combine output buffer sizing.
  override_env = os.environ.get("TE_EP_RECV_CAPACITY_PER_RANK", "").strip()
  if override_env:
    override_val = int(override_env)
    if override_val <= 0:
      raise ValueError(
          f"TE_EP_RECV_CAPACITY_PER_RANK must be positive, got {override_val}"
      )
    if min_dispatch_alignment > 0 and override_val % min_dispatch_alignment != 0:
      raise ValueError(
          f"TE_EP_RECV_CAPACITY_PER_RANK={override_val} must be a multiple of "
          f"moe_permutation_group_align_size={min_dispatch_alignment}."
      )
    max_logging.log(
        f"TE EP: overriding recv_capacity_per_rank from env "
        f"{derived_recv_capacity} -> {override_val}"
    )
    recv_capacity_per_rank = override_val
  else:
    recv_capacity_per_rank = derived_recv_capacity

  # Static capacity sanity check. This is the true routing worst-case for the
  # padded rows consumed by grouped GEMM. Keep this as a warning instead of a
  # hard failure so reduced-capacity experiments remain possible; such runs can
  # still underflow under sufficiently skewed routing.
  worst_case_padded_capacity = calculate_te_ep_padded_capacity_bound(
      max_tokens_per_rank=max_tokens_per_rank,
      ep_size=ep_size,
      top_k=int(config.num_experts_per_tok),
      num_local_experts=num_local_experts,
      dispatch_alignment=min_dispatch_alignment,
  )
  if recv_capacity_per_rank < worst_case_padded_capacity:
    max_logging.log(
        "TE EP: recv_capacity_per_rank="
        f"{recv_capacity_per_rank} is below the worst-case padded receive "
        f"bound {worst_case_padded_capacity}. This may underflow tail absorb "
        "under routing skew; increase te_ep_recv_capacity_factor or "
        "TE_EP_RECV_CAPACITY_PER_RANK for production runs."
    )

  # dispatch_alignment is the *small* per-expert padding granularity (typically
  # 128 = moe_permutation_group_align_size). The recv buffer holds variable-sized
  # per-expert blocks of `ceil(token_counts[k] / dispatch_alignment) * dispatch_alignment`
  # rows; the MoE GMM consumer computes those padded counts from `token_counts`
  # and uses them as `group_sizes` (mirroring HybridEP/DeepEP's pattern).
  dispatch_alignment = max(1, min_dispatch_alignment)
  needs_v1_tail_absorb = _needs_v1_tail_absorb()

  num_moe_layers = int(config.num_decoder_layers) - int(config.first_num_dense_layers)

  leading_axes = (*outer_axes, *ep_axes)
  leading_spec: Any = leading_axes[0] if len(leading_axes) == 1 else leading_axes
  hidden_spec: Any = tensor_axis
  config_key = (
      ep_axis,
      uses_compound_ep,
      outer_axes,
      outer_size,
      ep_size,
      tensor_axis,
      tensor_size,
      dense_tensor_size,
      uses_etp1_view,
      expected_world_size,
      int(config.num_experts),
      int(config.num_experts_per_tok),
      num_local_experts,
      max_tokens_per_rank,
      recv_capacity_per_rank,
      bool(config.te_ep_drop_on_overflow),
      dispatch_alignment,
      _hidden_dim(config),
      int(config.te_ep_max_num_sms),
      int(config.te_ep_em_unfused_num_sms),
      needs_v1_tail_absorb,
      num_moe_layers,
  )

  return TeEpState(
      mesh=te_mesh,
      mesh_resource=_build_mesh_resource(outer_axes, ep_axis, tensor_axis),
      ep_axis=ep_axis,
      ep_axes=ep_axes,
      uses_compound_ep=uses_compound_ep,
      outer_axes=outer_axes,
      outer_size=outer_size,
      ep_size=ep_size,
      tensor_axis=tensor_axis,
      tensor_size=tensor_size,
      dense_tensor_size=dense_tensor_size,
      uses_etp1_view=uses_etp1_view,
      expected_world_size=expected_world_size,
      num_experts=int(config.num_experts),
      num_local_experts=num_local_experts,
      max_tokens_per_rank=max_tokens_per_rank,
      recv_capacity_per_rank=recv_capacity_per_rank,
      drop_on_overflow=bool(config.te_ep_drop_on_overflow),
      dispatch_alignment=dispatch_alignment,
      hidden_dim=_hidden_dim(config),
      max_num_sms=int(config.te_ep_max_num_sms),
      em_unfused_num_sms=int(config.te_ep_em_unfused_num_sms),
      needs_v1_tail_absorb=needs_v1_tail_absorb,
      top_k=int(config.num_experts_per_tok),
      num_moe_layers=num_moe_layers,
      routing_spec_2d=PartitionSpec(leading_spec, None),
      input_spec_2d=PartitionSpec(leading_spec, hidden_spec),
      input_spec_3d=PartitionSpec(leading_spec, None, hidden_spec),
      ep_spec_2d=PartitionSpec(leading_spec, None),
      ep_spec_3d=PartitionSpec(leading_spec, None, hidden_spec),
      config_key=config_key,
  )


def init_te_ep_for_maxtext(config: Any, mesh: jax.sharding.Mesh) -> TeEpState:
  """Bootstrap TE NCCL EP exactly once per process.

  Must be called before ``setup_train_loop`` — model creation traces
  ``moe.py`` which dispatches into ``ep_dispatch``. Idempotent for matching
  ``config_key``; raises on shape/resource mismatch.
  """
  from maxtext.common.common_types import DecoderBlockType  # pylint: disable=import-outside-toplevel

  global _TE_EP_STATE

  # Guard: only the normal scanned DeepSeek MoE path is wired for
  # te_ep_layer_idx threading. Loosen guards as other paths are added.
  if not bool(getattr(config, "scan_layers", False)):
    raise ValueError(
        "use_te_ep=True requires scan_layers=True. Unrolled MoE stacks need "
        "separate per-layer handle plumbing that is not yet implemented."
    )
  if getattr(config, "decoder_block", None) != DecoderBlockType.DEEPSEEK:
    raise ValueError("use_te_ep=True currently only supports decoder_block=DEEPSEEK.")
  if bool(getattr(config, "using_pipeline_parallelism", False)):
    raise ValueError("use_te_ep=True does not support pipeline parallelism.")
  if bool(getattr(config, "use_batch_split_schedule", False)):
    raise ValueError("use_te_ep=True does not support use_batch_split_schedule.")
  if getattr(config, "engram_layers", None):
    raise ValueError("use_te_ep=True does not support engram_layers.")
  if int(getattr(config, "mhc_expansion_rate", 1)) > 1:
    raise ValueError("use_te_ep=True does not support mhc_expansion_rate > 1.")

  candidate = build_te_ep_state(config, mesh)
  if _TE_EP_STATE is not None:
    if _TE_EP_STATE.config_key != candidate.config_key:
      raise ValueError(
          "TE EP was already initialized with a different shape/resource contract. "
          f"Existing={_TE_EP_STATE.config_key}, requested={candidate.config_key}."
      )
    return _TE_EP_STATE

  world_size = jax.process_count()
  rank = jax.process_index()
  if world_size != candidate.expected_world_size:
    raise ValueError(
        "TE EP v1 expects one JAX process per active expert-view mesh slot. "
        f"process_count={world_size}, expected={candidate.expected_world_size}, "
        f"outer_axes={candidate.outer_axes}, ep_size={candidate.ep_size}, "
        f"expert_tensor_size={candidate.tensor_size}, dense_tensor_size={candidate.dense_tensor_size}."
    )

  from transformer_engine.jax.ep import ep_bootstrap  # pylint: disable=import-outside-toplevel
  from transformer_engine.jax.sharding import global_shard_guard  # pylint: disable=import-outside-toplevel

  with candidate.mesh, jax.set_mesh(candidate.mesh), global_shard_guard(candidate.mesh_resource):
    # TE EP branch pr-3036 signature: ep_size is derived internally from the
    # mesh (MeshResource.ep_resource, set in _build_mesh_resource), so it is no
    # longer passed explicitly. That branch also dropped the separate
    # max_num_permute_sms knob (only max_num_sms remains) and the
    # allow_handle_mem_reloc flag.
    ep_bootstrap(
        world_size=world_size,
        rank=rank,
        num_experts=candidate.num_experts,
        max_tokens_per_rank=candidate.max_tokens_per_rank,
        recv_capacity_per_rank=candidate.recv_capacity_per_rank,
        hidden_dim=candidate.hidden_dim,
        max_num_sms=candidate.max_num_sms,
        drop_on_overflow=candidate.drop_on_overflow,
    )

  _TE_EP_STATE = candidate
  max_logging.log(
      "TE EP bootstrapped: "
      f"outer_axes={candidate.outer_axes}, ep_axes={candidate.ep_axes}, "
      f"compound={candidate.uses_compound_ep}, "
      f"ep_size={candidate.ep_size}, outer_size={candidate.outer_size}, "
      f"tensor_axis={candidate.tensor_axis}, expert_tensor_size={candidate.tensor_size}, "
      f"dense_tensor_size={candidate.dense_tensor_size}, etp1_view={candidate.uses_etp1_view}, "
      f"num_moe_layers={candidate.num_moe_layers}, "
      f"max_tokens_per_rank={candidate.max_tokens_per_rank}, "
      f"recv_capacity_per_rank={candidate.recv_capacity_per_rank}, "
      f"drop_on_overflow={candidate.drop_on_overflow}, "
      f"dispatch_alignment={candidate.dispatch_alignment}"
  )
  return _TE_EP_STATE


def get_te_ep_state() -> TeEpState:
  if _TE_EP_STATE is None:
    raise ValueError(
        "TE EP has not been initialized. Call init_te_ep_for_maxtext(config, mesh) before tracing MoE."
    )
  return _TE_EP_STATE

def reset_te_ep_state_for_test() -> None:
  """Test-only: clear the singleton. Does NOT tear down the underlying TE NCCL state."""
  global _TE_EP_STATE
  _TE_EP_STATE = None
