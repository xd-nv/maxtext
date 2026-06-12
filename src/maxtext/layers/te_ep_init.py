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
  * MeshResource preserves tp/cp from the outer context (does not strip).
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
from typing import Any

import jax
from jax.sharding import PartitionSpec

from maxtext.utils import max_logging


_TE_EP_AXIS = "expert"
_TE_EP_STATE: "TeEpState | None" = None


@dataclass(frozen=True)
class TeEpState:
  """Process-local TE EP bootstrap state."""

  mesh_resource: Any
  ep_axis: str
  outer_axis: str | None
  outer_size: int
  ep_size: int
  expected_world_size: int
  num_experts: int
  num_local_experts: int
  max_tokens_per_rank: int
  recv_capacity_per_rank: int
  dispatch_alignment: int
  hidden_dim: int
  max_num_sms: int
  em_unfused_num_sms: int
  needs_v1_tail_absorb: bool
  input_spec_2d: PartitionSpec
  input_spec_3d: PartitionSpec
  ep_spec_2d: PartitionSpec
  ep_spec_3d: PartitionSpec
  config_key: tuple[Any, ...]


def _mesh_axis_size(mesh: jax.sharding.Mesh, axis: str) -> int:
  if axis not in mesh.shape:
    raise ValueError(
        f"TE EP requires mesh axis '{axis}'. Available axes: {tuple(mesh.shape.keys())}."
    )
  return int(mesh.shape[axis])


def _active_mesh_axes(mesh: jax.sharding.Mesh) -> dict[str, int]:
  return {axis: int(size) for axis, size in mesh.shape.items() if int(size) > 1}


def select_te_ep_outer_axis(mesh: jax.sharding.Mesh) -> str | None:
  """Pick the single non-EP outer mesh axis for TE EP v1.

  Preference order: ``fsdp`` > ``data``. Returns ``None`` when neither is
  present in the mesh (pure-EP, no DP/FSDP).
  """
  if "fsdp" in mesh.shape:
    return "fsdp"
  if "data" in mesh.shape:
    return "data"
  return None


def _validate_v1_mesh(mesh: jax.sharding.Mesh, outer_axis: str | None) -> None:
  """v1 only supports the expert axis plus one DP/FSDP outer axis being active."""
  active_axes = _active_mesh_axes(mesh)
  allowed = {_TE_EP_AXIS}
  if outer_axis is not None:
    allowed.add(outer_axis)
  unsupported = {axis: size for axis, size in active_axes.items() if axis not in allowed}
  if unsupported:
    raise ValueError(
        "use_te_ep=True v1 supports only the expert axis plus one outer data/FSDP axis. "
        f"Unsupported active mesh axes: {unsupported}."
    )


def _build_mesh_resource(outer_axis: str | None, ep_axis: str) -> Any:
  """Build a MeshResource for TE EP bootstrap.

  Sets ``fsdp_resource`` + ``ep_resource``; leaves ``tp_resource`` / ``cp_resource``
  / ``dp_resource`` unset to match :func:`maxtext.utils.max_utils.transformer_engine_context`
  under ``use_te_ep=true``. TE's ``_validate_mesh_resource_configuration`` calls
  ``get_mesh_axis_size`` on every set resource, which asserts when the named
  axis is missing from the active JAX mesh (e.g. inside ``jax.eval_shape``).
  The TE EP validator already gates TP=CP=1 so stripping those is harmless.
  """
  from transformer_engine.jax.sharding import MeshResource  # pylint: disable=import-outside-toplevel

  kwargs: dict[str, Any] = {
      "fsdp_resource": "fsdp",
      "ep_resource": ep_axis,
  }
  if outer_axis == "data":
    kwargs["dp_resource"] = "data"
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


def _max_tokens_per_rank(config: Any, leading_axis_size: int) -> int:
  global_tokens = int(config.micro_batch_size_to_train_on * config.max_target_length)
  return max(1, math.ceil(global_tokens / max(1, leading_axis_size)))


def _hidden_dim(config: Any) -> int:
  return int(config.moe_expert_input_dim if config.moe_expert_input_dim > 0 else config.emb_dim)


def _needs_v1_tail_absorb() -> bool:
  """True when local GPUs use TE's V1 GroupedQuantizeFFI path for MXFP8.

  V1 enforces ``sum(group_sizes) == m || sum == input_dims[0]`` (see
  ``transformer_engine/jax/csrc/extensions/quantization.cpp:385``). The V2
  path used on sm_100+ (Blackwell) doesn't have this assertion. Our path-C1
  variable-block layout has ``sum(padded_per_expert) < recv_capacity``, so
  on sm_90 we must absorb the unused tail into the last expert's group to
  satisfy the V1 assertion. On sm_100+ we skip tail-absorption to avoid the
  ~17% perf cost (extra GMM rows + non-uniform group_sizes scheduling).
  """
  try:
    from transformer_engine.jax.cpp_extensions.misc import (  # pylint: disable=import-outside-toplevel
        get_min_device_compute_capability,
    )
  except ImportError:
    # If TE isn't importable here we can't be using TE EP either; safe default.
    return False
  try:
    return int(get_min_device_compute_capability()) < 100
  except Exception:  # noqa: BLE001 - resilient: any failure → safe default
    return False


def build_te_ep_state(config: Any, mesh: jax.sharding.Mesh) -> TeEpState:
  """Build the TE EP state without mutating the process singleton.

  Pure function; tests can call this without triggering ``ep_bootstrap``.
  """
  ep_size = _mesh_axis_size(mesh, _TE_EP_AXIS)
  outer_axis = select_te_ep_outer_axis(mesh)
  outer_size = _mesh_axis_size(mesh, outer_axis) if outer_axis is not None else 1
  _validate_v1_mesh(mesh, outer_axis)

  if int(config.num_experts) % ep_size != 0:
    raise ValueError(
        f"num_experts ({config.num_experts}) must be divisible by TE EP size ({ep_size})."
    )

  num_local_experts = int(config.num_experts) // ep_size
  min_dispatch_alignment = int(config.moe_permutation_group_align_size)
  expected_world_size = outer_size * ep_size
  max_tokens_per_rank = _max_tokens_per_rank(config, expected_world_size)
  recv_capacity_per_rank = calculate_te_ep_capacity(
      max_tokens_per_rank=max_tokens_per_rank,
      ep_size=ep_size,
      num_experts=int(config.num_experts),
      top_k=int(config.num_experts_per_tok),
      num_local_experts=num_local_experts,
      recv_capacity_factor=float(config.te_ep_recv_capacity_factor),
      dispatch_alignment=min_dispatch_alignment,
  )
  # dispatch_alignment is the *small* per-expert padding granularity (typically
  # 128 = moe_permutation_group_align_size). The recv buffer holds variable-sized
  # per-expert blocks of `ceil(token_counts[k] / dispatch_alignment) * dispatch_alignment`
  # rows; the MoE GMM consumer computes those padded counts from `token_counts`
  # and uses them as `group_sizes` (mirroring HybridEP/DeepEP's pattern).
  dispatch_alignment = max(1, min_dispatch_alignment)
  needs_v1_tail_absorb = _needs_v1_tail_absorb()

  leading_spec: Any = (outer_axis, _TE_EP_AXIS) if outer_axis is not None else _TE_EP_AXIS
  config_key = (
      _TE_EP_AXIS,
      outer_axis,
      outer_size,
      ep_size,
      expected_world_size,
      int(config.num_experts),
      num_local_experts,
      max_tokens_per_rank,
      recv_capacity_per_rank,
      dispatch_alignment,
      _hidden_dim(config),
      int(config.te_ep_max_num_sms),
      int(config.te_ep_em_unfused_num_sms),
      needs_v1_tail_absorb,
  )

  return TeEpState(
      mesh_resource=_build_mesh_resource(outer_axis, _TE_EP_AXIS),
      ep_axis=_TE_EP_AXIS,
      outer_axis=outer_axis,
      outer_size=outer_size,
      ep_size=ep_size,
      expected_world_size=expected_world_size,
      num_experts=int(config.num_experts),
      num_local_experts=num_local_experts,
      max_tokens_per_rank=max_tokens_per_rank,
      recv_capacity_per_rank=recv_capacity_per_rank,
      dispatch_alignment=dispatch_alignment,
      hidden_dim=_hidden_dim(config),
      max_num_sms=int(config.te_ep_max_num_sms),
      em_unfused_num_sms=int(config.te_ep_em_unfused_num_sms),
      needs_v1_tail_absorb=needs_v1_tail_absorb,
      input_spec_2d=PartitionSpec(leading_spec, None),
      input_spec_3d=PartitionSpec(leading_spec, None, None),
      ep_spec_2d=PartitionSpec(leading_spec, None),
      ep_spec_3d=PartitionSpec(leading_spec, None, None),
      config_key=config_key,
  )


def init_te_ep_for_maxtext(config: Any, mesh: jax.sharding.Mesh) -> TeEpState:
  """Bootstrap TE NCCL EP exactly once per process.

  Must be called before ``setup_train_loop`` — model creation traces
  ``moe.py`` which dispatches into ``ep_dispatch``; the process-singleton
  must be live by then. Idempotent for matching ``config_key``; raises on
  shape/resource mismatch.
  """
  global _TE_EP_STATE

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
        "TE EP v1 expects one JAX process per active fsdp/expert mesh slot. "
        f"process_count={world_size}, expected={candidate.expected_world_size}, "
        f"outer_axis={candidate.outer_axis}, ep_size={candidate.ep_size}."
    )

  from transformer_engine.jax.ep import ep_bootstrap  # pylint: disable=import-outside-toplevel
  from transformer_engine.jax.sharding import global_shard_guard  # pylint: disable=import-outside-toplevel

  with mesh, jax.set_mesh(mesh), global_shard_guard(candidate.mesh_resource):
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
    )

  _TE_EP_STATE = candidate
  max_logging.log(
      "TE EP initialized: "
      f"outer_axis={candidate.outer_axis}, ep_axis={candidate.ep_axis}, "
      f"ep_size={candidate.ep_size}, outer_size={candidate.outer_size}, "
      f"max_tokens_per_rank={candidate.max_tokens_per_rank}, "
      f"recv_capacity_per_rank={candidate.recv_capacity_per_rank}, "
      f"dispatch_alignment={candidate.dispatch_alignment}, "
      f"max_num_sms={candidate.max_num_sms}, em_unfused_num_sms={candidate.em_unfused_num_sms}"
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
