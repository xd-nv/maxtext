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

"""Unit tests for the TE EP MaxText bootstrap helpers (pure-Python parts).

The TE NCCL EP and global_shard_guard imports inside
:func:`init_te_ep_for_maxtext` are lazy, so this test file does not require
``transformer_engine`` to be installed.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from maxtext.layers import te_ep_init


class CalculateTeEpCapacityTest(unittest.TestCase):
  """Worst-case capacity math; no GPU / TE needed."""

  def test_aligned(self):
    capacity = te_ep_init.calculate_te_ep_capacity(
        max_tokens_per_rank=16,
        ep_size=4,
        num_experts=8,
        top_k=2,
        num_local_experts=2,
        recv_capacity_factor=1.0,
        dispatch_alignment=128,
    )
    # T_per_ep_group=64, target=64*2=128. recv = ceil_to((128 + 2*128), 128) = 384.
    self.assertEqual(capacity, 384)

  def test_factor_before_alignment(self):
    capacity = te_ep_init.calculate_te_ep_capacity(
        max_tokens_per_rank=16,
        ep_size=4,
        num_experts=8,
        top_k=2,
        num_local_experts=2,
        recv_capacity_factor=3.0,
        dispatch_alignment=128,
    )
    # target = ceil(128 * 3) = 384. recv = ceil_to((384 + 2*128), 128) = 640.
    self.assertEqual(capacity, 640)

  def test_dsv3_671b_bs2_rcf1(self):
    """DSV3 671B at bs=2 rcf=1.0: production sizing under HybridEP-style padding.

    Regression test for path-C1 fix (dispatch_alignment passed to TE = 128, not
    slots_per_expert=4096). Earlier formulas that forced uniform per-expert blocks
    either overflowed (NLE*slots=262144) or had 2x memory inflation. This formula
    sizes the buffer to the routed-token total plus one NLE*align headroom block.
    """
    capacity = te_ep_init.calculate_te_ep_capacity(
        max_tokens_per_rank=8192,
        ep_size=4,
        num_experts=256,
        top_k=8,
        num_local_experts=64,
        recv_capacity_factor=1.0,
        dispatch_alignment=128,
    )
    # target = 8192*4*8 = 262144. recv = ceil_to((262144 + 64*128), 128) = 270336.
    self.assertEqual(capacity, 270336)

  def test_dsv3_671b_bs2_rcf2(self):
    """DSV3 671B at bs=2 rcf=2.0 should double the routed-token bound."""
    capacity = te_ep_init.calculate_te_ep_capacity(
        max_tokens_per_rank=8192,
        ep_size=4,
        num_experts=256,
        top_k=8,
        num_local_experts=64,
        recv_capacity_factor=2.0,
        dispatch_alignment=128,
    )
    # target = 262144*2 = 524288. recv = ceil_to((524288 + 64*128), 128) = 532480.
    self.assertEqual(capacity, 532480)

  def test_unaligned(self):
    capacity = te_ep_init.calculate_te_ep_capacity(
        max_tokens_per_rank=4,
        ep_size=4,
        num_experts=8,
        top_k=2,
        num_local_experts=2,
        recv_capacity_factor=1.5,
        dispatch_alignment=0,
    )
    # T_per_ep_group=16, worst=max(16*2, 16)*1=32, target=ceil(32*1.5)=48, no alignment.
    self.assertEqual(capacity, 48)

  def test_overconc_active_when_experts_exceed_pool(self):
    # T_per_ep_group * top_k = 1*1 = 1; num_experts=8 → overconc=8.
    capacity = te_ep_init.calculate_te_ep_capacity(
        max_tokens_per_rank=1,
        ep_size=1,
        num_experts=8,
        top_k=1,
        num_local_experts=8,
        recv_capacity_factor=1.0,
        dispatch_alignment=0,
    )
    # active=min(8, 1)=1, overconc=ceil(8/1)=8, worst=max(1, 16)*8=128.
    self.assertEqual(capacity, 128)


class CalculateTeEpPaddedCapacityBoundTest(unittest.TestCase):
  """Static bound for runtime padded token counts."""

  def test_dsv3_671b_bs2_worst_case_bound(self):
    capacity = te_ep_init.calculate_te_ep_padded_capacity_bound(
        max_tokens_per_rank=8192,
        ep_size=4,
        top_k=8,
        num_local_experts=64,
        dispatch_alignment=128,
    )
    # Routed-token bound = 8192*4*8 = 262144. Padding overhead is at most
    # 64*(128-1)=8128, then rounded up to alignment.
    self.assertEqual(capacity, 270336)

  def test_fewer_tokens_than_local_experts(self):
    capacity = te_ep_init.calculate_te_ep_padded_capacity_bound(
        max_tokens_per_rank=1,
        ep_size=1,
        top_k=1,
        num_local_experts=8,
        dispatch_alignment=128,
    )
    # Only one local expert can be active, so the padded bound is one block.
    self.assertEqual(capacity, 128)

  def test_unaligned(self):
    capacity = te_ep_init.calculate_te_ep_padded_capacity_bound(
        max_tokens_per_rank=4,
        ep_size=4,
        top_k=2,
        num_local_experts=2,
        dispatch_alignment=0,
    )
    self.assertEqual(capacity, 32)


class ModuleSurfaceTest(unittest.TestCase):
  """Surface-level checks that don't require TE/JAX to be importable as GPU."""

  def test_te_ep_axis_constant(self):
    self.assertEqual(te_ep_init._TE_EP_AXIS, "expert")

  def test_get_state_before_init_raises(self):
    te_ep_init.reset_te_ep_state_for_test()
    with self.assertRaisesRegex(ValueError, "not been initialized"):
      te_ep_init.get_te_ep_state()


class BuildTeEpStateTest(unittest.TestCase):
  """Pure-Python TE EP state construction checks."""

  def _config(self):
    return SimpleNamespace(
        num_experts=256,
        num_experts_per_tok=8,
        moe_permutation_group_align_size=128,
        te_ep_recv_capacity_factor=1.0,
        micro_batch_size_to_train_on=32,
        max_target_length=2048,
        num_decoder_layers=61,
        first_num_dense_layers=3,
        moe_expert_input_dim=-1,
        emb_dim=7168,
        te_ep_max_num_sms=0,
        te_ep_em_unfused_num_sms=-1,
    )

  def test_ici_tensor_parallelism_updates_world_size_not_token_capacity(self):
    mesh = SimpleNamespace(shape={"fsdp": 2, "expert": 8, "tensor": 2})

    with patch.object(te_ep_init, "_build_mesh_resource", return_value="mesh-resource"):
      state = te_ep_init.build_te_ep_state(self._config(), mesh)

    self.assertEqual(state.mesh_resource, "mesh-resource")
    self.assertEqual(state.outer_axis, "fsdp")
    self.assertEqual(state.outer_size, 2)
    self.assertEqual(state.ep_size, 8)
    self.assertEqual(state.tensor_axis, "tensor")
    self.assertEqual(state.tensor_size, 2)
    self.assertEqual(state.expected_world_size, 2 * 8 * 2)
    # Token routing capacity is partitioned by outer*EP only; TP shards hidden dim.
    self.assertEqual(state.max_tokens_per_rank, 4096)
    self.assertEqual(state.routing_spec_2d, te_ep_init.PartitionSpec(("fsdp", "expert"), None))
    self.assertEqual(state.input_spec_2d, te_ep_init.PartitionSpec(("fsdp", "expert"), "tensor"))


if __name__ == "__main__":
  unittest.main()
