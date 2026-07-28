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

"""Tests for max_utils.transformer_engine_context MeshResource composition.

Verifies that:
  * Without use_te_ep: ep_resource is None; dp_resource="data".
  * With use_te_ep=True: ep_resource="expert"; tp/dp/cp stay unset in the
    outer context so eval_shape does not validate resources before a mesh exists.
  * With TE EP ETP1: the outer context remains dense-compatible; EP's
    expert-DP view is local to EP operations.
"""

from types import SimpleNamespace
import unittest

from maxtext.utils import max_utils


class TransformerEngineContextTest(unittest.TestCase):

  def test_without_te_ep(self):
    config = SimpleNamespace(use_te_ep=False)
    resources = max_utils._transformer_engine_mesh_resource_kwargs(config)
    self.assertEqual(resources["dp_resource"], "data")
    self.assertEqual(resources["tp_resource"], "tensor")
    self.assertEqual(resources["fsdp_resource"], "fsdp")
    self.assertEqual(resources["cp_resource"], "context")
    self.assertIsNone(resources["ep_resource"])

  def test_with_te_ep_strips_tp_cp_dp(self):
    """The outer TE EP context keeps only fsdp/expert resources active."""
    config = SimpleNamespace(use_te_ep=True)
    resources = max_utils._transformer_engine_mesh_resource_kwargs(config)
    self.assertEqual(resources["ep_resource"], "expert")
    self.assertEqual(resources["fsdp_resource"], "fsdp")
    self.assertIsNone(resources["tp_resource"])
    self.assertIsNone(resources["cp_resource"])
    self.assertIsNone(resources["dp_resource"])

  def test_with_te_ep_and_ici_tp_still_strips_outer_tp(self):
    """TP is supplied by the MoE wrapper's mesh-scoped guard, not this context."""
    config = SimpleNamespace(use_te_ep=True, ici_tensor_parallelism=2)
    resources = max_utils._transformer_engine_mesh_resource_kwargs(config)
    self.assertEqual(resources["ep_resource"], "expert")
    self.assertEqual(resources["fsdp_resource"], "fsdp")
    self.assertIsNone(resources["tp_resource"])
    self.assertIsNone(resources["cp_resource"])
    self.assertIsNone(resources["dp_resource"])

  def test_with_te_ep_etp1_keeps_dense_compatible_outer_context(self):
    config = SimpleNamespace(
        use_te_ep=True,
        ici_tensor_parallelism=2,
        te_ep_expert_tensor_parallelism=1,
    )
    resources = max_utils._transformer_engine_mesh_resource_kwargs(config)
    self.assertEqual(resources["ep_resource"], "expert")
    self.assertIsNone(resources["dp_resource"])
    self.assertIsNone(resources["tp_resource"])
    self.assertEqual(resources["fsdp_resource"], "fsdp")
    self.assertIsNone(resources["cp_resource"])

  def test_none_config_defaults_to_no_te_ep(self):
    resources = max_utils._transformer_engine_mesh_resource_kwargs(None)
    self.assertEqual(resources["dp_resource"], "data")
    self.assertIsNone(resources["ep_resource"])


if __name__ == "__main__":
  unittest.main()
