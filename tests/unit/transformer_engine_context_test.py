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
"""

from types import SimpleNamespace
import unittest

try:
  from transformer_engine.jax.sharding import _GLOBAL_MESH_RESOURCE  # noqa: F401
  _HAS_TE = True
except ImportError:
  _HAS_TE = False

from maxtext.utils import max_utils


@unittest.skipUnless(_HAS_TE, "transformer_engine is required")
class TransformerEngineContextTest(unittest.TestCase):

  def _active_resource(self):
    import transformer_engine.jax.sharding as te_sharding  # pylint: disable=import-outside-toplevel

    return te_sharding._GLOBAL_MESH_RESOURCE

  def test_without_te_ep(self):
    config = SimpleNamespace(use_te_ep=False)
    with max_utils.transformer_engine_context(config):
      mr = self._active_resource()
      self.assertEqual(mr.dp_resource, "data")
      self.assertEqual(mr.tp_resource, "tensor")
      self.assertEqual(mr.fsdp_resource, "fsdp")
      self.assertEqual(mr.cp_resource, "context")
      self.assertIsNone(mr.ep_resource)

  def test_with_te_ep_strips_tp_cp_dp(self):
    """The outer TE EP context keeps only fsdp/expert resources active."""
    config = SimpleNamespace(use_te_ep=True)
    with max_utils.transformer_engine_context(config):
      mr = self._active_resource()
      self.assertEqual(mr.ep_resource, "expert")
      self.assertEqual(mr.fsdp_resource, "fsdp")
      self.assertIsNone(mr.tp_resource)
      self.assertIsNone(mr.cp_resource)
      self.assertIsNone(mr.dp_resource)

  def test_with_te_ep_and_ici_tp_still_strips_outer_tp(self):
    """TP is supplied by the MoE wrapper's mesh-scoped guard, not this context."""
    config = SimpleNamespace(use_te_ep=True, ici_tensor_parallelism=2)
    with max_utils.transformer_engine_context(config):
      mr = self._active_resource()
      self.assertEqual(mr.ep_resource, "expert")
      self.assertEqual(mr.fsdp_resource, "fsdp")
      self.assertIsNone(mr.tp_resource)
      self.assertIsNone(mr.cp_resource)
      self.assertIsNone(mr.dp_resource)

  def test_none_config_defaults_to_no_te_ep(self):
    with max_utils.transformer_engine_context(None):
      mr = self._active_resource()
      self.assertEqual(mr.dp_resource, "data")
      self.assertIsNone(mr.ep_resource)


if __name__ == "__main__":
  unittest.main()
