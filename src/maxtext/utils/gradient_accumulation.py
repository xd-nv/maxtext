# Copyright 2025-2026 Google LLC
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

"""Functions for gradient accumulation (GA)"""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from maxtext.common.common_types import ShardMode
from maxtext.utils.sharding import maybe_shard_with_name
from maxtext.utils.maxtext_utils import OVERWRITE_WITH_GRADIENT


def load_schedule(config):
  """Load the JaxPP pipeline schedule based on config.schedule.

  Only called when config.use_jaxpp is True; jaxpp is an optional dependency
  (see [[jaxpp_code_changes]]), so the import is lazy.
  """
  import jaxpp.api as jaxpp  # pylint: disable=import-outside-toplevel
  from jaxpp import __version__ as jaxpp_version  # pylint: disable=import-outside-toplevel
  from packaging.version import Version

  pipeline_parallel_dim = config.dcn_pipeline_parallelism * config.ici_pipeline_parallelism
  num_logical_stages = config.num_pipeline_repeats * pipeline_parallel_dim
  if config.schedule == "1f1b":
    assert num_logical_stages <= pipeline_parallel_dim
    schedule = jaxpp.Std1F1B(num_logical_stages)
  elif config.schedule == "eager_1f1b":
    assert num_logical_stages <= pipeline_parallel_dim
    schedule = jaxpp.Eager1F1B(num_logical_stages)
  elif config.schedule == "interleaved_1f1b":
    if Version(jaxpp_version) > Version("0.6.1"):
      schedule = jaxpp.Interleaved1F1B(num_logical_stages, pipeline_parallel_dim, config.fuse_steady_state)
    else:
      schedule = jaxpp.Interleaved1F1B(num_logical_stages, pipeline_parallel_dim)
  elif config.schedule == "kimik2":
    schedule = jaxpp.KimiK2(num_logical_stages, pipeline_parallel_dim, config.fuse_steady_state)
  elif config.schedule == "zero_bubble":
    assert num_logical_stages <= pipeline_parallel_dim
    schedule = jaxpp.ZeroBubble(num_logical_stages)
  elif config.schedule == "dualpipev":
    schedule = jaxpp.DualPipeV(num_logical_stages, pipeline_parallel_dim)
  else:
    raise NotImplementedError(f"Unknown schedule {config.schedule}")
  return schedule


def add_leading_axis(axis_name: str, path: jax.tree_util.KeyPath, s: jax.sharding.NamedSharding):
  """Add a leading axis to a NamedSharding for vmap."""
  assert isinstance(s, jax.sharding.NamedSharding)
  used = {n for ns in s.spec for n in (ns if isinstance(ns, tuple) else (ns,))}
  if axis_name in used:
    raise ValueError(
        f"mesh axis name {axis_name} cannot appear in "
        f"out_shardings. Found out_shardings{jax.tree_util.keystr(path)}={s.spec}"
    )
  return jax.sharding.NamedSharding(s.mesh, jax.sharding.PartitionSpec(axis_name, *s.spec), memory_kind=s.memory_kind)


def gradient_accumulation_loss_and_grad(
    _loss_fn,
    config,
    model,
    params,
    params_shardings,
    data,
    dropout_rng,
    extra_dpo_args,
):
  """
  Calculates gradients using gradient accumulation.

  This function computes the gradient of `_loss_fn` over multiple microbatches
  and accumulates them before returning a single, averaged gradient. It uses
  `jax.lax.scan` for efficient accumulation on device.

  It also supports a `shard_optimizer_over_data` mode (e.g., ZeRO-1) where
  parameters are cast to bf16 and sharded *before* the accumulation loop
  to perform the all-gather in lower precision.

  Args:
      _loss_fn: The loss function to differentiate. Its signature is expected
          to be: `(model, config, data, dropout_rng, params, *extra_args, is_train=True)`.
      config: Model and training configuration object. Must contain
          `gradient_accumulation_steps` and `shard_optimizer_over_data`.
      model: The model module.
      params: The model parameters (PyTree).
      params_shardings: The sharding constraints for the parameters (PyTree).
      data: A PyTree of batched data. The leading dimension is assumed
          to be the total batch size (microbatch_size * num_accumulations).
      dropout_rng: JAX PRNGKey for dropout.
      extra_dpo_args: A tuple of extra arguments to pass to the loss function.

  Returns:
      A tuple containing:
      - total_loss (Array): The mean loss, averaged over all microbatches.
      - final_aux (PyTree): Auxiliary outputs, summed across microbatches.
      - raw_grads (PyTree): The accumulated and averaged gradients.
  """

  def _maybe_shard_with_name(inputs, sharding_names):
    """Wrapper of maybe_shard_with_name with fixed shard_mode"""
    return maybe_shard_with_name(inputs, sharding_names, config.shard_mode, debug_sharding=config.debug_sharding)

  # For more efficient DP/ZeRO-1 + GA
  if config.shard_mode == ShardMode.EXPLICIT and config.ici_data_parallelism > 1:
    ga_params_shardings = jax.tree.map(update_sharding_for_reduced, params_shardings)
    grad_shardings = jax.tree.map(update_sharding_for_unreduced, params_shardings)
  else:
    ga_params_shardings = grad_shardings = params_shardings
  # When using Zero-1 optimizer sharding, cast params to lower precision and apply sharding constraints
  # so that all-gather is done once in the lower precision before the gradient accumulation loop
  if config.shard_optimizer_over_data:

    def convert_to_bf16(param):
      if param.dtype == jnp.float32:
        return param.astype(jnp.bfloat16)
      return param

    if not config.use_jaxpp:
      ga_params = jax.tree_util.tree_map(convert_to_bf16, params)
    else:
      # TODO(jaxpp): bf16 pre-cast is not yet supported together with treduce.
      ga_params = params
  else:
    ga_params = params

  ga_params = jax.tree.map(_maybe_shard_with_name, ga_params, ga_params_shardings)
  grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)

  if not config.use_jaxpp:
    # Standard scan-based gradient accumulation.
    def accumulate_gradient(acc_grad_and_loss, data):
      ga_params = acc_grad_and_loss["ga_params"]
      (_, aux), cur_batch_gradient = grad_func(
          model, config, data, dropout_rng, ga_params, *extra_dpo_args, is_train=True
      )
      acc_grad_and_loss["loss"] += aux["xent_sum"] + aux.get("dpo_loss", 0.0)
      acc_grad_and_loss["moe_lb_loss"] += aux["moe_lb_loss"]
      acc_grad_and_loss["indexer_loss"] += aux["indexer_loss"]
      acc_grad_and_loss["mtp_loss"] += aux["mtp_loss"]
      acc_grad_and_loss["grad"] = jax.tree_util.tree_map(
          lambda x, y: x + y, cur_batch_gradient, acc_grad_and_loss["grad"]
      )
      acc_grad_and_loss["total_weights"] += aux["total_weights"]
      return acc_grad_and_loss, aux

    def reshape_to_microbatch_accumulations(batch_arr):
      """Reshape global batch to microbatches, assuming batch axis is leading."""
      num_microbatches = config.gradient_accumulation_steps
      microbatch_shape = (batch_arr.shape[0] // num_microbatches, num_microbatches) + batch_arr.shape[1:]
      reshaped_batch_arr = jnp.reshape(batch_arr, microbatch_shape)
      return jnp.swapaxes(reshaped_batch_arr, 0, 1)

    data = jax.tree_util.tree_map(reshape_to_microbatch_accumulations, data)
    init_grad = jax.tree_util.tree_map(jnp.zeros_like, ga_params)
    init_grad = jax.tree.map(_maybe_shard_with_name, init_grad, grad_shardings)
    init_grad_and_loss = {
        "loss": 0.0,  # accumulates xent_sum across microbatches
        "grad": init_grad,
        "total_weights": 0,
        "moe_lb_loss": 0.0,
        "indexer_loss": 0.0,
        "mtp_loss": 0.0,
        "ga_params": ga_params,
    }

    grad_and_loss, aux = jax.lax.scan(
        accumulate_gradient, init_grad_and_loss, data, length=config.gradient_accumulation_steps
    )
    loss = (
        grad_and_loss["loss"] / grad_and_loss["total_weights"]
        + grad_and_loss["moe_lb_loss"] / config.gradient_accumulation_steps
        + grad_and_loss["indexer_loss"] / config.gradient_accumulation_steps
        + grad_and_loss["mtp_loss"] / config.gradient_accumulation_steps
    )
    raw_grads = grad_and_loss["grad"]
    raw_grads = jax.tree.map(_maybe_shard_with_name, raw_grads, params_shardings)
    raw_grads = jax.tree_util.tree_map(lambda arr: arr / grad_and_loss["total_weights"], raw_grads)
    aux = jax.tree.map(lambda x: jnp.sum(x, axis=0), aux)  # pytype: disable=module-attr

  else:
    # JaxPP treduce-based gradient accumulation with pipeline parallelism.
    import jaxpp.api as jaxpp  # pylint: disable=import-outside-toplevel

    mesh = jax.tree_util.tree_leaves(params_shardings)[0].mesh

    def compute_grads(data):
      """Compute gradients for a single microbatch."""
      if OVERWRITE_WITH_GRADIENT in ga_params:

        def place_owg(path, value):
          params = ga_params["params"]
          for key in path:
            if isinstance(key, jax.tree_util.DictKey) and key.key in params:
              params = params[key.key]
            else:
              return jaxpp.place_with(value, jax.tree.leaves(params)[0])
          assert False, "Should not reach here"

        ga_params[OVERWRITE_WITH_GRADIENT] = jax.tree.map_with_path(
            place_owg, ga_params[OVERWRITE_WITH_GRADIENT]
        )

      (loss, aux), raw_grads = grad_func(model, config, data, dropout_rng, ga_params, *extra_dpo_args, is_train=True)
      return (loss, aux), raw_grads

    def microbatched(a):
      """Reshape data for pipeline microbatching with optional data parallelism."""
      shape = (
          mesh.shape["data"],
          config.num_pipeline_microbatches,
          -1,
          config.max_target_length,
      )
      if shape[0] == 1:
        shape = shape[1:]
      out_sharding = None
      if config.shard_mode == ShardMode.EXPLICIT:
        out_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", "expert", None))
      return a.reshape(*shape, out_sharding=out_sharding)

    data = jax.tree.map(microbatched, data)
    data = _maybe_shard_with_name(data, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", "expert", None)))

    # Perform data parallelism manually through vmap.
    vmapped_compute_grads = compute_grads
    if mesh.shape["data"] > 1:
      vmapped_compute_grads = jax.vmap(compute_grads, spmd_axis_name="data")

    loss_aux_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    param_operation = {"params": jaxpp.Add}
    # FP8 quantization parameters use Max accumulation (not Add) across microbatches
    # per Flax documentation: amax values should be maxed, not summed.
    if OVERWRITE_WITH_GRADIENT in ga_params:
      param_operation[OVERWRITE_WITH_GRADIENT] = jaxpp.Max
    assert all(k in param_operation for k in ga_params.keys())

    axis = 1 if mesh.shape["data"] > 1 else 0
    (loss, aux), raw_grads = jaxpp.treduce(
        vmapped_compute_grads,
        data,
        axis=axis,
        schedule=load_schedule(config),
        operation=(jaxpp.Concat(axis=axis), param_operation),
    )

    if mesh.shape["data"] > 1:
      (loss, aux), raw_grads = jax.lax.with_sharding_constraint(
          ((loss, aux), raw_grads),
          jax.tree.map_with_path(
              functools.partial(add_leading_axis, "data"),
              (loss_aux_sharding, params_shardings),
          ),
      )
      # reduce-scatter gradients across "data"
      raw_grads = jax.tree.map(functools.partial(jnp.sum, axis=0), raw_grads)

    # Aggregate metrics similar to the scan path.
    total_weights = aux["total_weights"]
    total_weights = total_weights.sum() if mesh.shape["data"] > 1 else jnp.sum(total_weights)

    # TODO(jaxpp): normalize by total_weights and add moe_lb_loss/mtp_loss/indexer_loss,
    # matching the scan path in the non-jaxpp branch above.
    loss = loss.sum()

    raw_grads = jax.tree.map(_maybe_shard_with_name, raw_grads, params_shardings)

    # Sum aux across microbatches if needed.
    aux = jax.tree.map(lambda x: jnp.sum(x, axis=0) if x.ndim > 0 else x, aux)

  return loss, aux, raw_grads


# GA helper functions
def update_sharding_for_reduced(sharding: NamedSharding) -> NamedSharding:
  """
  Add reduced on data axis of given NamedSharding
  """
  return sharding.update(spec=sharding.spec.update(reduced={"data"}))


def update_sharding_for_unreduced(sharding: NamedSharding) -> NamedSharding:
  """
  Add unreduced on data axis of given NamedSharding
  """
  return sharding.update(spec=sharding.spec.update(unreduced={"data"}))
