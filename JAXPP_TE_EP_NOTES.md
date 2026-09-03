# JaxPP integration notes

Running log of the JaxPP + pipeline-parallelism merge onto this TE-EP branch,
and the follow-on investigation into making `use_te_ep` work under
`use_jaxpp`. Branch: `te-pr3429-nested-gemm-0826_jaxpp`.

## 1. Initial JaxPP port

Ported JaxPP support (config knobs, `pipeline_enter_stage` markers in
`decoders.py`, `MpmdMesh`/train-loop resharding in `train.py`, a
`treduce`-based path in `gradient_accumulation.py`, `third_party/jaxpp`
submodule) from the reference `jaxpp_dev_xiningd_fix` branch (source repo:
`/lustre/fsw/coreai_devtech_all/xiningd/jax/jaxpp/maxtext-jaxpp`). Kept
`jaxpp` an optional, lazily-imported dependency throughout so non-PP
training is unaffected. Deliberately left out of scope: `optimized_ep`/MoE
kernel work, `qwen3_moe` support, extra remat policies, and the reference's
own retuned `base.yml` defaults (hardware/attention/scan_layers/etc.) --
those aren't required for `use_jaxpp` to work, just that branch's own
experiment defaults.

## 2. Launcher wiring

`/lustre/fsw/coreai_devtech_all/xiningd/jax/maxtext-launcher`, branch
`add-jaxpp-pp-support`: wired PP config knobs into `launcher.py`/
`defaults.yaml`/`run.template.sh`, added
`configs/models/deepseek-v3-671b-pp8-15layer-smoke.yaml` (15-layer DSv3,
PP=8 x EP=8, 8 EOS nodes) as the smoke-test recipe. Along the way, caught
and fixed two real bugs in the launcher itself (not this branch): it had no
generic passthrough for arbitrary maxtext config keys -- `base_num_decoder_layers`/
`override_model_config`, then later `named_checkpoint_names`/
`use_batch_split_schedule`, were silently dropped from the generated
training command until added to `launcher.py`'s `moe_fields` allowlist.

## 3. Smoke-test debugging, in order (each got further than the last)

1. **ICI/DCN topology**: `ici_*` parallelism must fit within one node's
   device count (EOS = 8 GPUs/node); PP/EP spanning nodes need `dcn_*`.
   Config fix.
2. **`te_use_gmm` requires `megablox=False`**. Config fix.
3. **`te_use_gmm` requires `quantization` to start with `"te_"`**. Config
   fix.
4. **`use_te_ep` + `use_jaxpp` incompatible**: `te_ep_init.py`'s
   `init_te_ep_for_maxtext` unconditionally rejects
   `using_pipeline_parallelism` and requires `scan_layers=True`. Real,
   pre-existing gap, not a config issue -- switched the smoke test to
   `use_ring_of_experts` instead to keep validating JaxPP's own mechanics.
   Section 5 below is the real investigation into fixing this properly.
5. **JaxPP + `dcn_data_parallelism > 1`**: `gradient_accumulation.py`'s
   jaxpp path wraps `compute_grads` in `jax.vmap(..., spmd_axis_name="data")`,
   which conflicts with a `with_sharding_constraint` inside MoE that still
   mentions `"data"`. Worked around by using `dcn_fsdp_parallelism` instead
   for the smoke test's extra parallelism factor; **not fixed** as a
   general jaxpp+DP capability.
6. **`shard_map` mesh mismatch in `moe.py`'s `wrapper()`** (ring-of-experts
   dispatch): `mesh=self.mesh` (captured at model-construction time, full
   physical mesh) didn't match the ambient per-stage context JaxPP
   established during tracing. Two wrong fix attempts here (both reverted):
   `jax.sharding.get_abstract_mesh()` (returned empty -- this code traces
   via `jax.interpreters.partial_eval.trace_to_jaxpr_dynamic` inside
   `jaxpp/training.py`'s `pscan_wrapped`, which doesn't propagate the
   ambient `jax.set_mesh()` thread-local the way `jax.jit` does) and
   `jax.typeof(inputs).sharding.mesh` (returned the same stale full mesh as
   `self.mesh`).
7. **Root cause of #6, found by comparing directly against the reference
   branch's `maxtext_utils.py`**: `state`'s sharding was always built
   against the *full* mesh in this branch's port, because `MpmdMesh` was
   constructed in `train.py::train_loop` *after* `setup_train_loop` (and
   therefore `setup_initial_state`/`get_abstract_state`) had already run.
   The reference narrows `mesh` to `maybe_mpmd_mesh.lowering_mesh()`
   *before* `get_abstract_state`, so `state_mesh_shardings` is built
   against the narrowed, per-stage-local mesh from the start. **Fixed**:
   moved `MpmdMesh` construction into `train_utils.py::setup_train_loop`,
   right after `mesh = model.mesh` and before optimizer/checkpoint/data/state
   setup, so everything downstream (including `get_abstract_state`) sees
   the narrowed mesh consistently. `setup_train_loop` now returns
   `mpmd_mesh` as an extra tuple element; `train.py` no longer constructs
   its own. This got the smoke test all the way past mesh setup into
   JaxPP's own task-clustering pass -- a qualitatively different, much
   deeper failure point than anything reached before.
8. **JaxPP's own `AssertionError: Failed on loop body jaxpr`**
   (`jaxpp/core.py::cluster_jaxpr`, gated by
   `JAXPP_CONSERVATIVE_LOOP_CLUSTERING`, default on): some equation inside
   the microbatch-loop body can't be cleanly assigned to one pipeline
   stage. Tried matching the user's confirmed-working jaxpp/dev PP=8/EP=8
   config exactly (`schedule=interleaved_1f1b`, `num_pipeline_microbatches=32`,
   `remat_policy=named_checkpoint`, `per_device_batch_size=4`,
   `fuse_steady_state=true`) -- **same error persists**, ruling out those
   config differences as the cause.

## 4. Why `use_ring_of_experts` hits #8 but the reference's `optimized_ep`-based run doesn't

Compared `moe.py`'s `wrapper()` (ring-of-experts path, what we're using)
against `optimized_ep`'s internal dispatch
(`experimental/xtreme/deepseek_moe_stripped.py`). Both use `jax.shard_map`
-- the difference is `wrapper()` passes explicit, static `in_specs=(...)`
computed once outside the call, baking a rigid mesh declaration into the
`shard_map_p` jaxpr equation. `optimized_ep`'s internal `ring_of_experts`/
`ep_ffn` shard_map calls omit `in_specs` entirely and let JAX infer
partitioning from the arguments' already-attached sharding -- a more
dynamic, "follow the data" approach that doesn't need a separate rewrite
pass to reconcile against JaxPP's evolving per-stage context.

Checked mainline `AI-Hypercomputer/maxtext` (upstream, via a throwaway
shallow clone) as a third reference point: its own MoE `shard_map` call
uses the *same* explicit-`in_specs`/`mesh=self.mesh` pattern we do. So this
isn't a TE-EP-branch deviation from a "correct" pattern -- it's the
standard MaxText way to write this, just one that happens to collide with
JaxPP's per-stage mesh handling. The no-`in_specs` pattern is confined to
the experimental `optimized_ep`/`xtreme` code path.

**Not pursued further**: the user's actual interest is `use_te_ep`, not
`use_ring_of_experts` -- and TE_EP's dispatch (`ep_dispatch`/`ep_combine`)
already avoids `shard_map` for unrelated reasons (see section 5), so this
whole clustering investigation may not even be relevant to the path that
actually matters. Redirected effort there instead.

## 5. `use_te_ep` + `use_jaxpp`: what's actually needed

### 5a. The PP ban isn't arbitrary -- it's baked into a global world-size assumption

`te_ep_init.py::init_te_ep_for_maxtext`:
```python
world_size = jax.process_count()   # GLOBAL: whole SLURM job
rank = jax.process_index()
if world_size != candidate.expected_world_size:  # computed purely from EP-related mesh axes, no "stage" factor
    raise ValueError(...)
```
Under a flat SPMD job (no PP), global process count naturally equals the
EP collective size. Under JaxPP MPMD, the job has `PP_degree` independent
per-stage process groups, so global process count =
`PP_degree x per-stage-EP-size` -- structurally can never equal
`expected_world_size`. This is *why* `using_pipeline_parallelism` is
unconditionally rejected, not an arbitrary flag.

`MpmdMesh` (in `third_party/jaxpp/src/jaxpp/mesh.py`) already exposes what's
needed to compute stage-local values instead: `device_mpmd_idx` (device ->
stage index), `unstack[stage_idx]` (a plain per-stage `Mesh`), and
`lowering_mesh()` (== `unstack[my own stage]`, already threaded through as
`model.mesh` per fix #7 above).

### 5b. The real blocker: `_allgather_uid` is hardcoded to whole-job scope

Traced into `transformer_engine.jax.ep.ep_bootstrap` via this repo's local
overlay copy (`src/maxtext/te_overlay/transformer_engine/jax/ep.py` --
vendored for local TE testing, see `te_overlay_dir` launcher config below).
Its UID-broadcast helper:
```python
def _allgather_uid(uid_arr, world_size, uid_size):
    try:
        gathered = jmu.process_allgather(uid_arr, tiled=True)  # whole-job coordination service call
        if gathered.size == world_size * uid_size:
            return ...
    except Exception:
        pass
    devices = np.asarray(jax.devices())  # ALL devices in the entire job, not mesh-scoped
    if devices.size != world_size:
        raise RuntimeError(...)
    ...
```
Both paths require `world_size` to equal the *entire distributed job's*
device count. There's no way to pass a smaller, stage-local `world_size`
and have it correctly scope to just one stage's processes -- it either
silently fails the primary path's size check or hits the fallback's
`RuntimeError` outright. This is a real, structural limitation in TE's
current EP bootstrap code (third-party, vendored here), not something
fixable by changing call-site arguments.

### 5c. But JAX's distributed runtime *does* support scoped rendezvous -- JaxPP already uses it

`third_party/jaxpp/src/jaxpp/dime2.py::get_nccl_id`, used by JaxPP for its
own inter-stage NCCL communicators:
```python
class UniqueDevices(tuple):
    @cached_property
    def key(self) -> str:
        return ",".join(str(d.id) for d in self)  # deterministic key from an arbitrary device subset

def get_nccl_id(devs: UniqueDevices) -> UniqueId:
    if devs.leader.process_index == jax.process_index():
        uid = get_unique_id()
        get_distributed_client().key_value_set_bytes(devs.key, uid.as_bytes)
        return uid
    raw = get_distributed_client().blocking_key_value_get_bytes(devs.key, TIMEOUT)
    return UniqueId.from_bytes(raw)
```
`get_distributed_client()` is JAX's distributed-runtime coordination-service
client (the same one `jax.distributed.initialize()` sets up) -- a general
put/get key-value store that works for *any* subset of processes agreeing
on the same key, not just "all processes." This is exactly the primitive
needed to fix `_allgather_uid` for a stage-scoped EP bootstrap: replace the
whole-job gather with a keyed put/get scoped to the current stage's
device-ID list, mirroring `get_nccl_id`'s pattern.

This means the fix is local and implementable (not blocked on upstream TE),
though it's a correctness-sensitive patch to vendored NCCL bootstrap code,
not ordinary glue code.

### Testing mechanism already exists

The launcher (`maxtext-launcher/configs/defaults.yaml`, `te_overlay_dir`)
already supports overlaying local TE source edits into the container at
job start, specifically for testing local TE changes without rebuilding
the image -- built for exactly this kind of patch.

## 6. What's implemented (steps 1-4), and what's still needed (step 5)

### 6a. Implemented, not yet run-tested

**`src/maxtext/te_overlay/transformer_engine/jax/ep.py`** -- added
`_publish_and_fetch_domain_uid(uid_bytes, is_color_root, root_rank,
uid_size)`, a point-to-point KV rendezvous mirroring
`jaxpp/dime2.py::get_nccl_id`: the domain's root publishes its UID under
`f"te_ep_domain_uid:{root_rank}"` via the distributed client's key-value
store, every other member of that domain fetches it back under the same
key. **Deliberately does not touch rank numbering anywhere** (`root_rank`,
`rank_within_group`, `EpConfig.rank`/`world_size` all stay exactly what
`_ep_domain_for_rank` already computes) -- checked that `all_uids[root_rank]`
is the *only* thing ever read out of the original `_allgather_uid`'s
result, so this is a pure transport swap (whole-job gather -> targeted
point-to-point fetch), not a semantic change to what any of those values
mean. `ep_bootstrap` gained a new `scope_uid_exchange_to_mesh: bool = False`
parameter (opt-in, default matches 100% of existing behavior for every
current caller) that switches between the two.

Rejected a more complex first draft that renumbered ranks to be
stage-local end-to-end (touching `_ep_domain_for_rank`'s `device_to_rank`,
`_allgather_uid`'s indexing, and needing a caller-supplied local rank) --
that would have required a correctness judgment call on whether
`EpConfig.rank`/`world_size` (consumed by compiled C++/CUDA extensions,
`transformer_engine_jax`/`tex.ep`, not inspectable from this checkout) need
to be global or stage-local for the actual token-routing/addressing
kernels to work. Since `all_uids[root_rank]` was the only consumer, the
transport-only fix sidesteps that question entirely.

**`src/maxtext/layers/te_ep_init.py`** (`init_te_ep_for_maxtext`):
- `world_size` now derived from `candidate.mesh.devices.size` (the mesh's
  own device count) instead of `jax.process_count()`. For non-jaxpp callers
  `candidate.mesh` still spans the whole job, so this is unchanged in
  practice; under `use_jaxpp` it's already the narrowed per-stage mesh
  (see next point), so this naturally becomes the stage-local device count.
  `rank` (`jax.process_index()`, global) is untouched, consistent with not
  renumbering ranks anywhere (see above).
- `using_pipeline_parallelism` guard narrowed to `and not use_jaxpp` --
  MaxText-native pipeline parallelism is still rejected outright (untested,
  may not compose the same way); only `use_jaxpp` is now permitted through.
- Passes `scope_uid_exchange_to_mesh=bool(config.use_jaxpp)` to
  `ep_bootstrap`.

**`src/maxtext/utils/train_utils.py`** (`setup_train_loop`) and
**`src/maxtext/trainers/pre_train/train.py`** (`train_loop`): fixed the
ordering problem from section 5a/7 above at its root. `setup_train_loop`
now builds `mesh` via `maxtext_utils.get_mesh_from_config` *before* model
creation, constructs `MpmdMesh`/narrows `mesh` right after (when
`use_jaxpp`), then calls `te_ep_init.init_te_ep_for_maxtext(config, mesh)`
(when `use_te_ep`) -- all before `model_creation_utils.from_config(...)`,
which now takes the pre-built `mesh` explicitly (`from_config` already
supported a `mesh=` override; using it means `model.mesh` is correct from
construction, no more `model.clone(mesh=...)` post-hoc fixup needed).
`train.py` no longer calls `init_te_ep_for_maxtext` itself before
`setup_train_loop` -- that old call site is *removed*, not just left as a
harmless duplicate: since it used the full un-narrowed mesh and TE_EP's
bootstrap is idempotent-by-short-circuit for a matching `config_key`,
leaving it in place would have silently locked in the wrong (full-mesh)
bootstrap before the new, correct call ever got a chance to run.

### 6b. Still blocking: `scan_layers=True` requirement (step 5, not started)

Looked for where `te_ep_layer_idx` (threaded through `moe.py`'s
`sparse_matmul`/`te_ep_wrapper` signatures, supplied via `scan`'s `xs` in
`decoders.py`'s scanned MoE loop) is actually *consumed*. Found none: the
only place a per-layer `EpLayerConfig` gets built
(`moe.py::te_ep_wrapper`, ~line 2204) only sets `top_k` and
`dispatch_output_per_expert_alignment` -- no `lax.switch` or any other
per-layer-handle selection referencing `te_ep_layer_idx` exists anywhere
in the visible Python source. Two possibilities, not yet distinguished:
(a) it's genuinely unused/vestigial scaffolding for a design that got
simplified away, in which case the `scan_layers=True` guard may be more
conservative than actually necessary, or (b) there's a reason for the
scan requirement this repo's Python source doesn't show (e.g. something
in the compiled `transformer_engine_jax`/`tex.ep` extensions, or an NNX/
Linen scan-variable-capture concern). This needs real investigation before
touching the guard -- not something to relax on a guess, unlike the PP
guard (which had a clearly-identified, now-addressed root cause).

## 7. Next steps

1. Run-test everything in 6a (nothing here has been executed against real
   hardware/JAX -- this whole section is unverified). Start with `use_te_ep
   + use_jaxpp` alone at a small scale (matching the existing
   `deepseek-v3-671b-pp8-15layer-smoke.yaml` smoke-test pattern), expecting
   it to still fail at the `scan_layers=True` guard until 6b is resolved --
   that failure mode itself is a useful signal that everything up to that
   point (mesh scoping, TE_EP bootstrap ordering, PP guard) worked.
2. Investigate 6b properly: find out why `scan_layers=True` is really
   required (or confirm it isn't), before relaxing that guard.
3. Once both land, re-test end to end.
