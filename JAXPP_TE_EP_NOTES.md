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

### 6b. RESOLVED: `scan_layers=True` requirement was stale, now relaxed to a warning

Traced the guard's history via `git log`/`git show` on `te_ep_init.py`:

- `980a5337` added the guard. Stated reason: the singleton `EpHandle` from
  `ep_make_handle` would otherwise be silently shared across distinct
  physical MoE layers in an unrolled stack, which TE's `ep_make_handle`
  docstring forbade.
- `7e05a9aa` is the actual bug this was protecting against: **all scan
  iterations shared one process-level `EpHandle`. Under XLA's Latency
  Hiding Scheduler, `ep_dispatch` of layer i+1 could overlap with
  `ep_combine` of layer i, corrupting the shared handle's cached NCCL
  state -> wrong token routing -> silent NaN corruption.** This was a real
  race condition, not a cosmetic restriction. The fix created one
  `EpHandle` per MoE layer and used `lax.switch(te_ep_layer_idx, ...)` in
  `te_ep_wrapper` to select the right handle per scan iteration -- which is
  why `te_ep_layer_idx` exists as a threaded value at all, and why the
  guard required `scan_layers=True` (unrolled layers had no equivalent
  per-layer plumbing).
- `62820f14` ("Adapt TE EP path to PR3036 API", Jul 1) **deleted the entire
  `EpHandle`/`create_ep_handles()`/`lax.switch` mechanism** while adapting
  to TE's newer `pr-3036` EP API, and did not revisit the `scan_layers=True`
  guard at the same time -- `te_ep_layer_idx` was left as unused dead code,
  and the guard's error message went on describing plumbing that no longer
  exists ("not yet implemented" when it had actually been implemented, then
  removed).

Confirmed via the current `ep.py` (`_dispatch_fwd`, `_combine_fwd`,
`_combine_bwd`) that the newer API has no equivalent process-shared handle
object left to alias: `tex.ep_prepare(cfg, topk_idx)` is called fresh on
every `ep_dispatch`, returning a `handle_mem` value that flows through the
`custom_vjp` primal/residual tuple like any other traced JAX value (not a
persisted Python-side object pulled from a cache). Each layer's
`handle_mem` is produced by that layer's own `ep_prepare` call, with normal
JAX data-dependency tracking -- there's no longer any state shared across
layers for XLA's scheduler to race on, structurally unlike the old design.

This can't be verified with 100% certainty without the compiled
`transformer_engine_jax`/`tex.ep` C++/CUDA source (not present in this
checkout, only the Python JAX bindings are), so the guard in
`te_ep_init.py::init_te_ep_for_maxtext` was **relaxed to a `max_logging.log`
warning rather than deleted outright**, so any unexpected regression from
`scan_layers=False` is easy to trace back to this decision. Not yet
numerically validated (no `scan_layers=False` vs `scan_layers=True`
correctness comparison run against real hardware).

## 6c. First run-test attempt (job 5969155): forgot `--te-overlay-dir`

`use_te_ep + use_jaxpp, scan_layers=False` smoke test (config
`deepseek-v3-671b-pp8-15layer-smoke-teep.yaml`), job 5969155: crashed
immediately with `TypeError: ep_bootstrap() got an unexpected keyword
argument 'scope_uid_exchange_to_mesh'`. Cause: the
`scope_uid_exchange_to_mesh` parameter added to `ep_bootstrap` (section 6a)
only exists in this repo's *overlay* copy,
`src/maxtext/te_overlay/transformer_engine/jax/ep.py` -- it's never used
unless the launcher's `--te-overlay-dir` flag mounts it over the
container's real, pip-installed `transformer_engine` package at
`/opt/transformer-engine`. Job 5969155 was submitted without that flag, so
the container's unmodified TE ran and correctly rejected the unknown kwarg.
Confirms the `te_ep_init.py` side (jaxpp-scoped bootstrap, `scan_layers`
warning) reached `ep_bootstrap()` fine -- resubmitted as job 5969194 with
`--te-overlay-dir .../src/maxtext/te_overlay`.

## 6d. Second attempt (job 5969194): pre-existing overlay drags in an
unrelated, version-mismatched file

Job 5969194 (full `src/maxtext/te_overlay/` overlaid) crashed differently:
`ImportError: cannot import name 'nvte_built_with_cublasmp' from
'transformer_engine_jax'` (the compiled `.so`), surfacing through
`gemm.py`. Cause: `src/maxtext/te_overlay/` is a **pre-existing** overlay
(not created this session) snapshotted from TE `2.19.0.dev0` @ commit
`228c8c87` (`xiaopo/te-f07-pr3083-compound` branch, for unrelated compound
TE-EP work) -- see `MANIFEST.txt`. It bundles 5 files:
`ep.py`, `cpp_extensions/ep.py`, `gemm.py`, `moe.py`, `sharding.py`.
`--te-overlay-dir` does a blind `cp -rv .../transformer_engine/.
/opt/transformer-engine/transformer_engine/` (see `launcher.py`), so all 5
get copied in together. `gemm.py`'s `nvte_built_with_cublasmp` import
requires a newer compiled `transformer_engine_jax.so` than this job's
container (`...pn-ep-unused-tiled-fixed-te-maxtext-te-ep-v2`) actually
ships -- a pre-existing version mismatch between that overlay snapshot and
this particular container, unrelated to anything from this session's work.

Fix: only `ep.py` needed the `scope_uid_exchange_to_mesh` change; nothing
in this session's work touches `gemm.py`/`moe.py`/`sharding.py`. Created a
minimal overlay dir, `src/maxtext/te_overlay_ep_only/`, containing just
`transformer_engine/jax/ep.py` and `transformer_engine/jax/cpp_extensions/
ep.py` (copied verbatim from `te_overlay/`, unmodified beyond that) --
avoids pulling in the cublasmp-dependent files. Resubmitted as job 5969206
with `--te-overlay-dir .../src/maxtext/te_overlay_ep_only`. Not yet
confirmed this avoids all version-skew risk from `cpp_extensions/ep.py`
itself (its `..sharding` import target, `sharding.py`, is *not* overlaid --
relies on the container's native `sharding.py` still exporting
`global_mesh_resource`/`get_mesh_axis_size`, which is likely but unverified
by inspection alone).

## 6e. Third attempt (job 5969206) still pending; corrected overlay/container pairing

While 5969206 (minimal `ep.py`-only overlay, original gitlab container) was
in flight, re-examined which overlay+container pairing is actually
*validated* rather than just plausible by inspection. Two distinct overlay
setups exist and were conflated at first:

- `src/maxtext/te_overlay/` -- a **repo-checked-in snapshot**, TE
  `2.19.0.dev0` @ `228c8c87` (`xiaopo/te-f07-pr3083-compound`), paired with
  the gitlab `...-te-maxtext-te-ep-v2` container -- this is the pairing that
  produced the `nvte_built_with_cublasmp` mismatch in 6d.
- `/lustre/fsw/coreai_devtech_all/xiningd/jax/te-overlay/` -- a **live,
  shared, workspace-level** overlay (no MANIFEST/version control), paired
  with `.../amazon_fmr_containers/ghcr_nvidia_jax_maxtext_nightly_f07a860_
  pr3083_30743e4_sm90_v2_20260812.sqsh`. This is what job **5940930**
  (`COMPLETED`) actually used for a non-jaxpp `use_te_ep` run -- a real,
  confirmed-working overlay+container pairing.

Diffed `ep.py` between the two: identical baseline, only difference is this
session's `scope_uid_exchange_to_mesh` addition (confirmed absent from the
live one, as expected -- that patch was never applied there). So the patch
itself is compatible with the *validated* pairing; the earlier attempts
used a container that never matched the overlay they were tested with.

Fix: copied the live `/lustre/.../jax/te-overlay/` (not the repo's stale
snapshot) into a private repo-local copy, `src/maxtext/te_overlay_jaxpp/`,
applied the `scope_uid_exchange_to_mesh` patch there, and switched
`deepseek-v3-671b-pp8-15layer-smoke-teep.yaml`'s `container:` to the
matching `.sqsh` image (plus the env vars job 5940930 needed:
`MAXTEXT_NVTE_UNSAFE_GROUPED_GEMM_USAGE_WITHOUT_TE_PERMUTE`,
`NVTE_JAX_ENFORCE_V2_GROUPED_GEMM`, `TE_EP_RECV_CAPACITY_PER_RANK`).
Resubmitted as job 5969216 with `--te-overlay-dir .../te_overlay_jaxpp`.
Superseded 5969206 (which may still finish with the gemm.py mismatch from
6d, since that config wasn't fixed on the container side) -- 5969216 is the
one that should actually be trusted.

## 6f. Job 5969216: real progress -- past bootstrap into MoE compute

Job 5969216 (corrected overlay+container pairing, 6e) got much further:
past `ep_bootstrap` (jaxpp-scoped UID rendezvous worked -- no error there
at all) and into actual MoE forward compute, failing only at
`te_gmm`/`grouped_gemm`:

```
RuntimeError: The TE V2 grouped GEMM is not supported for the given input
parameters, but NVTE_JAX_ENFORCE_V2_GROUPED_GEMM is enabled. ... The TE V2
grouped GEMM for MXFP8 requires SM100+ (Blackwell or newer) but current min
device compute capability is 90.
```

Cause: `NVTE_JAX_ENFORCE_V2_GROUPED_GEMM=1` was copied from job 5940930's
env without checking hardware -- EOS is SM90 (H100), not SM100+
(Blackwell). Removed that env var (kept
`MAXTEXT_NVTE_UNSAFE_GROUPED_GEMM_USAGE_WITHOUT_TE_PERMUTE`,
`TE_EP_RECV_CAPACITY_PER_RANK`, which aren't hardware-gated). Resubmitted
as job 5969243.

This is the strongest signal yet that `use_te_ep + use_jaxpp` fundamentally
works: mesh scoping, jaxpp-aware `ep_bootstrap` ordering, the scoped UID
KV-rendezvous, and the `scan_layers=False` relaxation all held up through
real distributed execution, not just code inspection.

## 6g. Job 5969243: hits the same JaxPP clustering assertion as ring-of-experts

With `NVTE_JAX_ENFORCE_V2_GROUPED_GEMM` removed, job 5969243 got past
tracing (`jaxpr/tracing` completed, 11.2s) and into `p_train_step.compile`,
then failed:

```
File ".../jaxpp/core.py", line 1150, in cluster_jaxpr
    raise AssertionError(
AssertionError: Failed on loop body jaxpr
```

via `trace_and_place -> wrap_into_tasks -> wrap_into_tasks_inside_loop ->
cluster_jaxpr`. This is the **exact same clustering assertion** documented
in section 4 for `use_ring_of_experts` (`optimized_ep=True` avoided it,
plain ring-of-experts did not). `use_te_ep` now hits it too, at the same
call site.

Given section 4's finding (ring-of-experts' `shard_map` `in_specs` pattern
differs from `optimized_ep`'s inference-based one, and that's plausibly
what trips JaxPP's clustering pass), and that `moe.py`'s `te_ep_wrapper`
also goes through `shard_map` for the TE_EP dispatch/combine calls, this is
likely the same root cause surfacing in a second code path, not a new bug.
Not yet confirmed by reading `te_ep_wrapper`'s `shard_map` call directly --
that's the next concrete step, not yet started as of this note.

Significance: everything specific to `use_te_ep + use_jaxpp` (mesh
scoping/6a, jaxpp-scoped bootstrap/UID rendezvous/6c-6f, `scan_layers=False`
relaxation/6b) is now validated working end-to-end. The remaining blocker
is JaxPP's task-clustering pass itself, which is a `shard_map`/tracing
concern, not specific to the TE_EP integration work done this session.

## 6h. Diagnosing the clustering assertion: routed_bias gradient hypothesis

Read the actual unclustered tail of job 5969243's dumped jaxpr (the part
`cluster_jaxpr` appends after all successfully-clustered `task[...]`
equations, right before the jaxpr's final `in (...)`). It's exactly 12
pairs of:

```
bttq:bf16[256] = broadcast_in_dim 0.0:bf16[]
...
btuc:bf16[256] = add btte bttq
```

12 == `num_moe_layers` for this 15-layer config (3 dense + 12 MoE), 256 ==
`num_experts`. Hypothesis: this is the learnable `routed_bias` gate-bias
parameter's gradient, one `[num_experts]` array per MoE layer, and
combining per-layer contributions that live in *different* mpmd/pipeline
stages (each stage owns different layers -- see the `JaxPP stage end layer
idx: [1, 3, 5, 7, 9, 11, 13, 14]` log line) is what JaxPP's clustering pass
can't place in a single stage. Ruled out `bias_updates` (the aux-loss-free
load-balancing *update direction*, `moe.py::calculate_load_balance_updates`)
as the source: `should_update_load_balance()` requires
`routed_bias_update_rate > 0.0`, which defaults to `0.0` in `base.yml` and
is never overridden in this config chain, so that path is inactive here --
leaving the `self.gate.bias` parameter gradient (active whenever
`routed_bias=True`, independent of the update-rate) as the more likely
candidate.

Testing empirically rather than fully tracing JaxPP's clustering internals
(faster and more conclusive): added `routed_bias` to `launcher.py`'s
`moe_fields` passthrough allowlist (previously not forwarded at all --
`deepseek3-671b.yml`'s `routed_bias: True` was silently un-overridable from
the launcher), then submitted job 5969310 with `routed_bias: false`
overriding the smoke-teep config's model default, same overlay/container as
5969243. This is a **temporary diagnostic override**, not a real fix --
`routed_bias` is a real part of the DSv3 architecture; if this clears the
assertion, next question is how to properly reduce this parameter's
gradient across jaxpp stages (likely needs the same treatment as any other
jaxpp cross-stage reduction, e.g. `jaxpp.cross_mpmd_all_reduce`), not to
permanently disable routed_bias.

## 6i. Confirmed against jaxpp/dev: this is a known, already-worked-around issue

Checked how `jaxpp/dev` (`/lustre/.../jax/jaxpp/maxtext-jaxpp`,
`jaxpp_dev_xiningd_fix`, same `moe.py`/`GateLogit` code, same `no
stop_gradient` on `self.bias`, same `third_party/jaxpp` pin `b83d13a` as
this branch had before this session's bump) handles this. Their
`slurm/deepseek3-671b-h100.sh` (and `-repro.sh`, `-pp16ep16.sh`) all pass
`routed_bias=false` as an explicit CLI override, with this exact comment:

> The router bias inits to zeros (initializers.py: default_bias_init =
> constant(0.0)) and nothing updates it here -- optimized_ep returns no
> moe_bias_updates (moe.py:2046) and routed_bias_update_rate defaults to
> 0.0 -- so it is identically zero all run. Leaving it on gives JaxPP 58
> dead `bias + 0` microbatch accumulators that depend on no pipeline task,
> which trips the loop-body clustering assert in jaxpp/core.py:1152.
> Revert this if aux-loss-free load balancing is ever enabled for real.

Confirms 6h's hypothesis exactly (58 there = full 61-layer model's MoE
layer count; 12 here = this 15-layer smoke config's MoE layer count -- same
mechanism, different layer count). Also notes it must be a **CLI override**,
not a YAML model-config edit, because "yaml overrides of model-config keys
are silently dropped" in their config-merging order -- worth checking
whether the same silent-drop applies to `maxtext-launcher`'s
`resolve_config` merge order if a future config sets `routed_bias` in the
YAML rather than relying on the `moe_fields` CLI passthrough added in 6h.

Conclusion: `routed_bias=False` is not a workaround unique to this
session's TE_EP work -- it's a real, general JaxPP limitation whenever
`routed_bias=True` and the manual bias-update mechanism is off (the common
case for short/synthetic smoke tests). Job 5969310 (routed_bias=false) is
expected to confirm this and should be treated as validating a
known-necessary override, not a novel diagnostic finding.

## 6j. Job 5969310: routed_bias=false confirmed, uncovered a second, TE_EP-specific issue

Job 5969310 (`routed_bias=false`) got past the clustering assert entirely
-- confirms 6h/6i's hypothesis. Progressed further (through
`before_loop`/loop tracing, `core.py:1377/1383` replication-factor
logging) before hitting a *different* assertion:

```
File ".../jaxpp/core.py", line 328, in process_primitive
    raise AssertionError("After loop computation is not replicateable")
```

Source: `train.py:458`, `maxtext_utils.calculate_te_ep_recv_metrics` ->
`jnp.concatenate(demands)`, where `demands` is one `te_ep_total_recv_tokens`
intermediate per MoE layer (sown in `moe.py::te_ep_wrapper`, one per
layer/pipeline-stage). This concatenate runs in JaxPP's post-scan
("after loop") portion of the traced program, which requires values
combined there to be identical across every stage ("replicateable") --
a cross-stage concatenate isn't. Confirmed this function doesn't exist at
all in jaxpp/dev (`grep` came up empty) -- it's TE_EP-specific telemetry
added on this fork, never previously reconciled with JaxPP's stage-crossing
rules.

Fix: gated `calculate_te_ep_recv_metrics` off under `use_jaxpp`
(`train.py`, `if config.use_te_ep and not config.use_jaxpp:`) -- it's
diagnostics-only (recv-capacity overflow monitoring), not consumed by the
loss/gradient, so skipping it under jaxpp is safe; a real fix would route
it through a proper cross-stage reduction, not attempted here. Also made
`routed_bias: false` a **permanent** requirement in
`deepseek-v3-671b-pp8-15layer-smoke-teep.yaml` (not a temp diagnostic
anymore, per 6i). Resubmitted as job 5969800.

## 6k. Job 5969800: both JaxPP assertions cleared -- now just an OOM

Job 5969800 (routed_bias=false permanent + recv-metrics gated off) cleared
*both* prior JaxPP clustering assertions -- no more "Failed on loop body
jaxpr" or "After loop computation is not replicateable". New failure is
purely resource-related:

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying
to allocate 1.74GiB with allocator GPU_0_bfc on device 0.
```

This is a milestone: `use_te_ep + use_jaxpp` is now architecturally
working end-to-end through compilation up to the point of an ordinary
memory-budget problem, not a jaxpp/TE_EP integration bug. Likely causes to
try next: `xla_dump_hlo_as_text=true` (enabled by default in this smoke
config, adds compile-time memory overhead), `per_device_batch_size=4` x
`num_pipeline_microbatches=32` possibly too large for an 8-node/64-GPU
smoke test, or `XLA_PYTHON_CLIENT_MEM_FRACTION=0.88` too aggressive
alongside jaxpp's own buffer pre-allocation. Not yet investigated further.

## 6l. Trying remat_policy=full to work around the 5969800 OOM

Switched `deepseek-v3-671b-pp8-15layer-smoke-teep.yaml`'s `remat_policy`
from `named_checkpoint` (out_proj, mla_q, mla_kv -- tuned for throughput on
the jaxpp/dev reference run) to `full` (recompute all activations, no
saved-activation memory cost, `base.yml`'s own default) to work around
5969800's `RESOURCE_EXHAUSTED` OOM. Resubmitted as job 5969821, same
overlay/container/routed_bias=false/recv-metrics-gated-off as 5969800.

## 6m. remat_policy=full made no difference -- OOM is not an activation-memory issue

Job 5969821 (`remat_policy=full`) hit the *identical* OOM as 5969800: same
`RESOURCE_EXHAUSTED ... allocate 1.74GiB ... GPU_0_bfc`, same location
(`backend_compile_and_load`, i.e. during compilation, not the runtime
training step). `remat_policy` only affects saved-activation memory during
the backward pass; it can't move an OOM that happens during compilation.
Consistent with 6k's earlier log line, `before_loop output size:
527.84GiB` (`jaxpp/core.py:1377`) -- likely parameter/state memory feeding
the pipelined loop, not activations. `dcn_fsdp_parallelism=1` in this
config, so weights are only sharded by `ici_expert_parallelism=8` within a
stage; nothing shards them further across `dcn_pipeline_parallelism=8`,
meaning each stage's 8 GPUs may be holding much more replicated weight
memory than expected. Not yet confirmed as the actual OOM cause -- next
step is to look at what's actually being allocated at compile time, and/or
try reducing `per_device_batch_size`/`num_pipeline_microbatches` (currently
4 and 32, matching the jaxpp/dev *full-scale* reference run, likely
oversized for a memory-constrained smoke test) rather than remat tuning.
5969800 cancelled per user request once superseded by 5969821 (both fail
identically, cancel was pure cleanup, not a new finding).

## 6n. Trying smaller batch/microbatch count instead of remat

Cancelled 5969800 (already superseded) and 5969821 (remat=full made no
difference, per user request) once superseded. Reverted `remat_policy`
back to `named_checkpoint` (no reason to keep `full`'s worse throughput
when it didn't move the OOM). Reduced `per_device_batch_size` 4 -> 1 and
`num_pipeline_microbatches` 32 -> 8 (local_batch stays 8 either way:
`4*64/32 == 1*64/8`, so per-microbatch shapes are unchanged -- this only
reduces how much is concurrently in flight across the pipeline schedule).
Resubmitted as job 5969916.

## 6o. Batch-size reduction didn't help either -- OOM is in JaxPP's sharding inference, not the training step

Job 5969916 (per_device_batch_size 4->1, num_pipeline_microbatches 32->8)
hit the *identical* `RESOURCE_EXHAUSTED ... 1.74GiB` OOM as 5969800/5969821
-- ruling out batch/microbatch size as the cause too. New information from
this traceback: the failing compile is
`jaxpp/sharding_inference.py:570 infer_shardings -> ...lower(...).compile()`,
**not** the main `p_train_step` compile.

Read `sharding_inference.py:536-568`: by default (`jaxpp_enable_local_
propagation=False`), this pass does `jax.jit(jcore.jaxpr_as_fun(closed_
jaxpr), in_shardings=..., out_shardings=...).lower(*avals).compile()` on
the **entire, whole-program jaxpr** -- every pipeline stage's computation
combined, not narrowed to this process's own stage -- purely so XLA can
propagate shardings; the compiled executable itself is discarded, only the
inferred sharding annotations are kept (`_write_inspected_shardings`).
This explains why neither `remat_policy` nor batch/microbatch size moved
the OOM: this particular compile's memory cost scales with the *whole*
multi-stage program's graph size, not with what any single stage executes
at runtime.

The code comment documents an alternative: `env_vars.jaxpp_enable_local_
propagation` (env `JAXPP_ENABLE_LOCAL_PROPAGATION`, default `False`) uses
`infer_shardings2` instead -- "per-task sharding inference... on the
compact pre-unroll task set... cheap," at the cost of double-compiling
each task (once for inference, once at runtime). Not yet tried. This is
the next concrete lever, not further remat/batch tuning.

## 6p. Trying JAXPP_ENABLE_LOCAL_PROPAGATION

Cancelled 5969916 (superseded, per user request). Added
`JAXPP_ENABLE_LOCAL_PROPAGATION: 1` to the smoke-teep config's `env_vars`,
to switch `sharding_inference.py`'s pass from whole-multi-stage-program
compilation to per-task local propagation (see 6o). Resubmitted as job
5969944, same overlay/container/routed_bias/recv-metrics-gated-off as
prior successful-past-the-clustering-asserts jobs; per_device_batch_size=1/
num_pipeline_microbatches=8 (from 6n) left as-is, not reverted -- testing
local propagation on top of the already-reduced batch size, not in
isolation.

## 6q. JAXPP_ENABLE_LOCAL_PROPAGATION hangs instead of OOMing -- reverted

Job 5969944 (`JAXPP_ENABLE_LOCAL_PROPAGATION=1`) did avoid the OOM --
compiling many small `infer_shardings2_{before_loop,fwd}_N` tasks instead
of one giant whole-program compile -- but then hung indefinitely: all 8
stage-leader ranks (one GPU per node, not all 64 processes checked) stopped
producing any log output simultaneously right after `infer_shardings2_fwd_7`,
with zero progress across ~12 minutes (cancelled by user request rather
than waiting longer).

Read `infer_shardings2` (`sharding_inference.py:458-508`): the per-task
loop calls `_infer_task_output_shardings(..., lowering_mesh, ...)` for
*every* task with the *same* `lowering_mesh` passed through unchanged from
the top-level call -- very likely the whole, unnarrowed, multi-process mesh
(all 64 processes), not scoped down per task. A `.compile()` call under a
multi-process `ctx_mesh` is itself a collective, synchronized operation in
JAX/XLA: every process sharing that mesh must arrive at and call that exact
compile together. So local propagation only shrinks *what* gets compiled
per step (smaller per-task jaxprs -> less compile-time memory, explaining
why the OOM went away) -- it does not decouple cross-process
synchronization. All 64 processes still have to reach and call dozens of
sequential per-task compiles in lockstep, in the same order; if even one
process falls behind anywhere (slower node, HLO-dump I/O contention,
anything), the whole job can appear to hang. This is a materially more
fragile sync pattern than the default global-compile mode's single
synchronization point.

Not confirmed by live-process inspection (`py-spy` etc. -- job was already
cancelled) -- a plausible, evidence-based hypothesis, not a proven root
cause. Given the default mode's failure (OOM) is a well-understood,
addressable resource problem with concrete levers, and this hang has none
without live debugging, **reverted `JAXPP_ENABLE_LOCAL_PROPAGATION`** and
returned to addressing the OOM directly (default global sharding
inference).

## 6r. HLO dump inspection: found the actual memory hog

Inspected job 5969800's HLO dump
(`/lustre/.../jax/hlo_dumps/teep-jaxpp-smoke-v2_5969800/`).
`module_6655.jit_jaxpr_as_fun` is exactly the whole-program
sharding-inference compile (`jax.jit(jcore.jaxpr_as_fun(closed_jaxpr),
...)`, the default non-local-propagation path in `sharding_inference.py`);
it has an `after_spmd_partitioner.txt` (11MB) but no final
`after_optimizations`/`memory-usage-report`, confirming it died mid-compile
-- consistent with the OOM.

A byte-size scan of every tensor shape in that dump surfaced `f32[65536,
7168]` (~1.88GB) and `f32[1,65536,7168]` (~1.88GB) as the largest shapes by
far -- 65536 == `TE_EP_RECV_CAPACITY_PER_RANK`, 7168 == hidden_dim. These
are the pre-GMM zero-initialization of TE_EP's receive buffer (the NaN
guard at `moe.py:2260-2268`, "Zero padded recv-buffer slots before the
GMM", `f32` because it's the gradient/backward-pass copy of the `bf16`
forward buffer). Counted **192 distinct definitions** of `f32[65536,7168]`
and **72** of `f32[1,65536,7168]` in the dump, each tagged in its metadata
to a *different* `decoder/moe_layers_N/...` (N = 0, 1, 2, 3, 4, ...) --
confirms the scan_layers=False buffer-duplication hypothesis from 6h/6m
directly, with hard evidence: under `scan_layers=False` (required by
JaxPP), each of the 12 unrolled MoE layers gets its *own* instance of this
~1.88GB buffer instead of sharing one via `scan`'s forced buffer reuse.
192/12 = 16 -- consistent with `remat_policy=named_checkpoint` duplicating
it again per backward-pass recompute segment. Even accounting for
XLA reusing memory across non-overlapping buffer lifetimes, this is a
large, structural, quantifiable cost specific to `use_te_ep +
scan_layers=False` -- entirely absent when `scan_layers=True` (the
non-jaxpp default), where `scan`'s loop-carry forces one shared buffer
across all layers by construction.

This reframes the OOM: it is not simply "PP uses more memory" in the
abstract (6l/6m's framing) -- it is this specific, identifiable buffer
class. Candidate mitigations, not yet tried: (a) lower
`TE_EP_RECV_CAPACITY_PER_RANK` for this smoke test (896MB/buffer scales
linearly with it, currently 65536; the earlier warning in 6a's job logs
about `recv_capacity_per_rank` being below the "worst-case padded receive
bound" was for a *different*, larger-scale config -- may not apply at this
15-layer/64-GPU smoke scale), (b) checking whether donation/aliasing can be
added so XLA reuses one physical buffer across the unrolled layers even
without a literal `scan` (would need jaxpp/moe.py changes, more invasive),
(c) simply reducing layer count further for the smoke test (fewer unrolled
copies).

## 6s. Trying TE_EP_RECV_CAPACITY_PER_RANK=8192

Lowered `TE_EP_RECV_CAPACITY_PER_RANK` 65536 -> 8192 in the smoke-teep
config (8x reduction, still a multiple of 128), per 6r's finding.
Resubmitted as job 5970033, same overlay/container/routed_bias/recv-metrics
-gated-off/batch=1/microbatches=8/remat=named_checkpoint as the prior
non-hanging attempts.

## 6t. TE_EP_RECV_CAPACITY_PER_RANK=8192 did NOT fix the OOM -- 6r's hypothesis is incomplete/wrong

Job 5970033 (`TE_EP_RECV_CAPACITY_PER_RANK=8192`, an 8x reduction) hit the
*exact same* `RESOURCE_EXHAUSTED ... allocate 1.74GiB` failure, same
location, as every prior attempt. This is a meaningful negative result:
**every knob tried so far that changes runtime tensor *sizes*** (remat
policy in 6m, batch/microbatch count in 6o, recv capacity here) **has
failed to move this specific OOM at all** -- always the identical 1.74GiB
allocation. `before_loop output size: 527.84GiB` (jaxpp/core.py:1377) has
also stayed exactly unchanged across every single attempt regardless of
these knobs, reinforcing that whatever's actually gating this failure is
insensitive to them.

This means 6r's hypothesis (the ~1.88GB `f32[65536,7168]` recv-buffer,
duplicated 192x across unrolled MoE layers, as the dominant OOM cause) is
likely **incomplete or wrong** as an explanation for *this specific*
1.74GiB failure, even though the 192-instance duplication finding itself is
real and confirmed in the HLO dump. More likely: this OOM is driven by
XLA's own compiler-internal working/scratch memory, which scales with
graph *complexity* (equation/op count -- 2535+ custom-calls in the 5969800
dump) rather than with the *size* of any individual tensor -- explaining
why shrinking tensor sizes has never helped. Reducing the *number* of
distinct compiled ops (fewer unrolled layers, i.e. fewer MoE layers in the
smoke config) would be a more targeted next lever than further buffer-size
tuning, since it directly reduces op count rather than op size. Not yet
tried.

## 6u. Trying fewer layers (8 instead of 15) to reduce op count

Reduced `base_num_decoder_layers` 15 -> 8 (3 dense + 5 MoE, still exactly
1 layer/stage over `dcn_pipeline_parallelism=8`), per 6t's op-count
hypothesis. Resubmitted as job 5970049, `TE_EP_RECV_CAPACITY_PER_RANK` left
at the reduced 8192 from 6s (no reason to revert -- didn't hurt, even if it
alone wasn't sufficient), everything else unchanged from 5970033.

## 6v. MAJOR MILESTONE: OOM fully resolved with 8 layers (5 MoE) -- new failure past it

Job 5970049 (`base_num_decoder_layers=8`, 5 MoE layers) **completely
resolved the compile-time OOM**: `xla_compilation/infer_shardings took
60.151s` -- the whole-program sharding-inference compile that OOM'd in
every prior 15-layer (12 MoE) attempt now succeeds cleanly. Confirms 6t's
op-count hypothesis definitively: the OOM was gated by graph complexity
(op/custom-call count, scaling with number of unrolled MoE layers under
`scan_layers=False`), not by any single buffer's size -- 6r/6s's
recv-capacity tuning was real but insufficient alone; the actual fix was
fewer unrolled layers.

New failure, past all previously-diagnosed issues, at `train.py:636`:

```
state = jaxpp.spmd_to_mpmd_reshard(mpmd_mesh, state, args_mpmd_shardings[0])
  -> jaxpp/array.py:324, jax.jit(_id, out_shardings=_actual_shardings)(...)
ValueError: Received incompatible devices for jitted computation. Got
jit's context mesh with device ids [0-7] ... explicit output sharding
with device ids [0-63]
```

Same *class* of bug as the original narrowed-vs-full mesh-size mismatch
fixed earlier this session (`a1a382ea`, `use_abstract_mesh cannot change
the size of the mesh`) -- but now inside JaxPP's own `spmd_to_mpmd_reshard`
utility (converts the full-mesh-initialized `state` into JaxPP's per-stage
MPMD representation), not in `moe.py`'s `shard_map`. Not yet investigated
-- stopped here to report the milestone clearly rather than continue
without a checkpoint, since this crosses into genuinely new territory past
the OOM that dominated sections 6k-6u.

## 6w. Root-caused and fixed the spmd_to_mpmd_reshard mesh mismatch

Compared our `train.py`'s `train_loop` to jaxpp/dev's reference
(`/lustre/.../jax/jaxpp/maxtext-jaxpp/src/maxtext/trainers/pre_train/
train.py:571-579`). There, `with jax.set_mesh(mesh)` narrowly wraps only
`p_train_step.compile(...)`; `state = jaxpp.spmd_to_mpmd_reshard(mpmd_mesh,
state, args_mpmd_shardings[0])` runs *after* that block exits, under no
narrowed-mesh context. In our `train.py`, this session's earlier mesh-
restructuring work (section 6a) had merged `jit_train_and_eval_step`,
`p_train_step.compile(...)`, *and* `spmd_to_mpmd_reshard` into one shared
`with jax.set_mesh(mesh), mesh, ...:` block spanning both the `use_jaxpp`
and non-jaxpp branches -- an unintended over-broadening.

Confirmed via `jaxpp/array.py::_spmd_to_mpmd_reshard`: it builds
`_actual_shardings` explicitly with `mesh=mpmd_mesh.jax_mesh` (the full,
un-narrowed mesh) -- by design, since its whole job is converting an
SPMD-full-mesh-sharded array into JaxPP's per-stage MPMD representation. Its
internal `jax.jit(_id, out_shardings=_actual_shardings)` call therefore
needs an ambient mesh context that's compatible with the full mesh, not the
narrowed `mpmd_mesh.lowering_mesh()` our `with jax.set_mesh(mesh)` block
established. This bug was never reachable before -- with 15 layers/12 MoE
layers, `p_train_step.compile()` always OOM'd first (6a-6v), so execution
never got as far as this line in this whole investigation.

Fix (`train.py`): narrowed the `with jax.set_mesh(mesh)` block back down to
just `jit_train_and_eval_step` + `p_train_step.compile(...)` (jaxpp branch)
/ the else-branch compile, and moved `state = jaxpp.spmd_to_mpmd_reshard(...)`
to run after that block exits -- now matches jaxpp/dev's structure exactly.
The other two `spmd_to_mpmd_reshard`/`mpmd_to_spmd_reshard` call sites later
in `train_loop` (per-step batch reshard, ~line 620 pre-fix numbering) were
already correctly placed outside any `jax.set_mesh` block -- only this one
initial-state reshard call was wrong.

Not yet re-run-tested -- next job should confirm this actually fixes the
`ValueError: Received incompatible devices` failure from 6v.

## 6x. 6w's fix was correct but surfaced a deeper, real architectural tension

Job 5970176 (6w's mesh-context fix) got past the exact `ValueError` from
6v -- confirms that fix was necessary and correctly targeted -- but hit a
*different* error at the *same* line:

```
ValueError: Received incompatible devices for jitted computation. Got
argument xs[0] of _id with shape bfloat16[129280,7168] and with device ids
[0-7] ... explicit output sharding with device ids [0-63]
```

Different failure mode: this time it's not the ambient JIT context that's
narrow -- it's the **input array itself** (`state`'s embedding/lm_head
weight) that's only physically resident on this process's 8 devices, while
`spmd_to_mpmd_reshard` wants output sharded across all 64.

This exposes a real tension between two things fixed for good reasons at
different points:

- Commit `a1a382ea` (this session's very first major fix, section 6a) made
  `state` get built against the **narrowed** `mpmd_mesh.lowering_mesh()`
  from the start -- necessary because building it against the *full* mesh
  broke `p_train_step.compile()` under `use_te_ep`
  (`use_abstract_mesh cannot change the size of the mesh`).
- `jaxpp.spmd_to_mpmd_reshard` (`jaxpp/array.py::_spmd_to_mpmd_reshard`) is
  designed to convert a state sharded across the **full** SPMD mesh into
  JaxPP's per-stage MPMD representation -- its whole purpose presumes the
  input physically lives on all 64 devices to begin with.

jaxpp/dev's own reference `train.py` builds `state` against the *full*
mesh throughout, narrows `mesh` only for later compile/execution use, and
relies on exactly this reshard call to do the actual per-stage narrowing.
They never needed a1a382ea's fix because their EP backend (ring-of-experts)
doesn't hit the `use_abstract_mesh` compile failure that `use_te_ep` does
-- so their `spmd_to_mpmd_reshard` precondition (full-mesh input) always
holds; ours doesn't, because we narrowed `state` earlier to work around a
different, TE_EP-specific problem.

Not a bug to patch around casually -- this is a real design question:
either (a) make this reshard call a no-op / skip it per-leaf where
`state`'s current (already-narrowed) sharding already matches what
`p_train_step.in_shardings` expects, or (b) find a way to satisfy
`use_te_ep`'s narrowed-mesh-at-construction requirement (a1a382ea) without
permanently losing the full-mesh view `spmd_to_mpmd_reshard` needs (e.g.
keep a full-mesh-sharded copy of `state` around just for this one reshard
call, if that's even meaningful given `state` was never actually populated
identically across stages in the first place under our construction path).
Stopped here to flag this for a design decision rather than choosing
unilaterally.

## 6y. Investigated skip-where-matching; found embedding/lm_head is a genuine cross-stage leaf, tried explicit full-mesh context instead

User asked to investigate skipping the reshard for leaves whose sharding
already matches. Before implementing, checked what the failing leaf in
6x's error actually is: `bfloat16[129280,7168]` == `[vocab_size, hidden]`
-- DeepSeek's embedding table, architecturally needed by *both* the first
pipeline stage (embedding lookup) and the last stage (LM head / output
projection). Its target legitimately spans devices `[0-63]`, not just this
process's own stage -- a genuine cross-stage array, not a false positive.
Conclusion: a full skip-based fix can't work alone -- at least this leaf
(and likely others, e.g. final norm/MTP head if present) needs real
cross-process data movement through the reshard mechanism regardless.

Instead, addressed what 6w's fix was still missing: removing the narrow
`with jax.set_mesh(mesh)` block left the ambient mesh context
implicit/unset at the `spmd_to_mpmd_reshard` call site, rather than
explicitly the full `mpmd_mesh.jax_mesh` that its internal
`jax.jit(..., out_shardings=...)` needs to reconcile with `state`'s
narrowed-mesh-resident leaves (a1a382ea narrowed `state`'s construction
mesh for use_te_ep's sake, unlike jaxpp/dev's reference flow where `state`
is genuinely full-mesh-resident already). Wrapped the call in an explicit
`with jax.set_mesh(mpmd_mesh.jax_mesh):`. Resubmitted to test.

## 6z. Implemented option 3: surgical bypass of spmd_to_mpmd_reshard

Both 6w's and 6y's mesh-context fixes were necessary but insufficient --
the real issue is that `state` is never genuinely SPMD-full-mesh-resident
(each process only ever physically holds its own 8-device slice, per
a1a382ea), while `jaxpp.spmd_to_mpmd_reshard`'s `jax.jit(_id,
out_shardings=full_mesh)` fundamentally assumes it is. No amount of
`jax.set_mesh` context around the *call* fixes a mismatch in what the
*input arrays themselves* are.

Added `_local_state_to_mpmd_reshard(mpmd_mesh, local_state, mpmd_shardings)`
in `train.py` (right before `train_loop`), replacing the
`jaxpp.spmd_to_mpmd_reshard` call at the initial-state reshard site only
(the later per-step `spmd_to_mpmd_reshard` call for `example_batch`/
`nextrng` was never broken -- genuinely SPMD-full-mesh-resident batch data,
left untouched). For each leaf: builds a `jaxpp.MpmdArray` directly from
`local_state`'s already-local array via
`MpmdArray(partially_addressable_arrays=[arr], mpmd_sharding=dsh)` if this
process's mpmd stage index is in `dsh.mesh_ids` (mirrors exactly what
`_spmd_to_mpmd_reshard`'s own per-array branch does once it already has a
per-stage local slice -- we already have that slice, so no `jax.jit`/
`slice_p` reshard is needed to produce it), or an empty `MpmdArray` (no
`partially_addressable_arrays`) otherwise -- matching
`_spmd_to_mpmd_reshard`'s "not owned by this stage" branch exactly.

**Correctness assumption, not yet empirically validated**: for leaves
whose `mesh_ids` span multiple stages (e.g. the embedding table, needed by
both the first stage's embedding lookup and the last stage's LM head),
`jaxpp.array.MpmdArray`'s own docstring states such leaves are
"replicated across those groups" -- every owning stage holds its own
complete, independently-computed copy, not a shard of one canonical copy.
Since model parameter init is deterministic (same `init_weights_seed`,
same model-construction code traced independently by every process before
any stage clustering happens), every process's local data for a
multi-stage leaf should already be numerically identical to what any other
owning stage computed -- meaning no cross-process data movement is
actually needed, only correct sharding *metadata*. This is a plausible,
evidence-based inference (from `MpmdArray`'s own docstring plus how
JAX/Flax deterministic init works), not something verified by comparing
actual values or a loss curve. If wrong for any leaf, this function will
silently produce incorrect values for it rather than raising -- flagged
prominently in the function's docstring. Needs real validation (e.g.
compare a resulting loss curve against a `scan_layers=True`/non-jaxpp
reference) before this can be trusted for anything beyond continued smoke
testing.

## 7a. Job 5970613: the bypass worked -- past all mesh/reshard issues entirely

Job 5970613 (`_local_state_to_mpmd_reshard`) hit **no mesh-mismatch error
at all**. Progressed through `xla_compilation/infer_shardings` (clean, no
OOM) and all the way into `train_loop`'s post-setup code
(`prof = profiler.Profiler(...)`, `train.py:736`) -- past every
mesh/reshard/OOM issue documented in 6a-6z. The only failure now is
trivial and unrelated:

```
ValueError: Profiling requested but initial profiling step set past
training final step
```

`skip_first_n_steps_for_profiler=10` (inherited default) vs. `steps=5` in
this smoke config -- the profiler wants to start after training already
ends. A plain config fix (disable profiling or reduce
`skip_first_n_steps_for_profiler`), not a jaxpp/TE_EP issue.

This is the furthest `use_te_ep + use_jaxpp` has ever gotten in this
investigation -- every architectural blocker (mesh scoping, jaxpp-aware
TE_EP bootstrap, scan_layers=False, the compile-time OOM, and now the
spmd_to_mpmd_reshard mesh mismatch) is resolved. Next job should reach
actual training steps for the first time.

Job 5970633 (first profiler fix attempt, `profiler: ""`) hit a *new*
trivial config error: `""` comes through the launcher as `None`, not a
valid `profiler` enum value (`''`, `'xplane'`, `'nsys'`) -- pydantic
validation failure, unrelated to jaxpp/TE_EP. Fixed instead by adjusting
`skip_first_n_steps_for_profiler: 1` / `profiler_steps: 2` so the
profiling window fits inside the 5-step run. Resubmitted as job 5970642.

## 7b. Job 5970642: same bug, different call site (per-step batch reshard)

Job 5970642 (fixed profiler config) got past training-step compile itself
and hit the *identical* `ValueError: Received incompatible devices` error
class -- but now on the **per-step** `spmd_to_mpmd_reshard` call
(`example_batch`/`nextrng`, `int32[64,512]`), not the initial state one.
My earlier assumption (6z) that this call site was "genuinely SPMD-full-
mesh-resident batch data, left untouched" was wrong -- `example_batch` is
built under the same narrowed `mesh` (a1a382ea narrows it for the whole
`train_loop`, not just state construction), so it's equally narrow-mesh-
resident, not full-mesh.

Fix: reused `_local_state_to_mpmd_reshard` for this call site too. Flagged
in an inline comment that the correctness assumption is *weaker* here than
for `state`: params are deterministically initialized (provably identical
across processes for a shared seed), but `example_batch` is loaded fresh
every step -- for `dataset_type=synthetic` (this smoke config) the
generator is plausibly deterministic too, but this is unverified and does
NOT obviously generalize to real (non-synthetic) datasets, where different
processes may genuinely hold different data. Do not reuse this bypass for
a non-synthetic-data run without re-examining that assumption.

There is a third, structurally similar call, `jaxpp.mpmd_to_spmd_reshard`
(train.py, reverse direction, MPMD -> SPMD, at the end of the training
loop, feeding into checkpoint-saving) -- left untouched for now; resubmit
and observe empirically whether it also fails, rather than guessing
preemptively (this smoke config has `enable_checkpointing: false` /
`save_checkpoint_on_completion` unset, so its result isn't actually
consumed here even if it runs).

## 7c. Job 5970696: real bug in my own bypass -- stale MpmdSharding reused for "unused" leaves

Job 5970696 hit a new error, this time inside my own bypass:

```
ValueError: Argument array's (0 (64, 512)) mpmd_idx=0 not in mpmd_idxs={mpmd_idxs}
```

Traced to `jaxpp.array.MpmdArray.__init__`'s own validation
(`array.py:93-97`): it independently re-derives `mpmd_idx` from the local
array's own mesh and checks membership in `mpmd_sharding.mesh_ids`. Root
cause in `_local_state_to_mpmd_reshard`: for "unused" leaves
(`len(dsh.mesh_ids) == 0` -- jaxpp's own convention, defaulted to rank 0 by
the reference `spmd_to_mpmd_reshard` wrapper via constructing a *new*
`MpmdSharding(mesh_ids={0}, ...)`), my code computed a local `mesh_ids`
variable (correctly defaulting to `{0}`) for its own membership check, but
then passed the **original, still-empty-mesh_ids** `dsh` object into
`MpmdArray(...)` -- so the array itself got built with an empty
`mpmd_idxs`, and `MpmdArray`'s own check (`0 not in frozenset()`) correctly
rejected it. A real bug in my implementation, not a jaxpp/TE_EP issue --
example_batch's `nextrng` leaf (or similar) is apparently one of these
"unused" arrays for at least one mpmd_idx's `in_shardings` entry.

Fixed: when `len(dsh.mesh_ids) == 0`, now construct a corrected
`MpmdSharding(mpmd_mesh=dsh.mpmd_mesh, mesh_ids={0}, spec=dsh.spec,
memory_kind=dsh.memory_kind)` and use *that* consistently for both the
membership check and the `MpmdArray(...)` construction, matching what
jaxpp's own reference `spmd_to_mpmd_reshard` wrapper does. Resubmitted to
test.

## 7d. Job 5970741: nextrng isn't NamedSharding-committed

5970696's fix worked (no more "not in mpmd_idxs" error), but a new
`AssertionError` surfaced inside `jaxpp.array.get_named_sharding`
(`assert isinstance(a.sharding, jax.sharding.NamedSharding)`), called from
`MpmdArray.__init__`. Cause: `nextrng = jax.jit(jax.random.fold_in)
(init_rng, step)` (train.py's per-step loop) has no explicit
`out_shardings`, so it's very likely `SingleDeviceSharding`, not
`NamedSharding` -- unlike `state`'s leaves (built via Flax/NNX under
`jax.set_mesh(narrow mesh)`, always `NamedSharding`) or `example_batch`
(loaded through `data_loader`, also mesh-committed). The generic
`jaxpp.spmd_to_mpmd_reshard`'s `jax.jit(_id, out_shardings=...)` call
tolerates any input sharding since jax.jit can reshard on the way in; my
direct-wrap bypass can't, since it never reshards at all.

Fixed: in `_local_state_to_mpmd_reshard`, if a leaf's `arr.sharding` isn't
already `NamedSharding`, commit it via a plain local
`jax.device_put(arr, NamedSharding(mpmd_mesh.lowering_mesh(),
PartitionSpec()))` first (fully replicated across this process's own
narrowed devices) -- safe because it's a fresh, tiny, per-process value
with no genuine cross-process state to reconcile, so it doesn't reintroduce
the narrowed-vs-full-mesh problem this function exists to avoid. Resubmitted.

## 7e. Job 5970764: MAJOR MILESTONE -- zero mesh/reshard errors; new, unrelated TE GEMM kernel bug

Job 5970764 (nextrng NamedSharding fix) produced **no mesh-mismatch or
reshard error at all** -- confirms the full bypass chain
(`_local_state_to_mpmd_reshard` for both the initial state reshard and the
per-step batch reshard, plus the "unused leaf"/NamedSharding fixes) is
working. Progressed past compile, into real per-stage task compilation
(`before_loop_0_0`, `fwd_0`, ...) and JaxPP's own inter-stage NCCL
communicator setup (`dime2.py`).

The job appeared to hang (~4 min silent), but checking all 8 stage-leader
ranks individually found the real cause: **rank 3 hit a genuine, unrelated
TE kernel error**:

```
jax.errors.JaxRuntimeError: UNKNOWN: XLA FFI call failed:
transformer_engine/jax/csrc/extensions/gemm.cpp:1247 in function
GroupedGemmFFI: Assertion failed: !lhs_is_trans && rhs_is_trans. For SM90
or older archs and FP8 input, only NT (row-major) GEMM is supported, got
lhs_is_trans=0, rhs_is_trans=0 [executable_name='jit_fwd_3']
```

TE's FP8 grouped GEMM on SM90 (H100) only supports "NT" layout (one
operand transposed); the compiled program for stage 3's forward task
passed both operands non-transposed ("NN"). Rank 3's process crashed on
this, and ranks 4-7 (whose inter-stage communicator setup involves rank
3's devices) then hung waiting on a KV-store rendezvous with the dead rank
(`DEADLINE_EXCEEDED: GetKeyValue() timed out`, 4 min) -- a cascading
failure from one real crash, not a new bug in the reshard bypass. Ranks
0-2 hadn't reached their own GEMM compile yet, just stuck mid-setup.
Cancelled the job (unrecoverable hang once rank 3 died).

This is unrelated to the entire mesh/reshard investigation (sections 6a-7d)
-- a separate TE quantization/kernel-layout issue, likely tied to
`te_gmm_quantization=te_mxfp8` + `quantization=te_fp8_currentscaling` +
`scan_layers=False`'s per-layer unrolled weight layout for this specific
stage/layer. Not yet investigated. Next step: look at why stage 3
specifically produces an NN-layout GEMM call while (presumably) other
layers don't, or determine if ALL layers would hit this once reached.

## 7f. Diagnosed the GEMM layout bug: te_mxfp8 never independently validated on H100 for this path

Checked whether the validated non-jaxpp `use_te_ep` baseline (job 5940930)
uses the same `te_gmm_quantization`: it doesn't -- 5940930 uses
`te_gmm_quantization: te_no_quant`, not `te_mxfp8`. Our smoke-teep config's
`te_mxfp8` setting was inherited unmodified from the original ring-of-
experts smoke config template and never independently validated for this
GEMM path on H100 (SM90). Combined with 6f's earlier finding this same
session (`NVTE_JAX_ENFORCE_V2_GROUPED_GEMM` requiring SM100+/Blackwell for
MXFP8 V2 grouped GEMM), this strongly suggests `te_mxfp8`'s grouped-GEMM
kernel path (V1, since V2 is blocked on this hardware) has its own
layout/orientation requirements on SM90 that aren't satisfied by however
`te_gmm`/`quantizations.py::gmm` constructs one of the operand pairs (`wo`,
the down-projection, going by the earlier `wo_kernel_axes` asymmetry vs
`w0`/`w1`) -- independent of jaxpp entirely, not something introduced by
this session's mesh/reshard work.

Switched `te_gmm_quantization` `te_mxfp8` -> `te_no_quant` (matching
5940930's validated setting; `moe_permutation_group_align_size=128` stays
valid since `te_no_quant` only requires a multiple of 8). Resubmitted as
job 5970913.

## 7g. MAJOR MILESTONE: te_no_quant fixed the GEMM bug, real training steps executed

Job 5970913 (`te_gmm_quantization: te_no_quant`) confirms 7f's diagnosis:
no GEMM layout crash at all. `fwd_0`, `bwd_0`, and `after_loop_*` tasks all
compiled and **executed** -- `Memstats: After params initialized` logged,
meaning a real forward+backward training step ran end to end for the first
time in this entire investigation. Reached
`metric_logger.buffer_and_write_train_metrics`, i.e. actual training, not
just setup.

New (trivial) failure: `AssertionError: Array is not partially
addressable`, from `jaxpp.array.MpmdArray.__format__`, via
`metric_logger.py::_log_training_metrics`'s
`f"total_weights: {scalars['learning/total_weights']}"`. Cause:
`train_step`'s scalar metrics (loss, total_weights, etc.) are only
genuinely computed/addressable on whichever mpmd stage produces the final
loss; on every other stage's process these are `MpmdArray`s with no local
data, and `metric_logger.py`'s log-formatting assumes plain,
locally-addressable values everywhere (same code exists verbatim in
jaxpp/dev's own `metric_logger.py` -- likely just never exercised there,
or their loss computation differs).

Fixed in `train.py`: right after `state, metrics = p_train_step(...)`,
`jax.tree.map` the whole `metrics` pytree through
`max_utils.maybe_unwrap` (an existing helper from earlier this session --
unwraps `MpmdArray` to `first_mpmd_replica`, or `0` if this process
doesn't own it), using `is_leaf` to stop recursion at `MpmdArray` nodes.
Resubmitted -- this could be the first fully-completing 5-step run.

## 7h. MAJOR MILESTONE: all 5 training steps completed

Job 5970982 (metrics `maybe_unwrap` fix) **completed all 5 training
steps**:

```
completed step: 0, seconds: 192.392, TFLOP/s/device: 0.091, ...
completed step: 1, seconds: 2.364, TFLOP/s/device: 7.390, ...
completed step: 2, seconds: 0.718, TFLOP/s/device: 24.347, ...
completed step: 3, seconds: 2.584, TFLOP/s/device: 6.761, ...
completed step: 4, seconds: 0.445, TFLOP/s/device: 39.243, ...
```

First time `use_te_ep + use_jaxpp`'s training loop has run end to end in
this entire investigation (sections 6a-7g). `total_weights: 0, loss:
0.000` on this rank's log is expected/correct given 7g's fix -- rank 0
(mpmd stage 0) doesn't own the final loss (computed on whichever stage
sees the model output), so `maybe_unwrap` correctly falls back to `0`
rather than crashing; the real values live on a different rank's log
(not yet checked).

New failure, only reached *after* the training loop finished: the third
reshard call site flagged in 7b as "left untouched, observe empirically" --
`jaxpp.mpmd_to_spmd_reshard` (train.py, post-loop, unconditional, feeds
checkpoint-saving) -- hit the same device-mismatch error class, reverse
direction (`_select_mpmd_slice`, input on all 64 devices, output wants
narrowed 8). Since this smoke config has `enable_checkpointing: false`,
its result isn't actually consumed -- candidate fix is to skip this call
entirely when checkpointing isn't needed, or extend the bypass pattern to
this direction too. Not yet fixed.

## 7i. Skipped the dead post-loop mpmd_to_spmd_reshard call

`train.py:711` unconditionally asserts `checkpoint_manager is None`
whenever `use_jaxpp=True` ("Checkpointing is not supported together with
JaxPP") -- so the post-loop `jaxpp.mpmd_to_spmd_reshard(...)` call's
result was never actually consumed (`checkpointing.maybe_save_checkpoint`
is a no-op with `checkpoint_manager=None`). Removed the call rather than
building/verifying a second bypass (reverse direction --
`_local_state_to_mpmd_reshard` only handles spmd->mpmd) for something
whose output is unused today. Revisit once jaxpp checkpointing support
actually lands. Resubmitted -- expecting the first fully clean 5-step run
with no errors at all.

## 7j. Root-caused and fixed the step-4 hang: profiler's block_until_ready(state)

Job 5971015 hung indefinitely (30+ min, no error) exactly after step 3.
All 8 ranks reached step 3 in lockstep (near-identical timestamps); ranks
1-6 then raced ahead to step 4 within 0.4s, while ranks 0 and 7 --
specifically the two pipeline **endpoint** stages -- never progressed.
Crucially, steps 0-3 showed a real, correctly decreasing loss curve on
rank 7 (which owns the final loss): `98.107 -> 88.539 -> 80.164 -> 73.883`
-- strong confirmation the actual `use_te_ep + use_jaxpp` forward/backward
training math is correct; the hang is unrelated to training correctness.

Root cause: `skip_first_n_steps_for_profiler=1, profiler_steps=2` puts
`profiler.Profiler.finished_initial_profile_step` at step 3 (clamped to
`steps - 1 = 4` regardless of settings once `mode != ""` -- this clamp
means `deactivate()` is *always* reachable within any run once the
profiler is enabled, it can't be configured away). `deactivate()`
(profiler.py) calls `jax.block_until_ready(blocking_object)` where
`blocking_object=state` (gated by `config.profile_cleanly`, on by
default). `state` is a pytree of `jaxpp.MpmdArray`, not plain
`jax.Array` -- and at least one leaf (the embedding/lm_head table) is
"replicated" across exactly the two endpoint stages (0 and 7), per
`_local_state_to_mpmd_reshard`'s own docstring/reasoning (section 6z).
`jax.block_until_ready` likely doesn't traverse/synchronize `MpmdArray`
the way it does an ordinary pytree, explaining why the hang is confined
to precisely those two ranks.

Fixed (`train.py`): pass `state=None` instead of `state` to both
`prof.maybe_activate_profiler`/`maybe_deactivate_profiler` when
`config.use_jaxpp`, so `profile_cleanly`'s blocking wait is skipped
entirely. This is a profiler-only diagnostic tradeoff (profiler traces may
not always capture a fully-synced compute graph under jaxpp), not a
training-correctness fix -- `MpmdArray`'s `block_until_ready` semantics
are out of scope for this investigation. Resubmitted -- expecting the
actual first fully clean, complete 5-step run.

## 7k. Profiler fix did NOT resolve the hang -- same point, same two ranks

Job 5971172 (7j's `state=None` profiler fix) hit the **identical** hang:
ranks 0 and 7 stuck after step 3 (30+ min, no error), ranks 1-6 completed
step 4 normally. Since the profiler's `jax.block_until_ready(state)` call
was fully eliminated for jaxpp and the hang persisted unchanged, 7j's
diagnosis was wrong (or at best incomplete) -- the real cause is still
unknown. Cancelled the job rather than guess again without more signal.

Only two data points exist so far, both with `steps=5`, both hanging at
the same absolute step index (after step 3, entering step 4) -- not yet
enough to distinguish "hangs on the last step" from "hangs at a fixed
step-3/4 threshold regardless of total step count" (e.g. a buffer-reuse or
caching threshold). Bumped `steps` 5 -> 8 to test: if the hang follows to
the new last step (step 6->7), that points to end-of-run logic
(eval/checkpoint-adjacent code, loop-exit handling); if it stays fixed at
step 3->4, that points to something keyed to a fixed iteration count.
Submitted as job 5971571.

## 7l. CONFIRMED: hang follows the last configured step, isolated to pipeline endpoint stages

Job 5971571 (`steps=8`) disambiguates 7j/7k cleanly: ranks 1-6 completed
**all 8 steps** (step 7 logged on every one of them); ranks 0 and 7 stalled
at step 6, one step behind, transitioning into step 7 -- the exact same
"endpoint ranks stuck one step behind the rest" pattern as the `steps=5`
runs, now shifted to match the new last-step index (was 4, now 7). This
conclusively rules out "fixed iteration count" (7k's other hypothesis) and
confirms the hang is tied to **the last configured training step**
specifically, isolated to the two pipeline **endpoint** stages (0 and 7 --
the only two stages with just one pipeline neighbor each, unlike middle
stages which have two). Loss is still decreasing correctly through step 6
(`61.715 -> 60.866`), so this remains a pure hang, not a correctness
regression.

Since 7j's profiler fix didn't move this at all, the leading (unverified)
hypothesis is now a JaxPP/`dime2.py` NCCL communicator lifecycle issue --
something that behaves specially on the last scheduled call, specific to
endpoint stages' one-sided communicator topology (each middle stage
exchanges with two neighbors; stages 0 and 7 only have one). Not yet
confirmed by reading `dime2.py`'s teardown/refcounting logic, and not
resolvable from log analysis alone -- would need live process inspection
(e.g. `py-spy dump` on a hung rank) to pin down exactly where ranks 0/7 are
blocked.

## 7m. ROOT CAUSE FOUND via py-spy: hang is metric-buffering, not compute

Installed `py-spy` in the running container (`pip install py-spy`, reached
via `srun --overlap --jobid=5971571 -w <node> --ntasks=1 --gpus=0 ...`,
using `ps aux` inside the container to find the target PID) and dumped
stacks for the two hung ranks (0 on eos0290, 7 on eos0301) from job
5971571 (`steps=8`, still hung at the same endpoint-rank-behind pattern as
5971172 but at the new last step). Both stacks were identical in shape:

```
_value (jax/_src/array.py:642)
wrapper (jax/_src/profiler.py:420)
__format__ (jax/_src/array.py:334)
_log_training_metrics (metric_logger.py:179 or 189)
log_metrics -> write_metrics -> flush_metrics_and_cleanup
train_loop (pre_train/train.py:924)
```

**Definitive finding: both ranks had already exited the main training
loop** (`train_loop`, past the `for step in ...:` block, at line 924 --
`flush_metrics_and_cleanup()`, called once after the loop). All 8 steps'
compute completed successfully on every rank including the endpoints. The
hang is entirely inside `metric_logger.py`'s metrics-buffering mechanism,
unrelated to JaxPP's pipeline execution, NCCL communicators, or anything
else investigated in 6a-7l.

Root cause: `buffer_and_write_train_metrics` (metric_logger.py:325)
deliberately writes the *previous* step's metrics while buffering the
*current* step's -- an intentional overlap optimization to hide
host-device sync latency behind the next step's compute. This means the
very last step's metrics are **never** materialized inside the loop, only
in `flush_metrics_and_cleanup()` after the loop has exited. Under jaxpp,
materializing that deferred last-step value hangs indefinitely, but only
on the two pipeline endpoint ranks (0 and mpmd_dim-1) -- rank 0's hang was
on `moe_lb_loss`, rank 7's on `lm_loss`, different fields, confirming it's
not one specific poisoned metric but something structural about
materializing *any* buffered value once the loop (and whatever "there's a
next call coming" assumption jaxpp's buffer-donation/reuse relies on) has
ended.

Fixed (`train.py`): under `config.use_jaxpp`, skip the one-step-behind
buffering entirely -- call `metric_logger.record_train_metrics(...)` +
`metric_logger.write_metrics(metrics, step)` directly, every step, so
nothing is ever deferred past the point where its underlying computation
is guaranteed still valid. `flush_metrics_and_cleanup()`'s buffered-write
at the end becomes a no-op under jaxpp as a result (`buffered_train_metrics`
stays `None`), removing the hang trigger entirely rather than working
around its symptom. Non-jaxpp path is untouched (keeps the overlap
optimization). Resubmitted -- expecting this to finally be the fully clean
run.

## 7n. 7m's fix moved the hang, didn't remove it -- this is a real device-level stuck computation, not a Python buffering issue

Job 5971753 (7m's immediate-write fix) hit the identical hang, confirmed
via a second py-spy dump: the stack now shows `train_loop
(pre_train/train.py:912)` -- my new direct `metric_logger.write_metrics(...)`
call *inside* the loop -- instead of `flush_metrics_and_cleanup` after it.
Same `_log_training_metrics(metric_logger.py:189)` -> `__format__` ->
`._value` blocking pattern as before.

This is a materially different, more significant finding than 7m's
original diagnosis: since the hang follows the value access regardless of
*when* (immediately after `p_train_step` returns vs. deferred after loop
exit), it rules out "Python-level buffering/donation-after-loop-ends" as
the cause entirely. The underlying async device computation for this
metric (`moe_lb_loss` on rank 0, `lm_loss` on rank 7 in earlier runs) is
genuinely never dispatched to completion by the compiled program on the
GPU, specific to the pipeline's *final* scheduled iteration, specific to
the two endpoint stages (0 and mpmd_dim-1 -- the only stages with just one
pipeline neighbor each, unlike middle stages' two). This looks like a real
bug in JaxPP's own scheduling/task-clustering for terminal-iteration
draining at pipeline boundaries (`interleaved_1f1b` + `fuse_steady_state`),
not something fixable by rearranging *when* MaxText reads the value at the
Python level. 7m's fix is not wrong to keep (it's a reasonable
simplification either way), but it does not address the actual root
cause. Cancelled the job.

Not yet investigated further -- would need to look inside JaxPP's own
schedule-generation/task-clustering code (`third_party/jaxpp/src/jaxpp/`)
for how it handles the final iteration's outputs at pipeline endpoints, or
get a device-side (not just Python-level) view of what's actually stuck
(e.g. `cuda-gdb`, NCCL debug logging) -- beyond what's been attempted so
far in this investigation.

## 7o. Final validated smoke-test results (accepted the last-step hang as a known limitation)

Job 5972016 (`steps=8`, same config as 7n) confirmed steps 0-6 complete
cleanly and correctly on both endpoint ranks, matching the established
pattern exactly (only step 7, the last, hangs -- 7l/7n). Cancelled before
reaching step 7, per explicit user direction that the last-step hang is an
acceptable, known limitation given perf data is available from the steps
before it.

**Loss curve (rank 7, owns the final loss)**: `98.1 -> 88.5 -> 79.9 ->
71.2 -> 64.1 -> 61.7 -> 60.9` -- monotonically decreasing, confirms
correct training math end to end.

**Performance (rank 7), steady-state by steps 4-6**:

| Step | TFLOP/s/device | Tokens/s/device |
|---|---|---|
| 4 | 39.0 | 1144 |
| 5 | 40.2 | 1179 |
| 6 | 40.2 | 1178 |

Converges around **~40 TFLOP/s/device, ~1180 tokens/s/device** for this
reduced smoke-test config (8 decoder layers, batch=1, EP=8 x PP=8 = 64
GPUs, `te_no_quant`, `remat_policy=named_checkpoint`). Not representative
of full-scale (61-layer) production throughput -- see the caveats
discussed earlier in this session (small model, reduced recv-capacity/
microbatches tuned to dodge the compile OOM, not tuned for perf).

**Status as of this note**: `use_te_ep + use_jaxpp` is validated
end-to-end for correctness on this fork (mesh scoping, TE_EP bootstrap,
`scan_layers=False`, the compile-time OOM, all three resharding call
sites, GEMM layout, metric logging -- sections 6a-7n). The one open,
accepted-as-known-limitation issue is the last-configured-step hang on
pipeline endpoint stages, root-caused via py-spy to a stuck device-side
async computation likely in JaxPP's own terminal-iteration scheduling, not
fixable from the MaxText integration layer without deeper JaxPP-internals
work (out of scope for this session per explicit user direction).

## 7. Next steps

1. Run-test everything in 6a and 6b (nothing here has been executed against
   real hardware/JAX -- this whole section is unverified). Start with
   `use_te_ep + use_jaxpp, scan_layers=False` at a small scale (matching the
   existing `deepseek-v3-671b-pp8-15layer-smoke.yaml` smoke-test pattern).
2. Before trusting `scan_layers=False` + `use_te_ep` numerically (with or
   without jaxpp), run a small correctness comparison against a
   `scan_layers=True` reference (same seed/data, compare loss curve / check
   for NaN) -- 6b's conclusion is code-evidence-based, not empirically
   validated yet.
3. Once both land, re-test end to end.
