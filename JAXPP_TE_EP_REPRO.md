# Reproducing the current `use_te_ep` + `use_jaxpp` blocker

This is a short, focused guide for reproducing the *current* open issue
(scaling past 8 decoder layers). For the working, validated case, see
`JAXPP_TE_EP_SETUP.md`. For the full chronological investigation behind
every fix and finding referenced here, see `JAXPP_TE_EP_NOTES.md`
(sections 6a-8a).

## The one-paragraph situation

`use_te_ep` (TE's NCCL expert-parallel MoE) works together with
`use_jaxpp` (pipeline parallelism) and trains correctly, but only at a
reduced 8-decoder-layer scale. Scaling to the originally-targeted 15
layers hits two stacked memory problems. The first is fixed. The second is
open: a real, runtime out-of-memory error building the model's initial
parameters, which -- surprisingly -- does not improve when adding more
pipeline stages/nodes. That's the part worth a second pair of eyes.

## Where the code is

- `maxtext-te-ep-v2-xiaopo`, branch `te-pr3429-nested-gemm-0826_jaxpp`,
  pushed to `git@github.com:xd-nv/maxtext.git`. Tag
  `checkpoint-before-mesh-reorder-2026-09-08` marks the current state (all
  the reproduction steps below are unchanged since that tag).
- `maxtext-launcher`, branch `add-jaxpp-pp-support`, pushed to
  `ssh://gitlab-master.nvidia.com:12051/xiningd/maxtext-launcher.git`,
  same tag name.

```bash
git clone -b te-pr3429-nested-gemm-0826_jaxpp git@github.com:xd-nv/maxtext.git maxtext-te-ep-v2-xiaopo
git clone -b add-jaxpp-pp-support ssh://git@gitlab-master.nvidia.com:12051/xiningd/maxtext-launcher.git
```

## Step 1: reproduce the working baseline (8 layers)

```bash
cd maxtext-launcher
python3 launcher.py deepseek-v3-671b-pp8-15layer-smoke-teep --cluster eos --tag repro-baseline \
  --te-overlay-dir /path/to/maxtext-te-ep-v2-xiaopo/src/maxtext/te_overlay_jaxpp
```

8 nodes, PP=8 x EP=8, 8 decoder layers (3 dense + 5 MoE). Expect steps 0-6
to complete with a real, monotonically decreasing loss and steady-state
throughput around 40 TFLOP/s/device; step 7 (the last configured step)
hangs on the two pipeline endpoint ranks -- a separate, already-understood
issue (`JAXPP_TE_EP_NOTES.md` 7m/7n), not what this doc is about. Either
let it run and read the metrics from the earlier steps, or cancel once
you've seen enough.

This step is here so you have a known-good reference point before looking
at the broken case.

## Step 2: reproduce the (now-fixed) compile-time OOM at 15 layers

```bash
python3 launcher.py deepseek-v3-671b-pp8-15layer-teep --cluster eos --tag repro-compile-oom \
  --te-overlay-dir /path/to/maxtext-te-ep-v2-xiaopo/src/maxtext/te_overlay_jaxpp
```

Same 8-node/PP=8/EP=8 topology, `base_num_decoder_layers: 15` (3 dense +
12 MoE) instead of 8. As shipped, this config already includes
`JAXPP_FAST_INFER_SHARDINGS: 1` in its `env_vars`, so it will **not**
reproduce the compile OOM -- it's fixed. To see the original failure,
remove that one line and resubmit: you'll get
`RESOURCE_EXHAUSTED: ... allocate 1.74GiB ...` inside
`xla_compilation/infer_shardings`, which took ~45s to fail regardless of
node count. Root cause: that JaxPP compile pass runs with XLA autotuning
enabled by default, benchmarking candidate kernels on-GPU during
compilation -- real memory allocations scaling with the number of
distinct ops in the compiled program (which scales with unrolled MoE
layer count under `scan_layers=False`, required by JaxPP). Confirmed via
elimination testing (shrinking individual buffer sizes never helped;
shrinking layer count did) -- see `JAXPP_TE_EP_NOTES.md` 6r/6t/8.

## Step 3: reproduce the open runtime OOM -- the actual blocker

With `JAXPP_FAST_INFER_SHARDINGS: 1` left in place (so you get past step
2's issue), the same job from step 2 fails differently and later:

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while
trying to allocate 896.00MiB with allocator GPU_0_bfc on device 0.
[executable_name='jit_initialize_state']
```

This happens while actually materializing the model's initial parameters
(not at compile time this time -- the traceback goes through
`get_first_step` -> `state.step` materialization, surfacing a failure from
`jit_initialize_state`'s own execution).

**The specific thing worth a second pair of eyes**: this failure is
*exactly the same*, byte-for-byte, whether you use 8 pipeline stages or
15. To see this yourself:

```bash
python3 launcher.py deepseek-v3-671b-pp15-15layer-teep --cluster eos --tag repro-pp15 \
  --te-overlay-dir /path/to/maxtext-te-ep-v2-xiaopo/src/maxtext/te_overlay_jaxpp
```

This is the *same* 15-layer model, but `dcn_pipeline_parallelism: 15` /
`nodes: 15` instead of 8 -- exactly 1 decoder layer per pipeline stage
instead of 1-2. Naively, more pipeline stages (fewer layers per stage)
should reduce how much parameter data each device needs to hold. It
doesn't: both jobs fail with the identical `896.00MiB` allocation error.

You can confirm this precisely from each job's HLO dump (`xla_dump_to` in
the run command; look for `module_*.jit_initialize_state.sm_9.0a_gpu_
after_optimizations-memory-usage-report.txt`). Its `Total bytes:` line
reads **99.08GiB** in both the PP=8 and PP=15 runs -- identical -- versus
**42.75GiB** for the working 8-layer baseline from step 1. Also confirmed
via `module_*.jit_initialize_state.config.pbtxt`: `num_partitions: 8` in
*both* the PP=8 and PP=15 runs -- i.e. `jit_initialize_state` is compiled
against exactly the local pipeline stage's `EP=8` devices regardless of
how many total pipeline stages/nodes exist.

## What we believe the root cause is (not yet fixed)

`train_utils.py::setup_train_loop` narrows the mesh to just the local
pipeline stage's devices (`mpmd_mesh.lowering_mesh()`) **before** building
the model/state:

```python
mesh = maxtext_utils.get_mesh_from_config(config, devices)
mpmd_mesh = None
if config.use_jaxpp:
  mpmd_mesh = jaxpp.MpmdMesh(mesh, "stage")
  mesh = mpmd_mesh.lowering_mesh()          # <-- narrows here
...
model = model_creation_utils.from_config(config, devices, mesh=mesh)   # <-- built narrow
...
state, ... = maxtext_utils.setup_training_state(..., mesh, ...)        # <-- built narrow
```

That narrowing (commit `a1a382ea`, the very first fix in this whole
investigation) was required to make `use_te_ep`'s own compile succeed
under `use_jaxpp` at all -- without it, `p_train_step.compile()` raises
`use_abstract_mesh cannot change the size of the mesh`. But its side
effect is that state construction is permanently pinned to `EP`-sized
sharding, never able to use the "stage"/PP axis to spread parameters
across more devices as PP grows -- which is presumably how the reference
`jaxpp/dev` maxtext branch achieves memory that scales with PP degree, in
some form we have not yet pinned down with confidence (see below).

**We do not have a confirmed fix.** Three hypotheses were investigated
this session and each one, on closer inspection, failed to hold up or
remained unverified:

1. *Switch compilation strategy to `jaxpp.mpmd_jit_with_loop`* -- turned
   out to be a non-issue: our fork already uses this (`train_utils.py::
   jit_train_step`, ported unchanged from `jaxpp/dev`).
2. *Build state on the wide mesh, reshard narrow before compiling* --
   plausible, matches `jaxpp/dev`'s `jit_train_step`-level narrowing
   timing, but unverified whether `jaxpp.mpmd_jit_with_loop`'s `.compile()`
   actually tolerates a wide-sharded concrete `state` argument, and how to
   derive the correct narrow reshard target before `p_train_step` exists
   to provide `in_shardings`.
3. *Wire in `sharding.add_stage_to_sharding`* (already ported into this
   repo, in `src/maxtext/utils/sharding.py`, but never called) -- this is
   what `jaxpp/dev`'s `maxtext_utils.py::setup_initial_state` uses to tag
   `out_shardings` with a "stage" partition dimension. But it's called
   there with the *already-narrowed* `lowering_mesh()`, whose "stage" axis
   has size 1 -- meaning `add_stage_to_sharding`'s own divisibility check
   (`size % mesh.shape["stage"] == 0`) is trivially true for any size, so
   the tag it adds looks like a no-op for actual memory sharding at that
   point. Unclear how (or whether) this genuinely reduces per-device
   memory in `jaxpp/dev`'s own flow -- not confirmed empirically there
   either.

None of these were implemented. If you have a clearer read on how
`jaxpp/dev`'s memory scaling with PP degree is actually supposed to work
-- or want to A/B it empirically against a real `jaxpp/dev` job with
memory profiling -- that's exactly the kind of second opinion this doc is
for.

## Reference: known-good vs. broken, at a glance

| | 8 layers, PP=8 (baseline) | 15 layers, PP=8 | 15 layers, PP=15 |
|---|---|---|---|
| Compile-time `infer_shardings` OOM | n/a (never had it) | fixed via `JAXPP_FAST_INFER_SHARDINGS=1` | same fix applies |
| `jit_initialize_state` peak memory | 42.75GiB | 99.08GiB | 99.08GiB (**identical**) |
| Trains successfully | yes (steps 0-6) | no -- OOMs | no -- OOMs, identically |
