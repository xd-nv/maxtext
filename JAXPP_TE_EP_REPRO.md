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

## Prerequisites

Beyond the two `git clone`s below, three things are *not* carried by the
clone and must already be true of the environment you're running in:

1. **The container image.** Model configs reference a pre-built `.sqsh`
   file by absolute path:
   `/lustre/fsw/coreai_devtech_all/xiningd/amazon_fmr_containers/ghcr_nvidia_jax_maxtext_nightly_f07a860_pr3083_30743e4_sm90_v2_20260812.sqsh`
   (17GB). It's group-readable (`coreai_devtech_all`), so anyone in that
   group can use it as-is -- no rebuild needed -- but you do need that
   group membership, and the path itself is hardcoded in each model
   config (`container:` key), not something the clone brings with it.

   Since this lives on shared Lustre, a colleague who already has
   `coreai_devtech_all` group membership (the same group needed for SLURM
   access below) needs **nothing extra** -- the path just resolves.
   Confirm with:
   ```bash
   groups | tr ' ' '\n' | grep coreai_devtech_all
   ls -la /lustre/fsw/coreai_devtech_all/xiningd/amazon_fmr_containers/ghcr_nvidia_jax_maxtext_nightly_f07a860_pr3083_30743e4_sm90_v2_20260812.sqsh
   ```
   If they're **not** in that group, there's no clean self-service fix:
   this specific `.sqsh` is a hand-built nightly image with a TE overlay
   pairing baked in (must match `te_overlay_jaxpp`'s baseline, see
   `JAXPP_TE_EP_NOTES.md` 6e) and there's no documented `docker build`/
   `enroot import` recipe to reproduce this exact tag from scratch. The
   practical options are: ask whoever manages `coreai_devtech_all` group
   membership to add them, or have someone who already has access `cp`/
   `rsync` the 17GB file to a Lustre path they can read. (Overriding
   `container:` in the config to point at a different image is also
   possible, but only if you're confident that image's TE build actually
   matches `te_overlay_jaxpp` -- mismatched pairings crash on unrelated
   missing symbols, `JAXPP_TE_EP_NOTES.md` 6d.)
2. **SLURM account/partition access.** `maxtext-launcher/configs/
   defaults.yaml` hardcodes `account: coreai_devtech_all` and
   `partition: 36x2-a01r`. You need SLURM allocation rights under that
   account on EOS (or override `--partition`/`account` on the CLI for a
   different one you do have access to).
3. **A Python env with `pyyaml` for `launcher.py` itself** -- it runs on
   the login node, outside the container, just to render and submit the
   SLURM scripts.

No dataset staging or HF token is needed -- every config here uses
`dataset_type: synthetic`.

The output directory (`$REPO_ROOT/outputs/<tag>_<timestamp>/`, holding
logs, `run.sh`, `submit.sh`, `config.yaml`) is **not** separately
hardcoded -- `launcher.py` derives it directly from `workspace`
(`output_dir = f"{workspace}/outputs/{tag}_{timestamp}"`), so passing
`--workspace "$REPO_ROOT"` as shown below automatically puts outputs
under your own clone location too.

## Where the code is

- `maxtext-te-ep-v2-xiaopo`, branch `te-pr3429-nested-gemm-0826_jaxpp`,
  pushed to `git@github.com:xd-nv/maxtext.git`. Tag
  `checkpoint-before-mesh-reorder-2026-09-08` marks the current state (all
  the reproduction steps below are unchanged since that tag).
- `maxtext-launcher`, branch `add-jaxpp-pp-support`, pushed to
  `ssh://gitlab-master.nvidia.com:12051/xiningd/maxtext-launcher.git`,
  same tag name.

```bash
# Pick any empty parent directory -- these steps don't depend on its name
# or location, only that both repos land side by side inside it.
mkdir -p jaxpp-repro && cd jaxpp-repro
REPO_ROOT="$(pwd)"

# --recurse-submodules is required: third_party/jaxpp is a git submodule
# (pinned at mlsys2025-476-gc65f75e) and the launcher installs jaxpp from
# it directly (`pip install --no-deps -e "$maxtext_path/third_party/jaxpp"`,
# maxtext-launcher/templates/run.template.sh) -- an empty checkout there
# breaks the job at container startup, not at some obvious "missing repo"
# clone step.
git clone --recurse-submodules -b te-pr3429-nested-gemm-0826_jaxpp git@github.com:xd-nv/maxtext.git maxtext-te-ep-v2-xiaopo
git clone -b add-jaxpp-pp-support ssh://git@gitlab-master.nvidia.com:12051/xiningd/maxtext-launcher.git

# If you already cloned without --recurse-submodules, fix it up in place:
#   cd maxtext-te-ep-v2-xiaopo && git submodule update --init --recursive
```

Every `launcher.py` command below passes `--workspace "$REPO_ROOT"` and
`--te-overlay-dir "$REPO_ROOT/maxtext-te-ep-v2-xiaopo/src/maxtext/
te_overlay_jaxpp"`, both **absolute**, explicitly overriding whatever
`workspace:`/`maxtext_path:` the model config hardcodes. This is what
makes the steps location-independent -- without it, the model configs'
own `workspace: /lustre/fsw/coreai_devtech_all/xiningd/jax` gets bind-
mounted as `/opt/workspace` regardless of where you actually cloned, and
`maxtext_path: /opt/workspace/maxtext-te-ep-v2-xiaopo` silently resolves
inside *that* directory -- so if you clone anywhere else, the job runs
against whatever checkout (if any) happens to sit at that hardcoded path,
not the one you just made, and fails confusingly or silently trains the
wrong code. (`--maxtext-path` is also available if your checkout isn't
named `maxtext-te-ep-v2-xiaopo` directly under `$REPO_ROOT`; not needed
here since the clone command above uses that exact name.) Both
`--workspace` and `--maxtext-path` require the `add-jaxpp-pp-support`
branch of `maxtext-launcher` used above -- they were added specifically
to make this doc self-contained.

## Step 1: reproduce the working baseline (8 layers)

```bash
cd "$REPO_ROOT/maxtext-launcher"
python3 launcher.py deepseek-v3-671b-pp8-15layer-smoke-teep --cluster eos --tag repro-baseline \
  --workspace "$REPO_ROOT" \
  --te-overlay-dir "$REPO_ROOT/maxtext-te-ep-v2-xiaopo/src/maxtext/te_overlay_jaxpp"
```

8 nodes, PP=8 x EP=8, 8 decoder layers (3 dense + 5 MoE). Expect steps 0-6
to complete with a real, monotonically decreasing loss and steady-state
throughput around 40 TFLOP/s/device; step 7 (the last configured step)
hangs on the two pipeline endpoint ranks -- a separate, already-understood
issue, not what this doc is about.

**What the hang looks like, so it doesn't read as a crash**: `squeue`
still shows the job `RUNNING` (it is not stuck in a SLURM sense), rank 0's
and the last rank's per-rank logs (`outputs/<tag>_<timestamp>/output-
<jobid>-<node>-<rank>.txt`) stop after `completed step: 6` and never print
`completed step: 7`, and `nvidia-smi` on those two nodes shows the GPUs
still at **100% utilization** -- i.e. it's not idle/frozen, it's actively
spinning on some device-side op (most likely a JaxPP-scheduler-internal
NCCL collective at the pipeline's terminal/drain iteration) that never
returns. All the *other* ranks (every non-endpoint pipeline stage) do
print `completed step: 7` normally and exit cleanly -- only the two
endpoint stages (global rank 0 and the last global rank) get stuck. This
was confirmed live via `py-spy dump --pid <pid>` on both stuck ranks,
both sitting in `metric_logger.py`'s `_log_training_metrics` -> jax
`Array.__format__` -> `_value`, i.e. blocked materializing a metric value
whose underlying device computation never completes.

**How to handle it without confusion**: once you see `completed step: 6`
on every rank's log and no further step lines appear for a minute or two,
the run has given you everything it's going to -- `scancel <jobid>` is
safe at that point; you are not losing in-flight work or corrupting
partial results, since steps 0-6 already completed and logged before the
hang. Don't wait for the job to self-terminate at the walltime limit.

All three model configs used in this doc set `time: "00:15:00"` (rather
than the launcher's 1hr default) for exactly this reason -- an unattended
run hits the hang and then just burns 8-15 nodes at 100% GPU utilization
for the rest of the walltime doing nothing useful, on shared EOS
resources. 15 minutes comfortably covers compile (~3-4min) + the handful
of real training steps either way.

This step is here so you have a known-good reference point before looking
at the broken case.

## Step 2: reproduce the (now-fixed) compile-time OOM at 15 layers

```bash
python3 launcher.py deepseek-v3-671b-pp8-15layer-teep --cluster eos --tag repro-compile-oom \
  --workspace "$REPO_ROOT" \
  --te-overlay-dir "$REPO_ROOT/maxtext-te-ep-v2-xiaopo/src/maxtext/te_overlay_jaxpp"
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
  --workspace "$REPO_ROOT" \
  --te-overlay-dir "$REPO_ROOT/maxtext-te-ep-v2-xiaopo/src/maxtext/te_overlay_jaxpp"
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
