# Running `use_te_ep` + `use_jaxpp` together

This is a practical how-to for running DeepSeek V3 671B with TE's
NCCL-based expert-parallel MoE (`use_te_ep`) together with JaxPP pipeline
parallelism (`use_jaxpp`) on this branch. For the full investigation
history, root causes, and what was tried and rejected along the way, see
`JAXPP_TE_EP_NOTES.md` (sections 6a-8a) in this repo.

## Status

**Validated at reduced scale.** End-to-end on 8 EOS nodes (H100, 64 GPUs):
training runs with a correct, monotonically decreasing loss curve and
steady-state throughput of ~40 TFLOP/s/device on an 8-layer smoke config.
The last configured training step hangs on the two pipeline endpoint
ranks -- see [Known limitations](#known-limitations) below. Everything
short of that is working and numerically sane.

**Blocked at the originally-targeted 15-layer scale** (and, by the same
mechanism, at full 61-layer scale). Two distinct memory problems stack up
as layer count grows under `scan_layers=False`:

1. A compile-time OOM in JaxPP's whole-program sharding-inference compile
   (op count scales with unrolled MoE layer count) -- **fixed**, see
   [Known limitations](#known-limitations) item 5.
2. A separate, genuinely-runtime OOM in `jit_initialize_state`, which does
   **not** improve with more nodes/PP degree -- **open**, root-caused but
   not fixed; see item 5 for why it's a real architectural gap, not a
   config tweak.

This has **not** been run at full scale (61 layers, production
batch/microbatch counts) or validated against a `scan_layers=True`
numerical reference. Treat it as "architecturally working, smoke-tested at
reduced scale," not "production-ready" and not yet "scales with more
nodes."

## What's involved

Three repos/pieces, all required together:

1. **This repo**, branch `te-pr3429-nested-gemm-0826_jaxpp` -- has the
   JaxPP port plus the `use_te_ep`-under-jaxpp fixes (mesh scoping,
   `scan_layers` guard relaxation, the `spmd_to_mpmd_reshard` bypass, GEMM
   quantization, metric-logging fixes).
2. **`src/maxtext/te_overlay_jaxpp/`** (checked into this repo) -- a
   Python-source-only overlay for `transformer_engine`, carrying a patch
   to `ep_bootstrap`'s NCCL UID exchange (`scope_uid_exchange_to_mesh`) so
   it works when the ambient mesh has been narrowed to one pipeline
   stage's devices, instead of assuming it always spans the whole job.
   Must be passed to the launcher via `--te-overlay-dir`; it's copied over
   `/opt/transformer-engine/transformer_engine/` inside the container at
   job start (see `maxtext-launcher/launcher.py`'s `te_overlay_setup`).
3. **`maxtext-launcher`**, branch `add-jaxpp-pp-support` -- has the JaxPP
   CLI/config knobs and the validated smoke-test config,
   `configs/models/deepseek-v3-671b-pp8-15layer-smoke-teep.yaml`.

## How to run the validated smoke test

```bash
cd /lustre/fsw/coreai_devtech_all/<you>/jax/maxtext-launcher   # or wherever it's checked out
python3 launcher.py deepseek-v3-671b-pp8-15layer-smoke-teep --cluster eos --tag <your-tag> \
  --te-overlay-dir /path/to/maxtext-te-ep-v2-xiaopo/src/maxtext/te_overlay_jaxpp
```

This launches an 8-node (64-GPU) job: `EP=8` (ici) x `PP=8` (dcn), 8
decoder layers (3 dense + 5 MoE), `steps: 8`. Expect steps 0-6 to complete
normally with real, decreasing loss; step 7 will hang on two ranks (see
below) -- either let it run and manually cancel once you've read the
metrics you need, or reduce `steps` if you don't need the extra data
points. See that config file's own header comments for the reasoning
behind each setting -- several are load-bearing, not arbitrary (e.g.
`routed_bias: false`, `te_gmm_quantization: te_no_quant`).

### Container and container-overlay pairing matters

The smoke config's `container:` (a `.sqsh` file, not the gitlab image used
by other DSv3 configs) is specifically the one whose baked-in
`transformer_engine` matches `te_overlay_jaxpp`'s baseline. Don't swap the
container without checking the overlay is still compatible --
mismatched pairings crash on unrelated missing symbols (see
`JAXPP_TE_EP_NOTES.md` 6d for a concrete example of what goes wrong).

## Known limitations

1. **Last-step hang on pipeline endpoint ranks.** Root-caused via a live
   `py-spy` stack dump (`JAXPP_TE_EP_NOTES.md` 7m/7n) to a stuck
   device-side async computation for one metric value on the two pipeline
   endpoint stages (rank 0 and the last rank) -- not a crash, not a
   training-correctness issue (loss is already correct and decreasing by
   the time it happens). All non-endpoint pipeline stages complete the
   final step and exit cleanly; only the two endpoint stages get stuck.
   Confirmed live (job 6000960, 2026-09-09) that this is an active spin,
   not an idle wait: `nvidia-smi` on the two stuck nodes shows their GPUs
   at 100% utilization the whole time, while `py-spy` shows the host
   thread blocked in `metric_logger.py`'s `_log_training_metrics` ->
   `Array.__format__` -> `_value`, waiting on that device computation to
   finish. Points at a stuck kernel/NCCL collective inside JaxPP's own
   scheduler at the terminal/drain iteration, not something fixable from
   this integration layer -- root-causing exactly which call is stuck
   would need `NCCL_DEBUG=INFO` or an `nsys` capture on the two endpoint
   ranks while it's spinning; not yet attempted. Workaround: this is safe
   to work around, not just tolerate -- `squeue` still shows the job
   `RUNNING` and every earlier step's data is already fully logged before
   the hang, so once `completed step: N-1` appears on every rank's log and
   no further step lines show up for a minute or two, `scancel` the job
   rather than waiting for the walltime limit.

2. **`scan_layers=False` + `use_te_ep` numerical correctness is inferred,
   not verified.** The `scan_layers=True` requirement was relaxed to a
   warning based on tracing the code history (the original race condition
   it guarded against was structurally eliminated by a later TE API
   migration -- see `JAXPP_TE_EP_NOTES.md` 6h/6i) but no side-by-side loss
   comparison against a `scan_layers=True` reference has been run.

3. **The `spmd_to_mpmd_reshard` bypass (`_local_state_to_mpmd_reshard` in
   `train.py`) relies on an unverified assumption for multi-stage-owned
   state leaves** (e.g. a tied embedding/LM-head table): that every owning
   stage's independently-initialized copy is already numerically
   identical, so no real cross-process data movement is needed. This
   holds by construction for model parameters (deterministic init from a
   shared seed). For the equivalent per-step batch/rng reshard, the same
   bypass is used but the assumption is weaker and unverified for
   non-synthetic datasets -- see the docstring and the inline comment at
   that call site in `train.py` before reusing it for a real-data run.

4. Not tested at full model scale (61 layers), only the reduced 8-layer
   smoke config and a not-yet-fully-working 15-layer attempt (see item 5).

5. **Layer count is blocked at ~8 -- scaling to the original 15-layer
   target (or full 61-layer scale) hits two stacked memory problems,**
   both `JAXPP_TE_EP_NOTES.md` section 8/8a:

   - **Compile-time OOM (fixed)**: JaxPP's whole-program sharding-inference
     compile OOMs as unrolled MoE layer count grows -- driven by op/
     instruction *count*, not any individual tensor's size (confirmed by
     elimination: shrinking buffers didn't help, shrinking layer count
     did). Root cause: that compile runs with XLA autotuning fully
     enabled by default, which benchmarks candidate kernels *on the GPU*
     during compilation -- real device memory allocations scaling with op
     count. **Fix**: add `JAXPP_FAST_INFER_SHARDINGS: 1` to the config's
     `env_vars` (disables autotuning for that specific compile). Confirmed
     working at 15 layers (job 5994713).
   - **Runtime OOM in `jit_initialize_state` (open, not fixed)**: once the
     compile-time OOM is out of the way, materializing the initial
     parameters/optimizer state OOMs for real -- and, unlike the compile
     OOM, this **does not improve with more PP stages/nodes** (confirmed
     identical ~99GB peak memory requirement at both PP=8 and PP=15, same
     15-layer model). Root cause: `train_utils.py::setup_train_loop`
     narrows the mesh to just the local pipeline stage's `EP` devices
     *before* building the model/state (`a1a382ea`, this session's very
     first fix, needed for `use_te_ep`'s compile to work at all under
     jaxpp). That means state is always sharded across only the EP-sized
     device group, regardless of PP degree -- unlike `jaxpp/dev`'s
     reference flow, which builds state on the *full* wide mesh (so it
     naturally scales across more devices as PP grows) and only narrows
     right before compiling the train step.
     **Investigated a fix, did not implement it**: matching `jaxpp/dev`'s
     PP-scalable behavior turns out to require switching from this fork's
     compilation pattern (plain `jax.jit` + manual `.compile()` +
     `_local_state_to_mpmd_reshard`) to JaxPP's higher-level
     `jaxpp.mpmd_jit_with_loop` API, which `jaxpp/dev`'s `jit_train_step`
     actually uses and which handles the wide-state/narrow-compile
     reconciliation internally. That's a materially bigger, higher-risk
     rewrite than a mesh-reordering tweak -- it touches the compilation
     strategy several other fixes this session depend on (the reshard
     bypass, per-step reshard, metrics/profiler fixes), with real
     potential for silently incorrect sharding if done wrong. Not
     attempted; flagged as future work. A git tag,
     `checkpoint-before-mesh-reorder-2026-09-08`, marks the last known-good
     state in both this repo and `maxtext-launcher` if picking this up
     later.

## Where to look for more detail

- `JAXPP_TE_EP_NOTES.md` (this repo) -- the full chronological
  investigation, including every dead end, wrong hypothesis, and the
  reasoning behind each fix. Sections 6a-6z cover getting past mesh
  scoping, the compile-time OOM, and the GEMM/quantization issue; 7a-7o
  cover the resharding bypass and the metric-logging/hang investigation;
  section 8/8a covers the 15-layer scaling attempt and both memory issues
  above.
- `src/maxtext/layers/te_ep_init.py::init_te_ep_for_maxtext` -- the
  `scan_layers` guard and its NOTE(jaxpp) docstring block.
- `src/maxtext/trainers/pre_train/train.py::_local_state_to_mpmd_reshard`
  -- the reshard bypass, with its own detailed docstring.
- `maxtext-launcher/configs/models/deepseek-v3-671b-pp8-15layer-teep.yaml`
  and `deepseek-v3-671b-pp15-15layer-teep.yaml` -- the two configs used to
  isolate and test the layer-scaling issues above.
