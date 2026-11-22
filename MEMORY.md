# File containing AI-generated notes from previous sessions to allow
# agents to preserve key information between runs

## Session 2026-11-20: SAM3 install + video pipeline trace

### Goal
Install Meta's SAM3 (facebookresearch/sam3), verify real CUDA mask
generation on a short clip, and trace the video inference call chain, as
prep for a later session comparing eager / backbone-only compile / stock
compile modes. Optimization patch itself is explicitly deferred.

### State: blocked on Hugging Face checkpoint access, everything else ready

**What's done:**
- Hardware note: the user's prompt said RTX A4000 16GB, but `nvidia-smi`
  on this machine shows an **RTX 4070 SUPER with 12GB VRAM** (compute
  capability 8.9, driver reports CUDA UMD 13.3). Told the user; proceeded
  with actual hardware. Worth re-confirming if this is run on different
  hardware later — 12GB is tighter than 16GB for compile-mode memory
  comparisons.
- Repo cloned to `sam3/` at commit `31cb7e624e4fe95301b1be1d4e809111593d2e45`
  (2026-11-17). Contains both the regular SAM3 path and the newer SAM 3.1
  Object Multiplex path (see `RELEASE_SAM3p1.md`).
- Venv created at `.venv/` (project root, matches AGENTS.md convention).
  Installed: `torch==2.10.0+cu128`, `torchvision==0.25.0+cu128`,
  `sam3` (editable, `.[notebooks]` extras), plus `triton-windows==3.9.0.post30`
  (PyPI `triton` has no Windows wheels; `sam3/model/edt.py` imports `triton`
  directly at module load time, so the package is unimportable on Windows
  without this community fork — same import name, drop-in). Also had to
  manually pin `numpy<2` and `pandas<2.3` after installing `pandas` (not in
  any of sam3's declared extras, but required by `sam3/visualization_utils.py`
  for its `save_masklet_image`/overlay helpers) pulled in numpy 2.x which
  breaks sam3's `numpy>=1.26,<2` constraint.
- CUDA verified working: matmul on GPU succeeds, `torch.cuda.get_device_name(0)`
  → "NVIDIA GeForce RTX 4070 SUPER".
- Full `sam3` import chain verified (all builders in `model_builder.py`
  import cleanly after the triton fix).
- Frame-loading pipeline verified independent of any checkpoint: loaded
  all 270 bundled frames from `sam3/assets/videos/0001/` (1280x720) via
  `sam3.model.io_utils.load_resource_as_video_frames` in ~4s.
- Confirmed builder choice for this session: `build_sam3_video_predictor()`
  → `Sam3VideoPredictorMultiGPU` → `build_sam3_video_model()`, which
  downloads `facebook/sam3` / `sam3.pt` (the plain SAM3 checkpoint, NOT
  the 3.1 multiplex one). This is the correct, simpler non-multiplex path
  per the user's request.
- Wrote `scripts/smoke_test.py` (`--mode eager|compile`, optional
  `--timeout`): start_session → add_prompt(text="person") on frame 0 →
  propagate_in_video → verifies first nonempty mask → saves overlay PNG to
  `outputs/smoke_test_<mode>_overlay.png` → reports frame count, mask
  shape, elapsed time, peak `torch.cuda.max_memory_allocated()`.
  Classifies OOM / timeout / other error separately (needed for step 5,
  the compile-mode feasibility check). **Not yet run to completion** —
  blocked below. Model construction itself (full nn.Module graph: tracker,
  detector, transformer, segmentation head) was confirmed to succeed by
  running it anyway and observing where it failed.

**Blocker (updated):** HF login is now done — user ran
`.venv/Scripts/hf.exe auth login` (needed the venv's `hf.exe` directly
since it's not on system PATH), confirmed via `hf auth whoami` →
`user=EllieJean`. Reran the smoke test after login: the error changed
from **401 "not logged in"** to **403 "authenticated but not in the
authorized list"** — i.e. login succeeded but Meta hasn't approved the
`facebook/sam3` access request yet (manual-approval gate, per
`HfApi().model_info('facebook/sam3').gated == 'manual'`). This is purely
a waiting-on-Meta's-review-queue blocker now, not a config/auth problem.

**Official next step for the user:** wait for approval on
https://huggingface.co/facebook/sam3 (request already submitted), or
check that page to confirm a request was actually made. No further local
action needed — once approved, the existing login will just start
working.

Rerun once approved: `.venv/Scripts/python.exe scripts/smoke_test.py --mode eager`

### Design decisions
- Used the bundled `assets/videos/0001` JPEG sequence (270 frames) instead
  of downloading external video — avoids network dependency, matches the
  official example notebook's own test clip, and "person" is the prompt
  the notebook itself demonstrates on this exact clip (dancers).
- Kept eager and compile modes in one script (`--mode` flag) rather than
  two scripts, since they must share checkpoint/input/prompt/device per
  the user's requirement to control for everything except the compile
  flag.
- Did not attempt to work around the HF gate (e.g., searching for mirror
  checkpoints) — respecting the "official next step" instruction and
  because a gated model used without proper access would be improper.

### Code trace findings (for later compile-comparison work)
- **Call chain**: `build_sam3_video_predictor()` (`sam3/model_builder.py:818`)
  → `Sam3VideoPredictorMultiGPU.__init__` (`sam3/model/sam3_video_predictor.py:111-139`)
  → `Sam3VideoPredictor.__init__` (same file, line 25) builds model via
  `build_sam3_video_model()` (`model_builder.py:676-815`).
  `handle_request`/`handle_stream_request` dispatch is in
  `sam3/model/sam3_base_predictor.py` (`Sam3BasePredictor`, shared by
  SAM3 and SAM3.1): `start_session` → `init_state` (line 140),
  `add_prompt` (line 170) runs under `torch.autocast(bfloat16)` and calls
  `self.model.add_prompt`, `propagate_in_video` (line 302) is a generator
  that calls `self.model.propagate_in_video` for forward/backward passes.
- **Video model internals**: `sam3/model/sam3_video_inference.py`,
  class `Sam3VideoInference` (line 30) / `Sam3VideoInferenceWithInstanceInteractivity`
  (line 1028). `init_state` at line 55, `propagate_in_video` at line 253,
  `_run_single_frame_inference` at line 360, `_postprocess_output`
  (builds `out_binary_masks`, line 432-524).
- **Compile flag entry point**: `propagate_in_video` (line 253) calls
  `self._compile_model()` at **line 269**, deliberately NOT in
  `add_prompt`, so the first prompt call always runs eager to "fill in
  the decoder buffers" (their comment) before any compilation happens.
  `_compile_model()` itself is at **`sam3_video_inference.py:573-651`**:
  sets `torch._dynamo.config.cache_size_limit=128`,
  `accumulated_cache_size_limit=2048`, `capture_scalar_outputs=True`,
  and critically **`suppress_errors=True`** — this is the silent-fallback
  mechanism: Dynamo graph breaks/compile errors fall back to eager
  silently rather than raising. To detect silent fallback in later
  sessions: run with `TORCH_LOGS=graph_breaks,recompiles` (or
  `torch._dynamo.config.suppress_errors=False` temporarily) and compare
  wall-clock/profiler traces, since `_compile_model()` gives no feedback
  on its own. `scripts/bench_compile.py:136` references exactly this
  (`TORCH_LOGS=recompiles`) for validating zero serve-time recompiles.
- **What gets compiled** (all with `mode="max-autotune"` except where
  noted), all inside `_compile_model`:
  detector vision backbone forward (line 595), transformer encoder
  (line 602) and decoder (line 609, `dynamic=False`), segmentation head
  (line 618); tracker maskmem backbone (line 627,
  `max-autotune-no-cudagraphs`), tracker transformer encoder (line 634,
  `max-autotune-no-cudagraphs`, `dynamic=True`), tracker SAM mask decoder
  (line 644, `max-autotune`, `dynamic=False`).
- **Vision backbone wrapper**: `sam3/model/vl_combiner.py`,
  `SAM3VLBackbone.__init__` (line 26) takes `compile_visual: bool` — if
  true, wraps the whole visual neck in `torch.compile(visual)` at
  **line 42**. This is a *second*, separate compile path from
  `_compile_model()`'s per-submodule compilation — `compile_visual` is
  not wired up by any of the builders in `model_builder.py` (always
  default `False`), so it's currently dead/unused for the backbone-only
  idea. The actual ViT trunk (`sam3/model/vitdet.py`, class `ViT`, line
  759) has its own `compile_mode` param (line 802) that wraps
  `self.forward` directly at **line 952-955** when non-None;
  `model_builder.py`'s `_create_vit_backbone` (line 77) and
  `_create_vision_backbone` (line 512) both thread a `compile_mode`
  through, but `build_sam3_video_model` (line 676) does NOT currently
  pass `compile_mode` into `_create_vision_backbone()` (line 706) — only
  the *image* model builder (`build_sam3_image_model`, line 573) wires
  `compile=True` → `compile_mode="default"` → the ViT-level compile.
  **This is exactly where a backbone-only compile option for video would
  need to be added**: either (a) thread a `compile_mode` kwarg through
  `build_sam3_video_model` → `_create_vision_backbone` → `_create_vit_backbone`
  (compiles only the ViT trunk's `.forward`, bypassing `_compile_model()`'s
  encoder/decoder/segmentation-head/tracker compilation entirely), or
  (b) set `compile_visual=True` on the `SAM3VLBackbone` used by the video
  detector (`model_builder.py:709`, `backbone = SAM3VLBackbone(...)`),
  which compiles the whole neck (ViT + Sam3DualViTDetNeck) as one graph.
  Route (a) is more surgical/backbone-only; route (b) is coarser.
- **Upstream benchmark**: `scripts/bench_compile.py` (full file read) is
  Meta's own encoder-focused eager-vs-compile A/B harness. Builds
  `Sam3VideoPredictorMultiGPU` directly with `compile=do_compile`,
  synthesizes a moving-circles test video (no checkpoint-independent
  assets needed), measures steady-state propagation FPS/latency/peak
  memory via `scripts/measure_speed.py` (`main_loop`,
  `max_memory_allocated`, `synthesize_video_data` — not yet read this
  session). Has `--native-warmup` (calls `_compile_model()` directly,
  mirrors prod warmup) and `--full-warmup` (`model_wrapper.model.warm_up_compilation()`,
  object-count sweep) flags — useful reference for how Meta itself
  structures compile-mode timing trials. This is the natural template to
  adapt for the later eager/backbone-only/stock-compile three-way
  comparison.

### Repo cleanup (same session, after initial install/trace)
- Added `.gitignore` at project root. Ignores: `.venv/` (71MB), `sam3/`
  (71MB cloned repo — has its own `.git` history, so nested-repo; decided
  **not** to vendor or submodule it, just re-clone from the documented
  commit hash when needed), `.claude/` (machine-local tool config),
  plus standard `__pycache__/`, `*.pyc`, HF/torch cache dirs, `*.pt`/`*.pth`,
  OS cruft.
- Tracked files: `.gitignore`, `AGENTS.md`, `CLAUDE.md`, `MEMORY.md`,
  `env_freeze.txt`, `scripts/smoke_test.py`. `outputs/` exists but is
  still empty (no mask output yet — blocked on HF approval), so nothing
  from it is tracked yet; it'll pick up the overlay PNGs automatically
  once the smoke test runs successfully.
- Scanned `env_freeze.txt` for secrets before staging — clean (the one
  substring match on "token" was the `asttokens` package name).

### Open questions for next time
1. Has HF access been approved yet? Auth itself is already confirmed
   working (`hf auth whoami` → `EllieJean`) — just rerun
   `scripts/smoke_test.py --mode eager` to check.
2. After a successful eager run, run `--mode compile` per step 5 of the
   original plan (feasibility only, record success/timeout/fallback/OOM
   — don't do full perf trials yet).
3. For the later backbone-only compile patch: decide between route (a)
   (thread `compile_mode` into `build_sam3_video_model`'s ViT) vs route
   (b) (`compile_visual=True` on `SAM3VLBackbone`) — leaning toward (a)
   since it's more surgical and matches what "backbone-only" should mean,
   but route (b) is a one-line flip worth trying first as a sanity check.
4. Should read `scripts/measure_speed.py` before building the 3-way
   comparison harness — `bench_compile.py` depends on it and it's not
   yet been read this session.
5. 12GB VRAM (not 16GB as originally assumed) may constrain how large a
   test clip / how many tracked objects the later max-autotune compile
   trials can use without OOM — worth a dedicated memory-headroom check
   once eager baseline numbers exist.
6. `env_freeze.txt` at project root has the exact pinned versions from
   this session (torch, torchvision, triton-windows, numpy, pandas, sam3,
   etc.) — use it to recreate the environment if `.venv` is ever rebuilt.

## Session 2026-11-22: HF access cleared, both smoke-test modes run

### Goal
Resume from the HF-approval blocker: confirm `facebook/sam3` access,
rerun `scripts/smoke_test.py --mode eager` to get a real end-to-end
result, then do the step-5 compile-mode feasibility check (success /
timeout / fallback / OOM only — no perf trial yet).

### State: both modes run; eager fully succeeded, compile hit mid-run
recompilation stalls and timed out at 600s (not OOM, not a crash)

**What's done:**
- Confirmed HF access approved: `hf_hub_download(repo_id="facebook/sam3",
  filename="sam3.pt")` succeeded directly (previously hit the 403
  "authenticated but not in the authorized list" error from last
  session). Checkpoint now cached at
  `~/.cache/huggingface/hub/models--facebook--sam3/snapshots/<hash>/sam3.pt`
  (~4.3GB blob). `hf auth whoami` still reports `EllieJean`, unchanged.
- **Eager mode: full success.** `scripts/smoke_test.py --mode eager`
  ran the complete pipeline — model build (~13s), 270-frame load (~7s),
  `add_prompt(text="person")` on frame 0 detected 4 objects, full
  270-frame `propagate_in_video` in 125.96s (~2.14 fps throughput over
  the whole loop), peak CUDA memory 10333.5 MiB (well inside the 12GB
  4070 SUPER budget, ~1.9GB headroom), first nonempty mask at frame 0
  with shape `(4, 720, 1280)` and 220552 nonzero pixels across 4 objects.
  Overlay PNG written to `outputs/smoke_test_eager_overlay.png` (966KB) —
  **this is a real result, not a stub**, confirming the whole pipeline
  (checkpoint load → text prompt → video tracker → mask decode →
  visualization) works on this hardware.
- **Compile mode: feasibility check completed, hit TimeoutError (not
  OOM/crash) at 600s.** `scripts/smoke_test.py --mode compile --timeout
  600` ran: model build, frame load, `add_prompt` (4 objects, same as
  eager — compile doesn't touch the first prompt call, consistent with
  the `_compile_model()` trace from last session: it's invoked inside
  `propagate_in_video`, not `add_prompt`), then propagation. Early
  frames progressed at ~3.1 it/s (comparable/faster than eager's ~2.4
  it/s steady-state) once warmed up, but **around frame 172-173 new
  Inductor `AUTOTUNE addmm(...)` events fired for previously-unseen
  tensor shapes** (seen in log: shapes shifted from `8x256`/`8x128`
  early on to `5184x...`/`5188x...` later — consistent with object
  count or sequence-length changes triggering fresh kernel search),
  each costing several seconds of wall-clock (one gap recorded at
  ~9.36s for a single frame, another ~3.85s/it) before the 600s ceiling
  was hit at 159 frames counted (progress bar showed up to 172/270 by
  then, since the timeout check and the tqdm counter aren't perfectly
  synced). **GPU memory returned to baseline (~1.16GB used, same as
  idle) after the run** — confirms this is a genuine recompilation/
  retuning stall, not a memory crash or leak.
- **What this means for later work:** this is exactly the kind of
  "silent recompile" case flagged in last session's trace notes
  (`_compile_model()` sets `suppress_errors=True`). The mid-video
  AUTOTUNE events are *not* silent (they print), but they **do** show
  that `max-autotune` is re-tuning kernels well after the documented
  frame-0 warmup, i.e. warmup via the first `add_prompt` call is
  insufficient to avoid serve-time recompilation when object count /
  internal tensor shapes change over the course of a video. This is a
  concrete, measured instance of exactly the risk the project's planned
  three-way eager/backbone-only/stock-compile comparison needs to
  quantify — worth rerunning with `TORCH_LOGS=recompiles` (per last
  session's note on `scripts/bench_compile.py:136`) to get an exact
  recompile count/cause rather than inferring from AUTOTUNE shape
  changes.
- Verified `outputs/` and `.gitignore` still correct; only the eager
  overlay PNG is on disk (compile mode's TimeoutError path in
  `smoke_test.py` doesn't reach the `save_masklet_image` call, so no
  compile-mode overlay exists yet — not a bug, just that path requires a
  full completion).

### Design decisions
- Used `--timeout 600` for the compile feasibility check rather than
  letting it run to completion — per the plan's step 5 scope ("success/
  timeout/fallback/OOM only, don't do full perf trials yet"). 600s was
  picked as roughly 5x eager's full-video time (126s), generous enough
  to absorb expected compile warmup/autotune overhead while still
  bounding runtime for a feasibility check.
- Did not rerun compile mode with a longer timeout or investigate the
  recompile trigger further this session — that belongs to the later
  dedicated compile-mode comparison work, not this checkpoint-unblocking
  session.

### Open questions for next time
1. **Main unblocked next step**: build the actual 3-way comparison
   harness (eager / backbone-only compile / stock compile) using
   `scripts/bench_compile.py` and `scripts/measure_speed.py` as
   templates, per last session's trace of the compile entry points.
   Both eager and compile-mode feasibility are now confirmed to run on
   this hardware, so there's no remaining blocker to starting this.
2. Should the comparison harness run with `TORCH_LOGS=recompiles` (or
   `torch._dynamo.config.suppress_errors=False`) by default, given that
   this session caught a real mid-video recompile stall under default
   settings? Leaning yes — the stall found this session suggests
   recompile visibility should be on by default for the comparison, not
   opt-in.
3. For the backbone-only compile patch, still need to decide between
   route (a) (thread `compile_mode` into `build_sam3_video_model`'s ViT)
   vs route (b) (`compile_visual=True` on `SAM3VLBackbone`) — unchanged
   from last session, not yet investigated further.
4. Should still read `scripts/measure_speed.py` before building the
   3-way harness — not yet read as of this session either.
5. Given the mid-video recompile stall observed here, worth deciding
   whether the comparison should track "time to Nth frame" curves
   (to show the stall shape) rather than just total elapsed time /
   average fps, since a single aggregate number would hide this effect.
6. 12GB VRAM headroom looks comfortable for eager (10.3GB peak, ~1.9GB
   free) but compile mode's peak was never measured this session (the
   TimeoutError path exits before reaching the peak-memory print) —
   worth checking compile-mode peak memory specifically once the
   recompile-stall question above is resolved, since `max-autotune`
   kernel search can transiently use extra memory during autotuning.

### Addendum: formal proposal now exists, supersedes some open questions above
`CS_587_Final_Project_Proposal.pdf` (added to repo root 2026-11-22, author
Ellie Larence, title "Selective Compilation for SAM 3 Video Inference")
locks in scope, hypothesis, and schedule. It directly cites this
session's own numbers (125.96s eager full-propagation time, 600s compile
timeout), so project plan and this session's empirical results are
already in sync. Key takeaways for future sessions:

- **Research question**: does compiling only SAM3's vision backbone beat
  Meta's stock (full) compile policy on total time-to-mask for a fixed
  270-frame/1280x720 clip prompted with "person"?
- **Hypothesis (two-part, both must hold)**: (1) cold-process,
  empty-Inductor-cache backbone-only run beats stock-compile's
  launch-to-final-mask time by >=20% on the 270-frame clip; (2) in a
  second, same-process warm session (predictor/compiled code already
  loaded), backbone-only retains >=90% of stock's warmed propagation FPS.
- **Chosen patch approach resolves last session's open route (a) vs (b)
  question — it's actually a third, simpler route (c)**: add a
  backbone-only option directly at the existing `_compile_model()` entry
  point (`sam3/model/sam3_video_inference.py:573-651`), reusing the same
  wrapper/compiler settings already used there for the backbone line
  (line 595) but skipping the calls that compile the transformer
  encoder/decoder, segmentation head, and tracker submodules. This is
  neither route (a) (threading `compile_mode` through
  `build_sam3_video_model`'s ViT builder chain) nor route (b)
  (`compile_visual=True` on `SAM3VLBackbone`) from the 2026-11-20 trace
  notes — **go with this `_compile_model()`-level flag instead**, it's
  the proposal's committed design.
- **Evaluation plan specifics**: 5 trials per mode (eager / backbone-only
  / stock), alternating order; each cold trial = fresh process + empty
  TorchInductor cache; after cold run, open a fresh video session in the
  *same* process to measure warmed propagation FPS with compiled code
  already resident. Diagnostic runs use `TORCH_LOGS=recompiles` to
  distinguish real guard-failure recompiles from Inductor autotuning
  events (both were conflated by inference in this session's log —
  needs the env var to pin down for real). Common timeout set before
  formal trials; incomplete runs reported with last completed frame;
  compare frame coverage, unmatched object IDs, and mask IoU against
  eager output per trial so a timed-out/degraded run can't be
  miscounted as a speedup. Two measured variables: (1) process-launch to
  final-mask wall time, (2) warmed propagation FPS.
- **Also run a short-prefix workload** (subset of the 270 frames) to see
  how the comparison shifts when there's less video to amortize compile
  cost — not just the full clip.
- **Schedule** (from proposal, all dates 2026): Nov 22-26 implement the
  backbone-only `_compile_model()` patch + per-frame timing/timeout
  logging, verify mask agreement (vs eager) on a short clip; Nov 27-30
  run all three modes on the short clip, confirm which functions
  actually compiled (via logs/`TORCH_LOGS`), submit status update; Dec
  1-5 run the full repeated-trial comparison, produce the elapsed-time
  figure + peak-VRAM table; Dec 6-11 analyze completed runs/timeouts,
  document rerun steps, write final report. **We are at the very start
  of the Nov 22-26 window as of this session** — the literal next task
  is implementing the `_compile_model()` backbone-only flag, not yet
  started.
- **Fallback already specified** if stock compile can't be made to
  finish even with a longer timeout: narrow to eager-vs-backbone-only
  only, on the short-prefix workload, to test whether selective
  compilation at least repays its own startup cost — don't block the
  whole project on stock compile finishing.
- AI-workflow section of the proposal explicitly commits to: tracing the
  predictor and drafting the patch via Claude Code (already underway),
  reviewing the diff against the stock compile path, checking compiler
  logs/traces to verify only the intended functions compiled, and
  comparing resulting masks against eager output before trusting any
  performance numbers — i.e. mask-correctness checks are a hard
  prerequisite before reporting any timing result, not an optional
  nice-to-have.
