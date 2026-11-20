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
