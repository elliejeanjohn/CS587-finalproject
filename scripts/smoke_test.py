"""Minimal GPU smoke test for SAM3 video predictor (sam3 base, not 3.1 multiplex).

Loads the bundled assets/videos/0001 clip, prompts "person" on frame 0,
propagates through the video, verifies a non-empty binary mask, and saves
an overlay PNG. Reports frame count, mask shape, elapsed time, and peak
CUDA memory.

Usage:
    python scripts/smoke_test.py --mode eager
    python scripts/smoke_test.py --mode compile
"""

import argparse
import os
import sys
import time

import torch

SAM3_REPO = os.path.join(os.path.dirname(__file__), "..", "sam3")
sys.path.insert(0, SAM3_REPO)

from sam3.model_builder import build_sam3_video_predictor  # noqa: E402
from sam3.visualization_utils import save_masklet_image  # noqa: E402

VIDEO_PATH = os.path.join(SAM3_REPO, "assets", "videos", "0001")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
PROMPT_TEXT = "person"


def run(mode: str, timeout_sec: float | None):
    do_compile = mode == "compile"
    out_path = os.path.join(OUT_DIR, f"smoke_test_{mode}_overlay.png")
    os.makedirs(OUT_DIR, exist_ok=True)

    assert torch.cuda.is_available(), "CUDA is required for this smoke test"
    torch.cuda.reset_peak_memory_stats()

    print(f"Building SAM3 video predictor (compile={do_compile})...")
    predictor = build_sam3_video_predictor(compile=do_compile)

    t0 = time.time()

    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=VIDEO_PATH)
    )
    session_id = response["session_id"]
    print(f"session started: {session_id}")

    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=PROMPT_TEXT,
        )
    )
    add_prompt_out = response["outputs"]
    print(
        f"add_prompt on frame 0 -> "
        f"{len(add_prompt_out.get('out_obj_ids', []))} object(s) detected"
    )

    processed_frames = 0
    first_nonempty = None
    for stream_resp in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        if timeout_sec is not None and (time.time() - t0) > timeout_sec:
            raise TimeoutError(
                f"propagation exceeded {timeout_sec}s after {processed_frames} frames"
            )
        frame_idx = stream_resp["frame_index"]
        out = stream_resp["outputs"]
        processed_frames += 1
        masks = out["out_binary_masks"]  # (N, H, W) bool numpy array
        if masks.shape[0] > 0 and masks.any() and first_nonempty is None:
            first_nonempty = (frame_idx, out)

    elapsed = time.time() - t0
    peak_mem_bytes = torch.cuda.max_memory_allocated()

    predictor.handle_request(request=dict(type="close_session", session_id=session_id))

    print(f"\n=== RESULTS ({mode}) ===")
    print(f"processed_frames: {processed_frames}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print(f"peak_cuda_memory_allocated_MiB: {peak_mem_bytes / 1024**2:.1f}")

    if first_nonempty is None:
        print("FAILURE: no nonempty mask found across propagated frames")
        sys.exit(1)

    frame_idx, out = first_nonempty
    masks = out["out_binary_masks"]
    obj_ids = out["out_obj_ids"]
    print(f"first_nonempty_frame_index: {frame_idx}")
    print(f"mask_shape: {masks.shape}")
    print(f"object_ids: {obj_ids.tolist()}")
    print(f"mask_dtype: {masks.dtype}")
    print(f"mask_nonzero_pixels: {int(masks.sum())}")

    frame_path = os.path.join(VIDEO_PATH, f"{frame_idx}.jpg")
    save_masklet_image(frame_path, out, out_path, alpha=0.5, frame_idx=frame_idx)
    print(f"overlay_saved_to: {out_path}")
    return {
        "mode": mode,
        "processed_frames": processed_frames,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_allocated_MiB": peak_mem_bytes / 1024**2,
        "mask_shape": masks.shape,
        "overlay_path": out_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eager", "compile"], default="eager")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="abort propagation loop after this many seconds (feasibility check)",
    )
    args = parser.parse_args()

    try:
        run(args.mode, args.timeout)
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n=== RESULT ({args.mode}): OOM ===\n{e}")
        sys.exit(2)
    except TimeoutError as e:
        print(f"\n=== RESULT ({args.mode}): TIMEOUT ===\n{e}")
        sys.exit(3)
    except Exception as e:
        print(f"\n=== RESULT ({args.mode}): ERROR ({type(e).__name__}) ===\n{e}")
        raise


if __name__ == "__main__":
    main()
