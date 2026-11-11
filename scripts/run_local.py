"""
로컬 GPU에서 Qwen2.5-VL-7B로 클립 분석 → JSON 저장.

예:
  python scripts/run_local.py --video clips/demo1.mp4 --out fixtures/sample_result1.json
  python scripts/run_local.py --video clips/demo2.mp4 --out fixtures/sample_result2.json --clip-id id1_2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video_analyzer import analyze_clip
from video_analyzer.runtime_config import (
    DEFAULT_FPS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_PIXELS,
    MODEL_ID,
    USE_4BIT,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Qwen2.5-VL-7B video → JSON")
    parser.add_argument("--video", required=True, help="mp4 path")
    parser.add_argument(
        "--out",
        default="fixtures/sample_result.json",
        help="output JSON path",
    )
    parser.add_argument("--clip-id", default=None, help="override clip_id in result")
    parser.add_argument("--timestamp", default=None, help="override timestamp")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    print(f"model={MODEL_ID} 4bit={USE_4BIT} fps={args.fps} max_pixels={args.max_pixels}")
    print(f"loading + infer: {video} ...")

    result = analyze_clip(
        str(video.resolve()),
        mode="local",
        fps=args.fps,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
    )
    if args.clip_id:
        result["clip_id"] = args.clip_id
    if args.timestamp:
        result["timestamp"] = args.timestamp

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nsaved: {out.resolve()}")


if __name__ == "__main__":
    main()
