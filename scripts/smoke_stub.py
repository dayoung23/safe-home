# GPU 없이 테스트

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video_analyzer import analyze_clip


def main() -> None:
    result = analyze_clip("clips/demo.mp4", mode="stub")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert result["source"] == "video"
    assert "message" in result
    print("\n[OK] stub analyze_clip() works.")


if __name__ == "__main__":
    main()
