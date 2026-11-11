# Backup — legacy vision stack

Moved out of the realtime path when YOLO replaced pixel-motion trigger + Qwen observation.

| Path | Role |
|------|------|
| `detection/motion_trigger.py` | 160x90 pixel-diff motion trigger |
| `detection/clip_analyzer.py` | Qwen2.5-VL / fixture stub observation |
| `detection/fixtures/` | Precomputed stub JSON |
| `detection/vision_legacy.py` | Old vision loop (motion → clip → Qwen/stub) |
| `video_analyzer/` | Older team Qwen package (sources may be incomplete) |

Runtime backup entry: `detection/qwen_backup.py` imports `clip_analyzer` from here when `VISION_MODE=qwen|local|stub`.
