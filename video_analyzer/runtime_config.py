"""
RTX 5070 32GB 로컬 최대 성능 프리셋 (Qwen2.5-VL-7B).

- 4bit 미사용 → bf16 풀정밀
- fps / max_pixels / max_new_tokens 상향
"""

from __future__ import annotations

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# 32GB: False(bf16). VRAM 부족 시에만 True.
USE_4BIT = False

DEFAULT_FPS = 1.5
DEFAULT_MAX_PIXELS = 768 * 768
CAP_PIXELS_PER_FRAME = True
DEFAULT_MAX_NEW_TOKENS = 512

RECOMMENDED_CLIP_SECONDS_MAX = 12

# flash_attention_2 사용 시도 (미설치면 자동 폴백)
TRY_FLASH_ATTN = True
