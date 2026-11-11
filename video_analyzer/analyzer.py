"""
영상 분석 모듈 (로컬 GPU / stub).

Modes
-----
- local : 이 PC에서 Qwen2.5-VL-7B 추론 (RTX 5070 32GB 권장 설정)
- stub  : fixtures JSON 반환 (GPU 없는 팀원·시연 대체)

팀: from video_analyzer import analyze_clip
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from .prompts import DEFAULT_VIDEO_PROMPT
from .runtime_config import (
    CAP_PIXELS_PER_FRAME,
    DEFAULT_FPS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_PIXELS,
    MODEL_ID,
    TRY_FLASH_ATTN,
    USE_4BIT,
)
from .schema import VideoAnalysisResult, normalize_result
from .video_io import load_video_frames

Mode = Literal["stub", "local"]

DEFAULT_MODEL = MODEL_ID
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_STUB_PATH = REPO_ROOT / "fixtures" / "sample_result1.json"

_local_model = None
_local_processor = None


def analyze_clip(
    video_path: str,
    prompt: str | None = None,
    *,
    mode: Mode | None = None,
    model_id: str = DEFAULT_MODEL,
    stub_path: str | Path | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    fps: float = DEFAULT_FPS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> VideoAnalysisResult:
    """
    Analyze a home-cam clip → fixed schema dict.

    mode 기본값: env VIDEO_ANALYZER_MODE, 없으면 \"local\".
    GPU 없는 PC에서는 mode=\"stub\" 또는 VIDEO_ANALYZER_MODE=stub.
    """
    resolved_mode: Mode = mode or os.getenv("VIDEO_ANALYZER_MODE", "local")  # type: ignore[assignment]
    text_prompt = prompt or DEFAULT_VIDEO_PROMPT

    if resolved_mode == "stub":
        return _analyze_stub(stub_path or DEFAULT_STUB_PATH)

    if resolved_mode == "local":
        return _analyze_local(
            video_path,
            text_prompt,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            fps=fps,
            max_pixels=max_pixels,
        )

    raise ValueError(f"Unknown mode: {resolved_mode!r}. Use stub|local.")


def _analyze_stub(stub_path: str | Path) -> VideoAnalysisResult:
    path = Path(stub_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Stub result not found: {path}. "
            "Run scripts/run_local.py and save into fixtures/."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return normalize_result(data)


def _analyze_local(
    video_path: str,
    prompt: str,
    *,
    model_id: str,
    max_new_tokens: int,
    fps: float,
    max_pixels: int,
) -> VideoAnalysisResult:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(video_path)

    model, processor = _get_local_model(model_id)
    raw = run_qwen_vl_inference(
        model=model,
        processor=processor,
        video_path=str(path.resolve()),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        fps=fps,
        max_pixels=max_pixels,
    )
    parsed = try_parse_json_object(raw)
    if parsed is None:
        return normalize_result({"message": raw, "confidence": 0.0})
    return normalize_result(parsed)


def _get_local_model(model_id: str):
    global _local_model, _local_processor
    if _local_model is not None and _local_processor is not None:
        return _local_model, _local_processor

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Use mode='stub' or fix the torch CUDA install.")

    env_4bit = os.getenv("VIDEO_ANALYZER_4BIT")
    use_4bit = USE_4BIT if env_4bit is None else env_4bit == "1"

    # 32GB: 전체 모델을 GPU0에 올림 (device_map=auto 의 CPU 오프로딩 방지)
    kwargs: dict[str, Any] = {"device_map": {"": 0}}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    if TRY_FLASH_ATTN and not use_4bit:
        kwargs["attn_implementation"] = "flash_attention_2"

    try:
        _local_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    except Exception:
        kwargs.pop("attn_implementation", None)
        _local_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)

    _local_processor = AutoProcessor.from_pretrained(model_id)
    return _local_model, _local_processor


def run_qwen_vl_inference(
    *,
    model,
    processor,
    video_path: str,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    fps: float = DEFAULT_FPS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> str:
    """
    Qwen2.5-VL video → text.

    Windows + 최신 torchvision 에서는 io.read_video 가 없어 qwen-vl-utils 기본
    경로가 깨지므로, PyAV로 프레임을 뽑아 video=list[PIL] 로 넣는다.
    """
    from qwen_vl_utils import process_vision_info

    frames = load_video_frames(video_path, fps=fps, max_pixels=max_pixels)
    print(f"decoded frames: {len(frames)} @ target fps={fps}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "fps": fps,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    fps_val = video_kwargs.get("fps")
    if isinstance(fps_val, (list, tuple)):
        video_kwargs["fps"] = float(fps_val[0])

    proc_kwargs = dict(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    if CAP_PIXELS_PER_FRAME:
        proc_kwargs["cap_pixels_per_frame"] = True
    try:
        inputs = processor(**proc_kwargs)
    except TypeError:
        proc_kwargs.pop("cap_pixels_per_frame", None)
        inputs = processor(**proc_kwargs)

    import torch

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
    out_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return out_text.strip()


def try_parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None
