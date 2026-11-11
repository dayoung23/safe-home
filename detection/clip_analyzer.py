"""
detection/clip_analyzer.py — 트리거 후에만 쓰는 영상 관측 엔진 (라이브러리)

vision.py 가 모션/소리 트리거로 잘라 낸 짧은 클립 1개를 받아
"무엇이 보이는지"만 관측해 dict 로 돌려준다. 위험 등급은 정하지 않는다 (judge 담당).

모드
  stub  : detection/fixtures/*.json 에 미리 추론해 둔 관측 결과를 반환 (시연 기본, GPU·지연 없음)
  local : Qwen2.5-VL 로 클립을 실제 추론 (GPU 권장, 클립당 수~수십 초)

미리 추론 결과(fixture) 만들기 — 시연 클립을 local 로 한 번 돌려 저장:
  python -m detection.clip_analyzer clips/demo_intruder.mp4 --save detection/fixtures/02_intruder.json

반환 형식 (관측 결과)
  {"type": "person|fall|intruder|suspicious|fire|motion|none",
   "scene": "거실", "persons": [{"gender": "...", "desc": "..."}],
   "behavior": "...", "message": "사람이 읽는 한 문장", "confidence": 0~1}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
VISION_TYPES = ("person", "fall", "intruder", "suspicious", "fire", "motion", "none")

MODEL_ID = os.getenv("VL_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
VIDEO_FPS = float(os.getenv("VL_VIDEO_FPS", "1.0"))      # 클립에서 뽑을 초당 프레임 수
MAX_PIXELS = int(os.getenv("VL_MAX_PIXELS", str(360 * 420)))
MAX_NEW_TOKENS = 384

PROMPT = """너는 가정용 홈캠 영상 관측기야. 위험하다/안전하다는 판단은 하지 말고, 보이는 것만 기록해.
아래 JSON 하나로만 답해. 다른 말은 쓰지 마.
{
  "type": "person | fall | intruder | suspicious | fire | motion | none 중 하나",
  "scene": "장소 (예: 거실, 현관, 침실)",
  "persons": [{"gender": "남성|여성|불명", "desc": "옷차림과 행동 한 문장"}],
  "behavior": "클립 안에서 일어난 행동 요약 한 문장",
  "message": "사용자에게 보여 줄 관측 문장 한 개",
  "confidence": 0.0~1.0
}
type 기준: 사람이 쓰러짐=fall, 창문·문을 강제로 넘거나 여는 사람=intruder,
얼굴을 가리거나 서성이는 등 수상한 행동=suspicious, 불꽃·연기=fire,
그냥 사람이 보임=person, 사람 없이 움직임만=motion, 아무 변화 없음=none."""

# type 이 없는 관측(예전 형식 fixture 등)을 위한 키워드 보정
_TYPE_KEYWORDS = [
    ("fire", ("불꽃", "연기", "화재", "불이")),
    ("fall", ("쓰러", "넘어", "바닥에 누워", "의식")),
    ("intruder", ("침입", "창문을 넘", "창틀", "강제로", "문을 부수", "따고")),
    ("suspicious", ("마스크", "얼굴을 가", "서성", "배회", "두리번", "수상")),
]


# ───────────────────────── 공통: 결과 정규화 ─────────────────────────
def _guess_type(obs: dict[str, Any]) -> str:
    text = " ".join(
        [str(obs.get("message") or ""), str(obs.get("behavior") or "")]
        + [str(p.get("desc") or "") for p in obs.get("persons") or [] if isinstance(p, dict)]
    )
    for t, words in _TYPE_KEYWORDS:
        if any(w in text for w in words):
            return t
    if obs.get("persons"):
        return "person"
    return "motion" if text.strip() else "none"


def normalize(obs: dict[str, Any]) -> dict[str, Any]:
    """모델/fixture 출력 → 관측 결과 형식으로 통일."""
    msg = obs.get("message")
    if isinstance(msg, list):
        msg = " ".join(str(x) for x in msg if x)
    persons = obs.get("persons") or []
    if not isinstance(persons, list):
        persons = []
    out = {
        "type": str(obs.get("type") or "").strip().lower(),
        "scene": str(obs.get("scene") or ""),
        "persons": persons,
        "behavior": str(obs.get("behavior") or ""),
        "message": str(msg or ""),
        "confidence": float(obs.get("confidence") or 0.0),
    }
    if out["type"] not in VISION_TYPES:
        out["type"] = _guess_type(out)
    out["confidence"] = min(max(out["confidence"], 0.0), 1.0)
    return out


# ───────────────────────── stub: 미리 추론된 fixture ─────────────────────────
def list_fixtures() -> list[Path]:
    return sorted(FIXTURE_DIR.glob("*.json"))


def load_fixture(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return normalize(json.load(f))


# ───────────────────────── local: Qwen2.5-VL 실추론 ─────────────────────────
_model = None
_processor = None
_load_lock = threading.Lock()
_infer_lock = threading.Lock()   # GPU 1장 → 추론은 한 번에 하나


def _load_model():
    global _model, _processor
    with _load_lock:
        if _model is None:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            print(f"[clip_analyzer] {MODEL_ID} 로딩 중... (처음 한 번만)")
            _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID, torch_dtype="auto", device_map="auto"
            )
            _processor = AutoProcessor.from_pretrained(MODEL_ID)
            print("[clip_analyzer] 모델 준비 완료")
    return _model, _processor


def _parse_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"message": text.strip(), "confidence": 0.3}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"message": text.strip(), "confidence": 0.3}


def _infer_local(clip_path: str) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    model, processor = _load_model()
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": str(clip_path), "fps": VIDEO_FPS, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:   # qwen-vl-utils 신버전: fps 등 video_kwargs 도 같이 돌려줌
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs = {}
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt", **video_kwargs,
    ).to(model.device)

    with _infer_lock:
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return _parse_json(raw)


# ───────────────────────── 진입점 ─────────────────────────
def analyze_clip(clip_path: str | None, mode: str = "stub", fixture: str | Path | None = None) -> dict[str, Any]:
    """
    클립 1개 관측.
      stub : fixture 경로의 미리 추론 결과 반환 (clip_path 는 안 씀)
      local: clip_path 를 Qwen2.5-VL 로 추론
    """
    if mode == "stub":
        if fixture is None:
            raise FileNotFoundError("stub 모드인데 fixture 가 없습니다 (detection/fixtures/*.json)")
        return load_fixture(fixture)
    if mode == "local":
        if not clip_path or not Path(clip_path).exists():
            raise FileNotFoundError(f"클립 파일이 없습니다: {clip_path}")
        return normalize(_infer_local(str(clip_path)))
    raise ValueError(f"알 수 없는 mode: {mode}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="클립 1개를 local 추론 → (선택) fixture 로 저장")
    ap.add_argument("clip")
    ap.add_argument("--save", help="결과를 저장할 fixture 경로 (예: detection/fixtures/02_intruder.json)")
    a = ap.parse_args()
    result = analyze_clip(a.clip, mode="local")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[저장] {a.save}")
