"""Video analysis output schema shared with RiskJudge / team pipeline."""

from __future__ import annotations

from typing import Any, TypedDict

DEMO_CLIP_ID = "id1_1"
DEMO_TIMESTAMP = "2026-11-11T18:13:05"
ALLOWED_GENDERS = frozenset({"남성", "여성", "불명"})


class PersonInfo(TypedDict, total=False):
    gender: str
    desc: str


class VideoAnalysisResult(TypedDict, total=False):
    source: str
    scene: str
    persons: list[PersonInfo]
    message: str
    confidence: float
    clip_id: str
    timestamp: str


REQUIRED_KEYS = (
    "source",
    "scene",
    "persons",
    "message",
    "confidence",
    "clip_id",
    "timestamp",
)


def empty_result() -> VideoAnalysisResult:
    return {
        "source": "video",
        "scene": "",
        "persons": [],
        "message": "",
        "confidence": 0.0,
        "clip_id": DEMO_CLIP_ID,
        "timestamp": DEMO_TIMESTAMP,
    }


def _normalize_persons(raw: Any) -> list[PersonInfo]:
    if not isinstance(raw, list):
        return []
    out: list[PersonInfo] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        gender = str(item.get("gender") or "").strip()
        desc = str(item.get("desc") or "").strip()

        # legacy: count/desc only → try recover gender from desc
        if not gender and desc:
            if desc in ("none", ""):
                continue
            if "불명" in desc and ("남성" in desc or "여성" in desc):
                # "불명 남성" 류 → 성별 추정 가능하면 성별만, 아니면 불명
                if "남성" in desc and "여성" not in desc:
                    gender = "남성"
                elif "여성" in desc and "남성" not in desc:
                    gender = "여성"
                else:
                    gender = "불명"
            elif desc.startswith("남성") or "남성" in desc[:4]:
                gender = "남성"
            elif desc.startswith("여성") or "여성" in desc[:4]:
                gender = "여성"
            else:
                gender = "불명"

        if gender not in ALLOWED_GENDERS:
            # strip illegal combos like "불명 남성"
            if gender.startswith("불명"):
                gender = "불명"
            elif "남성" in gender and "여성" not in gender:
                gender = "남성"
            elif "여성" in gender and "남성" not in gender:
                gender = "여성"
            else:
                gender = "불명"

        if desc in ("none",) and gender == "불명":
            continue
        out.append({"gender": gender, "desc": desc})
    return out


def normalize_result(data: dict[str, Any], *, model: str | None = None) -> VideoAnalysisResult:
    """Fill missing keys; drop legacy `actions`; clean persons.gender."""
    del model
    base = empty_result()

    if "message" not in data and data.get("summary"):
        data = {**data, "message": data["summary"]}

    # If model still returned actions and message is weak, join once (no dup prefer message)
    if (not data.get("message")) and isinstance(data.get("actions"), list):
        data = {
            **data,
            "message": " ".join(str(a) for a in data["actions"] if a),
        }

    for key in REQUIRED_KEYS:
        if key == "persons":
            continue
        if key in data and data[key] is not None:
            base[key] = data[key]  # type: ignore[literal-required]

    base["persons"] = _normalize_persons(data.get("persons"))
    base["source"] = "video"
    if not base.get("clip_id"):
        base["clip_id"] = DEMO_CLIP_ID
    if not base.get("timestamp"):
        base["timestamp"] = DEMO_TIMESTAMP

    try:
        base["confidence"] = float(base.get("confidence", 0.0))
    except (TypeError, ValueError):
        base["confidence"] = 0.0

    return base
