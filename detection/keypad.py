import os
import glob
import json
import datetime
import threading
import time
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

# AST 모델은 sound.py에서 한 번만 로드해 공유합니다. (realtime.py와 같이 써도 모델이 중복 로딩되지 않음)
try:
    from detection.sound import classifier      # 루트에서 python main.py로 실행할 때
except ImportError:
    from sound import classifier                # detection 폴더에서 단독 실행할 때

LOCK_TOP_LABELS = ["Single-lens reflex camera", "Mechanisms", "Clicking"]

# ── 사용자별 타건 프로필 (MongoDB users + keypad_samples) ──
REGISTER_TARGET = 5          # 등록 때 받을 입력 횟수
REGISTER_TIMEOUT_SEC = 180   # 등록 모드 자동 종료
MIN_SAMPLES = 3              # 이보다 샘플이 적은 사용자는 비교에서 제외
OWNER_SIMILARITY = 0.70      # 이 이상이면 등록 사용자로 인정
PROFILE_CACHE_SEC = 30       # DB 조회 캐시 (새 샘플이 저장되면 즉시 무효화)

_lock = threading.Lock()
_register = None             # {"user_id","name","count","target","done","started"}
_profiles_cache = {"t": 0.0, "profiles": None}


def _db():
    from db import database   # 단독 실행(.env 없음)에서도 분석은 되도록 지연 import
    return database


# ───────────────────────── 1. 등록 (auth.py 가 호출) ─────────────────────────

def start_register(user_id: str, name: str, target: int = REGISTER_TARGET) -> dict:
    """로그인한 사용자의 키패드 패턴 등록 시작. 이후 도어락 입력 target 회를 이 사용자 샘플로 저장."""
    global _register
    with _lock:
        _register = {"user_id": str(user_id), "name": name, "count": 0,
                     "target": int(target), "done": False, "started": time.time()}
        return _public_status()


def get_register_status():
    """{"user_id","name","count","target","done"} 또는 None (등록 중 아님)"""
    with _lock:
        _expire_register()
        return _public_status()


def _public_status():
    if _register is None:
        return None
    return {k: _register[k] for k in ("user_id", "name", "count", "target", "done")}


def _expire_register():
    global _register
    if _register and not _register["done"] and time.time() - _register["started"] > REGISTER_TIMEOUT_SEC:
        print(f"[keypad] 등록 시간 초과 → 종료 ({_register['name']} {_register['count']}/{_register['target']})")
        _register = None


def _record_register_sample(speed_info: dict):
    """등록 모드면 이번 입력을 샘플로 저장하고 상태 반환, 아니면 None."""
    with _lock:
        _expire_register()
        reg = _register
        if reg is None or reg["done"]:
            return None
        intervals = speed_info.get("beep_intervals") or []
        if len(intervals) < 2:
            return {**_public_status(), "saved": False}
        try:
            _db().add_keypad_sample(reg["user_id"], {
                "avg_interval_sec": speed_info["avg_interval_sec"],
                "beep_count": speed_info["beep_count"],
                "typing_speed": speed_info["typing_speed"],
                "intervals": intervals,
            })
        except Exception as e:
            print(f"[keypad] 샘플 저장 실패: {e}")
            return {**_public_status(), "saved": False}
        reg["count"] += 1
        reg["done"] = reg["count"] >= reg["target"]
        _profiles_cache["profiles"] = None          # 새 샘플 → 프로필 다시 계산
        return {**_public_status(), "saved": True}


# ───────────────────────── 2. 프로필 계산 & 비교 엔진 ─────────────────────────

def build_profile(samples: list) -> dict | None:
    """keypad_samples → {"pin_length","mean_intervals","std_intervals","sample_count"}.
    자릿수가 섞여 있으면 가장 많은 자릿수의 샘플만 사용."""
    seqs = [s["intervals"] for s in samples if len(s.get("intervals") or []) >= 2]
    if not seqs:
        return None
    lengths = [len(x) for x in seqs]
    n = max(set(lengths), key=lengths.count)
    same = np.array([x for x in seqs if len(x) == n], dtype=float)
    if len(same) < MIN_SAMPLES:
        return None
    return {
        "pin_length": n + 1,
        "mean_intervals": same.mean(axis=0).round(3).tolist(),
        "std_intervals": same.std(axis=0).round(3).tolist(),
        "sample_count": int(len(same)),
    }


def load_profiles() -> list:
    """등록된 모든 사용자의 프로필 [{"user_id","name",...profile}] (캐시)."""
    now = time.time()
    with _lock:
        cached = _profiles_cache["profiles"]
        if cached is not None and now - _profiles_cache["t"] < PROFILE_CACHE_SEC:
            return cached
    profiles = []
    try:
        db = _db()
        for u in db.get_users():
            prof = build_profile(db.get_keypad_samples(u["id"]))
            if prof:
                profiles.append({"user_id": u["id"], "name": u.get("name") or u.get("username"), **prof})
    except Exception as e:
        print(f"[keypad] 프로필 조회 실패 → 비교 없이 진행: {e}")
        return []
    with _lock:
        _profiles_cache.update(t=now, profiles=profiles)
    return profiles


def verify_keystroke_pattern(current_intervals: list, profile: dict) -> dict:
    """현재 비프 간격과 프로필 1개를 비교 → 유사도(0~1)와 일치 여부."""
    if not current_intervals or len(current_intervals) < 2:
        return {"is_owner": False, "similarity": 0.0, "status": "insufficient_data",
                "message": "타건 데이터 부족 (판정 불가)"}

    ref_means = np.array(profile["mean_intervals"], dtype=float)
    ref_stds = np.array(profile["std_intervals"], dtype=float)
    curr = np.array(current_intervals, dtype=float)

    if len(curr) != len(ref_means):
        return {"is_owner": False, "similarity": 0.15, "similarity_pct": 15.0, "status": "length_mismatch",
                "message": f"자릿수 불일치 (등록 {len(ref_means)+1}자리 vs 입력 {len(curr)+1}자리)"}

    # 구간별 Z-score (허용 표준편차 최소 0.04초) → 지수 감쇠로 유사도 환산
    z_scores = np.abs((curr - ref_means) / np.maximum(ref_stds, 0.04))
    avg_z = float(np.mean(z_scores))
    similarity = float(np.exp(-0.4 * avg_z))
    return {
        "is_owner": similarity >= OWNER_SIMILARITY,
        "similarity": round(similarity, 4),
        "similarity_pct": round(similarity * 100, 1),
        "avg_z_score": round(avg_z, 2),
        "status": "owner_verified" if similarity >= OWNER_SIMILARITY else "rhythm_mismatch",
    }


def identify_member(current_intervals: list) -> dict:
    """등록된 모든 사용자와 비교해 가장 비슷한 사람을 찾음.
    반환: verification dict + member_id / member_name (일치하는 사람 없으면 None)"""
    profiles = load_profiles()
    if not profiles:
        return {"is_owner": False, "similarity": 0.0, "status": "no_profile",
                "member_id": None, "member_name": None,
                "message": "등록된 타건 패턴 없음 (비교 생략)"}

    best, best_p = None, None
    for p in profiles:
        v = verify_keystroke_pattern(current_intervals, p)
        if best is None or v["similarity"] > best["similarity"]:
            best, best_p = v, p
    if best["status"] == "insufficient_data":
        return {**best, "member_id": None, "member_name": None}

    sim = best.get("similarity_pct", 0)
    if best["is_owner"]:
        msg = f"{best_p['name']}님 타건 리듬 일치 (유사도 {sim}%)"
        member_id, member_name = best_p["user_id"], best_p["name"]
    elif best["status"] == "length_mismatch":
        msg = f"등록된 사용자와 자릿수가 달라요 - 도용 의심 (가장 가까운 {best_p['name']}님 기준)"
        member_id = member_name = None
    else:
        msg = f"등록된 사용자 리듬과 불일치 - 도용 의심 (가장 가까운 {best_p['name']}님 {sim}%)"
        member_id = member_name = None
    return {**best, "member_id": member_id, "member_name": member_name,
            "closest": best_p["name"], "message": msg}


# ───────────────────────── 3. 오디오 분석 및 특징 추출 ─────────────────────────

def calculate_pattern_confidence(beep_count: int, intervals: list, speed: str) -> float:
    if beep_count < 2 or not intervals:
        return 0.30

    score = 0.0
    if 4 <= beep_count <= 10:
        score += 0.40
    elif beep_count in [3, 11, 12]:
        score += 0.25
    else:
        score += 0.10

    valid_range = [i for i in intervals if 0.18 <= i <= 2.2]
    ratio = len(valid_range) / len(intervals)
    score += ratio * 0.30

    if speed == "slow":
        hesitations = sum(1 for i in intervals if i >= 1.2)
        score += min(hesitations * 0.15, 0.30)
    elif speed == "normal":
        fast_hits = sum(1 for i in intervals if i < 0.8)
        score += min(fast_hits * 0.15, 0.30)
    else:
        score += 0.15

    return round(score, 4)


def analyze_beep_intervals(data, sample_rate: int):
    try:
        data = np.asarray(data)
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        start_cut = int(sample_rate * 0.2)
        d_clean = data[start_cut:] if len(data) > start_cut else data

        threshold = np.max(np.abs(d_clean)) * 0.25
        min_distance = int(sample_rate * 0.15)
        peaks, _ = find_peaks(np.abs(d_clean), height=threshold, distance=min_distance)

        peak_times = (peaks + start_cut) / sample_rate
        beep_count = len(peak_times)

        if beep_count >= 2:
            intervals = list(np.diff(peak_times))
            if len(intervals) >= 3 and intervals[0] >= 1.2:
                intervals = intervals[1:]
            if len(intervals) >= 3 and intervals[-1] >= 1.2:
                intervals = intervals[:-1]

            has_consecutive_fast = any(
                intervals[i] < 0.8 and intervals[i + 1] < 0.8
                for i in range(len(intervals) - 1)
            )
            has_consecutive_slow = any(
                intervals[i] >= 1.2 and intervals[i + 1] >= 1.2
                for i in range(len(intervals) - 1)
            )

            core_median = float(np.median(intervals))

            if has_consecutive_slow or (core_median >= 0.9 and not has_consecutive_fast):
                speed = "slow"
                typical_interval = round(core_median, 2)
            else:
                speed = "normal"
                fast_hits = [i for i in intervals if i < 0.8]
                typical_interval = round(float(np.mean(fast_hits)), 2) if fast_hits else round(core_median, 2)

            pattern_conf = calculate_pattern_confidence(beep_count, intervals, speed)
            formatted_intervals = [round(float(i), 3) for i in intervals]
        else:
            typical_interval = 0.0
            speed = "single_or_none"
            pattern_conf = 0.30
            formatted_intervals = []

        return {
            "beep_count": beep_count,
            "avg_interval_sec": typical_interval,
            "typing_speed": speed,
            "pattern_confidence": pattern_conf,
            "beep_intervals": formatted_intervals
        }
    except Exception as e:
        print(f"[경고] 비프 분석 실패: {e}")
        return {
            "beep_count": 0,
            "avg_interval_sec": 0.0,
            "typing_speed": "unknown",
            "pattern_confidence": 0.0,
            "beep_intervals": []
        }


def analyze_keypad_audio(data, sample_rate: int, predictions: list, file_name: str = "stream") -> dict:
    top_label = predictions[0]["label"]
    raw_acoustic_score = round(float(predictions[0]["score"]), 4)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    top_3_labels = [p["label"] for p in predictions]
    is_keypad_candidate = (top_label == "Beep, bleep") or ("Beep, bleep" in top_3_labels) or ("Alarm clock" in top_3_labels)

    if is_keypad_candidate:
        speed_info = analyze_beep_intervals(data, sample_rate)
        typing_speed = speed_info["typing_speed"]
        beep_count = speed_info["beep_count"]
        avg_interval = speed_info["avg_interval_sec"]
        pattern_conf = speed_info["pattern_confidence"]
        beep_intervals = speed_info.get("beep_intervals", [])

        has_error_alarm = ("Alarm clock" in top_3_labels) or (beep_count >= 10) or (top_label == "Alarm clock")

        # 등록 모드면 이번 입력은 비교하지 않고 샘플로 저장 (오류음이 난 입력은 제외)
        reg = None if has_error_alarm else _record_register_sample(speed_info)
        if reg is not None:
            verification = {"is_owner": True, "status": "registering", "member_id": reg["user_id"],
                            "member_name": reg["name"], "message": "패턴 등록 중"}
        else:
            # ── 비교 엔진: 등록된 모든 사용자 패턴과 대조 ──
            verification = identify_member(beep_intervals)

        if reg is not None:
            auth_status = "registering"
            event_type = "keypad_register"
            done = " - 등록 완료" if reg["done"] else ""
            saved = "" if reg["saved"] else " (간격 인식 실패, 다시 입력해 주세요)"
            msg = f"{reg['name']}님 키패드 패턴 등록 {reg['count']}/{reg['target']}{done}{saved}"
        elif has_error_alarm:
            auth_status = "failed"
            event_type = "keypad_failed"
            msg = f"도어락 비밀번호 오류 경고음 감지 (입력 속도: {typing_speed})"
        elif typing_speed == "slow":
            auth_status = "pending"
            event_type = "keypad_slow_input"
            msg = f"키패드 입력 지연 감지 (유추 의심 | {verification['message']})"
        elif verification["status"] in ("rhythm_mismatch", "length_mismatch"):
            # 비밀번호는 맞췄으나 집주인 리듬과 현저히 다른 경우 -> 경고 격상
            auth_status = "mismatch"
            event_type = "keypad_anomaly"
            msg = f"비밀번호 입력 완료되나 타건 패턴 불일치 (도용 의심 | {verification['message']})"
        else:
            auth_status = "success"
            event_type = "keypad_input"
            if verification.get("member_id"):
                msg = f"정상 인증 완료 ({verification['message']})"
            else:   # 등록된 패턴이 없거나 간격을 못 잡음 → 비교 없이 입력 완료만 알림
                msg = f"도어락 비밀번호 입력 완료 ({verification['message']})"

        final_confidence = round(0.4 * raw_acoustic_score + 0.6 * pattern_conf, 4)
        breakdown = {
            "acoustic_score": raw_acoustic_score,
            "pattern_score": pattern_conf
        }

        detail_data = {
            "auth_status": auth_status,
            "typing_speed": typing_speed,
            "beep_count": beep_count,
            "avg_interval_sec": avg_interval,
            "beep_intervals": beep_intervals,
            "verification": verification,   # 등록 사용자 대조 결과 (유사도, 일치 여부)
            "member_id": verification.get("member_id"),
            "member_name": verification.get("member_name"),
        }

    elif top_label in LOCK_TOP_LABELS:
        event_type = "door_locked"
        msg = "도어락 걸쇠 체결(물리적 잠금음)이 감지되었습니다."
        mechanical_labels = ["Single-lens reflex camera", "Mechanisms", "Clicking", "Scrape", "Door"]
        lock_score = sum(p["score"] for p in predictions if p["label"] in mechanical_labels)
        calibrated_conf = round(min(float(lock_score * 2.2), 0.95), 4)

        final_confidence = calibrated_conf
        breakdown = {"acoustic_score": raw_acoustic_score, "pattern_score": calibrated_conf}
        detail_data = {
            "auth_status": "locked", "typing_speed": "none",
            "beep_count": 0, "avg_interval_sec": 0.0, "beep_intervals": [],
            "verification": {"is_owner": True, "similarity": 1.0, "status": "locked", "message": "도어락 잠금"},
            "member_id": None, "member_name": None,
        }
    else:
        event_type = top_label
        msg = f"도어락 주변 음향 감지: {top_label}"
        final_confidence = raw_acoustic_score
        breakdown = {"acoustic_score": raw_acoustic_score, "pattern_score": 0.0}
        detail_data = {
            "auth_status": "unknown", "typing_speed": "none",
            "beep_count": 0, "avg_interval_sec": 0.0, "beep_intervals": [],
            "verification": {"is_owner": False, "similarity": 0.0, "status": "none", "message": "일반 음향"},
            "member_id": None, "member_name": None,
        }

    return {
        "file_name": file_name,
        "source": "keypad",
        "type": event_type,
        "message": msg,
        "confidence": final_confidence,
        "confidence_breakdown": breakdown,
        "timestamp": timestamp,
        **detail_data,
        "top_3": [
            {"label": p["label"], "score": round(float(p["score"]), 4)}
            for p in predictions
        ]
    }


def detect_and_save_keypad(wav_path: str) -> dict:
    sample_rate, data = wavfile.read(wav_path)
    predictions = classifier(wav_path, top_k=527)[:3]
    return analyze_keypad_audio(data, sample_rate, predictions, os.path.basename(wav_path))


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(current_dir, "audio")
    wav_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))

    if not wav_files:
        print(f"[알림] '{audio_dir}' 폴더에 .wav 파일이 없습니다.")
    else:
        print(f"\n총 {len(wav_files)}개 오디오 키패드 패턴 대조 분석 시작 (등록 사용자 프로필은 DB에서 읽음)\n" + "=" * 75)
        all_results = []

        for fpath in wav_files:
            res = detect_and_save_keypad(fpath)
            all_results.append(res)
            v = res.get("verification", {})
            sim_str = f"{v.get('similarity_pct', 0)}%" if 'similarity_pct' in v else "-"
            pass_str = "PASS" if v.get("is_owner") else "FAIL"

            print(f"▶ {res['file_name']:<26} | 결과: {res['type']:<16} | 인증: {pass_str} ({sim_str}) | {res['message']}")

        output_path = os.path.join(current_dir, "keypad_result.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n[저장 완료] keypad_result.json에 실시간 대조 결과가 반영되었습니다.")