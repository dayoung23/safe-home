import os
import glob
import json
import datetime
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

# AST 모델은 sound.py에서 한 번만 로드해 공유합니다. (realtime.py와 같이 써도 모델이 중복 로딩되지 않음)
try:
    from detection.sound import classifier      # 루트에서 python main.py로 실행할 때
except ImportError:
    from sound import classifier                # detection 폴더에서 단독 실행할 때

LOCK_TOP_LABELS = ["Single-lens reflex camera", "Mechanisms", "Clicking"]
PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_profile.json")


# ───────────────────────── 1. 집주인 타건 프로필 관리 & 비교 엔진 ─────────────────────────

def load_or_create_owner_profile() -> dict:
    """
    집주인의 등록 타건 프로필(Baseline)을 불러옵니다.
    파일이 없으면 4자리 비밀번호(간격 3개, 평균 0.35초)를 기본 기준선으로 자동 생성합니다.
    """
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # 기본 집주인 프로필 (4자리 비밀번호 기준: 버튼 사이 간격 3개)
    default_profile = {
        "user_name": "집주인_A",
        "pin_length": 4,
        "mean_intervals": [0.35, 0.40, 0.32],  # 집주인 고유 타건 리듬
        "std_intervals": [0.06, 0.08, 0.05],   # 허용 오차 범위
        "enrolled_count": 5,
        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(default_profile, f, indent=2, ensure_ascii=False)
    return default_profile


def verify_keystroke_pattern(current_intervals: list, profile: dict) -> dict:
    """
    현재 입력된 비프음 간격 배열과 등록된 집주인 프로필을 비교하여
    유사도(0~100%)와 집주인 인증 여부를 판정합니다.
    """
    if not current_intervals or len(current_intervals) < 2:
        return {
            "is_owner": False,
            "similarity": 0.0,
            "status": "insufficient_data",
            "message": "타건 데이터 부족 (판정 불가)"
        }

    ref_means = np.array(profile.get("mean_intervals", [0.35, 0.40, 0.32]))
    ref_stds = np.array(profile.get("std_intervals", [0.06, 0.08, 0.05]))
    curr = np.array(current_intervals)

    # 1. 자릿수(간격 개수) 불일치: 비밀번호 자릿수부터 다른 경우
    if len(curr) != len(ref_means):
        return {
            "is_owner": False,
            "similarity": 0.15,
            "status": "length_mismatch",
            "message": f"자릿수 불일치 (등록 {len(ref_means)+1}자리 vs 입력 {len(curr)+1}자리)"
        }

    # 2. 각 구간별 Z-score 오차 계산
    # 허용 표준편차(최소 0.04초)로 나누어 각 버튼 사이의 시간 차이를 정규화
    stds_clipped = np.maximum(ref_stds, 0.04)
    z_scores = np.abs((curr - ref_means) / stds_clipped)
    avg_z = float(np.mean(z_scores))

    # 3. 유사도 환산 (Z-score가 0이면 100%, 오차가 커질수록 지수 감쇠)
    similarity = float(np.exp(-0.4 * avg_z))
    similarity_pct = round(similarity * 100, 1)

    # 70% 이상 일치 시 집주인으로 최종 승인
    is_owner = similarity >= 0.70

    if is_owner:
        status = "owner_verified"
        msg = f"집주인 타건 리듬 일치 (유사도 {similarity_pct}%)"
    else:
        status = "rhythm_mismatch"
        msg = f"타건 리듬 불일치 - 비번 유출/침입 의심 (유사도 {similarity_pct}%)"

    return {
        "is_owner": is_owner,
        "similarity": round(similarity, 4),
        "similarity_pct": similarity_pct,
        "avg_z_score": round(avg_z, 2),
        "status": status,
        "message": msg
    }


# ───────────────────────── 2. 오디오 분석 및 특징 추출 ─────────────────────────

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

    owner_profile = load_or_create_owner_profile()

    if is_keypad_candidate:
        speed_info = analyze_beep_intervals(data, sample_rate)
        typing_speed = speed_info["typing_speed"]
        beep_count = speed_info["beep_count"]
        avg_interval = speed_info["avg_interval_sec"]
        pattern_conf = speed_info["pattern_confidence"]
        beep_intervals = speed_info.get("beep_intervals", [])

        # ── 비교 엔진 가동: 집주인 타건 패턴과 실시간 대조 ──
        verification = verify_keystroke_pattern(beep_intervals, owner_profile)

        has_error_alarm = ("Alarm clock" in top_3_labels) or (beep_count >= 10) or (top_label == "Alarm clock")

        if has_error_alarm:
            auth_status = "failed"
            event_type = "keypad_failed"
            msg = f"도어락 비밀번호 오류 경고음 감지 (입력 속도: {typing_speed})"
        elif typing_speed == "slow":
            auth_status = "pending"
            event_type = "keypad_slow_input"
            msg = f"키패드 입력 지연 감지 (유추 의심 | {verification['message']})"
        elif not verification["is_owner"] and verification["status"] == "rhythm_mismatch":
            # 비밀번호는 맞췄으나 집주인 리듬과 현저히 다른 경우 -> 경고 격상
            auth_status = "mismatch"
            event_type = "keypad_anomaly"
            msg = f"비밀번호 입력 완료되나 타건 패턴 불일치 (도용 의심 | {verification['message']})"
        else:
            auth_status = "success"
            event_type = "keypad_input"
            msg = f"정상 인증 완료 ({verification['message']})"

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
            "verification": verification  # 집주인 대조 결과 (유사도, 일치 여부)
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
            "verification": {"is_owner": True, "similarity": 1.0, "status": "locked", "message": "도어락 잠금"}
        }
    else:
        event_type = top_label
        msg = f"도어락 주변 음향 감지: {top_label}"
        final_confidence = raw_acoustic_score
        breakdown = {"acoustic_score": raw_acoustic_score, "pattern_score": 0.0}
        detail_data = {
            "auth_status": "unknown", "typing_speed": "none",
            "beep_count": 0, "avg_interval_sec": 0.0, "beep_intervals": [],
            "verification": {"is_owner": False, "similarity": 0.0, "status": "none", "message": "일반 음향"}
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
        print(f"\n총 {len(wav_files)}개 오디오 집주인 타건 패턴 대조 분석 시작\n" + "=" * 75)
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