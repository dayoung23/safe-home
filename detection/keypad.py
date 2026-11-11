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

# 도어락 물리 잠금음으로 매핑하는 AST 라벨 (기존 규칙 그대로)
LOCK_TOP_LABELS = ["Single-lens reflex camera", "Mechanisms", "Clicking"]


def calculate_pattern_confidence(beep_count: int, intervals: list, speed: str) -> float:
    """
    타격 횟수, 간격의 물리적 타당성, 리듬 명확성을 평가하여
    도어락 키패드 입력 패턴과의 일치도(0.0 ~ 1.0)를 산출합니다.
    """
    if beep_count < 2 or not intervals:
        return 0.30

    score = 0.0

    # 1. 타격 횟수 적합도 (도어락 비밀번호 표준 4~10자리: 40점 만점)
    if 4 <= beep_count <= 10:
        score += 0.40
    elif beep_count in [3, 11, 12]:
        score += 0.25
    else:
        score += 0.10

    # 2. 인간 조작 물리 대역 타당성 (0.18초 ~ 2.2초 범위 비율: 30점 만점)
    valid_range = [i for i in intervals if 0.18 <= i <= 2.2]
    ratio = len(valid_range) / len(intervals)
    score += ratio * 0.30

    # 3. 속도 및 리듬 판정의 명확성 (30점 만점)
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
    """
    오디오 배열에서 양 끝 공백을 제외하고 비프음 속도 및 패턴 신뢰도를 계산합니다.
    (기존에는 wav 경로를 받았지만, 실시간 스트림에서도 쓰도록 배열을 받습니다. 판정 규칙은 동일)
    """
    try:
        data = np.asarray(data)
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # 1. 맨 앞 0.2초 마이크 터치 잡음 제외
        start_cut = int(sample_rate * 0.2)
        d_clean = data[start_cut:] if len(data) > start_cut else data

        # 2. 비프음 피크 감지 (0.21초 빠른 연타 보존을 위해 최소 간격 0.15초)
        threshold = np.max(np.abs(d_clean)) * 0.25
        min_distance = int(sample_rate * 0.15)
        peaks, _ = find_peaks(np.abs(d_clean), height=threshold, distance=min_distance)

        peak_times = (peaks + start_cut) / sample_rate
        beep_count = len(peak_times)

        if beep_count >= 2:
            intervals = list(np.diff(peak_times))

            # 3. 양 끝 공백 제거 (시작 터치 대기 및 마지막 성공/잠금 멜로디)
            if len(intervals) >= 3 and intervals[0] >= 1.2:
                intervals = intervals[1:]
            if len(intervals) >= 3 and intervals[-1] >= 1.2:
                intervals = intervals[:-1]

            # 4. 연속 빠른 입력 및 연속 주저함 탐색
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
        else:
            typical_interval = 0.0
            speed = "single_or_none"
            pattern_conf = 0.30

        return {
            "beep_count": beep_count,
            "avg_interval_sec": typical_interval,
            "typing_speed": speed,
            "pattern_confidence": pattern_conf
        }
    except Exception as e:
        print(f"[경고] 비프 분석 실패: {e}")
        return {
            "beep_count": 0,
            "avg_interval_sec": 0.0,
            "typing_speed": "unknown",
            "pattern_confidence": 0.0
        }


def analyze_keypad_audio(data, sample_rate: int, predictions: list, file_name: str = "stream") -> dict:
    """
    오디오 배열과 AST 상위 3개 예측(predictions)으로 도어락 이벤트를 판정합니다.
    판정 규칙은 기존 detect_and_save_keypad와 동일하며, 파일 대신 배열을 입력으로 받습니다.
    """
    top_label = predictions[0]["label"]
    raw_acoustic_score = round(float(predictions[0]["score"]), 4)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    top_3_labels = [p["label"] for p in predictions]

    # 1. 키패드 비프음 감지 시
    if top_label == "Beep, bleep":
        speed_info = analyze_beep_intervals(data, sample_rate)
        typing_speed = speed_info["typing_speed"]
        beep_count = speed_info["beep_count"]
        avg_interval = speed_info["avg_interval_sec"]
        pattern_conf = speed_info["pattern_confidence"]

        has_error_alarm = "Alarm clock" in top_3_labels or beep_count >= 10

        if has_error_alarm:
            auth_status = "failed"
            event_type = "keypad_failed"
            msg = f"도어락 비밀번호 오류 경고음 감지 (입력 속도: {typing_speed})"
        elif typing_speed == "slow":
            auth_status = "pending"
            event_type = "keypad_slow_input"
            msg = f"키패드 입력 지연 감지 (대표 간격 {avg_interval}초 - 유추 의심)"
        else:
            auth_status = "success"
            event_type = "keypad_input"
            msg = f"정상 속도 키패드 입력 감지 (대표 간격 {avg_interval}초)"

        # 종합 신뢰도 (음향 모델 40% + 행동 패턴 60%)
        final_confidence = round(0.4 * raw_acoustic_score + 0.6 * pattern_conf, 4)
        breakdown = {
            "acoustic_score": raw_acoustic_score,
            "pattern_score": pattern_conf
        }

        detail_data = {
            "auth_status": auth_status,
            "typing_speed": typing_speed,
            "beep_count": beep_count,
            "avg_interval_sec": avg_interval
        }

    # 2. 도어락 물리 잠금음 매핑 시
    elif top_label in LOCK_TOP_LABELS:
        event_type = "door_locked"
        msg = "도어락 걸쇠 체결(물리적 잠금음)이 감지되었습니다."

        # 기계음 계열 스코어 합산 후 도어락 도메인 신뢰도로 캘리브레이션
        mechanical_labels = ["Single-lens reflex camera", "Mechanisms", "Clicking", "Scrape", "Door"]
        lock_score = sum(p["score"] for p in predictions if p["label"] in mechanical_labels)
        calibrated_conf = round(min(float(lock_score * 2.2), 0.95), 4)

        final_confidence = calibrated_conf
        breakdown = {
            "acoustic_score": raw_acoustic_score,
            "pattern_score": calibrated_conf
        }

        detail_data = {
            "auth_status": "locked",
            "typing_speed": "none",
            "beep_count": 0,
            "avg_interval_sec": 0.0
        }

    # 3. 기타 일반 소음
    else:
        event_type = top_label
        msg = f"도어락 주변 음향 감지: {top_label}"
        final_confidence = raw_acoustic_score
        breakdown = {
            "acoustic_score": raw_acoustic_score,
            "pattern_score": 0.0
        }
        detail_data = {
            "auth_status": "unknown",
            "typing_speed": "none",
            "beep_count": 0,
            "avg_interval_sec": 0.0
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
    """wav 파일 한 개를 분석합니다. (기존 배치 방식 - 결과 형식 동일)"""
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
        print(f"\n총 {len(wav_files)}개 도어락 녹음본 신뢰도 통합 분석 시작\n" + "=" * 70)
        all_results = []

        for fpath in wav_files:
            res = detect_and_save_keypad(fpath)
            all_results.append(res)
            print(f"▶ {res['file_name']:<28} | type: {res['type']:<18} | 상태: {res['auth_status']:<7} | 종합신뢰도: {res['confidence']}")

        output_path = os.path.join(current_dir, "keypad_result.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n[저장 완료] keypad_result.json이 갱신되었습니다.")