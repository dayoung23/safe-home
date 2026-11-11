import os
import glob
import json
import datetime
import warnings
from transformers import pipeline

# numpy 쓰기 불가 경고는 동작에 영향이 없어 숨깁니다.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

# 1. AST 오디오 분류 파이프라인 초기화 (전체 527개 라벨 점수를 받아 감시 라벨 판정에 사용)
print("AST 오디오 분류 모델 초기화 중...")
classifier = pipeline(
    task="audio-classification",
    model="MIT/ast-finetuned-audioset-10-10-0.4593",
    top_k=527,
)
print("AST 모델 준비 완료.")

# 2. 위험 소리 감시 라벨과 임계값
#    ※ 임시값입니다. 새로 녹음한 데이터로 결과를 보면서 조정하세요.
WATCH_THRESHOLDS = {
    "Screaming": 0.05,
    "Shout": 0.10,
    "Yell": 0.10,
    "Slam": 0.05,
    "Door": 0.15,
    "Bang": 0.10,
    "Thump, thud": 0.10,
    "Smash, crash": 0.10,
    "Glass": 0.10,
    "Shatter": 0.10,
    "Breaking": 0.10,
    "Crying, sobbing": 0.10,
}


def detect_and_save_sound(audio_path: str) -> dict:
    """오디오 파일을 분류하고 표준 JSON 구조로 반환합니다."""
    file_name = os.path.basename(audio_path)
    all_preds = classifier(audio_path, top_k=527)   # 점수 내림차순
    scores = {p["label"]: float(p["score"]) for p in all_preds}
    top3 = all_preds[:3]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # 감시 라벨 점수 (높은 순)
    watch = {k: round(scores[k], 4) for k in WATCH_THRESHOLDS if k in scores}
    watch = dict(sorted(watch.items(), key=lambda x: -x[1]))

    # 임계값을 넘은 감시 라벨
    alerts = [k for k, v in watch.items() if v >= WATCH_THRESHOLDS[k]]

    if alerts:
        event_type = alerts[0]                 # watch가 점수순이라 가장 높은 라벨
        confidence = watch[event_type]
        message = f"위험 소리 감지: {event_type}"
    else:
        event_type = top3[0]["label"]
        confidence = round(float(top3[0]["score"]), 4)
        message = f"{event_type} 소리가 감지되었습니다."

    return {
        "file_name": file_name,
        "source": "audio",
        "type": event_type,
        "message": message,
        "confidence": confidence,
        "alert": bool(alerts),
        "alerts": alerts,
        "timestamp": timestamp,
        "top_3": [
            {"label": p["label"], "score": round(float(p["score"]), 4)}
            for p in top3
        ],
        "watch": watch,
    }


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(current_dir, "audio")

    # detection/audio 폴더 내 모든 wav 파일 탐색
    all_wavs = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))

    # 키패드 관련 음원 제외하고 일반 환경음만 필터링
    sound_targets = [
        f for f in all_wavs
        if not any(k in os.path.basename(f) for k in ["비번", "잠금"])
    ]

    if not sound_targets:
        print(f"[알림] '{audio_dir}' 폴더에 분석할 일반 환경음(.wav) 파일이 없습니다.")
    else:
        print(f"\n총 {len(sound_targets)}개 환경음 파일 분석 시작\n" + "=" * 60)
        all_results = []

        for fpath in sound_targets:
            res = detect_and_save_sound(fpath)
            all_results.append(res)
            flag = "  ⚠ ALERT" if res["alert"] else ""
            print(f"▶ {res['file_name']:<22} | 감지: {res['type']:<20} ({res['confidence']}){flag}")
            top_watch = list(res["watch"].items())[:4]
            print(f"    watch: {top_watch}")

        output_path = os.path.join(current_dir, "audio_result.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n[저장 완료] audio_result.json이 갱신되었습니다: {output_path}")