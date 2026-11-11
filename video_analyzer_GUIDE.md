# 로컬 구동 가이드 (RTX 5070 32GB)

Colab 없음. **이 PC에서 Qwen2.5-VL-7B bf16** 으로 돌린다.  
영상 디코딩은 **PyAV** (`video_analyzer/video_io.py`) — 최신 torchvision의 `read_video` 부재 우회.

## 목표 설정 (`runtime_config.py`)

| 항목 | 값 |
|------|-----|
| 정밀 | bf16 (`USE_4BIT=False`) |
| fps | 1.5 |
| max_pixels | 768x768 |
| max_new_tokens | 512 |

## 1) 환경 (venv Python 으로 실행할 것)


```powershell
cd D:\Hackaton 
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\activate

python -m pip install Jinja2 typing_extensions networkx sympy filelock fsspec

# CUDA 휠만 cu128에서 (버전 +cu128 명시)

python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128 
pip install -r requirements-video.txt
```

`python` / `pip` 가 venv 인지 확인.

```powershell
where python
where pip
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2) 클립

`clips/demo1.mp4` … 준비 (5~12초 권장).

## 3) 추론

```powershell
python scripts/run_local.py --video clips/demo1.mp4 --out fixtures/sample_result1.json --clip-id id1_1
python scripts/run_local.py --video clips/demo2.mp4 --out fixtures/sample_result2.json --clip-id id1_2
python scripts/run_local.py --video clips/demo3.mp4 --out fixtures/sample_result3.json --clip-id id1_3
python scripts/run_local.py --video clips/demo4.mp4 --out fixtures/sample_result4.json --clip-id id1_4
```

정상 시 로그에 `decoded frames: N @ target fps=3.0` 이 먼저 보인다.  
모델은 첫 실행만 HF 캐시 다운로드(이미 받았으면 스킵).

## 4) 팀 호출

```python
from video_analyzer import analyze_clip
r = analyze_clip("clips/demo1.mp4")  # local
r = analyze_clip("x.mp4", mode="stub", stub_path="fixtures/sample_result2.json")
```
