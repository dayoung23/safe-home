"""Decode mp4 with PyAV — avoids broken torchvision.io.read_video on new wheels."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_video_frames(
    video_path: str | Path,
    *,
    fps: float = 3.0,
    max_pixels: int = 1024 * 1024,
    max_frames: int = 64,
) -> list[Image.Image]:
    """
    Sample frames at ~fps and optionally shrink so width*height <= max_pixels.
    """
    import av

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)

    container = av.open(str(path))
    stream = container.streams.video[0]
    stream_fps = float(stream.average_rate) if stream.average_rate else 30.0
    step = max(stream_fps / max(fps, 0.1), 1.0)

    frames: list[Image.Image] = []
    next_i = 0.0
    for i, frame in enumerate(container.decode(video=0)):
        if i + 1e-6 < next_i:
            continue
        img = frame.to_image().convert("RGB")
        frames.append(_fit_max_pixels(img, max_pixels))
        next_i += step
        if len(frames) >= max_frames:
            break

    container.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def _fit_max_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    w, h = img.size
    area = w * h
    if area <= max_pixels or max_pixels <= 0:
        return img
    scale = (max_pixels / area) ** 0.5
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh), Image.Resampling.BICUBIC)
