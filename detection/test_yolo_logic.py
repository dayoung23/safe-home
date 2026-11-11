"""Offline checks for YOLO trigger + conservative obs mapper (no ultralytics needed)."""
from __future__ import annotations

from detection.yolo_detector import Det, FrameDetections
from detection.yolo_obs import map_observation
from detection.yolo_trigger import TriggerConfig, YoloTrigger


def _dets(persons: list[tuple], w=640, h=480) -> FrameDetections:
    ps = []
    for i, (xyxy, conf) in enumerate(persons):
        ps.append(Det(0, "person", conf, xyxy, track_id=i + 1))
    return FrameDetections(persons=ps, frame_w=w, frame_h=h, infer_ms=1.0)


def test_appear_and_cooldown():
    tr = YoloTrigger(TriggerConfig(min_interval_sec=8.0, warmup_frames=2, persist_frames=2))
    empty = _dets([])
    # warmup
    assert tr.update(empty, 0.0).fired is False
    assert tr.update(empty, 0.1).fired is False

    standing = _dets([((200, 80, 280, 400), 0.9)])  # tall box
    r1 = tr.update(standing, 1.0)  # persist 1
    assert r1.fired is False
    r2 = tr.update(standing, 1.1)  # persist 2 → appear
    assert r2.fired is True
    assert r2.reason == "person_appear"
    obs = map_observation(standing, r2)
    assert obs["type"] == "person"
    assert obs["persons"][0]["gender"] == "불명"

    # same count → no new emit reason (cooldown only applies when a reason exists)
    r3 = tr.update(standing, 2.0)
    assert r3.fired is False
    assert r3.reason == "stable"

    # count up during cooldown window → blocked
    two = _dets([
        ((200, 80, 280, 400), 0.9),
        ((320, 80, 400, 400), 0.88),
    ])
    r4 = tr.update(two, 2.5)  # streak already hot from prior person frames
    assert r4.fired is False
    assert r4.reason == "cooldown"
    assert r4.detail.get("blocked_reason") == "person_count_up"


def test_fall_candidate():
    tr = YoloTrigger(TriggerConfig(min_interval_sec=0.0, warmup_frames=0, persist_frames=2))
    # wide box in lower half
    fallen = _dets([((100, 300, 500, 450), 0.8)])
    assert tr.update(fallen, 1.0).fired is False
    r = tr.update(fallen, 1.1)
    assert r.fired is True
    assert r.reason == "fall_candidate"
    obs = map_observation(fallen, r)
    assert obs["type"] == "fall"


def test_force_none():
    tr = YoloTrigger(TriggerConfig(warmup_frames=0))
    empty = _dets([])
    r = tr.force_observe(empty, 5.0, reason="sound")
    assert r.fired is True
    obs = map_observation(empty, r)
    assert obs["type"] == "none"


def test_roi_suspicious():
    cfg = TriggerConfig(
        min_interval_sec=0.0,
        warmup_frames=0,
        persist_frames=2,
        roi=(0.0, 0.0, 0.3, 1.0),
    )
    tr = YoloTrigger(cfg)
    # person center in left 30%
    p = _dets([((20, 100, 100, 400), 0.85)])
    tr.update(p, 1.0)
    r = tr.update(p, 1.1)
    assert r.fired is True
    # appear may win over roi on first stable frame; either person or suspicious is ok,
    # but ROI enter should be suspicious when reason is roi_enter
    if r.reason == "roi_enter":
        assert map_observation(p, r)["type"] == "suspicious"
    else:
        assert r.reason in ("person_appear", "roi_enter")


if __name__ == "__main__":
    test_appear_and_cooldown()
    test_fall_candidate()
    test_force_none()
    test_roi_suspicious()
    print("ok: yolo trigger/obs logic")
