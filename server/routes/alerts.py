"""
server/routes/alerts.py — 알림 조회 라우트

판단은 하지 않는다. judge.py가 MongoDB alerts 컬렉션에 저장한 최종 결과를 화면에 넘겨주기만 함.
  sound/vision 스레드 → judge.on_event → database.save_alert / update_alert
                                              ↑ 여기서는 읽기만
"""
from flask import Blueprint, current_app, jsonify, request
from server.routes.auth import login_required
from db import database

bp = Blueprint("alerts", __name__, url_prefix="/api")


@bp.get("/alerts")
@login_required
def alerts():
    """최근 알림 목록 (최신순). 대시보드가 1.5초마다 폴링.
    [{"id","level","message","reason","action","sources","created_at","updated_at"}, ...]
    같은 id가 updated_at만 바뀌어 다시 오면 화면은 카드 내용만 교체(2단계 알림)."""
    limit = request.args.get("limit", default=20, type=int)
    limit = max(1, min(limit, 100))
    try:
        return jsonify(database.get_recent_alerts(limit))
    except Exception as e:  # DB 연결 문제로 화면 전체가 죽지 않게
        current_app.logger.exception("get_recent_alerts 실패")
        return jsonify({"ok": False, "error": f"알림 조회 실패: {e}"}), 500
