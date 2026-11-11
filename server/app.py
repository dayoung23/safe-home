"""
server/app.py — Flask 앱 생성 + 기능별 Blueprint 등록만 담당

실행: 프로젝트 루트(safe-home/)에서
    python main.py            ← 판단 스레드 + 서버 함께 실행
    python -m server.app      ← 서버만 단독 테스트
"""
import os

from flask import Flask, jsonify, request

from db import database
from server.routes.alerts import bp as alerts_bp
from server.routes.auth import bp as auth_bp
from server.routes.demo import bp as demo_bp
from server.routes.pages import bp as pages_bp

app = Flask(__name__)
app.json.ensure_ascii = False  # 한글 메시지가 \uXXXX로 깨지지 않게

app.secret_key = os.getenv("SECRET_KEY", "dev-only-change-me")  # 로그인 세션 쿠키 서명용
# MongoDB 연결 확인 + 인덱스 준비 (컬렉션은 처음 저장할 때 자동 생성됨)
database.init_db()

app.register_blueprint(pages_bp)   # /, /register
app.register_blueprint(auth_bp)    # /api/register, /api/register/status, /api/users
app.register_blueprint(alerts_bp)  # /api/alerts
app.register_blueprint(demo_bp)    # /api/status, /api/demo/* (시연·디버그용)


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "없는 API입니다"}), 404
    return "페이지를 찾을 수 없어요", 404


if __name__ == "__main__":
    # 단독 테스트용. main.py에서는 app.run(port=5000, use_reloader=False)로 실행할 것
    app.run(port=5000, debug=True, use_reloader=False)
