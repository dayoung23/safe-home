"""
server/routes/auth.py — 회원가입 / 로그인 / 로그아웃 / 내 정보

로그인 상태는 Flask session(쿠키)에 user_id만 넣어 유지.
다른 사람 정보를 주는 API는 없음. 조회는 항상 session의 user_id로만 한다.
"""
from functools import wraps
from detection import keypad
from flask import Blueprint, current_app, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash

from db import database

bp = Blueprint("auth", __name__, url_prefix="/api")


def login_required(f):
    """로그인 안 했으면 401. 다른 routes 파일에서도 import해서 씀."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"ok": False, "error": "로그인이 필요해요"}), 401
        return f(*args, **kwargs)
    return wrapper


@bp.post("/register")
def register():
    """회원가입. body: {"username","password","name"}"""
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    name = str(data.get("name", "")).strip()

    if not username or not password or not name:
        return jsonify({"ok": False, "error": "아이디, 비밀번호, 이름을 모두 입력해 주세요"}), 400
    if len(username) > 20 or len(name) > 20:
        return jsonify({"ok": False, "error": "아이디와 이름은 20자 이내로 입력해 주세요"}), 400
    if len(password) < 4:
        return jsonify({"ok": False, "error": "비밀번호는 4자 이상이어야 해요"}), 400

    try:
        user_id = database.add_user(username, generate_password_hash(password), name)
    except Exception as e:
        current_app.logger.exception("add_user 실패")
        return jsonify({"ok": False, "error": f"사용자 저장 실패: {e}"}), 500
    if user_id is None:
        return jsonify({"ok": False, "error": "이미 있는 아이디예요"}), 409

    return jsonify({"ok": True, "user_id": user_id})


@bp.post("/login")
def login():
    """로그인. body: {"username","password"}"""
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))

    try:
        user = database.get_user_by_username(username)
    except Exception as e:
        current_app.logger.exception("로그인 조회 실패")
        return jsonify({"ok": False, "error": f"로그인 실패: {e}"}), 500

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"ok": False, "error": "아이디 또는 비밀번호가 틀렸어요"}), 401

    session.clear()
    session["user_id"] = user["id"]
    return jsonify({"ok": True, "name": user["name"]})


@bp.post("/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


@bp.get("/me")
@login_required
def me():
    """내 정보만. {"id","username","name","sample_count","created_at"}"""
    try:
        user = database.get_user(session["user_id"])
    except Exception as e:
        current_app.logger.exception("get_user 실패")
        return jsonify({"ok": False, "error": f"사용자 조회 실패: {e}"}), 500
    if user is None:  # DB에서 지워진 계정
        session.clear()
        return jsonify({"ok": False, "error": "사용자를 찾을 수 없어요"}), 401
    return jsonify(user)

@bp.post("/keypad/register")
@login_required
def keypad_register():
    """로그인한 본인의 키패드 패턴 등록 시작. 측정·저장·학습은 detection/keypad.py가 담당."""
    user_id = session["user_id"]
    try:
        user = database.get_user(user_id)
        keypad.start_register(user_id, user["name"])
        return jsonify({"ok": True, "status": keypad.get_register_status()})
    except Exception as e:
        current_app.logger.exception("start_register 실패")
        return jsonify({"ok": False, "error": f"등록 시작 실패: {e}"}), 500

# 키패드 패턴 등록

@bp.get("/keypad/register/status")
@login_required
def keypad_register_status():
    """내 등록 진행률만. 다른 사람이 등록 중이면 active=False만 돌려줌."""
    try:
        st = keypad.get_register_status()
    except Exception as e:
        current_app.logger.exception("get_register_status 실패")
        return jsonify({"ok": False, "error": f"상태 조회 실패: {e}"}), 500
    if not st or st.get("user_id") != session["user_id"]:
        return jsonify({"ok": True, "active": False})
    return jsonify({**st, "active": True})