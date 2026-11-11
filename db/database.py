"""
db/database.py — MongoDB 연결 + 저장/조회 함수 모음

다른 파일은 MongoDB를 직접 만지지 않고 여기 함수만 부른다.
  - 회원가입 (server/routes/auth.py) : add_user, get_users
  - 알림     (risk/judge.py)          : save_alert, update_alert
  - 알림조회 (server/routes/alerts.py): get_recent_alerts
  - 키패드   (detection/keypad.py)    : add_keypad_sample, get_keypad_samples

규칙: MongoDB의 _id(ObjectId)는 JSON으로 못 보내므로 항상 문자열 "id"로 바꿔서 돌려준다.

.env (safe-home/ 바로 아래)
  MONGO_URI=mongodb+srv://아이디:비번@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
  MONGO_DB=safehome
"""
import os
from datetime import datetime
from pathlib import Path
from pymongo.errors import DuplicateKeyError

from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from pymongo import DESCENDING, MongoClient, ReturnDocument

# ─────────────────────────── 연결 ───────────────────────────
# 어디서 실행하든 safe-home/.env 를 읽도록 경로를 고정
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "safehome")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI가 없어요. safe-home/.env 파일에 MONGO_URI=... 를 넣어 주세요.")

# MongoClient는 스레드 여러 개가 같이 써도 안전 → 하나만 만들어 공유
# 3초 안에 연결 못 하면 오류 (설계 문서: 타임아웃 3초)
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
db = client[MONGO_DB]

users = db["users"]
alerts = db["alerts"]
keypad_samples = db["keypad_samples"]


# ─────────────────────────── 도우미 ───────────────────────────
def _now():
    """설계 문서의 시간 형식: "2026-11-11T22:13:05" """
    return datetime.now().isoformat(timespec="seconds")


def _clean(doc):
    """MongoDB 문서 → JSON으로 보낼 수 있는 dict (_id → 문자열 id)."""
    if doc is None:
        return None
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    return doc


def _oid(id_str):
    """문자열 id → ObjectId. 잘못된 형식이면 ValueError."""
    try:
        return ObjectId(id_str)
    except (InvalidId, TypeError):
        raise ValueError(f"잘못된 id 형식이에요: {id_str}")


def _sources(sources):
    """"sound,vision" 이든 ["sound","vision"] 이든 리스트로 통일."""
    if sources is None:
        return []
    if isinstance(sources, str):
        return [s.strip() for s in sources.split(",") if s.strip()]
    return list(sources)


# ─────────────────────────── 초기화 ───────────────────────────
def init_db():
    """연결 확인 + 인덱스 생성. 컬렉션은 처음 저장할 때 자동으로 생긴다.
    여러 번 호출해도 안전 (이미 있는 인덱스는 그대로)."""
    client.admin.command("ping")  # 연결 안 되면 여기서 바로 오류
    alerts.create_index([("created_at", DESCENDING)])
    keypad_samples.create_index("user_id")
    users.create_index("username" ,unique = True)
    print(f"[DB] MongoDB 연결 완료 (db: {MONGO_DB})")


# ─────────────────────────── 사용자 (회원가입) ───────────────────────────
def add_user(username, password_hash, name):
    """회원가입 → 새 id(문자열). 아이디가 이미 있으면 None."""
    try:
        result = users.insert_one({
            "username": username,
            "password_hash": password_hash,
            "name": name,
            "sample_count": 0,
            "created_at": _now(),
        })
    except DuplicateKeyError:
        return None
    return str(result.inserted_id)


def get_user_by_username(username):
    """로그인 확인용 (password_hash 포함). 화면으로 내보내지 말 것."""
    return _clean(users.find_one({"username": username}))


def get_user(user_id):
    """사용자 1명 (password_hash 제외). 없으면 None."""
    return _clean(users.find_one({"_id": _oid(user_id)}, {"password_hash": 0}))


def get_users():
    """모든 사용자 (password_hash 제외). keypad.py가 패턴 비교할 때만 씀 — API로 내보내지 않음."""
    return [_clean(u) for u in users.find({}, {"password_hash": 0}).sort("created_at", 1)]




# ─────────────────────────── 알림 ───────────────────────────
def save_alert(level, message, reason="", action="", sources=None):
    """새 알림 저장 → id(문자열) 반환. (judge.py가 1차 알림 때 호출)"""
    now = _now()
    result = alerts.insert_one({
        "level": level,               # 안전 / 주의 / 위험
        "message": message,
        "reason": reason,
        "action": action,
        "sources": _sources(sources),  # ["sound", "vision"]
        "created_at": now,
        "updated_at": now,             # 2단계 알림 업데이트 때 바뀜
    })
    return str(result.inserted_id)


def update_alert(alert_id, **fields):
    """기존 알림 수정 (2단계 알림). 바꿀 필드만 넘기면 됨.
    예) update_alert(aid, level="위험", message="...", sources="sound,vision")
    수정된 알림 dict 반환, 없으면 None."""
    allowed = {"level", "message", "reason", "action", "sources"}
    update = {k: v for k, v in fields.items() if k in allowed}
    if "sources" in update:
        update["sources"] = _sources(update["sources"])
    update["updated_at"] = _now()
    doc = alerts.find_one_and_update(
        {"_id": _oid(alert_id)},
        {"$set": update},
        return_document=ReturnDocument.AFTER,
    )
    return _clean(doc)


def get_recent_alerts(limit=20):
    """최근 알림 (최신순). 대시보드가 1.5초마다 호출."""
    cursor = alerts.find().sort("created_at", DESCENDING).limit(limit)
    return [_clean(a) for a in cursor]


# ─────────────────────────── 키패드 샘플 ───────────────────────────
def add_keypad_sample(user_id, result):
    """등록용 키패드 샘플 저장 → 샘플 id. users.sample_count +1.
    result: keypad.py 출력 dict (avg_interval_sec, beep_count 필수, intervals는 있으면 저장)"""
    doc = {
        "user_id": user_id,
        "avg_interval_sec": float(result["avg_interval_sec"]),
        "beep_count": int(result["beep_count"]),
        "typing_speed": result.get("typing_speed"),
        "created_at": _now(),
    }
    if result.get("intervals"):
        doc["intervals"] = [float(x) for x in result["intervals"]]
    inserted = keypad_samples.insert_one(doc)
    users.update_one({"_id": _oid(user_id)}, {"$inc": {"sample_count": 1}})
    return str(inserted.inserted_id)


def get_keypad_samples(user_id):
    """사용자의 등록 샘플 (오래된 순)."""
    return [_clean(s) for s in keypad_samples.find({"user_id": user_id}).sort("created_at", 1)]