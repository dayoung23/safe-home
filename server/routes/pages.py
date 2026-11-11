"""
server/routes/pages.py — 화면(HTML) 라우트. 로그인 안 했으면 /login으로 보냄.
"""
from flask import Blueprint, redirect, render_template, session, url_for

bp = Blueprint("pages", __name__)


def _logged_in():
    return "user_id" in session


@bp.get("/")
def dashboard():
    if not _logged_in():
        return redirect(url_for("pages.login_page"))
    return render_template("dashboard.html")


@bp.get("/login")
def login_page():
    return render_template("login.html")


@bp.get("/register")
def register_page():
    return render_template("register.html")


@bp.get("/settings")
def settings_page():
    if not _logged_in():
        return redirect(url_for("pages.login_page"))
    return render_template("settings.html")


@bp.get("/keypad")
def keypad_page():
    if not _logged_in():
        return redirect(url_for("pages.login_page"))
    return render_template("keypad.html")