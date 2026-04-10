"""
Flask Web Dashboard для OKX Grid Bot
Запускается на PythonAnywhere как WSGI приложение
"""

import json
import os
import threading
import time
from datetime import datetime

from flask import Flask, jsonify, render_template_string, request, session, redirect, url_for

# Импортируем бота
from okx_grid_bot import (
    bot,
    CONFIG,
    log,
    ALLOWED_CHAT_IDS,
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "okx-bot-secret-key-change-me")

# ─── НАСТРОЙКИ АВТОРИЗАЦИИ ───
ADMIN_USERNAME = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASS", "admin123")
GUEST_PASSWORD = os.environ.get("GUEST_PASS", "guest123")

# Храним состояние бота
bot_state = {
    "started": False,
    "start_time": None,
    "last_update": None,
    "last_price": 0,
    "balance": 0,
    "realized_pnl": 0,
    "unrealized_pnl": 0,
    "total_pnl": 0,
    "signal": "NEUTRAL",
    "orders_count": 0,
    "positions_count": 0,
    "errors": [],
    "positions": [],
}


def start_bot_in_background():
    """Запускает бота в фоновом потоке"""
    if bot_state["started"]:
        return

    bot_state["started"] = True
    bot_state["start_time"] = datetime.now().isoformat()

    def run_bot():
        try:
            log.info("🌐 Запуск бота из веб-приложения...")
            bot.start()
        except Exception as e:
            log.error(f"❌ Ошибка запуска бота: {e}")
            bot_state["errors"].append(str(e))

    thread = threading.Thread(target=run_bot, daemon=True)
    thread.start()
    log.info("✅ Бот запущен в фоновом потоке")


def update_bot_state():
    """Обновляет состояние бота для веб-интерфейса"""
    try:
        bot_state["last_update"] = datetime.now().isoformat()
        bot_state["last_price"] = bot.last_price if hasattr(bot, 'last_price') else 0

        # Баланс
        try:
            bot_state["balance"] = bot.get_balance()
        except Exception:
            pass

        # PnL — читаем напрямую из бота
        try:
            bot_state["realized_pnl"] = bot.realized_pnl if hasattr(bot, 'realized_pnl') else 0
        except Exception:
            bot_state["realized_pnl"] = 0

        try:
            bot_state["unrealized_pnl"] = bot.get_unrealized_pnl() if hasattr(bot, 'get_unrealized_pnl') else 0
        except Exception:
            bot_state["unrealized_pnl"] = 0

        bot_state["total_pnl"] = bot_state["realized_pnl"] + bot_state["unrealized_pnl"]

        # Сигнал
        try:
            bot_state["signal"] = bot.last_signal.get("signal", "NEUTRAL") if hasattr(bot, 'last_signal') else "NEUTRAL"
        except Exception:
            bot_state["signal"] = "NEUTRAL"

        # Ордера
        try:
            bot_state["orders_count"] = len(bot.active_orders) if hasattr(bot, 'active_orders') else 0
        except Exception:
            bot_state["orders_count"] = 0

        # Позиции
        try:
            r = bot.account_api.get_positions(instType="SWAP", instId=CONFIG["symbol"])
            if r.get("code") == "0":
                positions_data = r.get("data", [])
                bot_state["positions_count"] = len(positions_data)
                bot_state["positions"] = []
                for pos in positions_data:
                    pos_side = pos.get("posSide", "net")
                    sz = float(pos.get("pos", 0))
                    avg_px = float(pos.get("avgPx", 0))
                    upl = float(pos.get("upl", 0))
                    if sz > 0:
                        direction = "🟢 LONG" if pos_side == "long" else "🔴 SHORT" if pos_side == "short" else "⚪ NET"
                        bot_state["positions"].append({
                            "side": direction,
                            "size": sz,
                            "avg_price": avg_px,
                            "upl": upl,
                        })
        except Exception:
            bot_state["positions_count"] = 0
            bot_state["positions"] = []

    except Exception as e:
        log.warning(f"Ошибка обновления состояния: {e}")


# ─── HTML ШАБЛОНЫ ───

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>OKX Bot — Вход</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0a0a0a;
            color: #e0e0e0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .login-box {
            background: #1a1a1a;
            border-radius: 16px;
            padding: 40px;
            max-width: 400px;
            width: 100%;
            border: 1px solid #333;
        }
        h1 {
            color: #00d4aa;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #888;
        }
        select, input {
            width: 100%;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #333;
            background: #0a0a0a;
            color: #fff;
            font-size: 16px;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 14px;
            border-radius: 8px;
            border: none;
            background: #00d4aa;
            color: #000;
            font-weight: bold;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover { opacity: 0.9; }
        .error {
            color: #ff4757;
            text-align: center;
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="login-box">
        <h1>🤖 OKX Grid Bot</h1>
        <p class="subtitle">Вход в панель управления</p>
        {% if error %}
        <p class="error">{{ error }}</p>
        {% endif %}
        <form method="POST">
            <div class="form-group">
                <label>Режим доступа</label>
                <select name="role" id="role">
                    <option value="admin">👑 Администратор</option>
                    <option value="guest">👁 Гость (только просмотр)</option>
                </select>
            </div>
            <div class="form-group">
                <label>Пароль</label>
                <input type="password" name="password" placeholder="Введите пароль" required>
            </div>
            <button type="submit">🔓 Войти</button>
        </form>
    </div>
</body>
</html>
"""

ADMIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>OKX Bot — Admin</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0a0a0a;
            color: #e0e0e0;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { color: #00d4aa; text-align: center; margin-bottom: 10px; }
        .role-badge {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-bottom: 20px;
        }
        .role-badge a { color: #2196f3; text-decoration: none; }
        .status-card {
            background: #1a1a1a;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #333;
        }
        .status-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #333;
        }
        .status-row:last-child { border-bottom: none; }
        .label { color: #888; }
        .value { color: #fff; font-weight: bold; }
        .positive { color: #00d4aa; }
        .negative { color: #ff4757; }
        .btn {
            display: inline-block;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 8px;
            text-decoration: none;
            color: #fff;
            font-weight: bold;
            transition: opacity 0.2s;
        }
        .btn:hover { opacity: 0.8; }
        .btn-start { background: #00d4aa; }
        .btn-stop { background: #ff4757; }
        .btn-refresh { background: #2196f3; }
        .btn-logout { background: #666; }
        .actions { text-align: center; margin: 20px 0; }
        .signal-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        .signal-BUY, .signal-STRONG_BUY { background: rgba(0,212,170,0.2); color: #00d4aa; }
        .signal-SELL, .signal-STRONG_SELL { background: rgba(255,71,87,0.2); color: #ff4757; }
        .signal-NEUTRAL { background: rgba(255,165,2,0.2); color: #ffa502; }
        .position-card {
            background: #222;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .last-update { text-align: center; color: #666; font-size: 12px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 OKX Grid Bot V3</h1>
        <p class="role-badge">👑 Администратор | <a href="/logout">Выйти</a></p>

        {% if not state.started %}
        <div class="status-card">
            <p style="text-align:center;color:#888;">Бот не запущен</p>
            <div class="actions">
                <a href="/start" class="btn btn-start">▶️ Запустить бота</a>
            </div>
        </div>
        {% else %}
        <div class="status-card">
            <div class="status-row"><span class="label">📊 Состояние</span><span class="value positive">✅ Работает</span></div>
            <div class="status-row"><span class="label">⏱ Запущен</span><span class="value">{{ state.start_time or '—' }}</span></div>
            <div class="status-row"><span class="label">💰 Баланс</span><span class="value">{{ "%.2f"|format(state.balance) }} USDT</span></div>
            <div class="status-row">
                <span class="label">📈 Realized PnL</span>
                <span class="value {{ 'positive' if state.realized_pnl >= 0 else 'negative' }}">{{ "%+.2f"|format(state.realized_pnl) }} USDT</span>
            </div>
            <div class="status-row">
                <span class="label">📊 Unrealized PnL</span>
                <span class="value {{ 'positive' if state.unrealized_pnl >= 0 else 'negative' }}">{{ "%+.2f"|format(state.unrealized_pnl) }} USDT</span>
            </div>
            <div class="status-row">
                <span class="label">💎 Total PnL</span>
                <span class="value {{ 'positive' if state.total_pnl >= 0 else 'negative' }}" style="font-size:18px;">{{ "%+.2f"|format(state.total_pnl) }} USDT</span>
            </div>
            <div class="status-row"><span class="label">💱 Цена SOL</span><span class="value">${{ "%.2f"|format(state.last_price) }}</span></div>
            <div class="status-row">
                <span class="label">🧠 AI Сигнал</span>
                <span class="value"><span class="signal-badge signal-{{ state.signal }}">{{ state.signal }}</span></span>
            </div>
            <div class="status-row"><span class="label">📋 Ордеров</span><span class="value">{{ state.orders_count }}</span></div>
            <div class="status-row"><span class="label">💼 Позиций</span><span class="value">{{ state.positions_count }}</span></div>
        </div>

        {% if state.positions %}
        <div class="status-card">
            <h3 style="margin:0 0 15px 0;">💼 Открытые позиции</h3>
            {% for pos in state.positions %}
            <div class="position-card">
                <div class="status-row"><span class="label">{{ pos.side }}</span><span class="value">{{ pos.size }} SOL</span></div>
                <div class="status-row"><span class="label">Вход</span><span class="value">${{ "%.2f"|format(pos.avg_price) }}</span></div>
                <div class="status-row"><span class="label">PnL</span><span class="value {{ 'positive' if pos.upl >= 0 else 'negative' }}">{{ "%+.2f"|format(pos.upl) }} USDT</span></div>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <div class="actions">
            <a href="/refresh" class="btn btn-refresh">🔄 Обновить</a>
            <a href="/stop" class="btn btn-stop">⏹ Остановить</a>
            <a href="/logout" class="btn btn-logout">🚪 Выйти</a>
        </div>
        {% endif %}

        <p class="last-update">Последнее обновление: {{ state.last_update or '—' }}</p>
    </div>
    <script>setTimeout(() => window.location.reload(), 30000);</script>
</body>
</html>
"""

GUEST_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>OKX Bot — Guest</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0a0a0a;
            color: #e0e0e0;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { color: #00d4aa; text-align: center; margin-bottom: 10px; }
        .role-badge { text-align: center; color: #666; font-size: 14px; margin-bottom: 20px; }
        .role-badge a { color: #2196f3; text-decoration: none; }
        .status-card {
            background: #1a1a1a;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #333;
        }
        .status-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #333;
        }
        .status-row:last-child { border-bottom: none; }
        .label { color: #888; }
        .value { color: #fff; font-weight: bold; }
        .positive { color: #00d4aa; }
        .negative { color: #ff4757; }
        .signal-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        .signal-BUY, .signal-STRONG_BUY { background: rgba(0,212,170,0.2); color: #00d4aa; }
        .signal-SELL, .signal-STRONG_SELL { background: rgba(255,71,87,0.2); color: #ff4757; }
        .signal-NEUTRAL { background: rgba(255,165,2,0.2); color: #ffa502; }
        .position-card { background: #222; border-radius: 8px; padding: 15px; margin: 10px 0; }
        .last-update { text-align: center; color: #666; font-size: 12px; margin-top: 20px; }
        .actions { text-align: center; margin: 20px 0; }
        .btn {
            display: inline-block;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 8px;
            text-decoration: none;
            color: #fff;
            font-weight: bold;
        }
        .btn-refresh { background: #2196f3; }
        .btn-logout { background: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 OKX Grid Bot V3</h1>
        <p class="role-badge">👁 Гость (только просмотр) | <a href="/logout">Выйти</a></p>

        {% if not state.started %}
        <div class="status-card">
            <p style="text-align:center;color:#888;">Бот не запущен</p>
        </div>
        {% else %}
        <div class="status-card">
            <div class="status-row"><span class="label">📊 Состояние</span><span class="value positive">✅ Работает</span></div>
            <div class="status-row"><span class="label">⏱ Запущен</span><span class="value">{{ state.start_time or '—' }}</span></div>
            <div class="status-row"><span class="label">💰 Баланс</span><span class="value">{{ "%.2f"|format(state.balance) }} USDT</span></div>
            <div class="status-row">
                <span class="label">📈 Realized PnL</span>
                <span class="value {{ 'positive' if state.realized_pnl >= 0 else 'negative' }}">{{ "%+.2f"|format(state.realized_pnl) }} USDT</span>
            </div>
            <div class="status-row">
                <span class="label">📊 Unrealized PnL</span>
                <span class="value {{ 'positive' if state.unrealized_pnl >= 0 else 'negative' }}">{{ "%+.2f"|format(state.unrealized_pnl) }} USDT</span>
            </div>
            <div class="status-row">
                <span class="label">💎 Total PnL</span>
                <span class="value {{ 'positive' if state.total_pnl >= 0 else 'negative' }}" style="font-size:18px;">{{ "%+.2f"|format(state.total_pnl) }} USDT</span>
            </div>
            <div class="status-row"><span class="label">💱 Цена SOL</span><span class="value">${{ "%.2f"|format(state.last_price) }}</span></div>
            <div class="status-row">
                <span class="label">🧠 AI Сигнал</span>
                <span class="value"><span class="signal-badge signal-{{ state.signal }}">{{ state.signal }}</span></span>
            </div>
            <div class="status-row"><span class="label">📋 Ордеров</span><span class="value">{{ state.orders_count }}</span></div>
            <div class="status-row"><span class="label">💼 Позиций</span><span class="value">{{ state.positions_count }}</span></div>
        </div>

        {% if state.positions %}
        <div class="status-card">
            <h3 style="margin:0 0 15px 0;">💼 Открытые позиции</h3>
            {% for pos in state.positions %}
            <div class="position-card">
                <div class="status-row"><span class="label">{{ pos.side }}</span><span class="value">{{ pos.size }} SOL</span></div>
                <div class="status-row"><span class="label">Вход</span><span class="value">${{ "%.2f"|format(pos.avg_price) }}</span></div>
                <div class="status-row"><span class="label">PnL</span><span class="value {{ 'positive' if pos.upl >= 0 else 'negative' }}">{{ "%+.2f"|format(pos.upl) }} USDT</span></div>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <div class="actions">
            <a href="/refresh" class="btn btn-refresh">🔄 Обновить</a>
            <a href="/logout" class="btn btn-logout">🚪 Выйти</a>
        </div>
        {% endif %}

        <p class="last-update">Последнее обновление: {{ state.last_update or '—' }}</p>
    </div>
    <script>setTimeout(() => window.location.reload(), 30000);</script>
</body>
</html>
"""


# ─── РОУТЫ ───

@app.route("/", methods=["GET", "POST"])
def login():
    """Страница входа"""
    if request.method == "POST":
        role = request.form.get("role", "guest")
        password = request.form.get("password", "")

        if role == "admin" and password == ADMIN_PASSWORD:
            session["role"] = "admin"
            return redirect(url_for("dashboard"))
        elif role == "guest" and password == GUEST_PASSWORD:
            session["role"] = "guest"
            return redirect(url_for("dashboard"))
        else:
            return render_template_string(LOGIN_HTML, error="❌ Неверный пароль")

    return render_template_string(LOGIN_HTML, error=None)


@app.route("/logout")
def logout():
    """Выход"""
    session.clear()
    return redirect(url_for("login"))


@app.route("/bot")
def dashboard():
    """Главная страница — зависит от роли"""
    role = session.get("role")
    if not role:
        return redirect(url_for("login"))

    update_bot_state()

    if role == "admin":
        return render_template_string(ADMIN_HTML, state=bot_state)
    else:
        return render_template_string(GUEST_HTML, state=bot_state)


@app.route("/start")
def start_bot():
    """Запуск бота — только для админа"""
    if session.get("role") != "admin":
        return redirect(url_for("dashboard"))
    start_bot_in_background()
    return redirect(url_for("dashboard"))


@app.route("/stop")
def stop_bot():
    """Остановка бота — только для админа"""
    if session.get("role") != "admin":
        return redirect(url_for("dashboard"))
    if bot_state["started"]:
        bot.stop()
        bot_state["started"] = False
    return redirect(url_for("dashboard"))


@app.route("/refresh")
def refresh():
    """Обновить состояние"""
    update_bot_state()
    return redirect(url_for("dashboard"))


@app.route("/status")
def status_page():
    """JSON API"""
    update_bot_state()
    return jsonify({
        "running": bot_state["started"],
        "balance": bot_state["balance"],
        "realized_pnl": bot_state["realized_pnl"],
        "unrealized_pnl": bot_state["unrealized_pnl"],
        "total_pnl": bot_state["total_pnl"],
        "price": bot_state["last_price"],
        "signal": bot_state["signal"],
        "orders": bot_state["orders_count"],
        "positions": bot_state["positions_count"],
        "last_update": bot_state["last_update"],
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


# Автозапуск бота
@app.before_request
def ensure_bot_running():
    if not bot_state["started"]:
        start_bot_in_background()
