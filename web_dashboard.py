"""
Flask Web Dashboard для OKX Grid Bot
Запускается на PythonAnywhere как WSGI приложение
"""

import json
import threading
import time
from datetime import datetime

from flask import Flask, jsonify, render_template_string, request

# Импортируем бота
from okx_grid_bot import (
    bot,
    CONFIG,
    log,
    ALLOWED_CHAT_IDS,
    persistent_keyboard,
    main_keyboard,
)

app = Flask(__name__)

# Храним состояние бота
bot_state = {
    "started": False,
    "start_time": None,
    "last_update": None,
    "last_price": 0,
    "balance": 0,
    "pnl": 0,
    "signal": "NEUTRAL",
    "orders_count": 0,
    "positions_count": 0,
    "errors": [],
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
        bot_state["last_price"] = bot.last_price
        bot_state["balance"] = bot.get_balance()
        bot_state["pnl"] = bot.realized_pnl + bot.get_unrealized_pnl()
        bot_state["signal"] = bot.last_signal.get("signal", "NEUTRAL")
        bot_state["orders_count"] = len(bot.active_orders)
        
        # Получаем позиции
        try:
            r = bot.account_api.get_positions(instType="SWAP", instId=CONFIG["symbol"])
            if r.get("code") == "0":
                bot_state["positions_count"] = len(r.get("data", []))
        except Exception:
            bot_state["positions_count"] = 0
    except Exception as e:
        log.warning(f"Ошибка обновления состояния: {e}")


# HTML шаблон дашборда
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>OKX Grid Bot V3</title>
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
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        h1 {
            color: #00d4aa;
            text-align: center;
            margin-bottom: 30px;
        }
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
        .status-row:last-child {
            border-bottom: none;
        }
        .label {
            color: #888;
        }
        .value {
            color: #fff;
            font-weight: bold;
        }
        .positive { color: #00d4aa; }
        .negative { color: #ff4757; }
        .neutral { color: #ffa502; }
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
        .btn-telegram { background: #0088cc; }
        .actions {
            text-align: center;
            margin: 20px 0;
        }
        .signal-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        .signal-BUY, .signal-STRONG_BUY {
            background: rgba(0, 212, 170, 0.2);
            color: #00d4aa;
        }
        .signal-SELL, .signal-STRONG_SELL {
            background: rgba(255, 71, 87, 0.2);
            color: #ff4757;
        }
        .signal-NEUTRAL {
            background: rgba(255, 165, 2, 0.2);
            color: #ffa502;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #888;
        }
        .last-update {
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 OKX Grid Bot V3</h1>
        
        {% if not state.started %}
        <div class="status-card">
            <p style="text-align: center; color: #888;">Бот не запущен</p>
            <div class="actions">
                <a href="/start" class="btn btn-start">▶️ Запустить бота</a>
            </div>
        </div>
        {% else %}
        <div class="status-card">
            <div class="status-row">
                <span class="label">📊 Состояние</span>
                <span class="value">
                    {% if state.started %}
                    <span class="positive">✅ Работает</span>
                    {% else %}
                    <span class="negative">❌ Остановлен</span>
                    {% endif %}
                </span>
            </div>
            <div class="status-row">
                <span class="label">⏱ Uptime</span>
                <span class="value">{{ state.start_time or '—' }}</span>
            </div>
            <div class="status-row">
                <span class="label">💰 Баланс</span>
                <span class="value">{{ "%.2f"|format(state.balance) }} USDT</span>
            </div>
            <div class="status-row">
                <span class="label">📈 PnL</span>
                <span class="value {{ 'positive' if state.pnl >= 0 else 'negative' }}">
                    {{ "%+.2f"|format(state.pnl) }} USDT
                </span>
            </div>
            <div class="status-row">
                <span class="label">💱 Цена SOL</span>
                <span class="value">${{ "%.2f"|format(state.last_price) }}</span>
            </div>
            <div class="status-row">
                <span class="label">🧠 AI Сигнал</span>
                <span class="value">
                    <span class="signal-badge signal-{{ state.signal }}">
                        {{ state.signal }}
                    </span>
                </span>
            </div>
            <div class="status-row">
                <span class="label">📋 Ордеров</span>
                <span class="value">{{ state.orders_count }}</span>
            </div>
            <div class="status-row">
                <span class="label">💼 Позиций</span>
                <span class="value">{{ state.positions_count }}</span>
            </div>
        </div>

        <div class="actions">
            <a href="/refresh" class="btn btn-refresh">🔄 Обновить</a>
            <a href="/stop" class="btn btn-stop">⏹ Остановить</a>
            <a href="/status" class="btn btn-telegram">📊 Telegram Status</a>
        </div>
        {% endif %}

        <div class="last-update">
            Последнее обновление: {{ state.last_update or '—' }}
        </div>
    </div>

    <script>
        // Автообновление каждые 30 секунд
        setTimeout(() => {
            window.location.reload();
        }, 30000);
    </script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    """Главная страница дашборда"""
    update_bot_state()
    return render_template_string(DASHBOARD_HTML, state=bot_state)


@app.route("/start")
def start_bot():
    """Запуск бота"""
    start_bot_in_background()
    return "<h1>✅ Бот запущен!</h1><a href='/'>Назад к дашборду</a>"


@app.route("/stop")
def stop_bot():
    """Остановка бота"""
    if bot_state["started"]:
        bot.stop()
        bot_state["started"] = False
    return "<h1>⏹ Бот остановлен</h1><a href='/'>Назад к дашборду</a>"


@app.route("/refresh")
def refresh():
    """Обновить состояние и вернуться на дашборд"""
    update_bot_state()
    return dashboard()


@app.route("/status")
def status_page():
    """Полный статус бота (как в Telegram)"""
    update_bot_state()
    return jsonify(
        {
            "running": bot_state["started"],
            "balance": bot_state["balance"],
            "price": bot_state["last_price"],
            "pnl": bot_state["pnl"],
            "signal": bot_state["signal"],
            "orders": bot_state["orders_count"],
            "positions": bot_state["positions_count"],
            "last_update": bot_state["last_update"],
        }
    )


@app.route("/health")
def health():
    """Health check для PythonAnywhere"""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


# Автоматически запускаем бот при первом запросе к любому эндпоинту
@app.before_request
def ensure_bot_running():
    """Гарантирует что бот запущен"""
    if not bot_state["started"]:
        start_bot_in_background()
