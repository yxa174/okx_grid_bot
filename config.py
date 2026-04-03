from dotenv import load_dotenv
import os

load_dotenv()
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")

 
_raw_ids = os.getenv("ALLOWED_CHAT_IDS", "")
ALLOWED_CHAT_IDS: list[int] = (
    [int(x.strip()) for x in _raw_ids.split(",") if x.strip()]
    if _raw_ids.strip() else []
)
 
CONFIG: dict = {
    # ── Инструмент ────────────────────────────────────────────────
    "symbol":           "SOLUSDT",
    "leverage":         10,          # плечо
    "qty_step":         0.1,        # минимальный шаг qty для SOLUSDT
    "min_qty":          0.1,
    "price_decimals":   2,
 
    # ── Сетка ─────────────────────────────────────────────────────
    "grid_levels":      20,         # 20 уровней
    "atr_multiplier":   2.5,        # диапазон = цена ± 2.5 * ATR(14, 4h)
    "max_range_pct":    0.20,       # макс ширина сетки (20% от цены)
 
    # ── Размер позиции ────────────────────────────────────────────
    "max_qty":          0.5,        # максимум SOL на один ордер
    "max_balance_pct":  0.60,       # используем 60% баланса на все ордера
 
    # ── Глобальные защиты ─────────────────────────────────────────
    "global_sl_usdt":   25.0,       # стоп всего бота если unrealized < -25 USDT
    "global_tp_usdt":   40.0,       # стоп всего бота если realized > +40 USDT
 
    # ── Скорость ──────────────────────────────────────────────────
    "check_interval":   10,         # проверка каждые 10 секунд
 
    # ── Прочее ────────────────────────────────────────────────────
    "min_balance_usdt": 50.0,       # минимальный баланс для работы бота
}
 