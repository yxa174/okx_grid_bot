from dotenv import load_dotenv
import os

load_dotenv()

# OKX API credentials
OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_API_SECRET = os.getenv("OKX_API_SECRET")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE")
OKX_FLAG = os.getenv("OKX_FLAG", "1")  # 0 = live, 1 = demo

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# AI API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyClw4-sfNSZPy9EOGnR2UGkpWJxhzSdmB8")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

_raw_ids = os.getenv("ALLOWED_CHAT_IDS", "")
ALLOWED_CHAT_IDS: list[int] = (
    [int(x.strip()) for x in _raw_ids.split(",") if x.strip()]
    if _raw_ids.strip() else []
)

CONFIG: dict = {
    # ── Инструмент ────────────────────────────────────────────────
    "symbol":           "SOL-USDT",  # Spot trading

    # ── Сетка ─────────────────────────────────────────────────────
    "grid_levels":      20,
    "atr_multiplier":   2.5,
    "max_range_pct":    0.20,

    # ── Размер позиции ────────────────────────────────────────────
    "max_qty":          0.5,        # Максимум SOL на один ордер
    "max_balance_pct":  0.60,       # Используем 60% баланса на все ордера

    # ── Глобальные защиты ─────────────────────────────────────────
    "global_sl_usdt":   25.0,       # Стоп бота если loss > 25 USDT
    "global_tp_usdt":   40.0,       # Стоп бота если profit > 40 USDT

    # ── Скорость ──────────────────────────────────────────────────
    "check_interval":   10,         # Проверка каждые 10 секунд

    # ── Прочее ────────────────────────────────────────────────────
    "min_balance_usdt": 50.0,       # Минимальный баланс для работы бота
    "qty_step":         0.1,        # Минимальный шаг qty для SOL-USDT
    "min_qty":          0.1,
    "price_decimals":   2,

    # ── Мульти-AI ─────────────────────────────────────────────────
    "ai_analysis_interval": 900,    # Анализ каждые 15 минут (сек)
    "ai_cache_ttl":       900,      # Кэш ответов LLM (сек)
    "ai_memory_size":     20,       # Храним 20 последних решений
    "ai_adaptive_weights": True,    # Адаптивные веса вкл/выкл
    "ai_btc_corr_weight": 0.5,     # Вес корреляции BTC в решении
    "ai_role":            "профессиональный крипто-трейдер с 10-летним опытом",
    "ai_style":           "свинг-трейдинг на 4H таймфрейме",
    "ai_risk":            "умеренный",
    "ai_timeframe":       "4H",
    "ai_enabled_providers": ["groq", "openrouter", "cohere", "deepseek"],
    "ai_backtest_auto":   True,    # Авто-бэктест раз в 24ч
}
