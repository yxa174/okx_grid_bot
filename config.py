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
    "symbol":           "SOL-USDT-SWAP",  # Futures trading

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

    # ── Мульти-AI с разными стратегиями ───────────────────────────
    "ai_analysis_interval": 900,    # Анализ каждые 15 минут (сек)
    "ai_cache_ttl":       900,      # Кэш ответов LLM (сек)
    "ai_memory_size":     20,       # Храним 20 последних решений
    "ai_adaptive_weights": True,    # Адаптивные веса вкл/выкл

    # AI стратегии - разные подходы для разных провайдеров
    "ai_strategies": {
        # Groq - трендовая стратегия (следуй за трендом)
        "groq": {
            "role": "трендовый трейдер",
            "style": "следование за трендом на 4H таймфрейме",
            "risk": "умеренный",
            "instruction": "Следуй за основным трендом. Используй EMA 20/50 для определения тренда. Игнорируй краткосрочный шум. Давай сигнал только при подтверждённом тренде.",
            "min_confidence": 0.6,
            "enabled": True
        },
        # OpenRouter - контртрендовая (перекупленность/перепроданность)
        "openrouter": {
            "role": "контртрендовый трейдер",
            "style": "ловля разворотов на 1H таймфрейме",
            "risk": "высокий",
            "instruction": "Ищи перекупленность (RSI > 70) или перепроданность (RSI < 30). Используй дивергенции. Давай сигнал на разворот против текущего тренда.",
            "min_confidence": 0.65,
            "enabled": True
        },
        # Cohere - консервативная (высокая уверенность)
        "cohere": {
            "role": "консервативный трейдер",
            "style": "безопасная торговля с высоким подтверждением",
            "risk": "низкий",
            "instruction": "Давай сигнал ТОЛЬКО при высокой уверенности (confidence > 0.7). Дождись подтверждения от нескольких индикаторов. Избегай сигналов в периоды низкой волатильности.",
            "min_confidence": 0.7,
            "enabled": True
        },
        # DeepSeek - агрессивная (ранние входы)
        "deepseek": {
            "role": "агрессивный скальпер",
            "style": "ранние входы на 15m таймфрейме",
            "risk": "агрессивный",
            "instruction": "Лови ранние входы при первых признаках движения. Давай сигнал при confidence > 0.55. Работай на пробоях уровней поддержки/сопротивления.",
            "min_confidence": 0.55,
            "enabled": True
        },
        # Gemini - мульти-таймфрейм анализ
        "gemini": {
            "role": "мульти-таймфрейм аналитик",
            "style": "анализ M5, H1, H4 для принятия решения",
            "risk": "умеренный",
            "instruction": "Проанализируй данные на всех таймфреймах (M5, H1, H4). Если старшие таймфреймы против сигнала - игнорируй. Дай общее решение на основе консенсуса.",
            "min_confidence": 0.6,
            "enabled": True
        }
    },

    # Вето от старших таймфреймов
    "ai_veto_from_higher_tf": True,  # Если H4 против сигнала - не входить

    "ai_enabled_providers": ["groq", "openrouter", "cohere", "deepseek"],
    "ai_backtest_auto":   True,    # Авто-бэктест раз в 24ч
}
