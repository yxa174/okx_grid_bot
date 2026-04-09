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
    "grid_levels":      50,
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

    # ── TP/SL ──────────────────────────────────────────────────────
    "take_profit_pct":  0.5,        # Take Profit % (0.5 = 0.5%)
    "stop_loss_pct":    1.0,        # Stop Loss %

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
    # УЛУЧШЕНО: добавлены multi-timeframe analysis и adversarial roles
    "ai_strategies": {
        # Groq - Trend Following Agent (следует за трендом)
        "groq": {
            "role": "опытный trend-following трейдер с 15-летним стажем на криптовалютных рынках",
            "style": "следование за трендом с использованием EMA crossover на 4H",
            "risk": "умеренный",
            "instruction": "Ты TREND FOLLOWING агент. Твоя задача — определить текущий тренд и следовать ему.\n\nАНАЛИЗ ТРЕНДА:\n- Если EMA(20) > EMA(50) и цена выше обоих = БЫЧИЙ тренд\n- Если EMA(20) < EMA(50) и цена ниже обоих = МЕДВЕЖИЙ тренд\n- Если EMA пересекаются = НЕОПРЕДЕЛЕННОСТЬ\n\nПРАВИЛА:\n- В бычьем тренде отдавай предпочтение BUY/STRONG_BUY\n- В медвежьем тренде отдавай предпочтение SELL/STRONG_SELL\n- В неопределенности — NEUTRAL\n- Игнорируй краткосрочный шум, фокусируйся на 4H\n- Используй MACD для подтверждения силы тренда",
            "min_confidence": 0.6,
            "enabled": True,
            "timeframe": "4H",
            "agent_type": "trend_follower"
        },
        # OpenRouter - Mean Reversion Agent (ловит развороты)
        "openrouter": {
            "role": "агрессивный контртрендовый трейдер, специализирующийся на разворотах",
            "style": "mean reversion с RSI и Bollinger Bands на 1H",
            "risk": "высокий",
            "instruction": "Ты MEAN REVERSION агент. Твоя задача — находить точки разворота тренда.\n\nАНАЛИЗ РАЗВОРОТОВ:\n- RSI > 70 = ПЕРЕКУПЛЕННОСТЬ → ищи SELL сигналы\n- RSI < 30 = ПЕРЕПРОДАННОСТЬ → ищи BUY сигналы\n- Цена вышла за пределы Bollinger Bands = возможен возврат к середине\n\nПРАВИЛА:\n- При RSI > 75 и цене выше BB Upper = STRONG_SELL\n- При RSI < 25 и цене ниже BB Lower = STRONG_BUY\n- При RSI 60-70 или 30-40 = SELL или BUY (слабее)\n- Всегда проверяй: есть ли дивергенция RSI и цены?\n- Работай только на 1H таймфрейме",
            "min_confidence": 0.65,
            "enabled": True,
            "timeframe": "1H",
            "agent_type": "mean_reversion"
        },
        # Cohere - Risk Manager (консервативный контролёр)
        "cohere": {
            "role": "консервативный риск-менеджер крупного хедж-фонда",
            "style": "защита капитала, только высоковероятные сделки",
            "risk": "низкий",
            "instruction": "Ты RISK MANAGER агент. Твоя задача — защищать капитал и давать сигналы ТОЛЬКО при максимальной уверенности.\n\nКРИТЕРИИ ВЫСОКОЙ ВЕРОЯТНОСТИ:\n- Минимум 3 индикатора должны подтверждать сигнал\n- RSI + MACD + Bollinger Bands должны быть согласованы\n- Объём должен подтверждать движение\n- Fear & Greed должен соответствовать сигналу\n\nПРАВИЛА:\n- NEUTRAL по умолчанию — это нормальный сигнал\n- BUY только если ВСЕ индикаторы бычьи\n- SELL только если ВСЕ индикаторы медвежьи\n- Confidence должен быть > 0.75 для любого сигнала\n- Если сомневаешься — NEUTRAL (это правильный ответ)",
            "min_confidence": 0.75,
            "enabled": True,
            "timeframe": "1H+4H",
            "agent_type": "risk_manager"
        },
        # DeepSeek - Scalping Agent (агрессивные ранние входы)
        "deepseek": {
            "role": "агрессивный скальпер с фокусом на ранние входы",
            "style": "скальпинг на 15m с использованием уровней поддержки/сопротивления",
            "risk": "агрессивный",
            "instruction": "Ты SCALPING агент. Твоя задача — ловить ранние входы при первых признаках движения.\n\nАНАЛИЗ ДЛЯ СКАЛЬПИНГА:\n- Ищи пробои уровней поддержки/сопротивления\n- Используй RSI на 15m для раннего входа\n- Смотри на объём — резкий рост объёма = подтверждение\n- MACD histogram меняет знак = ранний сигнал\n\nПРАВИЛА:\n- Давай сигнал при confidence > 0.55\n- Работай на пробоях и отскоках от уровней\n- Бычий пробой сопротивления = BUY\n- Медвежий пробой поддержки = SELL\n- Не бойся ошибаться — скальперы работают на количестве\n- Фокусируйся на 15m таймфрейме",
            "min_confidence": 0.55,
            "enabled": True,
            "timeframe": "15m",
            "agent_type": "scalper"
        },
        # Gemini - Market Context Agent (мульти-таймфрейм аналитик)
        "gemini": {
            "role": "профессиональный аналитик с мульти-таймфрейм подходом",
            "style": "комплексный анализ M5, M15, H1, H4 для полного контекста",
            "risk": "умеренный",
            "instruction": "Ты MARKET CONTEXT агент. Твоя задача — дать полную картину по всем таймфреймам.\n\nМУЛЬТИ-ТАЙМФРЕЙМ АНАЛИЗ:\n- M5 (5 минут): краткосрочный импульс\n- M15 (15 минут): среднесрочный тренд\n- H1 (1 час): основной тренд\n- H4 (4 часа): глобальный тренд\n\nПРАВИЛА:\n- Если все таймфреймы согласны = высокая уверенность\n- Если старшие (H1, H4) против младших = следуй старшим\n- Если разногласие = NEUTRAL или низкая уверенность\n- Анализируй корреляцию с BTC\n- Учитывай Fear & Greed Index\n- Определи торговую сессию (Азиатская/Европейская/Американская)\n- Дай общее решение на основе консенсуса",
            "min_confidence": 0.6,
            "enabled": True,
            "timeframe": "M5+M15+H1+H4",
            "agent_type": "market_context"
        }
    },

    # Вето от старших таймфреймов
    "ai_veto_from_higher_tf": True,  # Если H4 против сигнала - не входить

    "ai_enabled_providers": ["groq", "openrouter", "cohere", "deepseek", "gemini"],
    "ai_backtest_auto":   True,    # Авто-бэктест раз в 24ч

    # Улучшенные настройки для AI промптов
    "ai_prompt_version": "v2",
    "ai_include_ohlcv": True,  # Включать OHLCV данные в промпт
    "ai_multi_timeframe": True,  # Использовать мульти-таймфрейм анализ
}
