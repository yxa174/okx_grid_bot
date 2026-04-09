"""
OKX TESTNET — SPOT GRID BOT V3 (MULTI-AI)
==========================================
Архитектура:

1. СПОТОВАЯ ТОРГОВЛЯ (Spot, без плеча)
   - Покупаем SOL за USDT, продаём SOL за USDT
   - Нет плеча → нет риска ликвидации
   - Нет шортов → только рост = прибыль

2. ДИНАМИЧЕСКИЙ ДИАПАЗОН (Auto-Range по ATR)
3. TRAILING UP (Следование вверх)
4. ГЛОБАЛЬНЫЙ STOP-LOSS / TAKE-PROFIT
5. МУЛЬТИ-AI АНАЛИЗ:
   - Gemini 2.0 Flash
   - Groq (Llama 3.1 70B)
   - OpenRouter (Llama 3 8B, free)
   - Cohere Command R+ (free tier)
   - DeepSeek V3 (free tier)
   - Адаптивные веса, память 20 решений
   - Внешние данные: Fear&Greed, BTC, корреляция, сессии
6. BACKTESTING точности AI
"""

import asyncio
import csv
import json
import logging
import logging.handlers
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
from okx import Trade, Account, MarketData

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch не установлен — LSTM отключён")

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters

from config import (
    OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE, OKX_FLAG,
    TELEGRAM_TOKEN, ALLOWED_CHAT_IDS, CONFIG,
    GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, HUGGINGFACE_API_KEY,
    COHERE_API_KEY, DEEPSEEK_API_KEY
)

# ── Константы биржи ───────────────────────────────────────────────
MAX_ORDERS = 50
MIN_QTY = 0.1

# Основной логгер бота
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.handlers.RotatingFileHandler(
            "bot.log",
            maxBytes=5*1024*1024,  # 5 MB
            backupCount=3,
            encoding="utf-8"
        ),
    ],
)

# Отдельный логгер для AI решений
ai_log_handler = logging.handlers.RotatingFileHandler(
    "ai_decisions.log",
    maxBytes=2*1024*1024,  # 2 MB
    backupCount=2,
    encoding="utf-8"
)
ai_logger = logging.getLogger("AI_Decisions")
ai_logger.setLevel(logging.INFO)
ai_logger.addHandler(ai_log_handler)
ai_logger.propagate = False  # Не дублировать в основной лог

# Отключаем HTTP логи от библиотек
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("okx").setLevel(logging.WARNING)

log = logging.getLogger("GridBotV3")


class StatusTracker:
    """Отслеживает ошибки и статус для отображения в консоли"""
    def __init__(self):
        self.critical_errors = 0    # Ошибки закрытия позиций, SL
        self.warnings = 0           # Предупреждения API, rate limits
        self.trades_count = 0       # Количество сделок
        self.total_pnl = 0.0        # Общий PnL
        self.open_positions = 0     # Открытые позиции
        self.open_orders = 0        # Открытые ордера
        self.balance = 0.0          # Баланс
        self.price = 0.0            # Текущая цена
        self.grid_range = ""        # Диапазон сетки
        self.last_update = ""       # Последнее важное событие
        self.start_time = time.time()
        self.running = False
        self._lines_rendered = 0    # Сколько строк было выведено

    def add_critical(self, msg: str):
        self.critical_errors += 1
        self.last_update = f"❌ {msg}"

    def add_warning(self, msg: str):
        self.warnings += 1
        self.last_update = f"⚠️ {msg}"

    def add_trade(self, pnl: float):
        self.trades_count += 1
        self.total_pnl += pnl
        sign = "+" if pnl >= 0 else ""
        self.last_update = f"💰 {sign}{pnl:.2f} USDT"

    def render(self) -> str:
        total_time = time.time() - self.start_time
        days = int(total_time / 86400)
        hours = int((total_time % 86400) / 3600)
        mins = int((total_time % 3600) / 60)
        secs = int(total_time % 60)
        if days > 0:
            uptime = f"{days}д {hours:02d}:{mins:02d}:{secs:02d}"
        else:
            uptime = f"{hours:02d}:{mins:02d}:{secs:02d}"

        status_icon = "🟢" if self.running else "🔴"
        pnl_color = "🟢" if self.total_pnl >= 0 else "🔴"
        sign = "+" if self.total_pnl >= 0 else ""

        lines = [
            f"{'='*52}",
            f"  {status_icon} GRID BOT V3 — OKX Testnet",
            f"  ⏱ Аптайм:  {uptime}   🔄 Циклов: {self.trades_count + self.open_orders}",
            f"{'─'*52}",
            f"  💵 Баланс:   {self.balance:>10.2f} USDT  |  📊 Цена: {self.price:.2f}",
            f"  📐 Сетка:    {self.grid_range:>20s}",
            f"  📈 Позиции:  {self.open_positions:>5d}  |  📋 Ордера: {self.open_orders}",
            f"  {pnl_color} PnL:       {sign}{self.total_pnl:>9.2f} USDT  |  💰 Сделок: {self.trades_count}",
            f"  ❌ Ошибки:  {self.critical_errors:>5d}  |  ⚠️ Warn: {self.warnings}",
            f"{'─'*52}",
            f"  📝 {self.last_update if self.last_update else 'Ожидание...'}",
            f"{'='*52}",
        ]
        self._lines_rendered = len(lines)
        return "\n".join(lines)


status = StatusTracker()


def render_status():
    """Отрисовывает статус в консоли — затирает предыдущий"""
    import sys
    output = status.render()
    # Поднимаем курсор на _lines_rendered строк вверх
    if status._lines_rendered > 0:
        sys.stdout.write(f"\033[{status._lines_rendered}A")
    sys.stdout.write("\033[J")  # Очищаем от курсора до конца экрана
    sys.stdout.write(output + "\n")
    sys.stdout.flush()


def retry_api(retries=3, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_err = None
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    log.warning(f"⚠️ Ошибка API ({func.__name__}), попытка {i+1}/{retries}: {e}")
                    time.sleep(delay * (i + 1))
            raise last_err
        return wrapper
    return decorator


def round_qty(qty: float) -> float:
    step = CONFIG.get("qty_step", 0.1)
    min_qty = CONFIG.get("min_qty", 0.1)
    steps = round(qty / step)
    result = max(min_qty, steps * step)
    return round(result, 2)


def round_price(price: float) -> float:
    decimals = CONFIG.get("price_decimals", 2)
    return round(max(0.01, price), decimals)


# ══════════════════════════════════════════════════════════════════
#  MARKET DATA PROVIDER — Внешние данные
# ══════════════════════════════════════════════════════════════════

class MarketDataProvider:
    """Собирает внешние данные: Fear&Greed, BTC, корреляция, сессии"""

    def __init__(self):
        self._fear_greed_cache = {"value": 50, "ts": 0, "ttl": 300}
        self._btc_cache = {"price": 0, "dominance": 0, "ts": 0, "ttl": 300}
        self._sol_prices = deque(maxlen=50)
        self._btc_prices = deque(maxlen=50)

    def get_fear_greed(self) -> dict:
        now = time.time()
        if now - self._fear_greed_cache["ts"] < self._fear_greed_cache["ttl"]:
            return self._fear_greed_cache
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            data = r.json()["data"][0]
            val = int(data["value"])
            label = data["value_classification"]
            self._fear_greed_cache = {"value": val, "label": label, "ts": now, "ttl": 300}
            return self._fear_greed_cache
        except Exception as e:
            log.warning(f"Fear&Greed ошибка: {e}")
            return self._fear_greed_cache

    def get_btc_data(self) -> dict:
        now = time.time()
        if now - self._btc_cache["ts"] < self._btc_cache["ttl"]:
            return self._btc_cache
        try:
            r = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd", timeout=10)
            btc_price = r.json()["bitcoin"]["usd"]
            self._btc_cache = {"price": btc_price, "dominance": 54.0, "ts": now, "ttl": 300}
            return self._btc_cache
        except Exception as e:
            log.warning(f"BTC данные ошибка: {e}")
            return self._btc_cache

    def update_prices(self, sol_price: float, btc_price: float):
        self._sol_prices.append(sol_price)
        self._btc_prices.append(btc_price)

    def get_correlation(self) -> float:
        if len(self._sol_prices) < 10 or len(self._btc_prices) < 10:
            return 0.7
        sol = np.array(list(self._sol_prices))
        btc = np.array(list(self._btc_prices))
        sol_ret = np.diff(sol) / (sol[:-1] + 1e-10)
        btc_ret = np.diff(btc) / (btc[:-1] + 1e-10)
        if np.std(sol_ret) < 1e-10 or np.std(btc_ret) < 1e-10:
            return 0.7
        corr = np.corrcoef(sol_ret, btc_ret)[0, 1]
        if np.isnan(corr):
            return 0.7
        return round(float(corr), 3)

    def get_trading_session(self) -> str:
        utc_hour = datetime.now(timezone.utc).hour
        if 0 <= utc_hour < 8:
            return "Азиатская"
        elif 8 <= utc_hour < 16:
            return "Европейская"
        else:
            return "Американская"

    def get_day_of_week(self) -> str:
        days = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
        return days[datetime.now(timezone.utc).weekday()]

    def get_multi_timeframe_data(self, market_api, symbol: str = "SOL-USDT-SWAP") -> dict:
        """Получает OHLCV данные с разных таймфреймов для multi-timeframe анализа"""
        timeframes = {
            "M5": "5m",
            "M15": "15m",
            "H1": "1H",
            "H4": "4H"
        }

        result = {}

        for tf_name, tf_value in timeframes.items():
            try:
                # Получаем последние 20 свечей для каждого таймфрейма
                candles = market_api.get_candlesticks(
                    instId=symbol,
                    bar=tf_value,
                    limit="20"
                )

                if candles.get('code') == '0' and candles.get('data'):
                    # OKX возвращает данные в обратном порядке (новые сначала)
                    data = candles['data']

                    # Преобразуем в удобный формат
                    ohlcv = []
                    for candle in reversed(data):  # Переворачиваем, чтобы старые были сначала
                        ts, o, h, l, c, vol, vol_ccy, vol_ccy_quote = candle
                        ohlcv.append({
                            'timestamp': int(ts),
                            'open': float(o),
                            'high': float(h),
                            'low': float(l),
                            'close': float(c),
                            'volume': float(vol)
                        })

                    # Вычисляем простые метрики
                    if len(ohlcv) >= 2:
                        current = ohlcv[-1]
                        previous = ohlcv[-2]
                        change_pct = ((current['close'] - previous['close']) / previous['close']) * 100

                        # Определяем тренд по EMA
                        closes = pd.Series([c['close'] for c in ohlcv])
                        ema_fast = closes.ewm(span=8).mean()
                        ema_slow = closes.ewm(span=20).mean()

                        trend = "BULLISH" if ema_fast.iloc[-1] > ema_slow.iloc[-1] else "BEARISH"

                        result[tf_name] = {
                            'current_price': current['close'],
                            'change_pct': change_pct,
                            'trend': trend,
                            'high': current['high'],
                            'low': current['low'],
                            'volume': current['volume'],
                            'ema_fast': ema_fast.iloc[-1],
                            'ema_slow': ema_slow.iloc[-1]
                        }
                    else:
                        result[tf_name] = None
                else:
                    result[tf_name] = None
            except Exception as e:
                log.warning(f"⚠️ Ошибка получения данных {tf_name}: {e}")
                result[tf_name] = None

        return result

    def get_market_context(self, sol_price: float) -> dict:
        btc = self.get_btc_data()
        fg = self.get_fear_greed()
        self.update_prices(sol_price, btc.get("price", 0))
        return {
            "fear_greed_value": fg.get("value", 50),
            "fear_greed_label": fg.get("label", "Нейтрально"),
            "btc_price": btc.get("price", 0),
            "btc_dominance": btc.get("dominance", 54.0),
            "sol_btc_correlation": self.get_correlation(),
            "trading_session": self.get_trading_session(),
            "day_of_week": self.get_day_of_week(),
        }


# ══════════════════════════════════════════════════════════════════
#  AI ANALYZER (технические индикаторы)
# ══════════════════════════════════════════════════════════════════

class AIAnalyzer:
    def __init__(self):
        self.price_history = deque(maxlen=200)
        self.volume_history = deque(maxlen=200)
        self.feature_buffer = deque(maxlen=200)
        log.info("AI Analyzer инициализирован (без LSTM)")

    def add_price(self, price: float, volume: float = 0.0):
        self.price_history.append(price)
        self.volume_history.append(volume)
        if len(self.price_history) >= 50:
            feats = self._compute_features()
            if feats is not None:
                self.feature_buffer.append(feats)

    def _compute_features(self):
        if len(self.price_history) < 50:
            return None
        c = pd.Series(list(self.price_history))
        v = pd.Series(list(self.volume_history))
        ret1 = c.pct_change(1).iloc[-1]
        ret3 = c.pct_change(3).iloc[-1]
        ret6 = c.pct_change(6).iloc[-1]
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi_series = 100 - (100 / (1 + gain / (loss + 1e-10)))
        rsi = rsi_series.iloc[-1]
        stoch_rsi = ((rsi_series - rsi_series.rolling(14).min()) /
                     (rsi_series.rolling(14).max() - rsi_series.rolling(14).min() + 1e-10)).iloc[-1]
        williams_r = ((c.rolling(14).max() - c) /
                      (c.rolling(14).max() - c.rolling(14).min() + 1e-10)).iloc[-1]
        tp = c
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        cci = ((tp - sma_tp) / (0.015 * mad + 1e-10) / 200).iloc[-1]
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        macd = ema12 - ema26
        macd_hist = ((macd - macd.ewm(span=9).mean()) / (c + 1e-10)).iloc[-1]
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        bb_width = (4 * std20 / (sma20 + 1e-10)).iloc[-1]
        bb_pos = ((c - (sma20 - 2 * std20)) / (4 * std20 + 1e-10)).iloc[-1]
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        obv_norm = (obv / (obv.rolling(50).std() + 1e-10)).iloc[-1]
        momentum = (c / (c.shift(10) + 1e-10) - 1).iloc[-1]
        tr = pd.concat([(c - c).abs(), (c - c.shift()).abs(), (c - c.shift()).abs()], axis=1).max(axis=1)
        atr_ratio = (tr.rolling(14).mean() / (c + 1e-10)).iloc[-1]
        ma_trend = ((c.rolling(20).mean() - c.rolling(50).mean()) / c).iloc[-1]
        vol_ratio = (v / (v.rolling(20).mean() + 1e-10)).iloc[-1]
        return [ret1, ret3, ret6, rsi, stoch_rsi, williams_r, cci,
                macd_hist, bb_width, bb_pos, obv_norm, momentum,
                atr_ratio, ma_trend, vol_ratio]

    def get_indicators(self) -> dict:
        if len(self.price_history) < 50:
            return {}
        c = pd.Series(list(self.price_history))
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        macd = ema12 - ema26
        sma = c.rolling(20).mean()
        std = c.rolling(20).std()
        return {
            "rsi": round(float(rsi.iloc[-1]), 2),
            "macd_hist": round(float((macd - macd.ewm(span=9).mean()).iloc[-1]), 2),
            "bb_upper": round(float((sma + 2 * std).iloc[-1]), 2),
            "bb_lower": round(float((sma - 2 * std).iloc[-1]), 2),
            "ma20": round(float(sma.iloc[-1]), 2),
            "ma50": round(float(c.rolling(50).mean().iloc[-1]), 2),
        }

    def get_signal(self) -> dict:
        ind = self.get_indicators()
        if not ind:
            return {"signal": "NEUTRAL", "score": 0, "indicators": {}}
        score = 0
        rsi = ind["rsi"]
        if rsi < 25: score += 3
        elif rsi < 35: score += 2
        elif rsi < 45: score += 1
        elif rsi > 75: score -= 3
        elif rsi > 65: score -= 2
        elif rsi > 55: score -= 1
        if ind["macd_hist"] > 0: score += 1
        else: score -= 1
        if score >= 4: signal = "STRONG_BUY"
        elif score >= 2: signal = "BUY"
        elif score <= -4: signal = "STRONG_SELL"
        elif score <= -2: signal = "SELL"
        else: signal = "NEUTRAL"
        return {"signal": signal, "score": score, "indicators": ind}


# ══════════════════════════════════════════════════════════════════
#  LLM PROVIDERS
# ══════════════════════════════════════════════════════════════════

class LLMProvider:
    """Базовый класс для LLM провайдеров с ограниченным кэшем"""
    name = "base"
    cache_ttl = CONFIG.get("ai_cache_ttl", 900)
    cache_maxsize = 100  # Ограничение кэша на провайдер

    def __init__(self):
        # Instance-level bounded cache
        self._cache_dict = {}  # key -> (timestamp, result)
        self._cache_order = []  # List to track insertion order (oldest first)

    def _is_cached(self, key: str) -> bool:
        if key in self._cache_dict:
            ts, _ = self._cache_dict[key]
            if time.time() - ts < self.cache_ttl:
                return True
            else:
                # TTL expired, remove from cache
                del self._cache_dict[key]
                return False
        return False

    def _get_cached(self, key: str):
        if key in self._cache_dict:
            return self._cache_dict[key][1]
        return None

    def _set_cache(self, key: str, value: dict):
        # Remove expired entries first
        cutoff = time.time() - self.cache_ttl
        expired_keys = [k for k, (ts, _) in self._cache_dict.items() if ts < cutoff]
        for k in expired_keys:
            del self._cache_dict[k]
            if k in self._cache_order:
                self._cache_order.remove(k)

        # Enforce maxsize: remove oldest entries if over limit
        while len(self._cache_dict) >= self.cache_maxsize and self._cache_order:
            oldest_key = self._cache_order.pop(0)
            if oldest_key in self._cache_dict:
                del self._cache_dict[oldest_key]

        # Add new entry
        self._cache_order.append(key)
        self._cache_dict[key] = (time.time(), value)

    def get_signal(self, prompt: str) -> dict:
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    name = "gemini"

    def get_signal(self, prompt: str) -> dict:
        cache_key = hash(prompt)
        if self._is_cached(cache_key):
            return self._get_cached(cache_key)
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 200}
            }
            r = requests.post(url, json=payload, timeout=30)
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            result = self._parse_response(text)
            self._set_cache(cache_key, result)
            log.info(f"🔮 Gemini: {result['signal']} (conf={result['confidence']:.2f})")
            ai_logger.info(
                f"GEMINI │ Signal: {result['signal']} │ Conf: {result['confidence']:.2f} │ "
                f"Reasoning: {result.get('reasoning', 'N/A')[:100]}"
            )
            return result
        except Exception as e:
            log.warning(f"Gemini ошибка: {e}")
            return {"signal": "NEUTRAL", "confidence": 0.5, "reasoning": "Ошибка API"}

    def _parse_response(self, text: str) -> dict:
        text_upper = text.upper()
        if "STRONG BUY" in text_upper or "STRONG_BUY" in text_upper:
            signal = "STRONG_BUY"
            confidence = 0.85
        elif "BUY" in text_upper:
            signal = "BUY"
            confidence = 0.7
        elif "STRONG SELL" in text_upper or "STRONG_SELL" in text_upper:
            signal = "STRONG_SELL"
            confidence = 0.85
        elif "SELL" in text_upper:
            signal = "SELL"
            confidence = 0.7
        else:
            signal = "NEUTRAL"
            confidence = 0.5
        for line in text.split("\n"):
            if "confidence" in line.lower():
                try:
                    nums = [float(s) for s in line.split() if s.replace(".", "").isdigit()]
                    if nums:
                        confidence = max(0.1, min(1.0, nums[0]))
                except:
                    pass
        return {"signal": signal, "confidence": confidence, "reasoning": text[:200]}


class GroqProvider(LLMProvider):
    name = "groq"

    def get_signal(self, prompt: str) -> dict:
        cache_key = hash(prompt)
        if self._is_cached(cache_key):
            return self._get_cached(cache_key)
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200
            }
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            result = self._parse_response(text)
            self._set_cache(cache_key, result)
            log.info(f"🔮 Groq: {result['signal']} (conf={result['confidence']:.2f})")
            ai_logger.info(
                f"GROQ │ Signal: {result['signal']} │ Conf: {result['confidence']:.2f} │ "
                f"Reasoning: {result.get('reasoning', 'N/A')[:100]}"
            )
            return result
        except Exception as e:
            log.warning(f"Groq ошибка: {e}")
            return {"signal": "NEUTRAL", "confidence": 0.5, "reasoning": "Ошибка API"}

    def _parse_response(self, text: str) -> dict:
        text_upper = text.upper()
        if "STRONG BUY" in text_upper or "STRONG_BUY" in text_upper:
            signal = "STRONG_BUY"
            confidence = 0.85
        elif "BUY" in text_upper:
            signal = "BUY"
            confidence = 0.7
        elif "STRONG SELL" in text_upper or "STRONG_SELL" in text_upper:
            signal = "STRONG_SELL"
            confidence = 0.85
        elif "SELL" in text_upper:
            signal = "SELL"
            confidence = 0.7
        else:
            signal = "NEUTRAL"
            confidence = 0.5
        for line in text.split("\n"):
            if "confidence" in line.lower():
                try:
                    nums = [float(s) for s in line.split() if s.replace(".", "").isdigit()]
                    if nums:
                        confidence = max(0.1, min(1.0, nums[0]))
                except:
                    pass
        return {"signal": signal, "confidence": confidence, "reasoning": text[:200]}


class OpenRouterProvider(LLMProvider):
    name = "openrouter"

    def get_signal(self, prompt: str) -> dict:
        cache_key = hash(prompt)
        if self._is_cached(cache_key):
            return self._get_cached(cache_key)
        if not OPENROUTER_API_KEY:
            return {"signal": "NEUTRAL", "confidence": 0.5, "reasoning": "Нет API ключа"}
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/bybit_bot",
                "X-Title": "GridBotV3"
            }
            payload = {
                "model": "meta-llama/llama-3-8b-instruct:free",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200
            }
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            result = self._parse_response(text)
            self._set_cache(cache_key, result)
            log.info(f"🔮 OpenRouter: {result['signal']} (conf={result['confidence']:.2f})")
            ai_logger.info(
                f"OPENROUTER │ Signal: {result['signal']} │ Conf: {result['confidence']:.2f} │ "
                f"Reasoning: {result.get('reasoning', 'N/A')[:100]}"
            )
            return result
        except Exception as e:
            log.warning(f"OpenRouter ошибка: {e}")
            return {"signal": "NEUTRAL", "confidence": 0.5, "reasoning": "Ошибка API"}

    def _parse_response(self, text: str) -> dict:
        text_upper = text.upper()
        if "STRONG BUY" in text_upper or "STRONG_BUY" in text_upper:
            signal = "STRONG_BUY"
            confidence = 0.85
        elif "BUY" in text_upper:
            signal = "BUY"
            confidence = 0.7
        elif "STRONG SELL" in text_upper or "STRONG_SELL" in text_upper:
            signal = "STRONG_SELL"
            confidence = 0.85
        elif "SELL" in text_upper:
            signal = "SELL"
            confidence = 0.7
        else:
            signal = "NEUTRAL"
            confidence = 0.5
        for line in text.split("\n"):
            if "confidence" in line.lower():
                try:
                    nums = [float(s) for s in line.split() if s.replace(".", "").isdigit()]
                    if nums:
                        confidence = max(0.1, min(1.0, nums[0]))
                except:
                    pass
        return {"signal": signal, "confidence": confidence, "reasoning": text[:200]}


class CohereProvider(LLMProvider):
    name = "cohere"

    def get_signal(self, prompt: str) -> dict:
        cache_key = hash(prompt)
        if self._is_cached(cache_key):
            return self._get_cached(cache_key)
        if not COHERE_API_KEY:
            return {"signal": "NEUTRAL", "confidence": 0.5, "reasoning": "Нет API ключа"}
        try:
            url = "https://api.cohere.ai/v1/chat"
            headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "command-r-plus",
                "message": prompt,
                "temperature": 0.3,
                "max_tokens": 200
            }
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            text = r.json()["text"]
            result = self._parse_response(text)
            self._set_cache(cache_key, result)
            log.info(f"🔮 Cohere: {result['signal']} (conf={result['confidence']:.2f})")
            ai_logger.info(
                f"COHERE │ Signal: {result['signal']} │ Conf: {result['confidence']:.2f} │ "
                f"Reasoning: {result.get('reasoning', 'N/A')[:100]}"
            )
            return result
        except Exception as e:
            log.warning(f"Cohere ошибка: {e}")
            return {"signal": "NEUTRAL", "confidence": 0.5, "reasoning": "Ошибка API"}

    def _parse_response(self, text: str) -> dict:
        text_upper = text.upper()
        if "STRONG BUY" in text_upper or "STRONG_BUY" in text_upper:
            signal = "STRONG_BUY"
            confidence = 0.85
        elif "BUY" in text_upper:
            signal = "BUY"
            confidence = 0.7
        elif "STRONG SELL" in text_upper or "STRONG_SELL" in text_upper:
            signal = "STRONG_SELL"
            confidence = 0.85
        elif "SELL" in text_upper:
            signal = "SELL"
            confidence = 0.7
        else:
            signal = "NEUTRAL"
            confidence = 0.5
        for line in text.split("\n"):
            if "confidence" in line.lower():
                try:
                    nums = [float(s) for s in line.split() if s.replace(".", "").isdigit()]
                    if nums:
                        confidence = max(0.1, min(1.0, nums[0]))
                except:
                    pass
        return {"signal": signal, "confidence": confidence, "reasoning": text[:200]}


class DeepSeekProvider(LLMProvider):
    name = "deepseek"

    def get_signal(self, prompt: str) -> dict:
        cache_key = hash(prompt)
        if self._is_cached(cache_key):
            return self._get_cached(cache_key)
        if not DEEPSEEK_API_KEY:
            return {"signal": "NEUTRAL", "confidence": 0.5, "reasoning": "Нет API ключа"}
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200
            }
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            result = self._parse_response(text)
            self._set_cache(cache_key, result)
            log.info(f"🔮 DeepSeek: {result['signal']} (conf={result['confidence']:.2f})")
            ai_logger.info(
                f"DEEPSEEK │ Signal: {result['signal']} │ Conf: {result['confidence']:.2f} │ "
                f"Reasoning: {result.get('reasoning', 'N/A')[:100]}"
            )
            return result
        except Exception as e:
            log.warning(f"DeepSeek ошибка: {e}")
            return {"signal": "NEUTRAL", "confidence": 0.5, "reasoning": "Ошибка API"}

    def _parse_response(self, text: str) -> dict:
        text_upper = text.upper()
        if "STRONG BUY" in text_upper or "STRONG_BUY" in text_upper:
            signal = "STRONG_BUY"
            confidence = 0.85
        elif "BUY" in text_upper:
            signal = "BUY"
            confidence = 0.7
        elif "STRONG SELL" in text_upper or "STRONG_SELL" in text_upper:
            signal = "STRONG_SELL"
            confidence = 0.85
        elif "SELL" in text_upper:
            signal = "SELL"
            confidence = 0.7
        else:
            signal = "NEUTRAL"
            confidence = 0.5
        for line in text.split("\n"):
            if "confidence" in line.lower():
                try:
                    nums = [float(s) for s in line.split() if s.replace(".", "").isdigit()]
                    if nums:
                        confidence = max(0.1, min(1.0, nums[0]))
                except:
                    pass
        return {"signal": signal, "confidence": confidence, "reasoning": text[:200]}


# ══════════════════════════════════════════════════════════════════
#  AI ENSEMBLE — Оркестратор мульти-AI
# ══════════════════════════════════════════════════════════════════

class AIEnsemble:
    """Управляет голосованием всех AI моделей, адаптивными весами, памятью и CSV"""

    def __init__(self, ai_analyzer: AIAnalyzer, market_data: MarketDataProvider, market_api=None):
        self.ai = ai_analyzer
        self.market = market_data
        self.market_api = market_api  # Для получения multi-timeframe данных
        self.weights = {
            "lstm": 1.0,
            "gemini": 1.0,
            "groq": 1.0,
            "openrouter": 1.0,
            "cohere": 1.0,
            "deepseek": 1.0,
        }
        self.memory = deque(maxlen=CONFIG.get("ai_memory_size", 20))
        self.last_llm_results = {}
        self.csv_file = "ai_decisions.csv"
        self._init_csv()
        self.providers = {
            "gemini": GeminiProvider(),
            "groq": GroqProvider(),
            "openrouter": OpenRouterProvider(),
            "cohere": CohereProvider(),
            "deepseek": DeepSeekProvider(),
        }
        self.last_analysis_time = 0
        self.total_correct = 0
        self.total_decisions = 0

    def _init_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "price", "btc_price", "fear_greed", "rsi", "macd",
                    "lstm_signal", "gemini_signal", "groq_signal", "openrouter_signal", "cohere_signal", "deepseek_signal",
                    "ensemble_signal", "confidence", "result_pnl", "correct"
                ])

    def _build_prompt(self, price: float, indicators: dict, position_size: float,
                      realized_pnl: float, grid_lower: float, grid_upper: float,
                      provider: str = None, multi_tf_data: dict = None) -> str:
        """Создаёт улучшенный промт для AI с multi-timeframe анализом"""
        ctx = self.market.get_market_context(price)

        # Получаем стратегию для провайдера
        strategies = CONFIG.get("ai_strategies", {})
        if provider and provider in strategies:
            strategy = strategies[provider]
            role = strategy.get("role", CONFIG.get("ai_role"))
            style = strategy.get("style", CONFIG.get("ai_style"))
            risk = strategy.get("risk", CONFIG.get("ai_risk"))
            instruction = strategy.get("instruction", "")
            min_conf = strategy.get("min_confidence", 0.6)
            agent_type = strategy.get("agent_type", "analyst")
            timeframe = strategy.get("timeframe", "1H")
        else:
            role = CONFIG.get("ai_role", "профессиональный крипто-трейдер")
            style = CONFIG.get("ai_style", "свинг-трейдинг")
            risk = CONFIG.get("ai_risk", "умеренный")
            instruction = ""
            min_conf = 0.6
            agent_type = "analyst"
            timeframe = "1H"

        # Формируем текст памяти (последние 5 решений)
        memory_text = ""
        if self.memory:
            recent = list(self.memory)[-5:]
            lines = []
            for i, m in enumerate(recent, 1):
                emoji = "✅" if m.get("correct", True) else "❌"
                pnl_str = f"{m.get('pnl', 0):+.2f}" if m.get('pnl') is not None else "—"
                lines.append(f"{i}. {m['signal']} @ {m.get('price', 0):.2f} → {pnl_str} USDT {emoji}")
            memory_text = "\n\n📝 ПОСЛЕДНИЕ РЕШЕНИЯ:\n" + "\n".join(lines)

        # Определяем тренд
        trend = "🟢 БЫЧИЙ" if indicators.get("ma20", 0) > indicators.get("ma50", 0) else "🔴 МЕДВЕЖИЙ"
        price_change_24h = 0
        if self.ai.price_history:
            prices = list(self.ai.price_history)
            if len(prices) > 10:
                price_change_24h = ((prices[-1] - prices[-10]) / prices[-10]) * 100

        # Multi-timeframe данные
        multi_tf_text = ""
        if multi_tf_data and any(v is not None for v in multi_tf_data.values()):
            multi_tf_lines = ["\n📊 MULTI-TIMEFRAME АНАЛИЗ:"]
            for tf_name in ["M5", "M15", "H1", "H4"]:
                tf_data = multi_tf_data.get(tf_name)
                if tf_data:
                    trend_icon = "🟢" if tf_data['trend'] == "BULLISH" else "🔴"
                    multi_tf_lines.append(
                        f"{tf_name}: {tf_data['current_price']:.2f} "
                        f"({tf_data['change_pct']:+.2f}%) "
                        f"{trend_icon} {tf_data['trend']}"
                    )
                else:
                    multi_tf_lines.append(f"{tf_name}: нет данных")
            multi_tf_text = "\n".join(multi_tf_lines)
        else:
            multi_tf_text = "\n📊 MULTI-TIMEFRAME АНАЛИЗ: данные недоступ"

        # Формируем OHLCV блок для 1H таймфрейма
        ohlcv_text = ""
        if multi_tf_data and multi_tf_data.get("H1"):
            h1_data = multi_tf_data["H1"]
            ohlcv_text = f"""
📈 ПОСЛЕДНИЕ СВЕЧИ (H1):
Текущая: O={h1_data['current_price']:.2f} H={h1_data['high']:.2f} L={h1_data['low']:.2f}
Объём: {h1_data['volume']:.0f} SOL"""

        prompt = f"""<SYSTEM>
{role}
Стиль торговли: {style}
Риск-профиль: {risk}
Тип агента: {agent_type}
Рабочий таймфрейм: {timeframe}
</SYSTEM>

<РЫНОЧНЫЕ ДАННЫЕ>
💰 SOL/USDT ТЕКУЩАЯ ЦЕНА: ${price:.2f}
📉 Изменение за период: {price_change_24h:+.1f}%

₿ BTC ДАННЫЕ:
• Цена: ${ctx['btc_price']:,.0f}
• Доминация: {ctx['btc_dominance']}%

😨 ИНДЕКС СТРАХА И ЖАДНОСТИ:
• Значение: {ctx['fear_greed_value']}/100
• Статус: {ctx['fear_greed_label']}

🔗 КОРРЕЛЯЦИЯ SOL/BTC: {ctx['sol_btc_correlation']}
{multi_tf_text}
{ohlcv_text}
</РЫНОЧНЫЕ ДАННЫЕ>

<ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ>
📊 RSI(14): {indicators.get('rsi', '—')}
📊 MACD Histogram: {indicators.get('macd_hist', '—')}
📊 Bollinger Bands: [{indicators.get('bb_lower', '—')} — {indicators.get('bb_upper', '—')}]
📊 MA(20): {indicators.get('ma20', '—')}
📊 MA(50): {indicators.get('ma50', '—')}
📊 Текущий тренд: {trend}
</ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ>

<КОНТЕКСТ ТОРГОВЛИ>
🕐 Торговая сессия: {ctx['trading_session']}
📅 День недели: {ctx['day_of_week']}

💼 СОСТОЯНИЕ БОТА:
• Позиция: {position_size:.2f} SOL
• Реализованный PnL: {realized_pnl:+.2f} USDT
• Диапазон сетки: {grid_lower:.2f} — {grid_upper:.2f}{memory_text}
</КОНТЕКСТ ТОРГОВЛИ>

<ИНСТРУКЦИЯ АГЕНТА>
{instruction}
</ИНСТРУКЦИЯ АГЕНТА>

<ФОРМАТ ОТВЕТА>
Ты должен вернуть сигнал и уверенность в следующем формате:

Signal: <один из: STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL>
Confidence: <число от 0.0 до 1.0>
Reasoning: <краткое обоснование решения в 1-3 предложениях>

Минимальная уверенность для сигнала: {min_conf}
Если уверенность ниже минимальной — верни NEUTRAL.
</ФОРМАТ ОТВЕТА>"""

        return prompt

    def analyze(self, price: float, indicators: dict, position_size: float,
                realized_pnl: float, grid_lower: float, grid_upper: float) -> dict:
        now = time.time()
        interval = CONFIG.get("ai_analysis_interval", 900)
        if now - self.last_analysis_time < interval:
            return self._get_last_ensemble()

        self.last_analysis_time = now

        # Получаем multi-timeframe данные один раз для всех провайдеров
        multi_tf_data = {}
        if self.market_api and CONFIG.get("ai_multi_timeframe", False):
            try:
                multi_tf_data = self.market.get_multi_timeframe_data(self.market_api)
                log.info(f"📊 Multi-timeframe данные получены: {sum(1 for v in multi_tf_data.values() if v is not None)}/4 таймфреймов")
            except Exception as e:
                log.warning(f"⚠️ Ошибка получения multi-timeframe данных: {e}")

        # AI Fallback - пробуем провайдеры по очереди
        result = self._analyze_with_fallback(
            price, indicators, position_size, realized_pnl,
            grid_lower, grid_upper, multi_tf_data
        )

        if result:
            self.last_llm_results = {"fallback": result}
            ensemble = self._vote({"fallback": result})
            self._save_to_csv(price, indicators, {"fallback": result}, ensemble)
            return ensemble

        # Все провайдеры недоступны - используем кэш
        return self._get_last_ensemble()

    def _analyze_with_fallback(self, price: float, indicators: dict, position_size: float,
                                realized_pnl: float, grid_lower: float, grid_upper: float,
                                multi_tf_data: dict = None) -> dict:
        """AI с fallback - пробуем провайдеры по очереди с multi-timeframe данными"""
        providers_order = ["gemini", "groq", "openrouter", "cohere", "deepseek"]

        for provider_name in providers_order:
            provider = self.providers.get(provider_name)
            if not provider:
                continue

            try:
                # Проверяем доступность провайдера
                if not self._check_provider_available(provider_name):
                    log.warning(f"⚠️ {provider_name} недоступен, пробую следующий")
                    continue

                # Создаём промт с multi-timeframe данными
                prompt = self._build_prompt(
                    price, indicators, position_size, realized_pnl,
                    grid_lower, grid_upper,
                    provider=provider_name,
                    multi_tf_data=multi_tf_data
                )

                # Вызываем AI
                result = provider.get_signal(prompt)

                # Успех - логируем и возвращаем
                log.info(f"🧠 {provider_name}: {result.get('signal', 'N/A')} (conf={result.get('confidence', 0):.2f})")
                ai_logger.info(
                    f"ENSEMBLE │ Provider: {provider_name.upper()} │ Signal: {result.get('signal', 'N/A')} │ "
                    f"Conf: {result.get('confidence', 0):.2f} │ Reasoning: {result.get('reasoning', 'N/A')[:100]}"
                )
                return result

            except Exception as e:
                log.warning(f"⚠️ {provider_name} ошибка: {e} → пробую следующий")
                continue

        # Все провайдеры упали
        log.error("🔴 Все AI провайдеры недоступны!")
        return None

    def _check_provider_available(self, provider_name: str) -> bool:
        """Проверка доступности провайдера (быстрый ping)"""
        try:
            if provider_name == "gemini":
                # Gemini - простой тест через API key
                return bool(GEMINI_API_KEY)
            elif provider_name == "groq":
                return bool(GROQ_API_KEY)
            elif provider_name == "openrouter":
                return bool(OPENROUTER_API_KEY)
            elif provider_name == "cohere":
                return bool(COHERE_API_KEY)
            elif provider_name == "deepseek":
                return bool(DEEPSEEK_API_KEY)
            return False
        except:
            return False

    def _vote(self, results: dict) -> dict:
        score_map = {"STRONG_BUY": 2, "BUY": 1, "NEUTRAL": 0, "SELL": -1, "STRONG_SELL": -2}
        weighted_score = 0
        total_weight = 0
        signals_count = {}

        for name, res in results.items():
            w = self.weights.get(name, 1.0)
            s = score_map.get(res["signal"], 0)
            weighted_score += s * w * res["confidence"]
            total_weight += w
            signals_count[name] = res["signal"]

        if total_weight > 0:
            normalized = weighted_score / total_weight
        else:
            normalized = 0

        if normalized >= 1.2: signal = "STRONG_BUY"
        elif normalized >= 0.5: signal = "BUY"
        elif normalized <= -1.2: signal = "STRONG_SELL"
        elif normalized <= -0.5: signal = "SELL"
        else: signal = "NEUTRAL"

        confidence = min(1.0, abs(normalized) / 2 + 0.5)

        return {
            "signal": signal,
            "score": round(normalized, 3),
            "confidence": round(confidence, 3),
            "providers": signals_count,
            "weights": dict(self.weights),
        }

    def _get_last_ensemble(self) -> dict:
        if self.last_llm_results:
            return self._vote(self.last_llm_results)
        return self.ai.get_signal()

    def record_outcome(self, pnl: float, correct: bool):
        if self.memory:
            last = self.memory[-1]
            last["pnl"] = pnl
            last["correct"] = correct
        self.total_decisions += 1
        if correct:
            self.total_correct += 1
        if CONFIG.get("ai_adaptive_weights", True):
            for name in self.weights:
                if name in self.last_llm_results:
                    res = self.last_llm_results[name]
                    res_correct = (res["signal"] in ("BUY", "STRONG_BUY") and pnl >= 0) or \
                                  (res["signal"] in ("SELL", "STRONG_SELL") and pnl < 0) or \
                                  (res["signal"] == "NEUTRAL")
                    if res_correct:
                        self.weights[name] *= 1.1
                    else:
                        self.weights[name] *= 0.9
            total = sum(self.weights.values())
            if total > 0:
                self.weights = {k: round(v / total, 4) for k, v in self.weights.items()}

    def _save_to_csv(self, price: float, indicators: dict, results: dict, ensemble: dict):
        # Rotate CSV if too large (keep last 5000 rows)
        max_csv_rows = 5000
        try:
            if os.path.exists(self.csv_file):
                with open(self.csv_file, "r", encoding="utf-8") as f:
                    row_count = sum(1 for _ in f) - 1  # Exclude header
                if row_count >= max_csv_rows:
                    # Keep only last 4000 rows
                    with open(self.csv_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    header = lines[0]
                    recent_rows = lines[-4000:]
                    with open(self.csv_file, "w", encoding="utf-8") as f:
                        f.writelines([header] + recent_rows)
                    log.info(f"🔄 CSV ротация: удалены старые строки, оставлено 4000")
        except Exception as e:
            log.warning(f"⚠️ Ошибка ротации CSV: {e}")

        ctx = self.market.get_market_context(price)
        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                price,
                ctx.get("btc_price", 0),
                ctx.get("fear_greed_value", 50),
                indicators.get("rsi", ""),
                indicators.get("macd_hist", ""),
                results.get("lstm", {}).get("signal", ""),
                results.get("gemini", {}).get("signal", ""),
                results.get("groq", {}).get("signal", ""),
                results.get("cohere", {}).get("signal", ""),
                results.get("deepseek", {}).get("signal", ""),
                ensemble["signal"],
                ensemble["confidence"],
                "",
                ""
            ])
        self.memory.append({
            "timestamp": datetime.now().isoformat(),
            "price": price,
            "signal": ensemble["signal"],
            "confidence": ensemble["confidence"],
            "pnl": None,
            "correct": None,
        })

    def get_accuracy_report(self, days: int = 7) -> str:
        if not os.path.exists(self.csv_file):
            return "📊 Нет данных для анализа"
        try:
            df = pd.read_csv(self.csv_file)
            if df.empty or "correct" not in df.columns:
                return "📊 Нет завершённых решений"
            cutoff = datetime.now().timestamp() - days * 86400
            recent = df[df["timestamp"] > datetime.fromtimestamp(cutoff).isoformat()]
            if recent.empty:
                return f"📊 Нет данных за последние {days} дней"
            lines = [f"📊 *Backtest за {days} дней*\n"]
            total = len(recent)
            correct = recent["correct"].sum() if "correct" in recent.columns else 0
            lines.append(f"Всего решений: `{total}`")
            lines.append(f"Правильных: `{int(correct)}` ({correct/total*100:.0f}%)")
            lines.append("")
            for name in ["lstm", "gemini", "groq", "openrouter", "cohere", "deepseek"]:
                col = f"{name}_signal"
                if col in recent.columns:
                    lines.append(f"{name}: {recent[col].value_counts().to_dict()}")
            return "\n".join(lines)
        except Exception as e:
            return f"📊 Ошибка backtest: {e}"

    def get_weights_text(self) -> str:
        lines = ["⚖️ *Адаптивные веса:*"]
        for name, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 20)
            lines.append(f"{name}: `{w:.3f}` {bar}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  BACKTESTER
# ══════════════════════════════════════════════════════════════════

class Backtester:
    """Анализ точности AI решений"""

    def __init__(self, ensemble: AIEnsemble):
        self.ensemble = ensemble
        self.last_auto_run = 0

    def run_analysis(self, days: int = 7) -> str:
        csv_file = self.ensemble.csv_file
        if not os.path.exists(csv_file):
            return "📊 Нет данных для backtest"
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                return "📊 CSV пуст"
            cutoff = datetime.now().timestamp() - days * 86400
            cutoff_str = datetime.fromtimestamp(cutoff).isoformat()
            recent = df[df["timestamp"] > cutoff_str]
            if recent.empty:
                return f"📊 Нет данных за последние {days} дней"
            lines = [f"📊 *Backtest за {days} дней*\n"]
            total = len(recent)
            if "correct" in recent.columns:
                valid = recent.dropna(subset=["correct"])
                if len(valid) > 0:
                    correct = valid["correct"].sum()
                    accuracy = correct / len(valid) * 100
                    lines.append(f"Решений с результатом: `{len(valid)}`")
                    lines.append(f"Точность: `{accuracy:.0f}%` ({int(correct)}/{len(valid)})")
                    if "result_pnl" in valid.columns:
                        total_pnl = valid["result_pnl"].sum()
                        lines.append(f"Общий PnL: `{total_pnl:+.4f} USDT`")
            lines.append("")
            lines.append("📈 *Сигналы по провайдерам:*")
            for name in ["lstm", "gemini", "groq", "openrouter", "cohere", "deepseek"]:
                col = f"{name}_signal"
                if col in recent.columns:
                    counts = recent[col].value_counts()
                    if len(counts) > 0:
                        parts = [f"{k}={v}" for k, v in counts.items()]
                        lines.append(f"{name}: {', '.join(parts)}")
            lines.append("")
            lines.append(self.ensemble.get_weights_text())
            return "\n".join(lines)
        except Exception as e:
            return f"📊 Ошибка backtest: {e}"

    def check_auto_run(self) -> str | None:
        if not CONFIG.get("ai_backtest_auto", True):
            return None
        now = time.time()
        if now - self.last_auto_run < 86400:
            return None
        self.last_auto_run = now
        return self.run_analysis(7)


# ══════════════════════════════════════════════════════════════════
#  GRID BOT V3 — LONG ONLY + MULTI-AI (OKX TESTNET)
# ══════════════════════════════════════════════════════════════════

class GridBotV3:
    def __init__(self):
        self.trade_api = Trade.TradeAPI(
            api_key=OKX_API_KEY,
            api_secret_key=OKX_API_SECRET,
            passphrase=OKX_PASSPHRASE,
            flag=OKX_FLAG,
            debug=False
        )
        self.account_api = Account.AccountAPI(
            api_key=OKX_API_KEY,
            api_secret_key=OKX_API_SECRET,
            passphrase=OKX_PASSPHRASE,
            flag=OKX_FLAG,
            debug=False
        )
        self.market_api = MarketData.MarketAPI(
            api_key=OKX_API_KEY,
            api_secret_key=OKX_API_SECRET,
            passphrase=OKX_PASSPHRASE,
            flag=OKX_FLAG,
            debug=False
        )
        self.lock = threading.Lock()
        self.running = False
        self.start_time = None
        self.start_balance = 0.0
        self.last_price = 0.0

        self.lower = 0.0
        self.upper = 0.0
        self.grid_levels = []
        self.active_orders = {}
        self._cancelled_ids = set()
        self._synced_positions = {}  # {f"{posSide}_{avg_px:.2f}": True} — уже обработанные позиции
        self._position_wait_counts = {}  # {f"{posSide}_{avg_px:.2f}": count} — счетчик циклов ожидания

        self.realized_pnl = 0.0
        self.trades_count = 0
        self.avg_buy_price = 0.0
        self.total_sol_bought = 0.0

        self._thread = None
        self._tg_notify = None
        self._tg_app = None

        self.ai = AIAnalyzer()
        self.market = MarketDataProvider()
        self.ensemble = AIEnsemble(self.ai, self.market, self.market_api)
        self.backtester = Backtester(self.ensemble)
        self.last_signal = {"signal": "NEUTRAL", "score": 0, "confidence": 0.5, "providers": {}, "indicators": {}}
        self.pending_sells = []  # [(price, qty), ...] — ждут появления SOL
        self.positions_without_tp = []  # Позиции открытые но без TP ордера

    def set_tg_notify(self, fn):
        self._tg_notify = fn

    def set_tg_app(self, app):
        self._tg_app = app

    def notify(self, msg: str):
        log.info(msg)
        if self._tg_notify:
            self._tg_notify(msg)

    # ── Биржа ─────────────────────────────────────────────────────

    def _check_okx_response(self, response: dict) -> dict:
        if response.get("code") != "0":
            raise Exception(f"OKX API error {response.get('code')}: {response.get('msg', 'Unknown')}")
        return response

    def _get_currency_detail(self, details: list, ccy: str) -> dict | None:
        for d in details:
            if d.get("ccy") == ccy:
                return d
        return None

    def setup_account(self):
        if "-SWAP" in CONFIG["symbol"]:
            log.info("⚡ Фьючерсный режим (с плечом)")
            # Пробуем переключить в long/short mode (нужно чтобы posSide работал)
            # Это НЕ сработает если есть открытые позиции — тогда работаем в net mode
            self.use_pos_side = False
            try:
                r = self.account_api.set_position_mode("long_short_mode")
                if r.get("code") == "0":
                    log.info("✅ Позиционный режим: long/short mode (posSide включён)")
                    self.use_pos_side = True
                else:
                    msg = r.get("msg", r.get("code"))
                    log.warning(f"⚠️ set_position_mode: {msg} — определяю режим по позициям")
                    # Проверяем, в каком режиме уже находятся позиции
                    try:
                        pr = self.account_api.get_positions(instType="SWAP", instId=CONFIG["symbol"])
                        if pr.get("code") == "0":
                            positions = pr.get("data", [])
                            for pos in positions:
                                ps = pos.get("posSide", "net")
                                if ps in ("long", "short"):
                                    self.use_pos_side = True
                                    log.info(f"✅ Обнаружен long/short mode по позиции {ps}")
                                    break
                    except Exception:
                        pass
                    if not self.use_pos_side:
                        log.warning("⚠️ Работаю в net mode (без posSide)")
            except Exception as e:
                log.warning(f"⚠️ Не удалось установить long/short mode: {e} — определяю режим по позициям")
                try:
                    pr = self.account_api.get_positions(instType="SWAP", instId=CONFIG["symbol"])
                    if pr.get("code") == "0":
                        positions = pr.get("data", [])
                        for pos in positions:
                            ps = pos.get("posSide", "net")
                            if ps in ("long", "short"):
                                self.use_pos_side = True
                                log.info(f"✅ Обнаружен long/short mode по позиции {ps}")
                                break
                except Exception:
                    pass
                if not self.use_pos_side:
                    log.warning("⚠️ Работаю в net mode (без posSide)")
        else:
            log.info("✅ Спотовый режим (без плеча)")
            self.use_pos_side = False

    @retry_api()
    def get_price(self) -> float:
        r = self.market_api.get_ticker(instId=CONFIG["symbol"])
        self._check_okx_response(r)
        return float(r["data"][0]["last"])

    @retry_api()
    def get_balance(self) -> float:
        try:
            r = self.account_api.get_account_balance()
            self._check_okx_response(r)
            details = r["data"][0].get("details", [])
            usdt = self._get_currency_detail(details, "USDT")
            if usdt:
                # For futures wallet, total balance is in eq field
                return float(usdt.get("eq", "0"))
            return 0.0
        except Exception as e:
            log.error(f"Баланс ошибка: {e}")
            return 0.0

    @retry_api()
    def get_available_balance(self) -> float:
        try:
            r = self.account_api.get_account_balance()
            self._check_okx_response(r)
            details = r["data"][0].get("details", [])
            usdt = self._get_currency_detail(details, "USDT")
            if usdt:
                # For futures trading, available balance is in availBal
                avail_bal = usdt.get("availBal", "0")
                return float(avail_bal) if avail_bal else 0.0
            return 0.0
        except Exception as e:
            log.error(f"Доступный баланс ошибка: {e}")
            return 0.0

    @retry_api()
    def get_sol_balance(self) -> float:
        """Для спота возвращаем баланс SOL"""
        try:
            r = self.account_api.get_account_balance()
            self._check_okx_response(r)
            details = r["data"][0].get("details", [])
            sol = self._get_currency_detail(details, "SOL")
            if sol:
                return float(sol.get("bal", "0"))
            return 0.0
        except Exception as e:
            log.error(f"Баланс SOL ошибка: {e}")
            return 0.0

    def get_unrealized_pnl(self) -> float:
        """Unrealized PnL: для фьючерсов берём upl из позиций, для спота считаем вручную"""
        if "-SWAP" in CONFIG["symbol"]:
            try:
                r = self.account_api.get_positions(instType="SWAP", instId=CONFIG["symbol"])
                if r.get("code") == "0":
                    total_upl = 0.0
                    for pos in r.get("data", []):
                        total_upl += float(pos.get("upl", 0))
                    return total_upl
            except Exception:
                pass
            return 0.0
        else:
            try:
                sol_bal = self.get_sol_balance()
                if sol_bal > 0 and self.last_price > 0 and self.avg_buy_price > 0:
                    return (self.last_price - self.avg_buy_price) * sol_bal
            except Exception:
                pass
            return 0.0

    def get_spot_holdings(self) -> float:
        """Для спота возвращаем баланс SOL"""
        return self.get_sol_balance()

    @retry_api()
    def get_open_order_count(self) -> int:
        try:
            r = self.trade_api.get_order_list(instType="SWAP", instId=CONFIG["symbol"], state="live")
            self._check_okx_response(r)
            return len(r["data"])
        except Exception:
            return MAX_ORDERS

    @retry_api()
    def get_klines(self, interval="4H", limit=50) -> pd.DataFrame:
        bar_map = {"240": "4H", "15": "15m", "60": "1H", "4H": "4H", "15m": "15m"}
        bar = bar_map.get(interval, interval)
        r = self.market_api.get_candlesticks(instId=CONFIG["symbol"], bar=bar, limit=str(limit))
        self._check_okx_response(r)
        data = r["data"]
        data.reverse()
        df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
        for col in ["open", "high", "low", "close", "vol"]:
            df[col] = df[col].astype(float)
        return df

    # ── Диапазон ──────────────────────────────────────────────────

    def calc_atr_range(self, price: float) -> tuple[float, float]:
        try:
            df = self.get_klines(interval="4H", limit=50)
            high, low, close = df["high"], df["low"], df["close"]
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            mult = CONFIG.get("atr_multiplier", 2.5)
            half_width = atr * mult
            max_half = price * (CONFIG.get("max_range_pct", 0.20) / 2)
            half = min(half_width, max_half)
            lower = round_price(max(price * 0.4, price - half))
            upper = round_price(price + half)
            log.info(f"📐 ATR={atr:.2f} -> Range: {lower} - {upper}")
            return lower, upper
        except Exception as e:
            log.error(f"ATR ошибка: {e}")
            return round_price(price * 0.95), round_price(price * 1.05)

    def build_grid(self, lower: float, upper: float) -> list[float]:
        n = CONFIG["grid_levels"]
        step = (upper - lower) / n
        return [round_price(lower + i * step) for i in range(n + 1)]

    # ── Ордера ────────────────────────────────────────────────────

    def _qty_for_price(self, price: float) -> float:
        avail_eq = self.get_available_balance()  # Available equity for futures
        # Apply leverage: x5 means we can use 5x the available equity for margin
        leveraged_avail = avail_eq * CONFIG.get("leverage", 5.0)
        n = CONFIG["grid_levels"]
        budget = (leveraged_avail * CONFIG["max_balance_pct"]) / n
        qty = budget / price  # Contract size for SOL-USDT-SWAP (1 contract = 1 SOL)
        qty = min(qty, CONFIG["max_qty"])
        qty = max(qty, MIN_QTY)
        return round_qty(qty)

    def place_buy(self, price: float, qty: float = None, buy_price: float = None, pos_side: str = None) -> str | None:
        if self.get_open_order_count() >= MAX_ORDERS:
            return None
        if qty is None:
            qty = self._qty_for_price(price)

        sig = self.last_signal.get("signal", "NEUTRAL")
        if sig == "STRONG_SELL":
            log.info(f"🧠 AI STRONG_SELL — пропуск BUY@{price}")
            return None
        if sig == "SELL":
            qty = round_qty(qty * 0.5)
        if qty * price < 5.0:
            return None
        try:
            params = {
                "instId": CONFIG["symbol"],
                "tdMode": "isolated",
                "side": "buy",
                "ordType": "limit",
                "sz": str(qty),
                "px": str(price)
            }
            if pos_side and "-SWAP" in CONFIG["symbol"] and getattr(self, 'use_pos_side', False):
                params["posSide"] = pos_side
            r = self.trade_api.place_order(**params)
            log.info(f"📤 BUY response: {r}")
            self._check_okx_response(r)
            oid = r["data"][0]["ordId"]
            # buy_price = цена входа в позицию (для расчёта PnL)
            self.active_orders[oid] = {"price": price, "type": "BUY", "qty": qty, "buy_price": buy_price, "pos_side": pos_side}
            if pos_side != "short":
                self.total_sol_bought += qty
                if self.total_sol_bought > 0:
                    self.avg_buy_price = ((self.avg_buy_price * (self.total_sol_bought - qty)) + (price * qty)) / self.total_sol_bought
            label = f"BUY (лонг)" if pos_side == "long" else (f"BUY (закрытие шорта)" if pos_side == "short" else "BUY")
            log.info(f"🟢 {label} @ {price} qty={qty} (x{CONFIG.get('leverage', 5.0)} leverage)")
            return oid
        except Exception as e:
            log.error(f"BUY ошибка: {e}")
            return None

    def place_sell(self, price: float, qty: float, buy_price: float = None, pos_side: str = None) -> str | None:
        if self.get_open_order_count() >= MAX_ORDERS:
            return None
        qty = round_qty(qty)
        if qty * price < 5.0:
            log.warning(f"⚠️ Пропуск SELL: qty={qty} * price={price} = {qty*price} < 5 USDT")
            return None
        try:
            params = {
                "instId": CONFIG["symbol"],
                "tdMode": "isolated",
                "side": "sell",
                "ordType": "limit",
                "sz": str(qty),
                "px": str(price)
            }
            if pos_side and "-SWAP" in CONFIG["symbol"] and getattr(self, 'use_pos_side', False):
                params["posSide"] = pos_side
            r = self.trade_api.place_order(**params)
            self._check_okx_response(r)
            oid = r["data"][0]["ordId"]
            self.active_orders[oid] = {"price": price, "type": "SELL", "qty": qty, "buy_price": buy_price, "pos_side": pos_side}
            label = f"SELL (тейк лонг)" if pos_side == "long" else (f"SELL (вход в шорт)" if pos_side == "short" else "SELL")
            log.info(f"🔴 {label} @ {price} qty={qty} (x{CONFIG.get('leverage', 5.0)} leverage)")
            return oid
        except Exception as e:
            if "Insufficient margin" in str(e) or "margin" in str(e).lower():
                log.warning(f"⚠️ SELL @ {price}: недостаточно маржи (qty={qty})")
            else:
                log.error(f"SELL ошибка: {e}")
            return None

    def close_all_positions(self):
        """Закрытие всех позиций на фьючерсах через close_positions API"""
        try:
            if "-SWAP" in CONFIG["symbol"]:
                if getattr(self, 'use_pos_side', False):
                    # Long/short mode — закрываем по сторонам
                    try:
                        r = self.trade_api.close_positions(instId=CONFIG["symbol"], mgnMode="isolated", posSide="long")
                        if r.get("code") == "0":
                            log.info("🔒 Лонг позиции закрыты")
                        else:
                            log.info(f"Закрытие лонгов: {r.get('msg', r.get('code'))}")
                    except Exception as e:
                        if "No positions" not in str(e) and "position" not in str(e).lower():
                            log.error(f"Закрытие лонгов ошибка: {e}")
                    try:
                        r = self.trade_api.close_positions(instId=CONFIG["symbol"], mgnMode="isolated", posSide="short")
                        if r.get("code") == "0":
                            log.info("🔒 Шорт позиции закрыты")
                        else:
                            log.info(f"Закрытие шортов: {r.get('msg', r.get('code'))}")
                    except Exception as e:
                        if "No positions" not in str(e) and "position" not in str(e).lower():
                            log.error(f"Закрытие шортов ошибка: {e}")
                else:
                    # Net mode — закрываем без posSide
                    try:
                        r = self.trade_api.close_positions(instId=CONFIG["symbol"], mgnMode="isolated")
                        if r.get("code") == "0":
                            log.info("🔒 Все позиции закрыты (net mode)")
                        else:
                            log.info(f"Закрытие позиций: {r.get('msg', r.get('code'))}")
                    except Exception as e:
                        if "No positions" not in str(e) and "position" not in str(e).lower():
                            log.error(f"Закрытие позиций ошибка: {e}")
            else:
                # Спот режим
                sol_holdings = self.get_spot_holdings()
                if sol_holdings <= 0:
                    return
                r = self.trade_api.place_order(
                    instId=CONFIG["symbol"],
                    tdMode="cash",
                    side="sell",
                    ordType="market",
                    sz=str(round_qty(sol_holdings)),
                )
                self._check_okx_response(r)
                log.info(f"🔒 Продано {sol_holdings} SOL по рынку")
        except Exception as e:
            log.error(f"Закрытие позиций ошибка: {e}")

    def cancel_all(self):
        try:
            r = self.trade_api.cancel_all_orders(instType="SWAP", instId=CONFIG["symbol"])
            self._check_okx_response(r)
        except Exception:
            pass
        self._cancelled_ids.update(self.active_orders.keys())
        self.active_orders.clear()
        log.info("🗑 Ордера отменены")

    # ── Сетка ─────────────────────────────────────────────────────

    def place_grid(self, price: float):
        self.cancel_all()
        time.sleep(0.5)

        # Лонг: BUY ордера ниже цены (вход в лонг)
        buy_levels = [p for p in self.grid_levels if p < price * 0.9995]

        placed = 0
        for lvl in sorted(buy_levels, reverse=True):
            if self.get_open_order_count() >= MAX_ORDERS:
                break
            qty = self._qty_for_price(lvl)
            result = self.place_buy(lvl, qty=qty, buy_price=lvl, pos_side="long")
            if result:
                placed += 1
            time.sleep(0.05)

        # Шорт: SELL ордера ниже цены (вход в шорт)
        sell_levels = [p for p in self.grid_levels if p < price * 0.995]

        for lvl in sorted(sell_levels, reverse=True):
            if self.get_open_order_count() >= MAX_ORDERS:
                break
            qty = self._qty_for_price(lvl)
            result = self.place_sell(lvl, qty, buy_price=None, pos_side="short")
            if result:
                placed += 1
            time.sleep(0.05)
        self.notify(f"📐 Сетка: {placed} ордера\nДиапазон: {self.lower:.2f} — {self.upper:.2f}\nБаланс: {self.get_balance():.2f} USDT")

    def _rebuild_grid_around_price(self, price: float):
        self.lower, self.upper = self.calc_atr_range(price)
        self.grid_levels = self.build_grid(self.lower, self.upper)
        to_cancel = [oid for oid, o in self.active_orders.items()
                     if o["type"] in ("BUY", "SELL") and (o["price"] < self.lower or o["price"] > self.upper)]
        for oid in to_cancel:
            try:
                self.trade_api.cancel_order(instId=CONFIG["symbol"], ordId=oid)
                self.active_orders.pop(oid, None)
            except Exception:
                pass
        existing_buy = {o["price"] for o in self.active_orders.values() if o["type"] == "BUY" and o.get("pos_side") == "long"}
        existing_sell = {o["price"] for o in self.active_orders.values() if o["type"] == "SELL" and o.get("pos_side") == "short"}
        buy_levels = [p for p in self.grid_levels if p < price * 0.9995 and p not in existing_buy]
        sell_levels = [p for p in self.grid_levels if p < price * 0.995 and p not in existing_sell]
        placed = 0
        for lvl in sorted(buy_levels, reverse=True):
            if self.get_open_order_count() >= MAX_ORDERS:
                break
            qty = self._qty_for_price(lvl)
            if self.place_buy(lvl, qty=qty, buy_price=lvl, pos_side="long"):
                placed += 1
            time.sleep(0.05)
        for lvl in sorted(sell_levels, reverse=True):
            if self.get_open_order_count() >= MAX_ORDERS:
                break
            qty = self._qty_for_price(lvl)
            if self.place_sell(lvl, qty, buy_price=None, pos_side="short"):
                placed += 1
            time.sleep(0.05)
        return placed

    def trailing_up(self, price: float):
        placed = self._rebuild_grid_around_price(price)
        self.notify(f"📈 Trailing UP: {placed} новых ордеров\nДиапазон: {self.lower:.2f} — {self.upper:.2f}")

    def trailing_down(self, price: float):
        placed = self._rebuild_grid_around_price(price)
        self.notify(f"📉 Trailing DOWN: {placed} новых ордеров\nДиапазон: {self.lower:.2f} — {self.upper:.2f}")

    # ── Исполненные ───────────────────────────────────────────────

    def check_filled(self):
        try:
            r = self.trade_api.get_order_list(instType="SWAP", instId=CONFIG["symbol"], state="live")
            self._check_okx_response(r)
            open_ids = {o["ordId"] for o in r["data"]}
        except Exception as e:
            log.error(f"check_filled ошибка: {e}")
            return
        step = (self.upper - self.lower) / CONFIG["grid_levels"]
        for oid in list(self.active_orders):
            if oid in open_ids:
                continue
            order = self.active_orders.pop(oid)
            if oid in self._cancelled_ids:
                self._cancelled_ids.discard(oid)
                continue
            price, otype, qty = order["price"], order["type"], order["qty"]
            buy_price_entry = order.get("buy_price")
            pos_side = order.get("pos_side")

            if otype == "BUY":
                if "-SWAP" in CONFIG["symbol"]:
                    if pos_side == "short":
                        # Это BUY для закрытия шорта - считаем PnL
                        short_entry_price = buy_price_entry
                        if short_entry_price is not None:
                            pnl = round((short_entry_price - price) * qty, 4)
                            self.realized_pnl += pnl
                            self.trades_count += 1
                            status.add_trade(pnl)
                            log.info(f"🔒 Шорт закрыт @ {price:.2f} | PnL: {pnl:+.4f} | Итого: {self.realized_pnl:+.4f}")
                            self.notify(f"🔒 Шорт: BUY @ {price:.2f} | PnL: {pnl:+.4f} | Итого: {self.realized_pnl:+.4f}")
                            self.ensemble.record_outcome(pnl, pnl > 0)

                            # Ставим новый SELL на уровень выше (вход в новый шорт)
                            new_sell_price = round_price(price + step * 2)
                            if new_sell_price <= self.upper:
                                time.sleep(0.2)
                                oid_new = self.place_sell(new_sell_price, qty, buy_price=None, pos_side="short")
                                if not oid_new:
                                    log.warning(f"⚠️ Не удалось поставить SELL для нового шорта @ {new_sell_price}, добавляю в retry")
                                    self.positions_without_tp.append({"type": "SELL", "price": new_sell_price, "qty": qty, "buy_price": None, "pos_side": "short", "retries": 0})
                        else:
                            log.warning(f"⚠️ Шорт закрыт @ {price:.2f}, но нет цены входа")
                    else:
                        # Это BUY для входа в лонг - ставим SELL TP
                        tp_pct = CONFIG.get("take_profit_pct", 0.5) / 100
                        sell_price = round_price(price * (1 + tp_pct))
                        log.info(f"📤 SELL TP (лонг) @ {sell_price:.2f} (+{tp_pct*100:.1f}%)")
                        oid_tp = self.place_sell(sell_price, qty, buy_price=price, pos_side="long")
                        if not oid_tp:
                            log.warning(f"⚠️ Не удалось поставить SELL TP @ {sell_price}, добавляю в retry")
                            self.positions_without_tp.append({"type": "SELL", "price": sell_price, "qty": qty, "buy_price": price, "pos_side": "long", "retries": 0})

            elif otype == "SELL":
                if "-SWAP" in CONFIG["symbol"]:
                    if pos_side == "long":
                        # Это SELL TP для лонга (buy_price = цена входа в лонг)
                        pnl = round((price - buy_price_entry) * qty, 4)
                        self.realized_pnl += pnl
                        self.trades_count += 1
                        status.add_trade(pnl)
                        log.info(f"💰 Лонг закрыт @ {price:.2f} | PnL: {pnl:+.4f} | Итого: {self.realized_pnl:+.4f}")
                        self.notify(f"💰 Лонг: SELL @ {price:.2f} | PnL: {pnl:+.4f} | Итого: {self.realized_pnl:+.4f}")
                        self.ensemble.record_outcome(pnl, pnl > 0)

                        # Ставим новый BUY на уровень ниже
                        new_buy_price = round_price(price - step * 2)
                        if new_buy_price >= self.lower:
                            time.sleep(0.2)
                            oid_new = self.place_buy(new_buy_price, pos_side="long")
                            if not oid_new:
                                log.warning(f"⚠️ Не удалось поставить BUY @ {new_buy_price}, добавляю в retry")
                                self.positions_without_tp.append({"type": "BUY", "price": new_buy_price, "qty": qty, "buy_price": new_buy_price, "pos_side": "long", "retries": 0})
                    elif pos_side == "short":
                        # Это SELL как вход в ШОРТ - ставим BUY TP (закрытие шорта)
                        tp_pct = CONFIG.get("take_profit_pct", 0.5) / 100
                        buy_tp_price = round_price(price * (1 - tp_pct))
                        log.info(f"📥 BUY TP (шорт) @ {buy_tp_price:.2f} (+{tp_pct*100:.1f}%)")
                        oid_tp = self.place_buy(buy_tp_price, qty, buy_price=price, pos_side="short")
                        if not oid_tp:
                            log.warning(f"⚠️ Не удалось поставить BUY TP @ {buy_tp_price}, добавляю в retry")
                            self.positions_without_tp.append({"type": "BUY", "price": buy_tp_price, "qty": qty, "buy_price": price, "pos_side": "short", "retries": 0})
                    else:
                        # pos_side не указан - старый режим: если buy_price=None, это вход в шорт
                        if buy_price_entry is None:
                            tp_pct = CONFIG.get("take_profit_pct", 0.5) / 100
                            buy_tp_price = round_price(price * (1 - tp_pct))
                            log.info(f"📥 BUY TP (шорт, legacy) @ {buy_tp_price:.2f} (+{tp_pct*100:.1f}%)")
                            oid_tp = self.place_buy(buy_tp_price, qty, buy_price=price, pos_side="short")
                            if not oid_tp:
                                log.warning(f"⚠️ Не удалось поставить BUY TP @ {buy_tp_price}, добавляю в retry")
                                self.positions_without_tp.append({"type": "BUY", "price": buy_tp_price, "qty": qty, "buy_price": price, "pos_side": "short", "retries": 0})
                        else:
                            pnl = round((price - buy_price_entry) * qty, 4)
                            self.realized_pnl += pnl
                            self.trades_count += 1
                            log.info(f"💰 Лонг закрыт (legacy) @ {price:.2f} | PnL: {pnl:+.4f} | Итого: {self.realized_pnl:+.4f}")
                            self.notify(f"💰 Лонг: SELL @ {price:.2f} | PnL: {pnl:+.4f} | Итого: {self.realized_pnl:+.4f}")
                            self.ensemble.record_outcome(pnl, pnl > 0)

                            new_buy_price = round_price(price - step * 2)
                            if new_buy_price >= self.lower:
                                time.sleep(0.2)
                                oid_new = self.place_buy(new_buy_price, pos_side="long")
                                if not oid_new:
                                    log.warning(f"⚠️ Не удалось поставить BUY @ {new_buy_price}, добавляю в retry")
                                    self.positions_without_tp.append({"type": "BUY", "price": new_buy_price, "qty": qty, "buy_price": new_buy_price, "pos_side": "long", "retries": 0})

    def place_pending_sells(self):
        """Пытается разместить отложенные SELL ордера когда SOL появился"""
        if not self.pending_sells:
            return
        remaining = self.get_spot_holdings()
        if remaining <= 0:
            return
        placed_any = False
        still_pending = []
        for sell_price, qty in self.pending_sells:
            if qty <= remaining:
                oid = self.place_sell(sell_price, qty)
                if oid:
                    remaining -= qty
                    placed_any = True
                    continue
            still_pending.append((sell_price, qty))
        self.pending_sells = still_pending
        if placed_any:
            log.info(f"📋 Размещено SELL, осталось в очереди: {len(self.pending_sells)}")

    def retry_missing_tp_orders(self):
        """Повторно пытается поставить TP ордера для позиций без защиты"""
        if not self.positions_without_tp:
            return
        still_missing = []
        for item in self.positions_without_tp:
            if item["retries"] >= 5:
                log.error(f"❌ Не удалось поставить ордер после 5 попыток: {item}")
                self.notify(f"❌ ОШИБКА! Позиция без TP после 5 попыток: {item['type']} @ {item['price']:.2f}")
                continue
            oid = None
            if item["type"] == "BUY":
                oid = self.place_buy(item["price"], qty=item["qty"], buy_price=item["buy_price"], pos_side=item["pos_side"])
            elif item["type"] == "SELL":
                oid = self.place_sell(item["price"], qty=item["qty"], buy_price=item["buy_price"], pos_side=item["pos_side"])
            if oid:
                log.info(f"✅ TP ордер успешно поставлен после {item['retries']+1} попытки: {item['type']} @ {item['price']:.2f}")
            else:
                item["retries"] += 1
                still_missing.append(item)
        self.positions_without_tp = still_missing
        if still_missing:
            log.warning(f"⚠️ {len(still_missing)} TP ордеров всё ещё не выставлены")

    # ── Защиты ────────────────────────────────────────────────────

    def check_global_stops(self) -> bool:
        unrealized = self.get_unrealized_pnl()
        balance = self.get_balance()
        max_loss = CONFIG.get("global_sl_usdt", 25.0)
        if unrealized < -max_loss:
            self.notify(f"🚨 СТОП-ЛОСС!\nUnrealized: {unrealized:.2f} < -{max_loss}")
            return True
        max_profit = CONFIG.get("global_tp_usdt", 40.0)
        if self.realized_pnl >= max_profit:
            self.notify(f"🎯 ТЕЙК-ПРОФИТ!\nRealized: {self.realized_pnl:.2f} >= +{max_profit}")
            return True
        min_bal = CONFIG.get("min_balance_usdt", 50.0)
        if balance < min_bal:
            self.notify(f"⚠️ Баланс {balance:.2f} < {min_bal}")
            return True
        return False

    def check_per_order_stop_loss(self):
        """Проверяет stop loss для каждой открытой позиции"""
        if "-SWAP" not in CONFIG["symbol"]:
            return
        sl_pct = CONFIG.get("stop_loss_pct", 1.0) / 100
        if sl_pct <= 0:
            return
        try:
            r = self.account_api.get_positions(instType="SWAP", instId=CONFIG["symbol"])
            if r.get("code") != "0":
                return
            positions = r.get("data", [])
            for pos in positions:
                if pos.get("instId") != CONFIG["symbol"]:
                    continue
                sz = float(pos.get("pos", 0))
                if sz <= 0:
                    continue
                pos_side = pos.get("posSide", "net")
                avg_px = float(pos.get("avgPx", 0))

                # Определяем направление: в net mode posSide="net", направление по стороне позиции
                is_long = (pos_side == "long") or (pos_side == "net" and sz > 0)
                is_short = (pos_side == "short") or (pos_side == "net" and sz < 0)

                if is_long:
                    sl_price = avg_px * (1 - sl_pct)
                    current_price = self.get_price()
                    if current_price <= sl_price:
                        log.warning(f"🛑 STOP LOSS лонг @ {avg_px:.2f}, текущая {current_price:.2f}, SL уровень {sl_price:.2f}")
                        status.add_critical(f"SL лонг @ {avg_px:.2f}")
                        self.notify(f"🛑 STOP LOSS лонг! Вход: {avg_px:.2f}, Текущая: {current_price:.2f}")
                        try:
                            if getattr(self, 'use_pos_side', False):
                                r = self.trade_api.close_positions(instId=CONFIG["symbol"], mgnMode="isolated", posSide="long")
                            else:
                                r = self.trade_api.close_positions(instId=CONFIG["symbol"], mgnMode="isolated")
                            log.info(f"📤 close_positions ответ: {r}")
                            if r.get("code") == "0":
                                log.info(f"✅ Позиция лонг закрыта по SL")
                            else:
                                log.error(f"❌ Ошибка закрытия лонга: code={r.get('code')}, msg={r.get('msg')}, full={r}")
                                self.notify(f"❌ Ошибка закрытия SL: {r}")
                                # Fallback: закрываем market ордером
                                try:
                                    sz = abs(sz)
                                    r2 = self.trade_api.place_order(
                                        instId=CONFIG["symbol"],
                                        tdMode="isolated",
                                        side="sell",
                                        ordType="market",
                                        sz=str(sz),
                                        posSide="long" if getattr(self, 'use_pos_side', False) else None,
                                    )
                                    log.info(f"📤 Fallback market close ответ: {r2}")
                                    if r2.get("code") == "0":
                                        log.info(f"✅ Позиция лонг закрыта market ордером")
                                    else:
                                        log.error(f"❌ Fallback market close ошибка: {r2}")
                                except Exception as e2:
                                    log.error(f"❌ Fallback market close ошибка: {e2}")
                        except Exception as e:
                            log.error(f"❌ Ошибка close_positions (лонг): {e}")
                            self.notify(f"❌ Ошибка закрытия SL: {e}")
                elif is_short:
                    sl_price = avg_px * (1 + sl_pct)
                    current_price = self.get_price()
                    if current_price >= sl_price:
                        log.warning(f"🛑 STOP LOSS шорт @ {avg_px:.2f}, текущая {current_price:.2f}, SL уровень {sl_price:.2f}")
                        status.add_critical(f"SL шорт @ {avg_px:.2f}")
                        self.notify(f"🛑 STOP LOSS шорт! Вход: {avg_px:.2f}, Текущая: {current_price:.2f}")
                        try:
                            if getattr(self, 'use_pos_side', False):
                                r = self.trade_api.close_positions(instId=CONFIG["symbol"], mgnMode="isolated", posSide="short")
                            else:
                                r = self.trade_api.close_positions(instId=CONFIG["symbol"], mgnMode="isolated")
                            log.info(f"📤 close_positions ответ: {r}")
                            if r.get("code") == "0":
                                log.info(f"✅ Позиция шорт закрыта по SL")
                            else:
                                log.error(f"❌ Ошибка закрытия шорта: code={r.get('code')}, msg={r.get('msg')}, full={r}")
                                self.notify(f"❌ Ошибка закрытия SL: {r}")
                                # Fallback: закрываем market ордером
                                try:
                                    sz = abs(sz)
                                    r2 = self.trade_api.place_order(
                                        instId=CONFIG["symbol"],
                                        tdMode="isolated",
                                        side="buy",
                                        ordType="market",
                                        sz=str(sz),
                                        posSide="short" if getattr(self, 'use_pos_side', False) else None,
                                    )
                                    log.info(f"📤 Fallback market close ответ: {r2}")
                                    if r2.get("code") == "0":
                                        log.info(f"✅ Позиция шорт закрыта market ордером")
                                    else:
                                        log.error(f"❌ Fallback market close ошибка: {r2}")
                                except Exception as e2:
                                    log.error(f"❌ Fallback market close ошибка: {e2}")
                        except Exception as e:
                            log.error(f"❌ Ошибка close_positions (шорт): {e}")
                            self.notify(f"❌ Ошибка закрытия SL: {e}")
        except Exception as e:
            log.error(f"check_per_order_stop_loss ошибка: {e}")

    # ── Управление рисками позиций ─────────────────────────────────

    def check_position_risks(self):
        """
        Проверяет открытые позиции и оценивает риск:
        1. Если есть позиция без TP ордера — выставляет TP
        2. Если позиция долго не закрывается — закрывает немедленно
        3. Если риск потери > threshold — закрывает немедленно
        """
        try:
            # Получаем баланс для расчета риска
            balance = self.get_balance()
            risk_threshold = balance * CONFIG.get("risk_close_threshold", 0.02)  # 2% от баланса
            max_wait = CONFIG.get("max_wait_cycles", 10) * CONFIG.get("check_interval", 10)  # 100 сек

            # Получаем все открытые позиции
            r_pos = self.account_api.get_positions(instType="SWAP", instId=CONFIG["symbol"])
            if r_pos.get("code") != "0":
                return

            positions_data = r_pos.get("data", [])
            if not positions_data:
                return

            # Получаем все открытые ордера
            r_orders = self.trade_api.get_order_list(instType="SWAP", instId=CONFIG["symbol"], state="live")
            open_orders = r_orders.get("data", []) if r_orders.get("code") == "0" else []

            for pos in positions_data:
                pos_side = pos.get("posSide", "net")
                avg_px = float(pos.get("avgPx", 0))
                sz = float(pos.get("pos", 0))
                upl = float(pos.get("upl", 0))
                margin = float(pos.get("margin", 0))

                if sz <= 0:
                    continue

                direction = "long" if pos_side == "long" else "short" if pos_side == "short" else "net"
                pos_key = f"{pos_side}_{avg_px:.2f}"

                # 1. Проверяем, есть ли TP ордер для этой позиции
                has_tp_order = False
                for order in open_orders:
                    order_side = order.get("side")
                    order_pos_side = order.get("posSide")

                    # Для лонг позиции TP — это SELL ордер
                    # Для шорт позиции TP — это BUY ордер
                    if direction == "long" and order_side == "sell" and order_pos_side == "long":
                        has_tp_order = True
                        break
                    elif direction == "short" and order_side == "buy" and order_pos_side == "short":
                        has_tp_order = True
                        break

                # 2. Проверяем, есть ли позиция в synced_positions и сколько она ждет
                wait_count = self._position_wait_counts.get(pos_key, 0)

                # 3. Оцениваем риск
                risk_level = abs(upl)  # Абсолютное значение убытка

                log.info(f"🔍 Проверка позиции {direction}: {sz} SOL @ {avg_px:.2f} | UPL: {upl:+.2f} | TP: {'✅' if has_tp_order else '❌'} | Wait: {wait_count}")

                # 4. Если нет TP ордера — выставляем
                if not has_tp_order:
                    log.warning(f"⚠️ Позиция {direction} без TP ордера! Выставляю...")

                    if direction == "long":
                        tp_price = round_price(avg_px * (1 + CONFIG.get("take_profit_pct", 0.5) / 100))
                        oid = self.place_sell(tp_price, sz, buy_price=avg_px, pos_side="long")
                        if oid:
                            log.info(f"✅ Выставлен TP SELL @ {tp_price:.2f} для лонг позиции")
                            self.notify(f"✅ Выставлен TP для лонг: {sz} SOL @ {avg_px:.2f} → TP: {tp_price:.2f}")
                        else:
                            log.error(f"❌ Не удалось выставить TP для лонг позиции")
                    elif direction == "short":
                        tp_price = round_price(avg_px * (1 - CONFIG.get("take_profit_pct", 0.5) / 100))
                        oid = self.place_buy(tp_price, qty=sz, buy_price=avg_px, pos_side="short")
                        if oid:
                            log.info(f"✅ Выставлен TP BUY @ {tp_price:.2f} для шорт позиции")
                            self.notify(f"✅ Выставлен TP для шорт: {sz} SOL @ {avg_px:.2f} → TP: {tp_price:.2f}")
                        else:
                            log.error(f"❌ Не удалось выставить TP для шорт позиции")

                    self._synced_positions[pos_key] = True
                    self._position_wait_counts[pos_key] = 0

                # 5. Если позиция долго ждет — закрываем немедленно
                elif wait_count >= CONFIG.get("max_wait_cycles", 10):
                    log.warning(f"⚠️ Позиция {direction} ждет {wait_count} циклов! Закрываю немедленно...")
                    self.notify(f"⚠️ Позиция {direction} ждет слишком долго! Закрываю: {sz} SOL @ {avg_px:.2f} | UPL: {upl:+.2f}")

                    try:
                        close_result = self.trade_api.close_positions(
                            instId=CONFIG["symbol"],
                            mgnMode="isolated",
                            posSide=pos_side if pos_side != "net" else "net"
                        )
                        if close_result.get("code") == "0":
                            log.info(f"✅ Позиция {direction} закрыта немедленно")
                            self.notify(f"✅ Позиция закрыта: {direction} {sz} SOL | UPL: {upl:+.2f}")
                            # Убираем из synced_positions
                            self._synced_positions.pop(pos_key, None)
                            self._position_wait_counts.pop(pos_key, None)
                        else:
                            log.error(f"❌ Ошибка закрытия позиции: {close_result}")
                    except Exception as e:
                        log.error(f"❌ Ошибка закрытия позиции: {e}")
                        # Fallback — маркет ордер
                        try:
                            if direction == "long":
                                self.trade_api.place_order(
                                    instId=CONFIG["symbol"],
                                    tdMode="isolated",
                                    side="sell",
                                    posSide="long",
                                    ordType="market",
                                    sz=str(sz)
                                )
                            elif direction == "short":
                                self.trade_api.place_order(
                                    instId=CONFIG["symbol"],
                                    tdMode="isolated",
                                    side="buy",
                                    posSide="short",
                                    ordType="market",
                                    sz=str(sz)
                                )
                            log.info(f"✅ Позиция закрыта маркет ордером")
                            self._synced_positions.pop(pos_key, None)
                            self._position_wait_counts.pop(pos_key, None)
                        except Exception as e2:
                            log.error(f"❌ Fallback маркет ошибка: {e2}")

                # 6. Если риск потери > threshold — закрываем немедленно
                elif risk_level > risk_threshold and upl < 0:
                    log.warning(f"🚨 Риск позиции {direction}: {upl:.2f} USDT > {risk_threshold:.2f} USDT! Закрываю...")
                    self.notify(f"🚨 Высокий риск! Закрываю позицию {direction}: {sz} SOL | UPL: {upl:+.2f} | Лимит: {risk_threshold:.2f}")

                    try:
                        close_result = self.trade_api.close_positions(
                            instId=CONFIG["symbol"],
                            mgnMode="isolated",
                            posSide=pos_side if pos_side != "net" else "net"
                        )
                        if close_result.get("code") == "0":
                            log.info(f"✅ Позиция {direction} закрыта по риску")
                            self._synced_positions.pop(pos_key, None)
                            self._position_wait_counts.pop(pos_key, None)
                        else:
                            log.error(f"❌ Ошибка закрытия позиции по риску: {close_result}")
                    except Exception as e:
                        log.error(f"❌ Ошибка закрытия позиции по риску: {e}")

                # 7. Увеличиваем счетчик ожидания
                else:
                    self._position_wait_counts[pos_key] = wait_count + 1

        except Exception as e:
            log.error(f"❌ check_position_risks ошибка: {e}")

    # ── Основной цикл ─────────────────────────────────────────────

    def _loop(self):
        self.setup_account()
        self.start_balance = self.get_balance()
        try:
            df15 = self.get_klines(interval="15m", limit=200)
            for c, v in zip(reversed(df15["close"].tolist()), reversed(df15["vol"].tolist())):
                self.ai.add_price(c, v)
            log.info(f"🧠 AI прогрет")
        except Exception as e:
            log.warning(f"AI прогрев ошибка: {e}")

        price = self.get_price()
        self.last_price = price
        self.lower, self.upper = self.calc_atr_range(price)
        self.grid_levels = self.build_grid(self.lower, self.upper)
        self.notify(f"🚀 Grid Bot V3 (Multi-AI) запущен!\nБаланс: {self.start_balance:.2f} USDT\nЦена: {price:.2f}\nСетка: {self.lower:.2f} — {self.upper:.2f}")
        self.place_grid(price)
        self.start_time = datetime.now()

        # Инициализируем статус
        status.running = True
        status.balance = self.start_balance
        status.price = price
        status.grid_range = f"{self.lower:.2f} — {self.upper:.2f}"
        status.last_update = "🚀 Бот запущен!"

        # Рисуем статус первый раз
        render_status()

        trailing_cooldown = 0
        ai_cooldown = 0
        sync_cooldown = 0  # Синхронизация позиций каждые 60 сек

        while self.running:
            try:
                time.sleep(CONFIG["check_interval"])
                if not self.running:
                    break
                price = self.get_price()
                self.last_price = price
                try:
                    df = self.get_klines(interval="15m", limit=1)
                    vol = float(df["vol"].iloc[-1])
                except Exception:
                    vol = 0.0
                self.ai.add_price(price, vol)

                # Обновляем статус
                status.price = price
                status.balance = self.get_balance()
                status.open_orders = self.get_open_order_count()
                status.grid_range = f"{self.lower:.2f} — {self.upper:.2f}"
                status.running = True

                # Считаем открытые позиции
                try:
                    pr = self.account_api.get_positions(instType="SWAP", instId=CONFIG["symbol"])
                    if pr.get("code") == "0":
                        pos_list = [p for p in pr.get("data", []) if p.get("instId") == CONFIG["symbol"] and float(p.get("pos", 0)) > 0]
                        status.open_positions = len(pos_list)
                except Exception:
                    pass

                if self.check_global_stops():
                    self.running = False
                    self.cancel_all()
                    self.close_all_positions()
                    break

                self.check_filled()
                self.check_per_order_stop_loss()
                self.check_position_risks()  # Проверка риска открытых позиций
                self.retry_missing_tp_orders()
                self.place_pending_sells()

                if ai_cooldown > 0:
                    ai_cooldown -= 1
                else:
                    indicators = self.ai.get_indicators()
                    pos_size = self.get_spot_holdings()
                    self.last_signal = self.ensemble.analyze(
                        price, indicators, pos_size, self.realized_pnl, self.lower, self.upper
                    )
                    ai_cooldown = CONFIG.get("ai_analysis_interval", 900) // CONFIG.get("check_interval", 10)

                # Синхронизация позиций каждые 60 сек
                if sync_cooldown > 0:
                    sync_cooldown -= 1
                else:
                    self.sync_positions()
                    sync_cooldown = 6  # 60 сек / 10 сек интервал

                auto_bt = self.backtester.check_auto_run()
                if auto_bt:
                    self.notify(auto_bt)

                if trailing_cooldown > 0:
                    trailing_cooldown -= 1
                elif price > self.upper:
                    self.notify(f"📈 Цена {price:.2f} > {self.upper:.2f}")
                    self.trailing_up(price)
                    trailing_cooldown = 30
                elif price < self.lower:
                    self.notify(f"📉 Цена {price:.2f} < {self.lower:.2f}")
                    self.trailing_down(price)
                    trailing_cooldown = 30

                buy_cnt = sum(1 for o in self.active_orders.values() if o["type"] == "BUY")
                if buy_cnt == 0 and self.get_open_order_count() == 0:
                    self.lower, self.upper = self.calc_atr_range(price)
                    self.grid_levels = self.build_grid(self.lower, self.upper)
                    self.place_grid(price)

                # Отрисовка статуса в консоли
                render_status()

            except Exception as e:
                log.error(f"Цикл ошибка: {e}")
                status.add_critical(f"Цикл: {e}")
                time.sleep(15)

    def sync_positions(self):
        """Синхронизация позиций - проверяет открытые позиции на бирже"""
        try:
            inst_type = "SWAP" if "-SWAP" in CONFIG["symbol"] else "SPOT"
            r = self.account_api.get_positions(instType=inst_type)
            if r.get("code") != "0":
                return

            positions = r.get("data", [])
            current_keys = set()

            for pos in positions:
                if pos.get("instId") != CONFIG["symbol"]:
                    continue

                sz = float(pos.get("pos", 0))
                if sz <= 0:
                    continue

                pos_side_api = pos.get("posSide", "net")
                avg_px = float(pos.get("avgPx", 0))
                pos_key = f"{pos_side_api}_{avg_px:.2f}"
                current_keys.add(pos_key)

                # Уже обрабатывали эту позицию?
                if pos_key in self._synced_positions:
                    continue

                # Ищем в active_orders по цене (приблизительно)
                found = False
                for oid, order in self.active_orders.items():
                    if abs(order.get("price", 0) - avg_px) < 0.5:
                        found = True
                        break

                if not found:
                    # Нашли "потерянную" позицию!
                    log.warning(f"⚠️ Found lost position: {pos_side_api} {sz} @ {avg_px}")
                    self.notify(f"⚠️ Найдена потерянная позиция: {pos_side_api} {sz} @ {avg_px}")

                    if pos_side_api == "long":
                        tp_price = avg_px * (1 + CONFIG.get("take_profit_pct", 0.5) / 100)
                        oid = self.place_sell(tp_price, sz, buy_price=avg_px, pos_side="long")
                        if oid:
                            self._synced_positions[pos_key] = True
                            log.info(f"📤 SELL TP (лонг) выставлен @ {tp_price:.2f}")
                    elif pos_side_api == "short":
                        tp_price = avg_px * (1 - CONFIG.get("take_profit_pct", 0.5) / 100)
                        oid = self.place_buy(tp_price, qty=sz, buy_price=avg_px, pos_side="short")
                        if oid:
                            self._synced_positions[pos_key] = True
                            log.info(f"📥 BUY TP (шорт) выставлен @ {tp_price:.2f}")

            # Удаляем из _synced_positions те, которых больше нет (позиция закрылась)
            for key in list(self._synced_positions):
                if key not in current_keys:
                    del self._synced_positions[key]

            # Safety: ограничиваем размер _synced_positions
            max_synced = 100
            if len(self._synced_positions) > max_synced:
                # Удаляем самые старые записи (если их слишком много)
                keys_to_remove = list(self._synced_positions.keys())[:len(self._synced_positions) - max_synced]
                for key in keys_to_remove:
                    del self._synced_positions[key]
                log.warning(f"🔄 _synced_positions очищен: удалено {len(keys_to_remove)} записей")

        except Exception as e:
            log.warning(f"Синхронизация позиций ошибка: {e}")

    def sync_existing_orders_and_positions(self):
        """Синхронизирует существующие ордеры и позиции с биржи при запуске"""
        log.info("🔄 Синхронизация ордеров и позиций с биржи...")

        try:
            # 1. Получаем все открытые ордера
            r = self.trade_api.get_order_list(instType="SWAP", instId=CONFIG["symbol"], state="live")
            self._check_okx_response(r)

            orders_data = r.get("data", [])
            log.info(f"📋 Найдено {len(orders_data)} открытых ордеров на бирже")

            # Синхронизируем ордера
            for order in orders_data:
                ord_id = order.get("ordId")
                price = float(order.get("px", 0))
                qty = float(order.get("sz", 0))
                side = order.get("side")  # "buy" или "sell"
                pos_side = order.get("posSide", "net")  # "long", "short", или "net"
                ord_type = order.get("ordType", "limit")

                # Определяем тип ордера для бота
                if side == "buy":
                    bot_type = "BUY"
                else:
                    bot_type = "SELL"

                # Добавляем в active_orders
                self.active_orders[ord_id] = {
                    "price": price,
                    "type": bot_type,
                    "qty": qty,
                    "pos_side": pos_side,
                    "buy_price": price if bot_type == "BUY" else None,
                    "ord_type": ord_type
                }

                log.info(f"  📌 Ордер {ord_id}: {bot_type} {qty} @ {price:.2f} ({pos_side})")

            log.info(f"✅ Синхронизировано {len(self.active_orders)} ордеров")

            # 2. Получаем все открытые позиции
            r_pos = self.account_api.get_positions(instType="SWAP", instId=CONFIG["symbol"])
            self._check_okx_response(r_pos)

            positions_data = r_pos.get("data", [])
            log.info(f"📋 Найдено {len(positions_data)} открытых позиций на бирже")

            if positions_data:
                for pos in positions_data:
                    pos_side = pos.get("posSide", "net")
                    avg_px = float(pos.get("avgPx", 0))
                    sz = float(pos.get("pos", 0))
                    upl = float(pos.get("upl", 0))
                    margin = float(pos.get("margin", 0))

                    direction = "🟢 LONG" if pos_side == "long" else "🔴 SHORT" if pos_side == "short" else "⚪ NET"

                    log.info(f"  💼 Позиция {direction}: {sz} SOL @ {avg_px:.2f} | UPL: {upl:+.2f} USDT | Margin: {margin:.2f}")

                    # Добавляем в synced_positions чтобы не обрабатывать повторно
                    pos_key = f"{pos_side}_{avg_px:.2f}"
                    self._synced_positions[pos_key] = True

                    # Обновляем avg_buy_price если это лонг
                    if pos_side == "long" and sz > 0:
                        self.avg_buy_price = avg_px
                        self.total_sol_bought = sz
                        log.info(f"    📊 Обновлен avg_buy_price: {self.avg_buy_price:.2f}")

                # Уведомление в Telegram
                pos_summary = f"🔄 Обнаружено позиций на бирже: {len(positions_data)}\n"
                for pos in positions_data:
                    pos_side = pos.get("posSide", "net")
                    avg_px = float(pos.get("avgPx", 0))
                    sz = float(pos.get("pos", 0))
                    upl = float(pos.get("upl", 0))

                    direction = "🟢 LONG" if pos_side == "long" else "🔴 SHORT" if pos_side == "short" else "⚪ NET"
                    pos_summary += f"\n{direction}: {sz} SOL @ {avg_px:.2f} | PnL: {upl:+.2f} USDT"

                pos_summary += f"\n\n📋 Ордеров: {len(self.active_orders)}"
                self.notify(pos_summary)
            else:
                log.info("✅ Нет открытых позиций")
                self.notify("✅ Бот запущен, нет открытых позиций")

            # Итоговая сводка
            total_orders = len(self.active_orders)
            total_positions = len(positions_data)

            log.info(f"✅ Синхронизация завершена: {total_orders} ордеров, {total_positions} позиций")

            return {
                "orders": total_orders,
                "positions": total_positions
            }

        except Exception as e:
            log.error(f"❌ Ошибка синхронизации: {e}")
            self.notify(f"⚠️ Ошибка синхронизации: {e}")
            return {
                "orders": 0,
                "positions": 0
            }

    def start(self) -> bool:
        if self.running:
            return False
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> bool:
        if not self.running:
            return False
        self.running = False
        self.cancel_all()
        return True

    def emergency_stop(self):
        self.running = False
        self.cancel_all()
        self.close_all_positions()
        self.notify("🛑 Экстренная остановка")

    def status_text(self) -> str:
        """Улучшенный формат статуса бота"""
        uptime = ""
        if self.start_time:
            d = datetime.now() - self.start_time
            h, m = divmod(int(d.total_seconds()), 3600)
            m, s = divmod(m, 60)
            uptime = f"{h:02d}:{m:02d}:{s:02d}"

        buys = sum(1 for o in self.active_orders.values() if o["type"] == "BUY")
        sells = sum(1 for o in self.active_orders.values() if o["type"] == "SELL")
        unrealized = self.get_unrealized_pnl()
        total_pnl = self.realized_pnl + unrealized
        ctx = self.market.get_market_context(self.last_price)

        # Эмодзи для статуса
        status_emoji = "🟢" if self.running else "🔴"
        status_text = "РАБОТАЕТ" if self.running else "ОСТАНОВЛЕН"

        # Эмодзи для PnL
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        pnl_sign = "+" if total_pnl >= 0 else ""

        lines = [
            f"{status_emoji} *СТАТУС БОТА* {status_text}",
            f"⏱ Uptime: `{uptime}`",
            "",
            "━" * 15 + "💰 *ПОРТФЕЛЬ*" + "━" * 15,
            f"💵 Баланс: `{self.get_balance():.2f} USDT`",
            f"📊 PnL Реализованный: `{self.realized_pnl:+.2f} USDT`",
            f"📈 PnL Нереализованный: `{unrealized:+.2f} USDT`",
            f"{pnl_emoji} *PnL Итого: `{pnl_sign}{total_pnl:.2f} USDT`*",
            "",
            "━" * 15 + "📊 *СЕТКА*" + "━" * 15,
            f"💱 Пара: `{CONFIG['symbol']}`",
            f"📏 Диапазон: `{self.lower:.2f} — {self.upper:.2f} USDT`",
            f"📐 Уровней: `{CONFIG['grid_levels']}`",
            f"🛒 BUY ордеров: `{buys}` | 📤 SELL ордеров: `{sells}`",
            f"📋 Всего ордеров: `{len(self.active_orders)}`",
            "",
            "━" * 15 + "🛡 *РИСК-МЕНЕДЖМЕНТ*" + "━" * 15,
            f"🔴 Stop-Loss: `-{CONFIG.get('global_sl_usdt', 25):.0f} USDT`",
            f"🟢 Take-Profit: `+{CONFIG.get('global_tp_usdt', 40):.0f} USDT`",
        ]

        # Информация о существующих позициях (если есть)
        if len(self._synced_positions) > 0:
            lines += [
                "",
                "━" * 15 + "💼 *ОТКРЫТЫЕ ПОЗИЦИИ*" + "━" * 15,
            ]
            # Получаем актуальные данные о позициях
            try:
                r_pos = self.account_api.get_positions(instType="SWAP", instId=CONFIG["symbol"])
                if r_pos.get("code") == "0":
                    positions_data = r_pos.get("data", [])
                    if positions_data:
                        for pos in positions_data:
                            pos_side = pos.get("posSide", "net")
                            avg_px = float(pos.get("avgPx", 0))
                            sz = float(pos.get("pos", 0))
                            upl = float(pos.get("upl", 0))

                            direction = "🟢 LONG" if pos_side == "long" else "🔴 SHORT" if pos_side == "short" else "⚪ NET"
                            lines.append(f"{direction}: `{sz}` SOL @ `{avg_px:.2f}` | PnL: `{upl:+.2f}` USDT")
                    else:
                        lines.append("✅ Нет открытых позиций")
            except Exception:
                lines.append("⚠️ Не удалось загрузить позиции")

        # AI сигнал
        sig = self.last_signal
        emoji = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "NEUTRAL": "⚪", "SELL": "🔴", "STRONG_SELL": "🔴🔴"}.get(
            sig.get("signal", "NEUTRAL"), "⚪"
        )
        lines += [
            "",
            "━" * 15 + "🧠 *AI СИГНАЛ*" + "━" * 15,
            f"{emoji} `{sig.get('signal', '—')}`",
            f"📊 Score: `{sig.get('score', 0):.2f}` | 🎯 Confidence: `{sig.get('confidence', 0.5):.2f}`",
        ]

        # Индикаторы
        ind = sig.get("indicators", {})
        if ind:
            rsi = ind.get('rsi', '—')
            macd = ind.get('macd_hist', '—')
            # RSI статус
            if isinstance(rsi, (int, float)):
                if rsi < 30:
                    rsi_status = "🟢 Перепроданность"
                elif rsi > 70:
                    rsi_status = "🔴 Перекупленность"
                else:
                    rsi_status = "⚪ Нейтрально"
                lines.append(f"📊 RSI: `{rsi}` {rsi_status}")
            else:
                lines.append(f"📊 RSI: `{rsi}`")
            lines.append(f"📊 MACD: `{macd}`")

        # Рынок
        lines += [
            "",
            "━" * 15 + "🌍 *РЫНОК*" + "━" * 15,
            f"😨 Fear&Greed: `{ctx.get('fear_greed_value', 50)}/100` ({ctx.get('fear_greed_label', '—')})",
            f"₿ BTC: `${ctx.get('btc_price', 0):,.0f}`",
            f"🔗 Корреляция SOL/BTC: `{ctx.get('sol_btc_correlation', 0.7)}`",
            f"🕐 Сессия: `{ctx.get('trading_session', '—')}` | `{ctx.get('day_of_week', '—')}`",
        ]

        return "\n".join(lines)

    def ai_signal_text(self) -> str:
        """Улучшенный формат AI сигнала"""
        sig = self.last_signal
        emoji = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "NEUTRAL": "⚪", "SELL": "🔴", "STRONG_SELL": "🔴🔴"}.get(
            sig.get("signal", "NEUTRAL"), "⚪"
        )

        text = f"🧠 *AI MULTI-PROVIDER SIGNAL*\n\n"
        text += f"{emoji} *{sig.get('signal', '—')}*\n"
        text += f"📊 Score: `{sig.get('score', 0):.2f}`\n"
        text += f"🎯 Confidence: `{sig.get('confidence', 0.5):.3f}`\n"

        # Голоса провайдеров
        text += "\n" + "━" * 12 + "🗳 *ГОЛОСА ПРОВАЙДЕРОВ*" + "━" * 12 + "\n"
        for name, s in sig.get("providers", {}).items():
            e = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "NEUTRAL": "⚪", "SELL": "🔴", "STRONG_SELL": "🔴🔴"}.get(s, "⚪")
            text += f"{e} {name.capitalize()}: `{s}`\n"

        # Веса
        text += "\n" + "━" * 12 + "⚖️ *АДАПТИВНЫЕ ВЕСА*" + "━" * 12 + "\n"
        for name, w in sorted(sig.get("weights", {}).items(), key=lambda x: -x[1]):
            text += f"• {name.capitalize()}: `{w:.3f}`\n"

        # Индикаторы
        ind = sig.get("indicators", {})
        if ind:
            text += "\n" + "━" * 12 + "📊 *ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ*" + "━" * 12 + "\n"
            rsi = ind.get('rsi', '—')
            if isinstance(rsi, (int, float)):
                if rsi < 30:
                    rsi_status = "🟢 Перепроданность"
                elif rsi > 70:
                    rsi_status = "🔴 Перекупленность"
                else:
                    rsi_status = "⚪ Нейтрально"
                text += f"📊 RSI(14): `{rsi}` {rsi_status}\n"
            else:
                text += f"📊 RSI(14): `{rsi}`\n"

            macd = ind.get('macd_hist', '—')
            if isinstance(macd, (int, float)):
                macd_emoji = "🟢" if macd > 0 else "🔴"
                text += f"{macd_emoji} MACD Hist: `{macd:.4f}`\n"
            else:
                text += f"📊 MACD Hist: `{macd}`\n"

            text += f"📊 MA20: `{ind.get('ma20', '—')}`\n"
            text += f"📊 MA50: `{ind.get('ma50', '—')}`\n"
            text += f"📊 BB: `{ind.get('bb_lower', '—')} — {ind.get('bb_upper', '—')}`\n"

        # Рынок
        ctx = self.market.get_market_context(self.last_price)
        text += "\n" + "━" * 12 + "🌍 *РЫНОЧНЫЙ КОНТЕКСТ*" + "━" * 12 + "\n"
        text += f"😨 Fear&Greed: `{ctx.get('fear_greed_value', 50)}/100` ({ctx.get('fear_greed_label', '—')})\n"
        text += f"₿ BTC: `${ctx.get('btc_price', 0):,.0f}`\n"
        text += f"📊 Доминация BTC: `{ctx.get('btc_dominance', 54.0):.1f}%`\n"
        text += f"🔗 Корреляция SOL/BTC: `{ctx.get('sol_btc_correlation', 0.7)}`\n"
        text += f"🕐 Сессия: `{ctx.get('trading_session', '—')}` ({ctx.get('day_of_week', '—')})\n"

        # Влияние на торговлю
        text += "\n" + "━" * 12 + "📌 *ВЛИЯНИЕ НА ТОРГОВЛЮ*" + "━" * 12 + "\n"
        signal = sig.get("signal", "NEUTRAL")
        if signal in ("STRONG_BUY", "BUY"):
            text += "✅ Полные BUY ордера\n"
            text += "📈 AI подтверждает покупку"
        elif signal == "NEUTRAL":
            text += "⚪ Стандартные ордера\n"
            text += "⚖️ AI не дает однозначного сигнала"
        elif signal == "SELL":
            text += "⚠️ BUY qty уменьшена на 50%\n"
            text += "🔻 AI рекомендует осторожность"
        elif signal == "STRONG_SELL":
            text += "🚫 BUY ордера пропущены\n"
            text += "🔴 AI сильно рекомендует продажу"

        return text


# ══════════════════════════════════════════════════════════════════
#  TELEGRAM
# ══════════════════════════════════════════════════════════════════

bot = GridBotV3()


def auth(update: Update) -> bool:
    return not ALLOWED_CHAT_IDS or update.effective_chat.id in ALLOWED_CHAT_IDS


def persistent_keyboard():
    """Улучшенная персистентная клавиатура с быстрыми действиями"""
    return ReplyKeyboardMarkup([
        [KeyboardButton("📊 Статус"), KeyboardButton("💰 Портфель"), KeyboardButton("📈 Позиции")],
        [KeyboardButton("🧠 AI Анализ"), KeyboardButton("📊 Backtest"), KeyboardButton("⚙️ Настройки")],
    ], resize_keyboard=True)


def main_keyboard():
    """Главное inline меню - улучшенная структура"""
    return InlineKeyboardMarkup([
        # Управление ботом
        [InlineKeyboardButton("▶️ Запустить", callback_data="start"),
         InlineKeyboardButton("⏸ Пауза", callback_data="pause"),
         InlineKeyboardButton("⏹ Остановить", callback_data="stop")],

        # Быстрая информация
        [InlineKeyboardButton("📊 Статус", callback_data="status"),
         InlineKeyboardButton("💰 Баланс", callback_data="balance"),
         InlineKeyboardButton("📈 Позиции", callback_data="positions")],

        # AI и аналитика
        [InlineKeyboardButton("🧠 AI Сигнал", callback_data="ai_signal"),
         InlineKeyboardButton("📊 Backtest", callback_data="backtest"),
         InlineKeyboardButton("📋 Ордера", callback_data="orders")],

        # Настройки
        [InlineKeyboardButton("⚙️ Настройки бота", callback_data="settings"),
         InlineKeyboardButton("🧠 Настройки AI", callback_data="ai_settings")],

        # Экстренные действия
        [InlineKeyboardButton("🚨 Экстренный стоп", callback_data="estop")],
    ])


def settings_keyboard():
    """Улучшенная клавиатура настроек"""
    return InlineKeyboardMarkup([
        # Риск-менеджмент
        [InlineKeyboardButton("🛡 Stop-Loss", callback_data="set_sl"),
         InlineKeyboardButton("🎯 Take-Profit", callback_data="set_tp")],

        # Параметры сетки
        [InlineKeyboardButton("📐 Уровни сетки", callback_data="set_levels"),
         InlineKeyboardButton("📊 ATR множитель", callback_data="set_atr")],

        # Размер позиции
        [InlineKeyboardButton("📦 Макс qty", callback_data="set_maxqty"),
         InlineKeyboardButton("💰 % Баланса", callback_data="set_balpct")],

        # Навигация
        [InlineKeyboardButton("🔙 Назад", callback_data="back_main")],
    ])


def ai_settings_keyboard():
    """Улучшенная клавиатура настроек AI"""
    return InlineKeyboardMarkup([
        # Роли и стили
        [InlineKeyboardButton("🎭 Роль", callback_data="set_ai_role"),
         InlineKeyboardButton("🎯 Стиль", callback_data="set_ai_style")],

        # Риск и веса
        [InlineKeyboardButton("⚠️ Риск-профиль", callback_data="set_ai_risk"),
         InlineKeyboardButton("📊 Вес BTC", callback_data="set_btc_weight")],

        # Частота и адаптация
        [InlineKeyboardButton("⏱ Частота AI", callback_data="set_ai_freq"),
         InlineKeyboardButton("⚖️ Адапт. веса", callback_data="set_adaptive")],

        # Навигация
        [InlineKeyboardButton("🔙 Назад", callback_data="back_main")],
    ])


HELP_TEXT = (
    "🤖 *OKX Grid Bot V3 — Multi-AI Trading*\n\n"
    "📌 *Стратегия:*\n"
    "• Фьючерсы SOL-USDT-SWAP с leverage 10x\n"
    "• Динамическая сетка на основе ATR\n"
    "• Multi-AI анализ с 5 LLM провайдерами\n"
    "• Adaptive weights + trade memory\n"
    "• Multi-timeframe analysis (M5, M15, H1, H4)\n\n"
    "📌 *AI Провайдеры:*\n"
    "• 🟢 Groq — Trend Following (4H)\n"
    "• 🟡 OpenRouter — Mean Reversion (1H)\n"
    "• 🔵 Cohere — Risk Manager (консервативный)\n"
    "• 🟣 DeepSeek — Scalping (15m)\n"
    "• 🟠 Gemini — Market Context (multi-TF)\n\n"
    "📌 *Команды:*\n"
    "`/start` — главное меню\n"
    "`/status` — быстрый статус\n"
    "`/ai` — AI сигнал и аналитика\n"
    "`/backtest` — точность AI за 7 дней\n"
    "`/sync` — синхронизация с биржей\n"
    "`/help` — эта справка\n"
    "`/set param value` — изменить параметр\n\n"
    "📌 *Быстрые кнопки:*\n"
    "📊 Статус — текущее состояние бота\n"
    "💰 Портфель — баланс и PnL\n"
    "📈 Позиции — активные позиции\n"
    "🧠 AI Анализ — детальный AI сигнал\n"
    "📊 Backtest — статистика AI\n"
    "⚙️ Настройки — параметры бота\n\n"
    "📌 *При запуске:*\n"
    "Бот автоматически синхронизирует\n"
    "существующие ордера и позиции\n"
    "с биржей и покажет отчет\n"
)


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown", reply_markup=persistent_keyboard())


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown", reply_markup=persistent_keyboard())


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    await update.message.reply_text(bot.status_text(), parse_mode="Markdown", reply_markup=persistent_keyboard())


async def cmd_ai(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    await update.message.reply_text(bot.ai_signal_text(), parse_mode="Markdown", reply_markup=persistent_keyboard())


async def cmd_backtest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    args = ctx.args
    days = int(args[0]) if args else 7
    report = bot.backtester.run_analysis(days)
    await update.message.reply_text(report, parse_mode="Markdown", reply_markup=persistent_keyboard())


async def cmd_set(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    args = ctx.args
    ai_params = {"ai_role", "ai_style", "ai_risk", "ai_timeframe"}
    if len(args) < 2:
        await update.message.reply_text(
            "Использование: `/set param value`\n\n"
            "Параметры бота:\n"
            "`global_sl_usdt`, `global_tp_usdt`, `grid_levels`,\n"
            "`atr_multiplier`, `max_qty`,\n"
            "`max_balance_pct`, `min_balance_usdt`\n\n"
            "AI параметры:\n"
            "`ai_role`, `ai_style`, `ai_risk`, `ai_timeframe`",
            parse_mode="Markdown", reply_markup=persistent_keyboard()
        )
        return
    key, val = args[0], " ".join(args[1:])
    if key in ai_params:
        CONFIG[key] = val
        await update.message.reply_text(f"✅ AI `{key}` = `{CONFIG[key]}`", parse_mode="Markdown", reply_markup=persistent_keyboard())
    elif key in CONFIG:
        try:
            if isinstance(CONFIG[key], bool): CONFIG[key] = val.lower() in ("true", "1", "yes")
            elif isinstance(CONFIG[key], int): CONFIG[key] = int(val)
            else: CONFIG[key] = float(val)
            await update.message.reply_text(f"✅ `{key}` = `{CONFIG[key]}`", parse_mode="Markdown", reply_markup=persistent_keyboard())
        except ValueError:
            await update.message.reply_text("❌ Неверное значение", reply_markup=persistent_keyboard())
    else:
        await update.message.reply_text(f"❌ Неизвестный параметр: `{key}`", parse_mode="Markdown", reply_markup=persistent_keyboard())


async def cmd_stop(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Остановка бота - закрывает все позиции и ордера"""
    if not auth(update): return

    bot = ctx.bot_data.get("bot")
    if not bot:
        await update.message.reply_text("❌ Бот не инициализирован", reply_markup=persistent_keyboard())
        return

    await update.message.reply_text("🔴 Останавливаю бота...", reply_markup=persistent_keyboard())

    try:
        # 1. Отменяем все ордера
        cancel_result = bot.trade_api.cancel_all_orders(instType="SWAP" if "-SWAP" in CONFIG["symbol"] else "SPOT", instId=CONFIG["symbol"])
        cancelled_count = len(cancel_result.get("data", [])) if cancel_result.get("code") == "0" else 0

        # 2. Закрываем все позиции
        positions_closed = 0
        try:
            close_result = bot.trade_api.close_positions(instId=CONFIG["symbol"], mgnMode="isolated")
            if close_result.get("code") == "0":
                positions_closed = len(close_result.get("data", []))
        except Exception as e:
            log.warning(f"close_positions не доступен: {e}")

        # 3. Получаем PnL
        pnl = 0
        try:
            bal = bot.get_balance()
            pnl = bal - 5000  # примерно
        except:
            pass

        log.info(f"🔴 STOP: отменено={cancelled_count}, закрыто={positions_closed}, PnL~={pnl:.2f}")

        # 4. Уведомление
        msg = f"""🔴 *БОТ ОСТАНОВЛЕН*

✅ Отменено ордеров: {cancelled_count}
✅ Закрыто позиций: {positions_closed}
💰 Примерный PnL: `{pnl:.2f} USDT`"""

        await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=persistent_keyboard())

        # 5. Останавливаем бота
        log.info("🔴 Бот остановлен пользователем")
        os._exit(0)

    except Exception as e:
        log.error(f"Ошибка при остановке: {e}")
        await update.message.reply_text(f"❌ Ошибка: {e}", reply_markup=persistent_keyboard())


async def cmd_sync(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Ручная синхронизация ордеров и позиций с биржей"""
    if not auth(update): return

    bot_instance = ctx.bot_data.get("bot")
    if not bot_instance:
        await update.message.reply_text("❌ Бот не инициализирован", reply_markup=persistent_keyboard())
        return

    msg = await update.message.reply_text("🔄 Синхронизация с биржей...", reply_markup=persistent_keyboard())

    try:
        # Вызываем синхронизацию
        sync_result = bot_instance.sync_existing_orders_and_positions()

        # Формируем ответ
        if sync_result["orders"] > 0 or sync_result["positions"] > 0:
            response = f"✅ *Синхронизация завершена*\n\n"
            response += f"📋 Ордеров найдено: `{sync_result['orders']}`\n"
            response += f"💼 Позиций найдено: `{sync_result['positions']}`\n\n"

            if sync_result["positions"] > 0:
                response += "💼 *Открытые позиции:*\n"
                try:
                    r_pos = bot_instance.account_api.get_positions(instType="SWAP", instId=CONFIG["symbol"])
                    if r_pos.get("code") == "0":
                        for pos in r_pos.get("data", []):
                            pos_side = pos.get("posSide", "net")
                            avg_px = float(pos.get("avgPx", 0))
                            sz = float(pos.get("pos", 0))
                            upl = float(pos.get("upl", 0))

                            direction = "🟢 LONG" if pos_side == "long" else "🔴 SHORT" if pos_side == "short" else "⚪ NET"
                            response += f"{direction}: `{sz}` SOL @ `{avg_px:.2f}` | PnL: `{upl:+.2f}` USDT\n"
                except Exception:
                    response += "⚠️ Не удалось загрузить детали позиций\n"

            if sync_result["orders"] > 0:
                response += f"\n📋 *Ордера:*\n"
                buys = sum(1 for o in bot_instance.active_orders.values() if o["type"] == "BUY")
                sells = sum(1 for o in bot_instance.active_orders.values() if o["type"] == "SELL")
                response += f"🛒 BUY: `{buys}`\n"
                response += f"📤 SELL: `{sells}`\n"
                response += f"📊 Всего: `{sync_result['orders']}`"
        else:
            response = "✅ *Синхронизация завершена*\n\n"
            response += "✅ Нет открытых ордеров или позиций"

        await msg.edit_text(response, parse_mode="Markdown")

    except Exception as e:
        log.error(f"Ошибка синхронизации: {e}")
        await msg.edit_text(f"❌ Ошибка синхронизации: {e}", parse_mode="Markdown")


def format_settings_page() -> str:
    return (
        "⚙️ *Настройки бота (спот)*\n\n"
        f"📊 Пара: `{CONFIG['symbol']}`\n"
        f"🔢 Уровней: `{CONFIG['grid_levels']}`\n"
        f"📐 ATR множитель: `{CONFIG.get('atr_multiplier', 2.5)}`\n"
        f"📦 Макс qty: `{CONFIG['max_qty']}` SOL\n"
        f"💰 % Баланса: `{CONFIG['max_balance_pct']*100:.0f}%`\n\n"
        f"🔒 Stop-Loss: `-{CONFIG.get('global_sl_usdt', 25):.0f} USDT`\n"
        f"🎯 Take-Profit: `+{CONFIG.get('global_tp_usdt', 40):.0f} USDT`\n"
        f"⚠️ Мин. баланс: `{CONFIG.get('min_balance_usdt', 50):.0f} USDT`\n\n"
        "Выбери параметр для изменения:"
    )


def format_ai_settings_page() -> str:
    return (
        "🧠 *Настройки AI*\n\n"
        f"🎭 Роль: `{CONFIG.get('ai_role', 'профессиональный крипто-трейдер')}`\n"
        f"🎯 Стиль: `{CONFIG.get('ai_style', 'свинг-трейдинг')}`\n"
        f"⚠️ Риск: `{CONFIG.get('ai_risk', 'умеренный')}`\n"
        f"📊 Вес BTC: `{CONFIG.get('ai_btc_corr_weight', 0.5)}`\n"
        f"⏱ Частота: `{CONFIG.get('ai_analysis_interval', 900)}с`\n"
        f"⚖️ Адапт. веса: `{'ВКЛ' if CONFIG.get('ai_adaptive_weights', True) else 'ВЫКЛ'}`\n"
        f"📝 Память: `{CONFIG.get('ai_memory_size', 20)} решений`\n\n"
        f"Провайдеры: `{', '.join(CONFIG.get('ai_enabled_providers', []))}`\n\n"
        "Выбери параметр для изменения:"
    )


async def callback_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass

    if q.data == "start":
        ok = bot.start()
        if ok:
            await q.edit_message_text("✅ *Бот запущен!*\n\n🔄 Строю сетку...\n🧠 AI анализирует...\n🛡 Защиты активны.", parse_mode="Markdown", reply_markup=main_keyboard())
        else:
            await q.edit_message_text("⚠️ *Бот уже работает!*", parse_mode="Markdown", reply_markup=main_keyboard())

    elif q.data == "stop":
        ok = bot.stop()
        if ok:
            await q.edit_message_text("⏹ *Бот остановлен*\n\n📋 Ордера отменены\n📍 Позиции остаются", parse_mode="Markdown", reply_markup=main_keyboard())
        else:
            await q.edit_message_text("ℹ️ *Бот не запущен*", parse_mode="Markdown", reply_markup=main_keyboard())

    elif q.data == "estop":
        bot.emergency_stop()
        await q.edit_message_text("🚨 *ЭКСТРЕННАЯ ОСТАНОВКА!*\n\n❌ Ордера отменены\n📉 Позиции закрыты\n🛑 Бот остановлен", parse_mode="Markdown", reply_markup=main_keyboard())

    elif q.data == "status":
        await q.edit_message_text(bot.status_text(), parse_mode="Markdown", reply_markup=main_keyboard())

    elif q.data == "balance":
        balance = bot.get_balance()
        avail = bot.get_available_balance()
        unrealized = bot.get_unrealized_pnl()
        price = bot.get_price()

        if "-SWAP" in CONFIG["symbol"]:
            # Фьючерсы — показываем позиции
            try:
                r = bot.account_api.get_positions(instType="SWAP", instId=CONFIG["symbol"])
                positions_text = ""
                if r.get("code") == "0":
                    positions = r.get("data", [])
                    if positions:
                        for pos in positions:
                            sz = float(pos.get("pos", 0))
                            if sz > 0:
                                ps = pos.get("posSide", "net")
                                avg_px = float(pos.get("avgPx", 0))
                                upl = float(pos.get("upl", 0))
                                direction = "🟢 Лонг" if ps in ("long", "net") else "🔴 Шорт"
                                positions_text += f"\n{direction}: `{sz}` @ `{avg_px:.2f}` | UPL: `{upl:+.4f}`"
                    else:
                        positions_text = "\nНет открытых позиций"
            except Exception:
                positions_text = "\nНе удалось загрузить позиции"

            await q.edit_message_text(
                f"💰 *Баланс (фьючерсы)*\n\n"
                f"💵 USDT: `{balance:.2f}`\n"
                f"💳 Доступно: `{avail:.2f} USDT`\n"
                f"💲 Цена: `{price:.2f}`\n"
                f"📊 Позиции:{positions_text}\n\n"
                f"📈 Realized PnL: `{bot.realized_pnl:+.4f} USDT`\n"
                f"📊 Unrealized PnL: `{unrealized:+.4f} USDT`\n"
                f"💎 *Итого PnL: `{bot.realized_pnl + unrealized:+.4f} USDT`*",
                parse_mode="Markdown", reply_markup=main_keyboard())
        else:
            # Спот
            sol_bal = bot.get_sol_balance()
            holdings = bot.get_spot_holdings()
            await q.edit_message_text(
                f"💰 *Баланс (спот)*\n\n"
                f"💵 USDT: `{balance:.2f}`\n"
                f"💳 Доступно USDT: `{avail:.2f}`\n"
                f"🪙 SOL в кошельке: `{sol_bal:.2f}`\n"
                f"📊 SOL в ордерах: `{holdings:.2f}`\n"
                f"💲 Цена SOL: `{price:.2f} USDT`\n"
                f"💎 Стоимость SOL: `{sol_bal * price:.2f} USDT`\n\n"
                f"📈 Realized PnL: `{bot.realized_pnl:+.4f} USDT`\n"
                f"📊 Avg buy: `{bot.avg_buy_price:.2f}`\n"
                f"💎 *Итого PnL: `{bot.realized_pnl + unrealized:+.4f} USDT`*",
                parse_mode="Markdown", reply_markup=main_keyboard())

    elif q.data == "orders":
        buys = {k: v for k, v in bot.active_orders.items() if v["type"] == "BUY"}
        sells = {k: v for k, v in bot.active_orders.items() if v["type"] == "SELL"}
        text = "📋 *Открытые ордера*\n\n"
        if buys:
            text += f"🟢 *BUY ({len(buys)})*\n"
            for oid, o in sorted(buys.items(), key=lambda x: -x[1]["price"])[:5]:
                text += f"  `{o['price']:.2f}` × `{o['qty']}`\n"
            if len(buys) > 5: text += f"  ... +{len(buys)-5}\n"
            text += "\n"
        else:
            text += "🟢 BUY: `нет`\n\n"
        if sells:
            text += f"🔴 *SELL ({len(sells)})*\n"
            for oid, o in sorted(sells.items(), key=lambda x: x[1]["price"])[:5]:
                text += f"  `{o['price']:.2f}` × `{o['qty']}`\n"
            if len(sells) > 5: text += f"  ... +{len(sells)-5}\n"
        else:
            text += "🔴 SELL: `нет`\n"
        await q.edit_message_text(text, parse_mode="Markdown", reply_markup=main_keyboard())

    elif q.data == "ai_signal":
        await q.edit_message_text(bot.ai_signal_text(), parse_mode="Markdown", reply_markup=main_keyboard())

    elif q.data == "ai_settings":
        await q.edit_message_text(format_ai_settings_page(), parse_mode="Markdown", reply_markup=ai_settings_keyboard())

    elif q.data == "settings":
        await q.edit_message_text(format_settings_page(), parse_mode="Markdown", reply_markup=settings_keyboard())

    elif q.data == "set_sl":
        await q.edit_message_text(f"🔒 *Stop-Loss*\n\nТекущее: `-{CONFIG.get('global_sl_usdt', 25):.0f} USDT`\n\nОтправь: `/set global_sl_usdt 30`", parse_mode="Markdown", reply_markup=settings_keyboard())
    elif q.data == "set_tp":
        await q.edit_message_text(f"🎯 *Take-Profit*\n\nТекущее: `+{CONFIG.get('global_tp_usdt', 40):.0f} USDT`\n\nОтправь: `/set global_tp_usdt 50`", parse_mode="Markdown", reply_markup=settings_keyboard())
    elif q.data == "set_levels":
        await q.edit_message_text(f"📐 *Уровни*\n\nТекущее: `{CONFIG['grid_levels']}`\n\nОтправь: `/set grid_levels 25`", parse_mode="Markdown", reply_markup=settings_keyboard())
    elif q.data == "set_atr":
        await q.edit_message_text(f"📊 *ATR Множитель*\n\nТекущее: `{CONFIG.get('atr_multiplier', 2.5)}`\n\nОтправь: `/set atr_multiplier 3.0`", parse_mode="Markdown", reply_markup=settings_keyboard())
    elif q.data == "set_balpct":
        await q.edit_message_text(f"💰 *% Баланса*\n\nТекущее: `{CONFIG['max_balance_pct']*100:.0f}%`\n\nОтправь: `/set max_balance_pct 0.50`", parse_mode="Markdown", reply_markup=settings_keyboard())
    elif q.data == "set_maxqty":
        await q.edit_message_text(f"📦 *Макс qty*\n\nТекущее: `{CONFIG['max_qty']}` SOL\n\nОтправь: `/set max_qty 1.0`", parse_mode="Markdown", reply_markup=settings_keyboard())

    elif q.data == "set_ai_role":
        await q.edit_message_text(f"🎭 *AI Роль*\n\nТекущее: `{CONFIG.get('ai_role', '—')}`\n\nОтправь: `/set ai_role профессиональный трейдер`", parse_mode="Markdown", reply_markup=ai_settings_keyboard())
    elif q.data == "set_ai_style":
        await q.edit_message_text(f"🎯 *AI Стиль*\n\nТекущее: `{CONFIG.get('ai_style', '—')}`\n\nОтправь: `/set ai_style скальпинг`", parse_mode="Markdown", reply_markup=ai_settings_keyboard())
    elif q.data == "set_ai_risk":
        await q.edit_message_text(f"⚠️ *AI Риск*\n\nТекущее: `{CONFIG.get('ai_risk', '—')}`\n\nОтправь: `/set ai_risk агрессивный`", parse_mode="Markdown", reply_markup=ai_settings_keyboard())
    elif q.data == "set_btc_weight":
        await q.edit_message_text(f"📊 *Вес BTC корреляции*\n\nТекущее: `{CONFIG.get('ai_btc_corr_weight', 0.5)}`\n\nОтправь: `/set ai_btc_corr_weight 0.7`", parse_mode="Markdown", reply_markup=ai_settings_keyboard())
    elif q.data == "set_ai_freq":
        await q.edit_message_text(f"⏱ *Частота AI анализа*\n\nТекущее: `{CONFIG.get('ai_analysis_interval', 900)}с`\n\nОтправь: `/set ai_analysis_interval 600`", parse_mode="Markdown", reply_markup=ai_settings_keyboard())
    elif q.data == "set_adaptive":
        cur = CONFIG.get("ai_adaptive_weights", True)
        CONFIG["ai_adaptive_weights"] = not cur
        await q.edit_message_text(f"⚖️ Адаптивные веса: `{'ВКЛ' if CONFIG['ai_adaptive_weights'] else 'ВЫКЛ'}`", parse_mode="Markdown", reply_markup=ai_settings_keyboard())

    elif q.data == "back_main":
        await q.edit_message_text("⬅️ *Главное меню*", parse_mode="Markdown", reply_markup=main_keyboard())


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений для кнопок"""
    if not auth(update): return
    text = update.message.text
    if text == "📊 Статус":
        await update.message.reply_text(bot.status_text(), parse_mode="Markdown", reply_markup=persistent_keyboard())
    elif text == "🧠 AI Сигнал":
        await update.message.reply_text(bot.ai_signal_text(), parse_mode="Markdown", reply_markup=persistent_keyboard())
    elif text == "💰 Баланс":
        balance = bot.get_balance()
        avail = bot.get_available_balance()
        unrealized = bot.get_unrealized_pnl()
        price = bot.get_price()

        if "-SWAP" in CONFIG["symbol"]:
            try:
                r = bot.account_api.get_positions(instType="SWAP", instId=CONFIG["symbol"])
                positions_text = ""
                if r.get("code") == "0":
                    positions = r.get("data", [])
                    if positions:
                        for pos in positions:
                            sz = float(pos.get("pos", 0))
                            if sz > 0:
                                ps = pos.get("posSide", "net")
                                avg_px = float(pos.get("avgPx", 0))
                                upl = float(pos.get("upl", 0))
                                direction = "🟢 Лонг" if ps in ("long", "net") else "🔴 Шорт"
                                positions_text += f"\n{direction}: `{sz}` @ `{avg_px:.2f}` | UPL: `{upl:+.4f}`"
                    else:
                        positions_text = "\nНет открытых позиций"
            except Exception:
                positions_text = "\nНе удалось загрузить позиции"

            await update.message.reply_text(
                f"💰 *Баланс (фьючерсы)*\n\n"
                f"💵 USDT: `{balance:.2f}`\n"
                f"💳 Доступно: `{avail:.2f} USDT`\n"
                f"💲 Цена: `{price:.2f}`\n"
                f"📊 Позиции:{positions_text}\n\n"
                f"📈 Realized PnL: `{bot.realized_pnl:+.4f} USDT`\n"
                f"📊 Unrealized PnL: `{unrealized:+.4f} USDT`\n"
                f"💎 *Итого PnL: `{bot.realized_pnl + unrealized:+.4f} USDT`*",
                parse_mode="Markdown", reply_markup=persistent_keyboard())
        else:
            sol_bal = bot.get_sol_balance()
            holdings = bot.get_spot_holdings()
            await update.message.reply_text(
                f"💰 *Баланс (спот)*\n\n"
                f"💵 USDT: `{balance:.2f}`\n"
                f"💳 Доступно: `{avail:.2f} USDT`\n"
                f"🪙 SOL в кошельке: `{sol_bal:.2f}`\n"
                f"📊 SOL в ордерах: `{holdings:.2f}`\n"
                f"💲 Цена: `{price:.2f}`\n"
                f"💎 Стоимость SOL: `{sol_bal * price:.2f} USDT`\n"
                f"💎 *Итого PnL: `{bot.realized_pnl + unrealized:+.4f} USDT`*",
                parse_mode="Markdown", reply_markup=persistent_keyboard())
    elif text == "📈 Backtest":
        report = bot.backtester.run_analysis(7)
        await update.message.reply_text(report, parse_mode="Markdown", reply_markup=persistent_keyboard())
    elif text == "⚙️ Настройки":
        await update.message.reply_text(format_settings_page(), parse_mode="Markdown", reply_markup=settings_keyboard())
    elif text == "🚨 Стоп":
        bot.emergency_stop()
        await update.message.reply_text("🚨 *ЭКСТРЕННАЯ ОСТАНОВКА!*\n\n❌ Ордера отменены\n📉 Позиции закрыты", parse_mode="Markdown", reply_markup=persistent_keyboard())


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    def tg_notify(msg: str):
        if not ALLOWED_CHAT_IDS:
            return
        async def _send():
            try:
                await app.bot.send_message(chat_id=ALLOWED_CHAT_IDS[0], text=msg, parse_mode="Markdown")
            except Exception as e:
                log.warning(f"Telegram notify error: {e}")
        try:
            loop = app.loop
            asyncio.run_coroutine_threadsafe(_send(), loop)
        except Exception:
            pass

    bot.set_tg_notify(tg_notify)
    bot.set_tg_app(app)

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("ai", cmd_ai))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("set", cmd_set))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("sync", cmd_sync))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    async def post_init(application):
        await application.bot.set_my_commands([
            ("start", "Главное меню"),
            ("status", "Статус бота"),
            ("ai", "AI сигнал"),
            ("backtest", "Точность AI"),
            ("help", "Справка"),
            ("set", "Изменить параметр"),
            ("stop", "Остановить бота"),
            ("sync", "Синхронизация с биржей"),
        ])

    app.post_init = post_init
    bot.start()

    # Синхронизация существующих ордеров и позиций при запуске
    log.info("🔄 Запуск синхронизации с биржей...")
    time.sleep(2)  # Даем боту 2 секунды на инициализацию
    sync_result = bot.sync_existing_orders_and_positions()

    # Добавляем информацию о синхронизации в лог запуска
    if sync_result["orders"] > 0 or sync_result["positions"] > 0:
        log.info(f"✅ Обнаружено: {sync_result['orders']} ордеров, {sync_result['positions']} позиций")
        log.info("⚠️ Бот будет работать с существующими ордерами/позициями")
    else:
        log.info("✅ Бот запущен без существующих ордеров/позиций")

    log.info("=" * 55)
    log.info("  GRID BOT V3 (Multi-AI) — OKX Testnet — запущен!")
    log.info("=" * 55)

    # Запускаем polling с обработкой конфликтов и авто-перезапуском
    while True:
        try:
            app.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
            )
            break  # Если polling завершился корректно — выходим
        except Exception as e:
            err_msg = str(e)
            if "Conflict" in err_msg or "terminated by other" in err_msg:
                log.error(f"🔴 Конфликт Telegram! Другой экземпляр бота запущен. Жду 60 сек...")
                time.sleep(60)
            elif "Server disconnected" in err_msg or "network" in err_msg.lower():
                log.warning(f"⚠️ Сетевая ошибка Telegram: {e}. Перезапуск через 15 сек...")
                time.sleep(15)
            else:
                log.error(f"🔴 Ошибка Telegram polling: {e}. Перезапуск через 30 сек...")
                time.sleep(30)


if __name__ == "__main__":
    MAX_RESTARTS = 10
    restart_count = 0

    while True:
        try:
            main()
            log.info("✅ Бот остановлен корректно")
            break
        except KeyboardInterrupt:
            log.info("🛑 Остановка по Ctrl+C")
            try:
                bot.stop()
            except Exception:
                pass
            break
        except Exception as e:
            restart_count += 1
            log.error(f"🔴 Бот упал: {e} (попытка {restart_count}/{MAX_RESTARTS})")
            if restart_count >= MAX_RESTARTS:
                log.error("❌ Достигнут лимит перезапусков. Остановка.")
                break
            wait_time = min(30 * restart_count, 120)
            log.info(f"🔄 Перезапуск через {wait_time} сек...")
            time.sleep(wait_time)
            # Сбрасываем состояние бота
            try:
                bot.running = False
            except Exception:
                pass
            bot = GridBotV3()
