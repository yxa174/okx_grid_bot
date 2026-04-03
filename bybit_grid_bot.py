"""
BYBIT TESTNET — LONG-ONLY GRID BOT v2
======================================
Архитектура основана на изученных лучших практиках:

1. ТОЛЬКО ЛОНГ-СЕТКА (Long Grid)
   - Бот покупает ниже текущей цены, продаёт выше
   - Нет шортов → нет риска бесконечного убытка при росте
   - Каждый исполненный Buy автоматически получает Sell на следующем уровне

2. ДИНАМИЧЕСКИЙ ДИАПАЗОН (Auto-Range)
   - Диапазон строится по ATR(14) на 4h свечах
   - Lower = текущая цена - 2*ATR, Upper = текущая цена + 2*ATR
   - Пересчитывается при выходе цены за границы

3. TRAILING UP (Следование вверх)
   - Если цена растёт и выходит за верхнюю границу:
     отменяем самый нижний ордер, добавляем новый сверху
   - Сетка следует за трендом, не теряя позиции

4. ГЛОБАЛЬНЫЙ STOP-LOSS БОТА
   - Если unrealized PnL < -X USDT — закрываем всё и стоп
   - Защита от катастрофических потерь

5. ГЛОБАЛЬНЫЙ TAKE-PROFIT БОТА
   - Если realized PnL > +Y USDT — закрываем всё и стоп
   - Фиксируем прибыль

6. ПРАВИЛЬНЫЙ QTY
   - Всегда кратен минимальному шагу биржи (0.1 для SOLUSDT)
   - Автоматически считается из баланса
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from config import (
    BYBIT_API_KEY, BYBIT_API_SECRET,
    TELEGRAM_TOKEN, ALLOWED_CHAT_IDS, CONFIG
)

# ── Константы биржи ───────────────────────────────────────────────
# Используем значения из CONFIG
MAX_ORDERS = 50    # лимит открытых ордеров
MIN_QTY = 0.1      # минимальное количество для ордера

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("grid_bot_v2.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("GridBotV2")


def retry_api(retries=3, delay=2):
    """Декоратор для повторных попыток при сетевых ошибках"""
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
    """Округляет qty до ближайшего допустимого шага"""
    step = CONFIG.get("qty_step", 0.1)
    min_qty = CONFIG.get("min_qty", 0.1)
    steps = round(qty / step)
    result = max(min_qty, steps * step)
    return round(result, 2)


def round_price(price: float) -> float:
    """Округляет цену согласно настройкам"""
    decimals = CONFIG.get("price_decimals", 2)
    return round(max(0.01, price), decimals)



# ══════════════════════════════════════════════════════════════════
#  LSTM / AI ANALYZER
# ══════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class LSTMModel(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            self.gru  = torch.nn.GRU(input_size, hidden_size, num_layers,
                                      batch_first=True,
                                      dropout=dropout if num_layers > 1 else 0)
            self.bn   = torch.nn.BatchNorm1d(hidden_size)
            self.drop = torch.nn.Dropout(dropout)
            self.fc1  = torch.nn.Linear(hidden_size, 32)
            self.relu = torch.nn.ReLU()
            self.fc2  = torch.nn.Linear(32, 1)
            self.sig  = torch.nn.Sigmoid()

        def forward(self, x):
            out, _ = self.gru(x)
            out = self.bn(out[:, -1, :])
            out = self.drop(self.relu(self.fc1(out)))
            return self.sig(self.fc2(out))


class AIAnalyzer:
    """
    LSTM ансамбль + технические индикаторы.
    Используется для:
    - Подтверждения входа в позицию (не ставим BUY при сильном медвежьем сигнале)
    - Динамической корректировки qty (больше на бычьем сигнале)
    - Паузы при экстремальных сигналах
    """
    def __init__(self, model_path="lstm_ensemble.pt"):
        self.price_history  = deque(maxlen=200)
        self.volume_history = deque(maxlen=200)
        self.feature_buffer = deque(maxlen=200)
        self.ensemble       = []
        self.seq_len        = 60
        self.scaler         = None
        self.n_models       = 0
        self._load_ensemble(model_path)

    def _load_ensemble(self, path):
        if not TORCH_AVAILABLE:
            log.info("PyTorch недоступен — LSTM отключён")
            return
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.seq_len  = ck.get("seq_len", 60)
            self.scaler   = ck["scaler"]
            self.n_models = ck["n_models"]
            input_size    = ck["input_size"]
            for i in range(self.n_models):
                cfg = ck["model_configs"][i]
                m = LSTMModel(input_size, cfg["hidden"], cfg["layers"], cfg["dropout"])
                m.load_state_dict(ck["models"][i])
                m.eval()
                self.ensemble.append(m)
            log.info(f"🧠 LSTM ансамбль загружен: {self.n_models} моделей")
        except Exception as e:
            log.warning(f"LSTM не загружен (работаю без него): {e}")

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
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi_series = 100 - (100 / (1 + gain / (loss + 1e-10)))
        rsi   = rsi_series.iloc[-1]
        stoch_rsi = ((rsi_series - rsi_series.rolling(14).min()) /
                     (rsi_series.rolling(14).max() - rsi_series.rolling(14).min() + 1e-10)).iloc[-1]
        williams_r = ((c.rolling(14).max() - c) /
                      (c.rolling(14).max() - c.rolling(14).min() + 1e-10)).iloc[-1]

        tp     = c
        sma_tp = tp.rolling(20).mean()
        mad    = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        cci    = ((tp - sma_tp) / (0.015 * mad + 1e-10) / 200).iloc[-1]

        ema12     = c.ewm(span=12).mean()
        ema26     = c.ewm(span=26).mean()
        macd      = ema12 - ema26
        macd_hist = ((macd - macd.ewm(span=9).mean()) / (c + 1e-10)).iloc[-1]

        sma20    = c.rolling(20).mean()
        std20    = c.rolling(20).std()
        bb_width = (4 * std20 / (sma20 + 1e-10)).iloc[-1]
        bb_pos   = ((c - (sma20 - 2 * std20)) / (4 * std20 + 1e-10)).iloc[-1]

        obv      = (np.sign(c.diff()) * v).fillna(0).cumsum()
        obv_norm = (obv / (obv.rolling(50).std() + 1e-10)).iloc[-1]
        momentum = (c / (c.shift(10) + 1e-10) - 1).iloc[-1]

        tr = pd.concat([(c - c).abs(),
                        (c - c.shift()).abs(),
                        (c - c.shift()).abs()], axis=1).max(axis=1)
        atr_ratio = (tr.rolling(14).mean() / (c + 1e-10)).iloc[-1]
        ma_trend  = ((c.rolling(20).mean() - c.rolling(50).mean()) / c).iloc[-1]
        vol_ratio = (v / (v.rolling(20).mean() + 1e-10)).iloc[-1]

        return [ret1, ret3, ret6, rsi, stoch_rsi, williams_r, cci,
                macd_hist, bb_width, bb_pos, obv_norm, momentum,
                atr_ratio, ma_trend, vol_ratio]

    def get_lstm_signal(self) -> float:
        """Возвращает вероятность роста от 0.0 до 1.0 (0.5 = нейтрально)"""
        if not TORCH_AVAILABLE or not self.ensemble or len(self.feature_buffer) < self.seq_len:
            return 0.5
        arr = np.array(list(self.feature_buffer)[-self.seq_len:], dtype=np.float32)
        if self.scaler:
            arr = self.scaler.transform(arr)
        x = torch.FloatTensor(arr).unsqueeze(0)
        probs = []
        with torch.no_grad():
            for m in self.ensemble:
                probs.append(float(m(x).item()))
        return round(sum(probs) / len(probs), 3)

    def get_indicators(self) -> dict:
        if len(self.price_history) < 50:
            return {}
        c     = pd.Series(list(self.price_history))
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = 100 - (100 / (1 + gain / (loss + 1e-10)))
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        macd  = ema12 - ema26
        sma   = c.rolling(20).mean()
        std   = c.rolling(20).std()
        return {
            "rsi":       round(float(rsi.iloc[-1]), 2),
            "macd_hist": round(float((macd - macd.ewm(span=9).mean()).iloc[-1]), 2),
            "bb_upper":  round(float((sma + 2 * std).iloc[-1]), 2),
            "bb_lower":  round(float((sma - 2 * std).iloc[-1]), 2),
            "ma20":      round(float(sma.iloc[-1]), 2),
            "ma50":      round(float(c.rolling(50).mean().iloc[-1]), 2),
        }

    def get_signal(self) -> dict:
        """
        Возвращает итоговый сигнал с учётом LSTM + индикаторов.
        signal: STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL
        """
        ind = self.get_indicators()
        if not ind:
            return {"signal": "NEUTRAL", "score": 0, "lstm": 0.5, "indicators": {}}

        score = 0
        rsi = ind["rsi"]

        if rsi < 25:    score += 3
        elif rsi < 35:  score += 2
        elif rsi < 45:  score += 1
        elif rsi > 75:  score -= 3
        elif rsi > 65:  score -= 2
        elif rsi > 55:  score -= 1

        if ind["macd_hist"] > 0: score += 1
        else:                    score -= 1

        lstm = self.get_lstm_signal()
        if lstm > 0.65:   score += 2
        elif lstm > 0.55: score += 1
        elif lstm < 0.35: score -= 2
        elif lstm < 0.45: score -= 1

        if score >= 4:    signal = "STRONG_BUY"
        elif score >= 2:  signal = "BUY"
        elif score <= -4: signal = "STRONG_SELL"
        elif score <= -2: signal = "SELL"
        else:             signal = "NEUTRAL"

        return {
            "signal":     signal,
            "score":      score,
            "lstm":       lstm,
            "indicators": ind,
        }

# ══════════════════════════════════════════════════════════════════
#  GRID BOT V2 — LONG ONLY
# ══════════════════════════════════════════════════════════════════

class GridBotV2:
    def __init__(self):
        self.client = HTTP(
            testnet=True,
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET
        )
        self.lock            = threading.Lock()
        self.running         = False
        self.start_time      = None
        self.start_balance   = 0.0
        self.last_price      = 0.0

        # Сетка
        self.lower           = 0.0
        self.upper           = 0.0
        self.grid_levels     = []       # список цен уровней
        self.active_orders   = {}       # {order_id: {price, type, qty}}
        self._cancelled_ids  = set()

        # Статистика
        self.realized_pnl    = 0.0
        self.trades_count    = 0
        self.closed_trades   = []

        self._thread         = None
        self._tg_notify      = None
        self.ai              = AIAnalyzer("lstm_ensemble.pt")
        self.last_signal     = {"signal": "NEUTRAL", "score": 0, "lstm": 0.5, "indicators": {}}

    def set_tg_notify(self, fn):
        self._tg_notify = fn

    def notify(self, msg: str):
        log.info(msg)
        if self._tg_notify:
            self._tg_notify(msg)

    # ── Биржа ─────────────────────────────────────────────────────

    def setup_account(self):
        """Настраивает аккаунт: плечо, режим маржи"""
        try:
            self.client.set_leverage(
                category="linear",
                symbol=CONFIG["symbol"],
                buyLeverage=str(CONFIG["leverage"]),
                sellLeverage=str(CONFIG["leverage"]),
            )
            log.info(f"✅ Плечо установлено: {CONFIG['leverage']}x")
        except Exception as e:
            log.warning(f"Не удалось установить плечо (возможно уже): {e}")

        try:
            self.client.switch_position_mode(
                category="linear",
                mode=3,  # BothSide (Hedge mode)
            )
            log.info("✅ Режим позиций: BothSide")
        except Exception as e:
            log.warning(f"Не удалось установить режим позиций (возможно уже): {e}")

    @retry_api()
    def get_price(self) -> float:
        r = self.client.get_tickers(category="linear", symbol=CONFIG["symbol"])
        return float(r["result"]["list"][0]["markPrice"])

    @retry_api()
    def get_balance(self) -> float:
        try:
            r = self.client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            val = r["result"]["list"][0].get("totalWalletBalance", "0")
            return float(val) if val else 0.0
        except Exception as e:
            log.error(f"Ошибка баланса: {e}")
            return 0.0

    @retry_api()
    def get_available_balance(self) -> float:
        """Возвращает ДОСТУПНЫЙ баланс (без учёта занятой маржи)"""
        try:
            r = self.client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            val = r["result"]["list"][0].get("availableToWithdraw", "0")
            return float(val) if val else 0.0
        except Exception as e:
            log.error(f"Ошибка доступного баланса: {e}")
            return 0.0

    @retry_api()
    def get_position_size(self) -> float:
        """Возвращает размер LONG позиции"""
        try:
            r = self.client.get_positions(category="linear", symbol=CONFIG["symbol"])
            for p in r["result"]["list"]:
                return float(p.get("size", 0))
        except Exception as e:
            log.error(f"Ошибка позиции: {e}")
        return 0.0

    @retry_api()
    def get_unrealized_pnl(self) -> float:
        try:
            r = self.client.get_positions(category="linear", symbol=CONFIG["symbol"])
            pnl = 0.0
            for p in r["result"]["list"]:
                v = p.get("unrealisedPnl") or "0"
                pnl += float(v)
            return pnl
        except Exception:
            return 0.0

    @retry_api()
    def get_open_order_count(self) -> int:
        try:
            r = self.client.get_open_orders(category="linear", symbol=CONFIG["symbol"])
            return len(r["result"]["list"])
        except Exception:
            return MAX_ORDERS

    @retry_api()
    def get_klines(self, interval="240", limit=50) -> pd.DataFrame:
        """Получает свечи (default: 4h)"""
        r = self.client.get_kline(
            category="linear",
            symbol=CONFIG["symbol"],
            interval=interval,
            limit=limit
        )
        df = pd.DataFrame(
            r["result"]["list"],
            columns=["ts", "open", "high", "low", "close", "vol", "turnover"]
        )
        for col in ["open", "high", "low", "close", "vol"]:
            df[col] = df[col].astype(float)
        return df

    # ── Расчёт диапазона по ATR ───────────────────────────────────

    def calc_atr_range(self, price: float) -> tuple[float, float]:
        """
        Считает диапазон сетки на основе ATR(14) на 4h свечах.
        Ограничивает ширину сетки для стабильности.
        """
        try:
            df = self.get_klines(interval="240", limit=50)
            high = df["high"]
            low  = df["low"]
            close = df["close"]

            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs()
            ], axis=1).max(axis=1)

            atr = tr.rolling(14).mean().iloc[-1]
            mult = CONFIG.get("atr_multiplier", 2.5)

            # Базовый расчет
            half_width = atr * mult
            
            # Ограничиваем: не более X% от цены (из конфига)
            max_half = price * (CONFIG.get("max_range_pct", 0.20) / 2)
            half = min(half_width, max_half)
            
            lower = round_price(max(price * 0.4, price - half)) # не ниже 40% от цены
            upper = round_price(price + half)

            log.info(f"📐 ATR={atr:.2f} (mult={mult}) -> Range: {lower} - {upper} (width={((upper/lower-1)*100):.1f}%)")
            return lower, upper
        except Exception as e:
            log.error(f"Ошибка ATR: {e}")
            # Фоллбэк: ±5%
            return round_price(price * 0.95), round_price(price * 1.05)

    def build_grid(self, lower: float, upper: float) -> list[float]:
        """Строит список уровней сетки"""
        n = CONFIG["grid_levels"]
        step = (upper - lower) / n
        return [round_price(lower + i * step) for i in range(n + 1)]

    # ── Ордера ────────────────────────────────────────────────────

    def _qty_for_price(self, price: float) -> float:
        """Вычисляет qty для одного ордера на основе ДОСТУПНОГО баланса"""
        avail_balance = self.get_available_balance()
        n = CONFIG["grid_levels"]

        # Используем не более max_balance_pct% ДОСТУПНОГО баланса на все ордера
        budget_per_order = (avail_balance * CONFIG["max_balance_pct"]) / n * CONFIG["leverage"]
        qty = budget_per_order / price

        qty = min(qty, CONFIG["max_qty"])
        qty = max(qty, MIN_QTY)
        return round_qty(qty)

    def place_buy(self, price: float) -> str | None:
        """Размещает лимитный BUY ордер"""
        if self.get_open_order_count() >= MAX_ORDERS:
            return None

        # AI фильтр: не ставим BUY при сильном медвежьем сигнале
        sig = self.last_signal.get("signal", "NEUTRAL")
        if sig == "STRONG_SELL":
            log.info(f"🧠 AI STRONG_SELL — пропускаю BUY@{price}")
            return None

        # При медвежьем сигнале уменьшаем qty на 50%
        qty = self._qty_for_price(price)
        if sig == "SELL":
            qty = round_qty(qty * 0.5)

        order_usdt = qty * price
        if order_usdt < 5.0:
            log.warning(f"⚠️ Buy@{price}: стоимость {order_usdt:.2f} USDT < 5 USDT, пропускаю")
            return None

        try:
            r = self.client.place_order(
                category="linear",
                symbol=CONFIG["symbol"],
                side="Buy",
                orderType="Limit",
                qty=str(qty),
                price=str(price),
                timeInForce="GTC",
                positionIdx=1,
                reduceOnly=False,
            )
            oid = r["result"]["orderId"]
            self.active_orders[oid] = {"price": price, "type": "BUY", "qty": qty}
            log.info(f"🟢 BUY limit @ {price} qty={qty} id={oid[:8]}..")
            return oid
        except Exception as e:
            log.error(f"Ошибка BUY@{price}: {e}")
            return None

    def place_sell(self, price: float, qty: float) -> str | None:
        """Размещает лимитный SELL ордер (закрытие лонга)"""
        if self.get_open_order_count() >= MAX_ORDERS:
            return None

        qty = round_qty(qty)
        order_usdt = qty * price
        if order_usdt < 5.0:
            return None

        try:
            r = self.client.place_order(
                category="linear",
                symbol=CONFIG["symbol"],
                side="Sell",
                orderType="Limit",
                qty=str(qty),
                price=str(price),
                timeInForce="GTC",
                positionIdx=1,
                reduceOnly=True,
            )
            oid = r["result"]["orderId"]
            self.active_orders[oid] = {"price": price, "type": "SELL", "qty": qty}
            log.info(f"🔴 SELL limit @ {price} qty={qty} id={oid[:8]}..")
            return oid
        except Exception as e:
            log.error(f"Ошибка SELL@{price}: {e}")
            return None

    def close_all_positions(self):
        """Закрывает все открытые лонг-позиции по рынку"""
        size = self.get_position_size()
        if size <= 0:
            return
        try:
            self.client.place_order(
                category="linear",
                symbol=CONFIG["symbol"],
                side="Sell",
                orderType="Market",
                qty=str(round_qty(size)),
                positionIdx=1,
                reduceOnly=True,
                timeInForce="IOC",
            )
            log.info(f"🔒 Закрыта позиция {size} SOL по рынку")
        except Exception as e:
            log.error(f"Ошибка закрытия позиции: {e}")

    def cancel_all(self):
        """Отменяет все ордера"""
        try:
            self.client.cancel_all_orders(
                category="linear", symbol=CONFIG["symbol"]
            )
        except Exception:
            pass
        self._cancelled_ids.update(self.active_orders.keys())
        self.active_orders.clear()
        log.info("🗑 Все ордера отменены")

    # ── Построение сетки ──────────────────────────────────────────

    def place_grid(self, price: float):
        """
        Строит полную лонг-сетку:
        - BUY ордера на всех уровнях НИЖЕ текущей цены
        - SELL ордера для существующей позиции на уровнях ВЫШЕ цены
        """
        self.cancel_all()
        time.sleep(0.5)

        buy_levels  = [p for p in self.grid_levels if p < price * 0.9995]
        placed = 0

        # Ставим BUY на уровни ниже цены
        for lvl in sorted(buy_levels, reverse=True):  # от ближайших к дальним
            if self.get_open_order_count() >= MAX_ORDERS:
                break
            oid = self.place_buy(lvl)
            if oid:
                placed += 1
            time.sleep(0.05)

        # Если есть позиция — ставим SELL ордера для неё
        pos_size = self.get_position_size()
        if pos_size > 0:
            sell_levels = [p for p in self.grid_levels if p > price * 1.0005]
            sell_placed = 0
            qty_per_sell = round_qty(pos_size / max(len(sell_levels), 1))
            for lvl in sorted(sell_levels):
                if self.get_open_order_count() >= MAX_ORDERS:
                    break
                oid = self.place_sell(lvl, qty_per_sell)
                if oid:
                    sell_placed += 1
                time.sleep(0.05)
            self.notify(
                f"📐 Сетка построена: {placed} BUY, {sell_placed} SELL ордеров\n"
                f"📊 Диапазон: {self.lower:.2f} — {self.upper:.2f}\n"
                f"💰 Баланс: {self.get_balance():.2f} USDT\n"
                f"📍 Позиция: {pos_size:.1f} SOL"
            )
        else:
            self.notify(
                f"📐 Сетка построена: {placed} BUY ордеров\n"
                f"📊 Диапазон: {self.lower:.2f} — {self.upper:.2f}\n"
                f"💰 Баланс: {self.get_balance():.2f} USDT"
            )

    # ── Trailing Up: сдвиг сетки вверх ───────────────────────────

    def _rebuild_grid_around_price(self, price: float):
        """Пересобирает сетку вокруг текущей цены, сохраняя существующие ордера"""
        old_lower, old_upper = self.lower, self.upper
        self.lower, self.upper = self.calc_atr_range(price)
        self.grid_levels = self.build_grid(self.lower, self.upper)

        # Отменяем только те BUY ордера, которые вышли за новый диапазон
        to_cancel = []
        for oid, order in self.active_orders.items():
            if order["type"] == "BUY" and (order["price"] < self.lower or order["price"] > self.upper):
                to_cancel.append(oid)

        for oid in to_cancel:
            try:
                self.client.cancel_order(category="linear", symbol=CONFIG["symbol"], orderId=oid)
                self.active_orders.pop(oid, None)
            except Exception:
                pass

        # Ставим новые BUY ордера на недостающих уровнях
        existing_prices = {o["price"] for o in self.active_orders.values() if o["type"] == "BUY"}
        buy_levels = [p for p in self.grid_levels if p < price * 0.9995 and p not in existing_prices]
        placed = 0
        for lvl in sorted(buy_levels, reverse=True):
            if self.get_open_order_count() >= MAX_ORDERS:
                break
            oid = self.place_buy(lvl)
            if oid:
                placed += 1
            time.sleep(0.05)

        return placed

    def trailing_up(self, price: float):
        """
        Если цена выросла выше верхней границы:
        Сдвигаем диапазон вверх и перестраиваем сетку.
        """
        placed = self._rebuild_grid_around_price(price)
        self.notify(
            f"📈 Trailing UP: сетка сдвинута вверх, {placed} новых BUY\n"
            f"📊 Новый диапазон: {self.lower:.2f} — {self.upper:.2f}"
        )

    def trailing_down(self, price: float):
        """
        Если цена упала ниже нижней границы:
        Сдвигаем диапазон вниз.
        """
        placed = self._rebuild_grid_around_price(price)
        self.notify(
            f"📉 Trailing DOWN: сетка сдвинута вниз, {placed} новых BUY\n"
            f"📊 Новый диапазон: {self.lower:.2f} — {self.upper:.2f}"
        )

    # ── Проверка исполненных ──────────────────────────────────────

    def check_filled(self):
        """
        Проверяет исполненные ордера.
        Если BUY исполнен → ставим SELL на следующий уровень вверх.
        Если SELL исполнен → ставим BUY обратно на этот уровень.
        """
        try:
            r = self.client.get_open_orders(
                category="linear", symbol=CONFIG["symbol"]
            )
            open_ids = {o["orderId"] for o in r["result"]["list"]}
        except Exception as e:
            log.error(f"Ошибка check_filled: {e}")
            return

        step = (self.upper - self.lower) / CONFIG["grid_levels"]

        for oid in list(self.active_orders):
            if oid in open_ids:
                continue  # ещё открыт

            order = self.active_orders.pop(oid)
            price = order["price"]
            otype = order["type"]
            qty   = order["qty"]

            # Пропускаем если сами отменили
            if oid in self._cancelled_ids:
                self._cancelled_ids.discard(oid)
                continue

            if otype == "BUY":
                # BUY исполнен — ставим SELL на следующий уровень вверх
                sell_price = round_price(price + step)
                self.trades_count += 1
                self.notify(f"✅ BUY исполнен @ {price:.2f} | SELL → {sell_price:.2f} | qty={qty}")

                # Ждём появления позиции
                for _ in range(20):
                    if self.get_position_size() > 0:
                        break
                    time.sleep(0.3)

                if sell_price <= self.upper:
                    self.place_sell(sell_price, qty)
                else:
                    log.info(f"SELL @ {sell_price:.2f} выходит за верхнюю границу, пропускаю")

            elif otype == "SELL":
                # SELL исполнен — фиксируем прибыль, ставим BUY обратно
                buy_price = round_price(price - step)
                pnl = round(step * qty, 4)
                self.realized_pnl += pnl
                self.trades_count += 1

                self.closed_trades.append({
                    "sell_price": price,
                    "buy_price":  buy_price,
                    "qty":        qty,
                    "pnl":        pnl,
                    "ts":         datetime.now().isoformat(),
                })

                self.notify(f"💰 SELL исполнен @ {price:.2f} | PnL: +{pnl:.4f} USDT | Итого: {self.realized_pnl:+.4f}")

                # Восстанавливаем BUY на том же уровне
                if buy_price >= self.lower:
                    time.sleep(0.2)
                    self.place_buy(buy_price)

    # ── Глобальные защиты ─────────────────────────────────────────

    def check_global_stops(self) -> bool:
        """
        Проверяет глобальный SL и TP бота.
        Возвращает True если нужно остановить бота.
        """
        unrealized = self.get_unrealized_pnl()
        balance    = self.get_balance()

        # Глобальный Stop-Loss: unrealized убыток > порога
        max_loss = CONFIG.get("global_sl_usdt", 30.0)
        if unrealized < -max_loss:
            self.notify(
                f"🚨 ГЛОБАЛЬНЫЙ СТОП-ЛОСС!\n"
                f"Unrealized PnL: {unrealized:.2f} USDT < -{max_loss} USDT\n"
                f"Закрываю все позиции и останавливаю бота."
            )
            return True

        # Глобальный Take-Profit: реализованная прибыль > порога
        max_profit = CONFIG.get("global_tp_usdt", 50.0)
        if self.realized_pnl >= max_profit:
            self.notify(
                f"🎯 ГЛОБАЛЬНЫЙ ТЕЙК-ПРОФИТ!\n"
                f"Realized PnL: {self.realized_pnl:.2f} USDT ≥ +{max_profit} USDT\n"
                f"Фиксирую прибыль и останавливаю бота."
            )
            return True

        # Защита баланса: если баланс упал ниже минимума
        min_balance = CONFIG.get("min_balance_usdt", 50.0)
        if balance < min_balance:
            self.notify(
                f"⚠️ Баланс {balance:.2f} USDT ниже минимума {min_balance} USDT\n"
                f"Останавливаю бота для защиты капитала."
            )
            return True

        return False

    # ── Основной цикл ─────────────────────────────────────────────

    def _loop(self):
        self.setup_account()
        self.start_balance = self.get_balance()

        # Загружаем историю цен для AI
        try:
            df15 = self.get_klines(interval="15", limit=200)
            closes = df15["close"].tolist()
            volumes = df15["vol"].tolist()
            for c, v in zip(reversed(closes), reversed(volumes)):
                self.ai.add_price(c, v)
            self.last_signal = self.ai.get_signal()
            log.info(f"🧠 AI прогрет: {self.last_signal['signal']} (LSTM={self.last_signal['lstm']:.2f})")
        except Exception as e:
            log.warning(f"Ошибка прогрева AI: {e}")

        price = self.get_price()
        self.last_price = price

        # Строим диапазон по ATR
        self.lower, self.upper = self.calc_atr_range(price)
        self.grid_levels = self.build_grid(self.lower, self.upper)

        self.notify(
            f"🚀 Grid Bot V2 запущен!\n"
            f"💰 Баланс: {self.start_balance:.2f} USDT\n"
            f"📍 Цена: {price:.2f}\n"
            f"📐 Сетка: {self.lower:.2f} — {self.upper:.2f}\n"
            f"📊 Уровней: {CONFIG['grid_levels']}\n"
            f"🛡 Глобальный SL: -{CONFIG.get('global_sl_usdt', 30):.0f} USDT\n"
            f"🎯 Глобальный TP: +{CONFIG.get('global_tp_usdt', 50):.0f} USDT"
        )

        self.place_grid(price)
        self.start_time = datetime.now()

        trailing_cooldown = 0

        while self.running:
            try:
                time.sleep(CONFIG["check_interval"])
                if not self.running:
                    break

                price = self.get_price()
                self.last_price = price
                try:
                    df = self.get_klines(interval="15", limit=1)
                    vol = float(df["vol"].iloc[-1])
                except Exception:
                    vol = 0.0
                self.ai.add_price(price, vol)
                self.last_signal = self.ai.get_signal()

                # Проверяем глобальные защиты
                if self.check_global_stops():
                    self.running = False
                    self.cancel_all()
                    self.close_all_positions()
                    break

                # Проверяем исполненные ордера
                self.check_filled()

                # Trailing: сдвиг сетки если цена вышла за границы (с кулдауном)
                if trailing_cooldown > 0:
                    trailing_cooldown -= 1
                elif price > self.upper:
                    self.notify(f"📈 Цена {price:.2f} выше верхней границы {self.upper:.2f}")
                    self.trailing_up(price)
                    trailing_cooldown = 30
                elif price < self.lower:
                    self.notify(f"📉 Цена {price:.2f} ниже нижней границы {self.lower:.2f}")
                    self.trailing_down(price)
                    trailing_cooldown = 30

                # Пополнение сетки если нет BUY ордеров и нет открытых ордеров вообще
                buy_cnt = sum(1 for o in self.active_orders.values() if o["type"] == "BUY")
                if buy_cnt == 0 and self.get_open_order_count() == 0:
                    log.info("⚠️ Нет BUY ордеров — пересобираю сетку")
                    self.lower, self.upper = self.calc_atr_range(price)
                    self.grid_levels = self.build_grid(self.lower, self.upper)
                    self.place_grid(price)

            except Exception as e:
                log.error(f"Ошибка в цикле: {e}")
                time.sleep(15)

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
        """Экстренная остановка: отменяет ордера и закрывает позиции"""
        self.running = False
        self.cancel_all()
        self.close_all_positions()
        self.notify("🛑 Экстренная остановка выполнена")

    def status_text(self) -> str:
        uptime = ""
        if self.start_time:
            d = datetime.now() - self.start_time
            h, m = divmod(int(d.total_seconds()), 3600)
            m, s = divmod(m, 60)
            uptime = f"{h}ч {m}м {s}с"

        buys  = sum(1 for o in self.active_orders.values() if o["type"] == "BUY")
        sells = sum(1 for o in self.active_orders.values() if o["type"] == "SELL")
        unrealized = self.get_unrealized_pnl()

        lines = [
            f"{'🟢 РАБОТАЕТ' if self.running else '🔴 ОСТАНОВЛЕН'}",
            "",
            "📊 *Сетка*",
            f"Пара: `{CONFIG['symbol']}`",
            f"Диапазон: `{self.lower:.2f} — {self.upper:.2f}`",
            f"Уровней: `{CONFIG['grid_levels']}`",
            "",
            "📈 *Статистика*",
            f"Uptime: `{uptime}`",
            f"Сделок: `{self.trades_count}`",
            f"PnL реализованный: `{self.realized_pnl:+.4f} USDT`",
            f"PnL нереализованный: `{unrealized:+.4f} USDT`",
            f"PnL итого: `{self.realized_pnl + unrealized:+.4f} USDT`",
            f"Баланс: `{self.get_balance():.2f} USDT`",
            "",
            "📋 *Ордера*",
            f"BUY: `{buys}` | SELL: `{sells}`",
            "",
            "🛡 *Защиты*",
            f"Глобальный SL: `-{CONFIG.get('global_sl_usdt', 30):.0f} USDT`",
            f"Глобальный TP: `+{CONFIG.get('global_tp_usdt', 50):.0f} USDT`",
        ]
        sig = self.last_signal
        ind = sig.get("indicators", {})
        emoji = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "NEUTRAL": "⚪", "SELL": "🔴", "STRONG_SELL": "🔴🔴"}.get(sig.get("signal", "NEUTRAL"), "⚪")
        lines += [
            "",
            "🧠 *AI Сигнал*",
            f"{emoji} `{sig.get('signal', '—')}` | LSTM: `{sig.get('lstm', 0.5):.2f}` | Score: `{sig.get('score', 0)}`",
        ]
        if ind:
            lines += [
                f"RSI: `{ind.get('rsi', '—')}` | MACD: `{ind.get('macd_hist', '—')}`",
                f"MA20: `{ind.get('ma20', '—')}` | MA50: `{ind.get('ma50', '—')}`",
            ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  TELEGRAM
# ══════════════════════════════════════════════════════════════════

bot = GridBotV2()


def auth(update: Update) -> bool:
    return not ALLOWED_CHAT_IDS or update.effective_chat.id in ALLOWED_CHAT_IDS


def keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ Старт",       callback_data="start"),
         InlineKeyboardButton("⏹ Стоп",         callback_data="stop")],
        [InlineKeyboardButton("🚨 Экстренный стоп", callback_data="estop")],
        [InlineKeyboardButton("📊 Статус",       callback_data="status"),
         InlineKeyboardButton("💰 Баланс",       callback_data="balance")],
        [InlineKeyboardButton("⚙️ Параметры",    callback_data="params")],
    ])


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    await update.message.reply_text(
        "🤖 *Grid Bot V2 — Long Only*\n\n"
        "Стратегия: покупаем на падении, продаём на росте.\n"
        "Нет шортов → нет риска при росте цены.\n\n"
        "Выбери действие:",
        parse_mode="Markdown", reply_markup=keyboard()
    )


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    await update.message.reply_text(
        bot.status_text(), parse_mode="Markdown", reply_markup=keyboard()
    )


async def cmd_set(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not auth(update): return
    args = ctx.args
    if len(args) != 2:
        await update.message.reply_text("Использование: `/set param value`", parse_mode="Markdown")
        return
    key, val = args[0], args[1]
    if key not in CONFIG:
        await update.message.reply_text(f"❌ Неизвестный параметр: `{key}`", parse_mode="Markdown")
        return
    try:
        if isinstance(CONFIG[key], bool):   CONFIG[key] = val.lower() in ("true", "1", "yes")
        elif isinstance(CONFIG[key], int):  CONFIG[key] = int(val)
        else:                               CONFIG[key] = float(val)
        await update.message.reply_text(
            f"✅ `{key}` = `{CONFIG[key]}`\n\nПерезапусти бот для применения.",
            parse_mode="Markdown"
        )
    except ValueError:
        await update.message.reply_text("❌ Неверное значение")


async def callback_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass

    if q.data == "start":
        ok = bot.start()
        await q.edit_message_text(
            "▶️ Бот запущен! Строю сетку..." if ok else "⚠️ Уже работает.",
            reply_markup=keyboard()
        )

    elif q.data == "stop":
        ok = bot.stop()
        await q.edit_message_text(
            "⏹ Бот остановлен. Позиции остаются открытыми." if ok else "Не запущен.",
            reply_markup=keyboard()
        )

    elif q.data == "estop":
        bot.emergency_stop()
        await q.edit_message_text(
            "🚨 Экстренная остановка!\nВсе ордера отменены, позиции закрыты.",
            reply_markup=keyboard()
        )

    elif q.data == "status":
        await q.edit_message_text(
            bot.status_text(), parse_mode="Markdown", reply_markup=keyboard()
        )

    elif q.data == "balance":
        unrealized = bot.get_unrealized_pnl()
        await q.edit_message_text(
            f"💰 *Баланс*\n\n"
            f"USDT: `{bot.get_balance():.2f}`\n"
            f"Цена SOL: `{bot.get_price():.2f}`\n"
            f"Позиция: `{bot.get_position_size():.1f} SOL`\n"
            f"PnL реализованный: `{bot.realized_pnl:+.4f} USDT`\n"
            f"PnL нереализованный: `{unrealized:+.4f} USDT`\n"
            f"PnL итого: `{bot.realized_pnl + unrealized:+.4f} USDT`",
            parse_mode="Markdown", reply_markup=keyboard()
        )

    elif q.data == "params":
        await q.edit_message_text(
            f"⚙️ *Параметры*\n\n"
            f"Пара: `{CONFIG['symbol']}`\n"
            f"Уровней: `{CONFIG['grid_levels']}`\n"
            f"Макс qty: `{CONFIG['max_qty']}`\n"
            f"ATR множитель: `{CONFIG.get('atr_multiplier', 2.5)}`\n"
            f"Плечо: `{CONFIG['leverage']}`\n"
            f"Глобальный SL: `-{CONFIG.get('global_sl_usdt', 30)} USDT`\n"
            f"Глобальный TP: `+{CONFIG.get('global_tp_usdt', 50)} USDT`\n\n"
            f"Изменить: `/set global_sl_usdt 50`",
            parse_mode="Markdown", reply_markup=keyboard()
        )


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    def tg_notify(msg: str):
        if not ALLOWED_CHAT_IDS:
            return
        async def _send():
            try:
                await app.bot.send_message(
                    chat_id=ALLOWED_CHAT_IDS[0], text=msg, parse_mode="Markdown"
                )
            except Exception as e:
                log.warning(f"Telegram notify error: {e}")
        try:
            loop = app.loop
            asyncio.run_coroutine_threadsafe(_send(), loop)
        except Exception:
            pass

    bot.set_tg_notify(tg_notify)

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("set",    cmd_set))
    app.add_handler(CallbackQueryHandler(callback_handler))

    # Автостарт при запуске из консоли
    bot.start()

    log.info("=" * 55)
    log.info("  GRID BOT V2 (Long Only) — запущен и торгует!")
    log.info("=" * 55)

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()