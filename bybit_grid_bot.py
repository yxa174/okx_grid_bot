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
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from config import (
    BYBIT_API_KEY, BYBIT_API_SECRET,
    TELEGRAM_TOKEN, ALLOWED_CHAT_IDS, CONFIG
)

# ── Константы биржи ───────────────────────────────────────────────
QTY_STEP   = 0.1   # минимальный шаг qty для SOLUSDT
MIN_QTY    = 0.1
MAX_ORDERS = 50    # лимит открытых ордеров

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


def round_qty(qty: float) -> float:
    """Округляет qty до ближайшего допустимого шага (0.1 для SOL)"""
    steps = round(qty / QTY_STEP)
    result = max(MIN_QTY, steps * QTY_STEP)
    return round(result, 1)


def round_price(price: float) -> float:
    """Округляет цену до 2 знаков"""
    return round(price, 2)


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

    def set_tg_notify(self, fn):
        self._tg_notify = fn

    def notify(self, msg: str):
        log.info(msg)
        if self._tg_notify:
            self._tg_notify(msg)

    # ── Биржа ─────────────────────────────────────────────────────

    def get_price(self) -> float:
        r = self.client.get_tickers(category="linear", symbol=CONFIG["symbol"])
        return float(r["result"]["list"][0]["markPrice"])

    def get_balance(self) -> float:
        try:
            r = self.client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            val = r["result"]["list"][0].get("totalWalletBalance", "0")
            return float(val) if val else 0.0
        except Exception as e:
            log.error(f"Ошибка баланса: {e}")
            return 0.0

    def get_position_size(self) -> float:
        """Возвращает размер LONG позиции"""
        try:
            r = self.client.get_positions(category="linear", symbol=CONFIG["symbol"])
            for p in r["result"]["list"]:
                if int(p.get("positionIdx", 0)) == 1:
                    return float(p.get("size", 0))
        except Exception as e:
            log.error(f"Ошибка позиции: {e}")
        return 0.0

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

    def get_open_order_count(self) -> int:
        try:
            r = self.client.get_open_orders(category="linear", symbol=CONFIG["symbol"])
            return len(r["result"]["list"])
        except Exception:
            return MAX_ORDERS

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
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        return df

    def setup_account(self):
        sym = CONFIG["symbol"]
        lev = str(CONFIG["leverage"])
        try:
            self.client.switch_position_mode(category="linear", symbol=sym, mode=3)
        except Exception:
            pass
        try:
            self.client.set_leverage(
                category="linear", symbol=sym,
                buyLeverage=lev, sellLeverage=lev
            )
        except Exception:
            pass

    # ── Расчёт диапазона по ATR ───────────────────────────────────

    def calc_atr_range(self, price: float) -> tuple[float, float]:
        """
        Считает диапазон сетки на основе ATR(14) на 4h свечах.
        Lower = price - multiplier*ATR
        Upper = price + multiplier*ATR
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

            lower = round_price(price - mult * atr)
            upper = round_price(price + mult * atr)

            log.info(f"📐 ATR={atr:.2f} → диапазон {lower} — {upper} (×{mult})")
            return lower, upper
        except Exception as e:
            log.error(f"Ошибка ATR: {e}")
            # Фоллбэк: ±8% от цены
            return round_price(price * 0.92), round_price(price * 1.08)

    def build_grid(self, lower: float, upper: float) -> list[float]:
        """Строит список уровней сетки"""
        n = CONFIG["grid_levels"]
        step = (upper - lower) / n
        return [round_price(lower + i * step) for i in range(n + 1)]

    # ── Ордера ────────────────────────────────────────────────────

    def _qty_for_price(self, price: float) -> float:
        """Вычисляет qty для одного ордера"""
        balance = self.get_balance()
        n = CONFIG["grid_levels"]

        # Используем не более max_balance_pct% баланса на все ордера
        budget_per_order = (balance * CONFIG["max_balance_pct"]) / n * CONFIG["leverage"]
        qty = budget_per_order / price

        qty = min(qty, CONFIG["max_qty"])
        qty = max(qty, MIN_QTY)
        return round_qty(qty)

    def place_buy(self, price: float) -> str | None:
        """Размещает лимитный BUY ордер"""
        if self.get_open_order_count() >= MAX_ORDERS:
            return None

        qty = self._qty_for_price(price)
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
        self._cancelled_ids.update(self.active_orders.keys())
        try:
            self.client.cancel_all_orders(
                category="linear", symbol=CONFIG["symbol"]
            )
        except Exception:
            pass
        self.active_orders.clear()
        log.info("🗑 Все ордера отменены")

    # ── Построение сетки ──────────────────────────────────────────

    def place_grid(self, price: float):
        """
        Строит полную лонг-сетку:
        - BUY ордера на всех уровнях НИЖЕ текущей цены
        - Уровни ВЫШЕ текущей цены пока пустые (туда встанут SELL после исполнения BUY)
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

        self.notify(
            f"📐 Сетка построена: {placed} BUY ордеров\n"
            f"📊 Диапазон: {self.lower:.2f} — {self.upper:.2f}\n"
            f"💰 Баланс: {self.get_balance():.2f} USDT"
        )

    # ── Trailing Up: сдвиг сетки вверх ───────────────────────────

    def trailing_up(self, price: float):
        """
        Если цена выросла выше верхней границы:
        Сдвигаем диапазон вверх и перестраиваем сетку.
        Сохраняем открытые SELL ордера (они уже в прибыли).
        """
        shift = (self.upper - self.lower) * 0.5  # сдвигаем на половину диапазона
        self.lower = round_price(self.lower + shift)
        self.upper = round_price(self.upper + shift)
        self.grid_levels = self.build_grid(self.lower, self.upper)

        self.notify(
            f"📈 Trailing UP: сетка сдвинута вверх\n"
            f"📊 Новый диапазон: {self.lower:.2f} — {self.upper:.2f}"
        )
        self.place_grid(price)

    def trailing_down(self, price: float):
        """
        Если цена упала ниже нижней границы:
        Сдвигаем диапазон вниз.
        """
        shift = (self.upper - self.lower) * 0.5
        self.lower = round_price(self.lower - shift)
        self.upper = round_price(self.upper - shift)
        self.grid_levels = self.build_grid(self.lower, self.upper)

        self.notify(
            f"📉 Trailing DOWN: сетка сдвинута вниз\n"
            f"📊 Новый диапазон: {self.lower:.2f} — {self.upper:.2f}"
        )
        self.place_grid(price)

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

        while self.running:
            try:
                time.sleep(CONFIG["check_interval"])
                if not self.running:
                    break

                price = self.get_price()
                self.last_price = price

                # Проверяем глобальные защиты
                if self.check_global_stops():
                    self.running = False
                    self.cancel_all()
                    self.close_all_positions()
                    break

                # Проверяем исполненные ордера
                self.check_filled()

                # Trailing: сдвиг сетки если цена вышла за границы
                if price > self.upper:
                    self.notify(f"📈 Цена {price:.2f} выше верхней границы {self.upper:.2f}")
                    self.trailing_up(price)

                elif price < self.lower:
                    self.notify(f"📉 Цена {price:.2f} ниже нижней границы {self.lower:.2f}")
                    self.trailing_down(price)

                # Пополнение пустых слотов раз в ~5 минут
                open_cnt = self.get_open_order_count()
                if open_cnt < 3:
                    log.info(f"⚠️ Мало ордеров ({open_cnt}), пополняю сетку")
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
        asyncio.run_coroutine_threadsafe(_send(), app.event_loop)

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