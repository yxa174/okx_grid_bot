"""
LSTM ENSEMBLE TRAINER для SOLUSDT
==================================
Скачивает историю свечей, готовит фичи, тренирует ансамбль моделей.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pybit.unified_trading import HTTP
from datetime import datetime
import time
import os

# ══════════════════════════════════════════════════════════════════
#  КОНФИГ
# ══════════════════════════════════════════════════════════════════

SYMBOL = "SOLUSDT"
INTERVAL = "15"  # 15-минутные свечи
TOTAL_CANDLES = 8000  # ~83 дня данных
SEQ_LEN = 60
N_MODELS = 3
EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LABEL_HORIZON = 12  # предсказываем движение через 12 свечей (3 часа)
SAVE_PATH = "lstm_ensemble.pt"

device = torch.device("cpu")

# ══════════════════════════════════════════════════════════════════
#  1. СКАЧИВАНИЕ ДАННЫХ
# ══════════════════════════════════════════════════════════════════

def download_klines(symbol=SYMBOL, interval=INTERVAL, total=TOTAL_CANDLES):
    """Скачивает свечи через Bybit API (пакетами по 1000)"""
    client = HTTP(testnet=False)  # mainnet для реальных данных
    all_data = []
    end_time = None
    limit = 1000
    
    print(f"📥 Скачиваю {total} свечей {interval}m для {symbol}...")
    
    while len(all_data) < total:
        try:
            kwargs = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            }
            if end_time:
                kwargs["end"] = end_time
            
            r = client.get_kline(**kwargs)
            candles = r["result"]["list"]
            if not candles:
                break
            
            all_data.extend(candles)
            oldest_ts = int(candles[-1][0])
            end_time = str(oldest_ts - 1)
            
            print(f"   Получено {len(all_data)} / {total} свечей...")
            time.sleep(0.3)  # rate limit
            
            if len(candles) < limit:
                break
        except Exception as e:
            print(f"   ⚠️ Ошибка: {e}")
            time.sleep(2)
    
    # Сортировка по времени (от старых к новым)
    all_data.sort(key=lambda x: int(x[0]))
    
    df = pd.DataFrame(
        all_data[:total],
        columns=["ts", "open", "high", "low", "close", "vol", "turnover"]
    )
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = df[col].astype(float)
    
    print(f"✅ Загружено {len(df)} свечей: {df['ts'].iloc[0]} -> {df['ts'].iloc[-1]}")
    return df


# ══════════════════════════════════════════════════════════════════
#  2. ПОДГОТОВКА ФИЧЕЙ (точно как в боте)
# ══════════════════════════════════════════════════════════════════

def compute_features(df):
    """Вычисляет 15 фичей из OHLCV данных"""
    c = df["close"]
    v = df["vol"]
    
    ret1 = c.pct_change(1)
    ret3 = c.pct_change(3)
    ret6 = c.pct_change(6)
    
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi_series = 100 - (100 / (1 + gain / (loss + 1e-10)))
    rsi = rsi_series
    stoch_rsi = ((rsi_series - rsi_series.rolling(14).min()) /
                 (rsi_series.rolling(14).max() - rsi_series.rolling(14).min() + 1e-10))
    williams_r = ((c.rolling(14).max() - c) /
                  (c.rolling(14).max() - c.rolling(14).min() + 1e-10))
    
    tp = c
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = ((tp - sma_tp) / (0.015 * mad + 1e-10) / 200)
    
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_hist = ((macd - macd.ewm(span=9).mean()) / (c + 1e-10))
    
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_width = (4 * std20 / (sma20 + 1e-10))
    bb_pos = ((c - (sma20 - 2 * std20)) / (4 * std20 + 1e-10))
    
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    obv_norm = (obv / (obv.rolling(50).std() + 1e-10))
    momentum = (c / (c.shift(10) + 1e-10) - 1)
    
    tr = pd.concat([(c - c).abs(),
                    (c - c.shift()).abs(),
                    (c - c.shift()).abs()], axis=1).max(axis=1)
    atr_ratio = (tr.rolling(14).mean() / (c + 1e-10))
    ma_trend = ((c.rolling(20).mean() - c.rolling(50).mean()) / c)
    vol_ratio = (v / (v.rolling(20).mean() + 1e-10))
    
    features = pd.DataFrame({
        "ret1": ret1, "ret3": ret3, "ret6": ret6,
        "rsi": rsi, "stoch_rsi": stoch_rsi, "williams_r": williams_r,
        "cci": cci, "macd_hist": macd_hist,
        "bb_width": bb_width, "bb_pos": bb_pos,
        "obv_norm": obv_norm, "momentum": momentum,
        "atr_ratio": atr_ratio, "ma_trend": ma_trend, "vol_ratio": vol_ratio,
    })
    
    return features


def create_labels(df, horizon=LABEL_HORIZON):
    """
    Создаёт метки: 1 если цена вырастет через horizon свечей, 0 иначе.
    """
    future_close = df["close"].shift(-horizon)
    returns = (future_close - df["close"]) / (df["close"] + 1e-10)
    labels = (returns > 0).astype(float)
    return labels


def prepare_sequences(features, labels, seq_len=SEQ_LEN):
    """Создаёт последовательности для LSTM"""
    # Убираем NaN
    valid_mask = ~(features.isna().any(axis=1) | labels.isna())
    features = features[valid_mask].values.astype(np.float32)
    labels = labels[valid_mask].values.astype(np.float32)
    
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
        y.append(labels[i + seq_len - 1])
    
    return np.array(X), np.array(y)


# ══════════════════════════════════════════════════════════════════
#  3. МОДЕЛЬ
# ══════════════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.bn(out[:, -1, :])
        out = self.drop(self.relu(self.fc1(out)))
        return self.sig(self.fc2(out))


def train_single_model(train_loader, val_loader, input_size, seed=42):
    """Тренирует одну модель"""
    torch.manual_seed(seed)
    
    model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()
    
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch).squeeze(-1)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).squeeze(-1)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
                
                predicted = (preds > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += len(y_batch)
        
        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.3f}")
        
        if patience_counter >= 15:
            print(f"   ⏹ Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model, best_val_loss


# ══════════════════════════════════════════════════════════════════
#  4. ГЛАВНАЯ
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  LSTM ENSEMBLE TRAINER — SOLUSDT")
    print("=" * 60)
    
    # 1. Скачиваем данные
    df = download_klines()
    
    # 2. Считаем фичи
    print("\n🔧 Вычисляю фичи...")
    features = compute_features(df)
    labels = create_labels(df)
    
    # 3. Готовим последовательности
    print("📊 Готовлю последовательности...")
    X, y = prepare_sequences(features, labels)
    print(f"   Всего сэмплов: {len(X)}")
    print(f"   Баланс классов: {y.mean():.3f} (1 = рост)")
    
    # 4. Разделяем на train/val
    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # 5. Нормализуем
    scaler = StandardScaler()
    n_features = X_train.shape[2]
    X_train_2d = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 6. Тренируем ансамбль
    print(f"\n🧠 Тренирую ансамбль из {N_MODELS} моделей...")
    models = []
    model_states = []
    model_configs = []
    
    for i in range(N_MODELS):
        print(f"\n📦 Модель {i+1}/{N_MODELS} (seed={i*7+42}):")
        model, val_loss = train_single_model(
            train_loader, val_loader,
            input_size=n_features,
            seed=i * 7 + 42
        )
        models.append(model)
        model_states.append(model.state_dict())
        model_configs.append({
            "hidden": HIDDEN_SIZE,
            "layers": NUM_LAYERS,
            "dropout": DROPOUT,
        })
        print(f"   ✅ Лучший val_loss: {val_loss:.4f}")
    
    # 7. Оцениваем ансамбль
    print("\n📊 Оценка ансамбля на валидации...")
    ensemble_correct = 0
    ensemble_total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            probs = []
            for m in models:
                probs.append(m(X_batch).squeeze(-1))
            ensemble_pred = torch.stack(probs).mean(dim=0)
            predicted = (ensemble_pred > 0.5).float()
            ensemble_correct += (predicted == y_batch).sum().item()
            ensemble_total += len(y_batch)
    
    ensemble_acc = ensemble_correct / ensemble_total if ensemble_total > 0 else 0
    print(f"   🎯 Ensemble Accuracy: {ensemble_acc:.3f}")
    
    # 8. Сохраняем
    checkpoint = {
        "seq_len": SEQ_LEN,
        "scaler": scaler,
        "n_models": N_MODELS,
        "input_size": n_features,
        "models": model_states,
        "model_configs": model_configs,
        "ensemble_accuracy": ensemble_acc,
        "trained_at": datetime.now().isoformat(),
        "total_samples": len(X),
        "label_horizon": LABEL_HORIZON,
    }
    
    torch.save(checkpoint, SAVE_PATH)
    file_size = os.path.getsize(SAVE_PATH) / 1024
    print(f"\n💾 Сохранено в {SAVE_PATH} ({file_size:.1f} KB)")
    print(f"   Фичей: {n_features}")
    print(f"   Seq len: {SEQ_LEN}")
    print(f"   Horizon: {LABEL_HORIZON} свечей")
    print("=" * 60)


if __name__ == "__main__":
    main()
