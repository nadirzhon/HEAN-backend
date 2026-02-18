#!/usr/bin/env python3
"""Train Oracle price-reversal models (TCN or LSTM).

Trains models compatible with the HEAN OracleIntegration signal-fusion stack:
  - TCN  → produces weights loadable by TCPriceReversalPredictor (TCN_MODEL_PATH)
  - LSTM → produces weights loadable by the ml_predictor pipeline (LSTM_MODEL_PATH)

Usage:
    python3 scripts/train_oracle.py --symbol BTCUSDT --days 30 --model-type tcn
    python3 scripts/train_oracle.py --symbol ETHUSDT --days 14 --model-type lstm \\
        --output models/lstm_eth.pt --epochs 50 --batch-size 64 --learning-rate 0.0005

Requires (optional deps, guarded at import):
    pip install torch numpy pandas pybit
    pip install scikit-learn          # for split/metrics helpers
    pip install duckdb                # to read local tick history
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional heavy dependencies — all guarded so the script fails gracefully
# ---------------------------------------------------------------------------
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from pybit.unified_trading import HTTP as BybitHTTP

    PYBIT_AVAILABLE = True
except ImportError:
    PYBIT_AVAILABLE = False

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging — mirrors HEAN convention (plain print with timestamps for scripts)
# ---------------------------------------------------------------------------
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("train_oracle")

# ---------------------------------------------------------------------------
# TCN architecture — mirrors TCNPredictor in hean-core (must stay in sync with
# hean/core/intelligence/tcn_predictor.py so checkpoints are interchangeable)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class Chomp1d(nn.Module):
        """Remove causal padding from the end of the time dimension."""

        def __init__(self, chomp_size: int) -> None:
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return x[:, :, : -self.chomp_size].contiguous()

    class TemporalBlock(nn.Module):
        """Dilated causal conv block with residual connection."""

        def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            kernel_size: int,
            dilation: int,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            pad = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(
                n_inputs, n_outputs, kernel_size, padding=pad, dilation=dilation
            )
            self.chomp1 = Chomp1d(pad)
            self.relu1 = nn.ReLU()
            self.drop1 = nn.Dropout(dropout)

            self.conv2 = nn.Conv1d(
                n_outputs, n_outputs, kernel_size, padding=pad, dilation=dilation
            )
            self.chomp2 = Chomp1d(pad)
            self.relu2 = nn.ReLU()
            self.drop2 = nn.Dropout(dropout)

            self.net = nn.Sequential(
                self.conv1,
                self.chomp1,
                self.relu1,
                self.drop1,
                self.conv2,
                self.chomp2,
                self.relu2,
                self.drop2,
            )
            self.downsample = (
                nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            )
            self.relu = nn.ReLU()
            self._init_weights()

        def _init_weights(self) -> None:
            self.conv1.weight.data.normal_(0, 0.01)
            self.conv2.weight.data.normal_(0, 0.01)
            if self.downsample is not None:
                self.downsample.weight.data.normal_(0, 0.01)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)

    class TCNModel(nn.Module):
        """Full TCN: stack of TemporalBlocks → linear head → sigmoid."""

        def __init__(
            self,
            input_size: int = 4,
            num_channels: list[int] | None = None,
            kernel_size: int = 3,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            if num_channels is None:
                num_channels = [32, 32, 32]
            layers: list[nn.Module] = []
            for i, out_ch in enumerate(num_channels):
                in_ch = input_size if i == 0 else num_channels[i - 1]
                layers.append(
                    TemporalBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout)
                )
            self.network = nn.Sequential(*layers)
            self.fc = nn.Linear(num_channels[-1], 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, features, seq_len)
            y = self.network(x)
            y = y[:, :, -1]  # last time-step
            return self.sigmoid(self.fc(y))

    class LSTMModel(nn.Module):
        """Multi-layer LSTM → dense head.  Outputs direction logits (3 horizons)."""

        def __init__(
            self,
            n_features: int,
            hidden_sizes: list[int] | None = None,
            n_outputs: int = 3,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            if hidden_sizes is None:
                hidden_sizes = [128, 64, 32]
            self.lstms = nn.ModuleList()
            in_size = n_features
            for i, h in enumerate(hidden_sizes):
                self.lstms.append(
                    nn.LSTM(
                        in_size,
                        h,
                        batch_first=True,
                        dropout=dropout if i < len(hidden_sizes) - 1 else 0.0,
                    )
                )
                in_size = h
            self.fc1 = nn.Linear(hidden_sizes[-1], 64)
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(64, n_outputs)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, seq_len, features)
            out = x
            for lstm in self.lstms:
                out, _ = lstm(out)
            last = out[:, -1, :]  # last time-step
            return self.fc2(self.drop(torch.relu(self.fc1(last))))


# ---------------------------------------------------------------------------
# Feature engineering (self-contained, no hean imports needed at script level)
# ---------------------------------------------------------------------------


def _rsi(prices: "pd.Series", period: int = 14) -> "pd.Series":
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _macd(prices: "pd.Series") -> "tuple[pd.Series, pd.Series]":
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.fillna(0.0), signal.fillna(0.0)


def _atr(df: "pd.DataFrame", period: int = 14) -> "pd.Series":
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean().bfill()


def build_feature_matrix(df: "pd.DataFrame") -> "tuple[np.ndarray, np.ndarray]":
    """Engineer features from an OHLCV DataFrame.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume].
            Must be sorted ascending by timestamp with no duplicate rows.

    Returns:
        (X, y) where:
          X: float32 array of shape (n_samples, seq_len, n_features) for LSTM
             or (n_samples, n_features, seq_len) for TCN — caller picks axis.
          y: float32 array of shape (n_samples,) — binary reversal label (TCN)
             OR (n_samples, 3) — signed return % for 1 h / 4 h / 24 h (LSTM).

    The function returns the raw (X_flat, y) for the *full* sequence; callers
    slice into train/val splits.
    """
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    # Price-based features
    ret = close.pct_change().fillna(0.0)
    log_ret = np.log1p(ret)
    vol_20 = ret.rolling(20).std().fillna(0.0)
    sma20 = close.rolling(20).mean().bfill()
    sma50 = close.rolling(50).mean().bfill()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    rsi = _rsi(close)
    macd, macd_sig = _macd(close)
    bb_mid = close.rolling(20).mean().bfill()
    bb_std = close.rolling(20).std().bfill()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pos = ((close - bb_lower) / (bb_upper - bb_lower + 1e-9)).clip(0, 1)
    atr = _atr(df)
    obv = (np.sign(close.diff()) * volume).fillna(0.0).cumsum()
    vol_norm = (volume / volume.rolling(20).mean().bfill()).fillna(1.0)

    feature_df = pd.DataFrame(
        {
            "log_ret": log_ret,
            "vol_20": vol_20,
            "rsi": rsi / 100.0,  # normalise to [0,1]
            "macd": macd,
            "macd_sig": macd_sig,
            "bb_pos": bb_pos,
            "sma_ratio": (sma20 / sma50).fillna(1.0) - 1.0,
            "ema_ratio": (ema12 / ema26).fillna(1.0) - 1.0,
            "atr_pct": (atr / close).fillna(0.0),
            "obv_norm": (obv / (obv.abs().rolling(100).mean().replace(0, 1))).fillna(0.0),
            "vol_norm": vol_norm,
        }
    )

    # Global z-score normalisation per column (fit on full history, slight lookahead
    # acceptable at training time — inference uses online normalisation in predictor)
    feature_arr = feature_df.values.astype(np.float32)
    col_mean = feature_arr.mean(axis=0, keepdims=True)
    col_std = feature_arr.std(axis=0, keepdims=True) + 1e-8
    feature_arr = (feature_arr - col_mean) / col_std

    return feature_arr  # shape (T, n_features)


def make_tcn_sequences(
    features: "np.ndarray",
    close: "np.ndarray",
    seq_len: int = 512,
    horizon: int = 10,
    reversal_threshold: float = 0.003,
) -> "tuple[np.ndarray, np.ndarray]":
    """Build (X, y) pairs for TCN binary reversal classification.

    A reversal label = 1 if price moves > reversal_threshold in the opposite
    direction of the last bar's return within the next `horizon` bars.

    Args:
        features: (T, n_features) float32 array.
        close: (T,) float32 array of close prices.
        seq_len: Lookback window length.
        horizon: Bars ahead to check for reversal.
        reversal_threshold: Minimum fractional move to count as reversal.

    Returns:
        X: (N, n_features, seq_len)  — TCN channel-first format.
        y: (N, 1) float32 binary labels.
    """
    T = len(features)
    Xs, ys = [], []
    for i in range(seq_len, T - horizon):
        x_slice = features[i - seq_len : i]  # (seq_len, n_features)
        last_ret = close[i] / close[i - 1] - 1.0 if close[i - 1] > 0 else 0.0
        future_ret = close[i + horizon] / close[i] - 1.0 if close[i] > 0 else 0.0
        # Reversal: price was going up but turns down (or vice-versa) by threshold
        label = float(
            abs(future_ret) > reversal_threshold
            and (last_ret * future_ret < 0)  # sign flip
        )
        Xs.append(x_slice.T)  # transpose → (n_features, seq_len)
        ys.append([label])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def make_lstm_sequences(
    features: "np.ndarray",
    close: "np.ndarray",
    seq_len: int = 60,
    horizons: list[int] | None = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """Build (X, y) pairs for LSTM multi-horizon direction prediction.

    Args:
        features: (T, n_features) float32 array.
        close: (T,) float32 array of close prices.
        seq_len: Lookback window length.
        horizons: List of bar-ahead horizons (default: [1, 4, 24]).

    Returns:
        X: (N, seq_len, n_features)  — LSTM sequence-first format.
        y: (N, len(horizons)) float32 signed return % targets.
    """
    if horizons is None:
        horizons = [1, 4, 24]
    T = len(features)
    max_h = max(horizons)
    Xs, ys = [], []
    for i in range(seq_len, T - max_h):
        x_slice = features[i - seq_len : i]  # (seq_len, n_features)
        targets = []
        for h in horizons:
            if close[i] > 0:
                targets.append(float((close[i + h] / close[i] - 1.0) * 100.0))
            else:
                targets.append(0.0)
        Xs.append(x_slice)
        ys.append(targets)
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading: DuckDB → CSV fallback → Bybit live
# ---------------------------------------------------------------------------


def load_from_duckdb(db_path: str, symbol: str, days: int) -> "pd.DataFrame | None":
    """Attempt to read historical ticks from local DuckDB store.

    Returns an OHLCV DataFrame (1-minute bars) or None if unavailable.
    """
    if not DUCKDB_AVAILABLE:
        logger.warning("duckdb not installed — skipping DuckDB data source")
        return None
    try:
        conn = duckdb.connect(db_path, read_only=True)
        cutoff_ts = time.time() - days * 86400
        # Aggregate micro-ticks into 1-min OHLCV
        df = conn.execute(
            """
            SELECT
                time_bucket(INTERVAL '1 minute', to_timestamp(timestamp)) AS ts,
                first(price ORDER BY timestamp) AS open,
                max(price) AS high,
                min(price) AS low,
                last(price ORDER BY timestamp) AS close,
                sum(volume) AS volume
            FROM ticks
            WHERE symbol = ? AND timestamp >= ?
            GROUP BY ts
            ORDER BY ts
            """,
            [symbol, cutoff_ts],
        ).df()
        conn.close()
        if df.empty:
            logger.warning(f"DuckDB returned no rows for {symbol} (last {days} days)")
            return None
        df = df.rename(columns={"ts": "timestamp"})
        logger.info(f"Loaded {len(df)} 1-min bars from DuckDB for {symbol}")
        return df
    except Exception as exc:
        logger.warning(f"DuckDB read failed ({exc}), trying CSV fallback")
        return None


def load_from_csv(data_dir: str, symbol: str) -> "pd.DataFrame | None":
    """Try to load OHLCV data from a CSV file in the data/ directory.

    Expects file named like `data/BTCUSDT.csv` or `data/BTCUSDT_ohlcv.csv`.
    Accepts any CSV with columns: timestamp, open, high, low, close, volume.
    """
    if not PANDAS_AVAILABLE:
        return None
    candidates = [
        Path(data_dir) / f"{symbol}.csv",
        Path(data_dir) / f"{symbol.lower()}.csv",
        Path(data_dir) / f"{symbol}_ohlcv.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path, parse_dates=["timestamp"])
                required = {"timestamp", "open", "high", "low", "close", "volume"}
                if not required.issubset(df.columns):
                    logger.warning(f"{path} missing required columns: {required - set(df.columns)}")
                    continue
                df = df.sort_values("timestamp").reset_index(drop=True)
                logger.info(f"Loaded {len(df)} rows from CSV: {path}")
                return df
            except Exception as exc:
                logger.warning(f"Failed to read {path}: {exc}")
    return None


def load_from_bybit(symbol: str, days: int) -> "pd.DataFrame | None":
    """Fetch 1-hour OHLCV klines from Bybit public endpoint (no auth required).

    Uses mainnet — historical data is the same as testnet price history.
    """
    if not PYBIT_AVAILABLE or not PANDAS_AVAILABLE:
        logger.error("pybit / pandas not installed — cannot fetch from Bybit")
        return None

    client = BybitHTTP(testnet=False)
    end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_rows: list[list] = []
    current_start = start_ms
    logger.info(f"Fetching klines from Bybit for {symbol} ({days} days, 1h interval)")

    while current_start < end_ms:
        try:
            result = client.get_kline(
                category="linear",
                symbol=symbol,
                interval="60",
                start=current_start,
                end=end_ms,
                limit=1000,
            )
        except Exception as exc:
            logger.error(f"Bybit API error: {exc}")
            break

        if result.get("retCode", -1) != 0:
            logger.error(f"Bybit returned error: {result.get('retMsg')}")
            break

        klines = result.get("result", {}).get("list", [])
        if not klines:
            break

        all_rows.extend(klines)
        last_ts = int(klines[-1][0])
        current_start = last_ts + 1
        time.sleep(0.1)  # respect rate limit

    if not all_rows:
        logger.error("No kline data returned from Bybit")
        return None

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    logger.info(f"Fetched {len(df)} 1-hour bars from Bybit")
    return df


def load_data(symbol: str, days: int, db_path: str = "data/hean.duckdb") -> "pd.DataFrame":
    """Load market data with DuckDB → CSV → Bybit fallback chain."""
    if not NUMPY_AVAILABLE or not PANDAS_AVAILABLE:
        logger.error("numpy and pandas are required. Install: pip install numpy pandas")
        sys.exit(1)

    df = load_from_duckdb(db_path, symbol, days)
    if df is None or len(df) < 200:
        df = load_from_csv("data", symbol)
    if df is None or len(df) < 200:
        df = load_from_bybit(symbol, days)
    if df is None or len(df) < 200:
        n_available = len(df) if df is not None else 0
        logger.error(
            f"Insufficient data for {symbol}: got {n_available} bars, need >= 200. "
            "Ensure DuckDB has tick history, or place a CSV at data/{symbol}.csv, "
            "or confirm Bybit connectivity."
        )
        sys.exit(1)

    logger.info(
        f"Data ready: {len(df)} bars for {symbol} "
        f"({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})"
    )
    return df


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------


def train_tcn(
    df: "pd.DataFrame",
    output_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Train TCN reversal predictor and save checkpoint.

    Returns:
        Dictionary of final evaluation metrics.
    """
    if not TORCH_AVAILABLE:
        logger.error("torch not installed. Install: pip install torch")
        sys.exit(1)

    close_arr = df["close"].values.astype(np.float32)
    features = build_feature_matrix(df)
    n_features = features.shape[1]

    # Sequence length: default 512 ticks (comparable to online predictor's buffer)
    seq_len = min(512, len(df) // 4)
    X, y = make_tcn_sequences(features, close_arr, seq_len=seq_len)
    logger.info(f"TCN dataset: {len(X)} samples, seq_len={seq_len}, features={n_features}")

    # 80/20 train/val split (chronological — no shuffle to avoid leakage)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    model = TCNModel(input_size=n_features, num_channels=[32, 32, 32], kernel_size=3, dropout=0.1)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(y_train).to(device),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).to(device),
        torch.from_numpy(y_val).to(device),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val_loss = float("inf")
    best_state: dict | None = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
                predicted = (pred > 0.5).float()
                correct += (predicted == yb).sum().item()
                total += yb.numel()
        val_loss /= len(val_ds)
        val_acc = correct / total if total > 0 else 0.0

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:>4}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.3f} | "
                f"elapsed={elapsed:.0f}s"
            )

    # Restore best weights before saving
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = (model(xb) > 0.5).float().cpu().numpy().flatten()
            truth = yb.cpu().numpy().flatten()
            for p, t in zip(pred, truth):
                if t == 1 and p == 1:
                    tp += 1
                elif t == 0 and p == 0:
                    tn += 1
                elif t == 0 and p == 1:
                    fp += 1
                else:
                    fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    metrics: dict[str, Any] = {
        "model_type": "tcn",
        "val_accuracy": round(accuracy, 4),
        "val_precision": round(precision, 4),
        "val_recall": round(recall, 4),
        "val_f1": round(f1, 4),
        "best_val_loss": round(best_val_loss, 6),
        "epochs_trained": epochs,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "seq_len": seq_len,
        "n_features": n_features,
        "training_seconds": round(time.time() - t0, 1),
    }

    # Save checkpoint — format compatible with TCPriceReversalPredictor.load_model()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sequence_length": seq_len,
            "input_size": n_features,
            "training_metrics": {
                "total_updates": epochs * len(train_ds),
                "total_loss": best_val_loss * len(val_ds),
                "recent_accuracy": accuracy,
            },
            "hean_metadata": metrics,
        },
        out_path,
    )
    logger.info(f"TCN model saved: {out_path}")

    # Write companion metadata JSON (consumed by promote_model.py)
    meta_path = out_path.with_suffix(".json")
    metrics["output_path"] = str(out_path.resolve())
    metrics["saved_at"] = datetime.now(tz=timezone.utc).isoformat()
    meta_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Metadata written: {meta_path}")

    return metrics


def train_lstm(
    df: "pd.DataFrame",
    output_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Train LSTM multi-horizon direction predictor and save checkpoint.

    Returns:
        Dictionary of final evaluation metrics.
    """
    if not TORCH_AVAILABLE:
        logger.error("torch not installed. Install: pip install torch")
        sys.exit(1)

    close_arr = df["close"].values.astype(np.float32)
    features = build_feature_matrix(df)
    n_features = features.shape[1]

    seq_len = min(60, len(df) // 4)
    horizons = [1, 4, 24]
    X, y = make_lstm_sequences(features, close_arr, seq_len=seq_len, horizons=horizons)
    logger.info(f"LSTM dataset: {len(X)} samples, seq_len={seq_len}, features={n_features}")

    # Convert regression targets to direction labels: sign of return
    y_dir = np.sign(y).astype(np.float32)  # -1, 0, +1  — treated as regression targets
    # Remap to [0, 1] for BCELoss: up=1, down or flat=0
    y_bin = (y_dir > 0).astype(np.float32)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_bin[:split], y_bin[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    model = LSTMModel(
        n_features=n_features,
        hidden_sizes=[128, 64, 32],
        n_outputs=len(horizons),
        dropout=0.2,
    )
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(y_train).to(device),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).to(device),
        torch.from_numpy(y_val).to(device),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val_loss = float("inf")
    best_state: dict | None = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_acc_sum = 0.0
        n_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                pred_dir = (torch.sigmoid(logits) > 0.5).float()
                val_acc_sum += (pred_dir == yb).float().mean().item()
                n_batches += 1
        val_loss /= len(val_ds)
        val_acc = val_acc_sum / n_batches if n_batches > 0 else 0.0

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:>4}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.3f} | "
                f"elapsed={elapsed:.0f}s"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Per-horizon direction accuracy on validation set
    model.eval()
    horizon_correct = [0] * len(horizons)
    horizon_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = (torch.sigmoid(model(xb)) > 0.5).float().cpu().numpy()
            truth = yb.cpu().numpy()
            for h_idx in range(len(horizons)):
                horizon_correct[h_idx] += int((pred[:, h_idx] == truth[:, h_idx]).sum())
            horizon_total += len(xb)

    horizon_accs = {
        f"val_accuracy_{horizons[i]}h": round(horizon_correct[i] / horizon_total, 4)
        for i in range(len(horizons))
    }
    avg_acc = sum(horizon_correct) / (horizon_total * len(horizons)) if horizon_total > 0 else 0.0

    metrics: dict[str, Any] = {
        "model_type": "lstm",
        "val_accuracy": round(avg_acc, 4),
        "best_val_loss": round(best_val_loss, 6),
        "epochs_trained": epochs,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "seq_len": seq_len,
        "n_features": n_features,
        "horizons": horizons,
        "training_seconds": round(time.time() - t0, 1),
        **horizon_accs,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features": n_features,
            "hidden_sizes": [128, 64, 32],
            "n_outputs": len(horizons),
            "seq_len": seq_len,
            "horizons": horizons,
            "hean_metadata": metrics,
        },
        out_path,
    )
    logger.info(f"LSTM model saved: {out_path}")

    meta_path = out_path.with_suffix(".json")
    metrics["output_path"] = str(out_path.resolve())
    metrics["saved_at"] = datetime.now(tz=timezone.utc).isoformat()
    meta_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Metadata written: {meta_path}")

    return metrics


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(symbol: str, model_type: str, output_path: str, metrics: dict[str, Any]) -> None:
    """Print a human-readable training summary to stdout."""
    width = 60
    sep = "=" * width
    print(f"\n{sep}")
    print(f"  HEAN Oracle Training — {model_type.upper()} — {symbol}")
    print(sep)
    print(f"  Output         : {output_path}")
    print(f"  Training time  : {metrics.get('training_seconds', '?')}s")
    print(f"  Epochs trained : {metrics.get('epochs_trained', '?')}")
    print(f"  Train samples  : {metrics.get('n_train', '?')}")
    print(f"  Val samples    : {metrics.get('n_val', '?')}")
    print(f"  Seq length     : {metrics.get('seq_len', '?')}")
    print(f"  Features       : {metrics.get('n_features', '?')}")
    print(sep)
    print("  Validation Metrics:")
    for k, v in metrics.items():
        if k.startswith("val_"):
            label = k.replace("val_", "").replace("_", " ").title()
            print(f"    {label:<22}: {v}")
    print(sep)
    val_acc = metrics.get("val_accuracy", 0.0)
    if isinstance(val_acc, float) and val_acc >= 0.60:
        verdict = "GOOD — directional accuracy >= 60%, suitable for production"
    else:
        verdict = "MARGINAL — directional accuracy < 60%, continue training or gather more data"
    print(f"  Verdict: {verdict}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Oracle price-reversal model (TCN or LSTM) for HEAN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (e.g. BTCUSDT, ETHUSDT)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of historical data to load",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["tcn", "lstm"],
        default="tcn",
        help="Model architecture to train",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path for model weights. "
            "Defaults to models/tcn_<symbol>.pt or models/lstm_<symbol>.pt"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/hean.duckdb",
        help="Path to local DuckDB tick database (used as primary data source)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    symbol = args.symbol.upper()
    model_type = args.model_type.lower()
    output_path = args.output or f"models/{model_type}_{symbol.lower()}.pt"

    logger.info(
        f"Starting Oracle training | symbol={symbol} | model={model_type} | "
        f"days={args.days} | epochs={args.epochs} | batch={args.batch_size} | "
        f"lr={args.learning_rate} | output={output_path}"
    )

    df = load_data(symbol, args.days, db_path=args.db_path)

    if model_type == "tcn":
        metrics = train_tcn(
            df,
            output_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    else:
        metrics = train_lstm(
            df,
            output_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    print_summary(symbol, model_type, output_path, metrics)


if __name__ == "__main__":
    main()
