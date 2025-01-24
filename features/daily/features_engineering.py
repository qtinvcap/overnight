import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial


def add_atr(g, atr_period=14):
    """Compute True Range and ATR"""
    # Calculate the previous close using shift before the apply
    prev_close = g["close"].shift(1)

    g["true_range"] = np.maximum.reduce(
        [g["high"] - g["low"], (g["high"] - prev_close).abs(), (g["low"] - prev_close).abs()]
    )
    g["atr"] = g["true_range"].rolling(window=atr_period).mean()
    return g


def add_macd(g, short_ema=12, long_ema=26, signal_ema=9):
    """Add MACD indicators"""
    g["ema_short"] = g["close"].ewm(span=short_ema, adjust=False).mean()
    g["ema_long"] = g["close"].ewm(span=long_ema, adjust=False).mean()
    g["macd_line"] = g["ema_short"] - g["ema_long"]
    g["macd_signal"] = g["macd_line"].ewm(span=signal_ema, adjust=False).mean()
    g["macd_hist"] = g["macd_line"] - g["macd_signal"]
    return g


def add_stoch_oscillator(g, stoch_period=14, d_period=3):
    """Add Stochastic Oscillator"""
    rolling_low = g["low"].rolling(window=stoch_period).min()
    rolling_high = g["high"].rolling(window=stoch_period).max()
    g["stoch_k"] = 100 * (g["close"] - rolling_low) / (rolling_high - rolling_low + 1e-9)
    g["stoch_d"] = g["stoch_k"].rolling(window=d_period).mean()
    return g


def add_obv(g):
    """Add On-Balance Volume"""
    close_diff = g["close"].diff()
    direction = np.sign(close_diff).fillna(0)
    g["obv"] = (direction * g["volume"]).fillna(0).cumsum()
    return g


def add_pivot_points(g):
    """Add Pivot Points"""
    g["pivot"] = (g["high"] + g["low"] + g["close"]) / 3.0
    g["r1"] = 2 * g["pivot"] - g["low"]
    g["s1"] = 2 * g["pivot"] - g["high"]
    range_y = g["high"] - g["low"]
    g["r2"] = g["pivot"] + range_y
    g["s2"] = g["pivot"] - range_y
    return g


def add_candlestick_patterns(g):
    """Add Candlestick Patterns"""
    body_size = (g["prev_day_open"] - g["close"]).abs()
    total_range = (g["high"] - g["low"]).abs()
    g["doji_flag"] = (body_size <= 0.1 * total_range).astype(int)

    real_body = (g["close"] - g["prev_day_open"]).abs()
    lower_shadow = (g[["close", "prev_day_open"]].min(axis=1) - g["low"]).abs()
    upper_shadow = (g["high"] - g[["close", "prev_day_open"]].max(axis=1)).abs()
    g["hammer_flag"] = ((lower_shadow >= 2 * real_body) & (upper_shadow <= 0.2 * real_body)).astype(int)
    return g


def per_ticker_features(
    g, rsi_period, boll_period, rolling_windows, short_ma_window, long_ma_window, extra_long_ma_window
):
    """Process features for a single ticker"""
    g = g.copy()
    g = g.set_index(pd.to_datetime(g["date"])).sort_index()

    # Store and shift original columns
    g["orig_close"] = g["close"]
    g["orig_high"] = g["high"]
    g["orig_low"] = g["low"]
    g["orig_open"] = g["open"]
    g["orig_vol"] = g["volume"]

    g["close"] = g["orig_close"].shift(1)
    g["high"] = g["orig_high"].shift(1)
    g["low"] = g["orig_low"].shift(1)
    g["open"] = g["orig_open"].shift(1)
    g["volume"] = g["orig_vol"].shift(1)

    g["current_day_open"] = g["orig_open"]
    g["next_day_open"] = g["orig_open"].shift(-1)
    g.rename(columns={"open": "prev_day_open"}, inplace=True)

    # Basic features
    g["prev_open_close_daily_return"] = g["close"] / g["prev_day_open"] - 1.0
    g["prev_close_close_daily_return"] = g["close"] / g["close"].shift(1) - 1.0
    g["open_open_daily_return"] = g["current_day_open"] / g["prev_day_open"] - 1.0
    g["daily_range"] = (g["high"] - g["low"]) / g["low"]
    g["today_open_gap"] = (g["current_day_open"] - g["close"]) / g["close"]

    # Price and volume metrics
    g["typical_price"] = (g["high"] + g["low"] + g["close"]) / 3
    g["dollar_volume"] = g["typical_price"] * g["volume"]
    g["volume_millions"] = g["volume"] / 1e6

    # Volatility metrics
    ln_hl = np.log(g["high"] / g["low"])
    ln_co = np.log(g["close"] / g["prev_day_open"])
    g["parkinson_daily"] = (1.0 / (4.0 * np.log(2))) * (ln_hl**2)
    g["gk_daily"] = 0.5 * ln_hl**2 - (2 * np.log(2) - 1) * ln_co**2

    # Technical indicators
    g = add_atr(g, atr_period=14)
    g = add_macd(g, short_ema=12, long_ema=26, signal_ema=9)
    g = add_stoch_oscillator(g, stoch_period=14, d_period=3)
    g = add_obv(g)
    g = add_pivot_points(g)
    g = add_candlestick_patterns(g)

    # RSI
    close_diff = g["close"].diff()
    gain = close_diff.where(close_diff > 0, 0.0)
    loss = -close_diff.where(close_diff < 0, 0.0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # Bollinger Bands
    ma_close = g["close"].rolling(window=boll_period).mean()
    std_close = g["close"].rolling(window=boll_period).std()
    g["boll_mid"] = ma_close
    g["boll_upper"] = ma_close + 2.0 * std_close
    g["boll_lower"] = ma_close - 2.0 * std_close

    # Moving averages
    g["short_ma"] = g["close"].rolling(window=short_ma_window).mean()
    g["long_ma"] = g["close"].rolling(window=long_ma_window).mean()
    g["extra_long_ma"] = g["close"].rolling(window=extra_long_ma_window).mean()

    # Rolling window calculations
    for w in rolling_windows:
        g[f"roll{w}_avg_dollar_volume"] = g["dollar_volume"].rolling(window=w).mean()
        g[f"roll{w}_std_return"] = g["prev_open_close_daily_return"].rolling(window=w).std()
        g[f"ratio_today_open_gap_to_roll{w}_avg_open_gap"] = (
            g["today_open_gap"] / g["today_open_gap"].rolling(window=w).mean()
        )
        g[f"roll{w}_avg_daily_range"] = g["daily_range"].rolling(window=w).mean()
        g[f"roll{w}_park_vol"] = g["parkinson_daily"].rolling(window=w).mean()
        g[f"roll{w}_gk_vol"] = g["gk_daily"].rolling(window=w).mean()
        g[f"roll{w}_price_momentum"] = g["close"] / g["close"].shift(w) - 1.0

        g[f"roll{w}_close_close_cum_return"] = (
            (1 + g["prev_close_close_daily_return"]).rolling(window=w).apply(lambda x: np.prod(x) - 1)
        )
        g[f"roll{w}_open_open_cum_return"] = (
            (1 + g["open_open_daily_return"]).rolling(window=w).apply(lambda x: np.prod(x) - 1)
        )

        opens_above_prev_close = (g["current_day_open"] > g["close"]).astype(int)
        g[f"roll{w}_opens_above_prev_close_count"] = opens_above_prev_close.rolling(window=w).sum() / w

    # Long rolling windows
    long_rolling_windows = [100, 250]
    opens_above_prev_close = (g["current_day_open"] > g["close"]).astype(int)
    for lw in long_rolling_windows:
        g[f"roll{lw}_opens_above_prev_close_count"] = opens_above_prev_close.rolling(window=lw).sum() / lw

    g.reset_index(drop=True, inplace=True)
    return g


def compute_advanced_daily_features(
    df_daily,
    rsi_period=14,
    boll_period=20,
    rolling_windows=[5, 10, 20, 50],
    short_ma_window=5,
    long_ma_window=20,
    extra_long_ma_window=200,
):
    """Parallel processing version of daily feature computation"""
    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df.sort_values(["ticker", "date"], inplace=True)

    # Split dataframe by ticker
    ticker_groups = [group for _, group in df.groupby("ticker")]

    # Create a pool of workers (use 75% of available CPUs)
    n_cores = max(1, int(cpu_count() * 0.75))
    print(f"Using {n_cores} CPU cores for parallel processing")

    # Run parallel processing
    with Pool(n_cores) as pool:
        parallel_func = partial(
            per_ticker_features,
            rsi_period=rsi_period,
            boll_period=boll_period,
            rolling_windows=rolling_windows,
            short_ma_window=short_ma_window,
            long_ma_window=long_ma_window,
            extra_long_ma_window=extra_long_ma_window,
        )
        results = pool.map(parallel_func, ticker_groups)

    # Combine results
    df_features = pd.concat(results, ignore_index=True)
    return df_features
