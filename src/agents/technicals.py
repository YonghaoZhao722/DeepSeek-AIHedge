import math

from langchain_core.messages import HumanMessage

from graph.state import AgentState, show_agent_reasoning

import json
import pandas as pd
import numpy as np

from tools.api import get_prices, prices_to_df
from utils.progress import progress


##### Technical Analyst #####
def technical_analyst_agent(state: AgentState):
    """
    Sophisticated technical analysis system that combines multiple trading strategies for multiple tickers:
    1. Trend Following
    2. Mean Reversion
    3. Momentum
    4. Volatility Analysis
    5. Statistical Arbitrage Signals
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize analysis for each ticker
    technical_analysis = {}

    for ticker in tickers:
        progress.update_status("technical_analyst_agent", ticker, "Analyzing price data")

        # Get the historical price data
        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        if not prices:
            progress.update_status("technical_analyst_agent", ticker, "Failed: No price data found")
            continue

        # Convert prices to a DataFrame
        prices_df = prices_to_df(prices)

        progress.update_status("technical_analyst_agent", ticker, "Calculating trend signals")
        trend_signals = calculate_trend_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Calculating mean reversion")
        mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Calculating momentum")
        momentum_signals = calculate_momentum_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Analyzing volatility")
        volatility_signals = calculate_volatility_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Statistical analysis")
        stat_arb_signals = calculate_stat_arb_signals(prices_df)

        # Combine all signals using a weighted ensemble approach
        strategy_weights = {
            "trend": 0.25,
            "mean_reversion": 0.20,
            "momentum": 0.25,
            "volatility": 0.15,
            "stat_arb": 0.15,
        }

        progress.update_status("technical_analyst_agent", ticker, "Combining signals")
        combined_signal = weighted_signal_combination(
            {
                "trend": trend_signals,
                "mean_reversion": mean_reversion_signals,
                "momentum": momentum_signals,
                "volatility": volatility_signals,
                "stat_arb": stat_arb_signals,
            },
            strategy_weights,
        )

        # Generate detailed analysis report for this ticker
        technical_analysis[ticker] = {
            "signal": combined_signal["signal"],
            "confidence": round(combined_signal["confidence"] * 100),
            "strategy_signals": {
                "trend_following": {
                    "signal": trend_signals["signal"],
                    "confidence": round(trend_signals["confidence"] * 100),
                    "metrics": normalize_pandas(trend_signals["metrics"]),
                },
                "mean_reversion": {
                    "signal": mean_reversion_signals["signal"],
                    "confidence": round(mean_reversion_signals["confidence"] * 100),
                    "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                },
                "momentum": {
                    "signal": momentum_signals["signal"],
                    "confidence": round(momentum_signals["confidence"] * 100),
                    "metrics": normalize_pandas(momentum_signals["metrics"]),
                },
                "volatility": {
                    "signal": volatility_signals["signal"],
                    "confidence": round(volatility_signals["confidence"] * 100),
                    "metrics": normalize_pandas(volatility_signals["metrics"]),
                },
                "statistical_arbitrage": {
                    "signal": stat_arb_signals["signal"],
                    "confidence": round(stat_arb_signals["confidence"] * 100),
                    "metrics": normalize_pandas(stat_arb_signals["metrics"]),
                },
            },
        }
        progress.update_status("technical_analyst_agent", ticker, "Done")

    # Create the technical analyst message
    message = HumanMessage(
        content=json.dumps(technical_analysis),
        name="technical_analyst_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(technical_analysis, "Technical Analyst")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def calculate_trend_signals(prices_df):
    """
    Advanced trend following strategy using multiple timeframes and indicators
    """
    # Calculate EMAs for multiple timeframes
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)

    # Calculate ADX for trend strength
    adx = calculate_adx(prices_df, 14)

    # Determine trend direction and strength
    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55

    # Combine signals with confidence weighting
    trend_strength = adx["adx"].iloc[-1] / 100.0

    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = "bullish"
        confidence = trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = "bearish"
        confidence = trend_strength
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": float(adx["adx"].iloc[-1]),
            "trend_strength": float(trend_strength),
        },
    }


def calculate_mean_reversion_signals(prices_df):
    """
    Mean reversion strategy using statistical measures and Bollinger Bands
    """
    # Calculate z-score of price relative to moving average
    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()
    z_score = (prices_df["close"] - ma_50) / std_50

    # Calculate Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)

    # Calculate RSI with multiple timeframes
    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)

    # Mean reversion signals
    price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    # Combine signals
    if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
        signal = "bullish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
        signal = "bearish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": float(z_score.iloc[-1]),
            "price_vs_bb": float(price_vs_bb),
            "rsi_14": float(rsi_14.iloc[-1]),
            "rsi_28": float(rsi_28.iloc[-1]),
        },
    }


def calculate_momentum_signals(prices_df):
    """Multi-factor momentum strategy"""
    # Price momentum with data length check
    returns = prices_df["close"].pct_change()
    
    # 根据可用数据长度调整周期
    available_periods = len(returns.dropna())
    mom_1m = returns.rolling(min(21, available_periods)).sum()
    mom_3m = returns.rolling(min(63, available_periods)).sum()
    mom_6m = returns.rolling(min(126, available_periods)).sum()

    # Volume momentum
    volume_ma = prices_df["volume"].rolling(min(21, available_periods)).mean()
    volume_momentum = prices_df["volume"] / volume_ma

    # Calculate momentum score with available data
    momentum_score = 0
    weights_sum = 0
    
    if not pd.isna(mom_1m.iloc[-1]):
        momentum_score += 0.4 * mom_1m.iloc[-1]
        weights_sum += 0.4
    if not pd.isna(mom_3m.iloc[-1]):
        momentum_score += 0.3 * mom_3m.iloc[-1]
        weights_sum += 0.3
    if not pd.isna(mom_6m.iloc[-1]):
        momentum_score += 0.3 * mom_6m.iloc[-1]
        weights_sum += 0.3
        
    if weights_sum > 0:
        momentum_score = momentum_score / weights_sum
    else:
        momentum_score = 0

    # Volume confirmation
    volume_confirmation = volume_momentum.iloc[-1] > 1.0

    if momentum_score > 0.05 and volume_confirmation:
        signal = "bullish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    elif momentum_score < -0.05 and volume_confirmation:
        signal = "bearish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_1m": float(mom_1m.iloc[-1]),
            "momentum_3m": float(mom_3m.iloc[-1]),
            "momentum_6m": float(mom_6m.iloc[-1]),
            "volume_momentum": float(volume_momentum.iloc[-1]),
        },
    }


def calculate_volatility_signals(prices_df):
    """Volatility-based trading strategy"""
    returns = prices_df["close"].pct_change()
    available_periods = len(returns.dropna())
    
    # 调整计算周期以适应可用数据
    vol_period = min(21, available_periods)
    regime_period = min(63, available_periods)
    
    # Historical volatility
    hist_vol = returns.rolling(vol_period).std() * math.sqrt(252)
    
    # Volatility regime detection with minimum periods check
    if available_periods >= regime_period:
        vol_ma = hist_vol.rolling(regime_period).mean()
        vol_regime = hist_vol / vol_ma.replace(0, float('nan'))
        vol_std = hist_vol.rolling(regime_period).std()
        vol_z_score = (hist_vol - vol_ma) / vol_std.replace(0, float('nan'))
        current_vol_regime = vol_regime.iloc[-1]
        vol_z = vol_z_score.iloc[-1]
    else:
        # 如果数据不足，使用简单的波动率比较
        current_vol_regime = 1.0  # 假设当前处于正常波动率区间
        recent_vol = hist_vol.iloc[-min(5, len(hist_vol)):]
        vol_z = (hist_vol.iloc[-1] - recent_vol.mean()) / (recent_vol.std() if len(recent_vol) > 1 else 1)

    # ATR ratio
    atr = calculate_atr(prices_df)
    atr_ratio = atr / prices_df["close"]

    # Generate signal based on available metrics
    if pd.notna(current_vol_regime) and pd.notna(vol_z):
        if current_vol_regime < 0.8 and vol_z < -1:
            signal = "bullish"
            confidence = min(abs(vol_z) / 3, 1.0)
        elif current_vol_regime > 1.2 and vol_z > 1:
            signal = "bearish"
            confidence = min(abs(vol_z) / 3, 1.0)
        else:
            signal = "neutral"
            confidence = 0.5
    else:
        # 使用简单的ATR比率作为备选指标
        if atr_ratio.iloc[-1] > atr_ratio.rolling(vol_period).mean().iloc[-1]:
            signal = "bearish"
            confidence = 0.6
        else:
            signal = "neutral"
            confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": float(hist_vol.iloc[-1]),
            "volatility_regime": float(current_vol_regime) if pd.notna(current_vol_regime) else 1.0,
            "volatility_z_score": float(vol_z) if pd.notna(vol_z) else 0.0,
            "atr_ratio": float(atr_ratio.iloc[-1]),
        },
    }


def calculate_stat_arb_signals(prices_df):
    """Statistical arbitrage signals based on price action analysis"""
    returns = prices_df["close"].pct_change().dropna()
    available_periods = len(returns)
    min_periods = min(63, available_periods)
    
    if min_periods < 5:  # 数据太少，返回中性信号
        return {
            "signal": "neutral",
            "confidence": 0.5,
            "metrics": {
                "hurst_exponent": 0.5,
                "skewness": 0.0,
                "kurtosis": 0.0,
            },
        }
    
    # 使用动态窗口计算统计量
    rolling_window = returns.rolling(window=min_periods, min_periods=5)
    
    # 计算偏度和峰度，使用较小的窗口以确保有值
    skew = rolling_window.skew().iloc[-1]
    kurt = rolling_window.kurt().iloc[-1]
    
    # 如果统计量为NaN，使用整个序列的统计量
    if pd.isna(skew):
        skew = returns.skew() if len(returns) > 0 else 0.0
    if pd.isna(kurt):
        kurt = returns.kurtosis() if len(returns) > 0 else 0.0
    
    # 计算Hurst指数
    hurst = calculate_hurst_exponent(prices_df["close"])
    
    # 根据可用的统计量生成信号
    if pd.notna(hurst) and pd.notna(skew):
        if hurst < 0.4 and skew > 1:
            signal = "bullish"
            confidence = (0.5 - hurst) * 2
        elif hurst < 0.4 and skew < -1:
            signal = "bearish"
            confidence = (0.5 - hurst) * 2
        else:
            signal = "neutral"
            confidence = 0.5
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "hurst_exponent": float(hurst),
            "skewness": float(skew) if pd.notna(skew) else 0.0,
            "kurtosis": float(kurt) if pd.notna(kurt) else 0.0,
        },
    }


def weighted_signal_combination(signals, weights):
    """
    Combines multiple trading signals using a weighted approach
    """
    # Convert signals to numeric values
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}

    weighted_sum = 0
    total_confidence = 0

    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal["signal"]]
        weight = weights[strategy]
        confidence = signal["confidence"]

        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence

    # Normalize the weighted sum
    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0

    # Convert back to signal
    if final_score > 0.2:
        signal = "bullish"
    elif final_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    return {"signal": signal, "confidence": abs(final_score)}


def normalize_pandas(obj):
    """Convert pandas Series/DataFrames to primitive Python types"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average

    Args:
        df: DataFrame with price data
        window: EMA period

    Returns:
        pd.Series: EMA values
    """
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)

    Args:
        df: DataFrame with OHLC data
        period: Period for calculations

    Returns:
        DataFrame with ADX values
    """
    # Calculate True Range
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

    # Calculate Directional Movement
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]

    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)

    # Calculate ADX
    df["+di"] = 100 * (df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["-di"] = 100 * (df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(span=period).mean()

    return df[["adx", "+di", "-di"]]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range

    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation

    Returns:
        pd.Series: ATR values
    """
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(period).mean()


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """Calculate Hurst Exponent with improved reliability"""
    # 确保数据量充足
    if len(price_series) < max_lag * 2:
        max_lag = len(price_series) // 2
    
    if max_lag < 4:  # 数据太少，返回0.5表示随机游走
        return 0.5
        
    lags = range(2, max_lag)
    tau = []
    
    # 计算每个lag的标准差
    for lag in lags:
        # 计算价格差异
        price_diff = np.subtract(price_series[lag:].values, price_series[:-lag].values)
        # 添加小的epsilon避免取log(0)
        tau.append(np.sqrt(np.std(price_diff) + 1e-10))
    
    # 使用对数回归计算Hurst指数
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        h = reg[0]  # Hurst指数是斜率
        
        # 限制Hurst指数在合理范围内
        h = max(0, min(1, h))
        return h
    except:
        return 0.5  # 计算失败时返回0.5表示随机游走
