from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from loguru import logger
import typer
from stock_prediction.config import PROCESSED_DATA_DIR

app = typer.Typer()

def calculate_bollinger_bands(data, window=20):
    sma = data['adj_close'].rolling(window=window).mean()
    std = data['adj_close'].rolling(window=window).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    return sma, upper_band, lower_band

def calculate_macd(data, short_window=12, long_window=26):
    short_ema = data['adj_close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['adj_close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd, signal_line

def engineer_features(df_ticker):
    df = df_ticker.copy()
    df['price_change'] = df['adj_close'].pct_change()
    df['volatility'] = df['adj_close'].rolling(window=20).std()
    if 'sentiment_reddit' in df.columns:
        df['sentiment_reddit_momentum'] = df['sentiment_reddit'].rolling(window=7).mean()
    if 'sentiment_finnhub' in df.columns:
        df['sentiment_finnhub_momentum'] = df['sentiment_finnhub'].rolling(window=7).mean()
    df['adj_close_lag1'] = df['adj_close'].shift(1)
    df['volume_lag1'] = df['volume'].shift(1)
    if 'sentiment_reddit' in df.columns and 'rsi_14' in df.columns:
        df['sentiment_reddit_rsi_interaction'] = df['sentiment_reddit'] * df['rsi_14']
    if 'sentiment_finnhub' in df.columns:
        df['sentiment_finnhub_volatility_interaction'] = df['sentiment_finnhub'] * df['volatility']
    df['price_momentum_5d'] = df['price_change'].rolling(window=5).mean()
    df['volatility_ratio'] = df['volatility'] / df['adj_close'].rolling(window=60).std()
    df['sma_20'], df['upper_band'], df['lower_band'] = calculate_bollinger_bands(df)
    df['macd'], df['signal_line'] = calculate_macd(df)
    return df

def perform_rfe(X, y, feature_cols, n_features=5):
    estimator = LinearRegression()
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector.fit(X[feature_cols], y)
    selected_features = [feature_cols[i] for i in range(len(selector.support_)) if selector.support_[i]]
    return selected_features

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "extended_stock_data.csv",
    output_base_dir: Path = PROCESSED_DATA_DIR / "engineered_features",
    lookback: int = 60,
):
    logger.info("Loading data from {}", input_path)
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    tickers = df['ticker'].unique()

    feature_cols_without_sentiment = [
        'adj_close', 'sma_20', 'rsi_14', 'volume', 'price_change', 'volatility',
        'adj_close_lag1', 'volume_lag1', 'price_momentum_5d', 'volatility_ratio',
        'upper_band', 'lower_band', 'macd', 'signal_line'
    ]
    feature_cols_with_sentiment = feature_cols_without_sentiment + [
        'sentiment_reddit', 'sentiment_finnhub', 'sentiment_reddit_momentum',
        'sentiment_finnhub_momentum', 'sentiment_reddit_rsi_interaction',
        'sentiment_finnhub_volatility_interaction'
    ]

    for ticker in tickers:
        logger.info("Processing {}", ticker)
        output_dir = output_base_dir / ticker
        output_dir.mkdir(parents=True, exist_ok=True)

        df_ticker = df[df['ticker'] == ticker].copy()
        df_engineered = engineer_features(df_ticker)
        df_engineered = df_engineered.dropna()

        if len(df_engineered) < 10:  # Arbitrary threshold to ensure enough data
            logger.warning("Insufficient data for {} after dropping NaNs.", ticker)
            continue

        # Perform RFE for non-sentiment features
        X = df_engineered
        y = df_engineered['adj_close']
        selected_features_no_sentiment = perform_rfe(X, y, feature_cols_without_sentiment)

        # Perform RFE for sentiment features
        selected_features_with_sentiment = perform_rfe(X, y, feature_cols_with_sentiment)

        # Save engineered data
        df_engineered.to_csv(output_dir / f"{ticker}_engineered_data.csv", index=False)

        # Save selected features
        with open(output_dir / "selected_features_no_sentiment.txt", "w") as f:
            f.write("\n".join(selected_features_no_sentiment))
        with open(output_dir / "selected_features_with_sentiment.txt", "w") as f:
            f.write("\n".join(selected_features_with_sentiment))

        # Write feature notes
        notes_no_sentiment = f"""Feature Engineering Notes (No Sentiment):
- New Features:
  - price_change: Daily percentage change in adj_close for stationarity (ARIMA).
  - volatility: 20-day rolling standard deviation of adj_close for risk measurement.
  - adj_close_lag1: 1-day lag of adj_close to capture temporal dependencies.
  - volume_lag1: 1-day lag of volume to capture trading activity trends.
  - price_momentum_5d: 5-day rolling mean of price_change for short-term momentum.
  - volatility_ratio: Ratio of 20-day to 60-day volatility to normalize risk.
  - sma_20: 20-day simple moving average for trend detection.
  - upper_band: Upper Bollinger Band for volatility measurement.
  - lower_band: Lower Bollinger Band for volatility measurement.
  - macd: Moving Average Convergence Divergence for momentum detection.
  - signal_line: 9-day EMA of MACD for signal detection.
- Rationale:
  - price_change: Captures daily returns, often stationary and useful for models like ARIMA.
  - volatility: Measures risk, crucial for portfolio optimization.
  - lag features: Help capture temporal dependencies in price and volume data.
  - price_momentum_5d: Provides a short-term momentum signal.
  - volatility_ratio: Normalizes volatility to highlight unusual market behavior.
- Selected Features: {', '.join(selected_features_no_sentiment)}
- Excluded: open, high, low, close (redundant with adj_close); all sentiment features.
- Lookback: {lookback} days, consistent with preprocessing for RL sequences.
"""
        with open(output_dir / f"{ticker}_feature_notes_no_sentiment.txt", "w") as f:
            f.write(notes_no_sentiment)

        notes_with_sentiment = f"""Feature Engineering Notes (With Sentiment):
- New Features:
  - price_change: Daily percentage change in adj_close for stationarity (ARIMA).
  - volatility: 20-day rolling standard deviation of adj_close for risk measurement.
  - sentiment_reddit_momentum: 7-day rolling mean of Reddit sentiment to smooth noise.
  - sentiment_finnhub_momentum: 7-day rolling mean of Finnhub sentiment to capture trends.
  - adj_close_lag1: 1-day lag of adj_close to capture temporal dependencies.
  - volume_lag1: 1-day lag of volume to capture trading activity trends.
  - sentiment_reddit_rsi_interaction: Interaction between Reddit sentiment and RSI.
  - sentiment_finnhub_volatility_interaction: Interaction between Finnhub sentiment and volatility.
  - price_momentum_5d: 5-day rolling mean of price_change for short-term momentum.
  - volatility_ratio: Ratio of 20-day to 60-day volatility to normalize risk.
  - sma_20: 20-day simple moving average for trend detection.
  - upper_band: Upper Bollinger Band for volatility measurement.
  - lower_band: Lower Bollinger Band for volatility measurement.
  - macd: Moving Average Convergence Divergence for momentum detection.
  - signal_line: 9-day EMA of MACD for signal detection.
- Rationale:
  - price_change: Captures daily returns, often stationary and useful for models like ARIMA.
  - volatility: Measures risk, crucial for portfolio optimization.
  - sentiment_momentum: Smooths out noise in sentiment data and captures trends.
  - lag features: Help capture temporal dependencies in price and volume data.
  - interaction features: Combine sentiment and technical indicators.
  - price_momentum_5d: Provides a short-term momentum signal.
  - volatility_ratio: Normalizes volatility to highlight unusual market behavior.
- Selected Features: {', '.join(selected_features_with_sentiment)}
- Excluded: open, high, low, close (redundant with adj_close).
- Lookback: {lookback} days, consistent with preprocessing for RL sequences.
"""
        with open(output_dir / f"{ticker}_feature_notes_with_sentiment.txt", "w") as f:
            f.write(notes_with_sentiment)

    logger.success("Feature engineering and selection complete for all tickers.")

if __name__ == "__main__":
    app()