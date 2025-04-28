from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from loguru import logger
import typer

from stock_prediction.config import PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()


def create_sequences(df, feature_cols, target_col, lookback):
    """Create sequences for time series modeling."""
    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(df.iloc[i - lookback:i][feature_cols].values)
        y.append(df.iloc[i][target_col])
    return np.array(X), np.array(y)


@app.command()
def main(
        data_path: Path = PROCESSED_DATA_DIR / "extended_stock_data.csv",
        metrics_path: Path = REPORTS_DIR / "model_performance.csv",
        figures_dir: Path = REPORTS_DIR / "figures",
        lookback: int = 60,
):
    logger.info("Loading data from {}", data_path)
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])

    tickers = df['ticker'].unique()
    feature_cols = ['adj_close', 'sma_20', 'rsi_14', 'volume', 'sentiment',
                    'sentiment_reddit', 'sentiment_wallstreetbets', 'sentiment_investing',
                    'sentiment_stocks', 'sentiment_StockMarket', 'sentiment_finnhub',
                    'pe_ratio', 'forward_eps']
    target_col = 'adj_close'

    results = []
    figures_dir.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        logger.info("Processing ticker: {}", ticker)
        df_ticker = df[df['ticker'] == ticker].copy().dropna(subset=feature_cols + [target_col])
        if len(df_ticker) < lookback + 1:
            logger.warning("Insufficient data for ticker {} (length: {}). Skipping.", ticker, len(df_ticker))
            continue

        X, y = create_sequences(df_ticker, feature_cols, target_col, lookback)
        tscv = TimeSeriesSplit(n_splits=5)

        arima_mapes, arima_rmses = [], []
        lr_mapes, lr_rmses = [], []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # ARIMA Model
            try:
                model = ARIMA(y_train, order=(1, 1, 0))
                fit = model.fit(method_kwargs={"maxiter": 200})
                arima_preds = fit.forecast(steps=len(y_test))
                arima_mape = mean_absolute_percentage_error(y_test, arima_preds)
                arima_rmse = np.sqrt(mean_squared_error(y_test, arima_preds))
                arima_mapes.append(arima_mape)
                arima_rmses.append(arima_rmse)
            except Exception as e:
                logger.error("ARIMA failed for {}: {}", ticker, e)
                arima_mapes.append(np.nan)
                arima_rmses.append(np.nan)

            # Linear Regression Model
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_flat)
            X_test_scaled = scaler.transform(X_test_flat)
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            lr_preds = model.predict(X_test_scaled)
            lr_mape = mean_absolute_percentage_error(y_test, lr_preds)
            lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
            lr_mapes.append(lr_mape)
            lr_rmses.append(lr_rmse)

        # Calculate mean metrics across splits
        arima_mape_mean = np.nanmean(arima_mapes)
        arima_rmse_mean = np.nanmean(arima_rmses)
        lr_mape_mean = np.mean(lr_mapes)
        lr_rmse_mean = np.mean(lr_rmses)

        # Save results
        results.append({
            'ticker': ticker,
            'model': 'ARIMA',
            'mape': arima_mape_mean,
            'rmse': arima_rmse_mean
        })
        results.append({
            'ticker': ticker,
            'model': 'Linear Regression',
            'mape': lr_mape_mean,
            'rmse': lr_rmse_mean
        })

        # Generate comparison plot
        plt.figure(figsize=(10, 6))
        models = ['ARIMA', 'Linear Regression']
        mapes = [arima_mape_mean, lr_mape_mean]
        rmses = [arima_rmse_mean, lr_rmse_mean]
        bar_width = 0.35
        x = np.arange(len(models))
        plt.bar(x - bar_width / 2, mapes, bar_width, label='MAPE', color='skyblue')
        plt.bar(x + bar_width / 2, rmses, bar_width, label='RMSE', color='salmon')
        plt.xlabel('Model')
        plt.ylabel('Error Metric')
        plt.title(f'{ticker} Model Performance Comparison')
        plt.xticks(x, models)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"{ticker}_model_comparison.png")
        plt.close()

    # Save all results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(metrics_path, index=False)
    logger.success("Baseline model training and evaluation complete. Results saved to {}", metrics_path)


if __name__ == "__main__":
    app()