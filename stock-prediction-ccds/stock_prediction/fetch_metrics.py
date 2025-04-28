import pandas as pd
import yfinance as yf
from loguru import logger
from stock_prediction.config import PROCESSED_DATA_DIR


def fetch_metrics(tickers):
    """Fetch P/E ratio and forward EPS for given tickers."""
    metric_dict = {}
    for ticker in tickers:
        try:
            yf_ticker = yf.Ticker(ticker)
            pe = yf_ticker.info.get('trailingPE', None)
            forward_eps = yf_ticker.info.get('forwardEps', None)
            metric_dict[ticker] = {'pe_ratio': pe, 'forward_eps': forward_eps}
            logger.info(f"Fetched metrics for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching metrics for {ticker}: {e}")
            metric_dict[ticker] = {'pe_ratio': None, 'forward_eps': None}
    return metric_dict


def main():
    # Load processed stock data
    input_path = PROCESSED_DATA_DIR / "processed_stock_data.csv"
    df = pd.read_csv(input_path)

    tickers = df['ticker'].unique()
    metric_dict = fetch_metrics(tickers)

    # Add metrics to dataframe
    df['pe_ratio'] = df['ticker'].map(lambda x: metric_dict[x]['pe_ratio'])
    df['forward_eps'] = df['ticker'].map(lambda x: metric_dict[x]['forward_eps'])

    # Save enriched dataframe
    output_path = PROCESSED_DATA_DIR / "enriched_stock_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Enriched data saved to {output_path}")


if __name__ == "__main__":
    main()