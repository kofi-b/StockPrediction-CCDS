import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import mplfinance as mpf
import numpy as np
from math import pi
from loguru import logger
import typer
from stock_prediction.config import PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()

# Static P/E and Forward EPS from output (should be dynamically fetched or loaded in a real scenario)
metrics = {
    'AAPL': {'pe_ratio': 33.887302, 'forward_eps': 8.31},
    'TSLA': {'pe_ratio': 121.94147, 'forward_eps': 3.24},
    'META': {'pe_ratio': 25.443886, 'forward_eps': 25.3},
    'AMZN': {'pe_ratio': 35.860508, 'forward_eps': 6.15},
    'MSFT': {'pe_ratio': 31.335485, 'forward_eps': 14.95},
    'NVDA': {'pe_ratio': 41.384354, 'forward_eps': 4.12},
    'CSCO': {'pe_ratio': 26.535088, 'forward_eps': 3.9},
    'MA': {'pe_ratio': 38.01441, 'forward_eps': 16.38}
}


def add_metrics_to_df(df, metrics):
    """Add P/E ratio and forward EPS to the dataframe for each ticker."""
    for ticker in metrics:
        df.loc[df['ticker'] == ticker, 'pe_ratio'] = metrics[ticker]['pe_ratio']
        df.loc[df['ticker'] == ticker, 'forward_eps'] = metrics[ticker]['forward_eps']
    return df


def generate_visualizations(df, ticker, output_dir):
    """Generate all visualizations for a given ticker."""
    df_ticker = df[df['ticker'] == ticker].copy()
    sentiment_types = ['sentiment', 'sentiment_newsapi', 'sentiment_reddit', 'sentiment_wallstreetbets',
                       'sentiment_investing', 'sentiment_stocks', 'sentiment_StockMarket', 'sentiment_finnhub']

    output_dir.mkdir(parents=True, exist_ok=True)

    # Time Series Plot of Adjusted Close Price
    plt.figure(figsize=(12, 6))
    plt.plot(df_ticker['date'], df_ticker['adj_close'], label='Adjusted Close')
    plt.title(f'{ticker} Adjusted Close Price Over Time (Scaled 0-1)')
    plt.xlabel('Date')
    plt.ylabel('Price (Scaled)')
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f"{ticker}_price_time_series.png")
    plt.close()

    # Dual Heatmap: Stock Features vs. Sentiment Features
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), height_ratios=[1, 1.5])
    stock_cols = ['adj_close', 'sma_20', 'rsi_14', 'volume']
    corr_matrix_stock = df_ticker[stock_cols].corr()
    sns.heatmap(corr_matrix_stock, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', ax=ax1)
    ax1.set_title(f'{ticker} Stock Features Correlation')
    corr_matrix_sentiment = df_ticker[sentiment_types].corr()
    sns.heatmap(corr_matrix_sentiment, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', ax=ax2)
    ax2.set_title(f'{ticker} Sentiment Features Correlation')
    plt.tight_layout()
    plt.savefig(output_dir / f"{ticker}_dual_heatmap.png")
    plt.close()

    # Sentiment vs. Price Stacked Area Chart
    fig, ax1 = plt.subplots(figsize=(14, 8))
    sentiment_agg = df_ticker[['date', 'sentiment', 'sentiment_reddit']].copy()
    sentiment_agg = sentiment_agg[(sentiment_agg['date'] >= '2024-01-01') & (sentiment_agg['date'] <= '2025-03-15')]
    sentiment_agg = sentiment_agg.set_index('date').reindex(
        pd.date_range('2024-01-01', '2025-03-15', freq='D'), fill_value=0).reset_index()
    ax1.stackplot(sentiment_agg['index'],
                  sentiment_agg['sentiment'], sentiment_agg['sentiment_reddit'],
                  labels=['News (NewsAPI+Finnhub)', 'Reddit (All Subreddits)'],
                  alpha=0.5, colors=['purple', 'blue'])
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Sentiment (-1 to 1)')
    ax1.set_title(f'{ticker} Sentiment and Price Over Time (2024-2025)')
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.get_xticklabels(), rotation=45)
    ax2 = ax1.twinx()
    price_df = df_ticker[(df_ticker['date'] >= '2024-01-01') & (df_ticker['date'] <= '2025-03-15')]
    ax2.plot(price_df['date'], price_df['adj_close'], label='Adjusted Close (Scaled)', color='black', linewidth=2)
    ax2.set_ylabel('Price (Scaled 0-1)')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir / f"{ticker}_sentiment_price_area.png")
    plt.close()

    # Heatmap Over Time
    melted_df = df_ticker.melt(id_vars=['date'], value_vars=sentiment_types,
                               var_name='sentiment_type', value_name='sentiment_value')
    melted_df = melted_df[(melted_df['date'] >= '2024-01-01') & (melted_df['date'] <= '2025-03-15')]
    melted_df['date'] = melted_df['date'].dt.strftime('%Y-%m-%d')
    pivot_df = melted_df.pivot(index='sentiment_type', columns='date', values='sentiment_value')
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_df, cmap='RdYlGn', vmin=-1, vmax=1, cbar_kws={'label': 'Sentiment'})
    plt.title(f'{ticker} Sentiment Heatmap (2024-2025)')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Source')
    plt.xticks(rotation=45)
    plt.gca().set_xticks(plt.gca().get_xticks()[::15])  # Every 30 days (~monthly)
    plt.tight_layout()
    plt.savefig(output_dir / f"{ticker}_sentiment_heatmap.png")
    plt.close()

    # Candlestick Chart with Sentiment Overlay
    plot_df = df_ticker[(df_ticker['date'] >= '2024-01-01') & (df_ticker['date'] <= '2025-03-15')].set_index('date')
    ap = [mpf.make_addplot(plot_df['sentiment_reddit'], color='blue', secondary_y=True)]
    mpf.plot(plot_df, type='candle', addplot=ap, title=f'{ticker} Price with Reddit Sentiment (2024-2025)',
             ylabel='Price (Scaled)', ylabel_lower='Sentiment',
             savefig=str(output_dir / f"{ticker}_candle_sentiment.png"))

    # Radar Chart for Sentiment Profiles
    means = [df_ticker[sent].mean() for sent in sentiment_types]
    angles = [n / float(len(sentiment_types)) * 2 * pi for n in range(len(sentiment_types))]
    angles += angles[:1]
    means += means[:1]
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, means, linewidth=2)
    ax.fill(angles, means, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([s.replace('sentiment_', '').capitalize() for s in sentiment_types])
    plt.title(f'{ticker} Average Sentiment Profile (2024-2025)')
    plt.savefig(output_dir / f"{ticker}_sentiment_radar.png")
    plt.close()

    # P/E and Forward EPS Plot
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(df_ticker['date'], df_ticker['adj_close'], label='Adjusted Close (Scaled)', color='black')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (Scaled 0-1)')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(df_ticker['date'], df_ticker['pe_ratio'], label='Trailing P/E', color='blue', alpha=0.7)
    ax2.plot(df_ticker['date'], df_ticker['forward_eps'], label='Forward EPS', color='green', alpha=0.7, linestyle='--')
    ax2.set_ylabel('P/E Ratio / EPS')
    ax2.legend(loc='upper right')
    plt.title(f'{ticker} Price, Trailing P/E, and Forward EPS Over Time')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f"{ticker}_price_pe_eps.png")
    plt.close()


@app.command()
def main():
    """Generate visualizations for stock data analysis."""
    # Load data
    input_path = PROCESSED_DATA_DIR / "processed_stock_data.csv"
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])

    # Add metrics to dataframe
    df = add_metrics_to_df(df, metrics)

    # Define output directory for figures
    output_dir = REPORTS_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # List of tickers
    tickers = df['ticker'].unique()

    # Generate visualizations for each ticker
    for ticker in tickers:
        logger.info(f"Generating visualizations for {ticker}")
        ticker_output_dir = output_dir / ticker

        generate_visualizations(df, ticker, output_dir)

    logger.info("All visualizations generated successfully.")


if __name__ == "__main__":
    app()