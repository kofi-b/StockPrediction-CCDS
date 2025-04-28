from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from newsapi import NewsApiClient
import praw
from sklearn.preprocessing import MinMaxScaler
import requests
import time
from datetime import datetime, timedelta
import nltk
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from loguru import logger
from tqdm import tqdm
import typer

from stock_prediction.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, config

app = typer.Typer()

# Load FinBERT model and tokenizer globally
model_name = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Define stock tickers and date range
stock_tickers = ['AAPL', 'TSLA', 'META', 'AMZN', 'MSFT', 'NVDA', 'CSCO', 'MA']
start_date = '2020-01-01'
end_date = '2025-03-15'
lookback = 60

# Rate limiting tracking
finnhub_calls_made = 0
FINNHUB_CALL_LIMIT = 60  # 60 calls/minute for the free tier

def get_finbert_sentiment(texts):
    """Compute sentiment scores using FinBERT."""
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_scores = probs[:, 2] - probs[:, 0]  # positive - negative
    return sentiment_scores.numpy()

def load_stock_data_yfinance(tickers, start, end, retries=3):
    """Load historical stock data from Yahoo Finance with retry mechanism."""
    for attempt in range(retries):
        try:
            df = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=False)
            df = df.stack(level=0).reset_index()
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            if 'adj_close' not in df.columns and 'adjusted_close' in df.columns:
                df.rename(columns={'adjusted_close': 'adj_close'}, inplace=True)
            logger.info("yfinance columns after load: {}", df.columns.tolist())
            return df
        except Exception as e:
            logger.warning("Attempt {} failed: {}. Retrying...", attempt + 1, e)
            time.sleep(2)
    raise Exception(f"Failed to load data from yfinance after {retries} attempts.")

def load_news_sentiment(tickers, api_key, start_date, end_date, output_file):
    """Load news data and compute sentiment with FinBERT, caching results."""
    if output_file.exists():
        logger.info("Loading news sentiment from {}", output_file)
        df = pd.read_csv(output_file)
        if 'title' in df.columns:
            titles = df['title'].tolist()
            sentiment_scores = get_finbert_sentiment(titles)
            df['sentiment'] = sentiment_scores
            df.to_csv(output_file, index=False)
            logger.info("Updated sentiment scores with FinBERT in {}", output_file)
        return df
    else:
        newsapi = NewsApiClient(api_key=api_key)
        all_articles = []
        all_titles = []
        for ticker in tickers:
            articles = newsapi.get_everything(q=f'{ticker} stock', language='en',
                                              from_param=start_date, to=end_date)
            for article in articles['articles']:
                all_articles.append({
                    'date': article['publishedAt'][:10],
                    'ticker': ticker,
                    'title': article['title']
                })
                all_titles.append(article['title'])
        if all_articles:
            sentiment_scores = get_finbert_sentiment(all_titles)
            for i, article in enumerate(all_articles):
                article['sentiment'] = sentiment_scores[i]
            df = pd.DataFrame(all_articles)
            df.to_csv(output_file, index=False)
            return df
        else:
            logger.info("No news data fetched. Returning empty DataFrame.")
            return pd.DataFrame(columns=['date', 'ticker', 'title', 'sentiment'])

def load_reddit_sentiment(tickers, client_id, client_secret, user_agent, start_date, end_date, output_file):
    """Load Reddit data and compute sentiment with FinBERT, caching results."""
    if output_file.exists():
        logger.info("Loading Reddit sentiment from {}", output_file)
        df = pd.read_csv(output_file)
        if 'title' in df.columns:
            titles = df['title'].tolist()
            sentiment_scores = get_finbert_sentiment(titles)
            df['sentiment'] = sentiment_scores
            df.to_csv(output_file, index=False)
            logger.info("Updated sentiment scores with FinBERT in {}", output_file)
        return df
    else:
        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
        try:
            logger.info("Connected to Reddit as: {}", reddit.user.me() or 'anonymous')
        except Exception as e:
            logger.error("Reddit authentication failed: {}", e)
            return pd.DataFrame()
        subreddits = ['wallstreetbets', 'investing', 'stocks', 'StockMarket']
        all_posts = []
        all_titles = []
        for ticker in tickers:
            for sub_name in subreddits:
                subreddit = reddit.subreddit(sub_name)
                try:
                    for submission in subreddit.search(ticker, limit=100):
                        post_date = pd.to_datetime(submission.created_utc, unit='s').strftime('%Y-%m-%d')
                        if start_date <= post_date <= end_date:
                            all_posts.append({
                                'date': post_date,
                                'ticker': ticker,
                                'title': submission.title,
                                'subreddit': sub_name
                            })
                            all_titles.append(submission.title)
                except Exception as e:
                    logger.warning("Error fetching {} data for {}: {}", sub_name, ticker, e)
                time.sleep(1)
        if all_posts:
            sentiment_scores = get_finbert_sentiment(all_titles)
            for i, post in enumerate(all_posts):
                post['sentiment'] = sentiment_scores[i]
            df = pd.DataFrame(all_posts)
            df.to_csv(output_file, index=False)
            return df
        else:
            return pd.DataFrame(columns=['date', 'ticker', 'title', 'subreddit', 'sentiment'])

def load_finnhub_news(tickers, api_key, start_date, end_date, output_file):
    """Load news from Finnhub and compute sentiment with FinBERT, caching results."""
    global finnhub_calls_made
    if output_file.exists():
        logger.info("Loading Finnhub sentiment from {}", output_file)
        df = pd.read_csv(output_file)
        if 'title' in df.columns:
            titles = df['title'].tolist()
            sentiment_scores = get_finbert_sentiment(titles)
            df['sentiment'] = sentiment_scores
            df.to_csv(output_file, index=False)
            logger.info("Updated sentiment scores with FinBERT in {}", output_file)
        return df
    else:
        all_articles = []
        all_titles = []
        for ticker in tickers:
            if finnhub_calls_made >= FINNHUB_CALL_LIMIT:
                logger.warning("Finnhub API limit reached ({} calls/minute). Skipping {}",
                               FINNHUB_CALL_LIMIT, ticker)
                break
            url = f'https://finnhub.io/api/v1/news?symbol={ticker}&token={api_key}'
            response = requests.get(url)
            if response.status_code == 200:
                finnhub_calls_made += 1
                articles = response.json()
                for article in articles:
                    date = datetime.fromtimestamp(article['datetime']).strftime('%Y-%m-%d')
                    if start_date <= date <= end_date:
                        all_articles.append({
                            'date': date,
                            'ticker': ticker,
                            'title': article['headline']
                        })
                        all_titles.append(article['headline'])
            else:
                logger.warning("Finnhub API error for {}: {} - {}",
                               ticker, response.status_code, response.text)
            time.sleep(1)
        if all_articles:
            sentiment_scores = get_finbert_sentiment(all_titles)
            for i, article in enumerate(all_articles):
                article['sentiment'] = sentiment_scores[i]
            df = pd.DataFrame(all_articles)
            df.to_csv(output_file, index=False)
            return df
        else:
            return pd.DataFrame(columns=['date', 'ticker', 'title', 'sentiment'])

def clean_stock_data(df):
    """Handle missing values, outliers, and inconsistencies in stock data."""
    logger.info("Stock DataFrame columns in clean_stock_data: {}", df.columns.tolist())
    df['date'] = pd.to_datetime(df['date'])
    price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
    df = df.dropna(subset=[price_col])
    df = df.ffill()
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            mean, std = df[col].mean(), df[col].std()
            df[col] = df[col].clip(lower=mean - 3 * std, upper=mean + 3 * std)
    if price_col != 'close' and price_col in df.columns:
        mean, std = df[price_col].mean(), df[price_col].std()
        df[price_col] = df[price_col].clip(lower=mean - 3 * std, upper=mean + 3 * std)
    df = df.sort_values('date').reset_index(drop=True)
    return df

def clean_sentiment_data(df, source='reddit'):
    """Clean sentiment data, handling different sources appropriately."""
    if source == 'reddit':
        expected_cols = ['date', 'ticker', 'sentiment_reddit', 'sentiment_wallstreetbets',
                         'sentiment_investing', 'sentiment_stocks', 'sentiment_StockMarket']
    elif source == 'newsapi':
        expected_cols = ['date', 'ticker', 'sentiment_newsapi']
    elif source == 'finnhub':
        expected_cols = ['date', 'ticker', 'sentiment_finnhub']
    else:
        raise ValueError(f"Unknown source: {source}")

    if df.empty or 'date' not in df.columns or 'title' not in df.columns:
        return pd.DataFrame(columns=expected_cols)

    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['title'])
    df['sentiment'] = df['sentiment'].clip(lower=-1, upper=1)

    if source == 'reddit':
        df_pivot = df.pivot_table(index=['date', 'ticker'], columns='subreddit',
                                  values='sentiment', aggfunc='mean', fill_value=0).reset_index()
        df_pivot.columns = ['date', 'ticker'] + [f'sentiment_{col}' for col in df_pivot.columns[2:]]
        df_unified = df.groupby(['date', 'ticker'])['sentiment'].mean().reset_index(name='sentiment_reddit')
        df_final = df_pivot.merge(df_unified, on=['date', 'ticker'], how='left').fillna(0)
    else:
        df_final = df[['date', 'ticker', 'sentiment']].copy()
        df_final = df_final.groupby(['date', 'ticker'])['sentiment'].mean().reset_index()
        df_final.columns = ['date', 'ticker', f'sentiment_{source}']

    return df_final

def calculate_technical_indicators(df):
    """Add technical indicators like RSI and moving averages."""
    df['sma_20'] = df['adj_close'].rolling(window=20).mean()
    delta = df['adj_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    return df

def normalize_data(df, columns):
    """Normalize numerical columns for LSTM input."""
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

def prepare_time_series(df, lookback=60):
    """Create sequences for LSTM input."""
    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(df.iloc[i-lookback:i][['adj_close', 'sma_20', 'rsi_14', 'volume']].values)
        y.append(df.iloc[i]['adj_close'])
    return np.array(X), np.array(y)

@app.command()
def main():
    """Download, process, and save stock and sentiment data."""
    logger.info("Starting dataset processing...")

    # Load API keys from config
    newsapi_key = config['API_KEYS']['newsapi_key']
    reddit_client_id = config['API_KEYS']['reddit_client_id']
    reddit_client_secret = config['API_KEYS']['reddit_client_secret']
    reddit_user_agent = config['API_KEYS']['reddit_user_agent']
    finnhub_api_key = config['API_KEYS']['finnhub_api_key']

    # Define output files
    processed_file = PROCESSED_DATA_DIR / 'processed_stock_data.csv'
    stock_file = RAW_DATA_DIR / 'stock_data_yf.csv'
    news_file = PROCESSED_DATA_DIR / 'news_sentiment.csv'
    reddit_file = PROCESSED_DATA_DIR / 'reddit_sentiment.csv'
    finnhub_file = PROCESSED_DATA_DIR / 'finnhub_sentiment.csv'

    # Check if processed data exists
    if processed_file.exists():
        logger.info("Loading existing processed data from {}", processed_file)
        merged_df = pd.read_csv(processed_file)
        merged_df['date'] = pd.to_datetime(merged_df['date'])
    else:
        # Load stock data
        logger.info("Fetching stock data...")
        if stock_file.exists():
            stock_df = pd.read_csv(stock_file)
        else:
            stock_df = load_stock_data_yfinance(stock_tickers, start_date, end_date)
            stock_df.to_csv(stock_file, index=False)
        stock_df_clean = clean_stock_data(stock_df)
        merged_df = stock_df_clean

        # Load sentiment data with progress tracking
        logger.info("Fetching sentiment data...")
        with tqdm(total=3, desc="Sentiment Sources") as pbar:
            news_df = load_news_sentiment(stock_tickers, newsapi_key, start_date, end_date, news_file)
            pbar.update(1)
            reddit_df = load_reddit_sentiment(stock_tickers, reddit_client_id, reddit_client_secret,
                                            reddit_user_agent, start_date, end_date, reddit_file)
            pbar.update(1)
            finnhub_df = load_finnhub_news(stock_tickers, finnhub_api_key, start_date, end_date, finnhub_file)
            pbar.update(1)

        # Clean sentiment data
        logger.info("Cleaning sentiment data...")
        news_df_clean = clean_sentiment_data(news_df, source='newsapi')
        reddit_df_clean = clean_sentiment_data(reddit_df, source='reddit')
        finnhub_df_clean = clean_sentiment_data(finnhub_df, source='finnhub')

        # Merge sentiment data
        logger.info("Merging data...")
        if not news_df_clean.empty:
            merged_df = merged_df.drop(columns=['sentiment_newsapi'], errors='ignore').merge(
                news_df_clean, on=['date', 'ticker'], how='left')
        if not reddit_df_clean.empty:
            reddit_cols = ['sentiment_reddit', 'sentiment_wallstreetbets', 'sentiment_investing',
                           'sentiment_stocks', 'sentiment_StockMarket']
            merged_df = merged_df.drop(columns=reddit_cols, errors='ignore').merge(
                reddit_df_clean, on=['date', 'ticker'], how='left')
        if not finnhub_df_clean.empty:
            merged_df = merged_df.drop(columns=['sentiment_finnhub'], errors='ignore').merge(
                finnhub_df_clean, on=['date', 'ticker'], how='left')

        # Calculate combined sentiment
        merged_df['sentiment'] = merged_df[['sentiment_newsapi', 'sentiment_finnhub']].mean(axis=1)

        # Ensure sentiment columns are clean
        sentiment_cols = ['sentiment', 'sentiment_newsapi', 'sentiment_reddit', 'sentiment_wallstreetbets',
                          'sentiment_investing', 'sentiment_stocks', 'sentiment_StockMarket', 'sentiment_finnhub']
        merged_df[sentiment_cols] = merged_df[sentiment_cols].astype(float).fillna(0)

        # Calculate technical indicators
        logger.info("Calculating technical indicators...")
        merged_df = merged_df.groupby('ticker').apply(calculate_technical_indicators).reset_index(drop=True)

        # Normalize data
        logger.info("Normalizing data...")
        columns_to_normalize = ['adj_close', 'sma_20', 'rsi_14', 'volume']
        merged_df, scaler = normalize_data(merged_df, columns_to_normalize)

        # Save processed data
        merged_df.to_csv(processed_file, index=False)
        logger.info("Processed data saved to {}", processed_file)

    # Prepare and save time series data
    logger.info("Preparing time series data...")
    for ticker in stock_tickers:
        df_ticker = merged_df[merged_df['ticker'] == ticker]
        if len(df_ticker) > lookback:
            X, y = prepare_time_series(df_ticker, lookback=lookback)
            np.save(PROCESSED_DATA_DIR / f'{ticker}_X.npy', X)
            np.save(PROCESSED_DATA_DIR / f'{ticker}_y.npy', y)
            logger.info("Time series data saved for {}", ticker)
        else:
            logger.warning("Insufficient data for {} to prepare time series (length: {})", ticker, len(df_ticker))

    logger.success("Dataset processing complete. Finnhub API calls: {}/{}", finnhub_calls_made, FINNHUB_CALL_LIMIT)

if __name__ == "__main__":
    app()