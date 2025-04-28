import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from loguru import logger
import typer
import yfinance as yf
from stock_prediction.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from stock_prediction.modeling.utils import StockLSTM, Attention

app = typer.Typer()

# Configuration
sp500_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
    'JNJ', 'PG', 'DIS', 'MA', 'HD', 'BAC', 'XOM', 'PFE', 'KO', 'CSCO'
]
start_date = "2010-01-01"
end_date = "2020-12-31"
lookback = 60
batch_size = 32
num_epochs = 10
learning_rate = 0.001
hidden_dim = 75
num_layers = 2
device = torch.device("cpu")

def preprocess_ticker(ticker, raw_data_dir, preprocessed_dir):
    """Preprocess data for a single ticker, computing technical indicators and creating sequences."""
    raw_file = raw_data_dir / f"{ticker}.csv"
    X_file = preprocessed_dir / f"{ticker}_X.npy"
    y_file = preprocessed_dir / f"{ticker}_y.npy"
    if X_file.exists() and y_file.exists():
        logger.info(f"Loading preprocessed data for {ticker}")
        return np.load(X_file), np.load(y_file)
    df = pd.read_csv(raw_file, index_col=0)
    df = df[['Close', 'Volume']].rename(columns={'Close': 'adj_close', 'Volume': 'volume'})
    df['price_change'] = df['adj_close'].pct_change()
    df['volatility'] = df['adj_close'].rolling(window=20).std()
    df['sma_20'] = df['adj_close'].rolling(window=20).mean()
    delta = df['adj_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['adj_close_lag1'] = df['adj_close'].shift(1)
    df['volume_lag1'] = df['volume'].shift(1)
    df['price_momentum_5d'] = df['price_change'].rolling(window=5).mean()
    df['volatility_ratio'] = df['volatility'] / df['adj_close'].rolling(window=60).std()
    sma = df['adj_close'].rolling(window=20).mean()
    std = df['adj_close'].rolling(window=20).std()
    df['upper_band'] = sma + 2 * std
    df['lower_band'] = sma - 2 * std
    short_ema = df['adj_close'].ewm(span=12, adjust=False).mean()
    long_ema = df['adj_close'].ewm(span=26, adjust=False).mean()
    df['macd'] = short_ema - long_ema
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    df = df.dropna()
    feature_cols = [
        'adj_close', 'sma_20', 'rsi_14', 'volume', 'price_change', 'volatility',
        'adj_close_lag1', 'volume_lag1', 'price_momentum_5d', 'volatility_ratio',
        'upper_band', 'lower_band', 'macd', 'signal_line'
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(X_scaled[i - lookback:i])
        y.append(df.iloc[i]['price_change'])
    X = np.array(X)
    y = np.array(y)
    np.save(X_file, X)
    np.save(y_file, y)
    logger.info(f"Preprocessed and saved data for {ticker}")
    return X, y

@app.command()
def main():
    """Pre-train an LSTM model on S&P 500 data and save it for transfer learning."""
    # Set up directories
    raw_data_dir = EXTERNAL_DATA_DIR / "sp500_raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir = PROCESSED_DATA_DIR / "sp500_preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "pretrained_lstm.pth"

    # Download data if not present
    logger.info("Checking and downloading S&P 500 data if necessary")
    for ticker in sp500_tickers:
        file_path = raw_data_dir / f"{ticker}.csv"
        if not file_path.exists():
            df = yf.download(ticker, start=start_date, end=end_date)
            df.to_csv(file_path)
            logger.info(f"Downloaded data for {ticker}")

    # Preprocess all tickers and combine data
    logger.info("Preprocessing S&P 500 data")
    all_X = []
    all_y = []
    for ticker in sp500_tickers:
        X, y = preprocess_ticker(ticker, raw_data_dir, preprocessed_dir)
        all_X.append(X)
        all_y.append(y)
    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    # Split into train and validation sets
    logger.info("Splitting data into training and validation sets")
    X_train, X_val, y_train, y_val = train_test_split(all_X, all_y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    input_dim = X_train.shape[2]
    model = StockLSTM(input_dim, hidden_dim, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    logger.info("Starting pre-training")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item() * batch_X.size(0)
            val_loss /= len(val_loader.dataset)

        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Save the pre-trained model
    torch.save(model.state_dict(), model_path)
    logger.success(f"Pre-training complete. Model saved to {model_path}")

if __name__ == "__main__":
    app()