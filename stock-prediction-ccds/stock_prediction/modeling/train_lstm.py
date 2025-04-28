import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from loguru import logger
import typer

from stock_prediction.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR
from stock_prediction.modeling.utils import StockLSTM, Attention, add_noise

app = typer.Typer()

def train_lstm_model(ticker: str, use_sentiment: bool, device: torch.device, lookback: int = 60, batch_size: int = 16, num_epochs: int = 50, patience: int = 10, learning_rate: float = 0.001, noise_level: float = 0.01):
    """Train and evaluate an LSTM model for a given ticker with or without sentiment features."""
    # Set input and output directories per CCDS structure
    input_dir = PROCESSED_DATA_DIR / "feature_engineering" / ticker
    output_dir = MODELS_DIR / "lstm" / ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed sequences
    sentiment_suffix = "_with_sentiment" if use_sentiment else "_no_sentiment"
    X = np.load(input_dir / f"{ticker}_X{sentiment_suffix}.npy")
    y = np.load(input_dir / f"{ticker}_y.npy")

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    # Split into train (60%), validation (20%), and test (20%) sets
    train_split = int(0.6 * len(X))
    val_split = int(0.8 * len(X))
    X_train, X_val, X_test = X[:train_split], X[train_split:val_split], X[val_split:]
    y_train, y_val, y_test = y[:train_split], y[train_split:val_split], y[val_split:]

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_size = X.shape[2]  # Number of features
    hidden_size = 75
    num_layers = 2
    model = StockLSTM(input_size, hidden_size, num_layers).to(device)

    # Define loss, optimizer, and scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    final_epoch = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # Add noise for data augmentation
            batch_X_noisy = add_noise(batch_X.cpu().numpy(), noise_level=noise_level)
            batch_X_noisy = torch.tensor(batch_X_noisy, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            outputs = model(batch_X_noisy)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item() * batch_X.size(0)
            val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / f"{ticker}_best_lstm_model{sentiment_suffix}.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch {}", epoch + 1)
                final_epoch = epoch + 1
                break
        logger.info("Ticker: {} - Epoch {}, Train Loss: {:.6f}, Val Loss: {:.6f}", ticker, epoch + 1, train_loss, val_loss)
        final_epoch = epoch + 1

    # Evaluate on test set
    model.load_state_dict(torch.load(output_dir / f"{ticker}_best_lstm_model{sentiment_suffix}.pth"))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze().cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        mse = mean_squared_error(y_test_np, y_pred)
        directional_accuracy = np.mean(
            (np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_test_np[1:] - y_test_np[:-1])).astype(float))

    # Save final model and performance notes
    torch.save(model.state_dict(), output_dir / f"{ticker}_lstm_model{sentiment_suffix}.pth")
    with open(output_dir / f"{ticker}_model_notes{sentiment_suffix}.txt", 'w') as f:
        f.write(f"Performance for {ticker} ({'With' if use_sentiment else 'Without'} Sentiment):\n")
        f.write(f"- MSE: {mse:.4f}\n")
        f.write(f"- Directional Accuracy: {directional_accuracy:.4f}\n")
        f.write(f"- Epochs: {final_epoch}\n")
    sentiment_status = "without" if not use_sentiment else "with"
    logger.info("Trained {} {} sentiment. MSE: {:.4f}, Directional Accuracy: {:.4f}.", ticker, sentiment_status, mse,
                directional_accuracy)

@app.command()
def train_all_tickers(
    tickers: list[str] = ['AAPL', 'TSLA', 'META', 'AMZN', 'MSFT', 'NVDA', 'CSCO', 'MA'],
    use_sentiment: bool = True,
    device: str = 'cpu',
):
    """Train LSTM models for all specified tickers."""
    device = torch.device(device)
    for ticker in tickers:
        train_lstm_model(ticker, use_sentiment, device)

if __name__ == "__main__":
    app()