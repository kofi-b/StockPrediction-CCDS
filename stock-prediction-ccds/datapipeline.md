# Data Pipeline Guide

This guide outlines the stock prediction project's data pipeline, enabling you to replicate the process from data collection to model evaluation. It’s structured for clarity and reproducibility, suitable for data scientists, researchers, or stakeholders.

## Project Structure
The project adopts the [Cookiecutter Data Science (CCDS)](https://drivendata.github.io/cookiecutter-data-science/) layout:
- **`data/`**: Subdirectories include `raw/`, `interim/`, `processed/`, and `external/`.
- **`stock_prediction/`**: Core scripts for data handling, feature engineering, modeling, and visualization.
- **`models/`**: Stores trained models.
- **`reports/`**: Contains logs, figures, and evaluation results.

Key files in `stock_prediction/`:
- `dataset.py`: Data collection and preprocessing.
- `features.py`: Feature engineering.
- `modeling/`: Model training scripts (`train.py`, `train_lstm.py`, `train_rl.py`).
- `plots.py`: Visualization tools.

## Replicating the Pipeline

### 1. Configure the Project
- **API Keys**: Add your NewsAPI, Reddit, Finnhub, and Alpha Vantage keys to `config.ini`.
- **Environment**: Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Collect and Preprocess Data
- **Script**: `dataset.py`
- **Purpose**: Pulls stock data from Yahoo Finance and sentiment data from APIs.
- **Run**:
  ```bash
  python stock_prediction/dataset.py
  ```
- **Output**: Processed data in `data/processed/`.
- **Note**: If API rate limits occur, adjust date ranges in `dataset.py`.

### 3. Engineer Features
- **Script**: `features.py`
- **Purpose**: Creates technical indicators (e.g., RSI, SMA) and sentiment features.
- **Run**:
  ```bash
  python stock_prediction/features.py
  ```
- **Output**: Features in `data/processed/engineered_features/`.

### 4. Train Models
Three model types are supported:

#### 4.1. Baseline Models (ARIMA, Linear Regression)
- **Script**: `modeling/train.py`
- **Run**:
  ```bash
  python stock_prediction/modeling/train.py
  ```
- **Output**: Models in `models/`; metrics in `reports/`.

#### 4.2. LSTM Models
- **Script**: `modeling/train_lstm.py`
- **Run**:
  ```bash
  python stock_prediction/modeling/train_lstm.py --tickers AAPL MSFT --use_sentiment True
  ```
- **Output**: Models in `models/lstm/`.

#### 4.3. Reinforcement Learning Models (DQN, PPO)
- **Script**: `modeling/train_rl.py`
- **Run**:
  ```bash
  python stock_prediction/modeling/train_rl.py
  ```
- **Output**: Models in `models/`; logs in `reports/`.

### 5. Evaluate and Visualize Results
- **Script**: `plots.py`
- **Purpose**: Plots model performance and stock trends.
- **Run**:
  ```bash
  python stock_prediction/plots.py
  ```
- **Output**: Figures in `reports/figures/`.
- **RL Visualization**: Use `modeling/visualize_rl.py` for RL-specific plots:
  ```bash
  python stock_prediction/modeling/visualize_rl.py
  ```

### 6. Aggregate RL Results
- **Script**: `modeling/aggregate_rl_results.py`
- **Purpose**: Combines RL test and validation rewards.
- **Run**:
  ```bash
  python stock_prediction/modeling/aggregate_rl_results.py
  ```
- **Output**: Results in `reports/aggregated_rl_results.csv`.

## Troubleshooting
- **Data Errors**: Confirm date ranges align with API limits.
- **Training Issues**: Check for NaN values or adjust hyperparameters.
- **Visualization**: Ensure required data columns exist.

#### Not yet implemented:
For additional help, see the [issue tracker](https://github.com/kofi-b/StockPrediction-CCDS/tree/main/stock-prediction-ccds/issues).