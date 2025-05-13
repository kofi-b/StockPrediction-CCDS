# stock-prediction-ccds

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

The overall goal of this project is to develop a data-driven approach to predict short-to-medium-term stock price movements and optimize portfolio allocation for a set of technology stocks
# Stock Prediction Project

## Project Overview
This project predicts stock prices by integrating traditional financial metrics, sentiment analysis from news and social media, and advanced machine learning techniques, including reinforcement learning (RL). Designed for modularity and reproducibility, it follows the [Cookiecutter Data Science (CCDS)](https://drivendata.github.io/cookiecutter-data-science/) structure. The pipeline spans data collection, preprocessing, feature engineering, modeling, and evaluation, delivering actionable insights for stock trading strategies.

## Key Features
- **Data Sources**: Stock data from Yahoo Finance; sentiment from NewsAPI, Reddit, and Finnhub.
- **Feature Engineering**: Technical indicators (RSI, SMA, Bollinger Bands) and sentiment scores.
- **Models**: Baseline (ARIMA, Linear Regression), deep learning (LSTM), and RL (DQN, PPO).
- **Outputs**: Visualizations, performance metrics, and trading strategy evaluations.

## Project Structure
- **`data/`**: Organized into `raw/`, `interim/`, `processed/`, and `external/` subdirectories to track data stages.
- **`stock_prediction/`**: Core scripts for the pipeline.
  - `dataset.py`: Data collection and preprocessing.
  - `features.py`: Feature generation.
  - `modeling/`: Model training and evaluation scripts (e.g., `train.py`, `train_lstm.py`, `train_rl.py`).
  - `plots.py`: Visualization tools.
- **`reports/`**: Stores figures, logs, and results.
- **`models/`**: Saves trained models.

## Replicating the Pipeline
1. **Configure**: Set API keys in `config.ini`.
2. **Collect Data**: Run `dataset.py`.
3. **Engineer Features**: Execute `features.py`.
4. **Train Models**: Use `train.py`, `train_lstm.py`, and `train_rl.py`.
5. **Evaluate**: Visualize results with `plots.py`.

For a detailed guide, see [Data Pipeline Guide](datapipeline.md).

## Project Organization

```
├── LICENSE.md         <- All rights reserved license for this project
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         stock_prediction and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── stock_prediction   <- Source code for use in this project.
    ├── __init__.py             <- Makes stock_prediction a Python module
    ├── config.py               <- Store useful variables and configuration
    ├── config.ini              <- Configuration file for API keys and other settings
    ├── dataset.py              <- Scripts to download or generate data
    ├── features.py             <- Code to create features for modeling
    ├── plots.py                <- Code to create visualizations
    ├── analysis
    │   └── financial_analysis.py  <- Code for financial metrics calculation (CAPM, DCF)
    └── modeling                
        ├── __init__.py 
        ├── aggregate_rl_results.py  <- Code to aggregate RL model results
        ├── predict.py          <- Code to run model inference with trained models          
        ├── pretrain_lstm.py    <- Code to pretrain LSTM model on S&P 500 data
        ├── train.py            <- Code to train baseline models (ARIMA, Linear Regression)
        ├── train_lstm.py       <- Code to train LSTM models
        ├── train_rl.py         <- Code to train RL models (DQN, PPO)
        ├── visualize_rl.py     <- Code to visualize RL model results
        ├── envs
        │   └── stock_trading_env.py  <- Custom Gym environment for stock trading
        └── utils.py            <- Utility functions and classes for modeling (e.g., StockLSTM)
```

## License

This project is licensed under an **All Rights Reserved** license.  
See [LICENSE.md](./LICENSE.md) for full details.

--------

