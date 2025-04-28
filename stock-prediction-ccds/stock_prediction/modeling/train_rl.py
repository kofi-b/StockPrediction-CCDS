import os
import pandas as pd
import numpy as np
import gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from loguru import logger
import typer

from stock_prediction.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR
from stock_prediction.modeling.envs.stock_trading_env import StockTradingEnv

app = typer.Typer()

def tune_hyperparameters(train_env, val_env, model_type, num_trials=20):
    """Tune hyperparameters using Optuna."""
    def objective(trial):
        if model_type == 'DQN':
            lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            gamma = trial.suggest_float('gamma', 0.9, 0.999)
            model = DQN("MlpPolicy", train_env, learning_rate=lr, gamma=gamma, buffer_size=10000, verbose=0)
        elif model_type == 'PPO':
            lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            n_steps = trial.suggest_int('n_steps', 16, 2048)
            gamma = trial.suggest_float('gamma', 0.9, 0.999)
            model = PPO("MlpPolicy", train_env, learning_rate=lr, n_steps=n_steps, gamma=gamma, verbose=0)
        else:
            raise ValueError("Unsupported model type")
        model.learn(total_timesteps=10000)
        mean_reward, _ = evaluate_policy(model, val_env, n_eval_episodes=5)
        return mean_reward
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=num_trials)
    return study.best_params

def train_and_evaluate(ticker, use_sentiment, model_type, best_params, full_train_env, test_env, total_timesteps=10000):
    """Train and evaluate the RL model."""
    eval_callback = EvalCallback(
        test_env,
        best_model_save_path=str(MODELS_DIR / f"{ticker}_{model_type}_best"),
        log_path=str(REPORTS_DIR / f"{ticker}_{model_type}_eval.log"),
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    if model_type == 'DQN':
        model = DQN(
            "MlpPolicy",
            full_train_env,
            learning_rate=best_params['lr'],
            gamma=best_params['gamma'],
            buffer_size=10000,
            verbose=1
        )
    elif model_type == 'PPO':
        model = PPO(
            "MlpPolicy",
            full_train_env,
            learning_rate=best_params['lr'],
            n_steps=best_params['n_steps'],
            gamma=best_params['gamma'],
            verbose=1
        )
    else:
        raise ValueError("Unsupported model type")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    mean_reward, _ = evaluate_policy(model, test_env, n_eval_episodes=5)
    model.save(MODELS_DIR / f"{ticker}_{model_type}_model")
    return {'mean_reward': mean_reward}

def process_combo(ticker, use_sentiment, model_type, lookback):
    """Process a single combination of ticker, sentiment, and model type."""
    try:
        data_file = PROCESSED_DATA_DIR / "engineered_features" / ticker / f"{ticker}_engineered_data.csv"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        df = pd.read_csv(data_file)
        T = len(df)
        idx_train_end = int(0.6 * T)
        idx_val_end = int(0.8 * T)
        train_df = df.iloc[:idx_train_end]
        val_df = df.iloc[idx_train_end - lookback:idx_val_end]
        test_df = df.iloc[idx_val_end - lookback:]
        full_train_df = df.iloc[:idx_val_end]

        train_env = DummyVecEnv([
            lambda: Monitor(
                StockTradingEnv(ticker, df=train_df, use_sentiment=use_sentiment),
                filename=str(REPORTS_DIR / f"{ticker}_{model_type}_train.log")
            )
        ])
        val_env = DummyVecEnv([
            lambda: Monitor(
                StockTradingEnv(ticker, df=val_df, use_sentiment=use_sentiment),
                filename=str(REPORTS_DIR / f"{ticker}_{model_type}_val.log")
            )
        ])
        test_env = DummyVecEnv([
            lambda: Monitor(
                StockTradingEnv(ticker, df=test_df, use_sentiment=use_sentiment),
                filename=str(REPORTS_DIR / f"{ticker}_{model_type}_test.log")
            )
        ])
        full_train_env = DummyVecEnv([
            lambda: Monitor(
                StockTradingEnv(ticker, df=full_train_df, use_sentiment=use_sentiment),
                filename=str(REPORTS_DIR / f"{ticker}_{model_type}_full_train.log")
            )
        ])

        best_params = tune_hyperparameters(train_env, val_env, model_type)
        metrics = train_and_evaluate(ticker, use_sentiment, model_type, best_params, full_train_env, test_env)
        logger.info(f"Completed {ticker} with model={model_type}, sentiment={use_sentiment}: {metrics}")
        return (ticker, use_sentiment, model_type), metrics
    except Exception as e:
        logger.error(f"Error processing {ticker} with model={model_type}, sentiment={use_sentiment}: {e}")
        return (ticker, use_sentiment, model_type), {'error': str(e)}

@app.command()
def main(lookback: int = 60):
    """Train RL models for stock trading in parallel."""
    tickers = ['AAPL', 'TSLA', 'META', 'AMZN', 'MSFT', 'NVDA', 'CSCO', 'MA']
    sentiment_options = [True, False]
    model_types = ['DQN', 'PPO']
    combos = list(product(tickers, sentiment_options, model_types))
    results = {}

    logger.info(f"Starting training for {len(combos)} combinations")
    with ProcessPoolExecutor() as executor:
        future_results = [
            executor.submit(process_combo, ticker, use_sentiment, model_type, lookback)
            for ticker, use_sentiment, model_type in combos
        ]
        for future in future_results:
            (ticker, use_sentiment, model_type), metrics = future.result()
            results[(ticker, use_sentiment, model_type)] = metrics

    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.index = pd.MultiIndex.from_tuples(results_df.index, names=['ticker', 'use_sentiment', 'model_type'])
    output_file = REPORTS_DIR / "trading_results.csv"
    results_df.to_csv(output_file)
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    app()