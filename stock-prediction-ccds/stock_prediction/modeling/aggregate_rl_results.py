import os
import pandas as pd
import glob
import re
import optuna
from loguru import logger
import typer
from stock_prediction.config import REPORTS_DIR

app = typer.Typer()

def aggregate_rl_results():
    """Aggregate RL test rewards and Optuna best validation rewards."""
    results = []
    pattern = re.compile(r'(.+?)_(.+?)_test\.log')

    # Find and process test monitor files
    test_monitor_files = glob.glob(str(REPORTS_DIR / '*_test.log'))
    for file in test_monitor_files:
        filename = os.path.basename(file)
        match = pattern.match(filename)
        if match:
            ticker, model_type = match.groups()
            try:
                df = pd.read_csv(file, skiprows=1)  # Monitor CSV format
                mean_reward = df['r'].mean()
                results.append({
                    'ticker': ticker,
                    'model_type': model_type,
                    'mean_reward': mean_reward
                })
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
                results.append({
                    'ticker': ticker,
                    'model_type': model_type,
                    'mean_reward': None
                })

    # Collect Optuna data
    optuna_results = []
    for optuna_db_file in glob.glob(str(REPORTS_DIR / 'optuna_*.db')):
        filename = os.path.basename(optuna_db_file)
        parts = filename.split('_')
        if len(parts) >= 3 and parts[0] == 'optuna':
            model_type = parts[1]
            ticker = parts[2].split('.')[0]
            storage_name = f"sqlite:///{optuna_db_file}"
            try:
                study = optuna.load_study(study_name=f"{model_type}_{ticker}_study", storage=storage_name)
                best_val_reward = study.best_value
                optuna_results.append({
                    'ticker': ticker,
                    'model_type': model_type,
                    'best_val_reward': best_val_reward
                })
            except Exception as e:
                logger.error(f"Error loading study from {optuna_db_file}: {e}")
                optuna_results.append({
                    'ticker': ticker,
                    'model_type': model_type,
                    'best_val_reward': None
                })

    # Merge results
    results_df = pd.DataFrame(results)
    optuna_df = pd.DataFrame(optuna_results)
    final_df = pd.merge(results_df, optuna_df, on=['ticker', 'model_type'], how='left')

    # Save aggregated results
    output_file = REPORTS_DIR / "aggregated_rl_results.csv"
    final_df.to_csv(output_file, index=False)
    logger.info(f"Aggregated RL results saved to {output_file}")
    return final_df

@app.command()
def main():
    """Run the RL results aggregation."""
    aggregate_rl_results()

if __name__ == "__main__":
    app()