import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import typer
from stock_prediction.config import REPORTS_DIR

app = typer.Typer()

def visualize_rl_results():
    """Generate visualizations from aggregated RL results."""
    results_path = REPORTS_DIR / "aggregated_rl_results.csv"
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return

    df = pd.read_csv(results_path)
    plot_df = df[df['model_type'].isin(['DQN', 'PPO'])]

    # Bar Chart: Model Performance by Ticker
    plt.figure(figsize=(12, 6))
    sns.barplot(x='ticker', y='mean_reward', hue='model_type', data=plot_df)
    plt.title('Model Performance by Ticker (Test Mean Reward)')
    plt.xlabel('Ticker')
    plt.ylabel('Mean Reward')
    plt.legend(title='Model Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "figures" / "model_performance_by_ticker.png")
    plt.close()

    # Line Plot: Mean Reward by Ticker and Model
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='ticker', y='mean_reward', hue='model_type', markers=True, data=plot_df)
    plt.title('Mean Reward by Ticker and Model')
    plt.xlabel('Ticker')
    plt.ylabel('Mean Reward')
    plt.legend(title='Model Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "figures" / "mean_reward_by_ticker.png")
    plt.close()

    logger.info("Visualizations saved to {}", REPORTS_DIR / "figures")

@app.command()
def main():
    """Run the RL results visualization."""
    visualize_rl_results()

if __name__ == "__main__":
    app()