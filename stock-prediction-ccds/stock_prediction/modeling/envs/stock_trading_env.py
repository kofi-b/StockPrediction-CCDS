import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from loguru import logger
from stock_prediction.config import PROCESSED_DATA_DIR, REPORTS_DIR
import matplotlib.pyplot as plt

class StockTradingEnv(gym.Env):
    def __init__(self, ticker: str, use_sentiment: bool = False, lookback: int = 60, transaction_cost: float = 0.001, initial_cash: float = 10000, risk_aversion: float = 0.1):
        super(StockTradingEnv, self).__init__()
        self.ticker = ticker
        self.use_sentiment = use_sentiment
        self.lookback = lookback
        self.transaction_cost = transaction_cost
        self.initial_cash = initial_cash
        self.risk_aversion = risk_aversion

        # Load data from PROCESSED_DATA_DIR
        data_path = PROCESSED_DATA_DIR / "engineered_features" / ticker / f"{ticker}_engineered_data.csv"
        self.df = pd.read_csv(data_path)

        # Load selected features based on sentiment usage
        features_file = "selected_features_with_sentiment.txt" if use_sentiment else "selected_features_no_sentiment.txt"
        with open(PROCESSED_DATA_DIR / "engineered_features" / ticker / features_file, 'r') as f:
            self.feature_cols = [line.strip() for line in f.readlines()]

        # Validate and clean data
        required_cols = ['adj_close'] + self.feature_cols
        self.df = self.df[required_cols].dropna()

        # Fit scaler on the entire data (as per original code)
        self.scaler = StandardScaler().fit(self.df[self.feature_cols])
        self.df[self.feature_cols] = self.scaler.transform(self.df[self.feature_cols])

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(lookback, len(self.feature_cols)), dtype=np.float32)

        # Initialize state variables
        self.current_step = lookback
        self.cash = initial_cash
        self.shares_held = 0
        self.previous_portfolio_value = initial_cash
        self.portfolio_values = [initial_cash]

    def reset(self, *, seed: int = None, options: dict = None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.current_step = self.lookback
        self.cash = self.initial_cash
        self.shares_held = 0
        self.previous_portfolio_value = self.initial_cash
        self.portfolio_values = [self.initial_cash]
        logger.info(f"Environment reset for {self.ticker}")
        return self._get_observation(), {}

    def step(self, action):
        """Take a step in the environment based on the action."""
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {'portfolio_value': self.get_portfolio_value()}

        current_price = self.df['adj_close'].iloc[self.current_step]

        # Execute action
        if action == 1:  # Buy
            cash_to_spend = 0.1 * self.cash  # Buy 10% of available cash
            shares_to_buy = cash_to_spend / (current_price * (1 + self.transaction_cost))
            if shares_to_buy > 0:
                self.shares_held += shares_to_buy
                self.cash -= shares_to_buy * current_price * (1 + self.transaction_cost)
        elif action == 2:  # Sell
            shares_to_sell = 0.1 * self.shares_held  # Sell 10% of held shares
            if shares_to_sell > 0:
                self.shares_held -= shares_to_sell
                self.cash += shares_to_sell * current_price * (1 - self.transaction_cost)

        # Update state
        self.current_step += 1
        portfolio_value = self.get_portfolio_value()
        self.portfolio_values.append(portfolio_value)

        # Compute risk-adjusted reward
        returns = np.diff(self.portfolio_values[-self.lookback:]) / self.portfolio_values[-self.lookback:-1] if len(self.portfolio_values) > 1 else [0]
        volatility = np.std(returns) if len(returns) > 0 else 0
        reward = (portfolio_value - self.previous_portfolio_value) - self.risk_aversion * volatility
        self.previous_portfolio_value = portfolio_value

        done = self.current_step >= len(self.df) - 1 or self.cash < 0
        info = {'portfolio_value': portfolio_value}
        return self._get_observation(), reward, done, False, info

    def _get_observation(self):
        """Get the current observation (sequence of features)."""
        window = self.df.iloc[self.current_step - self.lookback:self.current_step][self.feature_cols]
        return window.values

    def get_portfolio_value(self):
        """Calculate the current portfolio value."""
        current_price = self.df['adj_close'].iloc[self.current_step]
        return self.cash + self.shares_held * current_price

    def render(self, mode='human'):
        """Render the environment by saving a plot of portfolio values."""
        if mode == 'human':
            plot_path = REPORTS_DIR / "figures" / f"{self.ticker}_portfolio.png"
            plt.plot(self.portfolio_values)
            plt.title(f"Portfolio Value Over Time for {self.ticker}")
            plt.xlabel("Time Step")
            plt.ylabel("Portfolio Value")
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved portfolio plot to {plot_path}")