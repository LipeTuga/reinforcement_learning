"""Stock trading environment for reinforcement learning."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..utils.paths import PLOTS_DIR


class StockTradingEnv(gym.Env):
    """
    Custom Gym environment for stock trading with reinforcement learning.

    The environment simulates a simple stock trading scenario where an agent
    can buy, sell, or hold a single stock. The goal is to maximize net worth
    by making profitable trading decisions.

    Action Space:
        Discrete(3):
            0 - Hold: Do nothing
            1 - Buy: Purchase stock with all available cash
            2 - Sell: Sell all held shares

    Observation Space:
        Box(window_size, num_features): A sliding window of historical OHLCV data

    Reward:
        Based on change in net worth, with penalties for holding stocks too long
    """

    def __init__(self, df, normalize_fn=None):
        """
        Initialize the trading environment.

        Args:
            df: DataFrame with OHLCV columns
            normalize_fn: Optional function to normalize the data
        """
        super(StockTradingEnv, self).__init__()
        self.original_df = df.copy()
        if normalize_fn is None:
            # Default: do not normalize
            normalize_fn = lambda x: x
        self.df = normalize_fn(df)

        self.n_steps = len(self.df)
        self.current_step = 0

        # ACTIONS
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell

        # OBSERVATION SPACE
        self.window_size = 14
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.df.shape[1]),
            dtype=np.float32
        )

        # SET VARIABLES TO CONTROL REWARDS
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.last_net_worth = self.initial_balance
        self.number_days_hold = 1
        self.cumulative_reward = 0

    def _next_observation(self):
        """
        Get the next observation (sliding window of OHLCV data).

        Returns:
            numpy array of shape (window_size, num_features)
        """
        start = self.current_step - self.window_size + 1
        end = self.current_step + 1

        # Handle beginning of episode
        if start < 0:
            pad = np.zeros((abs(start), self.df.shape[1]))
            obs = np.vstack((pad, self.df.iloc[0:end].values))
        else:
            obs = self.df.iloc[start:end].values

        return obs.astype(np.float32)

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0=hold, 1=buy, 2=sell)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        current_price = float(self.original_df.iloc[self.current_step]["Close"])
        executed_action = 0

        if action == 1:  # Buy
            if self.balance > 0:
                self.shares_held = self.balance / current_price
                self.balance = 0
                executed_action = 1
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance = self.shares_held * current_price
                self.shares_held = 0
                executed_action = 2
                self.number_days_hold = 1

        if self.shares_held > 0:
            self.number_days_hold += 1  # Increment by 1 each day holding
        else:
            self.number_days_hold = 1  # Reset when not holding

        self.net_worth = self.balance + self.shares_held * current_price
        base_reward = (self.net_worth - self.last_net_worth) * 10

        if self.number_days_hold > 10:
            # Apply a negative penalty proportional to how long it's been held
            reward = -abs(base_reward) * (self.number_days_hold - 10)
        else:
            # Gradually decrease reward as days go by
            reward = base_reward / self.number_days_hold

        self.cumulative_reward += reward

        self.last_net_worth = self.net_worth
        self.net_worths.append(self.net_worth)
        self.actions.append(executed_action)

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # Or set appropriate truncation logic

        return self._next_observation(), reward, terminated, truncated, {
            "net_worth": self.net_worth,
            "shares_held": self.shares_held,
            "cumulative_reward": self.cumulative_reward,
            "balance": self.balance,
            "actions": self.actions,
        }

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options

        Returns:
            Tuple of (initial observation, info dict)
        """
        super().reset(seed=seed)
        self.current_step = self.window_size - 1
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.last_net_worth = self.initial_balance
        self.net_worths = []
        self.actions = []
        return self._next_observation(), {}

    def render(self, mode='human'):
        """
        Render the environment by plotting net worth over time.

        Saves a plot showing:
        - Net worth trajectory
        - Buy/sell/hold actions as colored markers
        """
        if not hasattr(self, "net_worths") or len(self.net_worths) == 0:
            print("No data to render yet.")
            return

        prices = self.original_df['Close'].iloc[
            self.window_size:self.window_size + len(self.net_worths)
        ].reset_index(drop=True)

        plt.figure(figsize=(12, 6))

        # Plot net worth
        plt.plot(self.net_worths, label='Net Worth', color='blue')

        # Optional: show buy/sell points
        for i, action in enumerate(self.actions):
            if action == 1:  # Buy
                plt.scatter(i, self.net_worths[i], marker='^', color='green')
            elif action == 2:  # Sell
                plt.scatter(i, self.net_worths[i], marker='v', color='red')
            elif action == 0:
                plt.scatter(i, self.net_worths[i], marker='o', color='black')

        plt.title("Net Worth Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Net Worth")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # Save to outputs/plots/ directory
        plot_path = PLOTS_DIR / 'net_worth.png'
        plt.savefig(plot_path)
