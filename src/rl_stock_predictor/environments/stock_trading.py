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
    Custom Gym environment for SHORT-ONLY stock trading with reinforcement learning.

    The environment simulates short selling where an agent profits when prices DROP.
    Short selling: Borrow shares → Sell at current price → Buy back later → Return shares

    Action Space:
        Discrete(3):
            0 - Hold: Do nothing
            1 - Short: Open short position (borrow and sell shares)
            2 - Cover: Close short position (buy back shares)

    Observation Space:
        Box(window_size, num_features): A sliding window of historical OHLCV data

    Reward:
        Based on change in net worth (profits when price goes DOWN)
    """

    def __init__(self, df, normalize_fn=None, transaction_cost=0.001, margin_requirement=0.5):
        """
        Initialize the trading environment.

        Args:
            df: DataFrame with OHLCV columns
            normalize_fn: Optional function to normalize the data
            transaction_cost: Transaction cost as a fraction (default 0.1%)
            margin_requirement: Fraction of balance to use as margin (default 50%)
        """
        super(StockTradingEnv, self).__init__()
        self.original_df = df.copy()
        if normalize_fn is None:
            # Default: do not normalize
            normalize_fn = lambda x: x
        self.df = normalize_fn(df)

        self.n_steps = len(self.df)
        self.current_step = 0

        # ACTIONS: 0=hold, 1=short, 2=cover
        self.action_space = spaces.Discrete(3)

        # OBSERVATION SPACE
        self.window_size = 14
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.df.shape[1]),
            dtype=np.float32
        )

        # TRADING PARAMETERS
        self.transaction_cost = transaction_cost  # 0.1% per trade
        self.margin_requirement = margin_requirement  # Use 50% of balance for shorting
        self.min_net_worth = 1000  # Bankruptcy threshold (10% of initial)

        # SHORT TRADING VARIABLES
        self.initial_balance = 10000
        self.balance = self.initial_balance  # Cash (increases when shorting)
        self.shares_shorted = 0              # Number of shares we owe
        self.short_entry_price = 0           # Price at which we entered short
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.last_net_worth = self.initial_balance
        self.number_days_hold = 1
        self.cumulative_reward = 0
        self.total_trades = 0
        self.profitable_trades = 0

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
            action: Action to take (0=hold, 1=short, 2=cover)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        current_price = float(self.original_df.iloc[self.current_step]["Close"])
        executed_action = 0
        trade_cost = 0
        invalid_action_penalty = 0

        if action == 1:  # Short (open short position)
            if self.shares_shorted == 0 and self.balance > 0:
                # Use margin_requirement of balance for shorting (default 50%)
                margin_used = self.balance * self.margin_requirement
                self.shares_shorted = margin_used / current_price
                self.short_entry_price = current_price

                # Receive cash from selling borrowed shares
                sale_proceeds = self.shares_shorted * current_price
                trade_cost = sale_proceeds * self.transaction_cost
                self.balance = self.balance + sale_proceeds - trade_cost

                self.total_trades += 1
                executed_action = 1
            else:
                # Invalid: trying to short when already in position
                invalid_action_penalty = -0.5

        elif action == 2:  # Cover (close short position)
            if self.shares_shorted > 0:
                # Buy back shares to close short position
                cost_to_cover = self.shares_shorted * current_price
                trade_cost = cost_to_cover * self.transaction_cost
                self.balance = self.balance - cost_to_cover - trade_cost

                # Track profitable trades
                if current_price < self.short_entry_price:
                    self.profitable_trades += 1

                self.shares_shorted = 0
                self.short_entry_price = 0
                self.total_trades += 1
                executed_action = 2
                self.number_days_hold = 1
            else:
                # Invalid: trying to cover when no position open
                invalid_action_penalty = -0.5

        # Track holding days for short position
        if self.shares_shorted > 0:
            self.number_days_hold += 1
        else:
            self.number_days_hold = 1

        # Calculate net worth for short position:
        # net_worth = cash - liability (what we owe = shares_shorted * current_price)
        short_liability = self.shares_shorted * current_price
        self.net_worth = self.balance - short_liability

        # Check for bankruptcy
        bankrupt = self.net_worth < self.min_net_worth

        # IMPROVED REWARD FUNCTION
        reward = self._calculate_reward(current_price, executed_action, trade_cost)
        reward += invalid_action_penalty  # Penalize invalid actions

        self.cumulative_reward += reward
        self.last_net_worth = self.net_worth
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        self.net_worths.append(self.net_worth)
        self.actions.append(executed_action)

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1 or bankrupt
        truncated = False

        # Calculate win rate for info
        win_rate = self.profitable_trades / max(1, self.total_trades // 2)  # Divide by 2 since each round trip is 2 trades

        return self._next_observation(), reward, terminated, truncated, {
            "net_worth": self.net_worth,
            "shares_shorted": self.shares_shorted,
            "short_liability": short_liability,
            "cumulative_reward": self.cumulative_reward,
            "balance": self.balance,
            "actions": self.actions,
            "current_price": current_price,
            "bankrupt": bankrupt,
            "total_trades": self.total_trades,
            "win_rate": win_rate,
        }

    def _calculate_reward(self, current_price, executed_action, trade_cost):
        """
        Calculate reward with improved signal quality.

        Reward components:
        1. Base reward: percentage change in net worth (scaled)
        2. Trade cost penalty: discourages excessive trading
        3. Holding penalty: stronger penalty for holding too long
        4. Profit bonus: extra reward for profitable trades
        5. Drawdown penalty: penalize large losses from peak
        """
        # 1. Base reward: percentage change in net worth
        if self.last_net_worth > 0:
            pct_change = (self.net_worth - self.last_net_worth) / self.last_net_worth
        else:
            pct_change = 0

        # Scale reward to reasonable range (multiply by 10 instead of 100)
        base_reward = pct_change * 10

        # 2. Trade cost penalty (already reflected in net worth, but add small explicit penalty)
        trade_penalty = -0.01 if executed_action > 0 else 0

        # 3. Holding penalty: stronger penalty after 5 days
        holding_penalty = 0
        if self.shares_shorted > 0 and self.number_days_hold > 5:
            holding_penalty = -0.05 * (self.number_days_hold - 5)

        # 4. Profit bonus for successful cover
        profit_bonus = 0
        if executed_action == 2 and self.net_worth > self.last_net_worth:
            profit_bonus = 0.5  # Bonus for profitable trade

        # 5. Drawdown penalty: penalize being far below peak
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        drawdown_penalty = -drawdown * 0.1 if drawdown > 0.1 else 0  # Only penalize >10% drawdown

        reward = base_reward + trade_penalty + holding_penalty + profit_bonus + drawdown_penalty

        return reward

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
        self.shares_shorted = 0
        self.short_entry_price = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.last_net_worth = self.initial_balance
        self.number_days_hold = 1
        self.cumulative_reward = 0
        self.total_trades = 0
        self.profitable_trades = 0
        self.net_worths = []
        self.actions = []
        return self._next_observation(), {}

    def render(self, mode='human'):
        """
        Render the environment by plotting net worth over time.

        Saves a plot showing:
        - Net worth trajectory
        - Short/Cover/Hold actions as colored markers
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

        # Show short/cover points
        for i, action in enumerate(self.actions):
            if action == 1:  # Short (open position)
                plt.scatter(i, self.net_worths[i], marker='v', color='red', s=100, label='Short' if i == 0 else '')
            elif action == 2:  # Cover (close position)
                plt.scatter(i, self.net_worths[i], marker='^', color='green', s=100, label='Cover' if i == 0 else '')

        plt.title("Net Worth Over Time (Short-Only Trading)")
        plt.xlabel("Time Step")
        plt.ylabel("Net Worth")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # Save to outputs/plots/ directory
        plot_path = PLOTS_DIR / 'net_worth.png'
        plt.savefig(plot_path)
