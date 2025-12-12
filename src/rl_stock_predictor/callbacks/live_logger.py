"""Training monitoring callback with live plotting."""
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np

from ..utils.paths import PLOTS_DIR


class LiveLoggerCallback(BaseCallback):
    """
    Custom callback for logging and visualizing training metrics in real-time.

    This callback tracks and plots:
    - Net worth over time
    - Cumulative rewards
    - Number of trades executed
    - Portfolio value

    Plots are saved periodically during training.
    """

    def __init__(self, plot_every: int = 1000, verbose: int = 0):
        """
        Initialize the callback.

        Args:
            plot_every: Save plots every N steps
            verbose: Verbosity level (0 = silent, 1 = info)
        """
        super().__init__(verbose)
        self.plot_every = plot_every
        self.net_worths = []
        self.cum_rewards = []
        self.trades = []
        self.portfolio_values = []
        self.steps = []
        self.total_trades = 0

    def _on_step(self) -> bool:
        """
        Called at each training step.

        Returns:
            True to continue training, False to stop
        """
        # `infos` is either a list of dicts (for VecEnv) or a dict (for regular Gym env)
        infos = self.locals.get("infos", [{}])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        rewards = self.locals.get("rewards", [0])
        reward = rewards[0] if isinstance(rewards, (list, tuple)) else rewards

        if "net_worth" in info:
            self.net_worths.append(info["net_worth"])
        else:
            self.net_worths.append(np.nan)

        if "cumulative_reward" in info:
            self.cum_rewards.append(info["cumulative_reward"])
        else:
            self.cum_rewards.append(np.nan)

        # Custom: track number of trades (buy/sell actions)
        if "actions" in info:
            acts = info["actions"]
            trades = sum(1 for a in acts if a in [1, 2])  # 1 = buy, 2 = sell
            self.trades.append(trades)
            self.total_trades += 1 if acts and acts[-1] in [1, 2] else 0
        else:
            self.trades.append(np.nan)

        # Custom: track portfolio value (use net_worth as approximation, or customize from info!)
        if "balance" in info and "shares_held" in info:
            # If you want to plot "portfolio value" as cash + stock value at each step:
            curr_step = self.locals["env"].envs[0].current_step
            df_prices = self.locals["env"].envs[0].original_df
            close_price = float(df_prices.iloc[curr_step]["Close"])
            port_value = info["balance"] + info["shares_held"] * close_price
            self.portfolio_values.append(port_value)
        else:
            self.portfolio_values.append(np.nan)

        self.steps.append(self.num_timesteps)

        # Live plot
        if self.n_calls % self.plot_every == 0:
            self._plot_metrics()

        return True

    def _plot_metrics(self):
        """Generate and save training metrics plots."""
        plt.figure(figsize=(18, 5))

        plt.subplot(1, 4, 1)
        plt.plot(self.steps, self.net_worths, label="Net Worth")
        plt.title("Net Worth")
        plt.grid()

        plt.subplot(1, 4, 2)
        plt.plot(self.steps, self.cum_rewards, label="Cumulative Reward", color="orange")
        plt.title("Cumulative Reward")
        plt.grid()

        plt.subplot(1, 4, 3)
        plt.plot(self.steps, np.cumsum(self.trades), label="Number of Trades", color="green")
        plt.title("Number of Trades")
        plt.grid()

        plt.subplot(1, 4, 4)
        plt.plot(self.steps, self.portfolio_values, label="Portfolio Value", color="purple")
        plt.title("Portfolio Value")
        plt.grid()

        plt.tight_layout()
        # Save to outputs/plots/ directory
        plot_path = PLOTS_DIR / "training_plot_live.png"
        plt.savefig(plot_path)
        plt.close()
