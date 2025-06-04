import yfinance as yf
import pandas as pd
from sb3_contrib import RecurrentPPO
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# List of tickers you want to combine

def get_stock_data(ticker, start="2020-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df


def normalize_df_zscore(data):
    data = data.copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean)/(std + 1e-8)
    data.dropna(inplace=True)
    return data


class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.original_df = df
        self.df= normalize_df_zscore(df)
        self.n_steps = len(self.df)
        self.current_step = 0

        #ACTIONS
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell

        #OBSERVATION SPACE
        self.window_size = 14
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.df.shape[1]), dtype=np.float32)

        #SET VARIABLES TO CONTROL REWARDS
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.last_net_worth = self.initial_balance
        self.number_days_hold = 1



    def _next_observation(self):
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
        current_price = float(self.original_df.iloc[self.current_step]["Close"])
        executed_action = 0
        if action == 1 :  # Buy
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

        self.net_worth = self.balance + self.shares_held * current_price
        if self.shares_held > 0:
            self.number_days_hold =min(max(self.number_days_hold*(self.number_days_hold +1), 1), 100_000_000)

        reward = (self.net_worth - self.last_net_worth)*10/self.number_days_hold
        self.last_net_worth = self.net_worth
        self.net_worths.append(self.net_worth)
        self.actions.append(executed_action)

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # Or set appropriate truncation logic

        return self._next_observation(), reward, terminated, truncated, {
            "net_worth": self.net_worth,
            "shares_held": self.shares_held,
            "balance": self.balance,
            "actions": self.actions,
        }



    def reset(self, *, seed=None, options=None):
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
        if not hasattr(self, "net_worths") or len(self.net_worths) == 0:
            print("No data to render yet.")
            return

        prices = self.original_df['Close'].iloc[self.window_size:self.window_size+len(self.net_worths)].reset_index(drop=True)
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
        plt.savefig("net_worth_plot.png")


partial_df = get_stock_data(ticker="GES", start="2024-01-01", end="2024-10-01")


env = StockTradingEnv(partial_df)
obs , _ = env.reset()

model = RecurrentPPO.load("models/ppo_multiasset_trader_1", device="cuda")

terminated = False
truncated = False

while not truncated and not terminated :
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, results = env.step(action)


env.render()