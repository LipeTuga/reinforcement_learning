import yfinance as yf
import pandas as pd
from sb3_contrib import RecurrentPPO
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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
        plt.show()




tickers = [ "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "UNH",
    "LLY", "JPM", "V", "AVGO", "JNJ", "XOM", "WMT", "PG", "MA", "HD",
    "MRK", "CVX", "ABBV", "COST", "PEP", "ADBE", "CRM", "MCD", "ACN", "BAC",
    "AMD", "TMO", "DIS", "LIN", "CSCO", "NFLX", "ABT", "WFC", "DHR", "VZ",
    "KO", "AMGN", "INTU", "TXN", "NEE", "PM", "ORCL", "CMCSA", "BMY", "NKE",
    "HON", "UPS", "SBUX", "QCOM", "LOW", "RTX", "UNP", "ISRG", "CAT", "INTC",
    "SPGI", "GS", "ELV", "MS", "AMT", "MDT", "LMT", "CVS", "DE", "TJX",
    "AXP", "BA", "BLK", "PLD", "GE", "TGT", "IBM", "NOW", "ADP", "MU",
    "SYK", "PGR", "ZTS", "COP", "GILD", "VRTX", "DUK", "CB", "REGN", "MMC",
    "C", "CSX", "SO", "BDX", "EQIX", "WM", "EOG", "SHW", "MO", "FCX"
]
start = "1990-01-01"
end = "2024-01-01"
dfs = [get_stock_data(ticker, start, end) for ticker in tickers]
common_index = set(dfs[0].index)
for df in dfs[1:]:
    common_index = common_index & set(df.index)
common_index = sorted(list(common_index))
aligned_dfs = [df.loc[common_index].copy() for df in dfs]

# -- Prepare one environment per ticker for DummyVecEnv
def make_env(df):
    #norm_df = normalize_df_zscore(df)
    return lambda: StockTradingEnv(df)

env_fns = [make_env(df) for df in aligned_dfs]
vec_env = DummyVecEnv(env_fns)

model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=1, device="cuda")
model.learn(total_timesteps=50_000_000)
model.save("models/ppo_multiasset_trader")

# -- Inference: predictions per ticker in parallel
obs = vec_env.reset()
terminated = [False] * len(tickers)
truncated = [False] * len(tickers)
while not all(terminated) and not all(truncated):
    actions, _ = model.predict(obs, deterministic=True)
    step_result = vec_env.step(actions)
    if len(step_result) == 5:
        obs, rewards, term, trunc, infos = step_result
        terminated = [a or b for a, b in zip(terminated, term)]
        truncated = [a or b for a, b in zip(truncated, trunc)]
    elif len(step_result) == 4:
        obs, rewards, done, infos = step_result
        terminated = [a or b for a, b in zip(terminated, done)]
        truncated = [False] * len(done)  # Mark as not truncated, or adjust as needed
    else:
        raise RuntimeError("Unexpected number of values returned by vec_env.step()")