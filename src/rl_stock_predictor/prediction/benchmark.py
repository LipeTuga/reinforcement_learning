"""Buy-and-hold benchmark for comparison."""
import pandas as pd
from typing import Dict


class BuyAndHoldBenchmark:
    """
    Simple buy-and-hold strategy for benchmarking.

    Strategy:
    - Buy as many shares as possible on the first day
    - Hold throughout the entire period
    - Sell on the last day

    This provides a baseline to compare against the RL model's active trading.
    """

    def __init__(self, initial_balance: float = 10000):
        """
        Initialize benchmark.

        Args:
            initial_balance: Starting capital (default: $10,000)
        """
        self.initial_balance = initial_balance

    def run(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Run buy-and-hold strategy on data.

        Args:
            df: DataFrame with OHLCV data (must have 'Close' column)

        Returns:
            Dictionary with:
            - buy_price: Price on first day
            - sell_price: Price on last day
            - shares_bought: Number of shares purchased
            - final_value: Final portfolio value
            - profit_pct: Profit percentage
            - profit_amount: Profit in dollars

        Example:
            >>> benchmark = BuyAndHoldBenchmark(initial_balance=10000)
            >>> results = benchmark.run(stock_data)
            >>> print(f"Buy-and-hold return: {results['profit_pct']:.2f}%")
        """
        if df.empty or 'Close' not in df.columns:
            return {
                'buy_price': 0.0,
                'sell_price': 0.0,
                'shares_bought': 0.0,
                'final_value': self.initial_balance,
                'profit_pct': 0.0,
                'profit_amount': 0.0,
            }

        # Buy on first day
        buy_price = float(df['Close'].iloc[0])
        shares_bought = self.initial_balance / buy_price

        # Sell on last day
        sell_price = float(df['Close'].iloc[-1])
        final_value = shares_bought * sell_price

        # Calculate profit
        profit_amount = final_value - self.initial_balance
        profit_pct = (profit_amount / self.initial_balance) * 100

        return {
            'buy_price': buy_price,
            'sell_price': sell_price,
            'shares_bought': shares_bought,
            'final_value': final_value,
            'profit_pct': profit_pct,
            'profit_amount': profit_amount,
        }

    def compare(self, model_return: float, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compare model performance to buy-and-hold.

        Args:
            model_return: Model's profit percentage
            df: DataFrame with stock data

        Returns:
            Dictionary with comparison metrics
        """
        benchmark_results = self.run(df)

        outperformance = model_return - benchmark_results['profit_pct']

        return {
            'model_return': model_return,
            'benchmark_return': benchmark_results['profit_pct'],
            'outperformance': outperformance,
            'beat_benchmark': outperformance > 0,
        }
