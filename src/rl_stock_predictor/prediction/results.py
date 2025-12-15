"""Container for prediction results."""
from dataclasses import dataclass, asdict
from typing import List, Optional
import pandas as pd


@dataclass
class PredictionResult:
    """
    Container for prediction results with performance metrics.

    Attributes:
        ticker: Stock ticker symbol
        start_date: Start date of prediction period
        end_date: End date of prediction period
        initial_balance: Starting balance ($10,000)
        final_net_worth: Final portfolio value
        total_steps: Number of trading days
        cumulative_reward: Sum of all RL rewards

        Performance metrics:
        sharpe_ratio: Risk-adjusted return metric
        max_drawdown: Largest peak-to-trough decline (negative %)
        volatility: Annualized volatility (%)
        sortino_ratio: Downside-risk adjusted return
        calmar_ratio: Return per unit of max drawdown

        Trading statistics:
        total_trades: Total buy + sell actions
        win_rate: Percentage of profitable trades
        profit_factor: Gross profit / Gross loss
        avg_holding_period: Average days between buy and sell
        num_buys: Number of buy actions
        num_sells: Number of sell actions

        Benchmark comparison:
        buy_hold_final_value: Buy-and-hold final value
        buy_hold_profit_pct: Buy-and-hold profit percentage
        model_vs_benchmark: Model outperformance vs buy-and-hold

        Historical data (optional):
        net_worth_history: Full net worth trajectory
        action_history: All actions taken
        price_history: Asset prices at each step
    """

    ticker: str
    start_date: str
    end_date: str
    initial_balance: float
    final_net_worth: float
    total_steps: int
    cumulative_reward: float

    # Performance metrics
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    sortino_ratio: float
    calmar_ratio: float

    # Trading statistics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_holding_period: float
    num_buys: int
    num_sells: int

    # Benchmark comparison
    buy_hold_final_value: float
    buy_hold_profit_pct: float
    model_vs_benchmark: float

    # Historical data (optional, not included in CSV export by default)
    net_worth_history: Optional[List[float]] = None
    action_history: Optional[List[int]] = None
    price_history: Optional[List[float]] = None

    @property
    def profit_pct(self) -> float:
        """Calculate model profit percentage."""
        return ((self.final_net_worth - self.initial_balance) / self.initial_balance) * 100

    @property
    def profit_amount(self) -> float:
        """Calculate model profit amount."""
        return self.final_net_worth - self.initial_balance

    def to_dict(self, include_history: bool = False) -> dict:
        """
        Convert to dictionary for JSON export.

        Args:
            include_history: Whether to include full historical data

        Returns:
            Dictionary representation
        """
        data = asdict(self)

        # Add calculated fields
        data['profit_pct'] = self.profit_pct
        data['profit_amount'] = self.profit_amount

        # Optionally exclude historical data for smaller output
        if not include_history:
            data.pop('net_worth_history', None)
            data.pop('action_history', None)
            data.pop('price_history', None)

        return data

    def to_series(self) -> pd.Series:
        """
        Convert to pandas Series for DataFrame row.

        Returns:
            Pandas Series with key metrics (excludes historical data)
        """
        return pd.Series({
            'ticker': self.ticker,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_balance': self.initial_balance,
            'final_net_worth': self.final_net_worth,
            'profit_amount': self.profit_amount,
            'profit_pct': self.profit_pct,
            'total_steps': self.total_steps,
            'cumulative_reward': self.cumulative_reward,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_holding_period': self.avg_holding_period,
            'num_buys': self.num_buys,
            'num_sells': self.num_sells,
            'buy_hold_final_value': self.buy_hold_final_value,
            'buy_hold_profit_pct': self.buy_hold_profit_pct,
            'model_vs_benchmark': self.model_vs_benchmark,
        })

    def __str__(self) -> str:
        """String representation for printing."""
        return (
            f"PredictionResult({self.ticker})\n"
            f"  Period: {self.start_date} to {self.end_date}\n"
            f"  Final: ${self.final_net_worth:,.2f} ({self.profit_pct:+.2f}%)\n"
            f"  Sharpe: {self.sharpe_ratio:.2f} | Max DD: {self.max_drawdown*100:.1f}%\n"
            f"  vs Buy-Hold: {self.model_vs_benchmark:+.2f}%"
        )
