"""Trading-specific statistics for RL model analysis."""
import numpy as np
from typing import List, Dict


def calculate_trading_stats(
    actions: List[int],
    net_worths: List[float],
    prices: List[float]
) -> Dict[str, float]:
    """
    Calculate comprehensive trading statistics.

    Args:
        actions: List of actions taken (0=hold, 1=buy, 2=sell)
        net_worths: List of net worth values at each step
        prices: List of asset prices at each step

    Returns:
        Dictionary containing:
        - total_trades: Total number of buy+sell actions
        - num_buys: Number of buy actions
        - num_sells: Number of sell actions
        - win_rate: Percentage of profitable trades
        - profit_factor: Gross profit / Gross loss
        - avg_profit_per_trade: Average profit when selling
        - avg_loss_per_trade: Average loss when selling
        - avg_holding_period: Average days held between buy and sell

    Example:
        >>> actions = [1, 0, 0, 2, 1, 2]  # Buy, hold, hold, sell, buy, sell
        >>> net_worths = [10000, 10100, 10200, 10300, 10200, 10400]
        >>> prices = [100, 101, 102, 103, 102, 104]
        >>> stats = calculate_trading_stats(actions, net_worths, prices)
    """
    actions_array = np.array(actions)
    net_worths_array = np.array(net_worths)

    # Count actions
    num_buys = np.sum(actions_array == 1)
    num_sells = np.sum(actions_array == 2)
    total_trades = num_buys + num_sells

    # Track trades for profit analysis
    buy_indices = np.where(actions_array == 1)[0]
    sell_indices = np.where(actions_array == 2)[0]

    profits = []
    losses = []
    holding_periods = []

    # Match buys with sells
    current_buy_idx = None
    for i, action in enumerate(actions_array):
        if action == 1:  # Buy
            current_buy_idx = i
        elif action == 2 and current_buy_idx is not None:  # Sell after buy
            # Calculate profit/loss
            net_worth_at_buy = net_worths_array[current_buy_idx]
            net_worth_at_sell = net_worths_array[i]
            pnl = net_worth_at_sell - net_worth_at_buy

            if pnl > 0:
                profits.append(pnl)
            else:
                losses.append(abs(pnl))

            # Calculate holding period
            holding_period = i - current_buy_idx
            holding_periods.append(holding_period)

            current_buy_idx = None  # Reset for next trade

    # Calculate statistics
    win_rate = len(profits) / (len(profits) + len(losses)) if (len(profits) + len(losses)) > 0 else 0.0

    total_profit = sum(profits) if profits else 0.0
    total_loss = sum(losses) if losses else 0.0
    profit_factor = total_profit / total_loss if total_loss > 0 else (float('inf') if total_profit > 0 else 0.0)

    avg_profit = np.mean(profits) if profits else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0

    return {
        "total_trades": int(total_trades),
        "num_buys": int(num_buys),
        "num_sells": int(num_sells),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor) if profit_factor != float('inf') else 999.99,
        "avg_profit_per_trade": float(avg_profit),
        "avg_loss_per_trade": float(avg_loss),
        "avg_holding_period": float(avg_holding_period),
        "num_winning_trades": len(profits),
        "num_losing_trades": len(losses),
    }
