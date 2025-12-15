"""Portfolio performance metrics for investment analysis."""
import numpy as np
from typing import List, Union


def calculate_returns(net_worths: List[float]) -> np.ndarray:
    """
    Calculate daily returns from net worth history.

    Args:
        net_worths: List of net worth values over time

    Returns:
        Array of daily returns (percentage change)

    Example:
        >>> net_worths = [10000, 10100, 10050, 10200]
        >>> returns = calculate_returns(net_worths)
        >>> # Returns: [0.01, -0.00495, 0.01493]
    """
    net_worths_array = np.array(net_worths)
    returns = np.diff(net_worths_array) / net_worths_array[:-1]
    return returns


def calculate_sharpe_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe Ratio (risk-adjusted return).

    Sharpe Ratio = (Annualized Return - Risk-Free Rate) / Annualized Volatility

    Args:
        returns: Array of daily returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Trading days per year (default: 252)

    Returns:
        Sharpe ratio value (higher is better, >1 is good, >2 is excellent)

    Example:
        >>> returns = np.array([0.01, -0.005, 0.015, 0.002])
        >>> sharpe = calculate_sharpe_ratio(returns)
    """
    if len(returns) == 0:
        return 0.0

    returns_array = np.array(returns)

    # Annualized return
    mean_return = np.mean(returns_array)
    annualized_return = mean_return * periods_per_year

    # Annualized volatility
    volatility = np.std(returns_array) * np.sqrt(periods_per_year)

    if volatility == 0:
        return 0.0

    sharpe_ratio = (annualized_return - risk_free_rate) / volatility
    return float(sharpe_ratio)


def calculate_max_drawdown(net_worths: List[float]) -> float:
    """
    Calculate maximum drawdown (largest peak-to-trough decline).

    Max Drawdown = (Trough Value - Peak Value) / Peak Value

    Args:
        net_worths: List of net worth values over time

    Returns:
        Maximum drawdown as a negative percentage (e.g., -0.15 for 15% drawdown)

    Example:
        >>> net_worths = [10000, 12000, 11000, 9000, 11000]
        >>> max_dd = calculate_max_drawdown(net_worths)
        >>> # Returns: -0.25 (25% drawdown from 12000 to 9000)
    """
    if len(net_worths) == 0:
        return 0.0

    net_worths_array = np.array(net_worths)

    # Calculate running maximum
    running_max = np.maximum.accumulate(net_worths_array)

    # Calculate drawdown at each point
    drawdown = (net_worths_array - running_max) / running_max

    # Return the maximum drawdown (most negative value)
    max_drawdown = np.min(drawdown)
    return float(max_drawdown)


def calculate_volatility(
    returns: Union[List[float], np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized volatility (standard deviation of returns).

    Args:
        returns: Array of daily returns
        periods_per_year: Trading days per year (default: 252)

    Returns:
        Annualized volatility as a percentage

    Example:
        >>> returns = np.array([0.01, -0.005, 0.015, 0.002])
        >>> vol = calculate_volatility(returns)
    """
    if len(returns) == 0:
        return 0.0

    returns_array = np.array(returns)
    daily_std = np.std(returns_array)
    annualized_vol = daily_std * np.sqrt(periods_per_year)
    return float(annualized_vol)


def calculate_sortino_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino Ratio (like Sharpe but only penalizes downside volatility).

    Sortino Ratio = (Annualized Return - Risk-Free Rate) / Downside Deviation

    Args:
        returns: Array of daily returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Trading days per year (default: 252)

    Returns:
        Sortino ratio value (higher is better)

    Example:
        >>> returns = np.array([0.01, -0.005, 0.015, 0.002])
        >>> sortino = calculate_sortino_ratio(returns)
    """
    if len(returns) == 0:
        return 0.0

    returns_array = np.array(returns)

    # Annualized return
    mean_return = np.mean(returns_array)
    annualized_return = mean_return * periods_per_year

    # Downside deviation (only consider negative returns)
    negative_returns = returns_array[returns_array < 0]

    if len(negative_returns) == 0:
        # No negative returns, use regular std
        downside_dev = np.std(returns_array) * np.sqrt(periods_per_year)
    else:
        downside_dev = np.std(negative_returns) * np.sqrt(periods_per_year)

    if downside_dev == 0:
        return 0.0

    sortino_ratio = (annualized_return - risk_free_rate) / downside_dev
    return float(sortino_ratio)


def calculate_calmar_ratio(
    returns: Union[List[float], np.ndarray],
    max_drawdown: float,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar Ratio (return / max drawdown).

    Calmar Ratio = Annualized Return / Absolute(Max Drawdown)

    Args:
        returns: Array of daily returns
        max_drawdown: Maximum drawdown (as negative value, e.g., -0.15)
        periods_per_year: Trading days per year (default: 252)

    Returns:
        Calmar ratio value (higher is better, >1 is good)

    Example:
        >>> returns = np.array([0.01, -0.005, 0.015, 0.002])
        >>> max_dd = -0.15
        >>> calmar = calculate_calmar_ratio(returns, max_dd)
    """
    if len(returns) == 0 or max_drawdown == 0:
        return 0.0

    returns_array = np.array(returns)

    # Annualized return
    mean_return = np.mean(returns_array)
    annualized_return = mean_return * periods_per_year

    # Calmar ratio
    calmar = annualized_return / abs(max_drawdown)
    return float(calmar)
