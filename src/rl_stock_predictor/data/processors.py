"""Data processing utilities for stock data."""
import pandas as pd
from typing import List


def align_dataframes(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Align multiple DataFrames to have common indices.

    This is useful when training on multiple stocks that may have different
    trading days (e.g., holidays, different listing dates). The function
    finds the intersection of all indices and filters each DataFrame to
    only include those common dates.

    Args:
        dfs: List of DataFrames to align (must have DatetimeIndex)

    Returns:
        List of aligned DataFrames with common index, sorted chronologically

    Example:
        >>> import pandas as pd
        >>> df1 = pd.DataFrame({'Close': [100, 101, 102]},
        ...                     index=pd.date_range('2020-01-01', periods=3))
        >>> df2 = pd.DataFrame({'Close': [200, 201, 202]},
        ...                     index=pd.date_range('2020-01-02', periods=3))
        >>> aligned = align_dataframes([df1, df2])
        >>> # Returns both DFs with only common dates (2020-01-02, 2020-01-03)
    """
    if not dfs:
        return []

    # Find common index across all dataframes
    common_index = set(dfs[0].index)
    for df in dfs[1:]:
        common_index = common_index & set(df.index)

    # Sort chronologically
    common_index = sorted(list(common_index))

    # Align all dataframes to common index
    aligned_dfs = [df.loc[common_index].copy() for df in dfs]

    return aligned_dfs


def create_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
    """
    Split a DataFrame into training and testing sets based on time.

    Args:
        df: DataFrame to split (must have DatetimeIndex)
        train_ratio: Ratio of data to use for training (default: 0.8)

    Returns:
        Tuple of (train_df, test_df)

    Example:
        >>> df = pd.DataFrame({'Close': range(100)},
        ...                   index=pd.date_range('2020-01-01', periods=100))
        >>> train, test = create_train_test_split(df, train_ratio=0.8)
        >>> # train has first 80 rows, test has last 20 rows
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to a stock DataFrame.

    Indicators added:
    - SMA_20: Simple Moving Average (20 day)
    - SMA_50: Simple Moving Average (50 day)
    - EMA_12: Exponential Moving Average (12 day)
    - EMA_26: Exponential Moving Average (26 day)
    - RSI_14: Relative Strength Index (14 day)
    - MACD: Moving Average Convergence Divergence
    - MACD_Signal: MACD Signal Line (9 day EMA of MACD)
    - MACD_Hist: MACD Histogram (MACD - Signal)
    - BB_Middle: Bollinger Band Middle (20 day SMA)
    - BB_Upper: Bollinger Band Upper (Middle + 2*std)
    - BB_Lower: Bollinger Band Lower (Middle - 2*std)
    - BB_Width: Bollinger Band Width ((Upper - Lower) / Middle)

    Args:
        df: DataFrame with OHLCV columns (must have 'Close' column)

    Returns:
        DataFrame with additional technical indicator columns

    Example:
        >>> df = downloader.download('AAPL', '2020-01-01', '2024-01-01')
        >>> df_with_indicators = add_technical_indicators(df)
        >>> print(df_with_indicators.columns)
    """
    df = df.copy()

    # Handle MultiIndex columns from yfinance (flatten if needed)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Get close prices as a Series
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]  # Take first column if DataFrame

    # =========================================================================
    # MOVING AVERAGES
    # =========================================================================

    # Simple Moving Averages
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()

    # Exponential Moving Averages (for MACD)
    df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = close.ewm(span=26, adjust=False).mean()

    # =========================================================================
    # RSI (Relative Strength Index)
    # =========================================================================
    # RSI measures momentum by comparing recent gains to recent losses
    # RSI = 100 - (100 / (1 + RS)), where RS = Avg Gain / Avg Loss

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Use exponential moving average for smoothing (Wilder's method)
    avg_gain = gain.ewm(com=13, adjust=False).mean()  # com=13 gives span=14
    avg_loss = loss.ewm(com=13, adjust=False).mean()

    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # =========================================================================
    # MACD (Moving Average Convergence Divergence)
    # =========================================================================
    # MACD Line = 12-period EMA - 26-period EMA
    # Signal Line = 9-period EMA of MACD Line
    # Histogram = MACD Line - Signal Line

    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # =========================================================================
    # BOLLINGER BANDS
    # =========================================================================
    # Middle Band = 20-period SMA
    # Upper Band = Middle + (2 × 20-period std)
    # Lower Band = Middle - (2 × 20-period std)

    bb_window = 20
    bb_std = 2

    df['BB_Middle'] = close.rolling(window=bb_window).mean()
    rolling_std = close.rolling(window=bb_window).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * rolling_std)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * rolling_std)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

    return df


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index) for a price series.

    Args:
        series: Price series (typically Close prices)
        period: RSI period (default: 14)

    Returns:
        Series with RSI values (0-100)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        series: Price series (typically Close prices)
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> tuple:
    """
    Calculate Bollinger Bands.

    Args:
        series: Price series (typically Close prices)
        window: Rolling window period (default: 20)
        num_std: Number of standard deviations (default: 2)

    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    return middle, upper, lower
