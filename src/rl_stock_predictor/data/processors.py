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

    Currently adds:
    - SMA (Simple Moving Average) - 20 day
    - RSI (Relative Strength Index) - 14 day

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with additional technical indicator columns

    Note:
        This is a placeholder for future implementation.
        Users can extend this to add their own indicators.
    """
    df = df.copy()

    # Simple Moving Average (20 day)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # Placeholder for RSI and other indicators
    # TODO: Implement RSI, MACD, Bollinger Bands, etc.

    return df
