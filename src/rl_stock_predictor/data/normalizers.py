"""Data normalization functions for stock trading environments."""
import pandas as pd


def normalize_df_zscore(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV columns using z-score normalization.

    Z-score normalization transforms each column to have mean=0 and std=1,
    making the data more suitable for neural network training.

    Args:
        data: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)

    Returns:
        Normalized DataFrame with same structure as input

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'Open': [100, 101, 102],
        ...     'High': [105, 106, 107],
        ...     'Low': [99, 100, 101],
        ...     'Close': [102, 103, 104],
        ...     'Volume': [1000000, 1100000, 1200000]
        ... })
        >>> normalized_df = normalize_df_zscore(df)
    """
    data = data.copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / (std + 1e-8)
    data.dropna(inplace=True)
    return data


def normalize_df_minmax(data: pd.DataFrame, feature_range: tuple = (0, 1)) -> pd.DataFrame:
    """
    Normalize OHLCV columns using min-max normalization.

    Min-max normalization scales each column to a specified range,
    typically [0, 1].

    Args:
        data: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)
        feature_range: Tuple (min, max) specifying the target range

    Returns:
        Normalized DataFrame with same structure as input
    """
    data = data.copy()
    min_val, max_val = feature_range

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        col_min = data[col].min()
        col_max = data[col].max()
        data[col] = (data[col] - col_min) / (col_max - col_min + 1e-8) * (max_val - min_val) + min_val

    data.dropna(inplace=True)
    return data


def normalize_df_robust(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV columns using robust scaler (median and IQR).

    Robust scaling uses median and interquartile range, making it more
    resistant to outliers than z-score normalization.

    Args:
        data: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)

    Returns:
        Normalized DataFrame with same structure as input
    """
    data = data.copy()

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        median = data[col].median()
        q25 = data[col].quantile(0.25)
        q75 = data[col].quantile(0.75)
        iqr = q75 - q25
        data[col] = (data[col] - median) / (iqr + 1e-8)

    data.dropna(inplace=True)
    return data
