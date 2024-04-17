import GPy
import joblib
import numpy as np
from typing import Tuple
from data_models.schema_validator import Frequency
from preprocessing.custom_transformers import TimeSeriesMinMaxScaler
from tqdm import tqdm
from config import paths

np.random.seed(1)

MAX_NUM_PRETRAINING_SERIES = 60_000


def calculate_max_N(T: int, D: int, target_ram_gb: float) -> int:
    """
    Calculate the maximum value of N for a numpy array with shape [N, T, D] that
    fits in the given RAM size.

    Args:
        T (int): The size of the second dimension of the array.
        D (int): The size of the third dimension of the array.
        target_ram_gb (float): The target RAM size in gigabytes.

    Returns:
        int: The maximum value of N.
    """
    element_size = np.dtype(np.float32).itemsize  # size of one float32 element in bytes
    target_ram_bytes = target_ram_gb * (1024**3)  # converting gigabytes to bytes

    # Calculating the maximum N that fits within the target RAM
    max_N = target_ram_bytes / (T * D * element_size)

    return int(min(max_N, MAX_NUM_PRETRAINING_SERIES))


def generate_seasonal_factors(
    len_series: int, num_periods: int, len_per_period: int
) -> np.ndarray:
    """
    Generates a vector of seasonal factors for a time series with repeating cycles,
    with neighboring seasonal factors being correlated.

    Args:
        len_series (int): The total length of the time series (T+H).
        num_periods (int): The number of distinct periods in the seasonality pattern.
        len_per_period (int): The length of each period in the seasonality pattern.

    Returns:
        np.ndarray: A vector of seasonal factors of length `len_series`.

    Example:
        >>> generate_seasonal_factors(24, 4, 5)
        [0.5, 0.5, ..., 1.2]
    """
    # Sample `num_periods` factors from a uniform distribution
    factors = np.random.uniform(0.5, 2, num_periods)

    # Calculate the rolling window length (num_periods // 3)
    rolling_window = num_periods // 3
    if rolling_window >= 3:
        # Apply circular rolling average
        factors = np.array(
            [
                np.mean(
                    np.take(
                        factors,
                        range(i - rolling_window // 2, i + rolling_window // 2 + 1),
                        mode="wrap",
                    )
                )
                for i in range(num_periods)
            ]
        )

    # Determine the offset within the first period
    first_period_start_idx = np.random.randint(0, len_per_period)

    # Create the initial partial period from the first_period_start_idx
    initial_partial_period = np.repeat(
        factors[0], len_per_period - first_period_start_idx
    )

    # Adjust the cycle to start from the remaining part of the first period
    remaining_cycle = np.repeat(factors[1:], len_per_period)
    adjusted_first_period = np.repeat(factors[0], first_period_start_idx)

    # Combine the initial partial period with the adjusted cycle
    seasonal_factors = np.concatenate(
        [initial_partial_period, remaining_cycle, adjusted_first_period]
    )
    seasonal_factors = np.tile(
        seasonal_factors, (len_series // len(seasonal_factors) + 1)
    )[:len_series]

    return seasonal_factors


def generate_multiplicative_trend_factors(len_series: int) -> np.ndarray:
    """
    Generates a vector of trend factors for a time series using a multiplicative trend.

    Args:
        len_series (int): The total length of the time series.

    Returns:
        np.ndarray: A vector of trend factors of length `len_series`.

    Example:
        >>> generate_trend_factors(10)
        [1, 1.001, 1.002001, ...]
    """
    # Calculate the minimum and maximum trend percentage factors
    min_trend_perc = np.exp(np.log(0.5e-1) / len_series)
    max_trend_perc = np.exp(np.log(0.5e1) / len_series)

    # Sample a trend percentage from a uniform distribution
    trend_perc = trend_perc = np.random.triangular(
        min_trend_perc, 1, max_trend_perc, size=1
    )

    # Create a vector of trend factors
    trend_factors = np.array([1 * (trend_perc**i) for i in range(len_series)])

    return np.squeeze(trend_factors)


def generate_linear_trend_factors(len_series: int) -> np.ndarray:
    """
    Generates a vector of linear trend factors for a time series.

    Args:
        len_series (int): The total length of the time series (T+H).

    Returns:
        np.ndarray: A vector of linear trend factors of length `len_series`.

    Example:
        >>> generate_linear_trend_factors(10)
        [1.0, 1.03, 1.06, 1.09, ...]
    """
    # Randomly sample a number for the trend change per step
    # Calculate the minimum and maximum trend percentage factors
    max_trend_perc = 3 / len_series

    # Sample a trend percentage from a uniform distribution
    trend_factor = np.random.triangular(-max_trend_perc, 0, max_trend_perc, size=1)

    # Create a vector of trend factors with a positive trend
    trend_factors = np.array([1 + i * trend_factor for i in range(len_series)])

    # With 50% probability, reverse the trend for a downward trend
    if np.random.rand() < 0.5:
        trend_factors = trend_factors[::-1]
    return np.squeeze(trend_factors)


def generate_random_walk(
    len_series: int, mean: float = 1.0, std_dev: float = 0.01
) -> np.ndarray:
    """
    Generates a multiplicative random walk sequence for a time series.

    Args:
        len_series (int): The length of the time series.
        mean (float): The mean of the multiplicative factor (default 1.0).
        std_dev (float): The standard deviation of the multiplicative factor (default 0.05).

    Returns:
        np.ndarray: A multiplicative random walk sequence of length `len_series`.

    Example:
        >>> generate_multiplicative_random_walk(10)
        [1, 1.03, 1.05, 1.07, 1.06, ..., 1.09]
    """
    # Initialize the first value of the series
    random_walk = [1]

    # Generate the rest of the series
    for _ in range(1, len_series):
        factor = np.random.normal(mean, std_dev)
        random_walk.append(random_walk[-1] * factor)

    return np.array(random_walk)


def generate_noise(len_series: int, std_dev: float = 0.02) -> np.ndarray:
    """
    Generates a vector of noise for a time series.

    Args:
        len_series (int): The length of the time series.
        std_dev (int): Std Dev of Noise (mean is 1).

    Returns:
        np.ndarray: A vector of noise of length `len_series`.

    Example:
        >>> generate_standard_normal_noise(10)
        [0.56, -1.32, ..., 0.97]
    """
    # Generate the series
    noise = []
    for _ in range(len_series):
        factor = np.random.normal(1, std_dev)
        noise.append(factor)
    return np.array(noise)


def generate_synthetic_data(
    num_series: int, len_series: int, frequency: Frequency
) -> np.ndarray:
    """
    Generates synthetic time series data for a given frequency with optional trend,
    seasonality, and noise.

    Args:
        num_series (int): The number of time series to generate.
        len_series (int): The length of each time series.
        frequency (Frequency): The frequency of the data.

    Returns:
        np.ndarray: A 2D numpy array of shape [num_series, len_series] containing
                    the synthetic time series data.
    """
    synthetic_data = np.ones((num_series, len_series))

    for i in range(num_series):
        # Apply trend
        if np.random.rand() < 0.75:

            # Use multiplicative trend with 50% probability
            if np.random.rand() < 0.5:
                trend_factors = generate_multiplicative_trend_factors(len_series)
            else:
                # Use linear trend with 50% probability
                trend_factors = generate_linear_trend_factors(len_series)
            synthetic_data[i] *= trend_factors

        # Apply seasonality based on frequency
        if np.random.rand() < 0.75:
            seasonal_factors = generate_seasonality_for_frequency(len_series, frequency)
            synthetic_data[i] *= seasonal_factors

        # Apply random walk
        if np.random.rand() < 0.75:
            rw_factors = generate_random_walk(len_series)
            synthetic_data[i] *= rw_factors

        # Apply noise
        if np.random.rand() < 0.75:
            noise_factors = generate_noise(len_series, std_dev=0.02)
            synthetic_data[i] *= noise_factors

    return synthetic_data


def generate_seasonality_for_frequency(
    len_series: int, frequency: Frequency
) -> np.ndarray:
    """
    Generates seasonality factors based on the given frequency.

    Args:
        len_series (int): The length of each time series.
        frequency (Frequency): The frequency of the data.

    Returns:
        np.ndarray: Seasonality factors.
    """
    if frequency == Frequency.QUARTERLY:
        # Assuming quarterly frequency for seasonality (num_periods=4, len_per_period=3)
        return generate_seasonal_factors(len_series, 4, 3)
    elif frequency == Frequency.MONTHLY:
        # Assuming monthly frequency for seasonality (num_periods=12, len_per_period=1)
        return generate_seasonal_factors(len_series, 12, 1)
    elif frequency == Frequency.WEEKLY:
        # Assuming weekly frequency for seasonality (num_periods=52, len_per_period=1)
        return generate_seasonal_factors(len_series, 52, 1)
    elif frequency == Frequency.DAILY:
        # Generate daily level seasonality
        # Example: weekly seasonality within daily data
        rand_num = np.random.rand()
        if rand_num < 0.5:
            # 7-day week
            week_len = 7
        elif rand_num < 0.75:
            # 6-day week
            week_len = 6
        else:
            # 5-day week
            week_len = 5
        # Weekday seasonality for weekly seasonality
        weekday_factors = generate_seasonal_factors(len_series, week_len, 1)
        # Week of year seasonality for annual seasonality
        wk_in_year_factors = generate_seasonal_factors(len_series, 52, week_len)
        return weekday_factors * wk_in_year_factors
    elif frequency == Frequency.HOURLY:
        # Determine the length of the hourly pattern
        rand_num = np.random.rand()
        if rand_num < 0.5:
            # 50% chance for a 24-hour pattern
            hours_in_day = 24
        else:
            # Evenly distribute the remaining 50% among 8 to 23-hour patterns
            hours_in_day = np.random.choice(range(8, 24))

        hourly_factors = generate_seasonal_factors(len_series, hours_in_day, 1)
        hourly_factors *= generate_seasonal_factors(
            len_series, 7, hours_in_day
        )  # 7 days in a week
        hourly_factors *= generate_seasonal_factors(
            len_series, 52, hours_in_day * 7
        )  # 52 weeks in a year
        return hourly_factors
    else:
        # Default case, no seasonality
        return np.ones(len_series)


def generate_with_kernels(
    window_length: int, n_kernels: int = 5, n_samples: int = 500
) -> np.ndarray:
    """
    Generates synthetic time series data for each window in the input data using GPy.

    Parameters:
    - window_length (int): The length of each window.
    - n_kernels (int): The number of kernels to sample for each window.
    - n_samples (int): The number of synthetic samples to generate.

    Returns:
    - np.ndarray: The synthetic time-series data for each window, maintaining the input shape.
    """
    w = window_length
    synthetic_series = np.zeros((n_samples, w))
    used_combinations = set()

    # Defining a variety of kernels
    kernel_bank = [
        GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0),
        GPy.kern.RBF(input_dim=1, variance=2.0, lengthscale=2.0),
        GPy.kern.RBF(input_dim=1, variance=2.0, lengthscale=1.0),
        GPy.kern.RBF(input_dim=1, variance=0.5, lengthscale=1.0),
        GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=0.5),
        GPy.kern.PeriodicExponential(1, lengthscale=1.0, period=1.0),
        GPy.kern.PeriodicExponential(1, lengthscale=1.0, period=2.0),
        GPy.kern.PeriodicExponential(1, lengthscale=1.0, period=0.5),
        GPy.kern.PeriodicExponential(1, lengthscale=2.0, period=1.0),
        GPy.kern.PeriodicExponential(1, lengthscale=2.0, period=2.0),
        GPy.kern.Linear(input_dim=1, variances=1.0),
        GPy.kern.Linear(input_dim=1, variances=2.0),
        GPy.kern.Linear(input_dim=1, variances=0.5),
    ]

    for i in tqdm(
        range(n_samples), desc="Generating synthetic series with GPy kernel synthesizer"
    ):
        # Sample kernels and combine them
        sampled_kernels = np.random.choice(kernel_bank, size=n_kernels, replace=True)
        combined_kernel = sampled_kernels[0]
        for k in sampled_kernels[1:]:
            if np.random.rand() > 0.5:
                combined_kernel += k
            else:
                combined_kernel *= k

        combination_key = "".join(sorted([str(k) for k in sampled_kernels]))

        if combination_key in used_combinations:
            continue  # Skip this iteration if the combination was used before

        # Dummy initial data for model instantiation
        X_dummy = np.linspace(0, w - 1, w).reshape(-1, 1)
        Y_dummy = np.zeros((w, 1))

        # Create a GP model with the combined kernel
        gp = GPy.models.GPRegression(X_dummy, Y_dummy, kernel=combined_kernel)

        # Assuming evenly spaced time points within each window
        time_points = np.linspace(0, w - 1, w).reshape(-1, 1)
        # Sample synthetic data from the GP model for the current window
        Y = gp.posterior_samples_f(time_points, full_cov=False, size=1)
        synthetic_series[i, :] = Y.squeeze()
        if i % 10 == 0:
            joblib.dump(synthetic_series, paths.KERNEL_SYNTH_DATA)
    return synthetic_series


def get_pretraining_data(
    series_len: int,
    forecast_length: int,
    frequency: Frequency,
    num_exog: int = 0,
    scale: bool = True,
) -> np.ndarray:
    """
    Generates synthetic data for pretraining, with optional exogenous features and scaling.

    Args:
        series_len (int): The length of each time series.
        forecast_length (int): The length of forecast window.
        frequency (Frequency): Frequency of the data such as MONTHLY, DAILY, etc.
        num_exog (int): The number of exogenous features to generate.
                        If 0, no exogenous features are added.
        scale (bool): Whether to scale the data or not.

    Returns:
        np.ndarray: A 3D numpy array of shape [num_series, series_len, 1 + num_exog]
                    containing the synthetic time series data and exogenous features.
    """
    # Calculate # of samples to generate
    num_series = calculate_max_N(series_len, 1 + num_exog, target_ram_gb=3.0)
    # Generate base synthetic data
    synthetic_data = generate_synthetic_data(num_series, series_len, frequency)

    # Expand to 3 dimensions
    synthetic_data = synthetic_data[:, :, np.newaxis]

    # Add exogenous features if num_exog is positive
    if num_exog > 0:
        exogenous_features = np.random.standard_normal(
            (num_series, series_len, num_exog)
        )
        synthetic_data = np.concatenate((synthetic_data, exogenous_features), axis=2)

    # Scale data
    synthetic_data = synthetic_data.astype(np.float32)
    if scale:
        scaler = TimeSeriesMinMaxScaler(encode_len=series_len - forecast_length)
        synthetic_data = scaler.fit_transform(synthetic_data)

    return synthetic_data


def get_kernel_pretrain_data(
    data: np.ndarray,
    window_length: int,
    num_windows: int,
    forecast_length: int,
    num_exog: int = 0,
    scale: bool = True,
) -> np.ndarray:
    """
    Sample windows of length `window_length` from the input data.

    Args:
        data (np.ndarray): The input data.
        window_length (int): The length of each window.
        num_windows (int): The number of windows to sample.
        forecast_length (int): The length of forecast window.
        num_exog (int): The number of exogenous features in the data.
        scale (bool): Whether to scale the data or not.

    Returns:
        np.ndarray: A 3D numpy array of shape [num_windows, window_length, num_features]
                    containing the sampled windows.
    """
    num_series, old_window_length = data.shape
    windows = np.zeros((num_windows, window_length))

    for i in range(num_windows):
        series_idx = np.random.randint(0, num_series)
        start_idx = np.random.randint(0, old_window_length - window_length)
        windows[i] = data[series_idx, start_idx : start_idx + window_length]

    windows = windows[:, :, np.newaxis]
    if num_exog > 0:
        exogenous_features = np.random.standard_normal(
            (windows.shape[0], windows.shape[1], num_exog)
        )
        windows = np.concatenate((windows, exogenous_features), axis=2)

    if scale:
        scaler = TimeSeriesMinMaxScaler(encode_len=window_length - forecast_length)
        windows = scaler.fit_transform(windows)

    return windows


def merge_data():
    pre_training_data = get_pretraining_data()


if __name__ == "__main__":

    pre_training_data = generate_with_kernels(
        window_length=1000, n_samples=10000, n_kernels=5
    )

    print(pre_training_data.shape)

    # sampled = sample_windows(
    #     pre_training_data,
    #     window_length=100,
    #     num_windows=10,
    #     forecast_length=5,
    #     num_exog=5,
    # )
    # print(sampled.shape)
    # import matplotlib.pyplot as plt

    # np.random.seed(0)
    # data = sampled[:, :, 0].squeeze().T  # 100 data points, 2 series
    # print(data[:, 0])
    # # Plotting
    # plt.figure(figsize=(10, 5))  # Set the figure size
    # plt.plot(data[:, 6], label="Series 1")  # Plot the first series
    # # plt.plot(data[:, 1], label="Series 2")  # Plot the second series
    # plt.title("Visualization of Two Series from a 2D Array")
    # plt.xlabel("Index")
    # plt.ylabel("Value")
    # plt.legend()  # Add a legend to distinguish the series
    # plt.grid(True)  # Add a grid for better readability
    # plt.show()
