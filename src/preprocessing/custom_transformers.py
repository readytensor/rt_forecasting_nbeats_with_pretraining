from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.gaussian_process.kernels import (
    RBF,
    ExpSineSquared,
    DotProduct,
    WhiteKernel,
)
from sklearn.gaussian_process import GaussianProcessRegressor


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects or drops specified columns."""

    def __init__(self, columns: List[str], selector_type: str = "keep"):
        """
        Initializes a new instance of the `ColumnSelector` class.

        Args:
            columns : list of str
                List of column names to select or drop.
            selector_type : str, optional (default='keep')
                Type of selection. Must be either 'keep' or 'drop'.
        """
        self.columns = columns
        assert selector_type in ["keep", "drop"]
        self.selector_type = selector_type

    def fit(self, X: pd.DataFrame, y=None):
        """
        No-op

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame):
        """
        Applies the column selection.

        Args:
            X : pandas.DataFrame - The input data.
        Returns:
            pandas.DataFrame: The transformed data.
        """
        if self.selector_type == "keep":
            retained_cols = [col for col in X.columns if col in self.columns]
            X = X[retained_cols].copy()
        elif self.selector_type == "drop":
            dropped_cols = [col for col in X.columns if col in self.columns]
            X = X.drop(dropped_cols, axis=1)
        return X


class TypeCaster(BaseEstimator, TransformerMixin):
    """
    A custom transformer that casts the specified variables in the input data
    to a specified data type.
    """

    def __init__(self, vars: List[str], cast_type: str):
        """
        Initializes a new instance of the `TypeCaster` class.

        Args:
            vars : list of str
                List of variable names to be transformed.
            cast_type : data type
                Data type to which the specified variables will be cast.
        """
        super().__init__()
        self.vars = vars
        self.cast_type = cast_type

    def fit(self, X: pd.DataFrame, y=None):
        """
        No-op.

        Returns:
            self
        """
        return self

    def transform(self, data: pd.DataFrame):
        """
        Applies the casting to given features in input dataframe.

        Args:
            data : pandas DataFrame
                Input data to be transformed.
        Returns:
            data : pandas DataFrame
                Transformed data.
        """
        data = data.copy()
        applied_cols = [col for col in self.vars if col in data.columns]
        for var in applied_cols:
            if data[var].notnull().any():  # check if the column has any non-null values
                data[var] = data[var].astype(self.cast_type)
            else:
                # all values are null. so no-op
                pass
        return data


class TimeColCaster(BaseEstimator, TransformerMixin):
    """
    A custom transformer that casts the time col in the input data
    to either a datetime type or the integer type, given its type
    in the schema.
    """

    def __init__(self, time_col: str, data_type: str):
        """
        Initializes a new instance of the `TimeColCaster` class.

        Args:
            time_col (str): Name of the time field.
            cast_type (str): Data type to which the specified variables
                             will be cast.
        """
        super().__init__()
        self.time_col = time_col
        self.data_type = data_type

    def fit(self, X: pd.DataFrame, y=None):
        """
        No-op.

        Returns:
            self
        """
        return self

    def transform(self, data: pd.DataFrame):
        """
        Applies the casting to given features in input dataframe.

        Args:
            data : pandas DataFrame
                Input data to be transformed.
        Returns:
            data : pandas DataFrame
                Transformed data.
        """
        data = data.copy()
        if self.data_type == "INT":
            data[self.time_col] = data[self.time_col].astype(int)
        elif self.data_type in ["DATETIME", "DATE"]:
            data[self.time_col] = pd.to_datetime(data[self.time_col])
        else:
            raise ValueError(f"Invalid data type for time column: {self.data_type}")
        return data


class DataFrameSorter(BaseEstimator, TransformerMixin):
    """
    Sorts a pandas DataFrame based on specified columns and their corresponding sort orders.
    """

    def __init__(self, sort_columns: list[str], ascending: list[bool]):
        """
        Initializes a new instance of the `DataFrameSorter` class.

        Args:
            sort_columns : list of str
                List of column names to sort by.
            ascending : list of bool
                List of boolean values corresponding to each column in `sort_columns`.
                Each value indicates whether to sort the corresponding column in ascending order.
        """
        assert len(sort_columns) == len(
            ascending
        ), "sort_columns and ascending must be of the same length"
        self.sort_columns = sort_columns
        self.ascending = ascending

    def fit(self, X: pd.DataFrame, y=None):
        """
        No-op

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame):
        """
        Sorts the DataFrame based on specified columns and order.

        Args:
            X : pandas.DataFrame - The input data.
        Returns:
            pandas.DataFrame: The sorted DataFrame.
        """
        X = X.sort_values(by=self.sort_columns, ascending=self.ascending)
        return X


class ReshaperToThreeD(BaseEstimator, TransformerMixin):
    def __init__(self, id_col, time_col, value_columns) -> None:
        super().__init__()
        self.id_col = id_col
        self.time_col = time_col
        if not isinstance(value_columns, list):
            self.value_columns = [value_columns]
        else:
            self.value_columns = value_columns
        self.id_vals = None
        self.fitted_value_columns = None
        self.time_periods = None

    def fit(self, X, y=None):
        self.id_vals = X[[self.id_col]].drop_duplicates().sort_values(by=self.id_col)
        self.id_vals.reset_index(inplace=True, drop=True)
        self.time_periods = sorted(X[self.time_col].unique())
        self.fitted_value_columns = [c for c in self.value_columns if c in X.columns]
        return self

    def transform(self, X):
        N = self.id_vals.shape[0]
        T = len(self.time_periods)
        D = len(self.value_columns)
        X = X[self.value_columns].values.reshape((N, T, D))
        return X

    def inverse_transform(self, preds_df):

        time_cols = list(preds_df.columns)
        preds_df = pd.concat([self.id_vals, preds_df], axis=1, ignore_index=True)

        cols = self.id_columns + time_cols
        preds_df.columns = cols

        # unpivot given dataframe
        preds_df = pd.melt(
            preds_df,
            id_vars=self.id_columns,
            value_vars=time_cols,
            var_name=self.time_column,
            value_name=self.value_columns[0],
        )

        return preds_df


class TimeSeriesWindowGenerator(BaseEstimator, TransformerMixin):
    """
    A transformer for generating windows from time-series data.
    """

    def __init__(self, window_size: int, stride: int = 1, max_windows: int = 10000):
        """
        Initializes the TimeSeriesWindowGenerator.

        Args:
            window_size (int): The size of each window (W).
            stride (int): The stride between each window.
        """
        self.window_size = window_size
        self.stride = stride
        self.max_windows = max_windows

    def fit(self, X, y=None):
        """
        No-op. This transformer does not require fitting.

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Transforms the input time-series data into windows using vectorized operations.

        Args:
            X (numpy.ndarray): Input time-series data of shape [N, T, D].

        Returns:
            numpy.ndarray: Transformed data of shape [N', W, D] where N' is the number of windows.
        """
        n_series, time_length, n_features = X.shape

        # Validate window size and stride
        if self.window_size > time_length:
            raise ValueError(
                "Window size must be less than or equal to the time dimension length"
            )

        # Calculate the total number of windows per series
        n_windows_per_series = 1 + (time_length - self.window_size) // self.stride

        # Create an array of starting indices for each window
        start_indices = np.arange(0, n_windows_per_series * self.stride, self.stride)

        # Use broadcasting to generate window indices
        window_indices = start_indices[:, None] + np.arange(self.window_size)

        # Generate windows for each series using advanced indexing
        windows = X[:, window_indices, :]

        # Reshape to the desired output format [N' (total windows across all series), W, D]
        windows = windows.transpose(1, 0, 2, 3).reshape(
            -1, self.window_size, n_features
        )

        if windows.shape[0] > self.max_windows:
            indices = np.random.choice(
                windows.shape[0], self.max_windows, replace=False
            )
            windows = windows[indices]
        return windows


class TimeSeriesMixer(BaseEstimator, TransformerMixin):
    """Transformer for applying TSMixing by taking a random weighted average of the input series."""

    def __init__(self, n_series: int, n_generated_windows: int):
        """Initializes the TimeSeriesMixer.
        Args:
            n_series (int): The number of series to mix.
            n_generated_windows (int): The number of mixed windows to generate.
        """
        self.n_series = n_series
        self.n_generated_windows = n_generated_windows

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the input time-series data by applying TSMixing.

        Args:
            X (np.ndarray): Input time-series data of shape [N, T, D].

        Returns:
            np.ndarray: Transformed data of shape [n_generated_windows, T, D].
        """
        n_windows, _, _ = X.shape
        mixed_series_list = []

        for _ in range(self.n_generated_windows):
            # Generate random weights for each series for the current window
            weights = np.random.dirichlet(np.ones(self.n_series), size=1).reshape(
                -1, 1, 1
            )

            # Randomly select series indices to mix
            selected_series_indices = np.random.choice(
                n_windows, self.n_series, replace=True
            )

            selected_series = X[selected_series_indices]

            # Initialize an empty array for the mixed series of the current window
            mixed_series = np.sum(selected_series * weights, axis=0)

            # mixed_series = (selected_series * weights).sum(axis=0)

            # Append the mixed series to the list
            mixed_series_list.append(mixed_series)

        # Stack the mixed series to form the output array
        mixed_series_output = np.array(mixed_series_list)
        transformed_data = np.concatenate([X, mixed_series_output], axis=0)

        return transformed_data


class KernelSynthesizer(BaseEstimator, TransformerMixin):
    """
    Transformer that generates synthetic time series data based on Gaussian processes
    with combined kernels. Each time series in the dataset is augmented by generating
    synthetic series that mimic the statistical properties of the input series.

    The transformer supports multiple covariates and generates synthetic series
    independently for each covariate in each time series. The final kernel for
    synthetic generation is constructed by randomly combining a set of base kernels
    via addition or multiplication, which allows capturing various time series
    behaviors such as trends, local variations, and seasonality.

    Attributes:
        num_samples (int): The number of base kernels to combine for synthetic series generation.
        length_scale (float): The length scale parameter for the RBF and Periodic kernels,
                              controlling the smoothness and periodicity.
        periodicity (float): The periodicity parameter for the Periodic kernel, controlling
                             the frequency of the seasonal pattern.
    """

    def __init__(self, num_kernel_samples=3, length_scale=1.0, periodicity=3.0):
        """
        Initializes the KernelSynthesizer.

        Args:
            num_kernel_samples (int): The number of base kernels to sample and combine for
                               generating synthetic series.
            length_scale (float): The length scale parameter for the RBF and Periodic kernels.
            periodicity (float): The periodicity parameter for the Periodic kernel.
        """
        self.num_kernel_samples = num_kernel_samples
        self.linear_kernel = DotProduct() + WhiteKernel()
        self.rbf_kernel = RBF(length_scale=length_scale)
        self.periodic_kernel = ExpSineSquared(
            length_scale=length_scale, periodicity=periodicity
        )

    def _combined_kernel(self, kernel_list):
        sampled_kernels = np.random.choice(
            kernel_list, size=self.num_kernel_samples, replace=True
        )
        combined = None
        for kernel in sampled_kernels:
            if combined is None:
                combined = kernel
            else:
                # Randomly decide to add or multiply the next kernel
                if np.random.rand() > 0.5:
                    combined += kernel
                else:
                    combined *= kernel
        return combined

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Generates synthetic time series data for each series and covariate in the input data.

        Args:
            X (np.ndarray): The input time-series data of shape [N, T, D], where N is the number
                            of series, T is the time length, and D is the number of features (covariates).

        Returns:
            np.ndarray: The synthetic time-series data with the same shape as the input.
                        Each series is independently augmented with synthetic data based on
                        the combined kernels.
        """

        # X is expected to be a 3D array (N, T, D)
        N, T, D = X.shape
        synthetic_series = np.zeros((N, T, D))

        for i in range(N):
            for d in range(D):
                # Generate synthetic series for each series in the dataset
                # For simplicity, we use the same kernel combination for each covariate
                kernel_list = [
                    self.linear_kernel,
                    self.rbf_kernel,
                    self.periodic_kernel,
                ]
                final_kernel = self._combined_kernel(kernel_list)
                gp = GaussianProcessRegressor(kernel=final_kernel)

                # Assuming evenly spaced time points
                time_points = np.linspace(0, T - 1, T).reshape(-1, 1)
                gp.fit(time_points, X[i, :, d])  # Fit to each series independently
                synthetic_series[i, :, d] = gp.sample_y(time_points, 1).flatten()

        return synthetic_series


class SeriesLengthTrimmer(BaseEstimator, TransformerMixin):
    """Transformer that trims the length of each series in the dataset.

    This transformer retains only the latest data points along the time dimension
    up to the specified length.

    Attributes:
        trimmed_len (int): The length to which each series should be trimmed.
    """

    def __init__(self, trimmed_len: int):
        """
        Initializes the SeriesLengthTrimmer.

        Args:
            trimmed_len (int): The length to which each series should be trimmed.
        """
        self.trimmed_len = trimmed_len

    def fit(self, X: np.ndarray, y: None = None) -> "SeriesLengthTrimmer":
        """Fit method for the transformer.

        This transformer does not learn anything from the data and hence fit is a no-op.

        Args:
            X (np.ndarray): The input data.
            y (None): Ignored. Exists for compatibility with the sklearn transformer interface.

        Returns:
            SeriesLengthTrimmer: The fitted transformer.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Trims each series in the input data to the specified length.

        Args:
            X (np.ndarray): The input time-series data of shape [N, T, D], where N is the number of series,
                            T is the time length, and D is the number of features.

        Returns:
            np.ndarray: The transformed data with each series trimmed to the specified length.
        """
        _, time_length, _ = X.shape
        if time_length > self.trimmed_len:
            X = X[:, -self.trimmed_len :, :]
        return X


class LeftRightFlipper(BaseEstimator, TransformerMixin):
    """
    A transformer that augments a dataset by adding a left-right flipped version of each tensor.

    This transformer flips each tensor in the dataset along a specified axis and then concatenates
    the flipped version with the original dataset, effectively doubling its size.

    Attributes:
        axis_to_flip (int): The axis along which the tensors will be flipped.
    """

    def __init__(self, axis_to_flip: int):
        """
        Initializes the LeftRightFlipper.

        Args:
            axis_to_flip (int): The axis along which the tensors will be flipped.
        """
        self.axis_to_flip = axis_to_flip

    def fit(self, X: np.ndarray, y: None = None) -> "LeftRightFlipper":
        """
        Fit method for the transformer. This transformer does not learn anything from the data
        and hence fit is a no-op.

        Args:
            X (np.ndarray): The input data.
            y (None): Ignored. Exists for compatibility with the sklearn transformer interface.

        Returns:
            LeftRightFlipper: The fitted transformer.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the input data by adding a flipped version of each tensor.

        Args:
            X (np.ndarray): The input data, a collection of tensors.

        Returns:
            np.ndarray: The augmented data, consisting of the original tensors and their
                        flipped versions.
        """
        X_flipped = np.flip(X, axis=self.axis_to_flip)
        return np.concatenate([X_flipped, X], axis=0)


class TimeSeriesMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Scales the history and forecast parts of a time-series based on history data.

    The scaler is fitted using only the history part of the time-series and is
    then used to transform both the history and forecast parts. Values are scaled
    to a range and capped to an upper bound.

    Attributes:
        encode_len (int): The length of the history (encoding) window in the time-series.
        upper_bound (float): The upper bound to which values are capped after scaling.
    """

    def __init__(self, encode_len: int, upper_bound: float = 3.5):
        """
        Initializes the TimeSeriesMinMaxScaler.

        Args:
            encode_len (int): The length of the history window in the time-series.
            upper_bound (float): The upper bound to which values are capped after scaling.
        """
        self.encode_len = encode_len
        self.upper_bound = upper_bound
        self.min_vals_per_d = None
        self.max_vals_per_d = None
        self.range_per_d = None

    def fit(self, X: np.ndarray, y=None) -> "TimeSeriesMinMaxScaler":
        """
        No-op

        Args:
            X (np.ndarray): Input time-series data of shape [N, T, D].
            y: Ignored. Exists for compatibility with the sklearn transformer interface.

        Returns:
            TimeSeriesMinMaxScaler: The fitted scaler.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the MinMax scaling transformation to the input data.

        Args:
            X (np.ndarray): Input time-series data of shape [N, T, D].

        Returns:
            np.ndarray: The transformed data of shape [N, T, D].
        """
        if self.encode_len > X.shape[1]:
            raise ValueError(
                f"Expected sequence length for scaling >= {self.encode_len}."
                f" Found length {X.shape[1]}."
            )
        # Calculate min, max, and range for scaling
        self.min_vals_per_d = np.min(X[:, : self.encode_len, :], axis=1, keepdims=True)
        self.max_vals_per_d = np.max(X[:, : self.encode_len, :], axis=1, keepdims=True)
        self.range_per_d = self.max_vals_per_d - self.min_vals_per_d
        self.range_per_d = np.where(self.range_per_d == 0, -1, self.range_per_d)
        X_scaled = np.where(
            self.range_per_d == -1, 0, (X - self.min_vals_per_d) / self.range_per_d
        )
        X_scaled = np.clip(X_scaled, -self.upper_bound, self.upper_bound)
        return X_scaled

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the inverse of the MinMax scaling transformation.

        Args:
            X (np.ndarray): Scaled time-series data of shape [N, T, D].

        Returns:
            np.ndarray: The inverse transformed data.
        """
        rescaled_X = np.where(
            self.range_per_d[:, :, :1] == -1,
            X[:, :, :1] + self.min_vals_per_d[:, :, :1],
            X[:, :, :1] * self.range_per_d[:, :, :1] + self.min_vals_per_d[:, :, :1],
        )
        return rescaled_X
