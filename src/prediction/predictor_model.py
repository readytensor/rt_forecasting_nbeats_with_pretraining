import os
import sys
import warnings
from typing import List, Union
import math

import joblib
import numpy as np
import pandas as pd
# from prophet import Prophet
from multiprocessing import Pool, cpu_count
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from prediction.nbeats_model import NBeatsNet
from prediction.pretraining_data_gen import get_pretraining_data


# Check for GPU availability
gpu_avai = (
    "GPU available (YES)"
    if tf.config.list_physical_devices("GPU")
    else "GPU not available"
)

print(gpu_avai)

PREDICTOR_FILE_NAME = "predictor.joblib"
MODEL_PARAMS_FNAME = "model_params.save"
MODEL_WTS_FNAME = "model_wts.save"
HISTORY_FNAME = "history.json"
COST_THRESHOLD = float("inf")


def get_patience_factor(N): 
    # magic number - just picked through trial and error
    patience = max(4, int(38 - math.log(N, 1.5)))
    return patience


class Forecaster:
    """A wrapper class for the Nbeats Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """
    GENERIC_BLOCK = 'generic'
    TREND_BLOCK = 'trend'
    SEASONALITY_BLOCK = 'seasonality'

    MIN_VALID_SIZE = 10

    MODEL_NAME = "Nbeats"

    def __init__(
            self,
            backcast_length:int,
            forecast_length:int,
            num_exog:int=0,
            num_generic_stacks:int=2,
            nb_blocks_per_stack:int=2,
            thetas_dim_per_stack:int=16,
            hidden_layer_units:int=32,
            share_weights_in_stack:bool=False,
            **kwargs
        ):
        """Construct a new Nbeats Forecaster.

        Args:
            backcast_length (int): Encoding (history) length.
            forecast_length (int): Decoding (forecast) length.
            num_exog (int, optional): Number of exogenous variables.
                                            Defaults to 0.
            num_generic_stacks (int, optional): Number of generic stacks.
                                            Defaults to 2.
            nb_blocks_per_stack (int, optional): Number of blocks per stack.
                                            Defaults to 2.
            thetas_dim_per_stack (int, optional): Number of expansion coefficients.
                                            Defaults to 16.
            hidden_layer_units (int, optional): Hidden layer units.
                                            Defaults to 32.
            share_weights_in_stack (boolean, optional): Whether to share weights within stacks.
                                            Defaults to False.
        """
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.num_exog = num_exog
        self.num_generic_stacks = num_generic_stacks
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim_per_stack = thetas_dim_per_stack
        self.hidden_layer_units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.model = self.build_model()
        self.loss = 'mse'
        self.learning_rate = 1e-4
        self.model.compile_model(self.loss, self.learning_rate)
        self.batch_size = 64

    def build_model(self): 
        """Build a new forecaster."""
        stack_types = []; thetas_dim = []
        for _ in range(self.num_generic_stacks):
            stack_types.append(self.GENERIC_BLOCK)
            thetas_dim.append(self.thetas_dim_per_stack)
        model = NBeatsNet(
            input_dim=1,
            exo_dim=self.num_exog,
            backcast_length=self.backcast_length,
            forecast_length=self.forecast_length,
            stack_types=[self.GENERIC_BLOCK] * self.num_generic_stacks,
            nb_blocks_per_stack=self.nb_blocks_per_stack,
            thetas_dim=[self.thetas_dim_per_stack] * self.num_generic_stacks,
            share_weights_in_stack=self.share_weights_in_stack,
            hidden_layer_units=self.hidden_layer_units,
            nb_harmonics=None,
        )
        return model

    def _get_X_y_and_E(self, data: np.ndarray, is_train:bool=True) -> np.ndarray:
        """Extract X (historical target series), y (forecast window target) and
            E (exogenous series) from given array of shape [N, T, D]

            When is_train is True, data contains both history and forecast windows.
            When False, only history is contained.
        """
        N, T, D = data.shape
        if D != 1 + self.num_exog:
            raise ValueError(
                f"Training data expected to have {self.num_exog} exogenous variables. "
                f"Found {D-1}"
            )
        if is_train:
            if T != self.backcast_length + self.forecast_length:
                raise ValueError(
                    f"Training data expected to have {self.backcast_length + self.forecast_length}"
                    f" length on axis 1. Found length {T}"
                )
            X = data[:, :self.backcast_length, :1]
            y = data[:, self.backcast_length:, :1]
            if D > 1:
                E = data[:, :self.backcast_length, 1:]
            else:
                E = None
        else:
            # for inference
            if T < self.backcast_length:
                raise ValueError(
                    f"Inference data length expected to be >= {self.backcast_length}"
                    f" on axis 1. Found length {T}"
                )
            X = data[:, -self.backcast_length:, :1]
            y = None
            if D > 1:
                E = data[:, -self.backcast_length:, 1:]
            else:
                E = None
        return X, y, E

    def _train_on_data(self, data, validation_split=0.1, verbose=1, max_epochs=500):
        """Train the model on the given data.

        Args:
            data (pandas.DataFrame): The training data.
        """
        X, y, E = self._get_X_y_and_E(data, is_train=True)
        loss_to_monitor = 'loss' if validation_split is None else 'val_loss'
        patience = get_patience_factor(X.shape[0])
        early_stop_callback = EarlyStopping(
            monitor=loss_to_monitor, min_delta = 1e-4, patience=patience)
        learning_rate_reduction = ReduceLROnPlateau(
            monitor=loss_to_monitor,
            patience=patience//2,
            factor=0.5,
            min_lr=1e-7
        )        
        history = self.model.fit(
            x=[X, E] if E is not None else X,
            y=y,
            validation_split=validation_split,
            verbose=verbose,
            epochs=max_epochs,
            callbacks=[early_stop_callback, learning_rate_reduction],
            batch_size=self.batch_size,
            shuffle=True
        )
        # recompile the model to reset the optimizer; otherwise re-training slows down
        self.model.compile_model(self.loss, self.learning_rate)
        return history

    def fit(self, training_data:np.ndarray, pre_training_data: Union[np.ndarray, None]=None,
            validation_split: Union[float, None]=0.15, verbose:int=0,
            max_epochs:int=2000):

        """Fit the Forecaster to the training data.
        A separate Prophet model is fit to each series that is contained
        in the data.

        Args:
            data (pandas.DataFrame): The features of the training data.
        """
        if pre_training_data is not None:
            print("Conducting pretraining...")
            pretraining_history = self._train_on_data(
                data=pre_training_data,
                validation_split=validation_split,
                verbose=verbose,
                max_epochs=max_epochs
            )
        
        print("Training on main data...")
        history = self._train_on_data(
            data=training_data,
            validation_split=validation_split,
            verbose=verbose,
            max_epochs=max_epochs
        )
        self._is_trained = True
        return history

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """Make the forecast of given length.

        Args:
            test_data (np.ndarray): Given test input for forecasting.
        Returns:
            numpy.ndarray: predictions as numpy array.
        """
        X, y, E = self._get_X_y_and_E(test_data, is_train=False)
        preds = self.model.predict(x=[X, E] if E is not None else X )
        return preds
    
    def evaluate(self, test_data: np.ndarray) -> np.ndarray:
        """Return loss for given evaluation X and y

        Args:
            test_data (np.ndarray): Given test data for evaluation.
        Returns:
            float: loss value (mse).
        """
        X, y, E = self._get_X_y_and_E(test_data, is_train=False)
        score = self.model.evaluate(
            x=[X, E] if E is not None else X,
            y=y
        )
        return score

    def save(self, model_dir_path: str) -> None:
        """Save the forecaster to disk.

        Args:
            model_dir_path (str): The dir path to which to save the model.
        """
        if self.model is None:
            raise NotFittedError("Model is not fitted yet.")
        model_params = {
            "backcast_length": self.backcast_length,
            "forecast_length": self.forecast_length,
            "num_exog": self.num_exog,
            "num_generic_stacks": self.num_generic_stacks,
            "nb_blocks_per_stack": self.nb_blocks_per_stack,
            "thetas_dim_per_stack": self.thetas_dim_per_stack,
            "hidden_layer_units": self.hidden_layer_units,
            "share_weights_in_stack": self.share_weights_in_stack,
        }
        joblib.dump(model_params, os.path.join(model_dir_path, MODEL_PARAMS_FNAME))
        self.model.save_weights(os.path.join(model_dir_path, MODEL_WTS_FNAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded forecaster.
        """
        if not os.path.exists(model_dir_path):
            raise FileNotFoundError(f"Model dir {model_dir_path} does not exist.")
        model_params = joblib.load(os.path.join(model_dir_path, MODEL_PARAMS_FNAME))
        forecaster_model = cls(**model_params)
        forecaster_model.model.load_weights(
            os.path.join(model_dir_path, MODEL_WTS_FNAME)
        ).expect_partial()
        return forecaster_model

    def __str__(self):
        return f"Model name: {self.MODEL_NAME}"


def train_predictor_model(
    history: pd.DataFrame,
    forecast_length: int,
    frequency: str,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the forecaster model.

    Args:
        history (np.ndarray): The training data inputs.
        forecast_length (int): Length of forecast window.
        frequency (str): Frequency of the data such as MONTHLY, DAILY, etc.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """
    pre_training_data = get_pretraining_data(
        series_len=history.shape[1],
        forecast_length=forecast_length,
        frequency=frequency,
        num_exog=history.shape[2]-1
    )
    model = Forecaster(
        backcast_length=history.shape[1] - forecast_length,
        forecast_length=forecast_length,
        num_exog=history.shape[2] - 1,
        **hyperparameters,
    )
    model.fit(
        training_data=history,
        pre_training_data=pre_training_data,
    )
    return model


def predict_with_model(
    model: Forecaster, test_data: np.ndarray
) -> np.ndarray:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (np.ndarray): The test input data for forecasting.

    Returns:
        np.ndarray: The forecast.
    """
    return model.predict(test_data)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
