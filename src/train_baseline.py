import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuration

# path to the cleaned dataset 
DATA_PATH = "../data/clean/noaaDataCleaned.csv"

MODEL_DIR = "models"
RESULTS_DIR = "results"

# Forecasting setup:
# - use the previous 72 hours of observations as input
# - predict weather conditions 24 hours into the future
INPUT_WINDOW = 72
FORECAST_HORIZON = 24

DATE_COLUMN = "date"
TARGET_COLUMNS = ["temperature", "dew_point", "wind_speed"]

# Chronological train/test split. The first 80% of windows are used for
# training, and the final 20% are used for testing.
TRAIN_RATIO = 0.8
RANDOM_STATE = 42


# Helper functions

def load_data(path):
    """
    Load the cleaned NOAA dataset and sort it chronologically.

    Sorting by date is important because this is a time-series forecasting
    setup. The model should only train and predict using correctly ordered
    historical observations.
    """
    df = pd.read_csv(path)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    return df


def create_windows(df, input_window, forecast_horizon, target_columns):
    """
    Convert the hourly weather dataset into supervised learning examples.

    Each example consists of:

    X:
        A window of the previous 72 hourly observations.

    y:
        The weather values 24 hours after the end of that input window.

    For this project, each target contains:
    - temperature
    - dew point
    - wind speed
    """
    values = df[target_columns].values

    X = []
    y = []
    target_dates = []

    max_start = len(df) - input_window - forecast_horizon + 1

    for i in range(max_start):
        input_start = i
        input_end = i + input_window

        target_index = input_end + forecast_horizon - 1

        X.append(values[input_start:input_end])
        y.append(values[target_index])
        target_dates.append(df.loc[target_index, DATE_COLUMN])

    X = np.array(X)
    y = np.array(y)
    target_dates = np.array(target_dates)

    return X, y, target_dates


def flatten_windows(X):
    """
    Converts X from:
        (samples, 72, 3)
    into:
        (samples, 216)

    This lets a traditional ML model use the time-windowed data.
    """
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])


def train_time_split(X, y, dates, train_ratio):
    """
    Split the dataset chronologically into training and testing sets.

    This avoids data leakage. A random split would mix future observations into
    the training set, which would make the forecasting task less realistic.
    """
    split_index = int(len(X) * train_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    dates_train = dates[:split_index]
    dates_test = dates[split_index:]

    return X_train, X_test, y_train, y_test, dates_train, dates_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained forecasting model on the held-out test set.

    The model predicts three outputs:
    - temperature
    - dew point
    - wind speed

    For each output, this function calculates:
    - MAE: average absolute prediction error
    - RMSE: square-rooted average squared prediction error

    MAE is the main metric used in the project because it is easy to interpret
    in the original weather units.
    """
    predictions = model.predict(X_test)

    results = {}

    for i, col in enumerate(TARGET_COLUMNS):
        mae = mean_absolute_error(y_test[:, i], predictions[:, i])
        mse = mean_squared_error(y_test[:, i], predictions[:, i])
        rmse = np.sqrt(mse)

        results[col] = {
            "MAE": mae,
            "RMSE": rmse
        }

    average_mae = np.mean([results[col]["MAE"] for col in TARGET_COLUMNS])
    average_rmse = np.mean([results[col]["RMSE"] for col in TARGET_COLUMNS])

    results["average"] = {
        "MAE": average_mae,
        "RMSE": average_rmse
    }

    return results, predictions


# Main training pipeline

def main():
    """
    Train and evaluate the Random Forest forecasting baseline.

    This script:
    1. Loads the cleaned NOAA weather dataset
    2. Creates 72-hour input windows and 24-hour-ahead targets
    3. Splits the data chronologically into training and testing sets
    4. Trains a multi-output Random Forest regressor
    5. Evaluates forecast accuracy
    6. Saves the trained model and result files

    Note:
    The saved model file, baseline_model.pkl, may be too large for GitHub.
    If it is not included in the repository, regenerate it by running this
    training script.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_data(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("Date range:", df[DATE_COLUMN].min(), "to", df[DATE_COLUMN].max())

    print("Creating 72-hour input windows and 24-hour-ahead targets...")
    X, y, target_dates = create_windows(
        df=df,
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
        target_columns=TARGET_COLUMNS
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_flat = flatten_windows(X)

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_time_split(
        X_flat,
        y,
        target_dates,
        TRAIN_RATIO
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    print("Training baseline Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")
    results, predictions = evaluate_model(model, X_test, y_test)

    print("\nFinal Baseline Results:")
    for target, metrics in results.items():
        print(f"{target}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

    print("Saving model...")
    joblib.dump(model, os.path.join(MODEL_DIR, "baseline_model.pkl"))

    print("Saving baseline results...")

    result_rows = []
    for target, metrics in results.items():
        result_rows.append({
            "target": target,
            "mae": metrics["MAE"],
            "rmse": metrics["RMSE"]
        })

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"), index=False)

    predictions_df = pd.DataFrame({
        "target_date": dates_test,
        "actual_temperature": y_test[:, 0],
        "actual_dew_point": y_test[:, 1],
        "actual_wind_speed": y_test[:, 2],
        "predicted_temperature": predictions[:, 0],
        "predicted_dew_point": predictions[:, 1],
        "predicted_wind_speed": predictions[:, 2],
    })

    predictions_df.to_csv(
        os.path.join(RESULTS_DIR, "baseline_predictions.csv"),
        index=False
    )

    print("Done.")


if __name__ == "__main__":
    main()
