import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = "../data/clean/noaaDataCleaned.csv"

MODEL_DIR = "models"
RESULTS_DIR = "results"

INPUT_WINDOW = 72
FORECAST_HORIZON = 24

DATE_COLUMN = "date"
TARGET_COLUMNS = ["temperature", "dew_point", "wind_speed"]

TRAIN_RATIO = 0.8
RANDOM_STATE = 42


# -----------------------------
# Helper functions
# -----------------------------

def load_data(path):
    df = pd.read_csv(path)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    return df


def create_windows(df, input_window, forecast_horizon, target_columns):
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
    split_index = int(len(X) * train_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    dates_train = dates[:split_index]
    dates_test = dates[split_index:]

    return X_train, X_test, y_train, y_test, dates_train, dates_test


def evaluate_model(model, X_test, y_test):
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


# -----------------------------
# Main training pipeline
# -----------------------------

def main():
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
