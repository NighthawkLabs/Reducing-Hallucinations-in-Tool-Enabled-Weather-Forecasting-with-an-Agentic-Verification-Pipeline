import joblib
import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


# Config

DATA_PATHS = [
    
    "../data/clean/noaaDataCleaned.csv",
]
# Possible paths to the trained Random Forest model.
#
# Note:
# The saved baseline_model.pkl file was not uploaded to GitHub because it
# exceeded GitHub's file size limit. To regenerate it, run:
#
#     python src/train_baseline.py
#
# After training, the model should be saved to models/baseline_model.pkl.

MODEL_PATHS = [
    "models/baseline_model.pkl",
]
# Weather variables used as both model inputs and prediction targets.
FEATURE_COLUMNS = ["temperature", "dew_point", "wind_speed"]
WINDOW_SIZE = 72
FORECAST_HORIZON = 24

OUTPUT_PATH = "results/baseline_verification.json"


# Helpers

def find_existing_path(possible_paths):
    """
    Return the first path in possible_paths that exists.

    This makes the script more flexible if files are moved or renamed during
    development. If no path exists, the function returns None.
    """
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def load_dataset():
    """
    Load and validate the cleaned NOAA weather dataset.

    The dataset must contain:
    - a date or DATE column
    - temperature
    - dew point
    - wind speed

    The date column is normalized to lowercase "date" so the rest of the
    script can use one consistent column name.
    """
    data_path = find_existing_path(DATA_PATHS)

    if data_path is None:
        raise FileNotFoundError(
            "Could not find cleaned weather dataset. "
            "Update DATA_PATHS in this script with the correct filename."
        )

    df = pd.read_csv(data_path)

    # Try to normalize the date column
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.rename(columns={"DATE": "date"})
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("Dataset must contain a DATE or date column.")

    df = df.sort_values("date").reset_index(drop=True)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    return df, data_path


def load_model():
    """
    Load the trained forecasting model.

    The model is normally saved with joblib. A pickle fallback is included in
    case the model file was saved using pickle in an earlier version of the
    project.
    """
    model_path = find_existing_path(MODEL_PATHS) 

    if model_path is None:
        raise FileNotFoundError(
            " Could not find trained model." 
            "Update MODEL_PATHS with the correct filename." 
        )

    try:
        model = joblib.load(model_path)
    except Exception as joblib_error:
        try: 
            with open(model_path, "rb") as f: 
                model = pickle.load(f)
        except Exception as pickle_error:
            raise RuntimeError( 
                f"Could not load model with joblib or pickle.\n"
                f"joblib error: {joblib_error}\n"
                f"pickle error: {pickle_error}"
            )
    return model, model_path



def create_windows(df):
    """
    Recreate the model input windows from the cleaned dataset.

    Each example uses:
    - the previous 72 hours of weather observations as input
    - the weather values 24 hours later as the target

    The windows are flattened because the Random Forest model expects 2D
    tabular input, not 3D sequence input.
    """
    values = df[FEATURE_COLUMNS].values

    X = []
    y = []
    target_dates = []

    for i in range(WINDOW_SIZE, len(values) - FORECAST_HORIZON):
        window = values[i - WINDOW_SIZE:i]
        target = values[i + FORECAST_HORIZON]

        X.append(window.flatten())
        y.append(target)
        target_dates.append(df.loc[i + FORECAST_HORIZON, "date"])

    return np.array(X), np.array(y), target_dates


def evaluate_model(model, X, y):
    """
    Evaluate the trained model using the same chronological test split style
    used in the training script.

    The final 20% of examples are treated as the test set. This avoids random
    mixing of past and future data, which is important for a forecasting task.
    """
    # Same style as your previous split: first 80% train, last 20% test
    split_index = int(len(X) * 0.8)

    X_test = X[split_index:]
    y_test = y[split_index:]

    predictions = model.predict(X_test)

    temp_mae = mean_absolute_error(y_test[:, 0], predictions[:, 0])
    dew_mae = mean_absolute_error(y_test[:, 1], predictions[:, 1])
    wind_mae = mean_absolute_error(y_test[:, 2], predictions[:, 2])
    avg_mae = (temp_mae + dew_mae + wind_mae) / 3

    return {
        "temperature_mae": float(temp_mae),
        "dew_point_mae": float(dew_mae),
        "wind_speed_mae": float(wind_mae),
        "average_mae": float(avg_mae),
        "test_examples": int(len(X_test)),
    }


def single_forecast_check(model, df):
    """
    Uses a known-safe request index from the test region.
    The request time is the time at the end of the 72-hour window.
    The target time is 24 hours later.
    """
    values = df[FEATURE_COLUMNS].values

    request_index = int(len(df) * 0.85)

    if request_index < WINDOW_SIZE:
        request_index = WINDOW_SIZE

    target_index = request_index + FORECAST_HORIZON

    if target_index >= len(df):
        raise ValueError("Not enough data after request_index for a 24-hour forecast.")

    window = values[request_index - WINDOW_SIZE:request_index]
    X_single = window.flatten().reshape(1, -1)

    prediction = model.predict(X_single)[0]
    actual = values[target_index]

    request_time = df.loc[request_index, "date"]
    target_time = df.loc[target_index, "date"]

    return {
        "request_time": str(request_time),
        "forecast_target_time": str(target_time),
        "prediction": {
            "temperature": float(prediction[0]),
            "dew_point": float(prediction[1]),
            "wind_speed": float(prediction[2]),
        },
        "actual": {
            "temperature": float(actual[0]),
            "dew_point": float(actual[1]),
            "wind_speed": float(actual[2]),
        },
        "absolute_error": {
            "temperature": float(abs(prediction[0] - actual[0])),
            "dew_point": float(abs(prediction[1] - actual[1])),
            "wind_speed": float(abs(prediction[2] - actual[2])),
        },
    }


# Main

def main():
    os.makedirs("results", exist_ok=True)

    df, data_path = load_dataset()
    model, model_path = load_model()

    X, y, target_dates = create_windows(df)

    metrics = evaluate_model(model, X, y)
    example_forecast = single_forecast_check(model, df)

    verification = {
        "status": "success",
        "data_path": data_path,
        "model_path": model_path,
        "dataset_rows": int(len(df)),
        "date_range": {
            "start": str(df["date"].min()),
            "end": str(df["date"].max()),
        },
        "feature_columns": FEATURE_COLUMNS,
        "window_size_hours": WINDOW_SIZE,
        "forecast_horizon_hours": FORECAST_HORIZON,
        "windowed_X_shape": list(X.shape),
        "windowed_y_shape": list(y.shape),
        "metrics": metrics,
        "example_forecast": example_forecast,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(verification, f, indent=2)

    print("\nBaseline verification complete.")
    print("--------------------------------")
    print(f"Dataset path: {data_path}")
    print(f"Model path:   {model_path}")
    print(f"Rows:         {len(df)}")
    print(f"Date range:   {df['date'].min()} to {df['date'].max()}")
    print()
    print("Windowed data:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print()
    print("MAE results:")
    print(f"Temperature MAE: {metrics['temperature_mae']:.4f}")
    print(f"Dew Point MAE:   {metrics['dew_point_mae']:.4f}")
    print(f"Wind Speed MAE:  {metrics['wind_speed_mae']:.4f}")
    print(f"Average MAE:     {metrics['average_mae']:.4f}")
    print()
    print("Example forecast:")
    print(json.dumps(example_forecast, indent=2))
    print()
    print(f"Saved verification file to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
