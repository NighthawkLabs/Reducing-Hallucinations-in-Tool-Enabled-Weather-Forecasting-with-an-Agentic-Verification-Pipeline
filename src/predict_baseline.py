import os
import joblib
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SRC_DIR, "models", "baseline_model.pkl")


DATA_PATH = "../data/clean/noaaDataCleaned.csv"

DATE_COLUMN = "date"
TARGET_COLUMNS = ["temperature", "dew_point", "wind_speed"]

INPUT_WINDOW = 72
FORECAST_HORIZON = 24


# -----------------------------
# Loading helpers
# -----------------------------

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Could not find model at: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Missing date column: {DATE_COLUMN}")

    missing_targets = [col for col in TARGET_COLUMNS if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    return df


# -----------------------------
# Prediction tool
# -----------------------------

def predict_weather_from_request_time(request_time):
    """
    Uses the previous 72 hours of weather data to predict temperature,
    dew point, and wind speed 24 hours after request_time.

    Example:
        request_time = "2005-04-12 08:00:00"

    Returns a dictionary with:
        - request time
        - input window range
        - forecast target time
        - predicted values
        - actual values, if available in the dataset
        - absolute errors
    """

    model = load_model()
    df = load_data()

    request_time = pd.to_datetime(request_time)

    matches = df.index[df[DATE_COLUMN] == request_time].tolist()

    if not matches:
        raise ValueError(f"Request time {request_time} not found in dataset.")

    request_index = matches[0]

    input_start = request_index - INPUT_WINDOW + 1
    input_end = request_index + 1
    target_index = request_index + FORECAST_HORIZON

    if input_start < 0:
        raise ValueError(
            f"Not enough historical data before {request_time}. "
            f"Need {INPUT_WINDOW} hours of prior observations."
        )

    if target_index >= len(df):
        raise ValueError(
            f"Not enough future data after {request_time} to evaluate the 24-hour forecast."
        )

    input_window = df.loc[input_start:input_end - 1, TARGET_COLUMNS].values

    X = input_window.reshape(1, INPUT_WINDOW * len(TARGET_COLUMNS))

    prediction = model.predict(X)[0]

    actual = df.loc[target_index, TARGET_COLUMNS].values
    target_time = df.loc[target_index, DATE_COLUMN]

    abs_errors = abs(prediction - actual)

    return {
        "request_time": str(request_time),
        "input_window_start": str(df.loc[input_start, DATE_COLUMN]),
        "input_window_end": str(df.loc[input_end - 1, DATE_COLUMN]),
        "forecast_horizon_hours": FORECAST_HORIZON,
        "forecast_target_time": str(target_time),

        "predicted_temperature": round(float(prediction[0]), 2), 
        "predicted_dew_point": round(float(prediction[1]), 2), 
        "predicted_wind_speed": round(float(prediction[2]), 2), 

        "actual_temperature": round(float(actual[0]), 2),
        "actual_dew_point": round(float(actual[1]), 2),
        "actual_wind_speed": round(float(actual[2]), 2),

        "temperature_abs_error": round(float(abs_errors[0]), 2),
        "dew_point_abs_error": round(float(abs_errors[1]), 2),
        "wind_speed_abs_error": round(float(abs_errors[2]), 2), 
        "average_abs_error": round(float(abs_errors.mean()), 2),
    }



if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1:
        test_time = sys.argv[1]
    else:
        test_time = "2005-04-12 08:00:00"

    result = predict_weather_from_request_time(test_time)

    print(json.dumps(result, indent=2))
