import os
import json
import joblib
import pandas as pd
import numpy as np

# Config

# paths to the dataset, trained model, and output files 
DATA_PATH = "../data/clean/noaaDataCleaned.csv"
MODEL_PATH = "models/baseline_model.pkl"

OUTPUT_JSONL = "../results/eval_forecast_cases.jsonl"
OUTPUT_CSV = "../results/eval_forecast_cases.csv"

# weather variables being used by the model 
FEATURE_COLUMNS = ["temperature", "dew_point", "wind_speed"]

# Prepare evaluation with the previous 72 hours of weather data to predict conditions 24-hours ahead. Generate 1000 cases 
WINDOW_SIZE = 72
FORECAST_HORIZON = 24
NUM_CASES = 1000


# Helper functions

def load_dataset():
    """
    Load and prep the dataset 
    """
    df = pd.read_csv(DATA_PATH)

    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.rename(columns={"DATE": "date"})
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("Dataset must contain a DATE or date column.")

    df = df.sort_values("date").reset_index(drop=True)

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    return df


def select_request_indices(df):
    """
    Select evenly spaced request indicies from the test portion of the dataset.

    Avoid the first 72 rows because it needs history.
    Avoid the final 24 rows because it needs a future target.

    using np.linspace gives coverage over the entire test period instead of taking the first NUM_CASES rows or randomly clustering cases
    """
    min_index = WINDOW_SIZE
    max_index = len(df) - FORECAST_HORIZON - 1

    # Use later data so these are aligned with the test region.
    test_start = int(len(df) * 0.8)
    start_index = max(test_start, min_index)

    indices = np.linspace(start_index, max_index, NUM_CASES, dtype=int)

    # Remove accidental duplicates caused by integer rounding.
    indices = sorted(set(indices.tolist()))

    return indices


def make_case(df, model, request_index, case_id):
    """
    Creates one forecast evaluation case 

    This function builds the 72-hour model input window, gets the Random Forest prediction, retrieves the actual NOAA observations, and stores both as a structured record 
    """
    values = df[FEATURE_COLUMNS].values

    target_index = request_index + FORECAST_HORIZON

    window = values[request_index - WINDOW_SIZE:request_index]
    X_single = window.flatten().reshape(1, -1)

    prediction = model.predict(X_single)[0]
    actual = values[target_index]

    request_time = df.loc[request_index, "date"]
    target_time = df.loc[target_index, "date"]

    case = {
        "case_id": case_id,
        "request_time": str(request_time),
        "forecast_target_time": str(target_time),
        "prediction": {
            "temperature": round(float(prediction[0]), 2),
            "dew_point": round(float(prediction[1]), 2),
            "wind_speed": round(float(prediction[2]), 2),
        },
        "actual": {
            "temperature": round(float(actual[0]), 2),
            "dew_point": round(float(actual[1]), 2),
            "wind_speed": round(float(actual[2]), 2),
        },
        "absolute_error": {
            "temperature": round(float(abs(prediction[0] - actual[0])), 2),
            "dew_point": round(float(abs(prediction[1] - actual[1])), 2),
            "wind_speed": round(float(abs(prediction[2] - actual[2])), 2),
        },
    }

    return case


def flatten_case(case):
    """
    Convert a nested evaluation case into a flat dictionary for CSV output 
    """
    return {
        "case_id": case["case_id"],
        "request_time": case["request_time"],
        "forecast_target_time": case["forecast_target_time"],
        "predicted_temperature": case["prediction"]["temperature"],
        "predicted_dew_point": case["prediction"]["dew_point"],
        "predicted_wind_speed": case["prediction"]["wind_speed"],
        "actual_temperature": case["actual"]["temperature"],
        "actual_dew_point": case["actual"]["dew_point"],
        "actual_wind_speed": case["actual"]["wind_speed"],
        "temperature_abs_error": case["absolute_error"]["temperature"],
        "dew_point_abs_error": case["absolute_error"]["dew_point"],
        "wind_speed_abs_error": case["absolute_error"]["wind_speed"],
    }



def main():
    """
    Generate forecast evlaution cases for the LLM systems 
    """
    os.makedirs("results", exist_ok=True)

    print("Loading dataset...")
    df = load_dataset()

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Selecting evaluation cases...")
    request_indices = select_request_indices(df)

    cases = []
    for i, request_index in enumerate(request_indices, start=1):
        case = make_case(df, model, request_index, case_id=i)
        cases.append(case)

    with open(OUTPUT_JSONL, "w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")

    flat_cases = [flatten_case(case) for case in cases]
    pd.DataFrame(flat_cases).to_csv(OUTPUT_CSV, index=False)

    print()
    print("Evaluation cases generated.")
    print("----------------------------")
    print(f"Dataset rows: {len(df)}")
    print(f"Number of cases: {len(cases)}")
    print(f"JSONL output: {OUTPUT_JSONL}")
    print(f"CSV output:   {OUTPUT_CSV}")
    print()
    print("First case:")
    print(json.dumps(cases[0], indent=2))
    print()
    print("Last case:")
    print(json.dumps(cases[-1], indent=2))


if __name__ == "__main__":
    main()
