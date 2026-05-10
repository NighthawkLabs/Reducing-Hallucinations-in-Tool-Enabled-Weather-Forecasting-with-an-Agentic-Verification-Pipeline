import argparse
import json
import os
import csv

# Config
# These paths are relative to this script's location. 
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Default in/output files used by the forecasting accuracy evaluation 
# Input JSONL file contains the generated evaluation cases, the model predictions, and the actual target values 

DEFAULT_INPUT_JSONL = os.path.join(RESULTS_DIR, "eval_forecast_cases.jsonl")
DEFAULT_DETAILED_CSV = os.path.join(RESULTS_DIR, "forecast_accuracy_detailed_scores.csv")
DEFAULT_SUMMARY_CSV = os.path.join(RESULTS_DIR, "forecast_accuracy_summary.csv")
DEFAULT_SUMMARY_JSON = os.path.join(RESULTS_DIR, "forecast_accuracy_summary.json")


# File helpers

def read_jsonl(path):
    """
    Read a JSON file where each non-empty line is one JSON object. 
    This project stores evaluation cases as JSONL files so each forecast case can be processed independenty while keeping the file easy to inspect
    """
    records = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find input file: {path}")

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def write_csv(rows, path):
    """
    Writes a list of dictionaries to a CSV file. The column names are taken from the keys of the first row
    """
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data, path):
    """
    Writes summary results to a JSONL 
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# Metric helpers


def mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def get_prediction(record):
    """
    Supports eval case format:

    "prediction": {
        "temperature": ...,
        "dew_point": ...,
        "wind_speed": ...
    }

    Also supports flat predicted_* fields if needed.
    """

    if "prediction" in record:
        pred = record["prediction"]

        return {
            "temperature": float(pred["temperature"]),
            "dew_point": float(pred["dew_point"]),
            "wind_speed": float(pred["wind_speed"]),
        }

    return {
        "temperature": float(record["predicted_temperature"]),
        "dew_point": float(record["predicted_dew_point"]),
        "wind_speed": float(record["predicted_wind_speed"]),
    }


def get_actual(record):
    """
    Supports eval case format:

    "actual": {
        "temperature": ...,
        "dew_point": ...,
        "wind_speed": ...
    }

    Also supports flat actual_* fields if needed.
    """

    if "actual" in record:
        actual = record["actual"]

        return {
            "temperature": float(actual["temperature"]),
            "dew_point": float(actual["dew_point"]),
            "wind_speed": float(actual["wind_speed"]),
        }

    return {
        "temperature": float(record["actual_temperature"]),
        "dew_point": float(record["actual_dew_point"]),
        "wind_speed": float(record["actual_wind_speed"]),
    }


def score_record(record):
    """
    Score one forecast case by comparing the ML model prediction against the actual observed values 

    This evaluation is separate from the LLM hallucination evaluation. 
    """
    pred = get_prediction(record)
    actual = get_actual(record)

    temperature_abs_error = abs(pred["temperature"] - actual["temperature"])
    dew_point_abs_error = abs(pred["dew_point"] - actual["dew_point"])
    wind_speed_abs_error = abs(pred["wind_speed"] - actual["wind_speed"])

    average_abs_error = mean([
        temperature_abs_error,
        dew_point_abs_error,
        wind_speed_abs_error,
    ])

    return {
        "case_id": record.get("case_id"),
        "request_time": record.get("request_time"),
        "forecast_target_time": record.get("forecast_target_time"),

        "predicted_temperature": round(pred["temperature"], 4),
        "actual_temperature": round(actual["temperature"], 4),
        "temperature_abs_error": round(temperature_abs_error, 4),

        "predicted_dew_point": round(pred["dew_point"], 4),
        "actual_dew_point": round(actual["dew_point"], 4),
        "dew_point_abs_error": round(dew_point_abs_error, 4),

        "predicted_wind_speed": round(pred["wind_speed"], 4),
        "actual_wind_speed": round(actual["wind_speed"], 4),
        "wind_speed_abs_error": round(wind_speed_abs_error, 4),

        "average_abs_error": round(average_abs_error, 4),
    }


def summarize(scores):
    """
    Aggregates all the pre-case scores into overall forecast accuracy metrics like MAE 
    """
    
    temperature_errors = [row["temperature_abs_error"] for row in scores]
    dew_point_errors = [row["dew_point_abs_error"] for row in scores]
    wind_speed_errors = [row["wind_speed_abs_error"] for row in scores]
    average_errors = [row["average_abs_error"] for row in scores]

    summary = {
        "system": "random_forest_ml_baseline",
        "total_cases": len(scores),

        "temperature_mae": round(mean(temperature_errors), 4),
        "dew_point_mae": round(mean(dew_point_errors), 4),
        "wind_speed_mae": round(mean(wind_speed_errors), 4),
        "average_mae": round(mean(average_errors), 4),

        "max_temperature_abs_error": round(max(temperature_errors), 4),
        "max_dew_point_abs_error": round(max(dew_point_errors), 4),
        "max_wind_speed_abs_error": round(max(wind_speed_errors), 4),
        "max_average_abs_error": round(max(average_errors), 4),
    }

    return summary


def print_summary(summary):
    print()
    print("Forecast Accuracy Summary")
    print("-------------------------")
    print(f"System:                  {summary['system']}")
    print(f"Total cases:             {summary['total_cases']}")
    print()
    print(f"Temperature MAE:         {summary['temperature_mae']}")
    print(f"Dew point MAE:           {summary['dew_point_mae']}")
    print(f"Wind speed MAE:          {summary['wind_speed_mae']}")
    print(f"Average MAE:             {summary['average_mae']}")
    print()
    print(f"Max temperature error:   {summary['max_temperature_abs_error']}")
    print(f"Max dew point error:     {summary['max_dew_point_abs_error']}")
    print(f"Max wind speed error:    {summary['max_wind_speed_abs_error']}")
    print(f"Max average error:       {summary['max_average_abs_error']}")



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ML forecast accuracy against actual NOAA target values."
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_JSONL,
        help="Input eval cases JSONL file.",
    )

    parser.add_argument(
        "--detailed-output",
        default=DEFAULT_DETAILED_CSV,
        help="Detailed per-case CSV output path.",
    )

    parser.add_argument(
        "--summary-output",
        default=DEFAULT_SUMMARY_CSV,
        help="Summary CSV output path.",
    )

    parser.add_argument(
        "--summary-json",
        default=DEFAULT_SUMMARY_JSON,
        help="Summary JSON output path.",
    )

    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    records = read_jsonl(args.input)

    if not records:
        print("No records found.")
        return

    scores = [score_record(record) for record in records]
    summary = summarize(scores)

    write_csv(scores, args.detailed_output)
    write_csv([summary], args.summary_output)
    write_json(summary, args.summary_json)

    print_summary(summary)

    print()
    print("Saved files:")
    print(f"- {args.detailed_output}")
    print(f"- {args.summary_output}")
    print(f"- {args.summary_json}")


if __name__ == "__main__":
    main()
