import json
import os
import re
import csv



# Config

# Build paths relative to this script 
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Input files for the two systems being compared 
# Each file contains one JSON record per line with the model/tool output 
INPUT_FILES = {
    "single_tool": os.path.join(RESULTS_DIR, "single_tool_eval_outputs.jsonl"),
    "agentic": os.path.join(RESULTS_DIR, "agentic_eval_outputs.jsonl"),
}

# Output files for the combined hallucination evaluation 
OUTPUT_DETAILED_CSV = os.path.join(
    RESULTS_DIR,
    "system_hallucination_detailed_scores.csv"
)

OUTPUT_SUMMARY_CSV = os.path.join(
    RESULTS_DIR,
    "system_hallucination_summary.csv"
)

OUTPUT_SUMMARY_JSON = os.path.join(
    RESULTS_DIR,
    "system_hallucination_summary.json"
)

# small tolerance for numeric comparisons 
NUMERIC_TOLERANCE = 0.05

# Helpers

def normalize_text(text):
    return text.lower().replace("_", " ")


def normalize_number(value):
    return round(float(value), 2)


def extract_datetime(text):
    pattern = r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"
    match = re.search(pattern, text)
    return match.group(0) if match else None


def extract_number_near_label(text, labels):
    """
    Extracts a number appearing near one of the provided labels. This is used to recover structured forecast values from the LLMs response
    """
    normalized = normalize_text(text)

    for label in labels:
        label = label.lower()

        patterns = [
            rf"{label}\s*(?:is|of|:|=)?\s*(-?\d+(?:\.\d+)?)",
            rf"{label}.*?(-?\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, normalized)

            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

    return None


def extract_all_numbers(text):
    """
    Extract all integer or decimal numbers. used as a backup when label-based extraction fails 
    """
    pattern = r"-?\d+(?:\.\d+)?"
    return [float(x) for x in re.findall(pattern, text)]


def extract_llm_forecast_values(llm_response):
    """
    Extract the forecast target time and predicted weather values from the final LLM response 
    """
    target_time = extract_datetime(llm_response)

    temperature = extract_number_near_label(
        llm_response,
        ["predicted temperature", "temperature", "temp"]
    )

    dew_point = extract_number_near_label(
        llm_response,
        ["predicted dew point", "dew point", "dewpoint"]
    )

    wind_speed = extract_number_near_label(
        llm_response,
        ["predicted wind speed", "wind speed", "wind"]
    )

    # Fallback extraction based on number order after the timestamp 
    
    if temperature is None or dew_point is None or wind_speed is None:
        text_after_time = llm_response

        if target_time and target_time in llm_response:
            text_after_time = llm_response.split(target_time, 1)[1]

        numbers = extract_all_numbers(text_after_time)

        if len(numbers) >= 3:
            if temperature is None:
                temperature = numbers[0]
            if dew_point is None:
                dew_point = numbers[1]
            if wind_speed is None:
                wind_speed = numbers[2]

    return {
        "forecast_target_time": target_time,
        "predicted_temperature": temperature,
        "predicted_dew_point": dew_point,
        "predicted_wind_speed": wind_speed,
    }


def numeric_match(expected, observed, tolerance=NUMERIC_TOLERANCE):
    if observed is None:
        return False

    return abs(float(expected) - float(observed)) <= tolerance


def get_response_text(record):
    """
    Gets the final text to score.

    Supports:
    - single-tool records with llm_response
    - agentic records with final_response
    - agentic records that still use llm_response
    """

    if "final_response" in record:
        return record.get("final_response", "")

    return record.get("llm_response", "")


def get_system_name(record, fallback_system):
    """
    Determines if the record belongs to single-tool LLM system or the agentic system 
    """
    if "system" in record:
        return record["system"]

    if "system_type" in record:
        system_type = record["system_type"]

        if "agentic" in system_type:
            return "agentic"
        if "single" in system_type:
            return "single_tool"

    return fallback_system


def get_expected_prediction(record):
    """
    Gets expected forecast values from the saved record.

    Supports:
    - tool_output from single-tool or agentic runs
    - prediction from generated eval cases
    - expected_prediction from older runs
    - flattened prediction fields
    """

    if "tool_output" in record:
        tool_output = record["tool_output"]

        return {
            "forecast_target_time": tool_output["forecast_target_time"],
            "predicted_temperature": tool_output["predicted_temperature"],
            "predicted_dew_point": tool_output["predicted_dew_point"],
            "predicted_wind_speed": tool_output["predicted_wind_speed"],
        }

    if "prediction" in record:
        prediction = record["prediction"]

        return {
            "forecast_target_time": record["forecast_target_time"],
            "predicted_temperature": prediction["temperature"],
            "predicted_dew_point": prediction["dew_point"],
            "predicted_wind_speed": prediction["wind_speed"],
        }

    if "expected_prediction" in record:
        prediction = record["expected_prediction"]

        return {
            "forecast_target_time": record["forecast_target_time"],
            "predicted_temperature": prediction["temperature"],
            "predicted_dew_point": prediction["dew_point"],
            "predicted_wind_speed": prediction["wind_speed"],
        }

    if all(key in record for key in [
        "forecast_target_time",
        "predicted_temperature",
        "predicted_dew_point",
        "predicted_wind_speed",
    ]):
        return {
            "forecast_target_time": record["forecast_target_time"],
            "predicted_temperature": record["predicted_temperature"],
            "predicted_dew_point": record["predicted_dew_point"],
            "predicted_wind_speed": record["predicted_wind_speed"],
        }

    raise KeyError(
        "Could not find expected forecast fields in record. "
        f"Available top-level keys: {list(record.keys())}"
    )


def detect_unsupported_terms(response):
    """
    Terms the model should not introduce unless explicitly included in the tool output.
    """

    unsupported_keywords = [
        "actual",
        "observed",
        "absolute error",
        "abs error",
        "average error",
        "error",
        "humidity",
        "precipitation",
        "rain",
        "snow",
        "storm",
        "cloudy",
        "cloud",
        "sunny",
        "fog",
        "visibility",
        "pressure",
        "confidence",
        "likely",
        "probably",
        "celsius",
        "°c",
        "meters per second",
        "m/s",
        "km/h",
        "kilometers per hour",
        "conversion",
        "converted",
        "additional details",
    ]

    lower_response = response.lower()

    found = []
    for term in unsupported_keywords:
        if term in lower_response:
            found.append(term)

    return found


# Scoring

def score_record(record, fallback_system):
    """
    Score one LLM response against its expected structured forecast output.

    The score separates two kinds of problems:

    1. Value hallucinations:
       Missing or mismatched forecast values.

    2. Policy violations:
       Extra unsupported terms or claims outside the tool output.

    A response issue is counted if either type of problem occurs.
    """
    response = get_response_text(record)
    expected = get_expected_prediction(record)
    extracted = extract_llm_forecast_values(response)

    expected_time = expected["forecast_target_time"]
    expected_temp = normalize_number(expected["predicted_temperature"])
    expected_dew = normalize_number(expected["predicted_dew_point"])
    expected_wind = normalize_number(expected["predicted_wind_speed"])

    observed_time = extracted["forecast_target_time"]
    observed_temp = extracted["predicted_temperature"]
    observed_dew = extracted["predicted_dew_point"]
    observed_wind = extracted["predicted_wind_speed"]

    target_time_match = observed_time == expected_time
    temperature_match = numeric_match(expected_temp, observed_temp)
    dew_point_match = numeric_match(expected_dew, observed_dew)
    wind_speed_match = numeric_match(expected_wind, observed_wind)

    unsupported_terms = detect_unsupported_terms(response)

    missing_fields = []
    mismatch_fields = []

    if observed_time is None:
        missing_fields.append("forecast_target_time")
    elif not target_time_match:
        mismatch_fields.append("forecast_target_time")

    if observed_temp is None:
        missing_fields.append("predicted_temperature")
    elif not temperature_match:
        mismatch_fields.append("predicted_temperature")

    if observed_dew is None:
        missing_fields.append("predicted_dew_point")
    elif not dew_point_match:
        mismatch_fields.append("predicted_dew_point")

    if observed_wind is None:
        missing_fields.append("predicted_wind_speed")
    elif not wind_speed_match:
        mismatch_fields.append("predicted_wind_speed")

    value_hallucinated = len(missing_fields) > 0 or len(mismatch_fields) > 0
    policy_violation = len(unsupported_terms) > 0
    response_issue = value_hallucinated or policy_violation

    return {
        "case_id": record.get("case_id"),
        "system": get_system_name(record, fallback_system),
        "request_time": record.get("request_time"),
        "forecast_target_time_expected": expected_time,
        "forecast_target_time_observed": observed_time,
        "target_time_match": target_time_match,

        "expected_temperature": expected_temp,
        "observed_temperature": observed_temp,
        "temperature_match": temperature_match,

        "expected_dew_point": expected_dew,
        "observed_dew_point": observed_dew,
        "dew_point_match": dew_point_match,

        "expected_wind_speed": expected_wind,
        "observed_wind_speed": observed_wind,
        "wind_speed_match": wind_speed_match,

        "missing_fields": ";".join(missing_fields),
        "mismatch_fields": ";".join(mismatch_fields),
        "unsupported_terms": ";".join(unsupported_terms),

        "value_hallucinated": value_hallucinated,
        "policy_violation": policy_violation,
        "response_issue": response_issue,

        "llm_response": response,
    }


def load_records():
    all_records = []

    for system_name, path in INPUT_FILES.items():
        if not os.path.exists(path):
            print(f"Warning: missing file for {system_name}: {path}")
            continue

        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    all_records.append((system_name, record))

    return all_records


def summarize(scores):
    summary = {}

    systems = sorted(set(score["system"] for score in scores))

    for system in systems:
        system_scores = [score for score in scores if score["system"] == system]
        total = len(system_scores)

        value_hallucinated_count = sum(
            1 for score in system_scores if score["value_hallucinated"]
        )
        policy_violation_count = sum(
            1 for score in system_scores if score["policy_violation"]
        )
        response_issue_count = sum(
            1 for score in system_scores if score["response_issue"]
        )

        target_time_matches = sum(
            1 for score in system_scores if score["target_time_match"]
        )
        temp_matches = sum(
            1 for score in system_scores if score["temperature_match"]
        )
        dew_matches = sum(
            1 for score in system_scores if score["dew_point_match"]
        )
        wind_matches = sum(
            1 for score in system_scores if score["wind_speed_match"]
        )

        summary[system] = {
            "total_cases": total,

            "value_hallucinated_cases": value_hallucinated_count,
            "value_hallucination_rate": (
                value_hallucinated_count / total if total else 0
            ),

            "policy_violation_cases": policy_violation_count,
            "policy_violation_rate": (
                policy_violation_count / total if total else 0
            ),

            "response_issue_cases": response_issue_count,
            "response_issue_rate": (
                response_issue_count / total if total else 0
            ),

            "target_time_match_rate": target_time_matches / total if total else 0,
            "temperature_match_rate": temp_matches / total if total else 0,
            "dew_point_match_rate": dew_matches / total if total else 0,
            "wind_speed_match_rate": wind_matches / total if total else 0,
        }

    return summary


def write_detailed_csv(scores):
    if not scores:
        return

    fieldnames = list(scores[0].keys())

    with open(OUTPUT_DETAILED_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scores)


def write_summary_csv(summary):
    rows = []

    for system, values in summary.items():
        row = {"system": system}
        row.update(values)
        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(OUTPUT_SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary):
    print()
    print("Hallucination Summary")
    print("---------------------")

    for system, values in summary.items():
        print(f"\nSystem: {system}")
        print(f"Total cases:                  {values['total_cases']}")

        print()
        print(f"Value hallucinated cases:     {values['value_hallucinated_cases']}")
        print(f"Value hallucination rate:     {values['value_hallucination_rate'] * 100:.2f}%")

        print()
        print(f"Policy violation cases:       {values['policy_violation_cases']}")
        print(f"Policy violation rate:        {values['policy_violation_rate'] * 100:.2f}%")

        print()
        print(f"Overall response issues:      {values['response_issue_cases']}")
        print(f"Response issue rate:          {values['response_issue_rate'] * 100:.2f}%")

        print()
        print(f"Target time match rate:       {values['target_time_match_rate'] * 100:.2f}%")
        print(f"Temperature match rate:       {values['temperature_match_rate'] * 100:.2f}%")
        print(f"Dew point match rate:         {values['dew_point_match_rate'] * 100:.2f}%")
        print(f"Wind speed match rate:        {values['wind_speed_match_rate'] * 100:.2f}%")


# -----------------------------
# Main
# -----------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    records = load_records()

    if not records:
        print("No records found. Make sure your LLM output files exist.")
        return

    scores = [
        score_record(record, fallback_system)
        for fallback_system, record in records
    ]

    summary = summarize(scores)

    write_detailed_csv(scores)
    write_summary_csv(summary)

    with open(OUTPUT_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)

    print()
    print("Saved files:")
    print(f"- {OUTPUT_DETAILED_CSV}")
    print(f"- {OUTPUT_SUMMARY_CSV}")
    print(f"- {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
