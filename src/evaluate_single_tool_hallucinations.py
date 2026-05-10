import os
import re
import json
import csv

# from predict_baseline import predict_weather_from_request_time


# Config
# Paths are relative to this script

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# input file containing the single-tool LLM responses to evaluate  
INPUT_JSONL_PATH = os.path.join(
    RESULTS_DIR,
    "single_tool_local_llm_outputs.jsonl"
)

# Output files for detailed results
OUTPUT_CSV_PATH = os.path.join(
    RESULTS_DIR,
    "single_tool_hallucination_scores.csv"
)

OUTPUT_JSONL_PATH = os.path.join(
    RESULTS_DIR,
    "single_tool_hallucination_scores.jsonl"
)

# Allows tiny formatting or rounding differences.
NUMERIC_TOLERANCE = 0.05

# File helpers

def read_jsonl(path):
    """
    Read a JSONL file where each non-empty line is one JSON object 

    single-tool output file stores one forecast/response record per line
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


def write_jsonl(scores, path):
    with open(path, "w") as f:
        for score in scores:
            f.write(json.dumps(score) + "\n")


def write_csv(scores, path):
    if not scores:
        return

    fieldnames = [
        "request_time",
        "forecast_target_time_expected",
        "forecast_target_time_observed",
        "target_time_match",

        "expected_temperature",
        "observed_temperature",
        "temperature_match",

        "expected_dew_point",
        "observed_dew_point",
        "dew_point_match",

        "expected_wind_speed",
        "observed_wind_speed",
        "wind_speed_match",

        "missing_fields",
        "mismatch_fields",
        "extra_terms",

        "value_hallucinated",
        "policy_violation",
        "response_issue",

        "llm_response",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for score in scores:
            row = score.copy()
            row["missing_fields"] = ";".join(row["missing_fields"])
            row["mismatch_fields"] = ";".join(row["mismatch_fields"])
            row["extra_terms"] = ";".join(row["extra_terms"])
            writer.writerow(row)



# Text parsing helpers


def normalize_text(text):
    return text.lower().replace("_", " ")


def extract_datetime(text):
    """
    Finds a datetime like:
        2005-04-13 08:00:00
    """

    pattern = r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"
    match = re.search(pattern, text)

    if match:
        return match.group(0)

    return None


def extract_number_near_label(text, labels):
    """
    Attempts to extract a number near a label.

    Example matches:
        temperature is 57.39
        predicted temperature: 57.39
        dew point of 54.07
        wind speed = 5.64
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
    Extracts every integer or decimal. used if label-based extraction fails 
    """
    pattern = r"-?\d+(?:\.\d+)?"
    return [float(x) for x in re.findall(pattern, text)]


def extract_llm_forecast_values(llm_response):
    """
    Extracts forecast target time, predicted temperature, predicted dew point,
    and predicted wind speed from the LLM's natural-language response.

    First tries label-based extraction. If that fails, it falls back to assuming
    the first three numbers after the timestamp are:
        temperature, dew point, wind speed
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

    # Fallback: infer values from number order after target time.
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
    """
    Check whether an observed LLM value matches the expected tool value with a small tolerance allowed for rounding errors
    """
    if observed is None:
        return False

    return abs(expected - observed) <= tolerance


def find_extra_terms(llm_response):
    """
    Finds content that is not part of the required final forecast format.

    Important distinction:
    - These are not necessarily hallucinations.
    - If the term came from the tool output, it is an instruction/policy violation,
      not a value hallucination.
    """

    text = normalize_text(llm_response)

    extra_terms_to_flag = [
        "actual",
        "absolute error",
        "abs error",
        "average error",
        "storm",
        "front",
        "pressure",
        "cloud",
        "rain",
        "precipitation",
        "humidity",
        "cause",
        "caused",
        "celsius", 
        "meters per second", 
        "°c",
        "m/s",
        "km/h",
        "kilometers per hour",
        "conversion",
        "converted",
        "note",
        "explanation",
        "additional details",
    ]

    found_terms = []

    for term in extra_terms_to_flag:
        if term in text:
            found_terms.append(term)

    return found_terms


# Scoring

def get_expected_prediction(record):
    """
    Gets the expected forecast values from the record itself
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
        "predicted_temperature": prediction["predicted_temperature"],
        "predicted_dew_point": prediction["predicted_dew_point"],
        "predicted_wind_speed": prediction["predicted_wind_speed"],
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


def score_record(record):
    """
    Score one LLM response at a time against the expected structured forecast output
    """
    request_time = record["request_time"]
    llm_response = record.get("llm_response", "")

    expected_tool_output = get_expected_prediction(record)
    extracted = extract_llm_forecast_values(llm_response)

    expected_time = expected_tool_output["forecast_target_time"]
    expected_temp = expected_tool_output["predicted_temperature"]
    expected_dew = expected_tool_output["predicted_dew_point"]
    expected_wind = expected_tool_output["predicted_wind_speed"]

    observed_time = extracted["forecast_target_time"]
    observed_temp = extracted["predicted_temperature"]
    observed_dew = extracted["predicted_dew_point"]
    observed_wind = extracted["predicted_wind_speed"]

    target_time_match = observed_time == expected_time
    temperature_match = numeric_match(expected_temp, observed_temp)
    dew_point_match = numeric_match(expected_dew, observed_dew)
    wind_speed_match = numeric_match(expected_wind, observed_wind)

    missing_fields = []

    if observed_time is None:
        missing_fields.append("forecast_target_time")
    if observed_temp is None:
        missing_fields.append("predicted_temperature")
    if observed_dew is None:
        missing_fields.append("predicted_dew_point")
    if observed_wind is None:
        missing_fields.append("predicted_wind_speed")

    mismatch_fields = []

    if observed_time is not None and not target_time_match:
        mismatch_fields.append("forecast_target_time")
    if observed_temp is not None and not temperature_match:
        mismatch_fields.append("predicted_temperature")
    if observed_dew is not None and not dew_point_match:
        mismatch_fields.append("predicted_dew_point")
    if observed_wind is not None and not wind_speed_match:
        mismatch_fields.append("predicted_wind_speed")

    extra_terms = find_extra_terms(llm_response)

    value_hallucinated = (
        len(missing_fields) > 0
        or len(mismatch_fields) > 0
    )

    policy_violation = len(extra_terms) > 0

    response_issue = value_hallucinated or policy_violation

    return {
        "request_time": request_time,

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

        "missing_fields": missing_fields,
        "mismatch_fields": mismatch_fields,
        "extra_terms": extra_terms,

        "value_hallucinated": value_hallucinated,
        "policy_violation": policy_violation,
        "response_issue": response_issue,

        "llm_response": llm_response,
    }


# Summary output


def print_summary(scores):
    total = len(scores)

    if total == 0:
        print("No records scored.")
        return

    value_hallucinated_count = sum(1 for s in scores if s["value_hallucinated"])
    policy_violation_count = sum(1 for s in scores if s["policy_violation"])
    response_issue_count = sum(1 for s in scores if s["response_issue"])

    target_time_matches = sum(1 for s in scores if s["target_time_match"])
    temp_matches = sum(1 for s in scores if s["temperature_match"])
    dew_matches = sum(1 for s in scores if s["dew_point_match"])
    wind_matches = sum(1 for s in scores if s["wind_speed_match"])

    print("\nSingle-Tool Local LLM Evaluation")
    print("--------------------------------")
    print(f"Records scored:              {total}")

    print()
    print(f"Value hallucinations:        {value_hallucinated_count}")
    print(f"Value hallucination rate:    {value_hallucinated_count / total:.2%}")

    print()
    print(f"Policy violations:           {policy_violation_count}")
    print(f"Policy violation rate:       {policy_violation_count / total:.2%}")

    print()
    print(f"Overall response issues:     {response_issue_count}")
    print(f"Response issue rate:         {response_issue_count / total:.2%}")

    print()
    print(f"Target time match rate:      {target_time_matches / total:.2%}")
    print(f"Temperature match rate:      {temp_matches / total:.2%}")
    print(f"Dew point match rate:        {dew_matches / total:.2%}")
    print(f"Wind speed match rate:       {wind_matches / total:.2%}")

    print("\nSaved CSV:")
    print(OUTPUT_CSV_PATH)

    print("\nSaved JSONL:")
    print(OUTPUT_JSONL_PATH)



def main():
    records = read_jsonl(INPUT_JSONL_PATH)

    scores = []

    for record in records:
        score = score_record(record)
        scores.append(score)

    write_csv(scores, OUTPUT_CSV_PATH)
    write_jsonl(scores, OUTPUT_JSONL_PATH)
    print_summary(scores)


if __name__ == "__main__":
    main()
