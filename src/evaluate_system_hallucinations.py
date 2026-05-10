import json
import os
import re
import csv


# -----------------------------
# Config
# -----------------------------

INPUT_FILES = {
    "single_tool": "results/single_tool_eval_outputs.jsonl",
    "agentic": "results/agentic_eval_outputs.jsonl",
}

OUTPUT_DETAILED_CSV = "results/system_hallucination_detailed_scores.csv"
OUTPUT_SUMMARY_CSV = "results/system_hallucination_summary.csv"
OUTPUT_SUMMARY_JSON = "results/system_hallucination_summary.json"


# -----------------------------
# Helpers
# -----------------------------

def normalize_number(value):
    return round(float(value), 2)


def response_contains_time(response, expected_time):
    return expected_time in response


def extract_numbers(response):
    """
    Extracts numeric values from the LLM response.

    This is intentionally simple because the expected responses should contain
    the exact forecast values from the tool output.
    """
    numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
    return [round(float(num), 2) for num in numbers]


def number_present(numbers, expected_value, tolerance=0.01):
    expected_value = normalize_number(expected_value)

    for num in numbers:
        if abs(num - expected_value) <= tolerance:
            return True

    return False


def detect_unsupported_terms(response):
    """
    These are terms the model should not introduce unless they were explicitly
    included in the final answer task.

    The tool output gives predictions only. It should not discuss actual values,
    errors, confidence, precipitation, humidity, or conditions.
    """
    unsupported_keywords = [
        "actual",
        "observed",
        "absolute error",
        "error",
        "humidity",
        "precipitation",
        "rain",
        "snow",
        "storm",
        "cloudy",
        "sunny",
        "fog",
        "visibility",
        "pressure",
        "confidence",
        "likely",
        "probably",
    ]

    lower_response = response.lower()

    found = []
    for term in unsupported_keywords:
        if term in lower_response:
            found.append(term)

    return found


def score_record(record):
    response = record.get("llm_response", "")

    expected_time = record["forecast_target_time"]
    expected_temp = normalize_number(record["expected_prediction"]["temperature"])
    expected_dew = normalize_number(record["expected_prediction"]["dew_point"])
    expected_wind = normalize_number(record["expected_prediction"]["wind_speed"])

    extracted_numbers = extract_numbers(response)

    target_time_match = response_contains_time(response, expected_time)
    temperature_match = number_present(extracted_numbers, expected_temp)
    dew_point_match = number_present(extracted_numbers, expected_dew)
    wind_speed_match = number_present(extracted_numbers, expected_wind)

    unsupported_terms = detect_unsupported_terms(response)

    missing_fields = []
    mismatch_fields = []

    if not target_time_match:
        missing_fields.append("forecast_target_time")
        mismatch_fields.append("forecast_target_time")

    if not temperature_match:
        missing_fields.append("temperature")
        mismatch_fields.append("temperature")

    if not dew_point_match:
        missing_fields.append("dew_point")
        mismatch_fields.append("dew_point")

    if not wind_speed_match:
        missing_fields.append("wind_speed")
        mismatch_fields.append("wind_speed")

    hallucinated = (
        not target_time_match
        or not temperature_match
        or not dew_point_match
        or not wind_speed_match
        or len(unsupported_terms) > 0
    )

    return {
        "case_id": record["case_id"],
        "system": record["system"],
        "request_time": record["request_time"],
        "forecast_target_time_expected": expected_time,
        "target_time_match": target_time_match,
        "expected_temperature": expected_temp,
        "temperature_match": temperature_match,
        "expected_dew_point": expected_dew,
        "dew_point_match": dew_point_match,
        "expected_wind_speed": expected_wind,
        "wind_speed_match": wind_speed_match,
        "missing_fields": ";".join(missing_fields),
        "mismatch_fields": ";".join(mismatch_fields),
        "unsupported_terms": ";".join(unsupported_terms),
        "hallucinated": hallucinated,
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
                    all_records.append(json.loads(line))

    return all_records


def summarize(scores):
    summary = {}

    systems = sorted(set(score["system"] for score in scores))

    for system in systems:
        system_scores = [score for score in scores if score["system"] == system]
        total = len(system_scores)

        hallucinated_count = sum(1 for score in system_scores if score["hallucinated"])
        target_time_matches = sum(1 for score in system_scores if score["target_time_match"])
        temp_matches = sum(1 for score in system_scores if score["temperature_match"])
        dew_matches = sum(1 for score in system_scores if score["dew_point_match"])
        wind_matches = sum(1 for score in system_scores if score["wind_speed_match"])
        unsupported_count = sum(1 for score in system_scores if score["unsupported_terms"])

        summary[system] = {
            "total_cases": total,
            "hallucinated_cases": hallucinated_count,
            "hallucination_rate": hallucinated_count / total if total else 0,
            "target_time_match_rate": target_time_matches / total if total else 0,
            "temperature_match_rate": temp_matches / total if total else 0,
            "dew_point_match_rate": dew_matches / total if total else 0,
            "wind_speed_match_rate": wind_matches / total if total else 0,
            "unsupported_claim_cases": unsupported_count,
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
        print(f"Total cases:              {values['total_cases']}")
        print(f"Hallucinated cases:       {values['hallucinated_cases']}")
        print(f"Hallucination rate:       {values['hallucination_rate'] * 100:.2f}%")
        print(f"Target time match rate:   {values['target_time_match_rate'] * 100:.2f}%")
        print(f"Temperature match rate:   {values['temperature_match_rate'] * 100:.2f}%")
        print(f"Dew point match rate:     {values['dew_point_match_rate'] * 100:.2f}%")
        print(f"Wind speed match rate:    {values['wind_speed_match_rate'] * 100:.2f}%")
        print(f"Unsupported claim cases:  {values['unsupported_claim_cases']}")


# -----------------------------
# Main
# -----------------------------

def main():
    os.makedirs("results", exist_ok=True)

    records = load_records()

    if not records:
        print("No records found. Make sure your LLM output files exist.")
        return

    scores = [score_record(record) for record in records]
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
