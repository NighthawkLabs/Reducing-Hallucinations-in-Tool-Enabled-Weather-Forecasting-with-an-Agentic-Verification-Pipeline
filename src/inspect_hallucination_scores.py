import pandas as pd

# path to the detailed hallucination scoring output created by the evaluation script 
SCORES_PATH = "results/system_hallucination_detailed_scores.csv"

# load the detailed per case hallucination scores into a dataframe 
df = pd.read_csv(SCORES_PATH)

print("\nColumns:")
print(df.columns.tolist())

print("\nCounts by system and hallucinated:")
print(df.groupby(["system", "hallucinated"]).size())
"""
# Aggregate summary by system

Compute the main evaluation metrics for each system:
total number of cases
number of hallucinated/problematic cases
field-level match rates
"""
print("\nSummary by system:")
summary = df.groupby("system").agg(
    total_cases=("case_id", "count"),
    hallucinated_cases=("hallucinated", "sum"),
    target_time_match_rate=("target_time_match", "mean"),
    temperature_match_rate=("temperature_match", "mean"),
    dew_point_match_rate=("dew_point_match", "mean"),
    wind_speed_match_rate=("wind_speed_match", "mean"),
)
summary["hallucination_rate"] = summary["hallucinated_cases"] / summary["total_cases"]
print(summary)

"""
Inspect sample hallucinated single-tool cases 
"""

print("\nSample hallucinated single-tool cases:")
hallucinated_single = df[(df["system"] == "single_tool") & (df["hallucinated"] == True)]

for _, row in hallucinated_single.head(5).iterrows():
    print("\n" + "-" * 80)
    print(f"Case ID: {row['case_id']}")
    print(f"Expected target time: {row['forecast_target_time_expected']}")
    print(f"Expected temp/dew/wind: {row['expected_temperature']}, {row['expected_dew_point']}, {row['expected_wind_speed']}")
    print(f"Missing fields: {row.get('missing_fields', '')}")
    print(f"Mismatch fields: {row.get('mismatch_fields', '')}")
    print(f"Unsupported terms: {row.get('unsupported_terms', '')}")
    print("LLM response:")
    print(row["llm_response"])

print("\nSample non-hallucinated single-tool cases:")
clean_single = df[(df["system"] == "single_tool") & (df["hallucinated"] == False)]

for _, row in clean_single.head(3).iterrows():
    print("\n" + "-" * 80)
    print(f"Case ID: {row['case_id']}")
    print(f"Expected target time: {row['forecast_target_time_expected']}")
    print(f"Expected temp/dew/wind: {row['expected_temperature']}, {row['expected_dew_point']}, {row['expected_wind_speed']}")
    print("LLM response:")
    print(row["llm_response"])
