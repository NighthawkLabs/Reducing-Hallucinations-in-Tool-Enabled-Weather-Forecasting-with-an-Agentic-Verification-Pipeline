import json
import os
from datetime import datetime
from llama_cpp import Llama


# Config

# Path to the local GGUF model used 
MODEL_PATH = "/home/sgibso34/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Build paths relative to this script so the runner can be executed from
# different working directories without hardcoding project-wide paths.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

INPUT_CASES = os.path.join(RESULTS_DIR, "eval_forecast_cases.jsonl")
OUTPUT_JSONL = os.path.join(RESULTS_DIR, "agentic_eval_outputs.jsonl")

# Set to a small number like 5 for testing, or None to run all cases.
MAX_CASES = None

# Reproducibility note
# ---------------------------------------------------------------------
# The trained Random Forest model file, baseline_model.pkl, was not uploaded
# to GitHub because it exceeded GitHub's file size limit. To regenerate the
# saved forecasting model, run the training script:
#
#     python src/train_baseline.py
#
# After training, the baseline model should be saved in the expected models
# directory and can be used to regenerate eval_forecast_cases.jsonl if needed.
# ---------------------------------------------------------------------



# File helpers

def read_jsonl(path):
    records = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find input file: {path}")

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records



# Agentic Steps

def build_tool_output(case):
    """
    Build the restricted forecast tool output.

    The agentic system should verify and report only these forecast values.
    Actual NOAA values and absolute errors are preserved in the saved record
    for later analysis, but they are not used in the final answer.
    """

    return {
        "request_time": case["request_time"],
        "forecast_target_time": case["forecast_target_time"],
        "predicted_temperature": case["prediction"]["temperature"],
        "predicted_dew_point": case["prediction"]["dew_point"],
        "predicted_wind_speed": case["prediction"]["wind_speed"],
    }


def build_planning_prompt(case):
    """
    Build the prompt used for the agent's planning step.

    The LLM is asked only to make a short plan. It is not allowed to produce
    the final weather forecast here. This separates the agentic reasoning step
    from the deterministic verified final-answer step.
    """
    tool_output = build_tool_output(case)

    return f"""You are part of an agentic weather forecasting system.

The user asks:
Give me the weather forecast for 24 hours after {case["request_time"]}.

The weather forecast tool has already returned this structured output:
{json.dumps(tool_output, indent=2)}

Your job is only to make a short plan before the final answer is produced.

Return a short plan with these steps:
1. Identify the request time.
2. Use the forecast tool output.
3. Verify the forecast target time, predicted temperature, predicted dew point, and predicted wind speed.
4. Produce a final answer using only verified values.

Do not produce the final weather forecast in this planning step.
"""


def generate_agent_plan(llm, prompt):
    """
    Generate the agent's short planning response using the local LLM.

    The planning text is saved for transparency, but it is not trusted as the
    final answer. The final response is built separately from verified
    structured values.
    """
    output = llm(
        prompt,
        max_tokens=160,
        temperature=0.1,
        top_p=0.9,
        stop=["</s>"],
    )

    return output["choices"][0]["text"].strip()


def verify_tool_output(tool_output):
    """
    Verification step.

    The agent checks that the restricted tool output contains all required
    fields before a final response is allowed.
    """

    required_fields = [
        "request_time",
        "forecast_target_time",
        "predicted_temperature",
        "predicted_dew_point",
        "predicted_wind_speed",
    ]

    missing_fields = []

    for field in required_fields:
        if field not in tool_output or tool_output[field] is None:
            missing_fields.append(field)

    verified = len(missing_fields) == 0

    verification = {
        "verified": verified,
        "missing_fields": missing_fields,
        "request_time": tool_output.get("request_time"),
        "forecast_target_time": tool_output.get("forecast_target_time"),
        "predicted_temperature": tool_output.get("predicted_temperature"),
        "predicted_dew_point": tool_output.get("predicted_dew_point"),
        "predicted_wind_speed": tool_output.get("predicted_wind_speed"),
    }

    return verification


def build_verified_final_answer(verification):
    """
    Build the final response from verified structured values.

    This intentionally does not ask the LLM to write the final answer. The goal
    is to test whether an agentic verification pipeline can prevent unsupported
    additions, timestamp changes, and numeric drift.
    """

    if not verification["verified"]:
        missing = ", ".join(verification["missing_fields"])
        return (
            "I cannot produce a forecast because the verified tool output is "
            f"missing required fields: {missing}."
        )

    return (
        f"The forecast target time is {verification['forecast_target_time']}. "
        f"The predicted temperature is {verification['predicted_temperature']}°F. "
        f"The predicted dew point is {verification['predicted_dew_point']}°F. "
        f"The predicted wind speed is {verification['predicted_wind_speed']} mph."
    )


# Main

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading LLM for agent planning...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )

    print("Loading evaluation cases...")
    cases = read_jsonl(INPUT_CASES)

    if MAX_CASES is not None:
        cases = cases[:MAX_CASES]

    print(f"Running agentic system on {len(cases)} cases...")
    print(f"Input cases: {INPUT_CASES}")
    print(f"Output file: {OUTPUT_JSONL}")

    run_timestamp = datetime.now().isoformat()

    with open(OUTPUT_JSONL, "w") as out:
        for case in cases:
            tool_output = build_tool_output(case)

            planning_prompt = build_planning_prompt(case)
            agent_plan = generate_agent_plan(llm, planning_prompt)

            verification = verify_tool_output(tool_output)
            final_answer = build_verified_final_answer(verification)

            record = {
                "run_timestamp": run_timestamp,
                "system": "agentic",
                "system_type": "agentic_verification_pipeline",
                "model_path": MODEL_PATH,

                "case_id": case["case_id"],
                "request_time": case["request_time"],
                "forecast_target_time": case["forecast_target_time"],

                # Restricted forecast values used by the agentic system.
                "tool_output": tool_output,

                # Preserve original generated eval case fields for later analysis.
                "prediction": case["prediction"],
                "actual": case["actual"],
                "absolute_error": case["absolute_error"],

                "user_request": (
                    f"Give me the weather forecast for 24 hours after "
                    f"{case['request_time']}."
                ),

                "agent_plan": agent_plan,
                "verification": verification,

                # Store both names so evaluators can read either one.
                "final_response": final_answer,
                "llm_response": final_answer,
            }

            out.write(json.dumps(record) + "\n")

            print(f"\nCase {case['case_id']} complete")
            print(final_answer)

    print()
    print("Agentic evaluation complete.")
    print(f"Saved outputs to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
