import json
import os
from datetime import datetime
from llama_cpp import Llama


# -----------------------------
# Config
# -----------------------------

MODEL_PATH = "/home/sgibso34/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

INPUT_CASES = os.path.join(RESULTS_DIR, "eval_forecast_cases.jsonl")
OUTPUT_JSONL = os.path.join(RESULTS_DIR, "single_tool_eval_outputs.jsonl")

# Set to a small number like 5 for testing, or None to run all cases.
MAX_CASES = None


# -----------------------------
# File helpers
# -----------------------------

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


# -----------------------------
# Prompting
# -----------------------------

def build_tool_output(case):
    """
    Build the restricted tool output shown to the LLM.

    Important:
    Do not include actual NOAA values or absolute errors here.
    The single-tool LLM should only see the forecast prediction.
    """

    return {
        "request_time": case["request_time"],
        "forecast_target_time": case["forecast_target_time"],
        "predicted_temperature": case["prediction"]["temperature"],
        "predicted_dew_point": case["prediction"]["dew_point"],
        "predicted_wind_speed": case["prediction"]["wind_speed"],
    }


def build_prompt(case):
    tool_output = build_tool_output(case)

    prompt = f"""You are a weather forecasting assistant.

The weather forecast tool has already been called once. Use ONLY the tool output below to answer the user.

Tool output:
{json.dumps(tool_output, indent=2)}

User request:
Give me the weather forecast for 24 hours after {case["request_time"]}.

Rules:
- Do not invent extra weather details.
- Do not report actual observed values.
- Do not report absolute error values.
- Preserve the forecast target time exactly.
- Preserve the predicted temperature, dew point, and wind speed exactly.
- Your answer should include only the forecast target time, predicted temperature, predicted dew point, and predicted wind speed.

Write a concise final answer.
"""

    return prompt


def generate_response(llm, prompt):
    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.2,
        top_p=0.9,
        stop=["</s>"],
    )

    return output["choices"][0]["text"].strip()


# -----------------------------
# Main
# -----------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading LLM...")
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

    print(f"Running single-tool baseline on {len(cases)} cases...")
    print(f"Input cases: {INPUT_CASES}")
    print(f"Output file: {OUTPUT_JSONL}")

    run_timestamp = datetime.now().isoformat()

    with open(OUTPUT_JSONL, "w") as out:
        for case in cases:
            prompt = build_prompt(case)
            response = generate_response(llm, prompt)
            tool_output = build_tool_output(case)

            record = {
                "run_timestamp": run_timestamp,
                "system_type": "single_tool_local_llm_baseline",
                "model_path": MODEL_PATH,

                "case_id": case["case_id"],
                "request_time": case["request_time"],
                "forecast_target_time": case["forecast_target_time"],

                # Preserve the restricted tool output the LLM actually saw.
                "tool_output": tool_output,

                # Preserve original generated eval case fields.
                # This keeps prediction and actual available for later evaluation.
                "prediction": case["prediction"],
                "actual": case["actual"],
                "absolute_error": case["absolute_error"],

                "user_request": (
                    f"Give me the weather forecast for 24 hours after "
                    f"{case['request_time']}."
                ),
                "llm_response": response,
            }

            out.write(json.dumps(record) + "\n")

            print(f"\nCase {case['case_id']} complete")
            print(response)

    print()
    print("Single-tool evaluation complete.")
    print(f"Saved outputs to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
