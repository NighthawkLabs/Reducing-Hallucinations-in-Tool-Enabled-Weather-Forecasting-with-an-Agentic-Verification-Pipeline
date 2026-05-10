import json
import os
from llama_cpp import Llama


# -----------------------------
# Config
# -----------------------------

MODEL_PATH = "/home/sgibso34/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

INPUT_CASES = "results/eval_forecast_cases.jsonl"
OUTPUT_JSONL = "results/agentic_eval_outputs.jsonl"

MAX_CASES = None  # set to 5 for quick testing, or None for all cases


# -----------------------------
# Agentic Steps
# -----------------------------

def build_planning_prompt(case):
    return f"""You are part of an agentic weather forecasting system.

The user asks:
Give me the weather forecast for 24 hours after {case["request_time"]}.

Your job is only to identify what information is needed before answering.

Return a short plan with these steps:
1. Identify the request time.
2. Use the forecast tool output.
3. Verify the required forecast fields.
4. Produce a final answer using only verified values.
"""


def generate_agent_plan(llm, prompt):
    output = llm(
        prompt,
        max_tokens=160,
        temperature=0.1,
        top_p=0.9,
        stop=["</s>"],
    )

    return output["choices"][0]["text"].strip()


def verify_tool_output(case):
    """
    This is the agent's verification step.

    The agent checks that the required fields exist before allowing a final
    response to be generated.
    """
    required_top_level = [
        "request_time",
        "forecast_target_time",
        "prediction",
    ]

    required_prediction_fields = [
        "temperature",
        "dew_point",
        "wind_speed",
    ]

    missing_fields = []

    for field in required_top_level:
        if field not in case:
            missing_fields.append(field)

    if "prediction" in case:
        for field in required_prediction_fields:
            if field not in case["prediction"]:
                missing_fields.append(f"prediction.{field}")

    verified = len(missing_fields) == 0

    verification = {
        "verified": verified,
        "missing_fields": missing_fields,
        "request_time": case.get("request_time"),
        "forecast_target_time": case.get("forecast_target_time"),
        "temperature": case.get("prediction", {}).get("temperature"),
        "dew_point": case.get("prediction", {}).get("dew_point"),
        "wind_speed": case.get("prediction", {}).get("wind_speed"),
    }

    return verification


def build_verified_final_answer(verification):
    """
    The final answer is intentionally constrained.

    This prevents the final response from adding unsupported actual values,
    errors, or invented weather conditions.
    """
    if not verification["verified"]:
        return (
            "I cannot produce a forecast because the tool output is missing "
            f"required fields: {verification['missing_fields']}"
        )

    return (
        f"For the 24-hour forecast target time {verification['forecast_target_time']}, "
        f"the predicted temperature is {verification['temperature']}, "
        f"the predicted dew point is {verification['dew_point']}, "
        f"and the predicted wind speed is {verification['wind_speed']}."
    )


# -----------------------------
# Main
# -----------------------------

def main():
    os.makedirs("results", exist_ok=True)

    print("Loading LLM for agent planning...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )

    print("Loading evaluation cases...")
    cases = []
    with open(INPUT_CASES, "r") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    if MAX_CASES is not None:
        cases = cases[:MAX_CASES]

    print(f"Running agentic system on {len(cases)} cases...")

    with open(OUTPUT_JSONL, "w") as out:
        for case in cases:
            planning_prompt = build_planning_prompt(case)
            agent_plan = generate_agent_plan(llm, planning_prompt)

            verification = verify_tool_output(case)
            final_answer = build_verified_final_answer(verification)

            record = {
                "case_id": case["case_id"],
                "system": "agentic",
                "request_time": case["request_time"],
                "forecast_target_time": case["forecast_target_time"],
                "expected_prediction": case["prediction"],
                "agent_plan": agent_plan,
                "verification": verification,
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
