"""
Legacy single-tool LLM baseline prototype.

This file was an early version of the single-tool baseline pipeline. It used
the OpenRouter/OpenAI API to generate a response from one forecast tool call
for a single request time.

This script was not used for the final evaluation results. The final project
uses the local TinyLlama-based batch runner, run_single_tool_on_eval_cases.py,
which processes the shared eval_forecast_cases.jsonl file so the single-tool
baseline and agentic verification system are evaluated on the same forecast
cases.

This file is kept in the repository for transparency and to show the evolution
of the project, but it should be treated as a legacy prototype rather than the
final experimental pipeline.
"""


import os
import json
from datetime import datetime

from openai import OpenAI

from predict_baseline import predict_weather_from_request_time


# -----------------------------
# Configuration
# -----------------------------

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SRC_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_LOG_PATH = os.path.join(RESULTS_DIR, "single_tool_llm_outputs.jsonl")

# You can change this model depending on what you use through OpenRouter.
MODEL_NAME = "openai/gpt-4o-mini"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)


# -----------------------------
# Single tool wrapper
# -----------------------------

def weather_forecast_tool(request_time):
    """
    This is the only tool the baseline LLM is allowed to use.

    It calls the frozen Random Forest baseline model and returns structured JSON.
    """

    return predict_weather_from_request_time(request_time)


# -----------------------------
# LLM response generation
# -----------------------------

def generate_single_tool_response(user_request, tool_output):
    """
    The baseline LLM does not plan, validate, or call multiple tools.
    It receives the user's request and the output of exactly one tool call.
    """

    system_prompt = """
You are a simple weather forecasting assistant.

You are given the user's request and the output of a weather forecasting tool.

Rules:
1. Use only the tool output.
2. Do not invent weather values.
3. Do not mention causes, storms, fronts, pressure systems, or outside weather details unless they appear in the tool output.
4. Report the predicted temperature, dew point, and wind speed.
5. Mention the forecast target time.
6. Keep the response short.
"""

    user_prompt = f"""
User request:
{user_request}

Tool output:
{json.dumps(tool_output, indent=2)}

Write the final forecast response.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        temperature=0
    )

    return response.choices[0].message.content


# -----------------------------
# Full baseline pipeline
# -----------------------------

def run_single_tool_baseline(request_time):
    """
    Runs the complete single-tool baseline:
    1. Build a user request.
    2. Call the weather forecasting tool once.
    3. Give the tool output to the LLM.
    4. Save the full result for evaluation.
    """

    user_request = (
        f"Using the available NOAA weather data, predict the temperature, "
        f"dew point, and wind speed 24 hours after {request_time}."
    )

    tool_output = weather_forecast_tool(request_time)

    llm_response = generate_single_tool_response(
        user_request=user_request,
        tool_output=tool_output
    )

    record = {
        "run_timestamp": datetime.now().isoformat(),
        "system_type": "single_tool_llm_baseline",
        "model_name": MODEL_NAME,
        "request_time": request_time,
        "user_request": user_request,
        "tool_output": tool_output,
        "llm_response": llm_response
    }

    with open(OUTPUT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


# -----------------------------
# Command line test
# -----------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        request_time = sys.argv[1]
    else:
        request_time = "2005-04-12 08:00:00"

    result = run_single_tool_baseline(request_time)

    print("\nSingle-Tool LLM Baseline Result")
    print("--------------------------------")
    print(result["llm_response"])

    print("\nSaved full output to:")
    print(OUTPUT_LOG_PATH)
