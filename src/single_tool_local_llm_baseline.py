"""
Legacy local single-tool LLM baseline prototype.

This file was an intermediate version of the single-tool baseline pipeline.
It replaced the earlier OpenRouter/OpenAI version with a local llama-cpp model,
but it still runs the forecasting tool for one request time at a time.

This script was not used for the final evaluation results. The final project
uses run_single_tool_on_eval_cases.py, which processes the shared
eval_forecast_cases.jsonl file so the single-tool baseline and agentic
verification system are evaluated on the same set of forecast cases.

This file is kept in the repository for transparency and to show the evolution
of the project pipeline. It should be treated as a legacy prototype rather than
the final experimental runner.
"""

import os
import json
from datetime import datetime

from llama_cpp import Llama

from predict_baseline import predict_weather_from_request_time


# Configuration

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SRC_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_LOG_PATH = os.path.join(RESULTS_DIR, "single_tool_local_llm_outputs.jsonl")

# Put your local GGUF model path here.
# Example:
# MODEL_PATH = "/home/sgibso34/models/tinyllama-1.1b-intermediate-step-1431k-3t.Q4_K_M.gguf"
MODEL_PATH = os.environ.get("LOCAL_GGUF_PATH")

if MODEL_PATH is None:
    raise ValueError(
        "LOCAL_GGUF_PATH is not set. Set it to the path of your local GGUF model.\n"
        "Example:\n"
        "export LOCAL_GGUF_PATH='/home/sgibso34/models/tinyllama-1.1b-intermediate-step-1431k-3t.Q4_K_M.gguf'"
    )


# Load local LLM

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)


# Forecasting tool

def weather_forecast_tool(request_time):
    """
    This is the only tool the local LLM baseline is allowed to use.
    """

    return predict_weather_from_request_time(request_time)


# Prompt formatting

def build_prompt(user_request, tool_output):
    """
    Builds a strict prompt for the local LLM.

    The model is instructed to use only the tool output.
    """

    prompt = f"""
You are a simple weather forecasting assistant.

You are given a user request and the output of a forecasting tool.

Rules:
- Use only the tool output.
- Do not invent weather values.
- Do not add explanations about storms, fronts, pressure, clouds, or causes.
- Report only the forecast target time, predicted temperature, predicted dew point, and predicted wind speed.
- Keep the answer short.

User request:
{user_request}

Tool output:
{json.dumps(tool_output, indent=2)}

Final answer:
"""

    return prompt.strip()


# LLM response generation

def generate_local_llm_response(user_request, tool_output):
    prompt = build_prompt(user_request, tool_output)

    output = llm(
        prompt,
        max_tokens=150,
        temperature=0.0,
        top_p=1.0,
        stop=["User request:", "Tool output:"]
    )

    response_text = output["choices"][0]["text"].strip()

    return response_text

# Full baseline pipeline

def run_single_tool_local_llm_baseline(request_time):
    """
    Runs the complete local LLM single-tool baseline:
    1. Build a weather request.
    2. Call the frozen forecasting tool exactly once.
    3. Give the tool output to the local LLM.
    4. Save the response for evaluation.
    """

    user_request = (
        f"Using the available NOAA weather data, predict the temperature, "
        f"dew point, and wind speed 24 hours after {request_time}."
    )

    tool_output = weather_forecast_tool(request_time)

    llm_response = generate_local_llm_response(
        user_request=user_request,
        tool_output=tool_output
    )

    record = {
        "run_timestamp": datetime.now().isoformat(),
        "system_type": "single_tool_local_llm_baseline",
        "model_path": MODEL_PATH,
        "request_time": request_time,
        "user_request": user_request,
        "tool_output": tool_output,
        "llm_response": llm_response
    }

    with open(OUTPUT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


# Command line test

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        request_time = sys.argv[1]
    else:
        request_time = "2005-04-12 08:00:00"

    result = run_single_tool_local_llm_baseline(request_time)

    print("\nSingle-Tool Local LLM Baseline Result")
    print("-------------------------------------")
    print(result["llm_response"])

    print("\nSaved full output to:")
    print(OUTPUT_LOG_PATH)
