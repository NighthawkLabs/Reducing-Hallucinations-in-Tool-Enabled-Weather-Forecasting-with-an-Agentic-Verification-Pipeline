# Reducing Hallucinations in Tool-Enabled Weather Forecasting with an Agentic Verification Pipeline

## Overview

This project compares two tool-enabled LLM systems in a structured weather forecasting task:

1. A **single-tool LLM baseline**
2. An **agentic verification pipeline**

The goal is to evaluate whether adding a verification step can reduce hallucinations in natural-language weather forecast responses.

The system uses historical NOAA weather data to train a Random Forest forecasting model. The forecasting model predicts temperature, dew point, and wind speed 24 hours ahead using a 72-hour historical input window. The LLM systems then convert the structured forecast output into a natural-language response.

The main research question is:

**Does an agentic verification pipeline reduce hallucinations compared to a single-tool LLM baseline when both systems are given the same structured weather forecasting tool output?**

---

## Project Motivation

Large language models can produce fluent responses that sound correct but contain unsupported or incorrect information. In weather forecasting, this is especially important because forecast values are measurable and easy to verify.

This project uses weather forecasting as a controlled test domain. The forecasting tool produces structured output, and the LLM is expected to explain that output without changing values, inventing new fields, or adding unsupported claims.

---

## Dataset

This project uses historical NOAA Local Climatological Data from 2003 to 2006.

The cleaned dataset contains hourly weather observations with the following fields:

- Date/time
- Temperature
- Dew point
- Wind speed

The cleaned dataset is used to create supervised learning examples:

- **Input:** 72 hours of previous weather observations
- **Target:** weather conditions 24 hours in the future

---

## Machine Learning Forecasting Model

The forecasting model is a `RandomForestRegressor` trained to predict three weather variables:

- Temperature
- Dew point
- Wind speed

The model uses flattened 72-hour windows as input features. Each input window contains temperature, dew point, and wind speed values from the previous 72 hours.

Representative model performance:

- Temperature MAE: approximately 4.3–4.5
- Dew Point MAE: approximately 4.6–4.7
- Wind Speed MAE: approximately 3.2–3.4
- Average MAE: approximately 4.1

The forecasting model output is used as the structured tool output for both LLM systems.

---

## LLM Systems Compared

### 1. Single-Tool LLM Baseline

The single-tool baseline receives the structured forecast output and directly generates a natural-language weather response.

The LLM is instructed not to:

- Change the forecast timestamp
- Change the predicted temperature
- Change the predicted dew point
- Change the predicted wind speed
- Add unsupported claims

This system represents a simple tool-enabled LLM workflow.

### 2. Agentic Verification Pipeline

The agentic system adds a verification step after the initial response is generated.

The verification stage:

1. Extracts forecast values from the generated response
2. Compares those values against the original tool output
3. Flags missing fields, mismatches, or unsupported terms
4. Produces a corrected final response

This system is designed to test whether an agentic verification step can reduce hallucinations.

---

## Hallucination Definition

In this project, a hallucination is defined as any LLM response that includes information not supported by the structured forecasting tool output.

A response is considered hallucinated if it contains:

- An incorrect forecast timestamp
- An incorrect temperature value
- An incorrect dew point value
- An incorrect wind speed value
- Missing required forecast fields
- Unsupported weather claims
- Extra values or terms not present in the tool output

The evaluation scripts compare the final LLM response against the expected structured output field by field.

---

## AI Disclosure

In this project, I used ChatGPT and Claude for assistance with code debugging, documentation, and brainstorming

## Repository Structure

```text
weatherProject/
│
├── data/
│   └── clean/
│       └── noaaDataCleaned.csv
│
├── results/
│   ├── eval_forecast_cases.jsonl
│   ├── single_tool_eval_outputs.jsonl
│   ├── agentic_eval_outputs.jsonl
│   ├── single_tool_hallucination_scores.csv
│   ├── system_hallucination_summary.csv
│   └── system_hallucination_summary.json
│
├── src/
│   ├── train_baseline.py
│   ├── predict_baseline.py
│   ├── verify_baseline.py
│   ├── generate_eval_cases.py
│   ├── single_tool_local_llm_baseline.py
│   ├── run_single_tool_on_eval_cases.py
│   ├── run_agentic_on_eval_cases.py
│   ├── evaluate_single_tool_hallucinations.py
│   └── evaluate_system_hallucinations.py
│
├── .gitignore
├── requirements.txt
└── README.md
