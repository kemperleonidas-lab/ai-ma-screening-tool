# AI M&A Financial Screening Tool

This project is an AI-supported financial screening tool designed to accelerate early-stage evaluation of potential M&A targets.

It uses Python, LangGraph-style workflows, and structured analysis logic to automate parts of financial research and company screening.

## Problem

Early-stage M&A analysis is time-intensive and requires manual collection and interpretation of company data. This project aims to reduce that workload by structuring the process into an automated pipeline.

## Features

- Automated financial data processing
- Rule-based and AI-assisted analysis
- Red flag detection and benchmarking
- Structured reporting of company insights

## Tech Stack

- Python
- Agent-based workflow logic
- Financial data processing
- Basic RAG components

## Project Structure

- `main.py` – main execution logic
- `query.py` – querying and analysis
- `agents/` logic spread across modules (parser, reporter, validator, etc.)
- `rag/` – data processing and retrieval components
- `utils/` – helper functions and shared logic

## Current Status

The project is currently focused on improving data quality and robustness in the parsing pipeline.

During development, it became clear that inconsistent or inaccurate input data propagates through the system and undermines downstream analysis. Therefore, the current priority is to establish a reliable and validated data foundation.

The `test_html_parser.py` module is used to systematically test and validate parsing logic, ensuring consistent extraction of financial data before it enters the analysis workflow.

This reflects a broader design focus on building reliable AI systems by addressing data quality and validation early in the pipeline.

## Key Learnings

- Designing multi-step AI workflows
- Structuring real-world data pipelines
- Applying AI to financial decision processes
- Identifying where automation creates real value
