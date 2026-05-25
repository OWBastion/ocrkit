# AGENTS.md

## Project Overview

This project is a specialized OCR and structured data extraction service for Overwatch custom challenge screenshots.

The goal is not to perform generic full-image OCR. The goal is to extract specific HUD fields from known screenshot layouts, including:

- Top-left challenge progress and statistics
- Center completion banner
- Top-right map, difficulty, and version information
- Optional bottom-left hero/status information

The system should return structured JSON suitable for use by a Web API, automation pipeline, or leaderboard service.

## Core Principle

Prefer deterministic image processing, ROI cropping, field parsing, and validation over model training.

Do not introduce custom model training unless the existing OCR + preprocessing + parsing pipeline has been measured and shown to be insufficient.

The expected processing pipeline is:

```text
image upload
→ normalize image size
→ crop configured ROIs
→ preprocess each ROI
→ OCR each ROI
→ parse fields with tolerant rules
→ normalize and validate fields
→ return structured JSON

## Non-Goals

Do not build a generic OCR system.
Do not OCR the entire screenshot and then attempt to infer fields from all detected text.
Do not use Core ML.
Do not design the system around Apple-only deployment.
Do not hardcode a single screenshot's values into the parser.
Do not use large multimodal LLMs for the production recognition path unless explicitly requested.

## Code Style

Write simple, explicit Python.
Prefer small pure functions for parsers and normalizers.
Avoid hidden global state.
Avoid hardcoded ROI coordinates in parser code.
Avoid mixing OCR logic, image preprocessing, and field parsing in a single function.
Use type hints for public functions.
Use Pydantic models for API input and output schemas.