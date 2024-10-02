
# EU DisinfoTest: a Benchmark for Evaluating Language Models’ Ability to Detect Disinformation Narratives

## Overview

This repository contains code for evaluating model performance on the EU DisinfoTest benchmark. The dataset for that benchamrk is contained in the `data` directory, file `EUDisinfoTest.csv`.

## Requirements

- Python 3.9+
- Pandas
- Numpy
- anthropic
- openai
- scikit-learn


## Installation

1. Ensure that Python 3.9 or higher is installed on your machine.
2. Clone this repository to your local machine.
3. Navigate to the project directory and install required Python libraries:
   ```bash
   pip install -r requirements.txt 
   ```

## Usage

1. Use the command-line interface to run the script. Here's how you can execute the script with default parameters:
   ```bash
   python run_evaluator.py
   ```
2. To specify parameters like model type, input path, output path, and API key, use the respective command-line arguments:
   ```bash
   python run_evaluator.py --model GPT3.5 --input_path data/EUDisinfoTest.csv --output_path your_result_file.csv --api_key YourAPIKey
   ```

### Arguments

- `--model`: Model type (default is `L3-8b`). Supported models include `L3-8b`, `L3-70b`, `Mixtral`, `GPT3.5`, `GPT4o`, `Sonnet`, `Haiku`, and `Opus`.
- `--input_path`: Path to the input CSV file (default is `data/EUDisinfoTest.csv`).
- `--output_path`: Path where the output will be saved (default is `evaluation_results_L3_8b.csv`).
- `--api_key`: API key for accessing the model evaluation services. The key you use depends on the model chosen:
  - **DeepInfra API key** for Llama models and Mixtral.
  - **Anthropic API key** for all Claude models.
  - **OpenAI API key** for OpenAI models.
  The default value is a placeholder.

## Output

The script outputs a JSON file containing aggregate metrics for the overall evaluation and for each rhetorical category. This file is saved to the location specified by the `--output_path` argument.

