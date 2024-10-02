from dataclasses import dataclass

ModelType = {
    "GPT3.5": "gpt-3.5-turbo-0125",
    "GPT4o": "gpt-4o-2024-05-13",
    "Mixtral": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "L3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "L3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "Sonnet": "claude-3-sonnet-20240229",
    "Haiku": "claude-3-haiku-20240307",
    "Opus": "claude-3-opus-20240229"
}


@dataclass
class EvaluationConfig:
    model: str
    input_path: str
    output_path: str
    api_key: str
    max_trials: int
