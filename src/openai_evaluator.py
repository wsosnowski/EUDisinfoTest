from src.base_evaluator import BaseEvaluator
from src.models import ModelType
import json
import logging
from openai import OpenAI


class OpenAIEvaluator(BaseEvaluator):
    def __init__(self, model_type: ModelType, api_key: str):
        super().__init__()
        self.client = OpenAI(
            api_key=api_key,
            timeout=30
        )
        self.model_name = model_type

    def evaluate(self, narrative):
        """
        Evaluates the narrative for disinformation and returns a structured response.
        """
        full_prompt = f"""
        You are an AI fact checker trained to assess whether the provided statement is disinformation.
        Statement: {narrative}
        Analyze the text. Explain your reasoning, stop, then think slowly, step-by-step.

        Provide your response in the following JSON format:
         {{
         "justification": "Your explanation here.",
          "confidence": a number between 0.0 and 1.0,
          "status": "disinformative" or "credible"
        }}
        """
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": full_prompt}],
                model=self.model_name,
            )
            res = response.choices[0].message.content.strip('```json\n').rstrip('```').strip()
            start = res.find('{')
            end = res.find('}') + 1
            json_part = res[start:end]
            json_res = json.loads(json_part)

            return json_res
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            return {"analysis": "", "status": "error"}
