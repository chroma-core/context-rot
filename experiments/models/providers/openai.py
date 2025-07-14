import os
from openai import OpenAI
from typing import Any
from ..base_provider import BaseProvider

class OpenAIProvider(BaseProvider):
    def process_single_prompt(self, prompt: str, model_name: str, max_output_tokens: int, index: int) -> tuple[int, str]:
        response = self.client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_completion_tokens=max_output_tokens,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        if response.choices and len(response.choices) > 0:
            if response.choices[0].message.content == "":
                print(response)
            return index, response.choices[0].message.content
        else:
            return index, "ERROR_NO_CONTENT"

    def get_client(self) -> Any:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
