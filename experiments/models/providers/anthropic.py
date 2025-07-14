import os
from anthropic import Anthropic
from typing import Any
from ..base_provider import BaseProvider

class AnthropicProvider(BaseProvider):
    def process_single_prompt(self, prompt: str, model_name: str, max_output_tokens: int, index: int) -> tuple[int, str]:
        response = self.client.messages.create(
            model=model_name,
            temperature=0,
            max_tokens=max_output_tokens,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            thinking = {
                "type": "disabled", # default is disabled, configure to test thinking modes
            }
        )

        if response.content and len(response.content) > 0:
            return index, response.content[0].text
        else:
            return index, "ERROR_NO_CONTENT"

    def get_client(self) -> Any:
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))