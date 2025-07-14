from google.cloud import aiplatform_v1
from google.cloud.aiplatform_v1.types import (
    GenerateContentRequest,
    GenerationConfig,
    Content,
    Part
)
import os
from typing import Any
from ..base_provider import BaseProvider

class GoogleProvider(BaseProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_path = self.get_model_path(model_name)
        super().__init__()

    def process_single_prompt(self, prompt: str, model_name: str, max_output_tokens: int, index: int) -> tuple[int, str]:
        gen_config = GenerationConfig( # modify this according to the model and thinking budget you want to use
            temperature=0,
            thinking_config=GenerationConfig.ThinkingConfig(
                thinking_budget=0
            ),
            max_output_tokens=max_output_tokens
        )

        request = GenerateContentRequest(
            model=self.model_path,
            contents=[
                Content(
                    role="user",
                    parts=[Part(text=prompt)]
                )
            ],
            generation_config=gen_config
        )
        
        response = self.client.generate_content(request=request)
        
        if response.candidates and len(response.candidates) > 0:
            if response.candidates[0].content.parts[0].text == "":
                print(response)
            return index, response.candidates[0].content.parts[0].text
        else:
            return index, "ERROR_NO_CONTENT"
            

    def get_client(self) -> Any:
        return aiplatform_v1.PredictionServiceClient()
        # this requires GOOGLE_APPLICATION_CREDENTIALS in your environment: export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
    
    def get_model_path(self, model_name: str) -> str:
        PATH = os.getenv("GOOGLE_MODEL_PATH")
        return PATH + model_name