# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ollama LLM API wrapper."""

import base64
import time
from typing import Any, Optional

import numpy as np
import ollama
from android_world.agents.llm_wrappers import base_wrapper


class OllamaWrapper(
    base_wrapper.LlmWrapper, base_wrapper.MultimodalLlmWrapper
):
    """
    Ollama wrapper for local LLM inference.
    
    This wrapper provides:
    - No rate limits (local inference)
    - Prompt isolation (new chat for every prompt)
    - Support for powerful models like deepseek-r1:8b
    """

    RETRY_WAITING_SECONDS = 5

    def __init__(
        self,
        model_name: str = "deepseek-r1:8b",
        max_retry: int = 3,
        temperature: float = 0.0,
        host: str = "localhost",
        port: int = 11434,
        timeout: int = 60,
    ):
        """Initialize Ollama wrapper.
        
        Args:
            model_name: The Ollama model to use (e.g., "deepseek-r1:8b").
            max_retry: Max number of retries when calling the LLM.
            temperature: Temperature for LLM generation.
            host: Ollama server host.
            port: Ollama server port.
            timeout: Request timeout in seconds.
        """
        self.model_name = model_name
        self.max_retry = min(max(1, max_retry), 5)
        self.temperature = temperature
        self.host = host
        self.port = port
        self.timeout = timeout
        
        # Initialize Ollama client
        try:
            self.client = ollama.Client(host=f"http://{host}:{port}")
            # Test connection and ensure model is available
            self._ensure_model_available()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama at {host}:{port}. "
                             f"Make sure Ollama is running. Error: {e}")

    def _ensure_model_available(self):
        """Ensure the specified model is available, pull if necessary."""
        try:
            # Check if model exists locally
            models_response = self.client.list()
            
            # Extract model names from the ListResponse
            model_names = []
            if hasattr(models_response, 'models'):
                for model in models_response.models:
                    if hasattr(model, 'model'):
                        model_names.append(model.model)
                    elif hasattr(model, 'name'):
                        model_names.append(model.name)
                    elif isinstance(model, dict):
                        model_names.append(model.get('model', model.get('name', str(model))))
                    else:
                        model_names.append(str(model))
            
            if self.model_name not in model_names:
                print(f"Model '{self.model_name}' not found locally. Available models: {model_names}")
                print(f"Attempting to pull model '{self.model_name}'...")
                self.client.pull(self.model_name)
                print(f"Successfully pulled model '{self.model_name}'")
            else:
                print(f"Model '{self.model_name}' is available locally")
                
        except Exception as e:
            # If we can't check, try to pull anyway
            print(f"Could not verify model availability: {e}")
            print(f"Attempting to pull model '{self.model_name}'...")
            try:
                self.client.pull(self.model_name)
                print(f"Successfully pulled model '{self.model_name}'")
            except Exception as pull_e:
                raise RuntimeError(f"Failed to pull model '{self.model_name}': {pull_e}")

    @classmethod
    def encode_image(cls, image: np.ndarray) -> str:
        """Encodes a numpy array image to a base64 string."""
        return base64.b64encode(base_wrapper.array_to_jpeg_bytes(image)).decode('utf-8')

    def predict(
        self,
        text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        """Predict with text-only prompt."""
        return self.predict_mm(text_prompt, [])

    def predict_mm(
        self, 
        text_prompt: str, 
        images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        """
        Predict with multimodal input (text + images).
        
        Creates a fresh chat session for each request to ensure prompt isolation.
        """
        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        
        while counter > 0:
            try:
                # Prepare messages for the chat
                messages = []
                
                if images:
                    # For multimodal input, encode images as base64
                    content = [{"type": "text", "text": text_prompt}]
                    for image in images:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.encode_image(image)}"
                            }
                        })
                    messages.append({"role": "user", "content": content})
                else:
                    # Text-only input
                    messages.append({"role": "user", "content": text_prompt})

                # Create a new chat session for prompt isolation
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options={
                        'temperature': self.temperature,
                        'num_predict': 1000,  # Max output tokens
                    },
                    stream=False,  # Get complete response atomically (no streaming)
                    keep_alive=0,  # Don't keep the session alive after this request
                )
                
                if 'message' in response and 'content' in response['message']:
                    return response['message']['content'], True, response
                else:
                    print(f"Unexpected response format from Ollama: {response}")
                    
            except ollama.ResponseError as e:
                print(f"Ollama API error: {e}")
                if "model not found" in str(e).lower():
                    print(f"Attempting to pull model '{self.model_name}'...")
                    try:
                        self.client.pull(self.model_name)
                        print(f"Successfully pulled model '{self.model_name}'. Retrying...")
                        continue  # Retry without decrementing counter
                    except Exception as pull_e:
                        print(f"Failed to pull model: {pull_e}")
                        
            except Exception as e:
                print(f"Error calling Ollama: {e}")
            
            counter -= 1
            if counter > 0:
                print(f'Retrying in {wait_seconds} seconds...')
                time.sleep(wait_seconds)
                wait_seconds *= 2

        return base_wrapper.ERROR_CALLING_LLM, None, None

    def get_available_models(self) -> list[str]:
        """Get list of available models from Ollama."""
        try:
            models_response = self.client.list()
            
            # Extract model names from the ListResponse
            model_names = []
            if hasattr(models_response, 'models'):
                for model in models_response.models:
                    if hasattr(model, 'model'):
                        model_names.append(model.model)
                    elif hasattr(model, 'name'):
                        model_names.append(model.name)
                    elif isinstance(model, dict):
                        model_names.append(model.get('model', model.get('name', str(model))))
                    else:
                        model_names.append(str(model))
            
            return model_names
        except Exception as e:
            print(f"Error getting available models: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            self.client.pull(model_name)
            print(f"Successfully pulled model '{model_name}'")
            return True
        except Exception as e:
            print(f"Failed to pull model '{model_name}': {e}")
            return False 