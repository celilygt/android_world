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

"""Ollama LLM API wrapper for local models."""

import base64
import time
from typing import Any, Optional

import numpy as np
import requests
from android_world.agents.llm_wrappers import base_wrapper


class OllamaWrapper(
    base_wrapper.LlmWrapper, base_wrapper.MultimodalLlmWrapper
):
  """
  Ollama wrapper for running local models like Gemma 3-4B.
  """

  RETRY_WAITING_SECONDS = 5

  def __init__(
      self,
      model_name: str = "gemma3:4b",
      max_retry: int = 3,
      temperature: float = 0.0,
      host: str = "localhost",
      port: int = 11434,
  ):
    """Initialize the Ollama wrapper.
    
    Args:
      model_name: The Ollama model to use (e.g., "gemma3:4b").
      max_retry: Max number of retries when calling the LLM.
      temperature: Temperature for LLM generation.
      host: Ollama host (default: localhost).
      port: Ollama port (default: 11434).
    """
    self.max_retry = min(max(1, max_retry), 5)
    self.temperature = temperature
    self.model = model_name
    self.base_url = f"http://{host}:{port}"

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    """Encodes a numpy array image to a base64 string."""
    return base64.b64encode(base_wrapper.array_to_jpeg_bytes(image)).decode(
        'utf-8'
    )

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """Predict with text only."""
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Predict with text and images using Ollama's API."""
    headers = {
        'Content-Type': 'application/json',
    }

    # For multimodal models in Ollama, we include images as base64
    payload = {
        'model': self.model,
        'prompt': text_prompt,
        'stream': False,
        'options': {
            'temperature': self.temperature,
        }
    }

    # Add images if provided
    if images:
      payload['images'] = [self.encode_image(image) for image in images]

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        response = requests.post(
            f'{self.base_url}/api/generate',
            headers=headers,
            json=payload,
            timeout=120,  # Longer timeout for local inference
        )
        response.raise_for_status()
        
        response_json = response.json()
        if 'response' in response_json:
          return (
              response_json['response'],
              True,  # Assume safe for local models
              response,
          )
        err_msg = response_json.get('error', 'Unknown error')
        print(f'Error calling Ollama API: {err_msg}')
        
      except requests.exceptions.ConnectionError as e:
        print(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running? Error: {e}")
      except requests.exceptions.Timeout as e:
        print(f"Timeout calling Ollama: {e}")
      except requests.exceptions.RequestException as e:
        print(f"Request error calling Ollama: {e}")
      except Exception as e:
        print(f"An unexpected error occurred: {e}")
      
      counter -= 1
      if counter > 0:
        print(f'Retrying in {wait_seconds} seconds...')
        time.sleep(wait_seconds)
        wait_seconds *= 2

    return base_wrapper.ERROR_CALLING_LLM, None, None

  def is_model_available(self) -> bool:
    """Check if the specified model is available in Ollama."""
    try:
      response = requests.get(f'{self.base_url}/api/tags', timeout=10)
      response.raise_for_status()
      models = response.json().get('models', [])
      return any(model['name'] == self.model for model in models)
    except Exception as e:
      print(f"Error checking model availability: {e}")
      return False

  def pull_model(self) -> bool:
    """Pull the model if it's not available."""
    try:
      print(f"Pulling model {self.model}...")
      response = requests.post(
          f'{self.base_url}/api/pull',
          json={'name': self.model},
          timeout=600,  # 10 minutes for model download
      )
      response.raise_for_status()
      print(f"Model {self.model} pulled successfully.")
      return True
    except Exception as e:
      print(f"Error pulling model: {e}")
      return False 