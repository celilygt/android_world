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

"""OpenRouter LLM API wrapper."""

import base64
import os
import time
from typing import Any, Optional

import numpy as np
import requests
from android_world.agents.llm_wrappers import base_wrapper


class OpenRouterWrapper(
    base_wrapper.LlmWrapper, base_wrapper.MultimodalLlmWrapper
):
  """
  OpenRouter AI wrapper for various open-source models.
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str = "google/gemma-3-27b-it:free",
      max_retry: int = 3,
      temperature: float = 0.0,
      site_url: str = None,
      site_name: str = None,
  ):
    if 'OPENROUTER_API_KEY' not in os.environ:
      raise RuntimeError('OPENROUTER_API_KEY environment variable not set.')
    self.openrouter_api_key = os.environ['OPENROUTER_API_KEY']
    self.max_retry = min(max(1, max_retry), 5)
    self.temperature = temperature
    self.model = model_name
    self.site_url = site_url or "https://github.com/google-research/android_world"
    self.site_name = site_name or "AndroidWorld"

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
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self.openrouter_api_key}',
        'HTTP-Referer': self.site_url,
        'X-Title': self.site_name,
    }

    content = [{'type': 'text', 'text': text_prompt}]
    for image in images:
      content.append({
          'type': 'image_url',
          'image_url': {'url': f'data:image/jpeg;base64,{self.encode_image(image)}'},
      })

    payload = {
        'model': self.model,
        'temperature': self.temperature,
        'messages': [{'role': 'user', 'content': content}],
        'max_tokens': 1000,
    }

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers,
            json=payload,
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        
        response_json = response.json()
        if 'choices' in response_json:
          return (
              response_json['choices'][0]['message']['content'],
              True,  # Assume safe, as OpenRouter doesn't classify
              response,
          )
        err_msg = response_json.get('error', {}).get('message', 'Unknown error')
        print(f'Error calling OpenRouter API: {err_msg}')
        
      except requests.exceptions.RequestException as e:
        print(f"Request error calling OpenRouter: {e}")
      except Exception as e:
        print(f"An unexpected error occurred: {e}")
      
      counter -= 1
      if counter > 0:
        print(f'Retrying in {wait_seconds} seconds...')
        time.sleep(wait_seconds)
        wait_seconds *= 2

    return base_wrapper.ERROR_CALLING_LLM, None, None 