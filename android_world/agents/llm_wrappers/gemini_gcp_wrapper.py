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

"""Gemini GCP LLM API wrapper."""

import os
import time
from typing import Any, Optional

import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import (
    answer_types,
    content_types,
    generation_types,
    safety_types,
)
import numpy as np
from PIL import Image

from android_world.agents.llm_wrappers import base_wrapper
from android_world.agents.llm_wrappers.gemini_gemma_wrapper import (
    SAFETY_SETTINGS_BLOCK_NONE,
)


class GeminiGcpWrapper(
    base_wrapper.LlmWrapper, base_wrapper.MultimodalLlmWrapper
):
  """
  Gemini GCP interface.
  """

  RETRY_WAITING_SECONDS = 1.0

  def __init__(
      self,
      model_name: str = 'gemini-1.5-pro-latest',
      max_retry: int = 3,
      temperature: float = 0.0,
      top_p: float = 0.95,
      enable_safety_checks: bool = True,
  ):
    if 'GCP_API_KEY' not in os.environ:
      raise RuntimeError('GCP_API_KEY environment variable not set.')
    genai.configure(api_key=os.environ['GCP_API_KEY'])

    self.llm = genai.GenerativeModel(
        model_name,
        safety_settings=(
            None if enable_safety_checks else SAFETY_SETTINGS_BLOCK_NONE
        ),
        generation_config=generation_types.GenerationConfig(
            temperature=temperature, top_p=top_p, max_output_tokens=1000
        ),
    )
    self.max_retry = min(max(1, max_retry), 5)
    self.enable_safety_checks = enable_safety_checks

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def is_safe(self, raw_response) -> bool:
    """Checks if the response from the LLM is safe."""
    try:
      return (
          raw_response.candidates[0].finish_reason
          != answer_types.FinishReason.SAFETY
      )
    except (IndexError, AttributeError):
      return True

  def predict_mm(
      self,
      text_prompt: str,
      images: list[np.ndarray],
  ) -> tuple[str, Optional[bool], Any]:
    return self.generate(
        [text_prompt] + [Image.fromarray(image) for image in images]
    )

  def generate(
      self,
      contents: content_types.ContentsType,
  ) -> tuple[str, Any]:
    """Exposes the generate_content API."""
    counter = self.max_retry
    retry_delay = self.RETRY_WAITING_SECONDS
    
    while counter > 0:
      try:
        response = self.llm.generate_content(
            contents=contents,
            safety_settings=(
                None if self.enable_safety_checks else SAFETY_SETTINGS_BLOCK_NONE
            ),
        )
        if self.is_safe(response):
          return response.text, response
        else:
          return base_wrapper.ERROR_CALLING_LLM, response
      except Exception as e:
        counter -= 1
        print(f"Error calling Gemini GCP: {e}. Retrying in {retry_delay}s...")
        if counter > 0:
          time.sleep(retry_delay)
          retry_delay *= 2
          
    raise RuntimeError(f'Error calling LLM after {self.max_retry} retries.') 