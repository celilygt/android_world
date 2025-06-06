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

"""Gemini API wrapper for Gemma models."""

import os
import time
from typing import Any, Optional

import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import answer_types, generation_types
import numpy as np
from PIL import Image

from android_world.agents.llm_wrappers import base_wrapper


SAFETY_SETTINGS_BLOCK_NONE = {
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: types.HarmBlockThreshold.BLOCK_NONE,
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: types.HarmBlockThreshold.BLOCK_NONE,
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: types.HarmBlockThreshold.BLOCK_NONE,
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: types.HarmBlockThreshold.BLOCK_NONE,
}


class GeminiGemmaWrapper(
    base_wrapper.LlmWrapper, base_wrapper.MultimodalLlmWrapper
):
  """
  Gemini API wrapper for free Gemma models.
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str = "gemma-3-27b",
      max_retry: int = 3,
      temperature: float = 0.0,
      top_p: float = 0.95,
      enable_safety_checks: bool = True,
  ):
    if 'GEMINI_API_KEY' not in os.environ:
      raise RuntimeError('GEMINI_API_KEY environment variable not set.')
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    
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
      # Assume safe if response is malformed or lacks safety attributes.
      return True

  def predict_mm(
      self,
      text_prompt: str,
      images: list[np.ndarray],
  ) -> tuple[str, Optional[bool], Any]:
    counter = self.max_retry
    retry_delay = self.RETRY_WAITING_SECONDS
    
    # Prepare content for the API call
    content = [text_prompt] + [Image.fromarray(image) for image in images]
    
    while counter > 0:
      try:
        output = self.llm.generate_content(
            content,
            safety_settings=(
                None if self.enable_safety_checks else SAFETY_SETTINGS_BLOCK_NONE
            ),
        )
        if self.is_safe(output):
            return output.text, True, output
        else:
            return base_wrapper.ERROR_CALLING_LLM, False, output
      except Exception as e:
        counter -= 1
        print(f"Error calling Gemini LLM: {e}. Retrying in {retry_delay}s...")
        if counter > 0:
          time.sleep(retry_delay)
          retry_delay *= 2

    return base_wrapper.ERROR_CALLING_LLM, None, None 