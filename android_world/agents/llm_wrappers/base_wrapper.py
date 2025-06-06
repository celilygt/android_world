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

"""Base classes and utilities for LLM wrappers."""

import abc
import io
from typing import Any, Optional

import numpy as np
from PIL import Image

ERROR_CALLING_LLM = 'Error calling LLM'


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
  """Converts a numpy array into a byte string for a JPEG image."""
  image = Image.fromarray(image)
  return image_to_jpeg_bytes(image)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
  """Converts a PIL image into a byte string for a JPEG image."""
  in_mem_file = io.BytesIO()
  image.save(in_mem_file, format='JPEG')
  in_mem_file.seek(0)
  return in_mem_file.read()


class LlmWrapper(abc.ABC):
  """Abstract interface for (text only) LLM."""

  @abc.abstractmethod
  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """
    Calls a text-based LLM with a prompt.

    Args:
      text_prompt: The text prompt.

    Returns:
      A tuple containing the text output, an optional safety flag, and the raw
      response from the API.
    """


class MultimodalLlmWrapper(abc.ABC):
  """Abstract interface for Multimodal LLM."""

  @abc.abstractmethod
  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """
    Calls a multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: The text prompt.
      images: A list of images as numpy ndarrays.

    Returns:
      A tuple containing the text output, an optional safety flag, and the raw
      response from the API.
    """ 