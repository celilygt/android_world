# Copyright 2024 The android_world Authors.
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

"""Backward compatibility module for LLM inference.

This module re-exports all LLM wrappers from the new llm_wrappers directory
to maintain compatibility with existing code that imports from android_world.agents.infer.
"""

# Re-export everything from the new llm_wrappers module for backward compatibility
from android_world.agents.llm_wrappers.base_wrapper import (
    LlmWrapper,
    MultimodalLlmWrapper,
    ERROR_CALLING_LLM,
    array_to_jpeg_bytes,
)

from android_world.agents.llm_wrappers.gemini_gcp_wrapper import GeminiGcpWrapper
from android_world.agents.llm_wrappers.gemini_gemma_wrapper import GeminiGemmaWrapper
from android_world.agents.llm_wrappers.gpt4_wrapper import Gpt4Wrapper
from android_world.agents.llm_wrappers.ollama_wrapper import OllamaWrapper
from android_world.agents.llm_wrappers.openrouter_wrapper import OpenRouterWrapper

# For backward compatibility, also export under the old names if they were different
GeminiWrapper = GeminiGcpWrapper  # Common alias

__all__ = [
    'LlmWrapper',
    'MultimodalLlmWrapper',
    'ERROR_CALLING_LLM',
    'array_to_jpeg_bytes',
    'GeminiGcpWrapper',
    'GeminiGemmaWrapper', 
    'GeminiWrapper',  # Alias
    'Gpt4Wrapper',
    'OllamaWrapper',
    'OpenRouterWrapper',
] 