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

"""Gemini API wrapper for Gemma models with smart routing and rate limiting."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Model configurations with rate limits (RPM = Requests Per Minute, TPM = Tokens Per Minute, RPD = Requests Per Day)
GEMINI_MODEL_CONFIGS = {
    "gemini-2.5-flash-lite-preview-0617": {
        "rpm_limit": 15,
        "tpm_limit": 250000,
        "rpd_limit": 1000,
        "priority": 1,  # Highest priority due to best RPD limit
        "description": "Best for high-volume requests"
    },
    "gemini-2.0-flash": {
        "rpm_limit": 15,
        "tpm_limit": 1000000,
        "rpd_limit": 200,
        "priority": 2,
        "description": "High token limit, good for complex prompts"
    },
    "gemini-2.0-flash-lite": {
        "rpm_limit": 30,
        "tpm_limit": 1000000,
        "rpd_limit": 200,
        "priority": 3,
        "description": "Highest RPM, good for quick requests"
    },
    "gemini-2.5-flash": {
        "rpm_limit": 10,
        "tpm_limit": 250000,
        "rpd_limit": 250,
        "priority": 4,
        "description": "Standard flash model"
    },
    "gemini-2.5-pro": {
        "rpm_limit": 5,
        "tpm_limit": 250000,
        "rpd_limit": 100,
        "priority": 5,
        "description": "Pro model with lower limits"
    },
}


class GeminiUsageTracker:
    """Tracks API usage across sessions with persistent storage."""
    
    def __init__(self, storage_file: str = "gemini_usage.json"):
        self.storage_file = Path.home() / ".android_world" / storage_file
        self.storage_file.parent.mkdir(exist_ok=True)
        self.usage_data = self._load_usage_data()
        
    def _load_usage_data(self) -> Dict:
        """Load usage data from persistent storage."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    # Clean old data (older than 1 day)
                    self._clean_old_data(data)
                    return data
            except (json.JSONDecodeError, KeyError):
                print(f"Warning: Corrupted usage file {self.storage_file}, starting fresh")
        
        return {"daily_usage": {}, "minute_usage": {}}
    
    def _clean_old_data(self, data: Dict) -> None:
        """Remove usage data older than 1 day."""
        current_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        current_minute = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M")
        
        # Keep only today's daily usage
        if "daily_usage" in data:
            data["daily_usage"] = {
                model: usage for model, usage in data["daily_usage"].items()
                if usage.get("date") == current_day
            }
        
        # Keep only last 2 minutes of minute usage
        if "minute_usage" in data:
            for model in list(data["minute_usage"].keys()):
                if model in data["minute_usage"]:
                    data["minute_usage"][model] = {
                        minute: usage for minute, usage in data["minute_usage"][model].items()
                        if minute >= current_minute[-2:]  # Keep last 2 minutes
                    }
    
    def _save_usage_data(self) -> None:
        """Save usage data to persistent storage."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save usage data: {e}")
    
    def get_daily_usage(self, model_name: str) -> Tuple[int, int]:
        """Get today's usage for a model (requests, tokens)."""
        current_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        if model_name not in self.usage_data["daily_usage"]:
            return 0, 0
            
        usage = self.usage_data["daily_usage"][model_name]
        if usage.get("date") != current_day:
            return 0, 0
            
        return usage.get("requests", 0), usage.get("tokens", 0)
    
    def get_minute_usage(self, model_name: str) -> Tuple[int, int]:
        """Get current minute's usage for a model (requests, tokens)."""
        current_minute = datetime.now(timezone.utc).strftime("%H-%M")
        
        if (model_name not in self.usage_data["minute_usage"] or 
            current_minute not in self.usage_data["minute_usage"][model_name]):
            return 0, 0
            
        usage = self.usage_data["minute_usage"][model_name][current_minute]
        return usage.get("requests", 0), usage.get("tokens", 0)
    
    def record_usage(self, model_name: str, tokens: int) -> None:
        """Record usage for a model."""
        current_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        current_minute = datetime.now(timezone.utc).strftime("%H-%M")
        
        # Initialize structures if needed
        if "daily_usage" not in self.usage_data:
            self.usage_data["daily_usage"] = {}
        if "minute_usage" not in self.usage_data:
            self.usage_data["minute_usage"] = {}
            
        # Record daily usage
        if model_name not in self.usage_data["daily_usage"]:
            self.usage_data["daily_usage"][model_name] = {"date": current_day, "requests": 0, "tokens": 0}
        elif self.usage_data["daily_usage"][model_name].get("date") != current_day:
            self.usage_data["daily_usage"][model_name] = {"date": current_day, "requests": 0, "tokens": 0}
            
        self.usage_data["daily_usage"][model_name]["requests"] += 1
        self.usage_data["daily_usage"][model_name]["tokens"] += tokens
        
        # Record minute usage
        if model_name not in self.usage_data["minute_usage"]:
            self.usage_data["minute_usage"][model_name] = {}
        if current_minute not in self.usage_data["minute_usage"][model_name]:
            self.usage_data["minute_usage"][model_name][current_minute] = {"requests": 0, "tokens": 0}
            
        self.usage_data["minute_usage"][model_name][current_minute]["requests"] += 1
        self.usage_data["minute_usage"][model_name][current_minute]["tokens"] += tokens
        
        self._save_usage_data()


class GeminiModelRouter:
    """Smart router that selects the best available Gemini model based on usage and limits."""
    
    def __init__(self, usage_tracker: GeminiUsageTracker):
        self.usage_tracker = usage_tracker
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text. Simple approximation: ~4 chars per token."""
        return max(1, len(text) // 4)
    
    def can_handle_request(self, model_name: str, estimated_tokens: int) -> Tuple[bool, str]:
        """Check if a model can handle the request based on current usage."""
        config = GEMINI_MODEL_CONFIGS[model_name]
        
        # Check daily limits
        daily_requests, daily_tokens = self.usage_tracker.get_daily_usage(model_name)
        if daily_requests >= config["rpd_limit"]:
            return False, f"Daily request limit reached ({daily_requests}/{config['rpd_limit']})"
        if daily_tokens + estimated_tokens > config["tpm_limit"] * 24 * 60:  # Rough daily token estimate
            return False, f"Daily token limit would be exceeded"
            
        # Check minute limits
        minute_requests, minute_tokens = self.usage_tracker.get_minute_usage(model_name)
        if minute_requests >= config["rpm_limit"]:
            return False, f"Minute request limit reached ({minute_requests}/{config['rpm_limit']})"
        if minute_tokens + estimated_tokens > config["tpm_limit"]:
            return False, f"Minute token limit would be exceeded"
            
        return True, "Available"
    
    def select_best_model(self, prompt: str, max_wait_seconds: int = 60) -> Optional[str]:
        """Select the best available model for the given prompt.
        
        Args:
            prompt: The text prompt to process
            max_wait_seconds: Maximum seconds to wait for minute limits to reset
            
        Returns:
            Selected model name or None if no model available after waiting
        """
        estimated_tokens = self.estimate_tokens(prompt)
        
        # Sort models by priority (lower number = higher priority)
        sorted_models = sorted(GEMINI_MODEL_CONFIGS.items(), key=lambda x: x[1]["priority"])
        
        # First, try to find an immediately available model
        available_models = []
        minute_limited_models = []
        daily_limited_models = []
        
        for model_name, config in sorted_models:
            can_handle, reason = self.can_handle_request(model_name, estimated_tokens)
            if can_handle:
                available_models.append((model_name, config))
            elif "minute" in reason.lower() or "rpm" in reason.lower() or "tpm" in reason.lower():
                minute_limited_models.append((model_name, config))
                print(f"Model {model_name} minute-limited: {reason}")
            else:
                daily_limited_models.append((model_name, config))
                print(f"Model {model_name} daily-limited: {reason}")
        
        # If we have available models, use the best one
        if available_models:
            selected_model = available_models[0][0]
            print(f"✅ Selected model: {selected_model} (estimated tokens: {estimated_tokens})")
            return selected_model
        
        # If no models available but some are just minute-limited, wait for next minute
        if minute_limited_models and not daily_limited_models:
            current_time = datetime.now(timezone.utc)
            seconds_to_next_minute = 60 - current_time.second
            
            if seconds_to_next_minute <= max_wait_seconds:
                print(f"⏳ All models minute-limited. Waiting {seconds_to_next_minute}s for next minute...")
                time.sleep(seconds_to_next_minute + 1)  # +1 second buffer
                
                # Try again after waiting
                return self.select_best_model(prompt, max_wait_seconds=0)  # Don't wait again
        
        # If we have minute-limited models but some daily limits prevent waiting
        if minute_limited_models:
            # Check if any minute-limited models might have daily capacity
            for model_name, config in minute_limited_models:
                daily_requests, daily_tokens = self.usage_tracker.get_daily_usage(model_name)
                if daily_requests < config["rpd_limit"]:
                    print(f"⚠️ Fallback to minute-limited model: {model_name} (may hit rate limits)")
                    return model_name
        
        # Last resort - use first model even if limited
        if sorted_models:
            fallback_model = sorted_models[0][0]
            print(f"⚠️ Warning: All models limited, falling back to {fallback_model}")
            return fallback_model
            
        print("❌ Error: No models configured")
        return None
    
    def get_usage_summary(self) -> str:
        """Get a summary of current usage across all models."""
        summary_lines = ["=== Gemini API Usage Summary ==="]
        
        for model_name, config in GEMINI_MODEL_CONFIGS.items():
            daily_requests, daily_tokens = self.usage_tracker.get_daily_usage(model_name)
            minute_requests, minute_tokens = self.usage_tracker.get_minute_usage(model_name)
            
            summary_lines.append(f"\n{model_name}:")
            summary_lines.append(f"  Daily: {daily_requests}/{config['rpd_limit']} requests, {daily_tokens:,} tokens")
            summary_lines.append(f"  Minute: {minute_requests}/{config['rpm_limit']} requests, {minute_tokens:,}/{config['tpm_limit']:,} tokens")
            
        return "\n".join(summary_lines)


class GeminiGemmaWrapper(
    base_wrapper.LlmWrapper, base_wrapper.MultimodalLlmWrapper
):
    """
    Gemini API wrapper with smart model routing and rate limiting.
    """

    RETRY_WAITING_SECONDS = 20

    def __init__(
        self,
        model_name: str = "auto",  # "auto" for smart routing
        max_retry: int = 3,
        temperature: float = 0.0,
        top_p: float = 0.95,
        enable_safety_checks: bool = True,
        verbose: bool = False,
    ):
        if 'GEMINI_API_KEY' not in os.environ:
            raise RuntimeError('GEMINI_API_KEY environment variable not set.')
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        
        self.max_retry = min(max(1, max_retry), 5)
        self.enable_safety_checks = enable_safety_checks
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose
        
        # Initialize smart routing components
        self.usage_tracker = GeminiUsageTracker()
        self.router = GeminiModelRouter(self.usage_tracker)
        self.fixed_model = model_name if model_name != "auto" else None
        
        if self.verbose:
            print(self.router.get_usage_summary())

    def _create_model(self, model_name: str) -> genai.GenerativeModel:
        """Create a GenerativeModel instance for the specified model."""
        return genai.GenerativeModel(
            model_name,
            safety_settings=(
                None if self.enable_safety_checks else SAFETY_SETTINGS_BLOCK_NONE
            ),
            generation_config=generation_types.GenerationConfig(
                temperature=self.temperature, 
                top_p=self.top_p, 
                max_output_tokens=1000
            ),
        )

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
        
        # Select the best model for this request
        if self.fixed_model:
            selected_model = self.fixed_model
            if self.verbose:
                print(f"Using fixed model: {selected_model}")
        else:
            selected_model = self.router.select_best_model(text_prompt, max_wait_seconds=60)
            if not selected_model or selected_model not in GEMINI_MODEL_CONFIGS:
                print(f"Warning: Invalid model selection, using fallback")
                selected_model = "gemini-2.5-flash"
        
        # Estimate tokens for usage tracking
        estimated_tokens = self.router.estimate_tokens(text_prompt)
        
        counter = self.max_retry
        retry_delay = self.RETRY_WAITING_SECONDS
        
        # Prepare content for the API call
        content = [text_prompt] + [Image.fromarray(image) for image in images]
        
        while counter > 0:
            try:
                # Create model instance for the selected model
                llm = self._create_model(selected_model)
                
                start_time = time.time()
                output = llm.generate_content(
                    content,
                    safety_settings=(
                        None if self.enable_safety_checks else SAFETY_SETTINGS_BLOCK_NONE
                    ),
                )
                
                if self.is_safe(output):
                    # Record successful usage
                    self.usage_tracker.record_usage(selected_model, estimated_tokens)
                    
                    if self.verbose:
                        elapsed = time.time() - start_time
                        print(f"✅ Request completed in {elapsed:.2f}s using {selected_model}")
                        print(f"📊 Response length: {len(output.text)} chars")
                    
                    return output.text, True, output
                else:
                    return base_wrapper.ERROR_CALLING_LLM, False, output
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle rate limiting specifically
                if "quota" in error_msg or "rate" in error_msg or "limit" in error_msg:
                    if not self.fixed_model:
                        # Try a different model if using auto-routing
                        print(f"Rate limit hit on {selected_model}, trying different model...")
                        # Remove this model from consideration temporarily by marking high usage
                        self.usage_tracker.record_usage(selected_model, 999999)  # Fake high usage
                        selected_model = self.router.select_best_model(text_prompt, max_wait_seconds=0)
                        continue
                
                counter -= 1
                print(f"Error calling Gemini LLM ({selected_model}): {e}. Retrying in {retry_delay}s...")
                if counter > 0:
                    time.sleep(retry_delay)
                    retry_delay *= 2

        return base_wrapper.ERROR_CALLING_LLM, None, None
    
    def get_usage_summary(self) -> str:
        """Get current usage summary."""
        return self.router.get_usage_summary()
    
    def reset_usage(self) -> None:
        """Reset usage tracking (useful for testing)."""
        if self.usage_tracker.storage_file.exists():
            self.usage_tracker.storage_file.unlink()
        self.usage_tracker = GeminiUsageTracker()
        self.router = GeminiModelRouter(self.usage_tracker)
        print("Usage tracking reset.") 