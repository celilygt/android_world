# Copyright 2024 The V-Droid+ Authors.
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

"""Qwen2.5 VL wrapper using OpenRouter (FREE) with smart rate limiting."""

import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
from android_world.agents.llm_wrappers import base_wrapper


# OpenRouter model configurations with rate limits
OPENROUTER_MODEL_CONFIGS = {
    "qwen/qwen2.5-vl-72b-instruct:free": {
        "rpm_limit": 20,  # Free models: 20 requests per minute
        "rpd_limit_low_credits": 50,    # <10 credits: 50 requests per day
        "rpd_limit_high_credits": 1000, # 10+ credits: 1000 requests per day  
        "priority": 1,  # Highest priority (our main model)
        "is_free": True,
        "description": "Free Qwen2.5 VL 72B - excellent for UI analysis"
    },
    "qwen/qwen2.5-vl-7b-instruct:free": {
        "rpm_limit": 20,
        "rpd_limit_low_credits": 50,
        "rpd_limit_high_credits": 1000,
        "priority": 2,  # Fallback option
        "is_free": True,
        "description": "Free Qwen2.5 VL 7B - smaller but still capable"
    },
    # Non-free fallbacks (in case free models are exhausted)
    "qwen/qwen2.5-vl-72b-instruct": {
        "rpm_limit": 60,  # Paid models typically have higher limits
        "rpd_limit_low_credits": 10000,  # Generous for paid
        "rpd_limit_high_credits": 10000,
        "priority": 10,  # Very low priority (only if free models exhausted)
        "is_free": False,
        "description": "Paid Qwen2.5 VL 72B - use only if free exhausted"
    },
}


class OpenRouterUsageTracker:
    """Tracks OpenRouter API usage across sessions with persistent storage."""
    
    def __init__(self, storage_file: str = "openrouter_usage.json"):
        self.storage_file = Path.home() / ".android_world" / storage_file
        self.storage_file.parent.mkdir(exist_ok=True)
        self.usage_data = self._load_usage_data()
        self.has_high_credits = self._detect_credit_status()
        
    def _load_usage_data(self) -> Dict:
        """Load usage data from persistent storage."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    self._clean_old_data(data)
                    return data
            except (json.JSONDecodeError, KeyError):
                print(f"Warning: Corrupted usage file {self.storage_file}, starting fresh")
        
        return {"daily_usage": {}, "minute_usage": {}, "credit_status": "unknown"}
    
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
    
    def _detect_credit_status(self) -> bool:
        """Try to detect if user has 10+ credits. Default to conservative assumption."""
        stored_status = self.usage_data.get("credit_status", "unknown")
        
        if stored_status == "high":
            return True
        elif stored_status == "low":
            return False
        else:
            # Conservative assumption: assume low credits until proven otherwise
            print("💰 Credit status unknown - assuming <10 credits (50 daily limit)")
            print("   If you have 10+ credits, the limit will auto-adjust after first 402 error")
            return False
    
    def update_credit_status(self, has_high_credits: bool) -> None:
        """Update credit status based on API responses."""
        self.has_high_credits = has_high_credits
        self.usage_data["credit_status"] = "high" if has_high_credits else "low"
        self._save_usage_data()
        
        limit = 1000 if has_high_credits else 50
        status = "10+ credits" if has_high_credits else "<10 credits"
        print(f"💰 Updated credit status: {status} (daily limit: {limit})")
    
    def _save_usage_data(self) -> None:
        """Save usage data to persistent storage."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save usage data: {e}")
    
    def get_daily_usage(self, model_name: str) -> int:
        """Get today's request count for a model."""
        current_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        if model_name not in self.usage_data["daily_usage"]:
            return 0
            
        usage = self.usage_data["daily_usage"][model_name]
        if usage.get("date") != current_day:
            return 0
            
        return usage.get("requests", 0)
    
    def get_minute_usage(self, model_name: str) -> int:
        """Get current minute's request count for a model."""
        current_minute = datetime.now(timezone.utc).strftime("%H-%M")
        
        if (model_name not in self.usage_data["minute_usage"] or 
            current_minute not in self.usage_data["minute_usage"][model_name]):
            return 0
            
        return self.usage_data["minute_usage"][model_name][current_minute].get("requests", 0)
    
    def record_usage(self, model_name: str) -> None:
        """Record a request for a model."""
        current_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        current_minute = datetime.now(timezone.utc).strftime("%H-%M")
        
        # Initialize structures if needed
        if "daily_usage" not in self.usage_data:
            self.usage_data["daily_usage"] = {}
        if "minute_usage" not in self.usage_data:
            self.usage_data["minute_usage"] = {}
            
        # Record daily usage
        if model_name not in self.usage_data["daily_usage"]:
            self.usage_data["daily_usage"][model_name] = {"date": current_day, "requests": 0}
        elif self.usage_data["daily_usage"][model_name].get("date") != current_day:
            self.usage_data["daily_usage"][model_name] = {"date": current_day, "requests": 0}
            
        self.usage_data["daily_usage"][model_name]["requests"] += 1
        
        # Record minute usage
        if model_name not in self.usage_data["minute_usage"]:
            self.usage_data["minute_usage"][model_name] = {}
        if current_minute not in self.usage_data["minute_usage"][model_name]:
            self.usage_data["minute_usage"][model_name][current_minute] = {"requests": 0}
            
        self.usage_data["minute_usage"][model_name][current_minute]["requests"] += 1
        
        self._save_usage_data()


class OpenRouterModelRouter:
    """Smart router that selects the best available OpenRouter model based on usage and limits."""
    
    def __init__(self, usage_tracker: OpenRouterUsageTracker):
        self.usage_tracker = usage_tracker
        
    def can_handle_request(self, model_name: str) -> Tuple[bool, str]:
        """Check if a model can handle the request based on current usage."""
        config = OPENROUTER_MODEL_CONFIGS[model_name]
        
        # Determine daily limit based on credit status
        if config["is_free"]:
            daily_limit = (config["rpd_limit_high_credits"] if self.usage_tracker.has_high_credits 
                         else config["rpd_limit_low_credits"])
        else:
            daily_limit = config["rpd_limit_high_credits"]  # Paid models use high limit
        
        # Check daily limits
        daily_requests = self.usage_tracker.get_daily_usage(model_name)
        if daily_requests >= daily_limit:
            return False, f"Daily request limit reached ({daily_requests}/{daily_limit})"
            
        # Check minute limits
        minute_requests = self.usage_tracker.get_minute_usage(model_name)
        if minute_requests >= config["rpm_limit"]:
            return False, f"Minute request limit reached ({minute_requests}/{config['rpm_limit']})"
            
        return True, "Available"
    
    def select_best_model(self, preferred_model: str = None, max_wait_seconds: int = 60) -> Optional[str]:
        """Select the best available model.
        
        Args:
            preferred_model: Preferred model name (if None, uses priority order)
            max_wait_seconds: Maximum seconds to wait for minute limits to reset
            
        Returns:
            Selected model name or None if no model available after waiting
        """
        # If specific model requested, try it first
        models_to_try = []
        if preferred_model and preferred_model in OPENROUTER_MODEL_CONFIGS:
            models_to_try.append((preferred_model, OPENROUTER_MODEL_CONFIGS[preferred_model]))
        
        # Add other models sorted by priority (free models first)
        other_models = sorted(
            [(name, config) for name, config in OPENROUTER_MODEL_CONFIGS.items() 
             if name != preferred_model],
            key=lambda x: (not x[1]["is_free"], x[1]["priority"])  # Free models first, then by priority
        )
        models_to_try.extend(other_models)
        
        # First pass: find immediately available models
        available_models = []
        minute_limited_models = []
        daily_limited_models = []
        
        for model_name, config in models_to_try:
            can_handle, reason = self.can_handle_request(model_name)
            if can_handle:
                available_models.append((model_name, config))
            elif "minute" in reason.lower():
                minute_limited_models.append((model_name, config))
                print(f"Model {model_name} minute-limited: {reason}")
            else:
                daily_limited_models.append((model_name, config))
                if config["is_free"]:  # Only warn about free model daily limits
                    print(f"Model {model_name} daily-limited: {reason}")
        
        # If we have available models, use the best one
        if available_models:
            selected_model = available_models[0][0]
            model_type = "FREE" if available_models[0][1]["is_free"] else "PAID"
            print(f"✅ Selected {model_type} model: {selected_model}")
            return selected_model
        
        # If only minute-limited models available, wait for next minute
        if minute_limited_models and not daily_limited_models:
            current_time = datetime.now(timezone.utc)
            seconds_to_next_minute = 60 - current_time.second
            
            if seconds_to_next_minute <= max_wait_seconds:
                print(f"⏳ All models minute-limited. Waiting {seconds_to_next_minute}s for next minute...")
                time.sleep(seconds_to_next_minute + 1)  # +1 second buffer
                
                # Try again after waiting
                return self.select_best_model(preferred_model, max_wait_seconds=0)
        
        # If we have minute-limited models with daily capacity, try them as fallback
        if minute_limited_models:
            fallback_model = minute_limited_models[0][0]
            print(f"⚠️ Fallback to minute-limited model: {fallback_model} (may hit rate limits)")
            return fallback_model
            
        # Last resort - return the preferred model even if limited
        if preferred_model:
            print(f"⚠️ Warning: Using requested model {preferred_model} despite limits")
            return preferred_model
        elif models_to_try:
            fallback = models_to_try[0][0]
            print(f"⚠️ Warning: All models limited, falling back to {fallback}")
            return fallback
            
        print("❌ Error: No models configured")
        return None
    
    def get_usage_summary(self) -> str:
        """Get a summary of current usage across all models."""
        summary_lines = ["=== OpenRouter API Usage Summary ==="]
        
        for model_name, config in OPENROUTER_MODEL_CONFIGS.items():
            daily_requests = self.usage_tracker.get_daily_usage(model_name)
            minute_requests = self.usage_tracker.get_minute_usage(model_name)
            
            # Determine daily limit
            if config["is_free"]:
                daily_limit = (config["rpd_limit_high_credits"] if self.usage_tracker.has_high_credits 
                             else config["rpd_limit_low_credits"])
            else:
                daily_limit = config["rpd_limit_high_credits"]
            
            model_type = "FREE" if config["is_free"] else "PAID"
            summary_lines.append(f"\n{model_name} ({model_type}):")
            summary_lines.append(f"  Daily: {daily_requests}/{daily_limit} requests")
            summary_lines.append(f"  Minute: {minute_requests}/{config['rpm_limit']} requests")
            
        credits_status = "10+ credits" if self.usage_tracker.has_high_credits else "<10 credits"
        summary_lines.append(f"\nCredit Status: {credits_status}")
        
        return "\n".join(summary_lines)


class QwenVLWrapper(base_wrapper.MultimodalLlmWrapper):
  """
  Qwen2.5 VL wrapper via OpenRouter with smart rate limiting.
  
  Features:
  - Automatic rate limiting (20 RPM for free models)
  - Credit-aware daily limits (50 vs 1000 requests)
  - Smart model selection and fallbacks
  - Persistent usage tracking
  - 402 error handling for negative balances
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str = "qwen/qwen2.5-vl-72b-instruct:free",
      max_retry: int = 3,
      temperature: float = 0.0,
      max_tokens: int = 2048,
      site_url: str = None,
      site_name: str = None,
      verbose: bool = True,
      high_credits: bool = None,  # None = auto-detect, True = 10+ credits, False = <10 credits
  ):
    """Initialize the Qwen VL wrapper with rate limiting."""
    if 'OPENROUTER_API_KEY' not in os.environ:
      raise RuntimeError('OPENROUTER_API_KEY environment variable not set.')
    
    super().__init__()
    self.openrouter_api_key = os.environ['OPENROUTER_API_KEY']
    self.max_retry = min(max(1, max_retry), 5)
    self.temperature = temperature
    self.preferred_model = model_name
    self.max_tokens = max_tokens
    self.site_url = site_url or "https://github.com/google-research/android_world"
    self.site_name = site_name or "AndroidWorld"
    self.verbose = verbose
    
    # Initialize smart routing components
    self.usage_tracker = OpenRouterUsageTracker()
    self.router = OpenRouterModelRouter(self.usage_tracker)
    
    # Set credit status if provided
    if high_credits is not None:
      self.usage_tracker.update_credit_status(high_credits)
      credit_msg = "10+ credits (1000 daily limit)" if high_credits else "<10 credits (50 daily limit)"
      print(f"💰 Credit status set: {credit_msg}")
    
    print(f"✅ Qwen2.5 VL wrapper initialized with rate limiting")
    print(f"🆓 Preferred model: {model_name}")
    
    if self.verbose:
      print(self.router.get_usage_summary())

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
    """Text-only prediction (no images)."""
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Multimodal prediction with images and smart rate limiting."""
    
    # Select the best available model
    selected_model = self.router.select_best_model(self.preferred_model, max_wait_seconds=60)
    if not selected_model:
      print("❌ No available models")
      return base_wrapper.ERROR_CALLING_LLM, None, None
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self.openrouter_api_key}',
        'HTTP-Referer': self.site_url,
        'X-Title': self.site_name,
    }

    # Build content with text and images
    content = [{'type': 'text', 'text': text_prompt}]
    for image in images:
      content.append({
          'type': 'image_url',
          'image_url': {'url': f'data:image/jpeg;base64,{self.encode_image(image)}'},
      })

    payload = {
        'model': selected_model,
        'temperature': self.temperature,
        'messages': [{'role': 'user', 'content': content}],
        'max_tokens': self.max_tokens,
    }

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    
    while counter > 0:
      try:
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=60,
        )
        
        # Handle specific status codes
        if response.status_code == 402:
          print("💳 Error 402: Negative credit balance detected")
          print("   Add credits to your OpenRouter account to continue using free models")
          self.usage_tracker.update_credit_status(False)  # Assume low credits
          return base_wrapper.ERROR_CALLING_LLM, None, None
        
        response.raise_for_status()
        
        response_json = response.json()
        if 'choices' in response_json and response_json['choices']:
          # Record successful usage
          self.usage_tracker.record_usage(selected_model)
          
          # Try to detect credit status from successful high-volume usage
          daily_usage = self.usage_tracker.get_daily_usage(selected_model)
          if (daily_usage > 50 and not self.usage_tracker.has_high_credits and 
              OPENROUTER_MODEL_CONFIGS[selected_model]["is_free"]):
            print("🎉 Detected 10+ credits based on successful high-volume usage!")
            self.usage_tracker.update_credit_status(True)
          
          return (
              response_json['choices'][0]['message']['content'],
              True,  # Assume safe for OpenRouter
              response,
          )
        
        # Handle OpenRouter errors
        error_info = response_json.get('error', {})
        err_msg = error_info.get('message', 'Unknown error')
        print(f'❌ OpenRouter API error: {err_msg}')
        
        # Handle rate limiting specifically
        if 'rate' in err_msg.lower() or 'limit' in err_msg.lower():
          print("🚦 Rate limit detected, trying different model...")
          # Mark this model as rate limited and try another
          selected_model = self.router.select_best_model(None, max_wait_seconds=30)
          if selected_model:
            payload['model'] = selected_model
            continue
          else:
            print("⏳ All models rate limited, waiting...")
            time.sleep(wait_seconds)
        
      except requests.exceptions.Timeout:
        print(f"⏰ Request timeout, retrying...")
      except requests.exceptions.RequestException as e:
        print(f"🌐 Request error calling OpenRouter: {e}")
      except Exception as e:
        print(f"💥 Unexpected error: {e}")
      
      counter -= 1
      if counter > 0:
        print(f'🔄 Retrying in {wait_seconds} seconds... ({counter} attempts left)')
        time.sleep(wait_seconds)
        wait_seconds = min(wait_seconds * 1.5, 120)

    print("❌ All retry attempts failed")
    return base_wrapper.ERROR_CALLING_LLM, None, None

  def get_model_info(self) -> dict:
    """Get information about the model and current usage."""
    return {
        'name': self.preferred_model,
        'provider': 'OpenRouter',
        'cost': 'FREE',
        'context_length': 131072,
        'multimodal': True,
        'rate_limits': OPENROUTER_MODEL_CONFIGS.get(self.preferred_model, {}),
        'usage_summary': self.router.get_usage_summary(),
        'capabilities': [
            'UI Understanding',
            'Layout Analysis', 
            'Icon Recognition',
            'Text Recognition',
            'Chart Analysis',
            'Android Screenshot Analysis'
        ]
    }
  
  def get_usage_summary(self) -> str:
    """Get current usage summary."""
    return self.router.get_usage_summary()
  
  def reset_usage(self) -> None:
    """Reset usage tracking (useful for testing)."""
    if self.usage_tracker.storage_file.exists():
      self.usage_tracker.storage_file.unlink()
    self.usage_tracker = OpenRouterUsageTracker()
    self.router = OpenRouterModelRouter(self.usage_tracker)
    print("Usage tracking reset.") 