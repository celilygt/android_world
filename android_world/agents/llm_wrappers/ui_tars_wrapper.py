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

"""UI-TARS wrapper using Ollama for the 0000/ui-tars-1.5-7b-q8_0 model.

This wrapper connects to Ollama running the UI-TARS-1.5-7b-q8_0 model specifically.
The model is optimized for Android GUI interaction and UI automation tasks.

To use this wrapper:
1. Install Ollama: https://ollama.ai/
2. Pull the UI-TARS model: `ollama run 0000/ui-tars-1.5-7b-q8_0`
3. Start Ollama server: `ollama serve` (if not running)

Example usage:
    wrapper = UITarsWrapper()
    response = wrapper.generate_text("Click on the login button", max_tokens=256)
"""

import time
import json
import requests
from typing import Any, Optional, Tuple
from android_world.agents.llm_wrappers.base_wrapper import MultimodalLlmWrapper


class UITarsWrapper(MultimodalLlmWrapper):
  """UI-TARS wrapper specifically for 0000/ui-tars-1.5-7b-q8_0 via Ollama."""

  def __init__(
      self,
      model_name: str = "avil/UI-TARS:latest",
      ollama_host: str = "http://localhost:11434",
      temperature: float = 0.0,
      max_new_tokens: int = 256,
      **kwargs
  ):
    """Initialize the UI-TARS wrapper.
    
    Args:
        model_name: Ollama model name (should be 0000/ui-tars-1.5-7b-q8_0)
        ollama_host: Ollama server host
        temperature: Generation temperature
        max_new_tokens: Maximum tokens to generate
        **kwargs: Additional arguments (ignored)
    """
    super().__init__()
    self.model_name = model_name
    self.ollama_host = ollama_host.rstrip('/')
    self.temperature = temperature
    self.max_new_tokens = max_new_tokens
    
    # Verify Ollama connection and model availability
    self._verify_connection()

  def _verify_connection(self) -> None:
    """Verify Ollama server is running and model is available."""
    try:
      # Check if Ollama server is running
      response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
      response.raise_for_status()
      
      models = response.json().get("models", [])
      model_names = [model["name"] for model in models]
      
      if self.model_name not in model_names:
        print(f"⚠️  Model {self.model_name} not found in Ollama.")
        print(f"Available models: {model_names}")
        print(f"To pull the model, run: ollama pull {self.model_name}")
        # Don't raise error - let it fail later if user tries to use it
        
    except requests.RequestException as e:
      print(f"⚠️  Could not connect to Ollama at {self.ollama_host}")
      print(f"Error: {e}")
      print("Make sure Ollama is running: ollama serve")

  def generate_text(
      self, 
      prompt: str, 
      max_tokens: Optional[int] = None,
      **kwargs
  ) -> str:
    """Generate text using UI-TARS via Ollama.
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate (overrides default)
        **kwargs: Additional generation parameters
        
    Returns:
        Generated text
    """
    if max_tokens is None:
      max_tokens = self.max_new_tokens
    
    try:
      # Prepare request payload
      payload = {
        "model": self.model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
          "temperature": self.temperature,
          "num_predict": max_tokens,
        }
      }
      
      # Add any additional options from kwargs
      if kwargs:
        payload["options"].update(kwargs)
      
      # Make request to Ollama
      response = requests.post(
        f"{self.ollama_host}/api/generate",
        json=payload,
        timeout=60
      )
      response.raise_for_status()
      
      result = response.json()
      return result.get("response", "").strip()
      
    except requests.RequestException as e:
      error_msg = f"Failed to generate text with UI-TARS: {e}"
      print(f"❌ {error_msg}")
      # Return empty string rather than raising to allow graceful degradation
      return ""

  def generate_text_with_images(
      self,
      prompt: str,
      images: list,
      max_tokens: Optional[int] = None,
      **kwargs
  ) -> Tuple[str, dict]:
    """Generate text with images using UI-TARS via Ollama.
    
    Args:
        prompt: Text prompt
        images: List of image data (numpy arrays or PIL Images)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (generated_text, metadata_dict)
    """
    import base64
    import io
    import numpy as np
    from PIL import Image
    
    if max_tokens is None:
      max_tokens = self.max_new_tokens
    
    try:
      # Convert images to base64
      image_data = []
      for img in images:
        if isinstance(img, np.ndarray):
          # Convert numpy array to PIL Image
          if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
          pil_img = Image.fromarray(img)
        elif hasattr(img, 'save'):
          # Assume it's already a PIL Image
          pil_img = img
        else:
          print(f"⚠️  Unsupported image type: {type(img)}")
          continue
          
        # Convert to base64
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        image_data.append(img_b64)
      
      # Prepare request payload with images
      payload = {
        "model": self.model_name,
        "prompt": prompt,
        "images": image_data,
        "stream": False,
        "options": {
          "temperature": self.temperature,
          "num_predict": max_tokens,
        }
      }
      
      # Add any additional options from kwargs
      if kwargs:
        payload["options"].update(kwargs)
      
      # Make request to Ollama
      response = requests.post(
        f"{self.ollama_host}/api/generate",
        json=payload,
        timeout=120  # Longer timeout for vision models
      )
      response.raise_for_status()
      
      result = response.json()
      generated_text = result.get("response", "").strip()
      
      # Create metadata
      metadata = {
        "model": self.model_name,
        "prompt_tokens": result.get("prompt_eval_count", 0),
        "completion_tokens": result.get("eval_count", 0),
        "total_time": result.get("total_duration", 0) / 1e9,  # Convert to seconds
        "images_processed": len(image_data)
      }
      
      return generated_text, metadata
      
    except Exception as e:
      error_msg = f"Failed to generate text with images: {e}"
      print(f"❌ {error_msg}")
      return "", {"error": error_msg}

  def get_model_info(self) -> dict:
    """Get information about the UI-TARS model.
    
    Returns:
        Dictionary with model information
    """
    try:
      response = requests.get(f"{self.ollama_host}/api/show", 
                            params={"name": self.model_name}, 
                            timeout=10)
      if response.status_code == 200:
        model_info = response.json()
        return {
          "name": self.model_name,
          "size": model_info.get("size", "Unknown"),
          "family": "UI-TARS",
          "parameters": "7B",
          "quantization": "Q8_0",
          "status": "Ready",
          "capabilities": ["GUI Interaction", "Android Automation", "UI Understanding"]
        }
    except Exception as e:
      print(f"Could not get model info: {e}")
    
    return {
      "name": self.model_name,
      "status": "Unknown",
      "capabilities": ["GUI Interaction", "Android Automation", "UI Understanding"]
    }

  def predict_mm(
      self, text_prompt: str, images: list
  ) -> tuple[str, Optional[bool], dict]:
    """Calls UI-TARS with a prompt and images.
    
    Args:
        text_prompt: The text prompt
        images: A list of images as numpy arrays
        
    Returns:
        A tuple containing (response_text, safety_flag, raw_response)
    """
    try:
      if images:
        response, metadata = self.generate_text_with_images(
            text_prompt, images, max_tokens=self.max_new_tokens
        )
        return response, True, metadata
      else:
        response = self.generate_text(
            text_prompt, max_tokens=self.max_new_tokens
        )
        return response, True, {}
        
    except Exception as e:
      print(f"❌ Error in predict_mm: {e}")
      return "", False, {"error": str(e)}

  def test_model(self) -> bool:
    """Test if the UI-TARS model is working correctly.
    
    Returns:
        True if model is working, False otherwise
    """
    test_prompt = "Generate a JSON action to click on a button at coordinates (100, 200)."
    
    try:
      response = self.generate_text(test_prompt, max_tokens=50)
      
      if response and len(response) > 0:
        print(f"✅ UI-TARS model test successful!")
        print(f"Test response: {response[:100]}...")
        return True
      else:
        print(f"❌ UI-TARS model test failed: Empty response")
        return False
        
    except Exception as e:
      print(f"❌ UI-TARS model test failed: {e}")
      return False 