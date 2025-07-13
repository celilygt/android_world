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

"""A perception module for the Celil agent."""

from PIL import Image
import pytesseract
import json
from typing import Optional
from android_world.env import interface
from android_world.agents.llm_wrappers.base_wrapper import MultimodalLlmWrapper

VISUAL_ANALYSIS_PROMPT = """Analyze this Android screen and provide a structured list of all interactive UI elements.

For each interactable element you can see, provide:
- text: The visible text (if any)
- type: The element type (button, text_field, checkbox, etc.)
- description: What this element does
- bounding_box: Approximate location as [x, y, width, height] (estimate based on visual position)

Return as JSON list: [{"text": "...", "type": "...", "description": "...", "bounding_box": [x, y, w, h]}, ...]

Be precise and only include elements that are clearly visible and interactable."""


class PerceptionModule:
  """Processes raw environment observations into a structured format."""

  def __init__(self, llm_wrapper: Optional[MultimodalLlmWrapper] = None):
    """Initialize the perception module.
    
    Args:
      llm_wrapper: Optional multimodal LLM for enhanced visual understanding.
                   If None, falls back to OCR-only processing.
    """
    self.llm_wrapper = llm_wrapper

  def _get_enhanced_visual_analysis(self, screenshot) -> list[dict]:
    """Use multimodal LLM to get enhanced visual analysis of the screen."""
    # This is the slow part. We disable it to improve performance and rely
    # on the action generator's multimodality instead.
    return []

  def process_observation(self, state: interface.State, deep_analysis: bool = False) -> dict:
    """Processes a raw state from the environment.

    Args:
      state: The raw state from the environment.
      deep_analysis: THIS IS NOW IGNORED. Deep analysis is disabled for speed.
        The action generator now receives the raw screenshot directly.

    Returns:
      A dictionary containing the structured observation.
    """
    screenshot = state.pixels

    # We have disabled the slow deep analysis by modifying _get_enhanced_visual_analysis.
    # This call is now fast and will always return [].
    enhanced_elements = self._get_enhanced_visual_analysis(screenshot)

    try:
      ocr_data = pytesseract.image_to_data(
        Image.fromarray(screenshot), output_type=pytesseract.Output.DICT
      )
    except pytesseract.TesseractNotFoundError:
      # Handle case where Tesseract is not installed or not in PATH
      # You might want to log a warning or raise an exception
      # For now, we'll return empty OCR results
      ocr_data = {
        'text': [],
        'left': [],
        'top': [],
        'width': [],
        'height': [],
        'conf': [],
      }

    ocr_results = []
    for i in range(len(ocr_data['text'])):
      if int(float(ocr_data['conf'][i])) > 0:  # Filter out empty boxes
        text = ocr_data['text'][i].strip()
        if text:
          left = ocr_data['left'][i]
          top = ocr_data['top'][i]
          width = ocr_data['width'][i]
          height = ocr_data['height'][i]
          ocr_results.append({
            'text': text,
            'bbox': (left, top, width, height),
          })

    # Create a concise summary for the LLM
    visible_text = [result['text'] for result in ocr_results]
    ui_elements_summary = []

    if hasattr(state, 'ui_elements') and state.ui_elements:
      for element in state.ui_elements[:10]:  # Limit to first 10 elements to avoid token bloat
        if hasattr(element, 'text') and element.text and element.text.strip():
          ui_elements_summary.append(f"'{element.text.strip()}'")
        elif hasattr(element, 'content_description') and element.content_description:
          ui_elements_summary.append(f"[{element.content_description}]")

    # Combine OCR, enhanced analysis, and UI elements for summary
    summary_parts = []
    if enhanced_elements:
      enhanced_text = [elem.get('text', elem.get('description', '')) for elem in enhanced_elements if elem.get('text') or elem.get('description')]
      if enhanced_text:
        summary_parts.append(f"AI-detected elements: {', '.join(enhanced_text[:10])}")

    if visible_text:
      summary_parts.append(f"OCR text: {', '.join(visible_text[:15])}")  # Limit text to avoid token bloat
    if ui_elements_summary:
      summary_parts.append(f"System UI elements: {', '.join(ui_elements_summary)}")

    summary = "; ".join(summary_parts) if summary_parts else "No significant UI content detected"

    return {
      'screenshot': screenshot,
      'ocr_results': ocr_results,
      'enhanced_elements': enhanced_elements,  # New: AI-powered visual analysis
      'ui_tree': state.ui_elements,
      'summary': summary,
    }