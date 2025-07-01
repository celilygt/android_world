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

"""The Maestro Planner for the Celil agent."""

from android_world.agents.llm_wrappers.base_wrapper import MultimodalLlmWrapper

PLANNER_PROMPT_TEMPLATE = """You are a master strategist for an Android agent. Analyze the screenshot and break down the goal into simple sub-goals.

Goal: "{goal}"

Current Screen Analysis:
Text visible: {ocr_summary}
Interactive elements: {ui_summary}

{retrieved_plan_section}

Based on the screenshot and this information, provide a numbered list of sub-goals. Each sub-goal should be a single, clear instruction.

Example:
1. Open the Files app.
2. Navigate to Downloads folder.
3. Select report.pdf file.
4. Tap delete button.
5. Confirm deletion.
"""

CORRECTIVE_PROMPT_TEMPLATE = """You are an expert agent supervisor. An agent has failed and needs a new plan. Analyze the current screenshot to understand the situation.

Original Goal: "{goal}"
Failed at step: "{failed_sub_goal}"
Attempted action: "{failed_action}"
Why it failed: "{verifier_feedback}"

Current screen analysis:
Text: {ocr_summary}
Elements: {ui_summary}

Based on the current screenshot, provide a new numbered plan to achieve the original goal from this state.
"""


class MaestroPlanner:
  """A high-level planner that uses an LLM to create and correct plans."""

  def __init__(self, llm_wrapper: MultimodalLlmWrapper):
    """Initializes the MaestroPlanner."""
    self.llm_wrapper = llm_wrapper

  def _create_ocr_summary(self, ocr_results: list[dict]) -> str:
    """Creates a concise summary of the OCR results."""
    texts = [result['text'] for result in ocr_results[:15]]  # Limit to 15 items
    return ", ".join(texts) if texts else "No text visible"

  def _create_ui_tree_summary(self, ui_tree) -> str:
    """Creates a concise summary of the UI tree."""
    if not ui_tree:
      return "No UI elements detected"
    
    summary_items = []
    element_count = 0
    
    for element in ui_tree:
      if element_count >= 20:  # Limit to 20 elements max
        break
        
      try:
        # Get clickable elements with text or content description
        if hasattr(element, 'clickable') and element.clickable:
          if hasattr(element, 'text') and element.text and element.text.strip():
            summary_items.append(f"Button: '{element.text.strip()}'")
            element_count += 1
          elif hasattr(element, 'content_description') and element.content_description:
            summary_items.append(f"Button: [{element.content_description}]")
            element_count += 1
        # Get text elements
        elif hasattr(element, 'text') and element.text and element.text.strip():
          summary_items.append(f"Text: '{element.text.strip()}'")
          element_count += 1
      except Exception:
        continue  # Skip problematic elements
    
    return "; ".join(summary_items[:15]) if summary_items else "No interactive elements"

  def generate_initial_plan(
      self, goal: str, observation: dict, retrieved_plan: list[str] | None
  ) -> list[str]:
    """Generates an initial plan using both screenshot and text analysis."""
    ocr_summary = self._create_ocr_summary(observation['ocr_results'])
    ui_summary = self._create_ui_tree_summary(observation['ui_tree'])

    if retrieved_plan:
      retrieved_plan_section = (
          "Similar task plan:\n" + "\n".join(retrieved_plan[:5])  # Limit to 5 steps
      )
    else:
      retrieved_plan_section = ""

    prompt = PLANNER_PROMPT_TEMPLATE.format(
        goal=goal,
        ocr_summary=ocr_summary,
        ui_summary=ui_summary,
        retrieved_plan_section=retrieved_plan_section,
    )
    
    # Use multimodal prediction with screenshot for better understanding (NOW FREE!)
    response, _, _ = self.llm_wrapper.predict_mm(prompt, [observation['screenshot']])
    return self._parse_plan(response)

  def generate_corrective_plan(self, failure_context: dict) -> list[str]:
    """Generates a corrective plan using screenshot analysis."""
    ocr_summary = self._create_ocr_summary(
        failure_context['observation']['ocr_results']
    )
    ui_summary = self._create_ui_tree_summary(
        failure_context['observation']['ui_tree']
    )

    # Ensure verifier_feedback is a string before slicing
    verifier_feedback = failure_context['verifier_feedback']
    if isinstance(verifier_feedback, dict):
      # Convert dict to string summary
      verifier_feedback = str(verifier_feedback)
    verifier_feedback_str = str(verifier_feedback)[:300]  # Truncate long feedback

    prompt = CORRECTIVE_PROMPT_TEMPLATE.format(
        goal=failure_context['goal'],
        failed_sub_goal=failure_context['failed_sub_goal'],
        failed_action=str(failure_context['failed_action'])[:200],  # Truncate long actions
        verifier_feedback=verifier_feedback_str,
        ocr_summary=ocr_summary,
        ui_summary=ui_summary,
    )
    
    # Use multimodal prediction with screenshot for better error recovery (NOW FREE!)
    response, _, _ = self.llm_wrapper.predict_mm(prompt, [failure_context['observation']['screenshot']])
    return self._parse_plan(response)

  def _parse_plan(self, response: str) -> list[str]:
    """Parses a numbered list response into a Python list of strings."""
    plan = []
    for line in response.split("\n"):
      line = line.strip()
      if line and line[0].isdigit():
        plan.append(line.split(".", 1)[1].strip())
    return plan