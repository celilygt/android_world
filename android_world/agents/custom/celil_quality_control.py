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

"""Quality control verifier ensemble for the Celil agent."""

from android_world.agents.llm_wrappers.base_wrapper import MultimodalLlmWrapper
import json

PRAGMATIST_PROMPT = """You are a pragmatist. Analyze the screenshot and determine if the proposed action is a logical step towards the sub-goal.

Sub-goal: {sub_goal}
Proposed Action: {proposed_action}
Context: {context_summary}
Current Screen: {observation_summary}

Look at the screenshot. Is this action correct and likely to work? JSON: {{"score": 0-10, "reasoning": "..."}}"""

SKEPTIC_PROMPT = """You are a skeptic. Examine the screenshot carefully for potential flaws and risks in the proposed action.

Sub-goal: {sub_goal}  
Proposed Action: {proposed_action}
Context: {context_summary}
Current Screen: {observation_summary}

Looking at the screenshot, how likely is this to fail? Consider UI misinterpretation, wrong elements, etc. JSON: {{"score": 0-10, "reasoning": "..."}}"""

EFFICIENCY_EXPERT_PROMPT = """You are an efficiency expert. Study the screenshot to determine if this is the most efficient way to achieve the sub-goal.

Sub-goal: {sub_goal}
Proposed Action: {proposed_action}
Context: {context_summary}
Current Screen: {observation_summary}

Based on the screenshot, is there a more direct or efficient approach? JSON: {{"score": 0-10, "reasoning": "..."}}"""


class VerifierEnsemble:
  """An ensemble of verifiers that score actions before execution using visual analysis."""

  def __init__(self, llm_wrapper: MultimodalLlmWrapper):
    """Initializes the VerifierEnsemble."""
    self.llm_wrapper = llm_wrapper

  def _create_observation_summary(self, observation: dict) -> str:
    """Creates a concise summary of the observation for the verifiers."""
    return observation.get('summary', 'No observation summary available')

  def verify_action(
      self, sub_goal: str, proposed_action: dict, observation: dict, context_summary: str
  ) -> tuple[float, dict]:
    """Verifies an action using an ensemble of verifiers with visual analysis."""
    scores = {}
    reasoning = {}
    
    # Create a concise observation summary and limit other inputs
    observation_summary = self._create_observation_summary(observation)
    action_str = str(proposed_action)[:200]
    limited_context = context_summary[:300] if context_summary else ""

    for persona, prompt_template in [
        ("pragmatist", PRAGMATIST_PROMPT),
        ("skeptic", SKEPTIC_PROMPT), 
        ("efficiency_expert", EFFICIENCY_EXPERT_PROMPT),
    ]:
      prompt = prompt_template.format(
          sub_goal=sub_goal,
          proposed_action=action_str,
          context_summary=limited_context,
          observation_summary=observation_summary,
      )
      
      # Use multimodal prediction with screenshot for better verification (NOW FREE!)
      response, _, _ = self.llm_wrapper.predict_mm(prompt, [observation['screenshot']])
      try:
        data = json.loads(response)
        scores[persona] = data.get("score", 0)
        reasoning[persona] = data.get("reasoning", "")
      except json.JSONDecodeError:
        # Try to extract score from response if JSON parsing fails
        try:
          if "score" in response.lower():
            # Look for patterns like "score": 7 or "score: 7"
            import re
            score_match = re.search(r'"?score"?\s*[:=]\s*(\d+)', response.lower())
            if score_match:
              scores[persona] = int(score_match.group(1))
            else:
              scores[persona] = 5  # Default middle score
          else:
            scores[persona] = 5
        except Exception:
          scores[persona] = 5
        reasoning[persona] = f"Parsing issue: {response[:100]}..."

    final_score = (
        0.4 * scores.get("pragmatist", 0)
        + 0.5 * scores.get("skeptic", 0)
        + 0.1 * scores.get("efficiency_expert", 0)
    )

    return final_score, reasoning