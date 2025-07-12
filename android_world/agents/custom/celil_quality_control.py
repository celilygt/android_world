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

CONSOLIDATED_VERIFIER_PROMPT = """You are a critical supervisor for an Android agent. Analyze the screenshot and proposed action.
    Evaluate the action from three perspectives:
    1.  **Pragmatist**: Is this a logical, direct step towards the sub-goal?
    2.  **Skeptic**: What's the risk of failure? Does it rely on misinterpreting the UI?
    3.  **Efficiency Expert**: Is there a more direct or simpler way to do this?

    Sub-goal: {sub_goal}
    Context: {context_summary}
    Proposed Action: {proposed_action}
    Current Screen: {observation_summary}

    Return a single JSON object with a score (0-10) and brief reasoning for each persona.
    Example: {{"pragmatist": {{"score": 9, "reasoning": "..."}}, "skeptic": {{"score": 8, "reasoning": "..."}}, "efficiency_expert": {{"score": 6, "reasoning": "..."}}}}
    """


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
        """Verifies an action using a single, consolidated verifier call."""
        observation_summary = self._create_observation_summary(observation)
        action_str = str(proposed_action)[:200]
        limited_context = context_summary[:300] if context_summary else ""

        prompt = CONSOLIDATED_VERIFIER_PROMPT.format(
            sub_goal=sub_goal,
            proposed_action=action_str,
            context_summary=limited_context,
            observation_summary=observation_summary,
        )

        response, _, _ = self.llm_wrapper.predict_mm(prompt, [observation['screenshot']])
        
        scores = {}
        reasoning = {}
        
        try:
            data = json.loads(response)
            prag_data = data.get("pragmatist", {})
            skep_data = data.get("skeptic", {})
            eff_data = data.get("efficiency_expert", {})

            scores["pragmatist"] = prag_data.get("score", 0)
            reasoning["pragmatist"] = prag_data.get("reasoning", "")
            
            scores["skeptic"] = skep_data.get("score", 0)
            reasoning["skeptic"] = skep_data.get("reasoning", "")

            scores["efficiency_expert"] = eff_data.get("score", 0)
            reasoning["efficiency_expert"] = eff_data.get("reasoning", "")

        except (json.JSONDecodeError, AttributeError):
            scores = {"pragmatist": 5, "skeptic": 5, "efficiency_expert": 5}
            reasoning = {"error": f"Failed to parse verifier response: {response[:100]}"}

        final_score = (
            0.4 * scores.get("pragmatist", 0)
            + 0.5 * scores.get("skeptic", 0)
            + 0.1 * scores.get("efficiency_expert", 0)
        )

        return final_score, reasoning