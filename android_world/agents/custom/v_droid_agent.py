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

"""V-DROID Agent: A verifier-based approach for Android device control.

This agent implements the V-DROID architecture described in the V-DROID paper,
which replaces the traditional generator-based approach with a verifier-based
multi-step process:
1. Extract all possible candidate actions from the screen
2. Use an LLM as a verifier to score each candidate  
3. Execute the highest-scoring candidate
"""

import time
from typing import Any, Optional

from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents.llm_wrappers.gemini_gemma_wrapper import GeminiGemmaWrapper
from android_world.agents.custom import candidate_generation
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils


# Enhanced verification prompt template for scoring candidate actions
VERIFICATION_PROMPT_TEMPLATE = """You are controlling an Android device to complete this task: {goal}

Current screen has these elements:
{ui_elements}

Candidate action: {candidate_action}

Is this action helpful for the task? Answer only "Yes" or "No".

For creating contacts, look for:
- "Add" or "+" buttons to create new contacts  
- Contact app icons
- Text fields for name/phone
- Save/Done buttons

Answer: """


class VDroidAgent(base_agent.EnvironmentInteractingAgent):
    """V-DROID Agent using verifier-based action selection."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        model_name: str = "gemma-3-27b-it",
        name: str = 'V-DROID-Agent',
        wait_after_action_seconds: float = 2.0,
        max_retry: int = 3,
        temperature: float = 0.0,
        top_p: float = 0.95,
        enable_safety_checks: bool = True,
        verbose: bool = True,
    ):
        """Initializes a V-DROID Agent using Gemini API with Gemma model as verifier.

        Args:
            env: The environment.
            model_name: The Gemini model to use as verifier.
            name: The agent name.
            wait_after_action_seconds: Seconds to wait for the screen to stabilize
                after executing an action.
            max_retry: Max number of retries when calling the LLM.
            temperature: Temperature for LLM generation.
            top_p: Top-p sampling parameter.
            enable_safety_checks: Whether to enable Gemini safety checks.
            verbose: Whether to enable verbose output during agent execution.
        """
        super().__init__(env, name)
        
        # Initialize the Gemini Gemma LLM wrapper as our verifier
        self.llm = GeminiGemmaWrapper(
            model_name=model_name,
            max_retry=max_retry,
            temperature=temperature,
            top_p=top_p,
            enable_safety_checks=enable_safety_checks,
        )
        
        self.history = []
        self.additional_guidelines = None
        self.wait_after_action_seconds = wait_after_action_seconds
        self.verbose = verbose

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        """Set additional task-specific guidelines."""
        self.additional_guidelines = task_guidelines

    def reset(self, go_home_on_reset: bool = False):
        """Reset the agent state."""
        super().reset(go_home_on_reset)
        # Hide the coordinates on screen which might affect the vision model
        self.env.hide_automation_ui()
        self.history = []

    def _score_candidate_actions(
        self, 
        goal: str, 
        history: list[str], 
        ui_elements: list[representation_utils.UIElement], 
        candidates: list[dict]
    ) -> list[tuple[float, dict]]:
        """Score candidate actions using the verifier LLM.
        
        Args:
            goal: The task goal.
            history: List of previous action descriptions.
            ui_elements: List of UIElement objects from current state.
            candidates: List of candidate action dictionaries.
            
        Returns:
            List of (score, candidate_action) tuples.
        """
        scored_candidates = []
        
        # Generate detailed UI description for the prompt
        ui_description = self._generate_ui_elements_description(ui_elements)
        
        # Prioritize candidates to reduce API calls
        prioritized_candidates = self._prioritize_candidates(candidates, goal)
        
        # Limit to top 10 candidates to avoid rate limiting
        candidates_to_score = prioritized_candidates[:10]
        
        for candidate in candidates_to_score:
            # Format the candidate action for verification
            action_description = self._format_candidate_action(candidate)
            
            # Create verification prompt
            verification_prompt = VERIFICATION_PROMPT_TEMPLATE.format(
                goal=goal,
                ui_elements=ui_description,
                candidate_action=action_description
            )
            
            if self.verbose:
                print(f"🔍 Verifying candidate: {action_description}")
            
            # Call verifier LLM
            try:
                response, is_safe, raw_response = self.llm.predict(verification_prompt)
                
                if not is_safe or not response:
                    score = 0.0
                else:
                    # Parse response - "Yes" = 1, anything else = 0
                    score = 1.0 if "yes" in response.lower().strip() else 0.0
                    
                if self.verbose:
                    print(f"   Score: {score} (Response: {response.strip() if response else 'None'})")
                    
            except Exception as e:
                print(f"Error calling verifier LLM: {e}")
                score = 0.0
            
            scored_candidates.append((score, candidate))
        
        # Add unscored candidates with score 0.0
        for candidate in candidates[len(candidates_to_score):]:
            scored_candidates.append((0.0, candidate))
        
        return scored_candidates

    def _prioritize_candidates(self, candidates: list[dict], goal: str) -> list[dict]:
        """Prioritize candidates based on likely relevance to the goal.
        
        Args:
            candidates: List of candidate action dictionaries.
            goal: The task goal.
            
        Returns:
            Candidates sorted by likely relevance.
        """
        def candidate_priority(candidate):
            score = 0
            action_type = candidate.get('action_type', '')
            text = candidate.get('text', '') or ''
            
            # Prioritize UI element actions over visual clicks
            if 'index' in candidate:
                score += 50
            
            # Prioritize certain action types
            if action_type == 'click':
                score += 30
            elif action_type == 'input_text':
                score += 40
            elif action_type in ['navigate_home', 'navigate_back']:
                score += 10
            
            # Boost candidates with relevant text
            if 'contact' in goal.lower():
                if any(word in text.lower() for word in ['contact', 'add', '+', 'new', 'create']):
                    score += 100
                if any(word in text.lower() for word in ['phone', 'number', 'name']):
                    score += 80
            
            return score
        
        return sorted(candidates, key=candidate_priority, reverse=True)

    def _generate_ui_elements_description(self, ui_elements: list[representation_utils.UIElement]) -> str:
        """Generate a description of current UI elements for the verification prompt.
        
        Args:
            ui_elements: List of UIElement objects from the current state.
            
        Returns:
            Formatted string describing the UI elements.
        """
        if not ui_elements:
            return "No UI elements detected on current screen."
        
        descriptions = []
        for i, element in enumerate(ui_elements[:20]):  # Limit to first 20 elements to avoid overwhelming the prompt
            parts = [f"Element {i}:"]
            
            if hasattr(element, 'text') and element.text:
                parts.append(f"text='{element.text}'")
            
            if hasattr(element, 'content_desc') and element.content_desc:
                parts.append(f"description='{element.content_desc}'")
                
            if hasattr(element, 'resource_id') and element.resource_id:
                parts.append(f"id='{element.resource_id}'")
            
            if hasattr(element, 'class_name') and element.class_name:
                class_name = str(element.class_name).split('.')[-1]  # Get just the class name
                parts.append(f"type={class_name}")
                
            properties = []
            if hasattr(element, 'clickable') and element.clickable:
                properties.append("clickable")
            if hasattr(element, 'scrollable') and element.scrollable:
                properties.append("scrollable")
            if hasattr(element, 'editable') and element.editable:
                properties.append("editable")
            if properties:
                parts.append(f"properties={','.join(properties)}")
                
            descriptions.append(" ".join(parts))
        
        if len(ui_elements) > 20:
            descriptions.append(f"... and {len(ui_elements) - 20} more elements")
            
        return "\n".join(descriptions)

    def _format_candidate_action(self, candidate: dict) -> str:
        """Format a candidate action dictionary into a readable description."""
        action_type = candidate.get('action_type', 'unknown')
        
        if action_type == 'click':
            return f"Click on element at index {candidate.get('index', 'unknown')}"
        elif action_type == 'input_text':
            text = candidate.get('text', '')
            return f"Input text '{text}' into element at index {candidate.get('index', 'unknown')}"
        elif action_type == 'scroll':
            direction = candidate.get('direction', 'unknown')
            return f"Scroll {direction} on element at index {candidate.get('index', 'unknown')}"
        elif action_type == 'long_press':
            return f"Long press on element at index {candidate.get('index', 'unknown')}"
        elif action_type == 'navigate_home':
            return "Navigate to home screen"
        elif action_type == 'navigate_back':
            return "Navigate back"
        elif action_type == 'status':
            status = candidate.get('goal_status', 'unknown')
            return f"Mark task as {status}"
        elif action_type == 'wait':
            return "Wait for screen to update"
        else:
            return f"Perform {action_type} action"

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """Perform a step using the V-DROID verifier-based approach.
        
        The V-DROID approach:
        1. Get current state
        2. Generate candidate actions
        3. Score candidates with verifier
        4. Execute highest-scoring action
        5. Update history
        
        Args:
            goal: The task goal.
            
        Returns:
            AgentInteractionResult with execution details.
        """
        step_data = {
            'step_number': len(self.history) + 1,
            'goal': goal,
            'candidates': [],
            'scores': [],
            'selected_action': None,
            'execution_success': False,
        }
        
        print(f'----------V-DROID Step {step_data["step_number"]}')
        
        # Get current state
        state = self.get_post_transition_state()
        
        # Generate candidate actions using the proper V-DROID candidate generation
        candidates = candidate_generation.generate_candidate_actions(state)
        step_data['candidates'] = candidates
        
        if self.verbose:
            print(f"📋 Generated {len(candidates)} candidate actions")
            candidate_summary = candidate_generation.format_candidates_for_display(candidates[:10])  # Show first 10
            print(candidate_summary)
        
        if not candidates:
            print("❌ No candidate actions generated")
            step_data['selected_action'] = {'action_type': 'status', 'goal_status': 'infeasible'}
            self.history.append(f"Step {step_data['step_number']}: No valid candidates found")
            return base_agent.AgentInteractionResult(True, step_data)
        
        # Score candidates using verifier
        scored_candidates = self._score_candidate_actions(
            goal, 
            self.history, 
            state.ui_elements, 
            candidates
        )
        step_data['scores'] = [score for score, _ in scored_candidates]
        
        # Select highest-scoring candidate
        if not scored_candidates:
            print("❌ No candidates were scored")
            step_data['selected_action'] = {'action_type': 'status', 'goal_status': 'infeasible'}
            self.history.append(f"Step {step_data['step_number']}: No candidates could be scored")
            return base_agent.AgentInteractionResult(True, step_data)
        
        # Find best action
        best_score, best_action = max(scored_candidates, key=lambda x: x[0])
        step_data['selected_action'] = best_action
        
        print(f"🎯 Selected action (score: {best_score}): {self._format_candidate_action(best_action)}")
        
        # Handle completion status
        if best_action.get('action_type') == 'status':
            print("🏁 Agent indicates task completion")
            action_description = f"Marked task as {best_action.get('goal_status', 'complete')}"
            self.history.append(f"Step {step_data['step_number']}: {action_description}")
            step_data['execution_success'] = True
            return base_agent.AgentInteractionResult(True, step_data)
        
        # Execute the selected action
        try:
            # Clean the action to remove fields not accepted by JSONAction
            cleaned_action = candidate_generation.clean_candidate_for_json_action(best_action)
            json_action_obj = json_action.JSONAction(**cleaned_action)
            self.env.execute_action(json_action_obj)
            
            time.sleep(self.wait_after_action_seconds)
            
            action_description = self._format_candidate_action(best_action)
            self.history.append(f"Step {step_data['step_number']}: {action_description}")
            step_data['execution_success'] = True
            
            print(f"✅ Action executed successfully")
            
        except Exception as e:
            print(f"❌ Failed to execute action: {e}")
            self.history.append(f"Step {step_data['step_number']}: Failed to execute action")
            step_data['execution_success'] = False
        
        return base_agent.AgentInteractionResult(False, step_data)


def create_v_droid_agent(
    env: interface.AsyncEnv,
    model_name: str = "gemma-3-27b-it",
    **kwargs
) -> VDroidAgent:
    """Convenience function to create a V-DROID agent.
    
    Args:
        env: The Android environment.
        model_name: The Gemini model to use as verifier.
        **kwargs: Additional arguments passed to VDroidAgent constructor.
        
    Returns:
        Configured VDroidAgent.
    """
    return VDroidAgent(env=env, model_name=model_name, **kwargs) 