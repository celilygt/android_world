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
import json
from typing import Any, Optional

from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents.llm_wrappers.gemini_gemma_wrapper import GeminiGemmaWrapper
from android_world.agents.custom import candidate_generation
from android_world.agents.custom import screen_analyzer
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils





class VDroidAgent(base_agent.EnvironmentInteractingAgent):
    """V-DROID Agent using Gemini for verifier-based action selection."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        model_name: str = "auto",
        name: str = 'V-DROID-Agent',
        wait_after_action_seconds: float = 2.0,
        max_retry: int = 3,
        temperature: float = 0.0,
        top_p: float = 0.95,
        enable_safety_checks: bool = True,
        verbose: bool = True,
        enable_batch_verification: bool = True,
        batch_size: int = 0,
        batch_delay_ms: int = 0,
    ):
        """Initializes a V-DROID Agent using Gemini API with smart router as verifier.

        Args:
            env: The environment.
            model_name: The Gemini model to use as verifier. Use "auto" for smart routing (default).
            name: The agent name.
            wait_after_action_seconds: Seconds to wait for the screen to stabilize
                after executing an action.
            max_retry: Max number of retries when calling the LLM.
            temperature: Temperature for LLM generation.
            top_p: Top-p sampling parameter.
            enable_safety_checks: Whether to enable Gemini safety checks.
            verbose: Whether to enable verbose output during agent execution.
            enable_batch_verification: Whether to use batch processing for candidate verification.
            batch_size: Maximum candidates to process in one batch (0 = no limit, recommended).
            batch_delay_ms: Optional additional delay in ms (0 = rely on router's smart delays).
        """
        super().__init__(env, name)
        
        # Always use smart router for optimal model selection, ignoring specific model requests
        router_model_name = "auto"
        if model_name != "auto" and verbose:
            print(f"🔄 V-DROID Agent: Ignoring specific model '{model_name}' and using smart router instead")
        
        # Initialize the Gemini Gemma LLM wrapper with smart routing
        self.llm = GeminiGemmaWrapper(
            model_name=router_model_name,
            max_retry=max_retry,
            temperature=temperature,
            top_p=top_p,
            enable_safety_checks=enable_safety_checks,
            verbose=verbose,
        )
        
        self.history = []
        self.additional_guidelines = None
        self.wait_after_action_seconds = wait_after_action_seconds
        self.verbose = verbose
        
        # Batch processing settings
        self.enable_batch_verification = enable_batch_verification
        self.batch_size = batch_size
        self.batch_delay_ms = batch_delay_ms

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
        candidates: list[dict],
        screen_contexts: list[str]
    ) -> list[tuple[float, dict]]:
        """Scores candidate actions using a single batch prompt to the Gemini LLM.

        This method constructs a single prompt containing all candidate actions and asks
        the LLM to return a JSON object with a score for each candidate. This
        approach is designed to be efficient and avoid rate-limiting issues.

        Args:
            goal: The task goal.
            history: List of previous action descriptions.
            ui_elements: List of UIElement objects from the current state.
            candidates: List of candidate action dictionaries.
            screen_contexts: List of detected screen contexts.

        Returns:
            A list of (score, candidate_action) tuples.
        """
        if not candidates:
            return []

        ui_description = self._generate_ui_elements_description(ui_elements)
        history_str = "\n".join(history) if history else "No actions taken yet."

        # Create a numbered list of candidate actions for the prompt
        candidate_lines = []
        for i, candidate in enumerate(candidates):
            action_desc = self._format_candidate_action(candidate)
            candidate_lines.append(f"{i + 1}. {action_desc}")
        candidates_str = "\n".join(candidate_lines)

        # Construct the single prompt for batch verification
        prompt = f"""You are controlling an Android device to complete this task: {goal}

History of actions:
{history_str}

Current screen elements:
{ui_description}

I have the following candidate actions:
{candidates_str}

Please evaluate these actions and return a JSON object where keys are the
candidate numbers and values are a score from 0.0 to 1.0, indicating
how helpful the action is for achieving the goal.
A score of 1.0 is for a clearly correct action.
A score of 0.0 is for a clearly incorrect or irrelevant action.
Provide only the JSON object in your response.
"""

        if self.verbose:
            print(f"🔍 Verifying {len(candidates)} candidates in a single batch...")

        try:
            response, is_safe, _ = self.llm.predict(prompt)
            if not is_safe or not response:
                print("❌ Verification failed due to safety settings or empty response.")
                return [(0.0, c) for c in candidates]

            # Extract the JSON object from the response
            json_str = response[response.find('{'):response.rfind('}') + 1]
            scores = json.loads(json_str)
            
            scored_candidates = []
            for i, candidate in enumerate(candidates):
                score = scores.get(str(i + 1), 0.0)
                scored_candidates.append((float(score), candidate))

            if self.verbose:
                for score, candidate in scored_candidates:
                    if score > 0:
                        print(f"  ✅ {self._format_candidate_action(candidate)}: {score}")
            
            return scored_candidates

        except (json.JSONDecodeError, TypeError) as e:
            print(f"❌ Error parsing LLM response: {e}")
            return [(0.0, c) for c in candidates]
        except Exception as e:
            print(f"❌ An unexpected error occurred during verification: {e}")
            return [(0.0, c) for c in candidates]

    

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

    def _detect_required_app_from_goal(self, goal: str) -> list[dict]:
        """Detect which app needs to be opened based on the goal text (like M3A logic).
        
        Args:
            goal: The task goal string.
            
        Returns:
            List of candidate actions for opening the required app.
        """
        app_candidates = []
        goal_lower = goal.lower()
        
        # App name mappings from goal text to actual app names
        app_mappings = {
            # Audio/Recording apps
            'audio recorder': ['Audio Recorder', 'com.dimowner.audiorecorder'],
            'record audio': ['Audio Recorder', 'com.dimowner.audiorecorder'],
            'recording': ['Audio Recorder', 'com.dimowner.audiorecorder'],
            
            # Contact apps
            'contact': ['Contacts', 'com.google.android.contacts', 'com.android.contacts'],
            
            # Messaging apps
            'message': ['Messages', 'com.google.android.apps.messaging', 'com.android.mms'],
            'sms': ['Messages', 'com.google.android.apps.messaging'],
            'text': ['Messages', 'com.google.android.apps.messaging'],
            
            # Email apps
            'email': ['Gmail', 'com.google.android.gm'],
            'gmail': ['Gmail', 'com.google.android.gm'],
            
            # Calendar apps
            'calendar': ['Calendar', 'com.google.android.calendar'],
            'schedule': ['Calendar', 'com.google.android.calendar'],
            'appointment': ['Calendar', 'com.google.android.calendar'],
            
            # Browser apps
            'browser': ['Chrome', 'com.android.chrome'],
            'chrome': ['Chrome', 'com.android.chrome'],
            'web': ['Chrome', 'com.android.chrome'],
            
            # Phone apps
            'call': ['Phone', 'com.google.android.dialer'],
            'dial': ['Phone', 'com.google.android.dialer'],
            'phone': ['Phone', 'com.google.android.dialer'],
            
            # Clock/Timer apps
            'timer': ['Clock', 'com.google.android.deskclock'],
            'alarm': ['Clock', 'com.google.android.deskclock'],
            'stopwatch': ['Clock', 'com.google.android.deskclock'],
            
            # Settings
            'setting': ['Settings', 'com.android.settings'],
            'wifi': ['Settings', 'com.android.settings'],
            'bluetooth': ['Settings', 'com.android.settings'],
            
            # Camera
            'camera': ['Camera', 'com.google.android.GoogleCamera'],
            'photo': ['Camera', 'com.google.android.GoogleCamera'],
            'picture': ['Camera', 'com.google.android.GoogleCamera'],
            
            # File manager
            'file': ['Files', 'com.google.android.apps.nbu.files'],
            'folder': ['Files', 'com.google.android.apps.nbu.files'],
        }
        
        # Find the best matching app
        for keyword, app_names in app_mappings.items():
            if keyword in goal_lower:
                for app_name in app_names:
                    app_candidates.append({
                        'action_type': 'open_app', 
                        'app_name': app_name,
                        'text': f'Open {app_name} app'
                    })
                break  # Use first match to avoid duplicates
        
        if self.verbose and app_candidates:
            print(f"🎯 Detected required app from goal: {[c['app_name'] for c in app_candidates]}")
        
        return app_candidates

    def _format_candidate_action(self, candidate: dict) -> str:
        """Format a candidate action dictionary into a readable description."""
        action_type = candidate.get('action_type', 'unknown')
        
        if action_type == 'click':
            # Show the actual text content for visual clicks
            if 'index' in candidate:
                text = candidate.get('text', 'no text')
                return f"Click UI element '{text}'"
            else:
                # Visual click - show the detected text
                text = candidate.get('text', 'unknown location')
                coords = f"({candidate.get('x', '?')}, {candidate.get('y', '?')})"
                return f"Click on '{text}' at {coords}"
        elif action_type == 'input_text':
            text = candidate.get('text', '')
            target = candidate.get('index', 'text field')
            return f"Input '{text}' into {target}"
        elif action_type == 'scroll':
            direction = candidate.get('direction', 'unknown')
            return f"Scroll {direction}"
        elif action_type == 'long_press':
            text = candidate.get('text', 'element')
            return f"Long press on '{text}'"
        elif action_type == 'navigate_home':
            return "Navigate to home screen"
        elif action_type == 'navigate_back':
            return "Navigate back"
        elif action_type == 'status':
            status = candidate.get('goal_status', 'unknown')
            return f"Mark task as {status}"
        elif action_type == 'wait':
            return "Wait for screen to update"
        elif action_type == 'open_app':
            app_name = candidate.get('app_name', 'app')
            return f"Open {app_name} app"
        else:
            return f"Perform {action_type} action"

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """Perform a step using the V-DROID verifier-based approach with Gemini.
        
        The V-DROID approach:
        1. Get current state
        2. Generate candidate actions
        3. Score candidates with Gemini verifier
        4. Execute highest-scoring action
        5. Update history
        
        Args:
            goal: The task goal.
            
        Returns:
            AgentInteractionResult with execution details.
        """
        step_data = {
            'v_droid_step': len(self.history) + 1,  # Renamed to avoid collision with constants.STEP_NUMBER
            'goal': goal,
            'candidates': [],
            'scores': [],
            'selected_action': None,
            'execution_success': False,
        }
        
        print(f'----------V-DROID Gemini Step {step_data["v_droid_step"]}')
        
        # Get current state
        state = self.get_post_transition_state()
        screen_contexts = screen_analyzer.analyze_screen_for_context(state.ui_elements)

        # Smart app opening based on goal analysis (like M3A)
        if step_data["v_droid_step"] == 1:
            app_candidates = self._detect_required_app_from_goal(goal)
            # Generate other candidates as fallback
            other_candidates = candidate_generation.generate_candidate_actions(state, goal)
            candidates = app_candidates + other_candidates
        else:
            # Generate candidate actions using the proper V-DROID candidate generation
            candidates = candidate_generation.generate_candidate_actions(state, goal)
        
        step_data['candidates'] = candidates
        
        if self.verbose:
            print(f"📋 Generated {len(candidates)} candidate actions")
            # Show first 10 for readability, but indicate total count
            if len(candidates) > 10:
                display_candidates = candidates[:10]
                print(f"First 10 candidates (out of {len(candidates)} total):")
                for i, candidate in enumerate(display_candidates):
                    action_desc = self._format_candidate_action(candidate)
                    print(f"  {i+1}. {action_desc}")
                print(f"  ... and {len(candidates) - 10} more candidates")
            else:
                candidate_summary = candidate_generation.format_candidates_for_display(candidates)
                print(candidate_summary)
        
        if not candidates:
            print("❌ No candidate actions generated")
            step_data['selected_action'] = {'action_type': 'status', 'goal_status': 'infeasible'}
            self.history.append(f"Step {step_data['v_droid_step']}: No valid candidates found")
            return base_agent.AgentInteractionResult(True, step_data)
        
        # Score candidates using Gemini verifier
        scored_candidates = self._score_candidate_actions(
            goal,
            self.history,
            state.ui_elements,
            candidates,
            screen_contexts
        )
        step_data['scores'] = [score for score, _ in scored_candidates]
        
        # Select highest-scoring candidate
        if not scored_candidates:
            print("❌ No candidates were scored")
            step_data['selected_action'] = {'action_type': 'status', 'goal_status': 'infeasible'}
            self.history.append(f"Step {step_data['v_droid_step']}: No candidates could be scored")
            return base_agent.AgentInteractionResult(True, step_data)
        
        # Find best action
        best_score, best_action = max(scored_candidates, key=lambda x: x[0])
        step_data['selected_action'] = best_action
        
        print(f"🎯 Selected action (score: {best_score}): {self._format_candidate_action(best_action)}")
        
        # Handle completion status
        if best_action.get('action_type') == 'status':
            print("🏁 Agent indicates task completion")
            action_description = f"Marked task as {best_action.get('goal_status', 'complete')}"
            self.history.append(f"Step {step_data['v_droid_step']}: {action_description}")
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
            self.history.append(f"Step {step_data['v_droid_step']}: {action_description}")
            step_data['execution_success'] = True
            
            print(f"✅ Action executed successfully")
            
        except Exception as e:
            print(f"❌ Failed to execute action: {e}")
            self.history.append(f"Step {step_data['v_droid_step']}: Failed to execute action")
            step_data['execution_success'] = False
        
        return base_agent.AgentInteractionResult(False, step_data)


def create_v_droid_agent(
    env: interface.AsyncEnv,
    model_name: str = "auto",
    **kwargs
) -> VDroidAgent:
    """Convenience function to create a V-DROID agent.
    
    Args:
        env: The Android environment.
        model_name: The Gemini model to use as verifier.
        **kwargs: Additional arguments passed to VDroidAgent constructor.
            Supports batch processing options:
            - enable_batch_verification: Whether to use batch processing (default: True)
            - batch_size: Maximum candidates per batch (default: 0 = no limit)
            - batch_delay_ms: Optional additional delay in ms (default: 0 = router handles delays)
        
    Returns:
        Configured VDroidAgent.
    """
    return VDroidAgent(env=env, model_name=model_name, **kwargs) 