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

"""A Multimodal Autonomous Agent for Android (M3A) using Ollama local models."""

import time
from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents.llm_wrappers import OllamaWrapper
from android_world.agents import m3a_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils

# Import all the same prompts and constants from the original M3A
from android_world.agents.m3a import (
    PROMPT_PREFIX,
    GUIDANCE,
    ACTION_SELECTION_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    _generate_ui_element_description,
    _generate_ui_elements_description_list,
    _action_selection_prompt,
    _summarize_prompt,
)


class M3AOllama(base_agent.EnvironmentInteractingAgent):
  """M3A using Ollama local models (like Gemma 3-4B) for privacy and cost efficiency."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      model_name: str = "gemma3:4b",
      name: str = 'M3A-Ollama',
      wait_after_action_seconds: float = 2.0,
      max_retry: int = 3,
      temperature: float = 0.0,
      host: str = "localhost",
      port: int = 11434,
      verbose: bool = True,
  ):
    """Initializes a M3A Agent using Ollama.

    Args:
      env: The environment.
      model_name: The Ollama model to use (defaults to gemma3:4b).
      name: The agent name.
      wait_after_action_seconds: Seconds to wait for the screen to stabilize
        after executing an action.
      max_retry: Max number of retries when calling the LLM.
      temperature: Temperature for LLM generation.
      host: Ollama host (default: localhost).
      port: Ollama port (default: 11434).
      verbose: Whether to enable verbose output during agent execution.
    """
    super().__init__(env, name)
    
    # Initialize the Ollama LLM wrapper
    self.llm = OllamaWrapper(
        model_name=model_name,
        max_retry=max_retry,
        temperature=temperature,
        host=host,
        port=port,
    )
    
    # Check if model is available and pull if necessary
    if not self.llm.is_model_available():
      print(f"Model {model_name} not found locally. Attempting to pull...")
      if not self.llm.pull_model():
        raise RuntimeError(f"Failed to pull model {model_name}. Please check your Ollama installation.")
    
    self.history = []
    self.additional_guidelines = None
    self.wait_after_action_seconds = wait_after_action_seconds
    self.verbose = verbose

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    # Hide the coordinates on screen which might affect the vision model.
    self.env.hide_automation_ui()
    self.history = []

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data = {
        'raw_screenshot': None,
        'before_screenshot_with_som': None,
        'before_ui_elements': [],
        'after_screenshot_with_som': None,
        'action_prompt': None,
        'action_output': None,
        'action_output_json': None,
        'action_reason': None,
        'action_raw_response': None,
        'summary_prompt': None,
        'summary': None,
        'summary_raw_response': None,
    }
    print('----------step ' + str(len(self.history) + 1))

    state = self.get_post_transition_state()
    logical_screen_size = self.env.logical_screen_size
    orientation = self.env.orientation
    physical_frame_boundary = self.env.physical_frame_boundary

    before_ui_elements = state.ui_elements
    step_data['before_ui_elements'] = before_ui_elements
    before_ui_elements_list = _generate_ui_elements_description_list(
        before_ui_elements, logical_screen_size
    )
    step_data['raw_screenshot'] = state.pixels.copy()
    before_screenshot = state.pixels.copy()
    for index, ui_element in enumerate(before_ui_elements):
      if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
        m3a_utils.add_ui_element_mark(
            before_screenshot,
            ui_element,
            index,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )
    step_data['before_screenshot_with_som'] = before_screenshot.copy()

    action_prompt = _action_selection_prompt(
        goal,
        [
            'Step ' + str(i + 1) + '- ' + step_info['summary']
            for i, step_info in enumerate(self.history)
        ],
        before_ui_elements_list,
        self.additional_guidelines,
    )
    step_data['action_prompt'] = action_prompt
    
    if self.verbose:
      print(f"🤖 LLM Call: Requesting action for goal: {goal[:100]}{'...' if len(goal) > 100 else ''}")
      
    action_output, is_safe, raw_response = self.llm.predict_mm(
        action_prompt,
        [
            step_data['raw_screenshot'],
            before_screenshot,
        ],
    )

    if self.verbose:
      print(f"🔍 LLM Call Result: action_output={action_output is not None}, is_safe={is_safe}, raw_response={raw_response is not None}")

    # Handle safety (local models are assumed safe)
    if is_safe == False:  # pylint: disable=singleton-comparison
      action_output = f"""Reason: Safety check triggered
Action: {{"action_type": "status", "goal_status": "infeasible"}}"""

    if not raw_response:
      if self.verbose:
        print(f"❌ LLM call failed - raw_response is None")
        print(f"   action_output: {action_output}")
        print(f"   is_safe: {is_safe}")
      raise RuntimeError('Error calling LLM in action selection phase.')
    
    if self.verbose:
      print(f"✅ LLM call successful")
      
    step_data['action_output'] = action_output
    step_data['action_raw_response'] = raw_response

    reason, action = m3a_utils.parse_reason_action_output(action_output)

    # Always show action and reason output
    print(f"\n🎯 STEP {len(self.history) + 1} RESULTS:")
    print(f"Action: {action}")
    print(f"Reason: {reason}")

    # If the output is not in the right format, add it to step summary which
    # will be passed to next step and return.
    if (not reason) or (not action):
      print('❌ Action prompt output is not in the correct format.')
      step_data['summary'] = (
          'Output for action selection is not in the correct format, so no'
          ' action is performed.'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )
    
    if self.verbose:
      print(f"Raw LLM Response: {raw_response}")

    step_data['action_reason'] = reason

    # Parse output.
    try:
      action_json = agent_utils.extract_json(action)
    except ValueError as e:
      print(f'Failed to parse action: {action}. Error: {e}')
      step_data['summary'] = 'Failed to parse action.'
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    step_data['action_output_json'] = action_json

    # Perform action.
    try:
      action_to_perform = json_action.JSONAction(action_json)
    except ValueError as e:
      print(f'Failed to create action: {action_json}. Error: {e}')
      step_data['summary'] = 'Failed to create action.'
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    if action_to_perform.is_check_status():
      goal_status = action_to_perform.get_goal_status()
      if goal_status == 'completed':
        summary = 'The goal is completed!'
      else:
        summary = (
            'The goal is set to '
            + goal_status
            + '. Reason: '
            + reason
        )
      step_data['summary'] = summary
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(
          goal_status == 'completed',
          step_data,
      )

    # Execute the action
    result = self.env.execute_action(action_to_perform)
    if self.verbose:
      print(f"Action execution result: {result}")

    # Wait to stabilize
    time.sleep(self.wait_after_action_seconds)

    # Capture state after the action
    after_state = self.get_post_transition_state()
    after_ui_elements = after_state.ui_elements
    after_screenshot = after_state.pixels.copy()
    for index, ui_element in enumerate(after_ui_elements):
      if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
        m3a_utils.add_ui_element_mark(
            after_screenshot,
            ui_element,
            index,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )
    step_data['after_screenshot_with_som'] = after_screenshot.copy()

    # Summarize.
    if action_to_perform.is_drag():
      action_description = 'swipe/drag from ' + str(
          action_to_perform.get_start_coordinate()
      ) + ' to ' + str(action_to_perform.get_end_coordinate())
    elif action_to_perform.is_dual_point():
      action_description = 'dual point action'
    else:
      coordinate = action_to_perform.get_coordinate()
      action_description = (
          action_to_perform.get_action_type()
          + ' at coordinate ('
          + str(coordinate[0])
          + ', '
          + str(coordinate[1])
          + ')'
      )
      if action_to_perform.is_type():
        action_description = (
            action_description
            + ' with text: "'
            + action_to_perform.get_text()
            + '"'
        )

    # Get summary with images
    summary_prompt = _summarize_prompt(
        goal,
        reason,
        action_description,
        before_ui_elements_list,
        _generate_ui_elements_description_list(
            after_ui_elements, logical_screen_size
        ),
    )
    step_data['summary_prompt'] = summary_prompt

    if self.verbose:
      print(f"🤖 LLM Call: Requesting summary")

    summary_output, summary_is_safe, summary_raw_response = self.llm.predict_mm(
        summary_prompt, [before_screenshot, after_screenshot]
    )

    if self.verbose:
      print(f"🔍 Summary Call Result: summary_output={summary_output is not None}, is_safe={summary_is_safe}, raw_response={summary_raw_response is not None}")

    if not summary_raw_response:
      if self.verbose:
        print(f"❌ Summary LLM call failed")
      step_data['summary'] = f'Performed {action_description}.'
    else:
      if self.verbose:
        print(f"Raw Summary LLM Response: {summary_raw_response}")
      step_data['summary'] = summary_output
    
    step_data['summary_raw_response'] = summary_raw_response

    self.history.append(step_data)

    return base_agent.AgentInteractionResult(
        False,
        step_data,
    )


def create_m3a_ollama_agent(
    env: interface.AsyncEnv,
    model_name: str = "gemma3:4b",
    **kwargs
) -> M3AOllama:
  """Creates an M3A agent using Ollama."""
  return M3AOllama(env=env, model_name=model_name, **kwargs) 