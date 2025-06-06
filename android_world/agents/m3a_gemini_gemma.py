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

"""A Multimodal Autonomous Agent for Android (M3A) using Gemini API with free Gemma 3-27B model."""

import time
from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
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


class M3AGeminiGemma(base_agent.EnvironmentInteractingAgent):
  """M3A using Gemini API with free Gemma 3-27B model."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      model_name: str = "gemma-3-27b-it",
      name: str = 'M3A-Gemini-Gemma',
      wait_after_action_seconds: float = 2.0,
      max_retry: int = 3,
      temperature: float = 0.0,
      top_p: float = 0.95,
      enable_safety_checks: bool = True,
  ):
    """Initializes a M3A Agent using Gemini API with Gemma model.

    Args:
      env: The environment.
      model_name: The Gemini model to use (defaults to free Gemma 3-27B).
      name: The agent name.
      wait_after_action_seconds: Seconds to wait for the screen to stabilize
        after executing an action.
      max_retry: Max number of retries when calling the LLM.
      temperature: Temperature for LLM generation.
      top_p: Top-p sampling parameter.
      enable_safety_checks: Whether to enable Gemini safety checks.
    """
    super().__init__(env, name)
    
    # Initialize the Gemini Gemma LLM wrapper
    self.llm = infer.GeminiGemmaWrapper(
        model_name=model_name,
        max_retry=max_retry,
        temperature=temperature,
        top_p=top_p,
        enable_safety_checks=enable_safety_checks,
    )
    
    self.history = []
    self.additional_guidelines = None
    self.wait_after_action_seconds = wait_after_action_seconds

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
    action_output, is_safe, raw_response = self.llm.predict_mm(
        action_prompt,
        [
            step_data['raw_screenshot'],
            before_screenshot,
        ],
    )

    # Handle Gemini safety classification
    if is_safe == False:  # pylint: disable=singleton-comparison
      action_output = f"""Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}
Action: {{"action_type": "status", "goal_status": "infeasible"}}"""

    if not raw_response:
      raise RuntimeError('Error calling LLM in action selection phase.')
    step_data['action_output'] = action_output
    step_data['action_raw_response'] = raw_response

    reason, action = m3a_utils.parse_reason_action_output(action_output)

    # If the output is not in the right format, add it to step summary which
    # will be passed to next step and return.
    if (not reason) or (not action):
      print('Action prompt output is not in the correct format.')
      step_data['summary'] = (
          'Output for action selection is not in the correct format, so no'
          ' action is performed.'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    print('Action: ' + action)
    print('Reason: ' + reason)
    step_data['action_reason'] = reason

    try:
      converted_action = json_action.JSONAction(
          **agent_utils.extract_json(action),
      )
      step_data['action_output_json'] = converted_action
    except Exception as e:  # pylint: disable=broad-exception-caught
      print('Failed to convert the output to a valid action.')
      print(str(e))
      step_data['summary'] = (
          'Can not parse the output to a valid action. Please make sure to pick'
          ' the action from the list with required parameters (if any) in the'
          ' correct JSON format!'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    action_index = converted_action.index
    num_ui_elements = len(before_ui_elements)
    if (
        converted_action.action_type
        in ['click', 'long_press', 'input_text', 'scroll']
        and action_index is not None
    ):
      if action_index >= num_ui_elements:
        print(
            f'Index out of range, prediction index is {action_index}, but the'
            f' UI element list only has {num_ui_elements} elements.'
        )
        step_data['summary'] = (
            'The parameter index is out of range. Remember the index must be in'
            ' the UI element list!'
        )
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)

      # Add mark to the target element.
      m3a_utils.add_ui_element_mark(
          step_data['raw_screenshot'],
          before_ui_elements[action_index],
          action_index,
          logical_screen_size,
          physical_frame_boundary,
          orientation,
      )

    if converted_action.action_type == 'status':
      if converted_action.goal_status == 'infeasible':
        print('Agent stopped since it thinks mission impossible.')
      step_data['summary'] = 'Agent thinks the request has been completed.'
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(
          True,
          step_data,
      )

    if converted_action.action_type == 'answer':
      print('Agent answered with: ' + converted_action.text)

    try:
      self.env.execute_action(converted_action)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print('Failed to execute action.')
      print(str(e))
      step_data['summary'] = (
          'Can not execute the action, make sure to select the action with'
          ' the required parameters (if any) in the correct JSON format!'
      )
      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    time.sleep(self.wait_after_action_seconds)

    state = self.env.get_state(wait_to_stabilize=False)
    logical_screen_size = self.env.logical_screen_size
    orientation = self.env.orientation
    physical_frame_boundary = self.env.physical_frame_boundary
    after_ui_elements = state.ui_elements
    after_ui_elements_list = _generate_ui_elements_description_list(
        after_ui_elements, logical_screen_size
    )
    after_screenshot = state.pixels.copy()
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

    m3a_utils.add_screenshot_label(
        step_data['before_screenshot_with_som'], 'before'
    )
    m3a_utils.add_screenshot_label(after_screenshot, 'after')
    step_data['after_screenshot_with_som'] = after_screenshot.copy()

    summary_prompt = _summarize_prompt(
        action,
        reason,
        goal,
        before_ui_elements_list,
        after_ui_elements_list,
    )
    summary, is_safe, raw_response = self.llm.predict_mm(
        summary_prompt,
        [
            before_screenshot,
            after_screenshot,
        ],
    )

    # Handle Gemini safety classification
    if is_safe == False:  # pylint: disable=singleton-comparison
      summary = """Summary triggered LLM safety classifier."""

    if not raw_response:
      print(
          'Error calling LLM in summarization phase. This should not happen: '
          f'{summary}'
      )
      step_data['summary'] = (
          'Some error occurred calling LLM during summarization phase: %s'
          % summary
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    step_data['summary_prompt'] = summary_prompt
    step_data['summary'] = f'Action selected: {action}. {summary}'
    print('Summary: ' + summary)
    step_data['summary_raw_response'] = raw_response

    self.history.append(step_data)
    return base_agent.AgentInteractionResult(
        False,
        step_data,
    )


def create_m3a_gemini_gemma_agent(
    env: interface.AsyncEnv,
    model_name: str = "gemma-3-27b-it",
    **kwargs
) -> M3AGeminiGemma:
  """Convenience function to create an M3A agent using Gemini API with Gemma.
  
  Args:
    env: The Android environment.
    model_name: The Gemini model to use.
    **kwargs: Additional arguments passed to M3AGeminiGemma constructor.
    
  Returns:
    Configured M3AGeminiGemma agent.
  """
  return M3AGeminiGemma(env=env, model_name=model_name, **kwargs) 