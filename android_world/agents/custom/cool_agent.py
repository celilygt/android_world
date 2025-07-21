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

"""A Multimodal Autonomous Agent for Android (M3A)."""

import os
import time
import json
import hashlib
import re
from collections import defaultdict, deque
from pathlib import Path
import cv2
from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents.llm_wrappers import gemini_gemma_wrapper
from android_world.agents.custom import cool_agent_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils

# MODIFICATION: Added a detailed list of openable applications to the prompt.
# This provides the agent with explicit knowledge of which apps can be opened directly.
# The 'app_name' values are chosen to be compatible with `adb_utils.launch_app`.
APP_LIST_GUIDANCE = (
    '- Open an app (nothing will happen if the app is not'
    ' installed): `{{\\"action_type\\": \\"open_app\\", \\"app_name\\": <name>}}`\n'
    '  You have the ability to directly open the following applications. If the'
    ' task description suggests using one of these apps, you should use this'
    ' action as your first step. Use the exact "app_name" from this list.\n'
    '  * google chrome: For browsing websites. (app_name: "chrome")\n'
    '  * google chat: For messaging and collaboration. (app_name: "google chat")\n'
    '  * settings: For changing device settings. (app_name: "settings")\n'
    '  * youtube: For watching videos. (app_name: "youtube")\n'
    '  * google play: For downloading apps and games. (app_name: "google play")\n'
    '  * gmail: For sending and receiving emails. (app_name: "gmail")\n'
    '  * google maps: For navigation and finding locations. (app_name: "google maps")\n'
    '  * google photos: For viewing and managing photos. (app_name: "google photos")\n'
    '  * google calendar: For managing schedules and events. (app_name: "google calendar")\n'
    '  * camera: For taking photos and videos. (app_name: "camera")\n'
    '  * audio recorder: For recording audio. (app_name: "audio recorder")\n'
    '  * google drive: For storing and sharing files. (app_name: "google drive")\n'
    '  * google keep: For taking notes and creating lists. (app_name: "google keep")\n'
    '  * clock: For setting alarms and timers. (app_name: "clock")\n'
    '  * contacts: For managing contacts. (app_name: "contacts")\n'
    '  * files: For managing files and folders. (app_name: "files")\n'
    '  * markor: For writing and editing notes in Markdown format. (app_name: "markor")\n'
    '  * clipper: For managing the clipboard. (app_name: "clipper")\n'
    '  * messages: For sending and receiving SMS and MMS messages. (app_name: "messages")\n'
    '  * dialer: For making phone calls. (app_name: "dialer")\n'
    '  * simple calendar pro: For managing schedules, events, and appointments. (app_name: "simple calendar pro")\n'
    '  * simple gallery pro: For viewing and managing photos and videos. (app_name: "simple gallery pro")\n'
    '  * miniwob: A specialized app for web-based tasks. (app_name: "miniwob")\n'
    '  * simple draw pro: For drawing and sketching. (app_name: "simple draw pro")\n'
    '  * pro expense: For tracking expenses. (app_name: "pro expense")\n'
    '  * broccoli app: A recipe application. (app_name: "broccoli")\n'
    '  * osmand: For maps and navigation. (app_name: "osmand")\n'
    '  * tasks: For managing to-do lists. (app_name: "tasks")\n'
    '  * open tracks sports tracker: For tracking sports activities. (app_name: "open tracks")\n'
    '  * joplin: For taking notes and creating to-do lists. (app_name: "joplin")\n'
    '  * vlc: For playing video files. (app_name: "vlc")\n'
    '  * retro music: For listening to music. (app_name: "retro music")\n'
)

PROMPT_PREFIX = (
        'You are an agent who can operate an Android phone on behalf of a user.'
        ' Based on user\'s goal/request, you may\n'
        '- Answer back if the request/goal is a question (or a chat message),'
        ' like user asks "What is my schedule for today?".\n'
        '- Complete some tasks described in the requests/goals by'
        ' performing actions (step by step) on the phone.\n\n'
        'When given a user request, you will try to complete it step by step.'
        ' At each step, you will be given the current screenshot (including the'
        ' original screenshot and the same screenshot with bounding'
        ' boxes and numeric indexes added to some UI elements) and a history of'
        ' what you have done (in text). Based on these pieces of information and'
        ' the goal, you must choose to perform one of the'
        ' action in the following list (action description followed by the JSON'
        ' format) by outputing the action in the correct JSON format.\n'
        '- If you think the task has been completed, finish the task by using the'
        ' status action with complete as goal_status:'
        ' `{{\\"action_type\\": \\"status\\", \\"goal_status\\": \\"complete\\"}}`\n'
        "- If you think the task is not feasible (including cases like you don't"
        ' have enough information or can not perform some necessary actions),'
        ' finish by using the `status` action with infeasible as goal_status:'
        ' `{{\\"action_type\\": \\"status\\", \\"goal_status\\": \\"infeasible\\"}}`\n'
        "- Answer user\'s question:"
        ' `{{\\"action_type\\": \\"answer\\", \\"text\\": \\"<answer_text>\\"}}\n'
        '- Click/tap on an element on the screen. We have added marks (bounding'
        ' boxes with numeric indexes on their TOP LEFT corner) to most of the UI'
        ' elements in the screenshot, use the numeric index to indicate which'
        ' element you want to click:'
        ' `{{\\"action_type\\": \\"click\\", \\"index\\": <target_index>}}`.\n'
        '- Long press on an element on the screen, similar with the click action'
        ' above, use the numeric label on the bounding box to indicate which'
        ' element you want to long press:'
        ' `{{\\"action_type\\": \\"long_press\\", \\"index\\": <target_index>}}`.\n'
        '- Type text into a text field (this action contains clicking the text'
        ' field, typing in the text and pressing the enter, so no need to click on'
        ' the target field to start), use the numeric label'
        ' on the bounding box to indicate the target text field:'
        ' `{{\\"action_type\\": \\"input_text\\", \\"text\\": <text_input>,'
        ' \\"index\\": <target_index>}}`\n'
        '- Press the Enter key: `{{\\"action_type\\": \\"keyboard_enter\\"}}`\n'
        '- Navigate to the home screen: `{{\\"action_type\\": \\"navigate_home\\"}}`\n'
        '- Navigate back: `{{\\"action_type\\": \\"navigate_back\\"}}`\n'
        '- Scroll the screen or a scrollable UI element in one of the four'
        ' directions, use the same numeric index as above if you want to scroll a'
        ' specific UI element, leave it empty when scroll the whole screen:'
        ' `{{\\"action_type\\": \\"scroll\\", \\"direction\\": <up, down, left, right>,'
        ' \\"index\\": <optional_target_index>}}`\n'
        + APP_LIST_GUIDANCE
        + '- Wait for the screen to update: `{{\\"action_type\\": \\"wait\\"}}`\n'
)

GUIDANCE = (
    'Here are some useful guidelines you need to follow:\n'
    'General:\n'
    '- Usually there will be multiple ways to complete a task, pick the'
    ' easiest one. Also when something does not work as expected (due'
    ' to various reasons), sometimes a simple retry can solve the problem,'
    " but if it doesn't (you can see that from the history),"
    ' SWITCH to other solutions.\n'
    '- Sometimes you may need to navigate the phone to gather information'
    ' needed to complete the task, for example if user asks'
    ' "what is my schedule tomorrow", then you may want to open the calendar'
    ' app (using the `open_app` action), look up information there, answer'
    " user's question (using the `answer` action) and finish (using"
    ' the `status` action with complete as goal_status).\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish!'
    ' Merely displaying the answer on the screen is NOT sufficient (unless'
    ' the goal is something like "show me ...").\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n"
    'Action Related:\n'
    '- Use the `open_app` action whenever possible to open an app. Do not use the'
    ' app drawer unless `open_app` has failed.\n'
    '- **SETUP SCREENS**: When setting up new apps (like Chrome or an audio recorder), you'
    ' will often see setup screens. Your goal is to get to the main app screen. Decline'
    ' optional features like "sync" or "add account" unless the task requires them. If you'
    ' cannot perform a primary action (like typing a custom filename) on a setup screen,'
    ' look for an "Apply", "Done", or "OK" button to exit the setup. You can often perform'
    ' the action later on the main screen.\n'
    '- **CRITICAL REASONING**: After an action, carefully observe the new screen to see what'
    ' *actually* happened. Do not assume the action did what you expected. For example,'
    ' clicking a "Record" button might start a timer, not open a "Save" dialog.\n'
    '- **AVOID LOOPS**: If the history says your last action was ineffective, the screen'
    ' did not change, or a loop was detected, you MUST try a different action or strategy.'
    ' DO NOT REPEAT the failed action. Common alternative strategies include trying a'
    ' different button, using `navigate_back`, or scrolling.\n'
    '- Use the `input_text` action whenever you want to type'
    ' something (including password) instead of clicking characters on the'
    ' keyboard one by one. Sometimes there is some default text in the text'
    ' field you want to type in, remember to delete them before typing.\n'
    '- For `click`, `long_press` and `input_text`, the index parameter you'
    ' pick must be VISIBLE in the screenshot and also in the UI element'
    ' list given to you (some elements in the list may NOT be visible on'
    ' the screen so you can not interact with them).\n'
    '- Consider exploring the screen by using the `scroll`'
    ' action with different directions to reveal additional content.\n'
    '- The direction parameter for the `scroll` action can be confusing'
    " sometimes as it's opposite to swipe, for example, to view content at the"
    ' bottom, the `scroll` direction should be set to "down". It has been'
    ' observed that you have difficulties in choosing the correct direction, so'
    ' if one does not work, try the opposite as well.\n'
    'Text Related Operations:\n'
    '- Normally to select certain text on the screen: <i> Enter text selection'
    ' mode by long pressing the area where the text is, then some of the words'
    ' near the long press point will be selected (highlighted with two pointers'
    ' indicating the range) and usually a text selection bar will also appear'
    ' with options like `copy`, `paste`, `select all`, etc.'
    ' <ii> Select the exact text you need. Usually the text selected from the'
    ' previous step is NOT the one you want, you need to adjust the'
    ' range by dragging the two pointers. If you want to select all text in'
    ' the text field, simply click the `select all` button in the bar.\n'
    "- At this point, you don't have the ability to drag something around the"
    ' screen, so in general you can not select arbitrary text.\n'
    '- To delete some text: the most traditional way is to place the cursor'
    ' at the right place and use the backspace button in the keyboard to'
    ' delete the characters one by one (can long press the backspace to'
    ' accelerate if there are many to delete). Another approach is to first'
    ' select the text you want to delete, then click the backspace button'
    ' in the keyboard.\n'
    '- To copy some text: first select the exact text you want to copy, which'
    ' usually also brings up the text selection bar, then click the `copy`'
    ' button in bar.\n'
    '- To paste text into a text box, first long press the'
    ' text box, then usually the text selection bar will appear with a'
    ' `paste` button in it.\n'
    '- When typing into a text field, sometimes an auto-complete dropdown'
    ' list will appear. This usually indicating this is a enum field and you'
    ' should try to select the best match by clicking the corresponding one'
    ' in the list.\n'
)

ACTION_SELECTION_PROMPT_TEMPLATE = (
        PROMPT_PREFIX
        + '\nThe current user goal/request is: {goal}\n\n'
          'Here is a history of what you have done so far:\n{history}\n\n'
          'The current screenshot and the same screenshot with bounding boxes'
          ' and labels added are also given to you.\n'
          'Here is a list of detailed'
          ' information for some of the UI elements (notice that some elements in'
          ' this list may not be visible in the current screen and so you can not'
          ' interact with it, can try to scroll the screen to reveal it first),'
          ' the numeric indexes are'
          ' consistent with the ones in the labeled screenshot:\n{ui_elements}\n'
        + GUIDANCE
        + '{additional_guidelines}'
        + '\nNow output an action from the above list in the correct JSON format,'
          ' following the reason why you do that. Your answer should look like:\n'
          'Reason: ...\nAction: {{\"action_type\":...}}\n\n'
          'Your Answer:\n'
)

SUMMARY_PROMPT_TEMPLATE = (
    'You are an agent summarizing the last step taken on an Android phone. '
    'Your overall goal is: {goal}\n\n'
    'You will be given the screenshot before you performed the action, the '
    'action you chose (with your reasoning), and the screenshot after the '
    'action was performed.\n'
    'This is the action you picked: {action}\n'
    'Based on the reason: {reason}\n\n'
    "By comparing the two screenshots (and UI element lists), provide a brief, "
    "critical, single-line summary of the action's **outcome**. "
    "State whether the outcome moved you closer to the goal. If the action "
    "resulted in completing the goal, state that clearly.\n\n"
    'Here is the list for the before screenshot:\n{before_elements}\n'
    'Here is the list for the after screenshot:\n{after_elements}\n\n'
    '--- RULES ---\n'
    "1. Be brief and critical. Focus on the *outcome* of the action.\n"
    "2. If the action did not work or the UI did not change as expected, "
    "state that clearly.\n"
    "3. If the action successfully completed the overall goal, say so.\n"
    "4. Your response MUST be a natural language sentence and in a single line.\n"
    "5. Do NOT include JSON or actions in your summary.\n\n"
    'Summary of this step: '
)


def _generate_ui_element_description(
        ui_element: representation_utils.UIElement, index: int
) -> str:
    """Generate a description for a given UI element with important information.

      Args:
        ui_element: UI elements for the current screen.
        index: The numeric index for the UI element.

      Returns:
        The description for the UI element.
      """
    element_description = f'UI element {index}: {{\"index\": {index}, '
    if ui_element.text:
        element_description += f'"text": "{ui_element.text}", '
    if ui_element.content_description:
        element_description += (
            f'"content_description": "{ui_element.content_description}", '
        )
    if ui_element.hint_text:
        element_description += f'"hint_text": "{ui_element.hint_text}", '
    if ui_element.tooltip:
        element_description += f'"tooltip": "{ui_element.tooltip}", '
    element_description += (
        f'"is_clickable": {"True" if ui_element.is_clickable else "False"}, '
    )
    element_description += (
        '"is_long_clickable":'
        f' {"True" if ui_element.is_long_clickable else "False"}, '
    )
    element_description += (
        f'"is_editable": {"True" if ui_element.is_editable else "False"}, '
    )
    if ui_element.is_scrollable:
        element_description += '"is_scrollable": True, '
    if ui_element.is_focusable:
        element_description += '"is_focusable": True, '
    element_description += (
        f'"is_selected": {"True" if ui_element.is_selected else "False"}, '
    )
    element_description += (
        f'"is_checked": {"True" if ui_element.is_checked else "False"}, '
    )
    return element_description[:-2] + '}'


def _generate_ui_elements_description_list(
        ui_elements: list[representation_utils.UIElement],
        screen_width_height_px: tuple[int, int],
) -> str:
    """Generate concise information for a list of UIElement.

      Args:
        ui_elements: UI elements for the current screen.
        screen_width_height_px: The height and width of the screen in pixels.

      Returns:
        Concise information for each UIElement.
      """
    tree_info = ''
    for index, ui_element in enumerate(ui_elements):
        if cool_agent_utils.validate_ui_element(ui_element, screen_width_height_px):
            tree_info += _generate_ui_element_description(ui_element, index) + '\n'
    return tree_info


def _action_selection_prompt(
        goal: str,
        history: list[str],
        ui_elements: str,
        additional_guidelines: list[str] | None = None,
) -> str:
    """Generate the prompt for the action selection.

      Args:
        goal: The current goal.
        history: Summaries for previous steps.
        ui_elements: A list of descriptions for the UI elements.
        additional_guidelines: Task specific guidelines.

      Returns:
        The text prompt for action selection that will be sent to gpt4v.
      """
    if history:
        history = '\n'.join(history)
    else:
        history = 'You just started, no action has been performed yet.'

    extra_guidelines = ''
    if additional_guidelines:
        extra_guidelines = 'For The Current Task:\n'
        for guideline in additional_guidelines:
            extra_guidelines += f'- {guideline}\n'

    return ACTION_SELECTION_PROMPT_TEMPLATE.format(
        goal=goal,
        history=history,
        ui_elements=ui_elements if ui_elements else 'Not available',
        additional_guidelines=extra_guidelines,
    )


def _summarize_prompt(
        action: str,
        reason: str,
        goal: str,
        before_elements: str,
        after_elements: str,
) -> str:
    """Generate the prompt for the summarization step.

      Args:
        action: Action picked.
        reason: The reason to pick the action.
        goal: The overall goal.
        before_elements: Information for UI elements on the before screenshot.
        after_elements: Information for UI elements on the after screenshot.

      Returns:
        The text prompt for summarization that will be sent to gpt4v.
      """
    return SUMMARY_PROMPT_TEMPLATE.format(
        goal=goal,
        before_elements=before_elements,
        after_elements=after_elements,
        action=action,
        reason=reason,
    )


class CoolAgent(base_agent.EnvironmentInteractingAgent):
    """CoolAgent which stands for Multimodal Autonomous Agent for Android."""

    def __init__(
            self,
            env: interface.AsyncEnv,
            llm: gemini_gemma_wrapper.GeminiGemmaWrapper,
            name: str = 'CoolAgent',
            wait_after_action_seconds: float | None = 2.0,
            run_log_dir: str | None = None,
            **kwargs,
    ):
        """Initializes a CoolAgent Agent.

            Args:
              env: The environment.
              llm: The multimodal LLM wrapper.
              name: The agent name.
              wait_after_action_seconds: Seconds to wait for the screen to stablize
                after executing an action. If None, uses the environment's auto-
                stabilization logic.
              run_log_dir: Directory to save run artifacts for debugging.
            """
        super().__init__(env, name, transition_pause=wait_after_action_seconds)
        self.llm = llm
        self.history = []
        self.additional_guidelines = None
        # (fingerprint, action) → counter  (for single‑state repeats)
        self._state_action_counter: defaultdict[tuple[str, str], int] = defaultdict(int)
        # Store most‑recent (state,action) pairs to detect A‑B‑A cycles
        self._recent_pairs: deque[tuple[str, str]] = deque(maxlen=8)
        # Pairs that became "forbidden" after detecting an A‑B‑A pattern
        self._blocked_pairs: set[tuple[str, str]] = set()
        self.set_run_log_dir(run_log_dir)

    def set_run_log_dir(self, run_log_dir: str | None) -> None:
        """Sets the directory for saving run artifacts and re-initializes subdirs."""
        self.run_log_dir = run_log_dir
        if self.run_log_dir:
            self.screenshot_dir = Path(self.run_log_dir) / 'screenshots'
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.screenshot_dir = None

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines

    def reset(self, go_home: bool = False):
        super().reset(go_home)
        # Hide the coordinates on screen which might affect the vision model.
        self.env.hide_automation_ui()
        self.history = []
        self._state_action_counter.clear()
        self._recent_pairs.clear()
        self._blocked_pairs.clear()

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        step_num = len(self.history) + 1
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
        print('----------step ' + str(step_num))

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
            if cool_agent_utils.validate_ui_element(ui_element, logical_screen_size):
                cool_agent_utils.add_ui_element_mark(
                    before_screenshot,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                )
        step_data['before_screenshot_with_som'] = before_screenshot.copy()

        if self.screenshot_dir:
            raw_path = self.screenshot_dir / f'{step_num}_before_raw.png'
            cv2.imwrite(str(raw_path), state.pixels.copy())
            marked_path = self.screenshot_dir / f'{step_num}_before_marked.png'
            cv2.imwrite(str(marked_path), before_screenshot)

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

        if is_safe == False:  # pylint: disable=singleton-comparison
            #  is_safe could be None
            action_output = f'"""Reason: {cool_agent_utils.TRIGGER_SAFETY_CLASSIFIER}\nAction: {{\"action_type\": \"status\", \"goal_status\": \"infeasible\"}}"""'

        if not raw_response:
            raise RuntimeError('Error calling LLM in action selection phase.')
        step_data['action_output'] = action_output
        step_data['action_raw_response'] = raw_response

        reason, action = cool_agent_utils.parse_reason_action_output(action_output)

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

        # ------------------------------------------------------------------ #
        # Improved loop detection                                            #
        # ------------------------------------------------------------------ #
        state_fingerprint = self._fingerprint_ui_state(before_ui_elements)
        canonical_action = self._canonicalize_action(action)
        state_action_pair = (state_fingerprint, canonical_action)
        if state_action_pair in self._blocked_pairs:
            print(f'LOOP DETECTED (A‑B‑A cycle): Blocked repetitive action: {action}')
            step_data['summary'] = (
                f'Action **disallowed** (part of an earlier A → B → A loop):\n'
                f'```json\n{canonical_action}\n```\n'
                'Please choose a different action.'
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        if self._state_action_counter[state_action_pair] >= self.MAX_SAME_ACTION_REPEATS:
            print(f'LOOP DETECTED: Refusing to execute repetitive action: {action}')
            step_data['summary'] = (
                f'Action **disallowed** after {self.MAX_SAME_ACTION_REPEATS + 1} '
                f'repeats in the same UI state:\n'
                f'```json\n{canonical_action}\n```\n'
                'Please pick another approach.'
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)
        # Record that we are attempting this pair once more.
        self._state_action_counter[state_action_pair] += 1
        # Save to recent list **before** execution so we record the exact order
        self._recent_pairs.append(state_action_pair)
        self._detect_two_cycle_loop()

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
                    f'Action failed: The generated index {action_index} is invalid. '
                    f'The number of available UI elements is {num_ui_elements}, so the'
                    f' index must be between 0 and {num_ui_elements - 1}. Please'
                    ' choose a valid index from the provided list.'
                )
                self.history.append(step_data)
                return base_agent.AgentInteractionResult(False, step_data)

            cool_agent_utils.add_ui_element_mark(
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
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )

        # Use the waiting logic from the base class, which respects
        # self.transition_pause and its 'auto' mode if set to None.
        state = self.get_post_transition_state()

        logical_screen_size = self.env.logical_screen_size
        orientation = self.env.orientation
        physical_frame_boundary = self.env.physical_frame_boundary
        after_ui_elements = state.ui_elements
        after_ui_elements_list = _generate_ui_elements_description_list(
            after_ui_elements, logical_screen_size
        )
        after_screenshot = state.pixels.copy()
        for index, ui_element in enumerate(after_ui_elements):
            if cool_agent_utils.validate_ui_element(ui_element, logical_screen_size):
                cool_agent_utils.add_ui_element_mark(
                    after_screenshot,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                )
        if self.screenshot_dir:
            after_marked_path = self.screenshot_dir / f'{step_num}_after_marked.png'
            cv2.imwrite(str(after_marked_path), after_screenshot)

        cool_agent_utils.add_screenshot_label(
            step_data['before_screenshot_with_som'], 'before'
        )
        cool_agent_utils.add_screenshot_label(after_screenshot, 'after')
        step_data['after_screenshot_with_som'] = after_screenshot.copy()

        is_ui_changing_action = converted_action.action_type in [
            'click',
            'long_press',
            'input_text',
            'scroll',
            'keyboard_enter',
            'navigate_back',
            'navigate_home',
        ]
        ui_did_not_change = before_ui_elements == after_ui_elements

        if is_ui_changing_action and ui_did_not_change:
            print(
                'Action appears to have had no effect on the UI tree. Overriding'
                ' summary.'
            )
            summary = (
                f'I performed action `{action}` but the screen did not change. The'
                ' action was ineffective. I need to re-evaluate and choose a'
                ' different action.'
            )
            step_data['summary_prompt'] = (
                'N/A - Summary overridden due to ineffective action.'
            )
            step_data['summary'] = f'Action selected: {action}. {summary}'
            step_data['summary_raw_response'] = (
                'Summary overridden by ineffective action detector.'
            )
            print('Summary: ' + summary)
        else:
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
                    step_data['before_screenshot_with_som'],
                    after_screenshot,
                ],
            )

            if is_safe == False:  # pylint: disable=singleton-comparison
                summary = '"""Summary triggered LLM safety classifier."""'

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

    # ========================= #
    #  Loop‑detection helpers   #
    # ========================= #

    @staticmethod
    def _fingerprint_ui_state(
            ui_elements: list[representation_utils.UIElement],
    ) -> str:
        """Return a stable MD5 fingerprint for the current UI.

        Only semantic attributes are kept, and the list is sorted so that
        element‑ordering changes (e.g. re‑layout, scrolling) do not break the
        equality check.
        """
        canonical_elements = []
        for el in ui_elements:
            # Skip status‑bar text that changes every minute / % tick.
            if CoolAgent._is_dynamic_text(el.text):
                continue
            canonical_elements.append(
                (
                    el.text or '',
                    el.content_description or '',
                    el.hint_text or '',
                    el.tooltip or '',
                    bool(el.is_selected),
                    bool(el.is_checked),
                )
            )
        canonical_elements.sort()
        serialised = json.dumps(canonical_elements, separators=(',', ':'))
        return hashlib.md5(serialised.encode('utf-8')).hexdigest()

    # Regexes to match "14:23", "14:23:05", "11 PM" or "87%"
    _DYNAMIC_TEXT_PATTERNS = [
        re.compile(r'^\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?$', re.I),
        re.compile(r'^\d{1,3}%$'),
    ]

    MAX_SAME_ACTION_REPEATS = 2  # Allow one extra retry in same state.

    @classmethod
    def _is_dynamic_text(cls, text: str | None) -> bool:
        if not text:
            return False
        for pat in cls._DYNAMIC_TEXT_PATTERNS:
            if pat.match(text.strip()):
                return True
        return False

    @staticmethod
    def _canonicalize_action(raw_action: str) -> str:
        """Return a canonical string representation of *any* action."""
        extracted = agent_utils.extract_json(raw_action)
        if extracted is None:
            return raw_action.strip()
        return json.dumps(extracted, sort_keys=True, separators=(',', ':'))

    # ------------------------------------------------------------------ #
    #  Two‑cycle (A,B,A,…) loop detector                                 #
    # ------------------------------------------------------------------ #

    def _detect_two_cycle_loop(self) -> None:
        """Detects pattern (A,B,A) and permanently blocks pair B."""
        if len(self._recent_pairs) < 3:
            return
        p_minus2, p_minus1, p_now = self._recent_pairs[-3], self._recent_pairs[-2], self._recent_pairs[-1]
        # Pattern: A B A   (A != B)
        if p_minus2 == p_now and p_minus2 != p_minus1:
            if p_minus1 not in self._blocked_pairs:
                print(
                    '🔄 Detected alternating loop: '
                    f'{p_minus2} ↔ {p_minus1}.  Blocking the second pair.'
                )
                self._blocked_pairs.add(p_minus1)
