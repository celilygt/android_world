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

"""Section Leader (action generator) for the Celil agent."""

import logging
from android_world.agents import agent_utils
from android_world.agents.llm_wrappers.ui_tars_wrapper import UITarsWrapper

# Configure logger for this module
uitars_logger = logging.getLogger('uitars_llm_calls')
uitars_logger.propagate = False
if not uitars_logger.handlers:
    uitars_logger.addHandler(logging.StreamHandler())
    uitars_logger.setLevel(logging.INFO)

UI_TARS_PROMPT_TEMPLATE = """You are an Android UI automation assistant. Your task: {sub_goal}

Context from recent actions:
{context_summary}

Current screen analysis:
{annotated_observation}

Generate a JSON action to accomplish the task. Use these exact formats:

For clicking: {{"action_type": "click", "coordinate": [x, y]}}
For typing: {{"action_type": "input_text", "text": "text to type"}}
For scrolling: {{"action_type": "scroll", "direction": "up"}}
For going home: {{"action_type": "navigate_home"}}
For going back: {{"action_type": "navigate_back"}}
For opening an app: {{"action_type": "open_app", "app_name": "app name"}}
For waiting: {{"action_type": "wait", "time": 1.0}}

Required format: JSON only, no extra text.
Response:"""


class UITarsActionGenerator:
    """UI-TARS based action generator using the 0000/ui-tars-1.5-7b-q8_0 model."""

    def __init__(
            self,
            model_name: str = "avil/UI-TARS:latest",
            temperature: float = 0.0,
            max_new_tokens: int = 256,
            **kwargs
    ):
        """Initialize the UI-TARS action generator.

        Args:
            model_name: UI-TARS model name (should be 0000/ui-tars-1.5-7b-q8_0)
            temperature: Generation temperature
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to UITarsWrapper
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        # Initialize UI-TARS wrapper
        try:
            self.ui_tars = UITarsWrapper(
                model_name=model_name,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            print(f"✅ UI-TARS Action Generator initialized with model: {model_name}")

            # Test the model connection
            model_info = self.ui_tars.get_model_info()
            print(f"📊 Model info: {model_info}")

        except Exception as e:
            print(f"❌ Failed to initialize UI-TARS: {e}")
            raise

    def generate_action(
            self,
            sub_goal: str,
            annotated_observation: str,
            context_summary: str = ""
    ) -> tuple[dict, float]:
        """Generate a JSON action based on the current screen and goal.

        Args:
            sub_goal: The current sub-goal to accomplish
            annotated_observation: Annotated screen description
            context_summary: Summary of recent actions taken

        Returns:
            A tuple containing the action dictionary and a confidence score.
        """
        try:
            # Handle "open app" sub-goals directly
            if "open the" in sub_goal.lower() and "app" in sub_goal.lower():
                # Extract app name from sub-goal like "Open the audio recorder app"
                import re
                match = re.search(r'open the (.+?) app', sub_goal.lower())
                if match:
                    app_name = match.group(1).strip()
                    # BUG FIX: Only proceed if the extracted app_name is not empty.
                    # This prevents the heuristic from incorrectly handling complex sentences
                    # where the regex might match but extract an empty string, causing errors.
                    if app_name:
                        # FIX: Add this line to clean the extracted app name
                        app_name = app_name.strip('`\'"')
                        action = {"action_type": "open_app", "app_name": app_name}
                        print(f"🎯 Direct app opening action: {action}")
                        return action, 9.5  # High confidence for direct match

            # Prepare the prompt
            prompt = UI_TARS_PROMPT_TEMPLATE.format(
                sub_goal=sub_goal,
                annotated_observation=annotated_observation,
                context_summary=context_summary or "No previous actions in current context."
            )

            uitars_logger.info("--- UI-TARS Request ---")
            uitars_logger.info(f"\n---PROMPT START---\n{prompt}\n---PROMPT END---")

            # Generate response using UI-TARS
            response = self.ui_tars.generate_text(
                prompt=prompt,
                max_tokens=self.max_new_tokens
            )

            uitars_logger.info("--- UI-TARS Response ---")
            uitars_logger.info(f"\n---RESPONSE START---\n{response}\n---RESPONSE END---")

            if not response:
                print("⚠️ UI-TARS returned empty response")
                return self._get_fallback_action(), 2.0

            # Try to extract and validate JSON from response
            cleaned_action = agent_utils.extract_json(response)

            if cleaned_action:
                print(f"✅ UI-TARS generated action: {cleaned_action}")
                # Placeholder confidence: high if successful, low if fallback
                confidence = 9.0
                return cleaned_action, confidence
            else:
                print(f"⚠️ Could not extract valid JSON from UI-TARS response: {response[:200]}...")
                fallback_action = self._get_fallback_action()
                confidence = 2.0
                return fallback_action, confidence

        except Exception as e:
            print(f"❌ Error in UI-TARS action generation: {e}")
            fallback_action = self._get_fallback_action()
            confidence = 2.0
            return fallback_action, confidence

    def generate_action_with_screenshot(
            self,
            sub_goal: str,
            screenshot,
            context_summary: str = ""
    ) -> str:
        """Generate action using screenshot input directly.

        Args:
            sub_goal: The current sub-goal to accomplish
            screenshot: Screenshot image (numpy array or PIL Image)
            context_summary: Summary of recent actions taken

        Returns:
            JSON string representing the action to take
        """
        try:
            # Prepare prompt for multimodal input
            prompt = f"""Task: {sub_goal}

Context: {context_summary or "No previous actions."}

Analyze the screenshot and generate the next JSON action to accomplish the task.

Respond with ONLY a single valid JSON action object:
{{"action_type": "click", "coordinate": [x, y]}}
{{"action_type": "input_text", "text": "text to type"}}
{{"action_type": "scroll", "direction": "up/down/left/right"}}
{{"action_type": "navigate_home"}}
{{"action_type": "navigate_back"}}
{{"action_type": "wait", "time": 1.0}}"""

            # Generate response with screenshot
            response, metadata = self.ui_tars.generate_text_with_images(
                prompt=prompt,
                images=[screenshot],
                max_tokens=self.max_new_tokens
            )

            if not response:
                print("⚠️ UI-TARS returned empty response for screenshot input")
                return self._get_fallback_action()

            # Try to extract and validate JSON from response
            cleaned_action = agent_utils.extract_json(response)

            if cleaned_action:
                print(f"✅ UI-TARS generated action from screenshot: {cleaned_action}")
                print(f"📊 Generation metadata: {metadata}")
                return cleaned_action
            else:
                print(f"⚠️ Could not extract valid JSON from screenshot response: {response[:200]}...")
                return self._get_fallback_action()

        except Exception as e:
            print(f"❌ Error in UI-TARS screenshot action generation: {e}")
            return self._get_fallback_action()

    def _get_fallback_action(self) -> str:
        """Get a safe fallback action when UI-TARS fails.

        Returns:
            JSON string for a safe wait action
        """
        fallback = '{"action_type": "wait", "time": 1.0}'
        print(f"🔄 Using fallback action: {fallback}")
        return fallback

    def test_connection(self) -> bool:
        """Test if the UI-TARS model is working.

        Returns:
            True if model is working, False otherwise
        """
        try:
            return self.ui_tars.test_model()
        except Exception as e:
            print(f"❌ UI-TARS connection test failed: {e}")
            return False

    def get_model_status(self) -> dict:
        """Get current model status and information.

        Returns:
            Dictionary with model status information
        """
        try:
            model_info = self.ui_tars.get_model_info()
            model_info["temperature"] = self.temperature
            model_info["max_tokens"] = self.max_new_tokens
            return model_info
        except Exception as e:
            return {
                "name": self.model_name,
                "status": "Error",
                "error": str(e)
            }