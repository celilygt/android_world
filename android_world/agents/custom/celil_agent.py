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

"""The Conductor Agent."""

import logging
import os
import re
from PIL import Image
import imagehash
from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import json_action
from android_world.agents.custom.celil_perception import PerceptionModule
from android_world.agents.custom.celil_memory import WorkingMemory, EpisodicMemory
from android_world.agents.custom.celil_maestro import MaestroPlanner
from android_world.agents.custom.celil_section_leader import UITarsActionGenerator
from android_world.agents.custom.celil_quality_control import VerifierEnsemble
from android_world.agents.llm_wrappers.qwen_vl_wrapper import QwenVLWrapper
from android_world.agents.llm_wrappers.gemini_gemma_wrapper import GeminiGemmaWrapper

# Configure detailed agent logging
agent_logger = logging.getLogger('celil_agent_flow')
agent_logger.setLevel(logging.INFO)

# Create handler if it doesn't exist
if not agent_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('🎭 CELIL [%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    agent_logger.addHandler(handler)
    agent_logger.propagate = False  # Prevent duplicate logs


class CelilAgent(base_agent.EnvironmentInteractingAgent):
    """An agent that uses a conductor architecture with proper model allocation."""

    def __init__(self, env: interface.AsyncAndroidEnv, **kwargs):
        """Initializes the CelilAgent."""
        # Only pass supported parameters to parent class
        name = kwargs.get("name", "CelilAgent")
        transition_pause = kwargs.get("transition_pause", 1.0)
        super().__init__(env, name=name, transition_pause=transition_pause)

        self.speed_mode = kwargs.get("speed_mode", False)
        self.qc_fast_threshold = kwargs.get("qc_fast_threshold", 8.0)
        agent_logger.info(f"⚡ SPEED MODE: {'Enabled' if self.speed_mode else 'Disabled'}")

        agent_logger.info("🚀 AGENT INITIALIZATION STARTED")

        self.run_log_dir = kwargs.get("run_log_dir")
        self.step_count = 0
        if self.run_log_dir:
            self.screenshot_dir = os.path.join(self.run_log_dir, "screenshots")
            os.makedirs(self.screenshot_dir, exist_ok=True)
            agent_logger.info(f"💾 Logging screenshots to: {self.screenshot_dir}")

            llm_log_path = os.path.join(self.run_log_dir, 'llm_interactions.log')
            file_handler = logging.FileHandler(llm_log_path, mode='w')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            for logger_name in ['celil_agent_flow', 'qwen_llm_calls', 'gemini_llm_calls', 'uitars_llm_calls']:
                logger = logging.getLogger(logger_name)
                # Remove existing stream handlers to avoid duplicate console logs if re-initializing
                for h in logger.handlers:
                    if isinstance(h, logging.StreamHandler):
                        logger.removeHandler(h)
                logger.addHandler(file_handler)
                logger.setLevel(logging.INFO)

        # Create separate LLM wrappers according to the architecture:
        qwen_wrapper = QwenVLWrapper(
            model_name="qwen/qwen2.5-vl-72b-instruct:free",
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 2048),
            high_credits=kwargs.get("high_credits", True),
        )
        gemini_wrapper = GeminiGemmaWrapper(
            model_name="auto",
            temperature=kwargs.get("temperature", 0.0),
            max_retry=kwargs.get("max_retry", 3),
            verbose=kwargs.get("verbose", True),
        )
        gemma_wrapper = GeminiGemmaWrapper(
            model_name="gemma-3-27b-it",
            temperature=kwargs.get("temperature", 0.0),
            max_retry=kwargs.get("max_retry", 3),
            verbose=False,
        )

        self.maestro = MaestroPlanner(gemini_wrapper)
        self.quality_control = VerifierEnsemble(gemma_wrapper)
        self.section_leader = UITarsActionGenerator(
            model_name=kwargs.get("ui_tars_model_name", "avil/UI-TARS:latest"),
            temperature=kwargs.get("ui_tars_temperature", 0.0),
            max_new_tokens=kwargs.get("ui_tars_max_new_tokens", 256),
        )
        self.episodic_memory = EpisodicMemory()
        self.working_memory = None
        self.perception = PerceptionModule(qwen_wrapper)
        self.tools = None

        self.last_observation_cache = None
        self.last_screenshot_phash = None
        self.micro_correction_retries = 0

        agent_logger.info("✅ AGENT INITIALIZATION COMPLETE")
        logging.info("🎭 CelilAgent initialized with proper architecture:")
        logging.info("   📸 Perception: Qwen VL (visual understanding)")
        logging.info("   🧠 Maestro: Gemini Pro (strategic planning)")
        logging.info("   🎯 Section Leader: UI-TARS (action generation)")
        logging.info("   ✅ Quality Control: Gemma 3n (verification ensemble)")

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """Takes a single step in the environment."""
        self.step_count += 1
        agent_logger.info("=" * 80)
        agent_logger.info(f"🎬 STEP {self.step_count} START: Goal = '{goal[:100]}{'...' if len(goal) > 100 else ''}'")

        if self.working_memory is None:
            agent_logger.info("🧠 MEMORY: Initializing working memory for new goal")
            self.working_memory = WorkingMemory(goal)

        current_state = self.get_post_transition_state()
        if self.run_log_dir:
            screenshot_path = os.path.join(self.screenshot_dir, f"step_{self.step_count:03d}.png")
            try:
                Image.fromarray(current_state.pixels).save(screenshot_path)
                agent_logger.info(f"📸 Saved screenshot to {screenshot_path}")
            except Exception as e:
                agent_logger.error(f"Failed to save screenshot: {e}")

        if not self.working_memory.get_plan():
            agent_logger.info("📋 PLANNING: No plan exists, generating initial plan...")

            agent_logger.info("🔍 EPISODIC MEMORY: Searching for similar tasks...")
            retrieved_plan = self.episodic_memory.find_similar_task(goal)
            if retrieved_plan:
                agent_logger.info(f"📚 EPISODIC MEMORY: Found similar plan with {len(retrieved_plan)} steps")
            else:
                agent_logger.info("📚 EPISODIC MEMORY: No similar tasks found")

            agent_logger.info("👁️ PERCEPTION: Processing current screen observation (fast mode)...")
            observation = self.perception.process_observation(current_state, deep_analysis=False)
            agent_logger.info(f"   ✅ PERCEPTION COMPLETE: {len(observation.get('summary', ''))} chars summary")

            agent_logger.info("🧠 MAESTRO: Generating initial strategic plan...")
            agent_logger.info("   🤖 LLM CALL INCOMING: Gemini for strategic planning")
            plan = self.maestro.generate_initial_plan(
                goal, observation, retrieved_plan
            )
            agent_logger.info(f"   ✅ MAESTRO COMPLETE: Generated plan with {len(plan)} steps")

            self.working_memory.set_plan(plan)
            agent_logger.info(f"📋 PLAN SET: {len(plan)} total steps")
            for i, step in enumerate(plan, 1):
                agent_logger.info(f"   {i}. {step}")

        if not self.working_memory.get_plan():
            agent_logger.info("🏁 TASK COMPLETE: Plan is empty, concluding task.")
            return base_agent.AgentInteractionResult(done=True, data={"reason": "Plan empty."})

        sub_goal = self.working_memory.get_plan()[0]
        agent_logger.info(f"🎯 SUB-GOAL: Executing next step: '{sub_goal}'")

        current_phash = imagehash.phash(Image.fromarray(current_state.pixels))
        agent_logger.info(f"🖼️  Screen p-hash: {current_phash}")

        # The slow Qwen perception call has been removed. We now always perform a fast
        # OCR/UI tree-based observation, which is still useful for the Maestro planner's context.
        # The action generator (SectionLeader) will get the raw screenshot.
        agent_logger.info("👁️  PERCEPTION (SHALLOW): Performing fast OCR/UI Tree analysis.")
        observation = self.perception.process_observation(current_state, deep_analysis=False)
        self.last_screenshot_phash = current_phash

        context_summary = self.working_memory.get_context_summary()
        agent_logger.info(f"🧠 CONTEXT: Retrieved {len(context_summary)} chars of context summary")

        agent_logger.info("🎯 SECTION LEADER: Generating action with UI-TARS (with screenshot)...")
        agent_logger.info("   🤖 LLM CALL INCOMING: UI-TARS for action generation")
        action_dict, confidence = self.section_leader.generate_action_with_screenshot(
            sub_goal,
            current_state.pixels,  # Pass the raw screenshot for visual analysis
            context_summary
        )
        agent_logger.info(f"   ✅ SECTION LEADER COMPLETE: Generated action with confidence {confidence:.1f}")

        try:
            import json
            proposed_action = json.loads(action_dict) if isinstance(action_dict, str) else action_dict
            agent_logger.info(f"📋 ACTION PARSED: {proposed_action}")
        except (json.JSONDecodeError, TypeError) as e:
            agent_logger.error(f"❌ ACTION PARSING FAILED: {e}. Raw response: {action_dict}")
            proposed_action = {"action_type": "wait", "time": 1.0}
            agent_logger.info(f"🔧 FALLBACK ACTION: Using wait action")

        if self.speed_mode and confidence >= self.qc_fast_threshold:
            agent_logger.info(f"⚡ SPEED MODE: Action generator is confident ({confidence:.1f}). Skipping verifier.")
            score, reasoning = 10.0, {"reasoning": "Fast-passed due to high generator confidence."}
        else:
            agent_logger.info(f"✅ QUALITY CONTROL: Verifying proposed action (Confidence: {confidence:.1f})...")
            agent_logger.info("   🤖 LLM CALL INCOMING: Gemini for action verification (Consolidated)")
            score, reasoning = self.quality_control.verify_action(sub_goal, proposed_action, observation,
                                                                  context_summary)

        agent_logger.info(f"   ✅ QUALITY CONTROL COMPLETE: Score = {score}/10")
        reasoning_str = str(reasoning) if reasoning else "No reasoning provided"
        agent_logger.info(f"   💭 REASONING: {reasoning_str[:150]}{'...' if len(reasoning_str) > 150 else ''}")

        action_approved = score >= 7.0
        step_succeeded = False

        if action_approved:
            agent_logger.info("✅ ACTION APPROVED: Score >= 7.0, executing action")
            outcome = 'execution_error'
            try:
                if 'coordinate' in proposed_action and proposed_action.get('action_type') == 'click':
                    coords = proposed_action.pop('coordinate')
                    if isinstance(coords, list) and len(coords) == 2:
                        proposed_action['x'], proposed_action['y'] = coords[0], coords[1]

                json_action_obj = json_action.JSONAction(**proposed_action)
                agent_logger.info(f"🚀 EXECUTING ACTION: {proposed_action}")
                self.env.execute_action(json_action_obj)

                agent_logger.info("🔍 POST-EXECUTION: Checking if screen changed...")
                new_state = self.get_post_transition_state()
                new_phash = imagehash.phash(Image.fromarray(new_state.pixels))
                screen_changed = new_phash != self.last_screenshot_phash

                outcome = 'success' if screen_changed else 'failure'
                agent_logger.info(f"   📊 OUTCOME: {outcome} (screen_changed = {screen_changed})")

                if screen_changed:
                    # No need for a slow perception call here either.
                    self.last_screenshot_phash = new_phash


            except Exception as e:
                agent_logger.error(f"❌ ACTION EXECUTION FAILED: {e}")
                outcome = 'execution_error'

            step_succeeded = (outcome == 'success')
            self.working_memory.add_step(
                {'sub_goal': sub_goal, 'action': proposed_action, 'score': score, 'reasoning': reasoning,
                 'outcome': outcome})
            agent_logger.info("💾 MEMORY: Recorded attempted step")
        else:
            agent_logger.warning(f"🚫 ACTION REJECTED: Score {score} < 7.0")
            self.working_memory.add_step(
                {'sub_goal': sub_goal, 'action': proposed_action, 'score': score, 'reasoning': reasoning,
                 'outcome': 'rejected_by_verifier'})

        if step_succeeded:
            agent_logger.info("✅ STEP SUCCEEDED: Popping completed sub-goal from plan.")
            self.working_memory.pop_sub_goal()
            self.micro_correction_retries = 0
        else:
            self.micro_correction_retries += 1
            agent_logger.warning(
                f"STEP FAILED: Not popping sub-goal. Failure count for this step: {self.micro_correction_retries}/3.")

            if self.micro_correction_retries == 2:
                agent_logger.info(
                    "🔄 MICRO-CORRECTION (LVL 2): Step failed twice. Trying 'navigate_back' to get unstuck.")
                try:
                    self.env.execute_action(json_action.JSONAction(action_type=json_action.NAVIGATE_BACK))
                except Exception as e:
                    agent_logger.error(f"❌ 'navigate_back' action failed during recovery: {e}")

            elif self.micro_correction_retries >= 3:
                agent_logger.error(
                    f"💥 RECOVERY FAILED: Too many retries ({self.micro_correction_retries}). Escalating to Maestro for full re-plan.")

                # Perform a fast observation for the replan context
                failure_observation = self.perception.process_observation(current_state, deep_analysis=False)

                failure_context = {
                    'goal': goal, 'original_plan': self.working_memory.get_plan(), 'failed_sub_goal': sub_goal,
                    'failed_action': proposed_action, 'verifier_feedback': reasoning, 'observation': failure_observation,
                }
                agent_logger.info("   🤖 LLM CALL INCOMING: Gemini for corrective planning")
                new_plan = self.maestro.generate_corrective_plan(failure_context)
                self.working_memory.set_plan(new_plan)
                self.micro_correction_retries = 0
                agent_logger.info(f"   ✅ MAESTRO COMPLETE: Generated new plan with {len(new_plan)} steps.")

            else:
                agent_logger.info(
                    "🔄 MICRO-CORRECTION (LVL 1): Will retry the same sub-goal on the next step with new context.")

        remaining_steps = len(self.working_memory.get_plan())
        agent_logger.info(f"📊 STEP SUMMARY: {remaining_steps} steps remaining in plan")

        if not self.working_memory.get_plan():
            agent_logger.info("🏁 TASK COMPLETE: No more steps in plan")
            return base_agent.AgentInteractionResult(done=True, data={"reason": "Plan complete."})

        agent_logger.info("➡️ STEP CONTINUING: Moving to next step")
        return base_agent.AgentInteractionResult(done=False, data={"reason": "Continuing plan."})