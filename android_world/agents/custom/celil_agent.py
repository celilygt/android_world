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
import re
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
    
    agent_logger.info("🚀 AGENT INITIALIZATION STARTED")
    
    # Create separate LLM wrappers according to the architecture:
    
    # 1. Qwen VL for visual perception tasks (FREE model with high daily limits)
    agent_logger.info("🔧 Initializing Qwen VL wrapper for perception...")
    qwen_wrapper = QwenVLWrapper(
        model_name="qwen/qwen2.5-vl-72b-instruct:free",
        temperature=kwargs.get("temperature", 0.0),
        max_tokens=kwargs.get("max_tokens", 2048),
        high_credits=kwargs.get("high_credits", True),
    )
    
    # 2. Gemini for strategic planning and reasoning (Maestro)
    agent_logger.info("🔧 Initializing Gemini wrapper for strategic planning...")
    gemini_wrapper = GeminiGemmaWrapper(
        model_name="auto",  # Smart routing to best available Gemini model
        temperature=kwargs.get("temperature", 0.0),
        max_retry=kwargs.get("max_retry", 3),
        verbose=kwargs.get("verbose", True),
    )
    
    # 3. Gemma for high-volume verification tasks (cheaper than Gemini)
    agent_logger.info("🔧 Initializing Gemma wrapper for verification...")
    gemma_wrapper = GeminiGemmaWrapper(
        model_name="gemma-3n-e4b-it",  # High-volume Gemma model
        temperature=kwargs.get("temperature", 0.0),
        max_retry=kwargs.get("max_retry", 3),
        verbose=False,  # Less verbose for verifiers
    )
    
    # Initialize modules with appropriate LLM wrappers
    agent_logger.info("🔧 Initializing agent modules...")
    self.maestro = MaestroPlanner(gemini_wrapper)  # Strategic planning with Gemini
    self.quality_control = VerifierEnsemble(gemma_wrapper)  # Verification with Gemma (cost-effective)
    
    # Initialize UI-TARS action generator with configuration
    self.section_leader = UITarsActionGenerator(
        model_name=kwargs.get("ui_tars_model_name", "avil/UI-TARS:latest"),
        temperature=kwargs.get("ui_tars_temperature", 0.0),
        max_new_tokens=kwargs.get("ui_tars_max_new_tokens", 256),
    )
    
    self.episodic_memory = EpisodicMemory()
    # Initialize working_memory as None - will be set when step() is called with a goal
    self.working_memory = None
    self.perception = PerceptionModule(qwen_wrapper)  # Enhanced visual understanding with Qwen VL
    self.tools = None
    
    # Log the proper architecture setup
    agent_logger.info("✅ AGENT INITIALIZATION COMPLETE")
    logging.info("🎭 CelilAgent initialized with proper architecture:")
    logging.info("   📸 Perception: Qwen VL (visual understanding)")
    logging.info("   🧠 Maestro: Gemini Pro (strategic planning)")
    logging.info("   🎯 Section Leader: UI-TARS (action generation)")
    logging.info("   ✅ Quality Control: Gemma 3n (verification ensemble)")

  def _detect_required_app_from_goal(self, goal: str) -> str | None:
    """Detect which app needs to be opened based on the goal text."""
    goal_lower = goal.lower()
    
    # App name mappings from goal text to actual app names
    app_mappings = {
        # Audio/Recording apps
        'audio recorder': 'audio recorder',
        'record audio': 'audio recorder',
        'recording': 'audio recorder',
        
        # Contact apps
        'contact': 'contacts',
        
        # Messaging apps
        'message': 'simple sms messenger',
        'sms': 'simple sms messenger',
        
        # Email apps
        'email': 'gmail',
        'gmail': 'gmail',
        
        # Calendar apps
        'calendar': 'simple calendar pro',
        'schedule': 'simple calendar pro',
        'appointment': 'simple calendar pro',
        
        # Browser apps
        'browser': 'chrome',
        'chrome': 'chrome',
        'web': 'chrome',
        
        # Phone apps
        'call': 'phone',
        'dial': 'phone',
        'phone': 'phone',
        
        # Clock/Timer apps
        'timer': 'clock',
        'alarm': 'clock',
        'stopwatch': 'clock',
        
        # Settings
        'setting': 'settings',
        'wifi': 'settings',
        'bluetooth': 'settings',
        
        # Camera
        'camera': 'camera',
        'photo': 'camera',
        'picture': 'camera',
        
        # File manager
        'file': 'files',
        'folder': 'files',
    }
    
    # Find the best matching app
    for keyword, app_name in app_mappings.items():
        if keyword in goal_lower:
            agent_logger.info(f"🎯 APP DETECTION: Detected required app from goal: {app_name}")
            logging.info(f"🎯 Detected required app from goal: {app_name}")
            return app_name
    
    agent_logger.info("🎯 APP DETECTION: No specific app detected in goal")
    return None

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    """Takes a single step in the environment."""
    agent_logger.info("=" * 80)
    agent_logger.info(f"🎬 STEP START: Goal = '{goal[:100]}{'...' if len(goal) > 100 else ''}'")
    
    if self.working_memory is None:
      agent_logger.info("🧠 MEMORY: Initializing working memory for new goal")
      self.working_memory = WorkingMemory(goal)

    if not self.working_memory.get_plan():
      agent_logger.info("📋 PLANNING: No plan exists, generating initial plan...")
      
      # Check if we need to open a specific app first
      required_app = self._detect_required_app_from_goal(goal)
      
      agent_logger.info("🔍 EPISODIC MEMORY: Searching for similar tasks...")
      retrieved_plan = self.episodic_memory.find_similar_task(goal)
      if retrieved_plan:
        agent_logger.info(f"📚 EPISODIC MEMORY: Found similar plan with {len(retrieved_plan)} steps")
      else:
        agent_logger.info("📚 EPISODIC MEMORY: No similar tasks found")
      
      agent_logger.info("👁️ PERCEPTION: Processing current screen observation...")
      agent_logger.info("   🤖 LLM CALL INCOMING: Qwen VL for visual understanding")
      observation = self.perception.process_observation(self.get_post_transition_state())
      agent_logger.info(f"   ✅ PERCEPTION COMPLETE: {len(observation.get('summary', ''))} chars summary")
      
      agent_logger.info("🧠 MAESTRO: Generating initial strategic plan...")
      agent_logger.info("   🤖 LLM CALL INCOMING: Gemini for strategic planning")
      plan = self.maestro.generate_initial_plan(
          goal, observation, retrieved_plan
      )
      agent_logger.info(f"   ✅ MAESTRO COMPLETE: Generated plan with {len(plan)} steps")
      
      # Prepend opening the app if needed
      if required_app:
        plan.insert(0, f"Open the {required_app} app")
        agent_logger.info(f"📱 APP OPENING: Prepended app opening to plan: {required_app}")
      
      self.working_memory.set_plan(plan)
      agent_logger.info(f"📋 PLAN SET: {len(plan)} total steps")
      for i, step in enumerate(plan, 1):
        agent_logger.info(f"   {i}. {step}")

    # Check if plan is empty before popping
    if not self.working_memory.get_plan():
      agent_logger.info("🏁 STEP COMPLETE: Plan is empty, task finished")
      return base_agent.AgentInteractionResult(
          done=True, data={"reason": "Plan complete."}
      )

    sub_goal = self.working_memory.pop_sub_goal()
    agent_logger.info(f"🎯 SUB-GOAL: Executing next step: '{sub_goal}'")
    
    agent_logger.info("👁️ PERCEPTION: Processing current screen observation...")
    agent_logger.info("   🤖 LLM CALL INCOMING: Qwen VL for visual understanding")
    observation = self.perception.process_observation(self.get_post_transition_state())
    agent_logger.info(f"   ✅ PERCEPTION COMPLETE: {len(observation.get('summary', ''))} chars summary")
    
    context_summary = self.working_memory.get_context_summary()
    agent_logger.info(f"🧠 CONTEXT: Retrieved {len(context_summary)} chars of context summary")

    # Generate action using UI-TARS (returns JSON string)
    agent_logger.info("🎯 SECTION LEADER: Generating action with UI-TARS...")
    agent_logger.info("   🤖 LLM CALL INCOMING: UI-TARS for action generation")
    action_json_str = self.section_leader.generate_action(
        sub_goal, observation["summary"], context_summary
    )
    agent_logger.info(f"   ✅ SECTION LEADER COMPLETE: Generated action")

    # Parse the JSON string to dict
    try:
      import json
      # Check if it's already a dict (from agent_utils.extract_json)
      if isinstance(action_json_str, dict):
        proposed_action = action_json_str
      else:
        proposed_action = json.loads(action_json_str)
      agent_logger.info(f"📋 ACTION PARSED: {proposed_action}")
    except (json.JSONDecodeError, TypeError) as e:
      agent_logger.error(f"❌ ACTION PARSING FAILED: {e}. Raw response: {action_json_str}")
      logging.error(f"Failed to parse action JSON: {e}. Raw response: {action_json_str}")
      # Use fallback action
      proposed_action = {"action_type": "wait", "time": 1.0}
      agent_logger.info(f"🔧 FALLBACK ACTION: Using wait action")

    agent_logger.info("✅ QUALITY CONTROL: Verifying proposed action...")
    agent_logger.info("   🤖 LLM CALL INCOMING: Gemini for action verification")
    score, reasoning = self.quality_control.verify_action(
        sub_goal, proposed_action, observation, context_summary
    )
    agent_logger.info(f"   ✅ QUALITY CONTROL COMPLETE: Score = {score}/10")
    reasoning_str = str(reasoning) if reasoning else "No reasoning provided"
    agent_logger.info(f"   💭 REASONING: {reasoning_str[:150]}{'...' if len(reasoning_str) > 150 else ''}")

    if score >= 7.0:
      agent_logger.info("✅ ACTION APPROVED: Score >= 7.0, executing action")
      # Convert proposed action to JSONAction and execute
      try:
        json_action_obj = json_action.JSONAction(**proposed_action)
        agent_logger.info(f"🚀 EXECUTING ACTION: {proposed_action}")
        action_result = self.env.execute_action(json_action_obj)
        
        # Simple post-execution check
        agent_logger.info("🔍 POST-EXECUTION: Checking if screen changed...")
        agent_logger.info("   🤖 LLM CALL INCOMING: Qwen VL for post-execution analysis")
        new_observation = self.perception.process_observation(self.get_post_transition_state())
        screen_changed = not (observation['screenshot'] == new_observation['screenshot']).all()
        
        outcome = 'success' if screen_changed else 'failure'
        agent_logger.info(f"   📊 OUTCOME: {outcome} (screen_changed = {screen_changed})")
        
        self.working_memory.add_step({
            'sub_goal': sub_goal,
            'action': proposed_action,
            'score': score,
            'reasoning': reasoning,
            'outcome': outcome
        })
        agent_logger.info("💾 MEMORY: Recorded successful step")
        
      except Exception as e:
        agent_logger.error(f"❌ ACTION EXECUTION FAILED: {e}")
        logging.error(f"Action execution failed: {e}")
        self.working_memory.add_step({
            'sub_goal': sub_goal,
            'action': proposed_action,
            'score': score,
            'reasoning': reasoning,
            'outcome': 'execution_error'
        })
        agent_logger.info("💾 MEMORY: Recorded execution error")
        
    else:
      agent_logger.warning(f"🚫 ACTION REJECTED: Score {score} < 7.0, triggering rehearsal loop")
      # Rehearsal loop trigger
      failure_context = {
          'goal': goal,
          'original_plan': self.working_memory.get_plan(),
          'failed_sub_goal': sub_goal,
          'failed_action': proposed_action,
          'verifier_feedback': reasoning,
          'observation': observation
      }
      agent_logger.info("🔄 REHEARSAL: Generating corrective plan...")
      agent_logger.info("   🤖 LLM CALL INCOMING: Gemini for corrective planning")
      new_plan = self.maestro.generate_corrective_plan(failure_context)
      agent_logger.info(f"   ✅ CORRECTIVE PLAN: Generated {len(new_plan)} new steps")
      
      self.working_memory.set_plan(new_plan)
      self.working_memory.add_step({
          'sub_goal': sub_goal,
          'action': proposed_action,
          'score': score,
          'reasoning': reasoning,
          'outcome': 'rejected'
      })
      agent_logger.info("💾 MEMORY: Recorded rejected action and updated plan")

    remaining_steps = len(self.working_memory.get_plan())
    agent_logger.info(f"📊 STEP SUMMARY: {remaining_steps} steps remaining in plan")

    if not self.working_memory.get_plan():
        agent_logger.info("🏁 TASK COMPLETE: No more steps in plan")
        return base_agent.AgentInteractionResult(
            done=True, data={"reason": "Plan complete."}
        )

    agent_logger.info("➡️ STEP CONTINUING: Moving to next step")
    return base_agent.AgentInteractionResult(
        done=False, data={"reason": "Continuing plan."}
    )
