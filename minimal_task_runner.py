#!/usr/bin/env python3
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

"""Runs a single task with a specified agent.

This script is a minimal version of run.py, designed for quick testing of a
single task with a chosen agent. It supports various agents via the
`--agent_name` flag.
"""

from collections.abc import Sequence
import os
import random
import sys
from typing import Type, Dict, Any

from absl import app
from absl import flags
from absl import logging
from android_world import registry
from android_world.agents import base_agent
from android_world.agents import registry as agent_registry
from android_world.env import env_launcher, interface
from android_world.task_evals import task_eval

# Agent-related imports are now handled by the agent_registry
# For example:
# from android_world.agents.custom.m3a_openrouter import M3AOpenRouter
# from android_world.agents.custom.m3a_gemini_gemma import M3AGeminiGemma
# from android_world.agents import t3a, infer

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = 'none'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
os.environ['GRPC_POLL_STRATEGY'] = 'poll'


def _find_adb_directory() -> str:
    """Returns the directory where adb is located."""
    potential_paths = [
        os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
        os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    raise EnvironmentError(
        'adb not found. Please install Android SDK and ensure adb is in one of'
        ' the expected directories or update the path.'
    )

# --- Core Flags ---
_ADB_PATH = flags.DEFINE_string(
    'adb_path', _find_adb_directory(), 'Path to adb.'
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup', False, 'Whether to perform initial emulator setup.'
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port', 5554, 'Console port of the Android device.'
)
_TASK = flags.DEFINE_string(
    'task', None, 'Specific task to run. If None, a random task is chosen.'
)
_AGENT_NAME = flags.DEFINE_string(
    'agent_name',
    'm3a_ollama_agent',
    'The agent to use for the task.',
)
_VERBOSE = flags.DEFINE_boolean(
    'verbose', True, 'Enable verbose output from the agent.'
)

# --- LLM/Agent Configuration Flags ---
_MODEL_NAME = flags.DEFINE_string(
    'model_name', None, 'Name of the LLM model to use (agent-specific).'
)
_TEMPERATURE = flags.DEFINE_float(
    'temperature', 0.0, 'LLM temperature for generation.'
)
_MAX_RETRY = flags.DEFINE_integer('max_retry', 3, 'Max retries for LLM calls.')
_WAIT_AFTER_ACTION = flags.DEFINE_float(
    'wait_after_action_seconds',
    2.0,
    'Seconds to wait after an action.',
)

# --- Gemini-Specific Flags ---
_TOP_P = flags.DEFINE_float('top_p', 0.95, 'Top-p for Gemini model sampling.')
_ENABLE_SAFETY_CHECKS = flags.DEFINE_boolean(
    'enable_safety_checks', True, 'Enable Gemini safety checks.'
)

# --- Ollama-Specific Flags ---
_HOST = flags.DEFINE_string('host', 'localhost', 'Ollama host.')
_PORT = flags.DEFINE_integer('port', 11434, 'Ollama port.')


def _check_api_keys():
    """Check for necessary API keys based on the selected agent."""
    if 'openrouter' in _AGENT_NAME.value.lower():
        if 'OPENROUTER_API_KEY' not in os.environ:
            print("❌ OPENROUTER_API_KEY not set.")
            sys.exit(1)
        print("✅ OpenRouter API key found.")
    elif 'gemini' in _AGENT_NAME.value.lower():
        if 'GEMINI_API_KEY' not in os.environ:
            print("❌ GEMINI_API_KEY not set.")
            sys.exit(1)
        print("✅ Gemini API key found.")
    elif 'ollama' in _AGENT_NAME.value.lower():
        print("✅ Using Ollama (no API key required).")


def _get_agent(env: interface.AsyncEnv) -> base_agent.EnvironmentInteractingAgent:
    """Dynamically initializes and returns the specified agent using the registry."""
    agent_name = _AGENT_NAME.value
    print(f"🤖 Initializing agent: {agent_name}...")

    # Collect all potential agent-related kwargs from flags.
    # The get_agent function in the registry will handle passing the correct ones.
    agent_kwargs = {
        'verbose': _VERBOSE.value,
        'temperature': _TEMPERATURE.value,
        'max_retry': _MAX_RETRY.value,
        'wait_after_action_seconds': _WAIT_AFTER_ACTION.value,
        'top_p': _TOP_P.value,
        'enable_safety_checks': _ENABLE_SAFETY_CHECKS.value,
        'host': _HOST.value,
        'port': _PORT.value,
    }
    # Handle model_name separately since it has defaults.
    if _MODEL_NAME.value:
        agent_kwargs['model_name'] = _MODEL_NAME.value

    return agent_registry.get_agent(agent_name, env, **agent_kwargs)


def _main(_: Sequence[str]) -> None:
    """Main function to run a single task."""
    _check_api_keys()

    env = env_launcher.load_and_setup_env(
        console_port=_DEVICE_CONSOLE_PORT.value,
        emulator_setup=_EMULATOR_SETUP.value,
        adb_path=_ADB_PATH.value,
    )
    env.reset(go_home=True)

    agent = _get_agent(env)

    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)

    if _TASK.value and _TASK.value in aw_registry:
        task_type: Type[task_eval.TaskEval] = aw_registry[_TASK.value]
        print(f"🚀 Running specified task: {task_type.__name__}")
    else:
        if _TASK.value:
            print(f"⚠️ Task '{_TASK.value}' not found. Selecting a random task.")
        task_type: Type[task_eval.TaskEval] = random.choice(list(aw_registry.values()))
        print(f"🎲 Running random task: {task_type.__name__}")

    params = task_type.generate_random_params()
    task = task_type(params)
    task.initialize_task(env)

    print('Goal: ' + str(task.goal))
    print(f"🎯 Max steps: {int(task.complexity * 10)}")
    print(f"🤖 Agent: {_AGENT_NAME.value} | Model: {_MODEL_NAME.value or 'default'}")
    print("=" * 80)
    
    is_done = False
    max_steps = int(task.complexity * 10)
    
    for step in range(max_steps):
        if _VERBOSE.value:
            print(f"\n{'='*30} Step {step + 1}/{max_steps} {'='*30}")
        else:
            print(f"Step {step + 1}...")
        
        try:
            response = agent.step(task.goal)
            if response.done:
                is_done = True
                print(f"✅ Agent indicated task completion at step {step + 1}.")
                break
            
            if _VERBOSE.value and 'summary' in response.data:
                print(f"Step summary: {response.data['summary']}")
                
        except Exception as e:
            print(f"❌ An error occurred during step {step + 1}: {e}")
            logging.exception("Exception details:")
            break
            
    if not is_done:
        print(f"⏰ Agent did not complete the task within {max_steps} steps.")

    # --- Evaluation ---
    task_success_score = task.is_successful(env)
    agent_successful = is_done and task_success_score == 1.0

    print("\n" + "="*30 + " FINAL RESULTS " + "="*30)
    if agent_successful:
        print(f'🎉 Task Passed ✅: {task.goal}')
    else:
        print(f'💔 Task Failed ❌: {task.goal}')
        if not is_done:
            print("   Reason: Agent reached max steps or an error occurred.")
        elif task_success_score != 1.0:
            print(f"   Reason: Task success check failed (score: {task_success_score}).")

    print("\n" + "="*30 + " RUN SUMMARY " + "="*31)
    print(f"  Task: {task_type.__name__}")
    print(f"  Agent: {_AGENT_NAME.value}")
    if _MODEL_NAME.value:
      print(f"  Model: {_MODEL_NAME.value}")
    print(f"  Steps Taken: {step + 1 if 'step' in locals() else 0}/{max_steps}")
    print(f"  Agent Completed: {'Yes' if is_done else 'No'}")
    print(f"  Task Successful: {'Yes' if task_success_score == 1.0 else 'No'}")
    print(f"  Overall Success: {'Yes' if agent_successful else 'No'}")
    print("=" * 75)

    env.close()


if __name__ == '__main__':
    app.run(_main) 