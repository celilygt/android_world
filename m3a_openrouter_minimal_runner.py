#!/usr/bin/env python3
# Copyright 2025 Celil Yiğit & Sina Mohammad Rezaei
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

"""Runs a single task with M3A OpenRouter Agent.

This script uses the M3A agent with OpenRouter's free models (like Gemma 3-27B)
instead of paid APIs. Requires OPENROUTER_API_KEY environment variable.
"""

from collections.abc import Sequence
import os
import random
import sys
from typing import Type

from absl import app
from absl import flags
from absl import logging
from android_world import registry
from android_world.agents.m3a_openrouter import M3AOpenRouter
from android_world.env import env_launcher
from android_world.task_evals import task_eval

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing
# Additional gRPC error suppression
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
      'adb not found in the common Android SDK paths. Please install Android'
      " SDK and ensure adb is in one of the expected directories. If it's"
      ' already installed, point to the installed location.'
  )


_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    _find_adb_directory(),
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    False,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)

_TASK = flags.DEFINE_string(
    'task',
    None,
    'A specific task to run. If not provided, a random task will be selected.',
)

_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    'google/gemma-3-27b-it:free',
    'OpenRouter model to use. Available free models: '
    'google/gemma-3-27b-it:free, meta-llama/llama-3.3-70b-instruct:free, '
    'mistralai/mistral-7b-instruct:free',
)

_TEMPERATURE = flags.DEFINE_float(
    'temperature',
    0.0,
    'Temperature for LLM generation. 0.0 = deterministic, 1.0 = more random.',
)

_MAX_RETRY = flags.DEFINE_integer(
    'max_retry',
    3,
    'Maximum number of retries when calling the LLM.',
)

_WAIT_AFTER_ACTION = flags.DEFINE_float(
    'wait_after_action_seconds',
    2.0,
    'Seconds to wait after each action for the screen to stabilize.',
)

_VERBOSE = flags.DEFINE_boolean(
    'verbose',
    True,
    'Whether to show verbose output from the M3A agent.',
)


def _check_openrouter_key() -> None:
  """Check if OpenRouter API key is set."""
  if 'OPENROUTER_API_KEY' not in os.environ:
    print("❌ OPENROUTER_API_KEY environment variable not set!")
    print("\n📋 Setup Instructions:")
    print("1. Sign up at https://openrouter.ai (free)")
    print("2. Get your API key from the dashboard")
    print("3. Set the environment variable:")
    print("   export OPENROUTER_API_KEY='your_api_key_here'")
    print("\n💡 Available free models:")
    print("   - google/gemma-3-27b-it:free (default)")
    print("   - meta-llama/llama-3.3-70b-instruct:free")
    print("   - mistralai/mistral-7b-instruct:free")
    sys.exit(1)
  else:
    print("✅ OpenRouter API key found!")


def _main() -> None:
  """Runs a single task with M3A OpenRouter Agent."""
  
  # Check OpenRouter API key
  _check_openrouter_key()
  
  print(f"🤖 Using M3A OpenRouter Agent with model: {_MODEL_NAME.value}")
  
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
  )
  env.reset(go_home=True)
  
  task_registry = registry.TaskRegistry()
  aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
  
  if _TASK.value:
    if _TASK.value not in aw_registry:
      raise ValueError(f'Task {_TASK.value} not found in registry.')
    task_type: Type[task_eval.TaskEval] = aw_registry[_TASK.value]
    print(f"Running task: {_TASK.value}")
  else:
    task_type: Type[task_eval.TaskEval] = random.choice(list(aw_registry.values()))
    print(f"Running random task: {task_type.__name__}")
  
  params = task_type.generate_random_params()
  task = task_type(params)
  task.initialize_task(env)
  
  # Create M3A OpenRouter agent with specified configuration
  agent = M3AOpenRouter(
      env=env,
      model_name=_MODEL_NAME.value,
      name='M3A-OpenRouter',
      temperature=_TEMPERATURE.value,
      max_retry=_MAX_RETRY.value,
      wait_after_action_seconds=_WAIT_AFTER_ACTION.value,
      verbose=_VERBOSE.value,
  )

  print('Goal: ' + str(task.goal))
  print(f"🎯 Max steps allowed: {int(task.complexity * 10)}")
  print(f"📋 Current Task: {task_type.__name__}")
  print(f"🤖 Model: {_MODEL_NAME.value}")
  print(f"🎮 Agent: M3A OpenRouter")
  print("=" * 80)
  
  is_done = False
  max_steps = int(task.complexity * 10)
  
  for step in range(max_steps):
    if _VERBOSE.value:
      print(f"\n{'='*50}")
      print(f"Step {step + 1}/{max_steps}")
      print(f"{'='*50}")
    else:
      print(f"Step {step + 1}...")
    
    try:
      response = agent.step(task.goal)
      if response.done:
        is_done = True
        print(f"✅ Agent indicated task completion at step {step + 1}")
        break
        
      if _VERBOSE.value and 'summary' in response.data:
        print(f"Step summary: {response.data['summary']}")
        
    except Exception as e:
      print(f"❌ Error during step {step + 1}: {str(e)}")
      break
  
  if not is_done:
    print(f"⏰ Agent did not indicate task completion within {max_steps} steps")
  
  # Check if task was actually successful
  task_success_score = task.is_successful(env)
  agent_successful = is_done and task_success_score == 1.0
  
  if agent_successful:
    print(f'\n🎉 Task Passed ✅; {task.goal}')
  else:
    print(f'\n💔 Task Failed ❌; {task.goal}')
    if not is_done:
      print("   Reason: Agent reached max steps without completing task")
    elif task_success_score != 1.0:
      print(f"   Reason: Task success score was {task_success_score}, expected 1.0")
  
  print(f"\n📊 Final Results:")
  print(f"   Task: {task_type.__name__}")
  print(f"   Goal: {task.goal}")
  print(f"   Steps taken: {step + 1 if 'step' in locals() else 0}")
  print(f"   Agent completed: {'Yes' if is_done else 'No'}")
  print(f"   Task successful: {'Yes' if task_success_score == 1.0 else 'No'}")
  print(f"   Overall success: {'Yes' if agent_successful else 'No'}")
  print(f"   Model used: {_MODEL_NAME.value}")
  
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main) 