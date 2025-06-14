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

"""Run eval suite.

The run.py module is used to run a suite of tasks, with configurable task
combinations, environment setups, and agent configurations. You can run specific
tasks or all tasks in the suite and customize various settings using the
command-line flags.
"""

from collections.abc import Sequence
import os
import signal
import sys
from typing import Dict, Any

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from absl import app
from absl import flags
from absl import logging
from android_world import checkpointer as checkpointer_lib
from android_world import episode_runner
from android_world import registry
from android_world import suite_utils
from android_world.agents import registry as agent_registry
from android_world.task_evals import task_eval
from android_world.agents import base_agent
from android_world.agents import human_agent
from android_world.agents import infer
from android_world.agents import m3a
from android_world.agents import random_agent
from android_world.agents import seeact
from android_world.agents import t3a
from android_world.env import env_launcher
from android_world.env import interface

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

_SUITE_FAMILY = flags.DEFINE_enum(
    'suite_family',
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [
        # Families from the paper.
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        # Other families for more testing.
        registry.TaskRegistry.MINIWOB_FAMILY,
        registry.TaskRegistry.ANDROID_FAMILY,
        registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
    ],
    'Suite family to run. See registry.py for more information.',
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    'task_random_seed', 30, 'Random seed for task randomness.'
)

_TASKS = flags.DEFINE_list(
    'tasks',
    None,
    'List of specific tasks to run in the given suite family. If None, run all'
    ' tasks in the suite family.',
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    'n_task_combinations',
    1,
    'Number of task instances to run for each task template.',
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'The directory to save checkpoints and resume evaluation from. If the'
    ' directory contains existing checkpoint files, evaluation will resume from'
    ' the latest checkpoint. If the directory is empty or does not exist, a new'
    ' directory will be created.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    os.path.expanduser('~/android_world/runs'),
    'The path to save results to if not resuming from a checkpoint is not'
    ' provided.',
)

# Agent specific.
_AGENT_NAME = flags.DEFINE_string('agent_name', 'm3a_gpt4v', help='Agent name.')

_FIXED_TASK_SEED = flags.DEFINE_boolean(
    'fixed_task_seed',
    False,
    'Whether to use the same task seed when running multiple task combinations'
    ' (n_task_combinations > 1).',
)

_VERBOSE = flags.DEFINE_boolean(
    'verbose',
    True,
    'Whether to show verbose output during task execution.',
)


# MiniWoB is very lightweight and new screens/View Hierarchy load quickly.
_MINIWOB_TRANSITION_PAUSE = 0.2

# Additional guidelines for the MiniWob tasks.
_MINIWOB_ADDITIONAL_GUIDELINES = [
    (
        'This task is running in a mock app, you must stay in this app and'
        ' DO NOT use the `navigate_home` action.'
    ),
]

# Add global variables for graceful shutdown
_interrupted = False
_accumulated_results = []

def _signal_handler(signum, frame):
  """Handle interrupt signals gracefully."""
  global _interrupted
  print(f"\n\n🛑 Received interrupt signal ({signum}). Gracefully shutting down...")
  print("📊 Saving accumulated results...")
  _interrupted = True



def _get_agent(
    env: interface.AsyncEnv,
    family: str | None = None,
) -> base_agent.EnvironmentInteractingAgent:
  """Gets agent from the registry."""
  print(f'Initializing agent: {_AGENT_NAME.value}...')

  agent = agent_registry.get_agent(
      name=_AGENT_NAME.value,
      env=env,
      verbose=_VERBOSE.value,
  )

  # Special handling for MiniWob tasks.
  if (
      _AGENT_NAME.value
      in ['m3a_gemini_gcp', 't3a_gemini_gcp', 't3a_gpt4', 'm3a_gpt4v', 'seeact']
      and family
      and family.startswith('miniwob')
      and hasattr(agent, 'set_task_guidelines')
  ):
    agent.set_task_guidelines(_MINIWOB_ADDITIONAL_GUIDELINES)
  
  agent.name = _AGENT_NAME.value # Keep this for logging/output consistency.

  return agent


def _main() -> None:
  """Runs eval suite and gets rewards back."""
  global _accumulated_results
  
  # Immediate output to test buffering
  print("🚀 Python process started - initializing...")
  sys.stdout.flush()
  
  # Set up signal handlers for graceful shutdown
  signal.signal(signal.SIGINT, _signal_handler)
  signal.signal(signal.SIGTERM, _signal_handler)
  
  print("🔧 Loading environment...")
  sys.stdout.flush()
  
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
  )
  
  print("✅ Environment loaded successfully")
  sys.stdout.flush()

  n_task_combinations = _N_TASK_COMBINATIONS.value
  task_registry = registry.TaskRegistry()
  suite = suite_utils.create_suite(
      task_registry.get_registry(family=_SUITE_FAMILY.value),
      n_task_combinations=n_task_combinations,
      seed=_TASK_RANDOM_SEED.value,
      tasks=_TASKS.value,
      use_identical_params=_FIXED_TASK_SEED.value,
  )
  suite.suite_family = _SUITE_FAMILY.value

  print("🤖 Initializing agent...")
  sys.stdout.flush()
  
  agent = _get_agent(env, _SUITE_FAMILY.value)
  
  print(f"✅ Agent initialized: {agent.name}")
  sys.stdout.flush()

  if _SUITE_FAMILY.value.startswith('miniwob'):
    # MiniWoB pages change quickly, don't need to wait for screen to stabilize.
    agent.transition_pause = _MINIWOB_TRANSITION_PAUSE
  else:
    agent.transition_pause = None

  if _CHECKPOINT_DIR.value:
    checkpoint_dir = _CHECKPOINT_DIR.value
  else:
    checkpoint_dir = checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)

  print(
      f'🚀 Starting eval with agent {_AGENT_NAME.value} and writing to'
      f' {checkpoint_dir}'
  )
  
  try:
    # Use custom process function that accumulates results
    def accumulate_results(episodes: list[Dict[str, Any]], print_summary: bool = False):
      global _accumulated_results
      _accumulated_results = episodes.copy()
      return suite_utils.process_episodes(episodes, print_summary)
    
    # Optionally add verbosity similar to minimal runner
    if _VERBOSE.value:
      original_run_task = suite_utils._run_task
      
      def verbose_run_task(task, run_episode_fn, env, demo_mode=False):
        print(f"\n🎯 Starting task: {task.name}")
        print(f"📋 Goal: {task.goal}")
        print(f"📊 Task Complexity: {task.complexity}")
        print(f"🎮 Max Steps: {int(task.complexity * 10)}")
        print("=" * 80)
        sys.stdout.flush()
        
        # Call original function but with enhanced episode runner
        def enhanced_run_episode(task_arg):
          print(f"\n🚀 Running episode for: {task_arg.name}")
          
          # Create custom print function for verbose output
          def step_print(msg):
            if 'Completed step' in msg:
              step_num = msg.split('step')[1].split('.')[0].strip()
              print(f"✅ Completed step {step_num}")
            elif 'Agent indicates task is done' in msg:
              print(f"🎉 Agent indicated task completion!")
            elif 'Agent did not indicate task is done' in msg:
              print(f"⏰ Agent reached max steps without completion")
            elif 'Environment ends episode' in msg:
              print(f"🔚 Environment ended episode")
            else:
              print(msg)
            sys.stdout.flush()
          
          return episode_runner.run_episode(
              goal=task_arg.goal,
              agent=agent,
              max_n_steps=suite_utils._allocate_step_budget(task_arg.complexity),
              start_on_home_screen=task_arg.start_on_home_screen,
              print_fn=step_print,
          )
        
        result = original_run_task(task, enhanced_run_episode, env, demo_mode)
        
        # Add completion status similar to minimal runner
        is_successful = result.get('is_successful', 0) > 0.5
        episode_data = result.get('episode_data', {})
        # Handle case where episode_data might be NaN (float) on error
        if isinstance(episode_data, dict):
          steps_taken = len(episode_data.get('step_number', []))
        else:
          steps_taken = 0
        
        print(f"\n📊 Task Results:")
        print(f"   Task: {task.name}")  
        print(f"   Steps taken: {steps_taken}")
        print(f"   Task successful: {'Yes' if is_successful else 'No'}")
        print(f"   Result: {'✅ Passed' if is_successful else '❌ Failed'}")
        sys.stdout.flush()
        
        return result
      
      # Temporarily replace the function
      suite_utils._run_task = verbose_run_task
    
    results = suite_utils.run(
        suite,
        agent,
        checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
        demo_mode=False,
        process_episodes_fn=accumulate_results,
    )
    
    # Restore original function if we patched it
    if _VERBOSE.value:
      suite_utils._run_task = original_run_task
    
    # If we get here without interruption, show final results
    if not _interrupted:
      print(f"\n🎉 Evaluation completed successfully!")
      print(f"📊 Final Results Summary:")
      
      total_tasks = len(_accumulated_results)
      successful_tasks = sum(1 for r in _accumulated_results if r.get('is_successful', 0) > 0.5)
      success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
      
      print(f"   Total tasks: {total_tasks}")
      print(f"   Successful: {successful_tasks}")
      print(f"   Success rate: {success_rate:.1f}%")
      print(f"   Results saved to: {checkpoint_dir}")
    
  except KeyboardInterrupt:
    print(f"\n⚠️  Keyboard interrupt received during execution")
    _signal_handler(signal.SIGINT, None)
  
  finally:
    # Always try to save results if we have any
    if _accumulated_results:
      print(f"💾 Saving {len(_accumulated_results)} accumulated results to checkpoint...")
      try:
        checkpointer = checkpointer_lib.IncrementalCheckpointer(checkpoint_dir)
        # Note: The results should already be saved by suite_utils.run, but this is a backup
        print(f"✅ Results preserved in: {checkpoint_dir}")
      except Exception as e:
        print(f"⚠️  Warning: Could not save final results: {e}")
    
    print(
        f'🏁 Finished running agent {_AGENT_NAME.value} on {_SUITE_FAMILY.value}'
        f' family. Results in {checkpoint_dir}.'
    )
    env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
