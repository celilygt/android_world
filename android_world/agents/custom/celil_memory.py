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

"""Memory systems for the Celil agent."""

import json
import os


class WorkingMemory:
  """Manages the short-term state for a single task."""

  def __init__(self, task_goal: str):
    """Initializes the working memory."""
    self.data = {'task_goal': task_goal, 'history': []}

  def add_step(self, step_details: dict):
    """Adds a step to the history."""
    self.data['history'].append(step_details)

  def get_context_summary(self, max_steps: int = 3) -> str:
    """Returns a formatted string of the last `max_steps` from the history."""
    summary = []
    for step in self.data['history'][-max_steps:]:
      summary.append(str(step))
    return "\n".join(summary)

  def set_plan(self, plan: list[str]):
    """Sets the Maestro's plan."""
    self.data['maestro_plan'] = plan

  def get_plan(self) -> list[str]:
    """Returns the current plan."""
    return self.data.get('maestro_plan', [])

  def pop_sub_goal(self) -> str:
    """Removes and returns the next sub-goal from the plan."""
    if not self.data.get('maestro_plan'):
      raise IndexError("Cannot pop from empty plan")
    return self.data['maestro_plan'].pop(0)

  def clear(self):
    """Resets the working memory."""
    self.data = {'task_goal': self.data['task_goal'], 'history': []}


class EpisodicMemory:
  """Manages long-term memory of successful plans."""

  def __init__(self, db_path: str = "celil_episodic_memory.json"):
    """Initializes the episodic memory."""
    self.db_path = db_path
    self._load()

  def _load(self):
    """Loads the memory from a JSON file."""
    if os.path.exists(self.db_path):
      with open(self.db_path, 'r') as f:
        self.memory = json.load(f)
    else:
      self.memory = []

  def _save(self):
    """Saves the memory to a JSON file."""
    with open(self.db_path, 'w') as f:
      json.dump(self.memory, f, indent=2)

  def add_successful_plan(self, goal_template: str, plan: list[str]):
    """Adds a successful plan to the memory."""
    self.memory.append({'goal_template': goal_template, 'plan': plan})
    self._save()

  def find_similar_task(self, goal: str) -> list[str] | None:
    """Finds a similar task in the memory."""
    for item in self.memory:
      if item['goal_template'] in goal:
        return item['plan']
    return None
