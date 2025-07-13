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

"""Agent registry for AndroidWorld."""

import inspect
from typing import Callable, Dict
from android_world.agents.custom import cool_agent
from android_world.agents import base_agent
from android_world.env import interface

# Agent factories can be a class or a function that returns an agent instance.
AgentFactory = Callable[..., base_agent.EnvironmentInteractingAgent]

_AGENT_REGISTRY: Dict[str, AgentFactory] = {}


def register(name: str, factory: AgentFactory):
    """Registers an agent factory."""
    if name in _AGENT_REGISTRY:
        raise ValueError(f"Agent '{name}' is already registered.")
    _AGENT_REGISTRY[name] = factory


def get_agent(name: str, env: interface.AsyncEnv, **kwargs) -> base_agent.EnvironmentInteractingAgent:
    """
    Initializes and returns an agent by its registered name, intelligently
    passing only the arguments that the agent's factory accepts.
    """
    if name not in _AGENT_REGISTRY:
        import difflib
        matches = difflib.get_close_matches(name, list_agents())
        err_msg = f"Unknown agent: '{name}'."
        if matches:
            err_msg += f" Did you mean: {', '.join(matches)}?"
        raise ValueError(err_msg)

    factory = _AGENT_REGISTRY[name]

    # Inspect the factory's signature to determine which kwargs to pass.
    target = factory.__init__ if inspect.isclass(factory) else factory
    sig = inspect.signature(target)
    
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        # If the factory accepts **kwargs, pass everything.
        final_kwargs = kwargs
    else:
        # Otherwise, only pass arguments that are explicitly in the signature.
        final_kwargs = {
            key: value for key, value in kwargs.items() if key in sig.parameters
        }

    return factory(env=env, **final_kwargs)


def list_agents() -> list[str]:
    """Returns a list of all registered agent names."""
    return sorted(_AGENT_REGISTRY.keys())

# --- Agent Registration ---
# Centralized registration to avoid circular dependencies and provide a clear
# overview of available agents.

def _register_all_agents():
    """Imports and registers all custom agents."""
    from android_world.agents.custom import m3a_openrouter
    from android_world.agents.custom import m3a_gemini_gemma
    from android_world.agents.custom import v_droid_agent
    from android_world.agents.custom import v_droid_ollama_agent
    from android_world.agents.custom import celil_agent

    # Register your custom agents here
    register('m3a_openrouter_agent', m3a_openrouter.M3AOpenRouter)
    register('m3a_gemini_gemma_agent', m3a_gemini_gemma.M3AGeminiGemma)
    register('v_droid_agent', v_droid_agent.VDroidAgent)
    register('v_droid_ollama_agent', v_droid_ollama_agent.VDroidOllamaAgent)
    register('celil_agent', celil_agent.CelilAgent)
    register('cool_agent', cool_agent.CoolAgent)

# Perform registration automatically when the module is imported.
_register_all_agents() 