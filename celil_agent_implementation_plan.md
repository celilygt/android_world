## **Implementation Plan: The Conductor Agent (`celil_agent`)**

### **FEATURE 1: Core Agent Scaffolding & Framework Integration**

*This feature establishes the basic structure of the `celil_agent`, registers it with the AndroidWorld framework, and makes it configurable.*

**Step 1.1: Create the Agent File and Class**
- **Action:** Create a new file at `android_world/agents/custom/celil_agent.py`.
- **Action:** Inside this file, define a new class `CelilAgent` that inherits from `base_agent.EnvironmentInteractingAgent`.
- **Action:** Implement the `__init__` method. It should accept `env`, `name`, and model/persona configuration parameters. For now, initialize placeholders for the main modules (`self.maestro`, `self.section_leader`, `self.quality_control`, `self.memory`, `self.perception`, `self.tools`).
- **Action:** Implement a skeleton `step(self, goal: str)` method that logs a message (e.g., "CelilAgent step called.") and returns a `base_agent.AgentInteractionResult(done=True, data={})` to prevent errors.
- **Acceptance Criteria:** The file `celil_agent.py` exists and contains the `CelilAgent` class definition without syntax errors.
- **Completed:** (True / False)

**Step 1.2: Register the Agent**
- **Action:** Open the file `android_world/agents/registry.py`.
- **Action:** Inside the `_register_all_agents()` function, add the following lines:
    ```python
    from android_world.agents.custom import celil_agent
    register('celil_agent', celil_agent.CelilAgent)
    ```
- **Acceptance Criteria:** The `celil_agent` is now available in the agent registry and can be selected by name.
- **Completed:** (True / False)

**Step 1.3: Add Configuration to YAML**
- **Action:** Open the file `config/default.yaml`.
- **Action:** Under `agent.configurations`, add a new configuration block for `celil_agent`.
    ```yaml
    celil_agent:
      registry_name: "celil_agent"
      maestro_verifier_model_name: "auto" # Use Gemini smart routing
      # We will add more config options later
    ```
- **Action:** Set `active_agent: "celil_agent"` in the YAML to make it the default for testing.
- **Acceptance Criteria:** Running `./run_benchmark.sh` now attempts to initialize and run `CelilAgent`. The skeleton `step` method should execute and the task should terminate immediately.
- **Completed:** (True / False)

---

### **FEATURE 2: Perception Module**

*This feature creates a dedicated module to process raw environment observations into a structured format that all other modules can use. It offloads OCR and UI parsing from the LLMs.*

**Step 2.1: Create the Perception Module File**
- **Action:** Create a new file: `android_world/agents/custom/celil_perception.py`.
- **Action:** Inside, define a new class `PerceptionModule`.
- **Acceptance Criteria:** The file and class exist.
- **Completed:** (True / False)

**Step 2.2: Implement Observation Processing**
- **Action:** In `PerceptionModule`, create a method `process_observation(self, state: interface.State) -> dict`.
- **Action:** This method will perform the following steps:
    1.  Import `pytesseract` and `PIL.Image`.
    2.  Get the screenshot from `state.pixels`.
    3.  Use `pytesseract.image_to_data(Image.fromarray(screenshot), output_type=pytesseract.Output.DICT)` to extract text and bounding boxes.
    4.  Iterate through the `pytesseract` output to build a list of OCR results, where each item is a dictionary: `{'text': '...', 'bbox': (left, top, width, height)}`.
    5.  Get the UI tree from `state.ui_elements`.
    6.  Return a single dictionary containing the structured observation:
        ```python
        {
            "screenshot": state.pixels, # The raw numpy array
            "ocr_results": ocr_list, # The list of dicts from step 4
            "ui_tree": state.ui_elements # The original UI elements list
        }
        ```
- **Acceptance Criteria:** The `process_observation` method correctly takes a `State` object and returns the specified dictionary structure.
- **Completed:** (True / False)

**Step 2.3: Integrate Perception Module into Agent**
- **Action:** In `celil_agent.py`, import `PerceptionModule`.
- **Action:** In the `CelilAgent.__init__`, instantiate it: `self.perception_module = PerceptionModule()`.
- **Acceptance Criteria:** The agent now has a perception module instance.
- **Completed:** (True /False)

---

### **FEATURE 3: Memory System**

*This feature implements the short-term working memory and long-term episodic memory to manage context.*

**Step 3.1: Create the Memory Module File**
- **Action:** Create a new file: `android_world/agents/custom/celil_memory.py`.
- **Action:** Inside this file, define two classes: `WorkingMemory` and `EpisodicMemory`.
- **Acceptance Criteria:** The file and classes exist.
- **Completed:** (True / False)

**Step 3.2: Implement Working Memory**
- **Action:** Implement the `WorkingMemory` class with the following methods:
    - `__init__(self, task_goal: str)`: Initializes an internal dictionary `self.data` with keys `task_goal` and `history` (an empty list).
    - `add_step(self, step_details: dict)`: Appends the `step_details` dictionary to the `self.data['history']` list.
    - `get_context_summary(self, max_steps=3) -> str`: Returns a formatted string of the last `max_steps` from the history for use in prompts.
    - `set_plan(self, plan: list[str])`: Sets a `maestro_plan` key in `self.data`.
    - `get_plan(self) -> list[str]`: Returns the current plan.
    - `pop_sub_goal(self) -> str`: Removes and returns the next sub-goal from the plan.
    - `clear(self)`: Resets the internal data.
- **Acceptance Criteria:** The `WorkingMemory` class functions as specified, managing state for a single task.
- **Completed:** (True / False)

**Step 3.3: Implement Episodic Memory (Simplified)**
- **Action:** Implement the `EpisodicMemory` class with the following methods:
    - `__init__(self, db_path: str = "celil_episodic_memory.json")`: Initializes the path to a JSON file. Calls a `_load()` method.
    - `_load(self)`: If the JSON file at `db_path` exists, load it into a list `self.memory`. If not, initialize `self.memory` as an empty list.
    - `_save(self)`: Dumps `self.memory` to the JSON file.
    - `add_successful_plan(self, goal_template: str, plan: list[str])`: Adds a new dictionary `{"goal_template": ..., "plan": ...}` to `self.memory` and calls `_save()`.
    - `find_similar_task(self, goal: str) -> list[str] | None`: A simple implementation for now. It iterates through `self.memory` and if `goal_template` is a substring of the current `goal`, it returns the associated plan. Otherwise, returns `None`.
- **Acceptance Criteria:** The `EpisodicMemory` class can persist successful plans to a file and retrieve them.
- **Completed:** (True / False)

**Step 3.4: Integrate Memory into Agent**
- **Action:** In `celil_agent.py`, import both memory classes.
- **Action:** In the `CelilAgent.__init__`, instantiate the episodic memory: `self.episodic_memory = EpisodicMemory()`.
- **Action:** The `WorkingMemory` will be instantiated at the start of each task in the `step` method logic. The `reset` method of the agent should ensure `self.working_memory` is cleared.
- **Acceptance Criteria:** The agent has access to both memory systems.
- **Completed:** (True / False)

---

### **FEATURE 4: The Maestro (Planner) Module**

*This feature implements the high-level planner that uses a powerful LLM to create and correct plans.*

**Step 4.1: Create the Maestro Module File**
- **Action:** Create a new file: `android_world/agents/custom/celil_maestro.py`.
- **Action:** Define a `MaestroPlanner` class.
- **Action:** The `__init__` method should accept an `llm_wrapper` instance (which will be a `GeminiGemmaWrapper`).
- **Acceptance Criteria:** The file and class exist.
- **Completed:** (True / False)

**Step 4.2: Implement Initial Plan Generation**
- **Action:** In `MaestroPlanner`, create `generate_initial_plan(self, goal: str, observation: dict, retrieved_plan: list[str] | None) -> list[str]`.
- **Action:** Construct the prompt using the following template. Note how it incorporates the retrieved plan from episodic memory if available.
    ```python
    PLANNER_PROMPT_TEMPLATE = """You are a master strategist for an Android agent. Your task is to break down a high-level goal into a series of simple, logical sub-goals.

    Goal: "{goal}"

    Current Screen Observation:
    {ocr_summary}
    {ui_tree_summary}

    {retrieved_plan_section}

    Based on all of this information, provide a numbered list of sub-goals. Each sub-goal should be a single, clear instruction.
    Example output:
    1. Open the Files app.
    2. Navigate to the 'Downloads' folder.
    3. Select the file 'report.pdf'.
    4. Activate the 'delete' function.
    5. Confirm the deletion.
    """
    # Helper functions _create_ocr_summary and _create_ui_tree_summary will be needed.
    ```
- **Action:** Call `self.llm_wrapper.predict_mm()` with the prompt and screenshot.
- **Action:** Parse the LLM's numbered list response into a Python `list` of strings.
- **Acceptance Criteria:** The method successfully queries the LLM and returns a list of sub-goal strings.
- **Completed:** (True / False)

**Step 4.3: Implement Corrective Plan Generation**
- **Action:** In `MaestroPlanner`, create `generate_corrective_plan(self, failure_context: dict) -> list[str]`. The `failure_context` will contain the original plan, failed action, verifier feedback, and new observation.
- **Action:** Construct the prompt using the following template:
    ```python
    CORRECTIVE_PROMPT_TEMPLATE = """You are an expert agent supervisor. An agent under your command has failed. Analyze the situation and provide a new, concise plan to recover and achieve the original goal.

    Original Goal: "{goal}"
    Original Plan: {original_plan}
    The agent failed at sub-goal: "{failed_sub_goal}"
    It tried to execute: "{failed_action}"
    Our verifiers said: "{verifier_feedback}"
    This resulted in the following screen:
    {ocr_summary}
    {ui_tree_summary}

    Provide a new, complete, numbered plan to achieve the original goal from this new state.
    """
    ```
- **Action:** Call the LLM and parse the response, similar to the initial plan generation.
- **Acceptance Criteria:** The method can generate a recovery plan based on failure context.
- **Completed:** (True / False)

---

### **FEATURE 5: The Section Leader (Action Generator) Module**

*This feature creates a module to run a local UI-TARS model for low-level action generation.*

**Step 5.1: Create the Section Leader Module File**
- **Action:** Create a new file: `android_world/agents/custom/celil_section_leader.py`.
- **Action:** Define a `UITarsActionGenerator` class. The `__init__` will take a local model name (e.g., a path to a GGUF file or an Ollama model tag).
- **Action:** In `__init__`, set up the client to communicate with the local LLM (e.g., using the `ollama` Python library).
- **Acceptance Criteria:** The file and class exist, ready to be implemented.
- **Completed:** (True / False)

**Step 5.2: Implement Action Generation**
- **Action:** In `UITarsActionGenerator`, create a method `generate_action(self, sub_goal: str, observation: dict, context_summary: str) -> dict`.
- **Action:** Construct the prompt for the UI-TARS model. This will be similar to the M3A prompt but simplified, as the high-level reasoning is handled by the Maestro.
    ```python
    UI_TARS_PROMPT_TEMPLATE = """You are a phone operator. Your current task is: {sub_goal}.
    Recent actions summary: {context_summary}
    Current screen elements:
    {annotated_observation}

    Based on the screen and your task, generate the single next JSON action to perform.
    Action:
    """
    # The annotated_observation will be a string representation of the OCR and UI tree data.
    ```
- **Action:** Call the local LLM via the client initialized in `__init__`.
- **Action:** Use the `agent_utils.extract_json` function from the provided files to parse the JSON from the model's output string.
- **Acceptance Criteria:** The method can query a local UI-TARS model and return a valid action dictionary.
- **Completed:** (True / False)

---

### **FEATURE 6: Quality Control Module**

*This feature implements the verifier ensemble to score actions before execution.*

**Step 6.1: Create the Quality Control Module File**
- **Action:** Create a new file: `android_world/agents/custom/celil_quality_control.py`.
- **Action:** Define a `VerifierEnsemble` class. The `__init__` should accept an `llm_wrapper`.
- **Acceptance Criteria:** The file and class exist.
- **Completed:** (True / False)

**Step 6.2: Implement the Verification Method**
- **Action:** In `VerifierEnsemble`, create the main method `verify_action(self, sub_goal: str, proposed_action: dict, observation: dict, context_summary: str) -> tuple[float, dict]`.
- **Action:** Inside this method:
    1.  Define the three persona prompt templates (Pragmatist, Skeptic, Efficiency Expert) as discussed previously.
    2.  Make three *parallel* (or sequential, for simplicity) calls to `self.llm_wrapper.predict_mm()`, one for each persona prompt.
    3.  For each response, parse the JSON to get the score and reasoning. Handle parsing errors gracefully (e.g., assign a low score).
    4.  Store the reasoning from all three verifiers.
    5.  Calculate the final score using the weighted average: `(0.4 * S_prag) + (0.5 * S_skep) + (0.1 * S_eff)`.
    6.  Return the final score and a dictionary of all the reasoning texts.
- **Acceptance Criteria:** The module can take a proposed action and return a single, robust confidence score based on the ensemble's feedback.
- **Completed:** (True / False)

---

### **FEATURE 7: Main Agent Loop & Integration**

*This is the most critical feature, where all the previously built modules are orchestrated within the `celil_agent.py`'s `step` method.*

**Step 7.1: Flesh out `CelilAgent.__init__`**
- **Action:** Update the `__init__` method in `celil_agent.py`.
- **Action:** Instantiate all the modules: `self.maestro`, `self.section_leader`, `self.quality_control`, etc. Pass the configured model wrappers to them. For the local UI-TARS, you might need a separate wrapper or a direct client. The `gemini_gemma_wrapper` can be used for both Maestro and Verifiers.
- **Acceptance Criteria:** The `CelilAgent` instance is fully initialized with all its components.
- **Completed:** (True / False)

**Step 7.2: Implement the `step` Method Logic**
- **Action:** Re-implement the `step` method in `celil_agent.py` to follow the Conductor logic.
    1.  At the first step, or if there is no plan, call `self.episodic_memory.find_similar_task` and then `self.maestro.generate_initial_plan`. Store the plan in `self.working_memory`.
    2.  Pop the next `sub_goal` from the working memory's plan.
    3.  Call `self.perception_module.process_observation` to get the current state.
    4.  Call `self.section_leader.generate_action` with the sub-goal and observation.
    5.  Call `self.quality_control.verify_action` on the proposed action.
    6.  **Decision Gate:** If the returned score is `>= 7.0`, proceed. If not, trigger the Rehearsal loop (see Feature 8) and return from the current step.
    7.  If the action is approved, use `self.env.execute_action` to perform it.
    8.  Perform a simple post-execution check (e.g., did the screen change?). Log the outcome.
    9.  Call `self.working_memory.add_step()` with all the details of the step (observation, action, scores, outcome).
    10. If the plan is now empty, generate a `status: complete` action.
    11. Return `AgentInteractionResult(done=False, data=...)`.
- **Acceptance Criteria:** The agent can execute a multi-step plan, using its components in the correct sequence.
- **Completed:** (True / False)

---

### **FEATURE 8: Self-Correction (Rehearsal) Loop**

*This feature implements the agent's ability to recover from failure.*

**Step 8.1: Implement the Rehearsal Trigger Logic**
- **Action:** Inside the `step` method of `CelilAgent`, after the Decision Gate (Step 7.6) and the post-execution check (Step 7.8), add the logic to handle failures.
- **Action:** If an action is rejected OR if it's executed but the post-check fails:
    1.  Gather the failure context: current goal, full plan, the failed sub-goal, the failed action, the verifier feedback, and the *new* observation after the failed action.
    2.  Log this entire context to the `working_memory`.
    3.  Call `self.maestro.generate_corrective_plan()` with this context.
    4.  Replace the old plan in `working_memory` with the new corrective plan.
    5.  End the current `step` call (return `AgentInteractionResult(done=False, ...)`). The next call to `step` will start executing the new corrective plan.
- **Acceptance Criteria:** When an action fails, the agent does not get stuck but instead generates and adopts a new plan to continue the task.
- **Completed:** (True / False)