#!/bin/bash

# AndroidWorld Benchmark Runner Script v3.1
# This script orchestrates benchmark runs using a central YAML configuration.
# It handles environment setup, agent configuration, and execution, with
# improved logging, graceful shutdown, and conditional output visibility.
#
# Revision Notes (v3.1):
# - Generalized code dump logic for both celil_agent and cool_agent.
# - Fixed name discrepancies causing empty analysis logs for cool_agent.
# - Made dump file handling dynamic and extensible for future agents.

set -e # Exit on any error

# --- Color Definitions ---
C_RED='\033[0;31m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[0;33m'
C_BLUE='\033[0;34m'
C_CYAN='\033[0;36m'
C_NC='\033[0m' # No Color

# --- Help Function ---
show_help() {
    cat << EOF
AndroidWorld Benchmark Runner
=============================
Uses 'config/default.yaml' or a custom config to run benchmarks.

Usage: ./run_benchmark.sh [options]

Options:
  --config=<path>     Path to a custom YAML config file.
  --override_agent=<agent_key>
                      Run with a different agent from the config, e.g., 'm3a_gemini'.
  --skip_completed    Skip tasks that are already marked as completed in 'config/completed_tasks.yaml'.
  --help, -h          Show this help message.
EOF
}

# --- Prerequisite: yq ---
check_yq() {
    if ! command -v yq &> /dev/null; then
        echo -e "${C_RED}❌ 'yq' is required. Please install it: https://github.com/mikefarah/yq/#install${C_NC}"
        exit 1
    fi
}
check_yq

# --- Configuration Loading and Parsing ---
CONFIG_FILE="config/default.yaml"
OVERRIDE_AGENT=""
SKIP_COMPLETED=true

for arg in "$@"; do
    case $arg in
        --config=*) CONFIG_FILE="${arg#*=}"; shift;;
        --override_agent=*) OVERRIDE_AGENT="${arg#*=}"; shift;;
        --skip_completed) SKIP_COMPLETED=true; shift;;
        -h|--help) show_help; exit 0;;
    esac
done

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${C_RED}❌ Config file not found: $CONFIG_FILE${C_NC}"; exit 1
fi

# Load script and run settings
eval "$(yq e '.script | to_entries | .[] | .key + "=\"" + .value + "\""' "$CONFIG_FILE")"
eval "$(yq e '.run | to_entries | .[] | "run_" + .key + "=\"" + .value + "\""' "$CONFIG_FILE")"

# Determine active agent and load its config
active_agent_key="${OVERRIDE_AGENT:-$(yq e '.agent.active_agent' "$CONFIG_FILE")}"
echo -e "${C_GREEN}✅ Active agent key: ${C_CYAN}$active_agent_key${C_NC}"

YQ_AGENT_PATH=".agent.configurations.$active_agent_key"
if [ "$(yq e "$YQ_AGENT_PATH" "$CONFIG_FILE")" = "null" ]; then
    echo -e "${C_RED}❌ Agent configuration for '$active_agent_key' not found in $CONFIG_FILE${C_NC}"
    exit 1
fi
eval "$(yq e "$YQ_AGENT_PATH | to_entries | .[] | \"agent_\" + .key + \"=\\\"\" + .value + \"\\\"\"" "$CONFIG_FILE")"

# Load task and environment settings
eval "$(yq e '.task | to_entries | .[] | "task_" + .key + "=\"" + .value + "\""' "$CONFIG_FILE")"
eval "$(yq e '.environment | to_entries | .[] | "env_" + .key + "=\"" + .value + "\""' "$CONFIG_FILE")"

# --- Setup Agent-Specific Logic (e.g., Code Dumps) ---
DUMP_SCRIPT_NAME=""
DUMP_FILE_NAME=""

if [[ "$active_agent_key" == "celil_agent" ]]; then
    DUMP_SCRIPT_NAME="create_celil_dump.py"
    DUMP_FILE_NAME="celil_dump.md"
elif [[ "$active_agent_key" == "cool_agent" ]]; then
    DUMP_SCRIPT_NAME="create_cool_agent_dump.py"
    DUMP_FILE_NAME="cool_agent_dump.md"
else
    echo -e "${C_YELLOW}⚠️ No specific code dump script configured for agent '$active_agent_key'.${C_NC}"
fi


# --- Setup Output Logging ---
LOG_DIR="runs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$LOG_DIR/${TIMESTAMP}_${active_agent_key}"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/benchmark_run.log"
touch "$LOG_FILE"


# --- Display Final Configuration ---
echo -e "${C_YELLOW}🚀 Starting AndroidWorld Benchmark...${C_NC}"
echo -e "  ${C_BLUE}Agent:${C_NC} $agent_registry_name (from key '$active_agent_key')"
echo -e "  ${C_BLUE}Run Mode:${C_NC} $run_mode"
echo -e "  ${C_BLUE}Task:${C_NC} ${task_name:-Default (Random/All)}"
echo -e "  ${C_BLUE}Log Directory:${C_NC} $RUN_DIR"
echo "-----------------------------------------------------"


# --- Check API Keys ---
check_api_keys() {
    if [[ "$agent_registry_name" == *"openrouter"* ]] && [ -z "$OPENROUTER_API_KEY" ]; then
        echo -e "${C_RED}❌ OPENROUTER_API_KEY is not set for agent '$agent_registry_name'.${C_NC}"
        exit 1
    elif [[ "$agent_registry_name" == *"gemini"* ]] && [ -z "$GEMINI_API_KEY" ]; then
        echo -e "${C_RED}❌ GEMINI_API_KEY is not set for agent '$agent_registry_name'.${C_NC}"
        exit 1
    elif [[ "$agent_registry_name" == *"ollama"* ]]; then
        echo -e "${C_GREEN}✅ Using Ollama (local inference, no API key required).${C_NC}"
        return 0
    fi
    echo -e "${C_GREEN}✅ API key for $agent_registry_name found.${C_NC}"
}
check_api_keys


# --- Environment Setup ---
# Resolve Android SDK path from config, with a default, and expand tilde.
sdk_path_unexpanded="${env_android_sdk_root:-~/Library/Android/sdk}"
if [[ "${sdk_path_unexpanded:0:1}" == "~" ]]; then
    sdk_path="${HOME}${sdk_path_unexpanded:1}"
else
    sdk_path="$sdk_path_unexpanded"
fi

export PATH="$PATH:${sdk_path}/platform-tools:${sdk_path}/emulator"
export PYTHONUNBUFFERED=1
export GRPC_VERBOSITY=FATAL # Suppress noisy gRPC logs from emulator communication.

if [ "$activate_conda" = "true" ]; then
    echo -e "${C_YELLOW}🔧 Activating conda environment: $conda_env_name...${C_NC}"
    # Adding more robust conda initialization
    if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        echo -e "${C_RED}❌ Could not find conda.sh. Please check your conda installation.${C_NC}"
        exit 1
    fi
    conda activate "$conda_env_name"
fi

# --- Ollama Management Functions ---
check_ollama() {
    if command -v ollama >/dev/null 2>&1; then
        curl -s http://localhost:11434/api/tags >/dev/null 2>&1
    else
        return 1
    fi
}

start_ollama() {
    if ! command -v ollama >/dev/null 2>&1; then
        echo -e "${C_RED}❌ Ollama is not installed. Please install it from https://ollama.com${C_NC}"
        exit 1
    fi

    echo -e "${C_YELLOW}🚀 Starting Ollama server...${C_NC}"
    nohup ollama serve > ollama.log 2>&1 &
    OLLAMA_PID=$!

    # Wait for Ollama to start
    for i in {1..30}; do
        if check_ollama; then
            echo -e "${C_GREEN}✅ Ollama server is ready.${C_NC}"
            return 0
        fi
        sleep 1
    done

    echo -e "${C_RED}❌ Ollama failed to start within 30 seconds.${C_NC}"
    exit 1
}

# Handle Ollama for Ollama-based agents
if [[ "$agent_registry_name" == *"ollama"* ]]; then
    if ! check_ollama; then
        start_ollama
    else
        echo -e "${C_GREEN}✅ Ollama server is already running.${C_NC}"
    fi
fi

check_emulator() { adb devices 2>/dev/null | grep -q "emulator"; }
if [ "$start_emulator" = "true" ] && ! check_emulator; then
    echo -e "${C_YELLOW}📱 Starting Android emulator...${C_NC}"
    pkill -f "emulator.*AndroidWorldAvd" || true
    # The emulator log path is now configurable via YAML.
    nohup emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554 > "${env_emulator_log_path:-emulator.log}" 2>&1 &

    echo "⏳ Waiting for emulator to connect..."
    if ! adb wait-for-device; then
        echo -e "${C_RED}❌ ADB could not connect to the device.${C_NC}"
        exit 1
    fi

    echo "⏳ Waiting for emulator to fully boot (max 3 minutes)..."
    boot_completed=0
    for ((i=0; i<90; i++)); do # Wait for up to 3 minutes (90 * 2s)
        # Using `2>/dev/null` to suppress errors when shell is not ready
        if [[ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" == "1" ]]; then
            boot_completed=1
            break
        fi
        sleep 2
    done

    if [ "$boot_completed" -eq 1 ]; then
        echo -e "${C_GREEN}✅ Emulator is booted and ready.${C_NC}"
        sleep 5 # Grace period for all services to stabilize
    else
        echo -e "${C_RED}❌ Emulator failed to boot within the time limit.${C_NC}"
        echo "Check ${env_emulator_log_path:-emulator.log} for details."
        exit 1
    fi
fi

# Handle emulator setup based on config
SETUP_MARKER=".emulator_setup_completed"
run_setup=false
if [ "$perform_emulator_setup" = "force_run" ]; then
    run_setup=true
elif [ "$perform_emulator_setup" = "run_once" ] && [ ! -f "$SETUP_MARKER" ]; then
    run_setup=true
fi

if [ "$run_setup" = "true" ]; then
    echo -e "${C_YELLOW}🔧 Running AndroidWorld emulator setup (agent: $agent_registry_name)...${C_NC}"
    if python run.py --perform_emulator_setup=True --agent_name="$agent_registry_name"; then
        echo -e "${C_GREEN}✅ Emulator setup completed.${C_NC}" && touch "$SETUP_MARKER"
    else
        echo -e "${C_RED}❌ Emulator setup failed!${C_NC}"; exit 1
    fi
else
    echo "⏭️  Skipping emulator setup."
fi


# --- Build Python Command ---
build_python_command() {
    local base_cmd
    local common_flags="--agent_name=\"$agent_registry_name\""

    # Append all agent-specific flags
    for var in $(compgen -v agent_); do
        if [ "$var" != "agent_registry_name" ]; then
            key_name=$(echo "$var" | sed 's/agent_//')
            value=${!var}
            [ -n "$value" ] && common_flags="$common_flags --$key_name=\"$value\""
        fi
    done

    # Append all environment-specific flags
    for var in $(compgen -v env_); do
        key_name=$(echo "$var" | sed 's/env_//')
        if [[ "$key_name" != "android_sdk_root" && "$key_name" != "emulator_log_path" ]]; then
            value=${!var}
            [ -n "$value" ] && common_flags="$common_flags --$key_name=\"$value\""
        fi
    done

    if [ "$run_mode" = "fast" ]; then
        base_cmd="python -u minimal_task_runner.py"
        [ -n "$task_name" ] && common_flags="$common_flags --task=\"$task_name\""
    elif [ "$run_mode" = "full" ]; then
        base_cmd="python -u run.py"
        [ -n "$task_name" ] && common_flags="$common_flags --tasks=\"$task_name\""
        common_flags="$common_flags --suite_family=\"${task_suite_family:-android_world}\""
        common_flags="$common_flags --n_task_combinations=${task_n_task_combinations:-1}"
    else
        echo -e "${C_RED}❌ Invalid run mode: $run_mode.${C_NC}"; exit 1
    fi

    # Add the run log directory flag
    common_flags="$common_flags --run_log_dir=\"$RUN_DIR\""

    # Add the skip_tasks flag if enabled
    if [ "$SKIP_COMPLETED" = true ]; then
        completed_tasks=$(yq e '.completed_tasks | join(",")' config/completed_tasks.yaml)
        if [ -n "$completed_tasks" ]; then
            common_flags="$common_flags --skip_tasks=\"$completed_tasks\""
        fi
    fi

    echo "$base_cmd $common_flags"
}

PYTHON_CMD=$(build_python_command)
echo -e "${C_CYAN}🐍 Executing command:${C_NC}"
echo "$PYTHON_CMD"
echo "-----------------------------------------------------"


# --- Function to create code dump ---
create_code_dump() {
    if [ -n "$DUMP_SCRIPT_NAME" ]; then
        echo -e "${C_YELLOW}Creating code state dump for $active_agent_key...${C_NC}"
        python "$DUMP_SCRIPT_NAME" > /dev/null 2>&1
        # Move the generated dump file to the run-specific directory
        if [ -f "$DUMP_FILE_NAME" ]; then
            mv "$DUMP_FILE_NAME" "$RUN_DIR/$DUMP_FILE_NAME"
        else
            echo -e "${C_RED}Warning: Dump script '$DUMP_SCRIPT_NAME' did not produce '$DUMP_FILE_NAME'.${C_NC}"
        fi
    fi
}

# --- Function to combine all logs ---
combine_logs() {
    local COMBINED_LOG_FILE="$RUN_DIR/full_analysis_log.md"
    echo -e "${C_YELLOW}📝 Assembling the final analysis file for debugging...${C_NC}"

    local INTRO_PROMPT="We are developing an android agent to control a mobile application, for the benchmark android_world. If this file is sent to you, this means there has been something wrong in the execution, either the agent couldn't finish the task, or had a runtime error. You will find the source code of the related files, run logs of the agent, and also the LLM interactions. Finally, you will also find the screenshots of the steps of the agent. Debug the error/mistake and lay it out clearly, then propose solutions. The proposed solutions should be in the format of full file changes, to accomodate direct copy and paste from this chat to the IDE"

    echo "$INTRO_PROMPT" > "$COMBINED_LOG_FILE"
    echo "" >> "$COMBINED_LOG_FILE"

    local SCREENSHOT_DIR="$RUN_DIR/screenshots"
    echo -e "\n\n## 📸 Screenshots Taken\n" >> "$COMBINED_LOG_FILE"
    echo 'The following screenshots were saved during the run. The names correspond to the step number.' >> "$COMBINED_LOG_FILE"
    echo '```text' >> "$COMBINED_LOG_FILE"
    if [ -d "$SCREENSHOT_DIR" ] && [ -n "$(ls -A "$SCREENSHOT_DIR" 2>/dev/null)" ]; then
        ls -1 "$SCREENSHOT_DIR" >> "$COMBINED_LOG_FILE"
    else
        echo "No screenshots were found or the directory is empty." >> "$COMBINED_LOG_FILE"
    fi
    echo '```' >> "$COMBINED_LOG_FILE"

    # Add the Agent-specific code dump (if configured)
    if [ -n "$DUMP_FILE_NAME" ]; then
        local AGENT_DUMP_FILE="$RUN_DIR/$DUMP_FILE_NAME"
        echo -e "\n\n## 💻 Agent Code State Dump ($DUMP_FILE_NAME)\n" >> "$COMBINED_LOG_FILE"
        if [ -f "$AGENT_DUMP_FILE" ]; then
            cat "$AGENT_DUMP_FILE" >> "$COMBINED_LOG_FILE"
        else
            echo '```text' >> "$COMBINED_LOG_FILE"
            echo "--> $DUMP_FILE_NAME not found in $RUN_DIR." >> "$COMBINED_LOG_FILE"
            echo '```' >> "$COMBINED_LOG_FILE"
        fi
    fi

    echo -e "\n\n## 📜 Main Benchmark Run Log (benchmark_run.log)\n" >> "$COMBINED_LOG_FILE"
    echo '```log' >> "$COMBINED_LOG_FILE"
    [ -f "$LOG_FILE" ] && cat "$LOG_FILE" >> "$COMBINED_LOG_FILE" || echo "--> benchmark_run.log not found." >> "$COMBINED_LOG_FILE"
    echo '```' >> "$COMBINED_LOG_FILE"

    local LLM_LOG_FILE="$RUN_DIR/llm_interactions.log"
    echo -e "\n\n## 🤖 LLM Interaction Log (llm_interactions.log)\n" >> "$COMBINED_LOG_FILE"
    echo '```log' >> "$COMBINED_LOG_FILE"
    [ -f "$LLM_LOG_FILE" ] && cat "$LLM_LOG_FILE" >> "$COMBINED_LOG_FILE" || echo "--> llm_interactions.log not found." >> "$COMBINED_LOG_FILE"
    echo '```' >> "$COMBINED_LOG_FILE"

    echo -e "${C_GREEN}✅ Combined analysis file created successfully: ${C_CYAN}${COMBINED_LOG_FILE}${C_NC}"
}


# --- Execution & Cleanup ---
PID=
trap '
  echo -e "\n${C_RED}🛑 Interrupted. Killing process tree...${C_NC}"
  [ ! -z "$PID" ] && kill -- -"$PID" 2>/dev/null
  create_code_dump
  if [[ "$agent_registry_name" == *"ollama"* ]] && [ ! -z "$OLLAMA_PID" ]; then
      echo -e "${C_YELLOW}🛑 Stopping Ollama server...${C_NC}"
      kill $OLLAMA_PID 2>/dev/null || true
  fi
  combine_logs
  echo -e "${C_YELLOW}Artifacts and combined log saved to $RUN_DIR${C_NC}"
  exit 130
' SIGINT SIGTERM

# Using a subshell to run the command in its own process group
(
  set -m # Enable job control
  eval "$PYTHON_CMD" &
  PID=$!
  wait $PID
) 2>&1 | tee "$LOG_FILE"

# Create code dump and combine logs after successful execution
create_code_dump
combine_logs

echo -e "${C_GREEN}🎉 Benchmark completed!${C_NC}"
exit 0