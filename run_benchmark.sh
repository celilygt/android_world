#!/bin/bash

# AndroidWorld Benchmark Runner Script v3.3
# This script orchestrates benchmark runs using a central YAML configuration.
# It handles environment setup, agent configuration, and execution, with
# improved logging, graceful shutdown, and conditional output visibility.
#
# Revision Notes (v3.3):
# - Fixed path mismatch where python script was creating its own run directory.
#   The directory created by this script is now passed as the checkpoint_dir.

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

    # Add the run log and checkpoint directory flags, pointing to the same place.
    common_flags="$common_flags --run_log_dir=\"$RUN_DIR\""
    common_flags="$common_flags --checkpoint_dir=\"$RUN_DIR\""

    # Add the skip_tasks flag if enabled
    if [ "$SKIP_COMPLETED" = true ]; then
        completed_tasks=$(yq e '.completed_tasks // [] | join(",")' config/completed_tasks.yaml)
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
            echo -e "${C_GREEN}✅ Code dump saved to $RUN_DIR/$DUMP_FILE_NAME${C_NC}"
        else
            echo -e "${C_RED}Warning: Dump script '$DUMP_SCRIPT_NAME' did not produce '$DUMP_FILE_NAME'.${C_NC}"
        fi
    fi
}

# --- Execution & Cleanup ---
PID=
trap '
  echo -e "\n${C_RED}🛑 Interrupted. Killing process tree...${C_NC}"
  [ ! -z "$PID" ] && kill -- -"$PID" 2>/dev/null
  # We cannot reliably generate the final analysis on interrupt, as it is now
  # done per-task. The user can inspect the last completed task folder.
  # The code dump is created at the start, so it will be present.
  echo -e "${C_YELLOW}Artifacts from completed tasks are saved in their respective folders within $RUN_DIR${C_NC}"
  if [[ "$agent_registry_name" == *"ollama"* ]] && [ ! -z "$OLLAMA_PID" ]; then
      echo -e "${C_YELLOW}🛑 Stopping Ollama server...${C_NC}"
      kill $OLLAMA_PID 2>/dev/null || true
  fi
  exit 130
' SIGINT SIGTERM

# Create code dump BEFORE running the python script so it is available to it.
create_code_dump

# Using a subshell to run the command in its own process group
(
  set -m # Enable job control
  eval "$PYTHON_CMD" &
  PID=$!
  wait $PID
) 2>&1 | tee "$LOG_FILE"

# Per-task analysis is now handled by the python script. This script's job is done.
echo -e "${C_GREEN}🎉 Benchmark completed!${C_NC}"
echo -e "${C_YELLOW}Per-task analysis and artifacts saved in subdirectories within: ${C_CYAN}$RUN_DIR${C_NC}"
exit 0