#!/bin/bash

# AndroidWorld Benchmark Runner Script v2
# This script orchestrates benchmark runs using a central YAML configuration.
# It handles environment setup, agent configuration, and execution.
#
# Usage:
#   ./run_benchmark.sh
#   ./run_benchmark.sh --config=my_config.yaml
#   ./run_benchmark.sh --override_agent=m3a_gemini

set -e # Exit on any error

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
  --help, -h          Show this help message.
EOF
}

# --- Prerequisite: yq ---
check_yq() {
    if ! command -v yq &> /dev/null; then
        echo "❌ 'yq' is required. Please install it: https://github.com/mikefarah/yq/#install"
        exit 1
    fi
}
check_yq

# --- Configuration Loading and Parsing ---
CONFIG_FILE="config/default.yaml"
OVERRIDE_AGENT=""

for arg in "$@"; do
    case $arg in
        --config=*) CONFIG_FILE="${arg#*=}"; shift;;
        --override_agent=*) OVERRIDE_AGENT="${arg#*=}"; shift;;
        -h|--help) show_help; exit 0;;
    esac
done

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"; exit 1
fi

# Load script and run settings
eval "$(yq e '.script | to_entries | .[] | .key + "=\"" + .value + "\""' "$CONFIG_FILE")"
eval "$(yq e '.run | to_entries | .[] | "run_" + .key + "=\"" + .value + "\""' "$CONFIG_FILE")"

# Determine active agent and load its config
active_agent_key="${OVERRIDE_AGENT:-$(yq e '.agent.active_agent' "$CONFIG_FILE")}"
echo "✅ Active agent key: $active_agent_key"

YQ_AGENT_PATH=".agent.configurations.$active_agent_key"
if [ "$(yq e "$YQ_AGENT_PATH" "$CONFIG_FILE")" = "null" ]; then
    echo "❌ Agent configuration for '$active_agent_key' not found in $CONFIG_FILE"
    exit 1
fi
eval "$(yq e "$YQ_AGENT_PATH | to_entries | .[] | \"agent_\" + .key + \"=\\\"\" + .value + \"\\\"\"" "$CONFIG_FILE")"

# Load task and environment settings
eval "$(yq e '.task | to_entries | .[] | "task_" + .key + "=\"" + .value + "\""' "$CONFIG_FILE")"
eval "$(yq e '.environment | to_entries | .[] | "env_" + .key + "=\"" + .value + "\""' "$CONFIG_FILE")"


# --- Display Final Configuration ---
echo "🚀 Starting AndroidWorld Benchmark..."
echo "  Agent: $agent_registry_name (from key '$active_agent_key')"
echo "  Run Mode: $run_mode"
echo "  Task: ${task_name:-Default (Random/All)}"
echo "-----------------------------------------------------"


# --- Check API Keys ---
check_api_keys() {
    if [[ "$agent_registry_name" == *"openrouter"* ]] && [ -z "$OPENROUTER_API_KEY" ]; then
        echo "❌ OPENROUTER_API_KEY is not set for agent '$agent_registry_name'."
        exit 1
    elif [[ "$agent_registry_name" == *"gemini"* ]] && [ -z "$GEMINI_API_KEY" ]; then
        echo "❌ GEMINI_API_KEY is not set for agent '$agent_registry_name'."
        exit 1
    fi
    echo "✅ API key for $agent_registry_name found."
}
check_api_keys


# --- Environment Setup ---
export PATH="$PATH:~/Library/Android/sdk/platform-tools:~/Library/Android/sdk/emulator"
export PYTHONUNBUFFERED=1

if [ "$activate_conda" = "true" ]; then
    echo "🔧 Activating conda environment: $conda_env_name..."
    source /opt/anaconda3/etc/profile.d/conda.sh
    conda activate "$conda_env_name"
fi

check_emulator() { adb devices 2>/dev/null | grep -q "emulator"; }
if [ "$start_emulator" = "true" ] && ! check_emulator; then
    echo "📱 Starting Android emulator..."
    pkill -f "emulator.*AndroidWorldAvd" || true
    nohup emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554 > emulator.log 2>&1 &
    # Simplified wait, can be replaced with more robust version if needed
    echo "⏳ Waiting for emulator to boot..." && sleep 15
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
    echo "🔧 Running AndroidWorld emulator setup (agent: $agent_registry_name)..."
    if python run.py --perform_emulator_setup=True --agent_name="$agent_registry_name"; then
        echo "✅ Emulator setup completed." && touch "$SETUP_MARKER"
    else
        echo "❌ Emulator setup failed!"; exit 1
    fi
else
    echo "⏭️  Skipping emulator setup."
fi


# --- Build Python Command ---
build_python_command() {
    local base_cmd
    local common_flags="--agent_name=\"$agent_registry_name\""
    
    # Append all agent-specific flags, excluding registry_name
    for var in $(compgen -v agent_); do
        if [ "$var" != "agent_registry_name" ]; then
            key_name=$(echo "$var" | sed 's/agent_//')
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
        echo "❌ Invalid run mode: $run_mode."; exit 1
    fi
    echo "$base_cmd $common_flags"
}

PYTHON_CMD=$(build_python_command)
echo "🐍 Executing command:"
echo "$PYTHON_CMD"
echo "-----------------------------------------------------"


# --- Execution & Cleanup ---
TEMP_OUTPUT=$(mktemp)
trap 'echo "🛑 Interrupted. Log: $TEMP_OUTPUT"; exit 130' SIGINT SIGTERM

eval "$PYTHON_CMD" 2>&1 | tee "$TEMP_OUTPUT" &
wait $!

echo "🎉 Benchmark completed! Full output log is at: $TEMP_OUTPUT"
exit 0 