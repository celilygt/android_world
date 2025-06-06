#!/bin/bash

# AndroidWorld Benchmark Runner Script
# This script sets up the environment and runs the benchmark with M3A agents
# Usage: ./run_benchmark.sh [fast|full] [task_name] [--agent=AGENT_TYPE]

set -e  # Exit on any error

# Parse command line arguments
MODE="full"  # default
SPECIFIC_TASK=""
AGENT_TYPE="gemini"  # default

# Help function
show_help() {
    cat << EOF
AndroidWorld Benchmark Runner
=============================

Usage: ./run_benchmark.sh [MODE] [TASK_NAME] [--agent=AGENT_TYPE]
       ./run_benchmark.sh [TASK_NAME] [--agent=AGENT_TYPE]
       ./run_benchmark.sh --help

MODES:
  fast    Run minimal runner (quick test)
  full    Run full benchmark (default)

AGENT TYPES:
  openrouter    Use M3A OpenRouter Agent with free models (default)
  gemini        Use M3A Gemini Agent with free Gemma 3-27B

EXAMPLES:
  ./run_benchmark.sh                                    # Full benchmark with OpenRouter
  ./run_benchmark.sh --agent=gemini                     # Full benchmark with Gemini  
  ./run_benchmark.sh fast --agent=openrouter            # Fast mode with OpenRouter
  ./run_benchmark.sh fast ContactsAddContact --agent=gemini  # Fast mode with specific task and Gemini

AVAILABLE TASKS:
  ContactsAddContact, ClockStopWatchRunning, and many more...
  (See AndroidWorld documentation for full task list)

OUTPUT:
  Results are saved to ./runs/ with timestamp:
  - benchmark_summary_YYYYMMDD_HHMMSS.txt (parsed summary)
  - benchmark_output_YYYYMMDD_HHMMSS.log (full output)

EOF
}

# Parse arguments
while [ $# -gt 0 ]; do
    case $1 in
        --help|-h|help)
            show_help
            exit 0
            ;;
        --agent=*)
            AGENT_TYPE="${1#*=}"
            ;;
        fast|full)
            MODE=$1
            ;;
        *)
            # Assume it's a task name if not recognized as other option
            if [ -z "$SPECIFIC_TASK" ]; then
                SPECIFIC_TASK=$1
            fi
            ;;
    esac
    shift
done

echo "🚀 Starting AndroidWorld Benchmark Setup..."
echo "📋 Mode: $MODE"
echo "🤖 Agent: $AGENT_TYPE"
if [ -n "$SPECIFIC_TASK" ]; then
    echo "🎯 Specific Task: $SPECIFIC_TASK"
fi

# Check API keys based on agent type
if [ "$AGENT_TYPE" = "openrouter" ]; then
    if [ -z "$OPENROUTER_API_KEY" ]; then
        echo "❌ OPENROUTER_API_KEY environment variable not set!"
        echo ""
        echo "📋 Setup Instructions:"
        echo "1. Sign up at https://openrouter.ai (free)"
        echo "2. Get your API key from the dashboard"
        echo "3. Set the environment variable:"
        echo "   export OPENROUTER_API_KEY='your_api_key_here'"
        echo ""
        echo "💡 Available free models:"
        echo "   - google/gemma-3-27b-it:free (default)"
        echo "   - meta-llama/llama-3.3-70b-instruct:free"
        echo "   - mistralai/mistral-7b-instruct:free"
        exit 1
    else
        echo "✅ OpenRouter API key found!"
    fi
elif [ "$AGENT_TYPE" = "gemini" ]; then
    if [ -z "$GEMINI_API_KEY" ]; then
        echo "❌ GEMINI_API_KEY environment variable not set!"
        echo ""
        echo "📋 Setup Instructions:"
        echo "1. Get a free Gemini API key from https://aistudio.google.com/"
        echo "2. Set the environment variable:"
        echo "   export GEMINI_API_KEY='your_api_key_here'"
        echo ""
        echo "💡 Available free Gemma models:"
        echo "   - gemma-3-27b-it (default)"
        echo "   - gemma-3-9b-it"
        echo ""
        echo "🆓 Note: Gemma models on Gemini API have free usage quotas"
        exit 1
    else
        echo "✅ Gemini API key found!"
    fi
else
    echo "❌ Unknown agent type: $AGENT_TYPE"
    echo "Available agent types: openrouter, gemini"
    exit 1
fi

# Set runner script and agent name based on agent type
if [ "$AGENT_TYPE" = "openrouter" ]; then
    FAST_RUNNER="m3a_openrouter_minimal_runner.py"
    AGENT_NAME="m3a_openrouter_agent"
elif [ "$AGENT_TYPE" = "gemini" ]; then
    FAST_RUNNER="m3a_gemini_gemma_minimal_runner.py"
    AGENT_NAME="m3a_gemini_gemma_agent"
fi

# Add Android SDK tools to PATH
export PATH="$PATH:~/Library/Android/sdk/platform-tools"
export PATH="$PATH:~/Library/Android/sdk/emulator"

# Function to check if emulator is running
check_emulator() {
    adb devices 2>/dev/null | grep -q "emulator" && return 0 || return 1
}

# Function to wait for emulator to be ready
wait_for_emulator() {
    echo "⏳ Waiting for emulator to be ready..."
    local timeout=120  # 2 minutes timeout
    local elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if adb shell getprop sys.boot_completed 2>/dev/null | grep -q "1"; then
            echo "✅ Emulator is ready!"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        echo "   Waiting... (${elapsed}s/${timeout}s)"
    done
    
    echo "❌ Timeout waiting for emulator to be ready"
    return 1
}

# Function to check if emulator setup was completed
check_emulator_setup_done() {
    [ -f ".emulator_setup_completed" ]
}

# Function to check if user chose to skip setup forever
check_skip_setup_forever() {
    [ -f ".skip_emulator_setup_forever" ]
}

# Function to handle emulator setup
handle_emulator_setup() {
    # If user chose to skip forever, don't ask again
    if check_skip_setup_forever; then
        echo "⏭️  Emulator setup skipped (user preference saved)"
        return 0
    fi
    
    # If setup was already completed, give user options
    if check_emulator_setup_done; then
        echo ""
        echo "🔧 Emulator Setup Status"
        echo "========================"
        echo "✅ Emulator setup appears to have been completed before."
        echo ""
        echo "What would you like to do?"
        echo "  [s] Skip setup for this run (default)"
        echo "  [r] Run setup again (re-initialize apps)"
        echo "  [n] Never ask again (skip setup forever)"
        echo ""
        read -p "Choose option [s/r/n]: " -r setup_choice
        
        case $setup_choice in
            r|R)
                echo "🔄 Re-running emulator setup..."
                return 1  # Signal to run setup
                ;;
            n|N)
                echo "💾 Saving preference to skip emulator setup forever..."
                touch ".skip_emulator_setup_forever"
                return 0  # Skip setup
                ;;
            s|S|"")
                echo "⏭️  Skipping emulator setup for this run..."
                return 0  # Skip setup
                ;;
            *)
                echo "⚠️  Invalid choice, skipping setup..."
                return 0  # Skip setup
                ;;
        esac
    else
        # First time setup
        echo ""
        echo "🔧 Emulator Setup Required"
        echo "=========================="
        echo "⚠️  AndroidWorld emulator setup has not been completed yet."
        echo "    This initializes required apps and should be run only once."
        echo ""
        echo "What would you like to do?"
        echo "  [y] Run emulator setup now (recommended for first time)"
        echo "  [s] Skip for now (you can run it later)"
        echo "  [n] Never run setup (skip forever - not recommended)"
        echo ""
        read -p "Choose option [y/s/n]: " -r setup_choice
        
        case $setup_choice in
            y|Y|"")
                echo "🚀 Running emulator setup..."
                return 1  # Signal to run setup
                ;;
            n|N)
                echo "💾 Saving preference to skip emulator setup forever..."
                touch ".skip_emulator_setup_forever"
                return 0  # Skip setup
                ;;
            s|S)
                echo "⏭️  Skipping emulator setup for this run..."
                return 0  # Skip setup
                ;;
            *)
                echo "⚠️  Invalid choice, skipping setup..."
                return 0  # Skip setup
                ;;
        esac
    fi
}

# Function to run emulator setup
run_emulator_setup() {
    echo "🔧 Running AndroidWorld emulator setup..."
    echo "   This may take several minutes as it installs required apps..."
    
    # Use the selected agent for emulator setup too
    if python run.py --perform_emulator_setup=True --agent_name="$AGENT_NAME"; then
        echo "✅ Emulator setup completed successfully!"
        touch ".emulator_setup_completed"
        return 0
    else
        echo "❌ Emulator setup failed!"
        return 1
    fi
}

# Function to parse and save results
parse_and_save_results() {
    local output_file=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local runs_dir="./runs"
    local run_subfolder="$runs_dir/run_$timestamp"
    local summary_file="$run_subfolder/benchmark_summary.txt"
    local full_log_file="$run_subfolder/benchmark_output.log"
    
    # Create runs directory and run subfolder if they don't exist
    mkdir -p "$run_subfolder"
    
    echo "📊 Parsing results and saving summary..."
    
    # Create summary file
    cat > "$summary_file" << EOF
AndroidWorld Benchmark Summary
==============================
Timestamp: $(date)
Mode: $MODE
Agent: M3A $(echo "$AGENT_TYPE" | tr '[:lower:]' '[:upper:]')
$([ -n "$SPECIFIC_TASK" ] && echo "Specific Task: $SPECIFIC_TASK" || echo "Tasks: All available tasks")

TASK RESULTS:
=============
EOF

    # Parse the output for task results
    local current_task=""
    local current_goal=""
    local step_count=0
    local max_step=0
    
    while IFS= read -r line; do
        if [[ $line == *"Running random task:"* ]]; then
            current_task=$(echo "$line" | sed 's/.*Running random task: //')
            echo "" >> "$summary_file"
            echo "$current_task" >> "$summary_file"
            echo "$(printf '=%.0s' {1..${#current_task}})" >> "$summary_file"
        elif [[ $line == *"Running task:"* ]]; then
            current_task=$(echo "$line" | sed 's/.*Running task: //')
            echo "" >> "$summary_file"
            echo "$current_task" >> "$summary_file"
            echo "$(printf '=%.0s' {1..${#current_task}})" >> "$summary_file"
        elif [[ $line == *"Goal:"* ]]; then
            current_goal=$(echo "$line" | sed 's/Goal: //')
            echo "Goal: $current_goal" >> "$summary_file"
        elif [[ $line == *"Step "* && $line == *"..."* ]]; then
            step_num=$(echo "$line" | grep -o 'Step [0-9]*' | grep -o '[0-9]*')
            if [ -n "$step_num" ] && [ "$step_num" -gt "$max_step" ]; then
                max_step=$step_num
            fi
        elif [[ $line == *"Task Failed ❌"* ]]; then
            echo "Result: ❌ Failed" >> "$summary_file"
            echo "Steps completed: $max_step" >> "$summary_file"
            if [[ $line == *";"* ]]; then
                failure_reason=$(echo "$line" | sed 's/.*; //')
                echo "Task description: $failure_reason" >> "$summary_file"
            fi
            max_step=0
        elif [[ $line == *"Task Passed ✅"* ]]; then
            echo "Result: ✅ Passed" >> "$summary_file"
            echo "Steps completed: $max_step" >> "$summary_file"
            if [[ $line == *";"* ]]; then
                success_info=$(echo "$line" | sed 's/.*; //')
                echo "Task description: $success_info" >> "$summary_file"
            fi
            max_step=0
        elif [[ $line == *"Agent did not indicate task is done"* ]]; then
            echo "Reason: Agent reached max steps without completing task" >> "$summary_file"
        fi
    done < "$output_file"
    
    # Add final statistics from the table if available
    if grep -A 10 "mean_success_rate" "$output_file" > /dev/null 2>&1; then
        echo "" >> "$summary_file"
        echo "SUMMARY STATISTICS:" >> "$summary_file"
        echo "==================" >> "$summary_file"
        grep -A 20 "mean_success_rate" "$output_file" | head -10 >> "$summary_file"
    fi
    
    # Count total tasks and success rate
    local total_tasks=$(grep -c "Running.*task:" "$output_file" 2>/dev/null | tr -d '\n' || echo "0")
    local passed_tasks=$(grep -c "Task Passed ✅" "$output_file" 2>/dev/null | tr -d '\n' || echo "0")
    local failed_tasks=$(grep -c "Task Failed ❌" "$output_file" 2>/dev/null | tr -d '\n' || echo "0")
    
    # Ensure variables are clean numbers
    total_tasks=$(echo "$total_tasks" | tr -d ' \n\r')
    passed_tasks=$(echo "$passed_tasks" | tr -d ' \n\r')
    failed_tasks=$(echo "$failed_tasks" | tr -d ' \n\r')
    
    # Set defaults if empty or non-numeric
    total_tasks=${total_tasks:-0}
    passed_tasks=${passed_tasks:-0}
    failed_tasks=${failed_tasks:-0}
    
    # Validate they are actually numbers
    if ! [[ "$total_tasks" =~ ^[0-9]+$ ]]; then total_tasks=0; fi
    if ! [[ "$passed_tasks" =~ ^[0-9]+$ ]]; then passed_tasks=0; fi
    if ! [[ "$failed_tasks" =~ ^[0-9]+$ ]]; then failed_tasks=0; fi
    
    if [ "$total_tasks" -gt 0 ]; then
        echo "" >> "$summary_file"
        echo "OVERALL SUMMARY:" >> "$summary_file"
        echo "===============" >> "$summary_file"
        echo "Total tasks run: $total_tasks" >> "$summary_file"
        echo "Tasks passed: $passed_tasks" >> "$summary_file"
        echo "Tasks failed: $failed_tasks" >> "$summary_file"
        
        # Calculate success rate safely
        local success_rate=0
        if [ "$total_tasks" -gt 0 ] && [ "$passed_tasks" -ge 0 ]; then
            success_rate=$(echo "$passed_tasks $total_tasks" | awk '{if ($2 > 0) printf "%.0f", ($1 * 100) / $2; else print "0"}')
        fi
        echo "Success rate: ${success_rate}%" >> "$summary_file"
    fi
    
    echo "" >> "$summary_file"
    echo "Full output saved to: benchmark_output.log" >> "$summary_file"
    
    # Copy full output to the same subfolder
    cp "$output_file" "$full_log_file"
    
    echo "✅ Results saved in subfolder: $run_subfolder"
    echo "   📄 Summary: benchmark_summary.txt"
    echo "   📋 Full log: benchmark_output.log"
}

# Activate conda environment
echo "🔧 Activating android_world conda environment..."
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate android_world

# Check if emulator is already running (required for both fast and full modes)
if check_emulator; then
    echo "✅ Android emulator is already running"
else
    echo "📱 Starting Android emulator..."
    
    # Kill any existing emulator processes first
    pkill -f "emulator.*AndroidWorldAvd" || true
    sleep 2
    
    # Start emulator in background
    nohup ~/Library/Android/sdk/emulator/emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554 > emulator.log 2>&1 &
    EMULATOR_PID=$!
    echo "   Emulator started with PID: $EMULATOR_PID"
    
    # Wait for emulator to be ready
    if ! wait_for_emulator; then
        echo "❌ Failed to start emulator. Check emulator.log for details."
        exit 1
    fi
fi

# Verify adb connection
echo "🔍 Checking ADB connection..."
adb devices

# Handle emulator setup (required for both fast and full modes)
if ! handle_emulator_setup; then
    # User chose to run setup
    if ! run_emulator_setup; then
        echo "❌ Cannot continue without successful emulator setup."
        exit 1
    fi
fi

# Create temporary output file
TEMP_OUTPUT=$(mktemp)

# Run based on mode
if [ "$MODE" = "fast" ]; then
    echo "⚡ Running fast mode with M3A $(echo "$AGENT_TYPE" | tr '[:lower:]' '[:upper:]') Agent..."
    if [ -n "$SPECIFIC_TASK" ]; then
        echo "🎯 Task: $SPECIFIC_TASK"
        python "$FAST_RUNNER" --task="$SPECIFIC_TASK" 2>&1 | tee "$TEMP_OUTPUT"
    else
        echo "🎲 Random task selection"
        python "$FAST_RUNNER" 2>&1 | tee "$TEMP_OUTPUT"
    fi
else
    echo "🎯 Running full benchmark with M3A $(echo "$AGENT_TYPE" | tr '[:lower:]' '[:upper:]') Agent..."
    if [ -n "$SPECIFIC_TASK" ]; then
        echo "🎯 Task: $SPECIFIC_TASK"
        python run.py \
            --suite_family=android_world \
            --agent_name="$AGENT_NAME" \
            --tasks="$SPECIFIC_TASK" \
            --n_task_combinations=1 2>&1 | tee "$TEMP_OUTPUT"
    else
        python run.py \
            --suite_family=android_world \
            --agent_name="$AGENT_NAME" \
            --n_task_combinations=1 2>&1 | tee "$TEMP_OUTPUT"
            #Zum beispiel : Parameter = 3 => with 116 tasks × 3 combinations = 348 total task instances to run!
    fi
fi

# Parse and save results
parse_and_save_results "$TEMP_OUTPUT"

# Cleanup
rm "$TEMP_OUTPUT"

echo "🎉 Benchmark completed!"
echo "📊 Results saved to: ./runs/" 