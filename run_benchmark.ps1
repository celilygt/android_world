# AndroidWorld Benchmark Runner Script v3 - Windows PowerShell Version
# This script orchestrates benchmark runs using a central YAML configuration.
# It handles environment setup, agent configuration, and execution, with
# improved logging, graceful shutdown, and conditional output visibility.
#
# Usage:
#   .\run_benchmark.ps1
#   .\run_benchmark.ps1 -Config my_config.yaml
#   .\run_benchmark.ps1 -OverrideAgent m3a_gemini

param(
    [string]$Config = "config\windows.yaml",
    [string]$OverrideAgent = "",
    [switch]$Help
)

# Error handling
$ErrorActionPreference = "Stop"

# --- Color Definitions ---
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    Cyan = "Cyan"
    White = "White"
}

# --- Help Function ---
function Show-Help {
    Write-Host @"
AndroidWorld Benchmark Runner - Windows
=======================================
Uses 'config\windows.yaml' or a custom config to run benchmarks.

Usage: .\run_benchmark.ps1 [options]

Options:
  -Config <path>          Path to a custom YAML config file.
  -OverrideAgent <agent>  Run with a different agent from the config, e.g., 'm3a_gemini'.
  -Help                   Show this help message.
"@
}

if ($Help) {
    Show-Help
    exit 0
}

# --- Prerequisite: yq ---
function Test-Yq {
    try {
        $null = Get-Command yq -ErrorAction Stop
        Write-Host "✅ yq found" -ForegroundColor $Colors.Green
    }
    catch {
        Write-Host "❌ 'yq' is required. Please install it from: https://github.com/mikefarah/yq/#install" -ForegroundColor $Colors.Red
        Write-Host "For Windows, you can install via: winget install MikeFarah.yq" -ForegroundColor $Colors.Yellow
        exit 1
    }
}
Test-Yq

# --- Configuration Loading and Parsing ---
if (-not (Test-Path $Config)) {
    Write-Host "❌ Config file not found: $Config" -ForegroundColor $Colors.Red
    exit 1
}

Write-Host "📁 Loading configuration from: $Config" -ForegroundColor $Colors.Blue

# Load script and run settings
$scriptSettings = @{}
(yq e '.script | to_entries | .[] | .key + "=" + .value' $Config) | ForEach-Object {
    $key, $value = $_ -split '=', 2
    $scriptSettings[$key] = $value
}

$runSettings = @{}
(yq e '.run | to_entries | .[] | .key + "=" + .value' $Config) | ForEach-Object {
    $key, $value = $_ -split '=', 2
    $runSettings["run_$key"] = $value
}

# Determine active agent and load its config
$activeAgentKey = if ($OverrideAgent) { $OverrideAgent } else { yq e '.agent.active_agent' $Config }
Write-Host "✅ Active agent key: $activeAgentKey" -ForegroundColor $Colors.Green

$yqAgentPath = ".agent.configurations.$activeAgentKey"
$agentCheck = yq e $yqAgentPath $Config
if ($agentCheck -eq "null") {
    Write-Host "❌ Agent configuration for '$activeAgentKey' not found in $Config" -ForegroundColor $Colors.Red
    exit 1
}

$agentSettings = @{}
(yq e "$yqAgentPath | to_entries | .[] | .key + `"=`" + .value" $Config) | ForEach-Object {
    $key, $value = $_ -split '=', 2
    $agentSettings["agent_$key"] = $value
}

# Load task and environment settings
$taskSettings = @{}
(yq e '.task | to_entries | .[] | .key + "=" + .value' $Config) | ForEach-Object {
    $key, $value = $_ -split '=', 2
    $taskSettings["task_$key"] = $value
}

$envSettings = @{}
(yq e '.environment | to_entries | .[] | .key + "=" + .value' $Config) | ForEach-Object {
    $key, $value = $_ -split '=', 2
    $envSettings["env_$key"] = $value
}

# --- Setup Output Logging ---
$logDir = "runs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "$logDir\${timestamp}_${activeAgentKey}.log"
New-Item -ItemType File -Path $logFile | Out-Null

# --- Display Final Configuration ---
Write-Host "🚀 Starting AndroidWorld Benchmark..." -ForegroundColor $Colors.Yellow
Write-Host "  Agent: $($agentSettings['agent_registry_name']) (from key '$activeAgentKey')" -ForegroundColor $Colors.Blue
Write-Host "  Run Mode: $($runSettings['run_mode'])" -ForegroundColor $Colors.Blue
Write-Host "  Task: $(if ($taskSettings['task_name'] -and $taskSettings['task_name'] -ne 'null') { $taskSettings['task_name'] } else { 'Default (Random/All)' })" -ForegroundColor $Colors.Blue
Write-Host "  Log File: $logFile" -ForegroundColor $Colors.Blue
Write-Host "-----------------------------------------------------"

# --- Check API Keys ---
function Test-ApiKeys {
    $agentName = $agentSettings['agent_registry_name']
    
    if ($agentName -like "*openrouter*" -and -not $env:OPENROUTER_API_KEY) {
        Write-Host "❌ OPENROUTER_API_KEY is not set for agent '$agentName'." -ForegroundColor $Colors.Red
        exit 1
    }
    elseif ($agentName -like "*gemini*" -and -not $env:GEMINI_API_KEY) {
        Write-Host "❌ GEMINI_API_KEY is not set for agent '$agentName'." -ForegroundColor $Colors.Red
        exit 1
    }
    Write-Host "✅ API key for $agentName found." -ForegroundColor $Colors.Green
}
Test-ApiKeys

# --- Environment Setup ---
# Resolve Android SDK path from config, with a default, and expand environment variables
$sdkPathTemplate = if ($envSettings['env_android_sdk_root']) { $envSettings['env_android_sdk_root'] } else { "%LOCALAPPDATA%\Android\Sdk" }
$sdkPath = [Environment]::ExpandEnvironmentVariables($sdkPathTemplate)

# Add Android SDK tools to PATH
$env:PATH += ";$sdkPath\platform-tools;$sdkPath\emulator"
$env:PYTHONUNBUFFERED = "1"
$env:GRPC_VERBOSITY = "FATAL" # Suppress noisy gRPC logs from emulator communication

# Conda activation for Windows
if ($scriptSettings['activate_conda'] -eq 'true') {
    Write-Host "🔧 Activating conda environment: $($scriptSettings['conda_env_name'])..." -ForegroundColor $Colors.Yellow
    
    # Try different conda initialization paths for Windows
    $condaPaths = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
        "$env:ProgramData\miniconda3\Scripts\conda.exe",
        "$env:ProgramData\anaconda3\Scripts\conda.exe"
    )
    
    $condaFound = $false
    foreach ($condaPath in $condaPaths) {
        if (Test-Path $condaPath) {
            & $condaPath activate $scriptSettings['conda_env_name']
            $condaFound = $true
            break
        }
    }
    
    if (-not $condaFound) {
        Write-Host "⚠️ Conda not found in standard locations. Make sure conda is in your PATH." -ForegroundColor $Colors.Yellow
    }
}

# Check emulator function
function Test-Emulator {
    try {
        $devices = adb devices 2>$null
        return $devices -like "*emulator*"
    }
    catch {
        return $false
    }
}

# Start emulator if needed
if ($scriptSettings['start_emulator'] -eq 'true' -and -not (Test-Emulator)) {
    Write-Host "📱 Starting Android emulator..." -ForegroundColor $Colors.Yellow
    
    # Kill existing emulator processes
    Get-Process -Name "emulator*" -ErrorAction SilentlyContinue | Stop-Process -Force
    Get-Process -Name "qemu-system*" -ErrorAction SilentlyContinue | Stop-Process -Force
    
    # Start emulator
    $emulatorLogPath = if ($envSettings['env_emulator_log_path']) { $envSettings['env_emulator_log_path'] } else { "emulator.log" }
    Start-Process -FilePath "emulator" -ArgumentList "-avd", "AndroidWorldAvd", "-no-snapshot", "-grpc", "8554" -RedirectStandardOutput $emulatorLogPath -RedirectStandardError $emulatorLogPath -NoNewWindow
    
    Write-Host "⏳ Waiting for emulator to connect..."
    try {
        adb wait-for-device
    }
    catch {
        Write-Host "❌ ADB could not connect to the device." -ForegroundColor $Colors.Red
        exit 1
    }
    
    Write-Host "⏳ Waiting for emulator to fully boot (max 3 minutes)..."
    $bootCompleted = $false
    for ($i = 0; $i -lt 90; $i++) {
        try {
            $bootStatus = adb shell getprop sys.boot_completed 2>$null
            if ($bootStatus.Trim() -eq "1") {
                $bootCompleted = $true
                break
            }
        }
        catch {
            # Ignore errors during boot check
        }
        Start-Sleep 2
    }
    
    if ($bootCompleted) {
        Write-Host "✅ Emulator is booted and ready." -ForegroundColor $Colors.Green
        Start-Sleep 5 # Grace period for all services to stabilize
    }
    else {
        Write-Host "❌ Emulator failed to boot within the time limit." -ForegroundColor $Colors.Red
        Write-Host "Check $emulatorLogPath for details."
        exit 1
    }
}

# Handle emulator setup based on config
$setupMarker = ".emulator_setup_completed"
$runSetup = $false

if ($scriptSettings['perform_emulator_setup'] -eq 'force_run') {
    $runSetup = $true
}
elseif ($scriptSettings['perform_emulator_setup'] -eq 'run_once' -and -not (Test-Path $setupMarker)) {
    $runSetup = $true
}

if ($runSetup) {
    Write-Host "🔧 Running AndroidWorld emulator setup (agent: $($agentSettings['agent_registry_name']))..." -ForegroundColor $Colors.Yellow
    try {
        python run.py --perform_emulator_setup=True --agent_name="$($agentSettings['agent_registry_name'])"
        Write-Host "✅ Emulator setup completed." -ForegroundColor $Colors.Green
        New-Item -ItemType File -Path $setupMarker | Out-Null
    }
    catch {
        Write-Host "❌ Emulator setup failed!" -ForegroundColor $Colors.Red
        exit 1
    }
}
else {
    Write-Host "⏭️  Skipping emulator setup."
}

# --- Build Python Command ---
function Build-PythonCommand {
    $commonFlags = "--agent_name=`"$($agentSettings['agent_registry_name'])`""
    
    # Append all agent-specific flags, excluding registry_name
    foreach ($key in $agentSettings.Keys) {
        if ($key -ne "agent_registry_name") {
            $flagName = $key -replace "^agent_", ""
            $value = $agentSettings[$key]
            if ($value) {
                $commonFlags += " --$flagName=`"$value`""
            }
        }
    }
    
    # Append all environment-specific flags, excluding variables used only by this script
    foreach ($key in $envSettings.Keys) {
        $flagName = $key -replace "^env_", ""
        if ($flagName -notin @("android_sdk_root", "emulator_log_path")) {
            $value = $envSettings[$key]
            if ($value) {
                $commonFlags += " --$flagName=`"$value`""
            }
        }
    }
    
    if ($runSettings['run_mode'] -eq 'fast') {
        $baseCmd = "python -u minimal_task_runner.py"
        if ($taskSettings['task_name'] -and $taskSettings['task_name'] -ne 'null') {
            $commonFlags += " --task=`"$($taskSettings['task_name'])`""
        }
    }
    elseif ($runSettings['run_mode'] -eq 'full') {
        $baseCmd = "python -u run.py"
        if ($taskSettings['task_name'] -and $taskSettings['task_name'] -ne 'null') {
            $commonFlags += " --tasks=`"$($taskSettings['task_name'])`""
        }
        $suiteFamily = if ($taskSettings['task_suite_family']) { $taskSettings['task_suite_family'] } else { "android_world" }
        $commonFlags += " --suite_family=`"$suiteFamily`""
        $nTaskCombinations = if ($taskSettings['task_n_task_combinations']) { $taskSettings['task_n_task_combinations'] } else { "1" }
        $commonFlags += " --n_task_combinations=$nTaskCombinations"
    }
    else {
        Write-Host "❌ Invalid run mode: $($runSettings['run_mode'])." -ForegroundColor $Colors.Red
        exit 1
    }
    
    return "$baseCmd $commonFlags"
}

$pythonCmd = Build-PythonCommand
Write-Host "🐍 Executing command:" -ForegroundColor $Colors.Cyan
Write-Host $pythonCmd
Write-Host "-----------------------------------------------------"

# --- Execution & Cleanup ---
# Set up Ctrl+C handling
$job = $null
$cleanup = {
    Write-Host "`n🛑 Interrupted. Stopping process..." -ForegroundColor $Colors.Red
    if ($job) {
        Stop-Job $job
        Remove-Job $job
    }
    Write-Host "Logs saved to $logFile" -ForegroundColor $Colors.Yellow
    exit 130
}

# Register the cleanup handler
Register-EngineEvent PowerShell.Exiting -Action $cleanup

try {
    # Execute the command and tee output to log file, filtering out raw LLM responses
    Invoke-Expression $pythonCmd | Tee-Object -FilePath $logFile | Where-Object { $_ -notlike "*Raw LLM Response*" }
    
    Write-Host "🎉 Benchmark completed! Full output log is at: $logFile" -ForegroundColor $Colors.Green
}
catch {
    Write-Host "❌ Benchmark failed with error: $_" -ForegroundColor $Colors.Red
    Write-Host "Full output log is at: $logFile" -ForegroundColor $Colors.Yellow
    exit 1
}
finally {
    # Cleanup
    if ($job) {
        Remove-Job $job -Force
    }
}

exit 0 