
"""
A command-line utility to generate a single markdown file containing the
full context of the CelilAgent for analysis by an LLM.
"""

import os
import glob

# --- Configuration ---
# The files and glob patterns to include in the dump.
# Paths are relative to the project root.
FILE_PATTERNS = [
    "android_world/agents/custom/celil_*.py",
    "android_world/agents/base_agent.py",
    "android_world/env/interface.py",
    "android_world/env/json_action.py",
    "android_world/agents/llm_wrappers/qwen_vl_wrapper.py",
    "android_world/agents/llm_wrappers/gemini_gemma_wrapper.py",
    "config/default.yaml",
    "run_benchmark.sh",
    "run.py",
]

OUTPUT_FILE = "celil_dump.md"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    """
    Generates a single markdown file concatenating all specified project files
    for providing context to an LLM.
    """
    # Ensure we are in the project root
    os.chdir(PROJECT_ROOT)
    print(f"Working directory set to: {os.getcwd()}")

    # Remove the old dump file if it exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Removed existing {OUTPUT_FILE}")

    # Collect all file paths
    all_files = []
    for pattern in FILE_PATTERNS:
        all_files.extend(glob.glob(pattern, recursive=True))

    # Remove duplicates and sort for a consistent output order
    all_files = sorted(list(set(all_files)))

    print(f"Found {len(all_files)} files to include in the dump.")

    # Write the content to the output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as dump_file:
        dump_file.write("# Celil Agent Context Dump\n\n")
        dump_file.write(
            "This file contains a concatenation of multiple source files to provide "
            "context for an LLM. Below are the contents of the key files related "
            "to the `CelilAgent`.\n\n"
        )

        for file_path in all_files:
            print(f"  -> Adding {file_path}")
            try:
                with open(
                    file_path, "r", encoding="utf-8", errors="ignore"
                ) as source_file:
                    content = source_file.read()

                dump_file.write(f"\n---\n\n")
                dump_file.write(f"## File: `{file_path}`\n\n")
                # Use a language identifier for better syntax highlighting
                lang = "python" if file_path.endswith(".py") else "yaml" if file_path.endswith(".yaml") else "bash" if file_path.endswith(".sh") else ""
                dump_file.write(f"```{lang}\n")
                dump_file.write(content)
                dump_file.write("\n```\n")

            except Exception as e:
                print(f"    [!] Error reading {file_path}: {e}")

    print(f"\nSuccessfully created {OUTPUT_FILE} with the full context.")


if __name__ == "__main__":
    main()

