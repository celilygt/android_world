#!/usr/bin/env python3
"""Example usage of M3A with OpenRouter free models.

This script demonstrates how to use the M3A agent with OpenRouter's free models
instead of paid APIs like OpenAI or Google Gemini.
"""

import os
from android_world.agents.m3a_openrouter import M3AOpenRouter, create_m3a_openrouter_agent


def setup_openrouter_environment():
    """Setup instructions for using OpenRouter.
    
    You need to:
    1. Sign up at https://openrouter.ai (free)
    2. Get your API key from the dashboard
    3. Set the OPENROUTER_API_KEY environment variable
    """
    if 'OPENROUTER_API_KEY' not in os.environ:
        print("❌ OPENROUTER_API_KEY environment variable not set!")
        print("\n📋 Setup Instructions:")
        print("1. Sign up at https://openrouter.ai (it's free)")
        print("2. Get your API key from the dashboard")
        print("3. Set the environment variable:")
        print("   export OPENROUTER_API_KEY='your_api_key_here'")
        print("\n💡 Available free models include:")
        print("   - google/gemma-3-27b-it:free")
        print("   - meta-llama/llama-3.3-70b-instruct:free")
        print("   - mistralai/mistral-7b-instruct:free")
        return False
    return True


def example_agent_creation():
    """Example of how to create and configure the M3A OpenRouter agent."""
    
    # Check if environment is set up
    if not setup_openrouter_environment():
        return None
    
    print("✅ OpenRouter API key found!")
    
    # This would normally be your Android environment
    # env = your_android_environment_here
    
    # Example 1: Using default Gemma 3-27B free model
    print("\n🤖 Creating M3A agent with default Gemma 3-27B model...")
    # agent = create_m3a_openrouter_agent(env)
    
    # Example 2: Using a different free model
    print("🤖 Creating M3A agent with Llama 3.3 70B model...")
    # agent = create_m3a_openrouter_agent(
    #     env,
    #     model_name="meta-llama/llama-3.3-70b-instruct:free"
    # )
    
    # Example 3: Custom configuration
    print("🤖 Creating M3A agent with custom settings...")
    # agent = M3AOpenRouter(
    #     env=env,
    #     model_name="google/gemma-3-27b-it:free",
    #     name="My-Custom-M3A",
    #     temperature=0.1,
    #     max_retry=5,
    #     wait_after_action_seconds=3.0,
    #     site_url="https://my-research-project.com",
    #     site_name="My Research Project"
    # )
    
    print("✅ Agent creation examples complete!")
    print("\n📝 Key benefits of using OpenRouter:")
    print("   - 🆓 Completely free (no API costs)")
    print("   - 🔓 Open source models")
    print("   - 🚀 Same M3A functionality")
    print("   - 🔧 Easy to switch between models")
    
    return True


def show_model_comparison():
    """Show comparison between paid and free model options."""
    print("\n📊 Model Comparison:")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│                    PAID MODELS                          │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│ • OpenAI GPT-4V: ~$0.01-0.03 per image + text          │")
    print("│ • Google Gemini Pro Vision: ~$0.0025 per image         │")
    print("│ • Requires API keys with billing setup                 │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│                    FREE MODELS (OpenRouter)             │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│ • Gemma 3-27B: google/gemma-3-27b-it:free             │")
    print("│ • Llama 3.3 70B: meta-llama/llama-3.3-70b-instruct:free│")
    print("│ • Mistral 7B: mistralai/mistral-7b-instruct:free       │")
    print("│ • Completely free, no billing required                 │")
    print("└─────────────────────────────────────────────────────────┘")


def main():
    """Main function to demonstrate OpenRouter setup and usage."""
    print("🚀 M3A OpenRouter Setup and Usage Example")
    print("=" * 50)
    
    show_model_comparison()
    example_agent_creation()
    
    print("\n🎯 Next Steps:")
    print("1. Set up your OPENROUTER_API_KEY environment variable")
    print("2. Replace the original M3A agent with M3AOpenRouter in your code")
    print("3. Enjoy free multimodal AI for Android automation!")


if __name__ == "__main__":
    main() 