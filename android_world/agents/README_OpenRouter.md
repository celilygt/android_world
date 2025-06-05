# M3A with OpenRouter Free Models

This directory contains an implementation of the M3A (Multimodal Autonomous Agent for Android) that uses **free open-source models** via OpenRouter.ai instead of paid APIs like OpenAI GPT-4 or Google Gemini.

## 🆓 Why Use OpenRouter?

- **Completely Free**: No API costs, no billing setup required
- **Open Source Models**: Use state-of-the-art open models like Gemma 3-27B, Llama 3.3 70B
- **Same Functionality**: Drop-in replacement for the original M3A agent
- **Easy Setup**: Just need a free OpenRouter API key

## 🚀 Quick Start

### 1. Get OpenRouter API Key (Free)
```bash
# 1. Sign up at https://openrouter.ai (free)
# 2. Get your API key from the dashboard
# 3. Set environment variable
export OPENROUTER_API_KEY="your_api_key_here"
```

### 2. Use the OpenRouter M3A Agent
```python
from android_world.agents.m3a_openrouter import create_m3a_openrouter_agent

# Create agent with default free Gemma 3-27B model
agent = create_m3a_openrouter_agent(env)

# Or use a different free model
agent = create_m3a_openrouter_agent(
    env, 
    model_name="meta-llama/llama-3.3-70b-instruct:free"
)
```

## 📁 Files Added

- `infer.py`: Added `OpenRouterWrapper` class implementing the LLM interface
- `m3a_openrouter.py`: M3A agent using OpenRouter instead of paid APIs
- `example_openrouter_usage.py`: Example usage and setup instructions

## 🔄 Migration from Paid APIs

### Before (Paid APIs)
```python
from android_world.agents.m3a import M3A
from android_world.agents.infer import GeminiGcpWrapper, Gpt4Wrapper

# Required paid API keys
llm = GeminiGcpWrapper(model_name="gemini-1.5-pro-vision")  # $$$
# or
llm = Gpt4Wrapper(model_name="gpt-4-vision-preview")  # $$$

agent = M3A(env=env, llm=llm)
```

### After (Free Models)
```python
from android_world.agents.m3a_openrouter import M3AOpenRouter

# Only free OpenRouter API key needed
agent = M3AOpenRouter(
    env=env,
    model_name="google/gemma-3-27b-it:free"  # FREE!
)
```

## 🤖 Available Free Models

| Model | Identifier | Size | Strengths |
|-------|------------|------|-----------|
| **Gemma 3-27B** | `google/gemma-3-27b-it:free` | 27B | Good balance of capability and speed |
| **Llama 3.3 70B** | `meta-llama/llama-3.3-70b-instruct:free` | 70B | High capability, slower |
| **Mistral 7B** | `mistralai/mistral-7b-instruct:free` | 7B | Fast, good for simple tasks |

## ⚙️ Configuration Options

```python
agent = M3AOpenRouter(
    env=env,
    model_name="google/gemma-3-27b-it:free",
    name="My-Custom-M3A",
    temperature=0.0,              # Deterministic responses
    max_retry=3,                  # Retry attempts on failure
    wait_after_action_seconds=2.0, # Wait time between actions
    site_url="https://your-site.com",  # Optional: for OpenRouter rankings
    site_name="Your Project Name"      # Optional: for OpenRouter rankings
)
```

## 🔍 Technical Details

### OpenRouterWrapper Implementation
- Implements both `LlmWrapper` and `MultimodalLlmWrapper` interfaces
- Uses the same image encoding (base64 JPEG) as the original GPT-4 wrapper
- Includes retry logic with exponential backoff
- Compatible with the existing M3A architecture

### API Compatibility
- Uses OpenAI-compatible chat completions format
- Supports multimodal input (text + images)
- Returns same interface as paid models: `(text_output, is_safe, raw_response)`

## 🛠️ Testing

```bash
# Run the example to test setup
python android_world/agents/example_openrouter_usage.py

# Test the wrapper directly
python -c "
from android_world.agents.infer import OpenRouterWrapper
wrapper = OpenRouterWrapper()
print('✅ OpenRouter wrapper created successfully!')
"
```

## 🔧 Troubleshooting

### Common Issues

1. **"OpenRouter API key not set" error**
   ```bash
   export OPENROUTER_API_KEY="your_key_here"
   ```

2. **Rate limiting**
   - Free models have rate limits
   - Increase `max_retry` and wait times if needed

3. **Model not found**
   - Check available models at https://openrouter.ai/models
   - Ensure using `:free` suffix for free models

### Performance Tips

- **For Speed**: Use `mistralai/mistral-7b-instruct:free`
- **For Quality**: Use `meta-llama/llama-3.3-70b-instruct:free`
- **Balanced**: Use `google/gemma-3-27b-it:free` (default)

## 📈 Cost Comparison

| API | Cost per 1000 images* | Setup |
|-----|---------------------|--------|
| OpenAI GPT-4V | ~$10-30 | Requires billing |
| Google Gemini | ~$2.50 | Requires billing |
| **OpenRouter Free** | **$0** | **Just sign up** |

*Approximate costs including text tokens

## 🤝 Contributing

If you find issues with the OpenRouter implementation or want to add support for more free models, please:

1. Check the model is available at https://openrouter.ai/models
2. Test with the existing `OpenRouterWrapper`
3. Update model lists in documentation
4. Submit a pull request

## 📚 Resources

- [OpenRouter.ai](https://openrouter.ai) - Free API access to open models
- [AndroidWorld](https://github.com/google-research/android_world) - Original project
- [Available Models](https://openrouter.ai/models) - Browse all free models 