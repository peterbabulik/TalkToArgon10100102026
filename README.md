# TalkToArgon10100102026 🤖⚡

Google wants 99 USD to talk to argon,  
Kaggle gives you 100 USD to talk to argon (and other models)  
All available models listed below 👇

---

## 📋 Available Frontier & Open Models in Kaggle Model Proxy

You can switch to any of these models directly inside the client without entering any private API keys or credit cards (covered by Kaggle's free $10/day / $100/mo AI spend quota):

### 🌐 Google
- `google/gemini-4-argon` *(Default)*
- `google/gemini-3.8-flash`
- `google/gemini-3.6-flash`
- `google/gemini-3.5-flash-lite`
- `google/gemini-3.1-pro-preview`
- `google/gemini-3.1-flash-lite-preview`
- `google/gemini-3-flash-preview`
- `google/gemini-2.5-pro`
- `google/gemini-2.5-flash`
- `google/gemma-4-31b`
- `google/gemma-4-26b-a4b`

### ⚡ OpenAI
- `openai/gpt-6.1-sol`
- `openai/gpt-6-sol`
- `openai/gpt-6-luna`
- `openai/gpt-6-astra`
- `openai/gpt-5.6-terra`
- `openai/gpt-5.6-sol`
- `openai/gpt-5.6-luna`
- `openai/gpt-5.5-2026-04-23`
- `openai/gpt-5.4-2026-03-05`
- `openai/gpt-5.4-mini-2026-03-17`
- `openai/gpt-5.4-nano-2026-03-17`
- `openai/gpt-oss-120b`
- `openai/gpt-oss-20b`

### 🧠 Anthropic
- `anthropic/claude-opus-5-5@default`
- `anthropic/claude-opus-5@default`
- `anthropic/claude-opus-4-8@default`
- `anthropic/claude-opus-4-7@default`
- `anthropic/claude-opus-4-6@default`
- `anthropic/claude-opus-4-5@20251101`
- `anthropic/claude-sonnet-5-5@default`
- `anthropic/claude-sonnet-5@default`
- `anthropic/claude-sonnet-4-6@default`
- `anthropic/claude-haiku-5-5`
- `anthropic/claude-haiku-4-5@20251001`

### 🔬 DeepSeek, Qwen, xAI & Zhipu AI
- `deepseek-ai/deepseek-r1-0528`
- `qwen/qwen3-coder-480b-a35b-instruct`
- `qwen/qwen3-235b-a22b-instruct-2507`
- `qwen/qwen3-next-80b-a3b-thinking`
- `qwen/qwen3-next-80b-a3b-instruct`
- `xai/grok-4.6`
- `xai/grok-4.5-0708`
- `zai/glm-5`

---

## ✨ Features

- **Multi-Turn Chat with Full History:** Model remembers your code and conversation across multiple queries.
- **Model Switcher Dropdown:** Seamlessly switch between Gemini 4 Argon, Claude Opus, GPT-6, or Grok in the same chat.
- **Mobile-Friendly One-Click Copy:** Copies the raw response text to clipboard even inside sandboxed browser iframes.
- **Auto-Export to Disk:** Automatically writes the conversation history to `/kaggle/working/argon_output.json` and extracts Python blocks as standalone `.py` files.
- **In-Notebook Download Links:** Clickable download links right inside the output cell (no need to wrestle with Kaggle's mobile sidebar).

---

## 🚀 Quick Start (In under 60 seconds)

1. Ensure your Kaggle account is phone-verified (required to unlock the free AI quota).
2. Go to **[kaggle.com/benchmarks/tasks/new](https://www.kaggle.com/benchmarks/tasks/new)** to create a new Task notebook (where the proxy credentials are pre-configured).
3. Copy the code from `TalkToArgon.ipynb` (or the snippet below), paste it into a code cell, and hit **Run**.
4. Type your prompts or paste multiline code and start talking to Argon!