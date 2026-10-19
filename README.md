# TalkToArgon10100102026 🤖⚡

Google wants 99 USD to talk to argon,  
Kaggle gives you 100 USD to talk to argon (and other models)  
All available models listed below 👇

---

## 🚀 Quick Start Guide

1. **Verify your Kaggle account:** Complete SMS/phone verification in your Kaggle profile to unlock free model proxy quotas.
2. **Open a Task Environment:** Visit [kaggle.com/benchmarks/tasks/new](https://www.kaggle.com/benchmarks/tasks/new) to spin up an environment with injected model proxy access.
3. **Load the Mini Client v3:** Copy the code from **[TalkToArgon.ipynb](https://github.com/peterbabulik/TalkToArgon10100102026/blob/805d61eca6df149015b9681a7011eaef8c31dd0c/TalkToArgon.ipynb)** into a code cell.
4. **Run & Iterate:** 
   - Ask Argon to generate a script (e.g. `Write a script that benchmarks numpy matrix multiplication with %%writefile bench.py`).
   - Click `▶️ Run bench.py`.
   - Inspect the terminal output or feed errors back directly with `Send Output/Error to AI`.

---


### 📦 Code Resource: **Mini Client v3**
Get the latest interactive workbench notebook:  
👉 **[Mini Client v3 (`TalkToArgon.ipynb`)](https://github.com/peterbabulik/TalkToArgon10100102026/blob/805d61eca6df149015b9681a7011eaef8c31dd0c/TalkToArgon.ipynb)**

---

## ⚡ What's New in Mini Client v3 (`KaggleLLMWorkbench`)

Mini Client v3 transforms the Kaggle notebook from a simple chat interface into a **full-fledged local execution and debugging IDE**:

- 🏃 **Live Execution Engine:** Extracted Python (`.py`) and Bash (`.sh`) scripts get dynamic `▶️ Run <file>` buttons to execute directly inside your running Kaggle kernel.
- 🔁 **Instant Debugging Feedback Loop:** Encountered an error or crash? Click **"Send Output/Error to AI"** to automatically feed the captured stdout/stderr traceback straight back into the prompt box for immediate fixes.
- ⚙️ **Collapsible Advanced Configuration:**
  - **Dynamic Temperature Slider:** Adjust sampling precision from `0.00` (deterministic code) to `1.50` (creative brainstorming).
  - **Sliding-Window Memory Turns:** Control context retention (1 to 50 turns) to prevent context blowup and manage quota efficiently.
  - **Custom System Prompt:** Define roles (e.g., Senior Python Architect, Security Auditor, Competitive Programmer) on the fly.
- ↩️ **Undo Last Turn:** Made a typo or dissatisfied with a response? Hit **Undo Last** to pop the exchange and restore your prompt back into the editor.
- 📁 **Smart Multi-Language File Extraction:** Auto-detects `%%writefile filename.ext` headers or automatically maps code fences (`py`, `sh`, `js`, `html`, `css`, `sql`, `cpp`, `json`) to clean files in `/kaggle/working/`.
- 🎨 **Rich Code & Table Rendering:** Styled Markdown viewer with dark-themed syntax blocks, tabular layouts, and mobile-safe raw markdown copy buttons.

---

## 📋 Available Frontier & Open Models in Kaggle Model Proxy

Switch between any of these models directly inside the client without entering private API keys or credit cards (covered by Kaggle's free $10/day / $100/mo AI quota):

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

