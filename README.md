# TalkToArgon10100102026 🤖⚡

Google wants 99 USD to talk to argon,  
Kaggle gives you 100 USD to talk to argon (and other models)  
All available models listed below 👇

---

## 🚀 Quick Start Guide (Single-Cell Install — Recommended)

1. **Verify your Kaggle account:** Complete SMS/phone verification in your Kaggle profile to unlock free model proxy quotas.
2. **Open a Task Environment:** Visit [kaggle.com/benchmarks/tasks/new](https://www.kaggle.com/benchmarks/tasks/new) to spin up an environment with injected model proxy access.
3. **Load Mini Client v5 (`argon_code_v5.py`):** Paste and run the snippet below in a notebook cell. It will download v5 and automatically populate the next cell with the full workbench code:

```python
import urllib.request

# Raw link to Version 5
url = "https://raw.githubusercontent.com/peterbabulik/TalkToArgon10100102026/1f3d462fdf7b6502bbfac97bc21bb32f24c9b38d/argon_code_v5.py"
filename = "argon_code_v5.py"

# 1. Download the file
urllib.request.urlretrieve(url, filename)
print(f"File {filename} downloaded successfully.")

# 2. Automatically load the code into the next cell
with open(filename, "r", encoding="utf-8") as f:
    get_ipython().set_next_input(f.read())
```

4. **Run the populated cell** to launch the `KaggleLLMWorkbench` UI.
5. **Run & Iterate:** 
   - Attach existing files from your workspace or upload local code files.
   - Ask Argon to generate a script (e.g. `Write a script that benchmarks numpy matrix multiplication with %%writefile bench.py`).
   - Click `▶️ Run bench.py` or `📋 Copy bench.py`.
   - Inspect the terminal output or feed errors back directly with `Send Output/Error to AI`.

---

### 📦 Code Resource: **Mini Client v5**
Get the latest standalone Python workbench script:  
👉 **[Mini Client v5 (`argon_code_v5.py`)](https://github.com/peterbabulik/TalkToArgon10100102026/blob/1f3d462fdf7b6502bbfac97bc21bb32f24c9b38d/argon_code_v5.py)**

---

## ⚡ What's New in Mini Client v5 (`KaggleLLMWorkbench`)

Mini Client v5 upgrades the Kaggle notebook into a **complete local execution, file-context, and debugging IDE**:

- 📎 **File Attachments (Workspace & Direct Upload):**
  - Attach existing files directly from `/kaggle/working/` or upload multiple files from your computer (`.py`, `.js`, `.ts`, `.html`, `.css`, `.sh`, `.sql`, `.json`, `.yaml`, `.md`, `.cpp`, `.c`, `.java`, `.csv`, `.txt`).
  - Automatic UTF-8 validation, line-count chips, and dynamic backtick fence scaling so nested code blocks never break your prompt.
- 📋 **Iframe-Safe One-Click Copy Everywhere:**
  - Dedicated **Copy Last** button in the main toolbar.
  - Per-message **Copy Markdown** and individual **Copy `<lang> #N`** buttons on every model response card.
  - **Copy `<filename>`** buttons alongside generated workspace files, powered by a kernel-to-browser JS bridge that bypasses Jupyter HTML sanitization and cross-origin iframe restrictions.
- 🏃 **Live Execution Engine:** Extracted Python (`.py`) and Bash (`.sh`) scripts get dynamic `▶️ Run <file>` buttons to execute directly inside your running Kaggle kernel—capturing stdout, stderr, tables, and plots.
- 🔁 **Instant Debugging Feedback Loop:** Encountered an error or crash? Click **"Send Output/Error to AI"** to automatically append the captured execution log/traceback straight into the prompt box for immediate fixes.
- ⚙️ **Collapsible Advanced Configuration:**
  - **Dynamic Temperature Slider:** Adjust sampling precision from `0.00` (deterministic code) to `1.50` (creative brainstorming).
  - **Sliding-Window Memory Turns:** Control context retention (1 to 50 turns) to prevent context blowup and manage quota efficiently.
  - **Custom System Prompt:** Define roles (e.g., Elite Software Architect, Security Auditor, Competitive Programmer) on the fly.
- ↩️ **Full-State Undo Last Turn:** Made a typo or dissatisfied with a response? Hit **Undo Last** to pop the exchange and restore both your prompt text **and** its attached files back into the editor.
- 📁 **Smart Multi-Language File Extraction:** Auto-detects `%%writefile filename.ext` headers or maps code fences (`py`, `sh`, `js`, `ts`, `html`, `css`, `sql`, `cpp`, `c`, `java`, `yaml`, `json`) to clean files in `/kaggle/working/` plus full session history in `argon_output.json`.
- 🎨 **Modern Adaptive UI (Light & Dark Mode):** Refined glassmorphic header, status badges, and automatic `prefers-color-scheme: dark` styling.

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
