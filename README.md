# TalkToArgon10100102026 🤖⚡

Google wants 99 USD to talk to argon,  
Kaggle gives you 100 USD to talk to argon (and other models)  
All available models listed below 👇

---

## 🚀 Quick Start Guide (Single-Cell Install — Recommended)

1. **Verify your Kaggle account:** Complete SMS/phone verification in your Kaggle profile to unlock free model proxy quotas.
2. **Open a Task Environment:** Visit [kaggle.com/benchmarks/tasks/new](https://www.kaggle.com/benchmarks/tasks/new) to spin up an environment with injected model proxy access.
3. **Load Workbench v7 (`argon_code_v7.py`):** Paste and run the snippet below in a notebook cell. It will download v7 and automatically populate the next cell with the full workbench code:

```python
import urllib.request

# Raw link to Version 7
url = "https://raw.githubusercontent.com/peterbabulik/TalkToArgon10100102026/5a532bb89ca47a03c319154629880e8f0333e646/argon_code_v7.py"
filename = "argon_code_v7.py"

# 1. Download the file
urllib.request.urlretrieve(url, filename)
print(f"File {filename} downloaded successfully.")

# 2. Automatically load the code into the next cell
with open(filename, "r", encoding="utf-8") as f:
    get_ipython().set_next_input(f.read())
```

4. **Run the populated cell** to launch the `KaggleLLMWorkbench` v7 UI.
5. **Run Single Prompt or SuperAgent:**
   - **Single Prompt:** Select any frontier model, attach workspace/local files, and click `Send Prompt`.
   - **Autonomous SuperAgent:** Enter a high-level goal (e.g. *"Build an optimized prime sieve with benchmarks and test suite in `prime.py`"*), configure your agent roles, and click `🚀 Launch SuperAgent`.

---

### 📦 Code Resource: **Workbench v7**
Get the latest standalone Python script:  
👉 **[Workbench v7 (`argon_code_v7.py`)](https://github.com/peterbabulik/TalkToArgon10100102026/blob/5a532bb89ca47a03c319154629880e8f0333e646/argon_code_v7.py)**

---

## ⚡ What's New in Version 7 (`KaggleLLMWorkbench`)

Version 7 transforms the notebook into an **autonomous multi-agent coding IDE and live execution environment**:

### 🤖 1. Autonomous Multi-Model SuperAgent Swarm
Orchestrate specialized frontier models working as an agile engineering squad:
- 👑 **Master Orchestrator:** Analyzes the objective, breaks tasks down into structured JSON plans, and coordinates the pipeline.
- 🛠️ **Lead Coder:** Implements complete, self-contained scripts starting with `%%writefile <filename>`.
- ⚡ **Kernel Sandbox Executor:** Executes generated Python (`.py`) and Bash (`.sh`) scripts directly inside your running IPython session to test outputs in real time.
- 🧪 **Critic / QA Reviewer:** Investigates runtime errors, inspects tracebacks, and provides root-cause fixes back to the Coder.
- ⏹️ **SuperAgent Controls:** Configurable maximum step slider (2–15 steps), autonomous execution toggle, and an instant **Stop Agent** kill-switch.

### ⬇️ 2. Inverted Timeline (Newest Responses at the Top)
- **Zero Scrolling:** Latest model responses and agent steps appear immediately at the **top** of the timeline.
- **Timeline Direction Toggle:** Switch seamlessly between **⬇️ Newest First (Top)** and **⬆️ Oldest First (Bottom)** at any time.

### 🏃 3. Live Kernel Execution & 750px Console
- **Direct Notebook Execution:** Runs extracted `.py` and `.sh` files inside the active kernel session with full capture of `stdout`, `stderr`, tables, and tracebacks.
- **Spacious Terminal View:** Expanded 750px full-width dark console.
- **Instant Debugging Feedback Loop:** Click **"Send Output/Error to AI"** to immediately pipe console stdout/stderr back into the prompt box for rapid troubleshooting.
- **Clipboard Output:** Dedicated **📋 Copy Console Output** button.

### 📋 4. Dual-Header & Code-Level Clipboard Engine
- **Dual Copy Action Bars:** Every model card features copy buttons at both the **top** and **bottom**, so you never have to scroll past long code blocks.
- **Granular Block Copy:** Individual `📋 Copy <language> #N` buttons for every detected snippet.
- **Iframe-Safe JS Bridge:** Kernel-to-browser clipboard bridge that works reliably inside Jupyter/Kaggle iframe sandboxes.

### 📎 5. Advanced File Attachments & Context Management
- Attach workspace files directly from `/kaggle/working/` or upload local files (`.py`, `.js`, `.ts`, `.sh`, `.sql`, `.json`, `.yaml`, `.md`, `.cpp`, `.java`, etc.).
- Dynamic backtick fence scaling to prevent nested code blocks from corrupting prompts.
- **Undo Last Turn:** Restores your prompt text and attached files back into the editor with full state preservation.

---

## 📋 Available Frontier & Open Models in Kaggle Model Proxy

Switch between any of these models inside the workbench without personal API keys (covered by Kaggle's free quota):

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
