# Experiment 9: Autonomous Software Self-Update — JIT Shebang Dependencies & Adaptive Watchdogs 🔄🛡️

[![GitHub Tree](https://img.shields.io/badge/GitHub-Experiments%2FExperiment9-181717?logo=github)](https://github.com/peterbabulik/TalkToArgon10100102026/tree/main/Experiments/Experiment9)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Substrate](https://img.shields.io/badge/ArgonOS-v9.8%20Micro--Kernel-6366f1.svg)](https://github.com/peterbabulik/TalkToArgon10100102026)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An architectural evolution of the **ArgonOS v9.8** micro-kernel introducing native **Software Self-Update capabilities**: Just-In-Time (JIT) shebang dependency management and directive-driven adaptive execution timeouts. 

Tested and verified via a rigorous 5-part automated regression suite (Experiment 10`test_v98_features.py`), ArgonOS can now autonomously inspect generated script headers, resolve and install missing pip requirements on the fly, and dynamically scale its execution watchdog limits for heavy artificial intelligence and deep learning workloads.

---


## 🏛️ Architectural Upgrades in v9.8 (Experiment 9)

### 1. Just-In-Time (JIT) Shebang Dependency Management (`_ensure_script_dependencies`)
* **The Problem:** When Swarm agents generate standalone scripts utilizing specialized Python packages (e.g., `torch`, `flask`, or `triton`), fresh container runtimes frequently raise `ModuleNotFoundError`.
* **The Solution:** Before executing any script via `execute_file()`, the kernel parses the first 50 lines for shebang-style directives:
  ```python
  # requirements: torch, triton, pandas
  ```
  It checks importability via `importlib.util.find_spec()`, automatically runs `pip install -q` for missing packages under the active `sys.executable`, and streams installation logs safely to the console.

### 2. Directive-Driven & Heuristic Adaptive Timeouts (`_resolve_adaptive_timeout`)
* **The Problem:** The hardcoded 120-second execution watchdog (`SIGKILL` on timeout) prematurely terminated legitimate long-running ML training jobs (such as the 3-minute Tiny Shakespeare language modeling benchmark).
* **The Solution:** Implements a 3-tier adaptive timeout resolver:
  1. **Explicit In-File Directive:** Parses `# timeout: <seconds>` (e.g., `# timeout: 600` or `# timeout: 1200`) directly from the script header.
  2. **AI/ML Heuristics:** Automatically scans for computational keywords (`torch`, `triton`, `nn.Module`, `epochs =`, `train_losses`) and escalates the limit to **600 seconds (10 minutes)**.
  3. **Standard Default:** Falls back to `120` seconds for routine scripts.

---

## 📁 Repository & Experiment Artifacts

All files for Experiment 9 are archived in the [**`Experiments/Experiment9`**](https://github.com/peterbabulik/TalkToArgon10100102026/tree/main/Experiments/Experiment9) directory:

| File Artifact | Description |
| :--- | :--- |
| **[`argon_code_97.py`](argon_code_97.py)** | The core ArgonOS v9.8 micro-kernel source release featuring JIT dependency installation and adaptive watchdogs. |
| **[`PromptForUpdate.md`](PromptForUpdate.md)** | The exact senior-engineer prompt used to guide the Swarm through the autonomous self-update implementation. |
| **[`argon_output (9).json`](argon_output%20(9).json)** | The complete autonomous execution flight log and multi-agent reasoning trace. |
| **`test_v98_features.py`** | The automated end-to-end regression test suite verifying all 5 architectural test vectors. |
| **`test_v98_results.json`** | Structured JSON execution telemetry from the regression suite. |
| **`sample_heavy_workload.py`** | A reference script demonstrating top-of-file Swarm directive compliance (`# requirements` and `# timeout`). |

---

## 🚀 Quick Start & Reproduction

### Prerequisites
* Python 3.10+
* Isolated Linux container or Jupyter/Kaggle notebook environment.

### Running the Regression Suite
```bash
# 1. Clone repository and navigate to Experiment 9
git clone https://github.com/peterbabulik/TalkToArgon10100102026.git
cd TalkToArgon10100102026/Experiments/Experiment9

# 2. Execute the v9.8 regression and verification suite
python3 test_v98_features.py
```

---

## 📜 Citation

If you utilize the self-update architecture, JIT dependency installer, or adaptive watchdog engine in your research, please cite:

```bibtex
@misc{babulik2026argon_exp9,
  author       = {Babulik, Peter and the ArgonOS Autonomous Research Collective},
  title        = {Experiment 9: Autonomous Software Self-Update via JIT Shebang Dependencies and Adaptive Watchdogs in ArgonOS v9.8},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/peterbabulik/TalkToArgon10100102026/tree/main/Experiments/Experiment9}}
}
```
