# Experiment 10: Closed-Loop Self-Testing & Automated Regression Suite 🧪✅

[![GitHub Tree](https://img.shields.io/badge/GitHub-Experiments%2FExperiment10-181717?logo=github)](https://github.com/peterbabulik/TalkToArgon10100102026/tree/main/Experiments/Experiment10)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Substrate](https://img.shields.io/badge/ArgonOS-v9.8%20Micro--Kernel-6366f1.svg)](https://github.com/peterbabulik/TalkToArgon10100102026)
[![Status](https://img.shields.io/badge/Verification-5%2F5%20PASS-10b981.svg)](https://github.com/peterbabulik/TalkToArgon10100102026/tree/main/Experiments/Experiment10)

The formal **Self-Testing and Regression Campaign** for **ArgonOS v9.8**. 

While Experiment 9 focused on implementing autonomous self-update mechanics (JIT Shebang installation and adaptive watchdogs), Experiment 10 establishes a **closed-loop verification harness** (`test_v98_features.py`) that programmatically exercises and validates every safety, dependency, and preemption invariant of the micro-kernel inside the Kaggle Linux container.

---

## 📊 Self-Test Verification Scorecard

The test suite executed successfully, validating all 5 functional vectors of the ArgonOS v9.8 release:

| Test ID | Architectural Feature / Invariant | Status | Duration (s) | Verification Details |
| :--- | :--- | :---: | :---: | :--- |
| **TEST-01** | **JIT Shebang Auto-Installer** (`_ensure_script_dependencies`) | **PASS ✓** | 1.451s | Discovered `# requirements: cowsay` header in `test_jit_shebang.py`, dynamically installed `cowsay` into `sys.executable`, and verified correct ASCII banner execution. |
| **TEST-02** | **Watchdog Directive Timeout Enforcement** (`# timeout: N`) | **PASS ✓** | 5.028s | Subtest 2a (`# timeout: 5`, 2s workload) completed cleanly (`PASS ✓`). Subtest 2b (`# timeout: 3`, 10s workload) was terminated by watchdog `SIGKILL` at 3.01s (`FAIL ✕` expected timeout). |
| **TEST-03** | **Heuristic AI Timeout Escalation** (`_resolve_adaptive_timeout`) | **PASS ✓** | <0.001s | Confirmed AI/ML code containing `import torch`, `nn.Module`, `epochs`, and `train_losses` automatically escalates to `600s` while standard utility scripts resolve to `120s`. |
| **TEST-04** | **Clean Boot Sandbox Invariants** (`verify_kernel_integrity`) | **PASS ✓** | <0.001s | Confirmed zero unsolicited browser download hooks (`FileDownload` / `javascript`) on boot and validated sandbox file paths. |
| **TEST-05** | **Swarm Directive Header Compliance** | **PASS ✓** | <0.001s | Verified `sample_heavy_workload.py` declares `# requirements: torch, triton` and `# timeout: 600` on lines 1–2, ensuring agent self-governance. |

---

## 🛠️ The Self-Testing Architecture

Experiment 10 proves that an autonomous micro-kernel can evaluate its own runtime hardening without human intervention. The self-test harness exercises the following kernel routines:

1. **JIT Shebang Parsing (`_ensure_script_dependencies`):**
   Scans the header lines of candidate scripts for `# requirements:` directives, inspects site-packages via `importlib.util.find_spec()`, and provisions missing dependencies on demand using pip subprocessing under `sys.executable`.
   * *Verified Script:* [`test_jit_shebang.py`](test_jit_shebang.py)
2. **Directive-Driven Watchdog Escalation (`_resolve_adaptive_timeout`):**
   Parses `# timeout: <seconds>` directives or scans for deep learning keywords (`torch`, `epochs`, `nn.Module`) to dynamically adjust execution timeouts, preventing premature termination of heavy training scripts while maintaining safety against infinite loops.
   * *Verified Scripts:* [`test_timeout_pass.py`](test_timeout_pass.py), [`test_timeout_kill.py`](test_timeout_kill.py), [`sample_heavy_workload.py`](sample_heavy_workload.py)
3. **Headless Invariant Regression:**
   Executes programmatic assertions over window manager portals, UI bindings, and telemetry loops to guarantee stable zero-regression boots.
   * *Verified Suite:* [`test_v98_features.py`](test_v98_features.py)

---

## 📁 Repository & Experiment Artifacts

All files for Experiment 10 are archived in the [**`Experiments/Experiment10`**](https://github.com/peterbabulik/TalkToArgon10100102026/tree/main/Experiments/Experiment10) directory:

| File Artifact | Description |
| :--- | :--- |
| **[`argon_code_98.py`](argon_code_98.py)** | The verified ArgonOS v9.8 micro-kernel source release featuring full self-testing capabilities. |
| **[`PromptForTest.md`](PromptForTest.md)** | The exact senior-engineer verification prompt used to instruct the Swarm to build the regression harness. |
| **[`argon_output (10).json`](argon_output%20(10).json)** | **The Complete Autonomous Execution Flight Recorder.** Contains the full multi-agent diagnostic trace of the DyTopo Swarm validating the self-test suite. |
| **`test_v98_features.py`** | The comprehensive 5-part end-to-end regression test suite script. |
| **`test_v98_results.json`** | Structured JSON execution telemetry proving the 5/5 `PASS` outcomes. |
| **`sample_heavy_workload.py`** | Reference script demonstrating top-of-file Swarm directive compliance. |
| **`test_jit_shebang.py`**, **`test_timeout_pass.py`**, **`test_timeout_kill.py`** | Isolated unit test scripts exercised by the kernel runner during test execution. |

---

## 🚀 Quick Start & Reproduction

### Prerequisites
* Python 3.10+
* Isolated Linux container or Jupyter/Kaggle notebook environment.

### Running the Self-Test Suite
```bash
# 1. Clone the repository and navigate to Experiment 10
git clone https://github.com/peterbabulik/TalkToArgon10100102026.git
cd TalkToArgon10100102026/Experiments/Experiment10

# 2. Execute the v9.8 automated regression and self-test suite
python3 test_v98_features.py
```

---

## 📜 Citation

If you utilize the closed-loop self-testing framework or the ArgonOS regression harness in your research, please cite:

```bibtex
@misc{babulik2026argon_exp10,
  author       = {Babulik, Peter and the ArgonOS Autonomous Research Collective},
  title        = {Experiment 10: Closed-Loop Self-Testing and Automated Regression Suite in ArgonOS v9.8},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/peterbabulik/TalkToArgon10100102026/tree/main/Experiments/Experiment10}}
}
```
