You are a Senior Systems Architect and Python Runtime Engineer specializing in container sandboxing, process lifecycles, and autonomous developer tooling.

### Context:
In my cloud micro-kernel operating system (`ArgonOS v9.x` running inside an isolated Kaggle Linux container), code execution is handled by `execute_file(self, filename: str) -> tuple` using isolated subprocesses (`subprocess.Popen(..., start_new_session=True)`). 

We recently encountered two real-world operational friction points:
1. **Missing Dependencies on Clean Boot:** Long-running training scripts failed with `ModuleNotFoundError: No module named 'torch'` (or `flask`, `triton`) because fresh containers lack certain packages in `sys.executable`.
2. **Premature Watchdog Termination:** The hardcoded 120-second execution watchdog (`SIGKILL` on timeout) killed legitimate long-running ML training jobs (e.g., Tiny Shakespeare language modeling which requires 3–5 minutes).

---

### Objective:
Upgrade `execute_file()` and the Swarm agent instructions in `ArgonOS` with two features:
1. **Just-In-Time (JIT) Header Dependency Installer**
2. **Adaptive & Directive-Driven Execution Timeout Engine**

---

### Detailed Technical Specifications:

#### 1. JIT Shebang Dependency Management (`_ensure_script_dependencies`)
Implement a helper method `_ensure_script_dependencies(self, script_path: Path) -> List[str]`:
1. Read the script text before spawning the child process.
2. Search for a header directive matching `# requirements: <pkg1>, <pkg2>, ...` or `# pip: <pkg1>, <pkg2>, ...`.
3. For each declared package, check if it is importable in the current runtime environment using `importlib.util.find_spec(pkg)`.
4. If any packages are missing:
   - Update UI status via `self._set_status(f"📦 Installing missing dependencies: {missing_str}...", "#f59e0b")`.
   - Run `subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing_pkgs, check=True)`.
   - Log the installation success to `self.console_log_html`.
5. Gracefully catch installation errors without crashing the kernel (pipe any pip error to the console log and return failure if fatal).

#### 2. Adaptive & Directive-Driven Timeout (`_resolve_adaptive_timeout`)
Implement a helper method `_resolve_adaptive_timeout(self, code: str) -> int`:
1. **Priority 1 (Explicit In-File Directive):** Check if the script specifies `# timeout: <seconds>` (e.g., `# timeout: 600` or `# timeout: 1200`). If found, parse and enforce this exact integer value.
2. **Priority 2 (Adaptive Heuristic for AI/Compute Tasks):** If no explicit timeout directive is present, scan the script for compute-heavy keywords:
   - Framework imports: `torch`, `triton`, `tensorflow`, `jax`, `nn.Module`.
   - Training loop indicators: `epochs =`, `train_losses`, `for ep in range`, `hru_benchmark`.
   - If heavy AI/training patterns are detected, automatically escalate the timeout from the default 120s to **600 seconds (10 minutes)**.
3. **Priority 3 (Instance Config / Standard Default):** If neither directive nor heavy AI patterns are detected, fall back to `getattr(self, "exec_timeout_seconds", 120)` (default 120 seconds).

#### 3. Update `execute_file(self, filename: str) -> tuple`:
- Wire `_ensure_script_dependencies` and `_resolve_adaptive_timeout` directly before spawning `subprocess.Popen`.
- Retain all existing security invariants:
  - Working directory sandboxing (`self.work_dir`).
  - `start_new_session=True` with dedicated POSIX process group (`PGID`).
  - Active process pointer assignment `self._active_proc = proc` for instant `< 100ms` cancellation on **Stop Swarm** (`os.killpg(pgid, signal.SIGKILL)`).
  - Clean `self._active_proc = None` cleanup in a `finally:` block.

#### 4. Swarm Agent Prompt Directive Alignment:
Update the system prompt injected into the **Lead Architect** and **Specialist Coder** in `_run_swarm_worker`:
- Instruct the agents:
  *"When generating standalone scripts that require specific libraries or heavy computation, declare them at the very top of the file using directives:*
  *`# requirements: torch, triton`*
  *`# timeout: 600`"*

---

### Deliverables:
Provide clean, robust, fully commented Python code:
1. `_ensure_script_dependencies(self, file_path: Path) -> Tuple[bool, str]`
2. `_resolve_adaptive_timeout(self, code: str) -> int`
3. The complete refactored `execute_file(self, filename: str) -> tuple`
4. The updated Swarm agent prompt snippet showing the directive instructions.
