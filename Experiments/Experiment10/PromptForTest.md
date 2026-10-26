🚀 [DyTopo Swarm Mission • Mode: ADAPTIVE]
Execute a comprehensive end-to-end regression and verification suite for the newly deployed ArgonOS v98 micro-kernel upgrades in a standalone script `/kaggle/working/test_v98_features.py`.

### Objectives:
Systematically verify that all 4 architectural improvements in ArgonOS v98 operate correctly without breaking existing process isolation or sandbox invariants:

1. **Test 1: JIT Shebang Dependency Auto-Installer (`_ensure_script_dependencies`):**
   - Create a test script `test_jit_shebang.py` with the header directive `# requirements: cowsay` (a tiny, fast-installing package).
   - Verify that `execute_file("test_jit_shebang.py")` automatically:
     a) Detects `cowsay` is missing in `sys.executable`.
     b) Silently invokes `pip install -q cowsay`.
     c) Successfully imports and executes the script.
     d) Captures the expected stdout output in the console.

2. **Test 2: Explicit Directive Timeout Enforcement (`# timeout: <seconds>`):**
   - Create `test_timeout_pass.py` with `# timeout: 5` that sleeps for 2.0s -> Assert it completes with status `PASS ✓`.
   - Create `test_timeout_kill.py` with `# timeout: 3` that loops for 10.0s -> Assert the watchdog terminates it via `SIGKILL` in ~3.0s and returns status `FAIL ✕` with the timeout report.

3. **Test 3: Heuristic AI Timeout Escalation (`_resolve_adaptive_timeout`):**
   - Pass code containing heavy AI keywords (`import torch`, `nn.Module`, `epochs = 12`, `train_losses`) without any `# timeout:` directive -> Assert `_resolve_adaptive_timeout(code)` escalates the limit to `600` seconds.
   - Pass plain utility code without AI keywords -> Assert `_resolve_adaptive_timeout(code)` resolves to default `120` seconds.

4. **Test 4: Clean Boot Verification (No Auto-Downloads):**
   - Verify that calling `self.verify_kernel_integrity()` passes with zero triggered browser downloads or unexpected file artifacts.

5. **Test 5: Swarm Directive Generation Verification:**
   - Prompt the Specialist Coder to generate a sample heavy workload script.
   - Assert that the generated script begins with both `# requirements: ...` and `# timeout: ...` directives at the top of the file as instructed.

---

### Execution Protocol:
1. Implement the automated assertions in `/kaggle/working/test_v98_features.py`.
2. Execute `/kaggle/working/test_v98_features.py` via `execute_file()`.
3. Output a structured Markdown scorecard summarizing the status of each test (PASS / FAIL, timing, and log captures).
4. Save the telemetry payload to `/kaggle/working/test_v98_results.json`.
