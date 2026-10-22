### MISSION: Architect, Implement, and Sandbox-Verify "ArgonOS v9" (Mobile-Touch Multi-Window Operating Platform)

You are an autonomous engineering squad (Lead Architect, Lead Coder, Test Engineer, and QA Critic). Your mission is to build, integrate, and verify `argon_code_v9.py`—a complete, touch-native, multi-window Web Desktop Operating System running entirely within a Kaggle notebook kernel.

---

### 🏛️ ARCHITECTURAL REQUIREMENTS

1. **ArgonOS Micro-Kernel & Window Manager Engine:**
   - **Modular App Architecture:** Build a clean `ArgonOS` class with a `register_app(app_instance)` API to decouple apps from the desktop engine.
   - **100% Touch-Safe Pointer Dragging:** Implement pointer-capture drag handles on window titlebars (`setPointerCapture`) that completely isolate mobile drag gestures from native browser page scrolling.
   - **Z-Index Layering (Tap-to-Focus):** Tapping or clicking any window must bring it to the foreground with an active neon accent border.
   - **Per-Window Controls:** Every window must include a drag header, an in-window Fullscreen/Maximize toggle (`□` / `🗗`) that expands the window to 100% of the desktop space, and a Close button (`✕`).
   - **Three-Section Bottom Taskbar:**
     - *Left:* System Telemetry Screen (Kernel status, Kaggle quota monitor, memory chip).
     - *Center:* Hardware Fullscreen button (`⛶ Fullscreen` / `🗗 Exit`) that toggles the desktop to 100vw × 100vh with a bulletproof CSS fixed fallback.
     - *Right:* Dynamic App Launcher Dock featuring app icons with active indicator styling.
   - **GPU-Accelerated RGB LED Edge Loading Strip:** Conic-gradient ambient neon sweep running around the desktop perimeter that automatically pulses and glows when querying LLM models or executing scripts.

2. **Core Application 1: 🤖 TalkToArgon & DyTopo Swarm Studio:**
   - Integrated LLM chat interface connected to `kaggle_benchmarks.llms` (defaulting to `google/gemini-4-argon`).
   - DyTopo interaction topology modes (Adaptive, Star, Chain, Debate, Tree) and role assignments.
   - Prompt input box with action triggers (`Send Prompt`, `Launch Swarm`, file attachment chips).
   - Inverted timeline (newest answers on top) with header & footer copy buttons.
   - Hooked to activate the RGB LED perimeter strip during generation cycles.

3. **Core Application 2: 📁 Argon Files (Workspace File Explorer):**
   - Live filesystem scanner for `/kaggle/working/` displaying real `.py`, `.sh`, `.json`, and `.txt` files with size chips.
   - Action buttons on every file: `✏️ Edit`, `▶️ Run`, `📋 Copy`, `👁️ Info`.

4. **Core Application 3: ✏️ Argon Code Editor (Mobile-Touch IDE):**
   - Dark-themed code editor window with a monospace editing textarea.
   - Capable of opening any workspace file, editing code with touch input, and saving changes directly back to disk (`Path.write_text`) with a `💾 Save File` button.

5. **Core Application 4: ⚡ Sandbox Console & Execution Terminal:**
   - Live script execution engine for `.py` (via IPython `%run`) and `.sh` (via bash) capturing stdout, stderr, and tracebacks into a dedicated terminal window.
   - Action bar featuring `▶️ Run Selected`, `📋 Copy Console Output`, and `↩️ Send Error/Output to AI` feedback loop.

6. **Core Application 5: ⚙️ System Settings & Telemetry (BIOS):**
   - Sliders for sampling Temperature and Memory turns, plus a System Prompt instruction editor.
   - Manual toggle button for the RGB LED perimeter lighting effect (`⚡ LED FX: ON / OFF`).

---

### 📋 EXECUTION POLICY & VERIFICATION RULES

- **Self-Contained File Export:** You MUST write the complete, functional, ready-to-execute codebase into `/kaggle/working/argon_code_v9.py` using `%%writefile argon_code_v9.py`.
- **Sandbox Execution:** You MUST invoke `RUN_FILE` on `argon_code_v9.py` inside the live IPython kernel.
- **Verification:** Inspect the execution log to confirm the kernel boots cleanly without syntax errors, mounts all registered apps, renders the desktop layout, and reports `PASS`.
- **Completion:** Conclude with a comprehensive final summary once `argon_code_v9.py` is written, executed, and confirmed working.
