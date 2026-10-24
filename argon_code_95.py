import base64
import datetime
import html
import json
import math
import os
import signal
from pathlib import Path
import re
import shlex
import shutil
import sys
import time
import traceback

from IPython import get_ipython
from IPython.display import HTML, Javascript, display
from IPython.utils.capture import capture_output
import ipywidgets as widgets

try:
    import kaggle_benchmarks as kbench
    HAS_KBENCH = True
except Exception:
    kbench = None
    HAS_KBENCH = False

try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False


# Programmatic markdown fence to avoid literal triple backticks
CODE_FENCE = "`" * 3

UID = int(time.time() * 1000)
WORK_DIR = Path("/kaggle/working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

ATTACH_MAX_BYTES = 250_000
ATTACH_MAX_FILES = 10

TEXT_LANG_BY_EXT = {
    ".py": "python",
    ".sh": "bash",
    ".json": "json",
    ".txt": "text",
    ".md": "markdown",
    ".js": "javascript",
    ".ts": "typescript",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".csv": "text",
}

EXTENSION_MAP = {
    "python": "py",
    "py": "py",
    "bash": "sh",
    "shell": "sh",
    "sh": "sh",
    "json": "json",
    "javascript": "js",
    "js": "js",
    "typescript": "ts",
    "ts": "ts",
    "html": "html",
    "css": "css",
    "sql": "sql",
    "yaml": "yaml",
    "yml": "yml",
    "markdown": "md",
    "md": "md",
    "text": "txt",
    "txt": "txt",
}

DYTOPO_MODES = [
    ("Adaptive (Auto-Evolving Graph)", "adaptive"),
    ("Star (Master Orchestrator)", "star"),
    ("Chain (Sequential Pipeline)", "chain"),
    ("Peer Debate (Round-Robin Consensus)", "debate"),
    ("Hierarchical Tree (Tiered Delegation)", "tree"),
]

ROLE_METADATA = {
    "architect": {
        "title": "Lead Architect",
        "icon": "🏛️",
        "color": "#6366f1",
        "description": "System decomposition, edge routing & overall plan coordination",
    },
    "coder": {
        "title": "Specialist Coder",
        "icon": "🛠️",
        "color": "#0ea5e9",
        "description": "Implementation, production code & modular architecture",
    },
    "tester": {
        "title": "Test Engineer",
        "icon": "🧪",
        "color": "#10b981",
        "description": "Unit testing, assertion suites & edge case validation",
    },
    "critic": {
        "title": "Security / QA Critic",
        "icon": "🛡️",
        "color": "#f59e0b",
        "description": "Vulnerability scanning, sanity checks & code verification",
    },
    "benchmarker": {
        "title": "Benchmark Analyst",
        "icon": "⚡",
        "color": "#ec4899",
        "description": "Complexity analysis, memory/latency profiling & optimization",
    },
}


class ArgonApp:
    """Modular Application Descriptor for the ArgonOS v9 Micro-Kernel."""

    def __init__(
        self,
        app_id: str,
        title: str,
        icon: str,
        header_color: str,
        initial_pos: tuple = (50, 20),
        initial_size: tuple = ("840px", "530px"),
        content_html: str = "",
        widget: widgets.Widget = None,
        visible: bool = True,
        maximized: bool = False,
    ):
        self.app_id = app_id
        self.title = title
        self.icon = icon
        self.header_color = header_color
        self.top, self.left = initial_pos
        self.width, self.height = initial_size
        self.content_html = content_html
        self.widget = widget
        self.visible = visible
        self.maximized = maximized


class ArgonOS:
    """
    ArgonOS v9 Micro-Kernel & Mobile-Touch Multi-Window Operating System.
    Core Registered Applications:
      1. 'talk'    -> 🤖 TalkToArgon & DyTopo Swarm Studio
      2. 'files'   -> 📁 Argon Files (Workspace File Explorer)
      3. 'editor'  -> ✏️ Argon Code Editor (Mobile-Touch IDE)
      4. 'console' -> ⚡ Sandbox Console & Execution Terminal
      5. 'sysinfo' -> ⚙️ System Settings & Telemetry (BIOS)
    """

    def __init__(self, uid: int = UID, work_dir: Path = WORK_DIR):
        self.uid = uid
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.apps = {}

        # Runtime & LLM state
        self.chat_history = []
        self.attachments = []
        self._last_exec_log = ""
        self._last_raw_exec_output = ""
        self._saved_files = []
        self._agent_stop_requested = False
        self._active_proc = None
        self.led_active = False

        # DyTopo interaction graph state
        self.dytopo_graph = {
            "nodes": {
                role_id: {"status": "idle", "messages_handled": 0}
                for role_id in ROLE_METADATA
            },
            "edges": {},
            "transitions": [],
            "current_mode": "adaptive",
        }

        # Discover models from kaggle_benchmarks.llms
        if HAS_KBENCH and getattr(kbench, "llms", None):
            try:
                self.available_models = list(kbench.llms.keys())
            except Exception:
                self.available_models = ["google/gemini-4-argon"]
        else:
            self.available_models = ["google/gemini-4-argon"]

        if not self.available_models:
            self.available_models = ["google/gemini-4-argon"]

        self.default_model = self._pick_preferred_model(
            [
                "google/gemini-4-argon",
                "openai/gpt-6.1-sol",
                "anthropic/claude-opus-5-5@default",
            ]
        )
        self.default_coder_model = self._pick_preferred_model(
            [
                "anthropic/claude-opus-5-5@default",
                "qwen/qwen3-coder-480b-a35b-instruct",
                "google/gemini-4-argon",
                "openai/gpt-6.1-sol",
            ]
        )
        self.default_critic_model = self._pick_preferred_model(
            [
                "openai/gpt-6.1-sol",
                "deepseek-ai/deepseek-r1-0528",
                "google/gemini-4-argon",
                "anthropic/claude-opus-5-5@default",
            ]
        )

        # Hidden HTML/JS bridge (avoids nested Output capture clobbering)
        self._js_bridge = widgets.HTML(
            value="",
            layout=widgets.Layout(width="0px", height="0px", overflow="hidden"),
        )
        self._js_nonce = 0

        self._build_all_app_widgets()
        self._init_core_apps()
        self._bind_events()

    def _pick_preferred_model(self, candidates: list) -> str:
        for c in candidates:
            if c in self.available_models:
                return c
        return self.available_models[0]

    def _run_js(self, js_code: str):
        """Executes client-side JavaScript safely without clobbering IPython capture_output."""
        self._js_nonce += 1
        b64 = base64.b64encode(js_code.encode("utf-8")).decode("ascii")
        self._js_bridge.value = (
            f"<img src=\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='1' height='1' id='n_{self._js_nonce}'></svg>\" "
            f"style='display:none;' "
            f"onload=\"try{{Function(atob('{b64}'))();}}catch(e){{console.error(e);}}\" "
            f"data-nonce='{self._js_nonce}'>"
        )

    def register_app(self, app: ArgonApp):
        """Registers an ArgonApp instance with the ArgonOS Window Manager."""
        if not isinstance(app, ArgonApp):
            raise TypeError("Expected an ArgonApp instance.")
        self.apps[app.app_id] = app

    # ==========================================
    # 1. SYSTEM TELEMETRY & RGB LED CONTROLLER
    # ==========================================
    def _get_system_telemetry(self) -> dict:
        ram_str = "RAM: OK"
        try:
            meminfo = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        meminfo[parts[0].strip()] = int(parts[1].strip().split()[0])
            total_mb = meminfo.get("MemTotal", 0) / 1024
            avail_mb = meminfo.get("MemAvailable", 0) / 1024
            used_gb = (total_mb - avail_mb) / 1024
            total_gb = total_mb / 1024
            if total_gb > 0:
                ram_str = f"RAM: {used_gb:.1f}/{total_gb:.1f} GB"
        except Exception:
            pass

        disk_str = "Disk: OK"
        try:
            usage = shutil.disk_usage(self.work_dir)
            free_gb = usage.free / (1024 ** 3)
            total_d_gb = usage.total / (1024 ** 3)
            disk_str = f"Quota: {free_gb:.1f}GB free / {total_d_gb:.0f}GB"
        except Exception:
            pass

        return {
            "ram": ram_str,
            "disk": disk_str,
            "models": len(self.available_models),
            "files": len(self._list_workspace_files()),
        }

    def set_led_state(self, active: bool):
        """Activates or deactivates the GPU-Accelerated RGB LED Edge Loading Strip."""
        self.led_active = bool(active)
        if hasattr(self, "btn_led_toggle"):
            if self.led_active:
                self.btn_led_toggle.description = "⚡ LED FX: ON"
                self.btn_led_toggle.button_style = "warning"
            else:
                self.btn_led_toggle.description = "⚡ LED FX: OFF"
                self.btn_led_toggle.button_style = "info"

        if hasattr(self, "sysinfo_telemetry_html"):
            self.sysinfo_telemetry_html.value = self._render_sysinfo_telemetry_html()

        state_js = "true" if self.led_active else "false"
        self._run_js(
            f"if (window.argonSetLED) {{ window.argonSetLED('{self.uid}', {state_js}); }}"
        )

    def focus_app_window(self, app_id: str):
        """Brings a specific ArgonOS window to the front and ensures it is visible."""
        if app_id in self.apps:
            self.apps[app_id].visible = True
        self._run_js(
            f"if (window.argonOpenAppById) {{ window.argonOpenAppById('{app_id}', '{self.uid}'); }}"
        )

    # ==========================================
    # 2. BUILD ALL 5 CORE APP WIDGETS
    # ==========================================
    def _build_all_app_widgets(self):
        self._build_sysinfo_app_widget()
        self._build_talk_dytopo_app_widget()
        self._build_editor_app_widget()
        self._build_console_app_widget()
        self._build_files_app_widget()

    # ------------------------------------------
    # APP 5: ⚙️ SYSTEM SETTINGS & TELEMETRY BIOS
    # ------------------------------------------
    def _build_sysinfo_app_widget(self):
        self.temp_slider = widgets.FloatSlider(
            value=0.2,
            min=0.0,
            max=1.5,
            step=0.05,
            description="Temperature",
            readout_format=".2f",
            continuous_update=False,
            style={"description_width": "95px"},
            layout=widgets.Layout(width="300px", max_width="100%"),
        )

        self.max_turns_slider = widgets.IntSlider(
            value=10,
            min=1,
            max=50,
            step=1,
            description="Memory turns",
            continuous_update=False,
            style={"description_width": "95px"},
            layout=widgets.Layout(width="300px", max_width="100%"),
        )

        self.sys_prompt_input = widgets.Textarea(
            value=(
                "You are an elite coding assistant and software architect running inside ArgonOS v9. "
                "Analyze, write, test, and optimize production-ready code. "
                "Maintain full context across the conversation."
            ),
            layout=widgets.Layout(width="100%", height="90px"),
        )
        self.sys_prompt_input.add_class("kwb-system-prompt")

        self.btn_led_toggle = widgets.Button(
            description="⚡ LED FX: OFF",
            button_style="info",
            icon="bolt",
            tooltip="Toggle the RGB LED perimeter neon lighting strip",
            layout=widgets.Layout(width="165px", height="36px"),
        )

        self.btn_refresh_telemetry = widgets.Button(
            description="🔄 Refresh BIOS Telemetry",
            button_style="primary",
            layout=widgets.Layout(width="195px", height="36px"),
        )

        self.sysinfo_telemetry_html = widgets.HTML(
            value=self._render_sysinfo_telemetry_html()
        )

        self.sysinfo_widget = widgets.VBox(
            [
                self.sysinfo_telemetry_html,
                widgets.HBox(
                    [self.btn_led_toggle, self.btn_refresh_telemetry],
                    layout=widgets.Layout(margin="8px 0", flex_flow="row wrap"),
                ),
                widgets.HTML(
                    value="<div class='kwb-toolbar-label' style='margin-top:6px;'>LLM Sampling & Context Memory</div>"
                ),
                widgets.HBox(
                    [self.temp_slider, self.max_turns_slider],
                    layout=widgets.Layout(flex_flow="row wrap"),
                ),
                widgets.HTML(
                    value="<div class='kwb-toolbar-label' style='margin-top:6px;'>Global System Prompt Instruction</div>"
                ),
                self.sys_prompt_input,
            ],
            layout=widgets.Layout(width="100%", padding="10px"),
        )

    def _render_sysinfo_telemetry_html(self) -> str:
        t = self._get_system_telemetry()
        led_badge = "ACTIVE (ON)" if self.led_active else "STANDBY (OFF)"
        return f"""
        <div style="padding:12px; background:#090d16; border:1px solid #334155; border-radius:10px; color:#f1f5f9; font-size:12px; line-height:1.6;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <b style="color:#38bdf8; font-size:13px;">⚙️ ArgonOS v9 Micro-Kernel & BIOS Telemetry</b>
                <span style="background:#1e1b4b; color:#a5b4fc; padding:2px 8px; border-radius:999px; font-size:10.5px; font-weight:bold;">UID: {self.uid}</span>
            </div>
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(175px, 1fr)); gap:6px; font-size:11.5px;">
                <div>• Kernel Status: <span style="color:#10b981; font-weight:bold;">ONLINE (v9.0)</span></div>
                <div>• Touch Engine: <span style="color:#10b981; font-weight:bold;">PointerCapture 100%</span></div>
                <div>• Memory Chip: <span style="color:#38bdf8; font-weight:bold;">{html.escape(t['ram'])}</span></div>
                <div>• Kaggle Quota: <span style="color:#f59e0b; font-weight:bold;">{html.escape(t['disk'])}</span></div>
                <div>• Models Mounted: <span style="color:#a5b4fc; font-weight:bold;">{t['models']} LLMs</span></div>
                <div>• RGB Perimeter Strip: <span style="color:#ec4899; font-weight:bold;">{led_badge}</span></div>
            </div>
        </div>
        """

    # ------------------------------------------
    # APP 1: 🤖 TALKTOARGON & DYTOPO SWARM STUDIO
    # ------------------------------------------
    def _build_talk_dytopo_app_widget(self):
        self.model_dropdown = widgets.Dropdown(
            options=self.available_models,
            value=self.default_model,
            description="Active LLM",
            style={"description_width": "85px"},
            layout=widgets.Layout(width="360px", max_width="100%"),
        )

        self.dytopo_mode_dropdown = widgets.Dropdown(
            options=DYTOPO_MODES,
            value="adaptive",
            description="Topology",
            style={"description_width": "75px"},
            layout=widgets.Layout(width="310px", max_width="100%"),
        )

        self.role_selectors = {}
        for role_id, meta in ROLE_METADATA.items():
            preferred = (
                self.default_coder_model
                if role_id in ("coder", "tester")
                else (
                    self.default_critic_model
                    if role_id in ("critic", "benchmarker")
                    else self.default_model
                )
            )
            self.role_selectors[role_id] = widgets.Dropdown(
                options=self.available_models,
                value=preferred,
                description=f"{meta['icon']} {meta['title']}",
                style={"description_width": "140px"},
                layout=widgets.Layout(width="340px", max_width="100%"),
            )

        self.agent_max_steps_slider = widgets.IntSlider(
            value=6,
            min=2,
            max=20,
            step=1,
            description="Max swarm steps",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="240px", max_width="100%"),
        )

        self.agent_auto_exec_checkbox = widgets.Checkbox(
            value=True,
            description="Auto-run & verify generated scripts in Sandbox",
            indent=False,
            layout=widgets.Layout(width="auto"),
        )

        self.topology_graph_widget = widgets.HTML(
            value=self._render_dytopo_svg(),
            layout=widgets.Layout(width="100%"),
        )

        self.topology_history_html = widgets.HTML(
            value="<div style='color:#94a3b8; font-size:11.5px;'>No topology transitions recorded yet.</div>",
            layout=widgets.Layout(width="100%"),
        )

        role_grid = widgets.VBox(
            [
                widgets.HBox(
                    [self.role_selectors["architect"], self.role_selectors["coder"]],
                    layout=widgets.Layout(flex_flow="row wrap"),
                ),
                widgets.HBox(
                    [self.role_selectors["tester"], self.role_selectors["critic"]],
                    layout=widgets.Layout(flex_flow="row wrap"),
                ),
                widgets.HBox(
                    [self.role_selectors["benchmarker"]],
                    layout=widgets.Layout(flex_flow="row wrap"),
                ),
            ],
            layout=widgets.Layout(width="100%"),
        )

        dytopo_panel = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.dytopo_mode_dropdown,
                        self.agent_max_steps_slider,
                        self.agent_auto_exec_checkbox,
                    ],
                    layout=widgets.Layout(
                        width="100%", flex_flow="row wrap", align_items="center"
                    ),
                ),
                widgets.HTML(
                    value="<div class='kwb-toolbar-label' style='margin-top:4px;'>Specialist Role Model Assignments</div>"
                ),
                role_grid,
                widgets.HTML(
                    value="<div class='kwb-toolbar-label' style='margin-top:6px;'>Live DyTopo Interaction Graph</div>"
                ),
                self.topology_graph_widget,
                self.topology_history_html,
            ],
            layout=widgets.Layout(width="100%"),
        )

        self.dytopo_accordion = widgets.Accordion(
            children=[dytopo_panel],
            layout=widgets.Layout(width="100%"),
        )
        try:
            self.dytopo_accordion.set_title(
                0, "🕸️ DyTopo Topology Graph & Role Assignments"
            )
        except Exception:
            try:
                self.dytopo_accordion.titles = (
                    "🕸️ DyTopo Topology Graph & Role Assignments",
                )
            except Exception:
                pass
        self.dytopo_accordion.selected_index = None

        # File Attachment Controls (with FileUpload integration)
        self.attach_select = widgets.Dropdown(
            options=self._list_workspace_files(),
            description="Attach File",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="270px", max_width="100%"),
        )
        self.btn_attach_existing = widgets.Button(
            description="📎 Attach",
            button_style="info",
            layout=widgets.Layout(width="95px", height="32px"),
        )
        self.file_upload = widgets.FileUpload(
            accept="",
            multiple=True,
            description="⬆️ Upload",
            button_style="warning",
            layout=widgets.Layout(width="125px", height="32px"),
        )
        self.btn_clear_attachments = widgets.Button(
            description="Clear Chips",
            button_style="danger",
            layout=widgets.Layout(width="100px", height="32px"),
            disabled=True,
        )
        self.attachment_list = widgets.HTML(
            value="<div style='color:#94a3b8; font-size:11.5px;'>No files attached.</div>"
        )

        # Prompt Box
        self.textarea = widgets.Textarea(
            value="",
            placeholder=(
                "Ask google/gemini-4-argon a question, paste code, or describe a multi-step DyTopo Swarm mission...\n"
                "Tip: Code blocks with %%writefile filename.py are automatically saved to /kaggle/working/."
            ),
            layout=widgets.Layout(width="100%", height="110px"),
        )
        self.textarea.add_class("kwb-prompt")

        # Action Triggers
        self.btn_send = widgets.Button(
            description="Send Prompt",
            button_style="primary",
            icon="paper-plane",
            layout=widgets.Layout(width="135px", height="36px"),
        )
        self.btn_superagent = widgets.Button(
            description="Launch Swarm",
            button_style="warning",
            icon="share-alt",
            layout=widgets.Layout(width="145px", height="36px"),
        )
        self.btn_stop_agent = widgets.Button(
            description="Stop Swarm",
            button_style="danger",
            icon="stop",
            layout=widgets.Layout(width="115px", height="36px"),
            disabled=True,
        )
        self.btn_undo = widgets.Button(
            description="Undo",
            button_style="info",
            icon="undo",
            layout=widgets.Layout(width="85px", height="36px"),
            disabled=True,
        )
        self.btn_copy_last = widgets.Button(
            description="Copy Latest",
            button_style="info",
            icon="copy",
            layout=widgets.Layout(width="115px", height="36px"),
            disabled=True,
        )
        self.btn_save = widgets.Button(
            description="Export JSON",
            button_style="success",
            icon="download",
            layout=widgets.Layout(width="115px", height="36px"),
        )
        self.btn_reset = widgets.Button(
            description="Clear",
            button_style="danger",
            icon="trash",
            layout=widgets.Layout(width="85px", height="36px"),
        )

        self.status_label = widgets.HTML(
            value="<div class='kwb-status' style='color:#38bdf8;'>● Ready (TalkToArgon & DyTopo v9)</div>"
        )

        # Inverted timeline toggle (Newest on top by default)
        self.sort_order_toggle = widgets.ToggleButtons(
            options=[
                ("⬇️ Newest First (Top)", "newest"),
                ("⬆️ Oldest First (Bottom)", "oldest"),
            ],
            value="newest",
            style={"button_width": "155px", "font_weight": "bold"},
        )

        self.chat_empty_html = widgets.HTML(value=self._get_empty_chat_html())
        self.chat_container = widgets.VBox(
            [self.chat_empty_html],
            layout=widgets.Layout(width="100%", padding="10px"),
        )
        self.chat_container.add_class("kwb-chat-shell")

        top_bar = widgets.HBox(
            [self.model_dropdown, self.status_label],
            layout=widgets.Layout(
                width="100%",
                justify_content="space-between",
                align_items="center",
                flex_flow="row wrap",
            ),
        )

        attach_bar = widgets.HBox(
            [
                self.attach_select,
                self.btn_attach_existing,
                self.file_upload,
                self.btn_clear_attachments,
                self.attachment_list,
            ],
            layout=widgets.Layout(
                width="100%", align_items="center", flex_flow="row wrap"
            ),
        )

        action_bar = widgets.HBox(
            [
                self.btn_send,
                self.btn_superagent,
                self.btn_stop_agent,
                self.btn_undo,
                self.btn_copy_last,
                self.btn_save,
                self.btn_reset,
            ],
            layout=widgets.Layout(
                width="100%", align_items="center", flex_flow="row wrap"
            ),
        )

        timeline_header = widgets.HBox(
            [
                widgets.HTML(
                    value="<b style='color:#e2e8f0; font-size:13px;'>💬 Inverted Conversation & Swarm Timeline</b>"
                ),
                self.sort_order_toggle,
            ],
            layout=widgets.Layout(
                width="100%",
                justify_content="space-between",
                align_items="center",
                flex_flow="row wrap",
                margin="6px 0 2px 0",
            ),
        )

        self.talk_widget = widgets.VBox(
            [
                top_bar,
                self.dytopo_accordion,
                self.textarea,
                attach_bar,
                action_bar,
                timeline_header,
                self.chat_container,
            ],
            layout=widgets.Layout(width="100%", padding="10px"),
        )

    # ------------------------------------------
    # APP 2: 📁 ARGON FILES (WORKSPACE EXPLORER)
    # ------------------------------------------
    def _build_files_app_widget(self):
        self.files_upload_widget = widgets.FileUpload(
            accept="",
            multiple=True,
            description="⬆️ Upload to Workspace",
            button_style="info",
            layout=widgets.Layout(width="195px", height="34px"),
        )
        self.btn_files_refresh = widgets.Button(
            description="🔄 Rescan /kaggle/working",
            button_style="success",
            layout=widgets.Layout(width="195px", height="34px"),
        )
        self.files_summary_html = widgets.HTML()
        self.files_rows_box = widgets.VBox(layout=widgets.Layout(width="100%"))
        self.files_info_banner = widgets.HTML(
            value="<div style='color:#94a3b8; font-size:11.5px;'>Select an action (✏️ Edit, ▶️ Run, 📋 Copy, 👁️ Info) on any workspace file.</div>"
        )

        self.files_widget = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.files_summary_html,
                        widgets.HBox(
                            [self.files_upload_widget, self.btn_files_refresh],
                            layout=widgets.Layout(align_items="center", flex_flow="row wrap"),
                        ),
                    ],
                    layout=widgets.Layout(
                        width="100%",
                        justify_content="space-between",
                        align_items="center",
                        flex_flow="row wrap",
                    ),
                ),
                self.files_info_banner,
                self.files_rows_box,
            ],
            layout=widgets.Layout(width="100%", padding="10px"),
        )
        self.refresh_files_explorer()

    def refresh_files_explorer(self):
        """Scans /kaggle/working and builds interactive file rows + static HTML fallback."""
        try:
            all_files = [
                p
                for p in sorted(self.work_dir.iterdir())
                if p.is_file() and not p.name.startswith(".")
            ]
        except Exception:
            all_files = []

        self.files_summary_html.value = (
            f"<div style='color:#f1f5f9; font-size:12.5px;'>"
            f"📂 <b style='color:#38bdf8;'>{html.escape(str(self.work_dir))}</b> "
            f"<span style='background:#0f172a; border:1px solid #334155; padding:2px 8px; border-radius:999px; color:#10b981; font-size:11px; margin-left:6px;'>"
            f"{len(all_files)} files</span></div>"
        )

        rows = []
        for fpath in all_files:
            fname = fpath.name
            size_kb = fpath.stat().st_size / 1024.0
            ext = fpath.suffix.lower()
            badge_color = (
                "#38bdf8"
                if ext == ".py"
                else (
                    "#10b981"
                    if ext == ".sh"
                    else ("#f59e0b" if ext == ".json" else "#a5b4fc")
                )
            )

            label_html = widgets.HTML(
                value=(
                    f"<div style='display:flex; align-items:center; gap:8px; min-width:170px;'>"
                    f"<span style='color:{badge_color}; font-weight:700; font-size:12.5px;'>📄 {html.escape(fname)}</span>"
                    f"<span style='background:#090d16; border:1px solid #334155; color:#94a3b8; border-radius:6px; padding:1px 6px; font-size:10.5px;'>{size_kb:.1f} KB</span>"
                    f"</div>"
                )
            )

            btn_edit = widgets.Button(
                description="✏️ Edit",
                button_style="info",
                layout=widgets.Layout(width="74px", height="28px", margin="1px"),
            )
            btn_run = widgets.Button(
                description="▶️ Run",
                button_style="success",
                layout=widgets.Layout(width="74px", height="28px", margin="1px"),
            )
            btn_copy = widgets.Button(
                description="📋 Copy",
                layout=widgets.Layout(width="78px", height="28px", margin="1px"),
            )
            btn_copy.add_class("kwb-copy-widget-btn")
            btn_info = widgets.Button(
                description="👁️ Info",
                button_style="warning",
                layout=widgets.Layout(width="74px", height="28px", margin="1px"),
            )

            btn_edit.on_click(lambda _, fn=fname: self.open_file_in_editor(fn))
            btn_run.on_click(lambda _, fn=fname: self.run_file_in_console(fn))
            btn_copy.on_click(lambda btn, fn=fname: self._copy_workspace_file(fn, btn))
            btn_info.on_click(lambda _, fn=fname: self._show_file_info(fn))

            row = widgets.HBox(
                [
                    label_html,
                    widgets.HBox(
                        [btn_edit, btn_run, btn_copy, btn_info],
                        layout=widgets.Layout(flex_flow="row wrap"),
                    ),
                ],
                layout=widgets.Layout(
                    width="100%",
                    justify_content="space-between",
                    align_items="center",
                    padding="6px 10px",
                    margin="2px 0",
                    border="1px solid #334155",
                ),
            )
            rows.append(row)

        if not rows:
            rows.append(
                widgets.HTML(
                    value="<div style='color:#64748b; padding:14px; text-align:center;'>Workspace directory is currently empty.</div>"
                )
            )

        self.files_rows_box.children = rows

        # Sync dropdowns across other apps
        ws_list = self._list_workspace_files()
        if hasattr(self, "attach_select"):
            self.attach_select.options = ws_list
        if hasattr(self, "editor_file_dropdown"):
            try:
                self.editor_file_dropdown.unobserve(self._on_editor_dropdown_changed, names="value")
            except Exception:
                pass
            self.editor_file_dropdown.options = ws_list
            if hasattr(self, "editor_filename_input") and self.editor_filename_input.value in ws_list:
                self.editor_file_dropdown.value = self.editor_filename_input.value
            self.editor_file_dropdown.observe(self._on_editor_dropdown_changed, names="value")
        if hasattr(self, "console_file_dropdown"):
            exec_files = [
                f for f in ws_list if f.endswith((".py", ".sh"))
            ] or ws_list
            self.console_file_dropdown.options = exec_files

    def _show_file_info(self, filename: str):
        path = self.work_dir / filename
        if not path.is_file():
            self.files_info_banner.value = (
                f"<div style='color:#ef4444; font-size:12px;'>⚠ File {html.escape(filename)} not found.</div>"
            )
            return
        stat = path.stat()
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        try:
            lines = len(
                path.read_text(encoding="utf-8", errors="replace").splitlines()
            )
        except Exception:
            lines = 0
        self.files_info_banner.value = (
            f"<div style='padding:8px 10px; background:#090d16; border:1px solid #38bdf8; border-radius:8px; color:#e2e8f0; font-size:11.5px;'>"
            f"👁️ <b>{html.escape(filename)}</b> • Path: <code>{html.escape(str(path))}</code> • "
            f"Size: <b>{stat.st_size:,} bytes</b> • Lines: <b>{lines:,}</b> • Modified: {mtime}</div>"
        )

    # ------------------------------------------
    # APP 3: ✏️ ARGON CODE EDITOR (MOBILE IDE)
    # ------------------------------------------
    def _build_editor_app_widget(self):
        ws_files = self._list_workspace_files()
        initial_file = ws_files[0] if ws_files else "scratchpad.py"

        self.editor_file_dropdown = widgets.Dropdown(
            options=ws_files,
            value=ws_files[0] if ws_files else None,
            description="Open File",
            style={"description_width": "75px"},
            layout=widgets.Layout(width="250px", max_width="100%"),
        )
        self.editor_filename_input = widgets.Text(
            value=initial_file,
            placeholder="filename.py",
            description="Target Name",
            style={"description_width": "85px"},
            layout=widgets.Layout(width="250px", max_width="100%"),
        )
        self.btn_editor_load = widgets.Button(
            description="📂 Load",
            button_style="info",
            layout=widgets.Layout(width="85px", height="34px"),
        )
        self.btn_editor_save = widgets.Button(
            description="💾 Save File",
            button_style="success",
            layout=widgets.Layout(width="115px", height="34px"),
        )
        self.btn_editor_save_run = widgets.Button(
            description="▶️ Save & Run",
            button_style="warning",
            layout=widgets.Layout(width="125px", height="34px"),
        )
        self.btn_editor_copy = widgets.Button(
            description="📋 Copy Code",
            layout=widgets.Layout(width="115px", height="34px"),
        )
        self.btn_editor_copy.add_class("kwb-copy-widget-btn")

        self.editor_status_html = widgets.HTML(
            value="<div style='color:#38bdf8; font-size:11.5px;'>✏️ Touch-ready Monospace IDE</div>"
        )

        initial_code = (
            "# ArgonOS v9 Mobile-Touch Code Editor\nprint('Hello from ArgonOS v9!')\n"
        )
        if ws_files:
            try:
                initial_code = (self.work_dir / ws_files[0]).read_text(
                    encoding="utf-8", errors="replace"
                )
            except Exception:
                pass

        self.editor_textarea = widgets.Textarea(
            value=initial_code,
            placeholder="Write or edit Python / Bash / JSON code here...",
            layout=widgets.Layout(width="100%", height="300px"),
        )
        self.editor_textarea.add_class("kwb-editor-textarea")

        self.editor_widget = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.editor_file_dropdown,
                        self.btn_editor_load,
                        self.editor_filename_input,
                    ],
                    layout=widgets.Layout(
                        width="100%", flex_flow="row wrap", align_items="center"
                    ),
                ),
                widgets.HBox(
                    [
                        self.btn_editor_save,
                        self.btn_editor_save_run,
                        self.btn_editor_copy,
                        self.editor_status_html,
                    ],
                    layout=widgets.Layout(
                        width="100%", flex_flow="row wrap", align_items="center"
                    ),
                ),
                self.editor_textarea,
            ],
            layout=widgets.Layout(width="100%", padding="10px"),
        )

    def _on_editor_dropdown_changed(self, change):
        """Immediately syncs and loads file whenever the dropdown selection changes."""
        new_val = (
            change.get("new")
            if isinstance(change, dict)
            else getattr(change, "new", None)
        )
        if new_val:
            self.editor_filename_input.value = str(new_val)
            self.open_file_in_editor(str(new_val))

    def _on_editor_load_clicked(self, _):
        """Loads file prioritizing text input or dropdown selection."""
        target = (
            self.editor_filename_input.value.strip()
            or (self.editor_file_dropdown.value or "").strip()
        )
        if target:
            self.open_file_in_editor(target)
        else:
            self.editor_status_html.value = (
                "<span style='color:#f59e0b; font-size:12px;'>⚠ Enter filename or select from dropdown.</span>"
            )

    def open_file_in_editor(self, filename: str):
        """Loads a workspace file into the Argon Code Editor and brings the Editor window to front."""
        safe_name = os.path.basename(filename.strip())
        path = self.work_dir / safe_name
        if not path.is_file():
            self.editor_status_html.value = (
                f"<span style='color:#ef4444; font-size:12px;'>⚠ {html.escape(safe_name)} not found.</span>"
            )
            return
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            self.editor_filename_input.value = safe_name
            if safe_name in self.editor_file_dropdown.options:
                try:
                    self.editor_file_dropdown.unobserve(self._on_editor_dropdown_changed, names="value")
                except Exception:
                    pass
                self.editor_file_dropdown.value = safe_name
                self.editor_file_dropdown.observe(self._on_editor_dropdown_changed, names="value")
            self.editor_textarea.value = content
            self.editor_status_html.value = (
                f"<span style='color:#10b981; font-size:12px;'>✓ Loaded <b>{html.escape(safe_name)}</b> ({len(content):,} chars)</span>"
            )
            if "editor" in self.apps:
                self.apps["editor"].visible = True
            self.focus_app_window("editor")
        except Exception as exc:
            self.editor_status_html.value = (
                f"<span style='color:#ef4444; font-size:12px;'>✕ Error: {html.escape(str(exc))}</span>"
            )

    def save_editor_file(self, _=None) -> str:
        """Saves the current code in the Argon Code Editor directly to disk via Path.write_text."""
        raw_name = self.editor_filename_input.value.strip() or "untitled.py"
        safe_name = os.path.basename(raw_name)
        target_path = self.work_dir / safe_name
        content = self.editor_textarea.value
        target_path.write_text(content, encoding="utf-8")

        if safe_name not in self._saved_files:
            self._saved_files.append(safe_name)

        self.refresh_files_explorer()
        self.editor_filename_input.value = safe_name
        if safe_name in self.editor_file_dropdown.options:
            try:
                self.editor_file_dropdown.unobserve(self._on_editor_dropdown_changed, names="value")
            except Exception:
                pass
            self.editor_file_dropdown.value = safe_name
            self.editor_file_dropdown.observe(self._on_editor_dropdown_changed, names="value")

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.editor_status_html.value = (
            f"<span style='color:#10b981; font-size:12px;'>💾 Saved <b>{html.escape(safe_name)}</b> ({len(content):,} bytes) at {ts}</span>"
        )
        self._set_status(f"💾 Saved <b>{html.escape(safe_name)}</b> to disk.", "#10b981")
        return safe_name

    # ------------------------------------------
    # APP 4: ⚡ SANDBOX CONSOLE & TERMINAL
    # ------------------------------------------
    def _build_console_app_widget(self):
        ws_files = self._list_workspace_files()
        exec_files = [f for f in ws_files if f.endswith((".py", ".sh"))] or ws_files

        self.console_file_dropdown = widgets.Dropdown(
            options=exec_files,
            value=exec_files[0] if exec_files else None,
            description="Script",
            style={"description_width": "55px"},
            layout=widgets.Layout(width="260px", max_width="100%"),
        )
        self.btn_console_run = widgets.Button(
            description="▶️ Run Selected",
            button_style="success",
            layout=widgets.Layout(width="135px", height="34px"),
        )
        self.btn_copy_console = widgets.Button(
            description="📋 Copy Console Output",
            button_style="info",
            layout=widgets.Layout(width="180px", height="34px"),
        )
        self.btn_feed_output = widgets.Button(
            description="↩️ Send Error/Output to AI",
            button_style="warning",
            layout=widgets.Layout(width="200px", height="34px"),
        )
        self.btn_clear_console = widgets.Button(
            description="🧹 Clear",
            button_style="danger",
            layout=widgets.Layout(width="85px", height="34px"),
        )

        # Render terminal log in HTML + Output widget without entering Output context during __init__
        self.console_log_html = widgets.HTML(
            value=self._format_console_terminal(
                f"⚡ ArgonOS v9 Sandbox Execution Terminal Ready.\n📁 Working Directory: {self.work_dir}"
            )
        )
        self.exec_output = widgets.Output(
            layout=widgets.Layout(width="100%", max_height="220px", overflow_y="auto")
        )

        self.console_widget = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.console_file_dropdown,
                        self.btn_console_run,
                        self.btn_copy_console,
                        self.btn_feed_output,
                        self.btn_clear_console,
                    ],
                    layout=widgets.Layout(
                        width="100%", flex_flow="row wrap", align_items="center"
                    ),
                ),
                self.console_log_html,
                self.exec_output,
            ],
            layout=widgets.Layout(width="100%", padding="10px"),
        )

    @staticmethod
    def _format_console_terminal(text: str, status_color: str = "#10b981") -> str:
        return (
            f"<pre style='margin:6px 0; padding:12px; min-height:180px; max-height:320px; overflow-y:auto; "
            f"background:#050811; color:{status_color}; border:1px solid #334155; border-radius:10px; "
            f"font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size:12px; line-height:1.5; white-space:pre-wrap;'>"
            f"{html.escape(text)}</pre>"
        )

    def run_file_in_console(self, filename: str = None):
        target = filename or self.console_file_dropdown.value
        if not target:
            self._set_status("⚠ Select a script to execute first.", "#f59e0b")
            return
        if target in self.console_file_dropdown.options:
            self.console_file_dropdown.value = target
        self.focus_app_window("console")
        self.execute_file(target)

    # ==========================================
    # 3. REGISTER CORE APPS IN WINDOW MANAGER
    # ==========================================
    def _init_core_apps(self):
        # 1. TalkToArgon & DyTopo Swarm Studio
        self.register_app(
            ArgonApp(
                app_id="talk",
                title="TalkToArgon & DyTopo Swarm Studio",
                icon="🤖",
                header_color="linear-gradient(90deg, #312e81, #4f46e5, #0ea5e9)",
                initial_pos=(14, 14),
                initial_size=("680px", "540px"),
                content_html=self._get_empty_chat_html(),
                widget=self.talk_widget,
                visible=True,
            )
        )

        # 2. Argon Files (Workspace File Explorer)
        self.register_app(
            ArgonApp(
                app_id="files",
                title="Argon Files Explorer",
                icon="📁",
                header_color="linear-gradient(90deg, #065f46, #059669, #10b981)",
                initial_pos=(46, 360),
                initial_size=("540px", "430px"),
                content_html=self._render_static_files_fallback(),
                widget=self.files_widget,
                visible=True,
            )
        )

        # 3. Argon Code Editor (Mobile-Touch IDE)
        self.register_app(
            ArgonApp(
                app_id="editor",
                title="Argon Code Editor (Mobile IDE)",
                icon="✏️",
                header_color="linear-gradient(90deg, #0369a1, #0284c7, #38bdf8)",
                initial_pos=(78, 120),
                initial_size=("620px", "470px"),
                content_html="<div style='padding:12px; color:#38bdf8;'>✏️ Argon Code Editor Ready</div>",
                widget=self.editor_widget,
                visible=False,
            )
        )

        # 4. Sandbox Console & Execution Terminal
        self.register_app(
            ArgonApp(
                app_id="console",
                title="Sandbox Console & Execution Terminal",
                icon="⚡",
                header_color="linear-gradient(90deg, #9a3412, #ea580c, #f59e0b)",
                initial_pos=(105, 180),
                initial_size=("650px", "430px"),
                content_html="<div style='padding:12px; color:#10b981;'>⚡ Sandbox Execution Terminal Ready</div>",
                widget=self.console_widget,
                visible=False,
            )
        )

        # 5. System Settings & Telemetry (BIOS)
        self.register_app(
            ArgonApp(
                app_id="sysinfo",
                title="System Settings & Telemetry (BIOS)",
                icon="⚙️",
                header_color="linear-gradient(90deg, #4c1d95, #6d28d9, #8b5cf6)",
                initial_pos=(130, 260),
                initial_size=("580px", "450px"),
                content_html=self._render_sysinfo_telemetry_html(),
                widget=self.sysinfo_widget,
                visible=False,
            )
        )

    def _render_static_files_fallback(self) -> str:
        """Provides immediate HTML file list inside the window prior to widget portal mount."""
        files = self._list_workspace_files()
        chips = "".join(
            f"<span style='display:inline-block; margin:2px; padding:3px 8px; background:#0f172a; border:1px solid #334155; border-radius:6px; color:#38bdf8; font-size:11px;'>📄 {html.escape(f)}</span>"
            for f in files[:12]
        )
        return f"<div style='padding:8px; color:#94a3b8; font-size:11px;'>Workspace ({len(files)} files): {chips}</div>"

    # ==========================================
    # FILE UPLOAD PARSERS & HANDLERS
    # ==========================================
    def _extract_uploaded_files(self, raw_value) -> list:
        """
        Extracts uploaded files safely supporting both ipywidgets v7 (dict) and v8 (tuple/list of dicts).
        Returns a list of dicts: [{'name': filename, 'content': bytes, 'size': int}].
        """
        extracted = []
        if not raw_value:
            return extracted

        # ipywidgets v8: tuple or list of dicts [{'name': ..., 'content': <memoryview/bytes>, ...}]
        if isinstance(raw_value, (list, tuple)):
            for item in raw_value:
                if isinstance(item, dict):
                    name = item.get("name", "uploaded_file")
                    raw_content = item.get("content", b"")
                    if isinstance(raw_content, memoryview):
                        content_bytes = raw_content.tobytes()
                    elif isinstance(raw_content, bytes):
                        content_bytes = raw_content
                    elif isinstance(raw_content, str):
                        content_bytes = raw_content.encode("utf-8")
                    else:
                        content_bytes = bytes(raw_content)
                    extracted.append(
                        {
                            "name": os.path.basename(name),
                            "content": content_bytes,
                            "size": len(content_bytes),
                        }
                    )
        # ipywidgets v7: dict mapping filename -> {'metadata': ..., 'content': bytes}
        elif isinstance(raw_value, dict):
            for name, meta in raw_value.items():
                if isinstance(meta, dict):
                    raw_content = meta.get("content", b"")
                    if isinstance(raw_content, memoryview):
                        content_bytes = raw_content.tobytes()
                    elif isinstance(raw_content, bytes):
                        content_bytes = raw_content
                    elif isinstance(raw_content, str):
                        content_bytes = raw_content.encode("utf-8")
                    else:
                        content_bytes = bytes(raw_content)
                    extracted.append(
                        {
                            "name": os.path.basename(name),
                            "content": content_bytes,
                            "size": len(content_bytes),
                        }
                    )
        return extracted

    @staticmethod
    def _clear_upload_widget(widget):
        """Safely clears a FileUpload widget value across ipywidgets v7 and v8."""
        try:
            widget.value.clear()
        except Exception:
            try:
                widget.value = ()
            except Exception:
                try:
                    widget.value = {}
                except Exception:
                    pass
        try:
            widget._counter = 0
        except Exception:
            pass

    def on_file_upload_changed(self, change):
        """Processes uploads from TalkToArgon attach bar, saves to disk, and attaches text files to context."""
        raw_val = (
            change.get("new")
            if isinstance(change, dict)
            else getattr(change, "new", None)
        )
        if not raw_val:
            return

        files = self._extract_uploaded_files(raw_val)
        if not files:
            return

        saved_count = 0
        attached_count = 0

        for f in files:
            fname = f["name"]
            content_bytes = f["content"]
            dest = self.work_dir / fname
            try:
                dest.write_bytes(content_bytes)
                saved_count += 1
                if fname not in self._saved_files:
                    self._saved_files.append(fname)
            except Exception:
                continue

            if len(self.attachments) < ATTACH_MAX_FILES:
                ext = Path(fname).suffix.lower()
                lang = TEXT_LANG_BY_EXT.get(ext, "text")
                try:
                    text_str = content_bytes[:ATTACH_MAX_BYTES].decode(
                        "utf-8", errors="replace"
                    )
                    self.attachments = [
                        a for a in self.attachments if a["name"] != fname
                    ]
                    self.attachments.append(
                        {
                            "name": fname,
                            "content": text_str,
                            "lang": lang,
                        }
                    )
                    attached_count += 1
                except Exception:
                    pass

        self._clear_upload_widget(self.file_upload)
        self.refresh_files_explorer()
        self._refresh_attachment_view()
        self._set_status(
            f"⬆️ Uploaded {saved_count} file(s) ({attached_count} attached to prompt context).",
            "#10b981",
        )

    def on_workspace_upload_changed(self, change):
        """Processes uploads from Argon Files Explorer directly into /kaggle/working."""
        raw_val = (
            change.get("new")
            if isinstance(change, dict)
            else getattr(change, "new", None)
        )
        if not raw_val:
            return

        files = self._extract_uploaded_files(raw_val)
        if not files:
            return

        saved_names = []
        for f in files:
            fname = f["name"]
            content_bytes = f["content"]
            dest = self.work_dir / fname
            try:
                dest.write_bytes(content_bytes)
                saved_names.append(fname)
                if fname not in self._saved_files:
                    self._saved_files.append(fname)
            except Exception:
                continue

        self._clear_upload_widget(self.files_upload_widget)
        self.refresh_files_explorer()

        if saved_names:
            names_str = ", ".join(saved_names[:3])
            if len(saved_names) > 3:
                names_str += f" (+{len(saved_names) - 3} more)"
            self.files_info_banner.value = (
                f"<div style='padding:8px 10px; background:#064e3b; border:1px solid #10b981; border-radius:8px; color:#ecfdf5; font-size:11.5px;'>"
                f"✓ Uploaded <b>{len(saved_names)} file(s)</b> to workspace: <code>{html.escape(names_str)}</code></div>"
            )
            self._set_status(
                f"⬆️ Successfully uploaded {len(saved_names)} file(s) to workspace.",
                "#10b981",
            )

    def _bind_events(self):
        # TalkToArgon & DyTopo events
        self.btn_send.on_click(self.on_send_clicked)
        self.btn_superagent.on_click(self.on_dytopo_launch_clicked)
        self.btn_stop_agent.on_click(self.on_stop_agent_clicked)
        self.btn_undo.on_click(self.on_undo_clicked)
        self.btn_copy_last.on_click(self.on_copy_last_clicked)
        self.btn_save.on_click(self.on_save_clicked)
        self.btn_reset.on_click(self.on_reset_clicked)
        self.sort_order_toggle.observe(
            lambda _: self.refresh_chat_view(), names="value"
        )
        self.dytopo_mode_dropdown.observe(
            self._on_topology_mode_changed, names="value"
        )

        # File Attachments & Uploads
        self.btn_attach_existing.on_click(self.on_attach_existing_clicked)
        self.btn_clear_attachments.on_click(self.on_clear_attachments_clicked)
        self.file_upload.observe(self.on_file_upload_changed, names="value")

        # Files Explorer & Workspace Upload
        self.btn_files_refresh.on_click(lambda _: self.refresh_files_explorer())
        self.files_upload_widget.observe(
            self.on_workspace_upload_changed, names="value"
        )

        # Code Editor
        self.btn_editor_load.on_click(self._on_editor_load_clicked)
        self.editor_file_dropdown.observe(
            self._on_editor_dropdown_changed, names="value"
        )
        self.btn_editor_save.on_click(self.save_editor_file)
        self.btn_editor_save_run.on_click(self._on_editor_save_and_run)
        self.btn_editor_copy.on_click(
            lambda btn: self._copy_text_to_clipboard(
                self.editor_textarea.value,
                label=self.editor_filename_input.value or "Editor Code",
                button=btn,
            )
        )

        # Sandbox Console
        self.btn_console_run.on_click(
            lambda _: self.run_file_in_console(self.console_file_dropdown.value)
        )
        self.btn_copy_console.on_click(self._on_copy_console_clicked)
        self.btn_feed_output.on_click(self._on_feed_output_clicked)
        self.btn_clear_console.on_click(self._on_clear_console_clicked)

        # System Settings & BIOS
        self.btn_led_toggle.on_click(
            lambda _: self.set_led_state(not self.led_active)
        )
        self.btn_refresh_telemetry.on_click(self._on_refresh_bios_clicked)

    def _on_editor_save_and_run(self, _):
        saved_name = self.save_editor_file()
        self.run_file_in_console(saved_name)

    def _on_clear_console_clicked(self, _):
        self.exec_output.clear_output(wait=True)
        self.console_log_html.value = self._format_console_terminal(
            "⚡ Console cleared."
        )

    def _on_refresh_bios_clicked(self, _):
        self.sysinfo_telemetry_html.value = self._render_sysinfo_telemetry_html()
        self.refresh_files_explorer()

    # ==========================================
    # 4. DYTOPO TOPOLOGY GRAPH & ROUTING ENGINE
    # ==========================================
    def _on_topology_mode_changed(self, change):
        new_mode = change["new"]
        self.dytopo_graph["current_mode"] = new_mode
        self._record_topology_transition(
            step="Config",
            source="Operator",
            action=f"Switched topology mode to {new_mode.upper()}",
            channel="Orchestration",
        )
        self._refresh_topology_view()

    def _record_topology_transition(self, step, source, action, channel):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.dytopo_graph["transitions"].append(
            {
                "step": step,
                "source": source,
                "action": action,
                "channel": channel,
                "time": ts,
            }
        )
        items_html = []
        for t in reversed(self.dytopo_graph["transitions"][-5:]):
            items_html.append(
                f"<div style='margin-bottom:3px; font-size:11px;'>"
                f"<span style='color:#a5b4fc; font-weight:700;'>[{t['time']}]</span> "
                f"<span style='color:#38bdf8; font-weight:700;'>{html.escape(str(t['source']))}</span> "
                f"➔ <span style='color:#e2e8f0;'>{html.escape(str(t['action']))}</span> "
                f"<span style='color:#94a3b8;'>({html.escape(str(t['channel']))})</span></div>"
            )
        self.topology_history_html.value = (
            "<div style='max-height:85px; overflow-y:auto; padding:6px 8px; background:#090e1a; border-radius:8px; border:1px solid #1e293b;'>"
            + "".join(items_html)
            + "</div>"
        )

    def _record_edge_interaction(self, src: str, dst: str):
        key = f"{src}->{dst}"
        self.dytopo_graph["edges"][key] = self.dytopo_graph["edges"].get(key, 0) + 1
        if src in self.dytopo_graph["nodes"]:
            self.dytopo_graph["nodes"][src]["messages_handled"] += 1
        if dst in self.dytopo_graph["nodes"]:
            self.dytopo_graph["nodes"][dst]["messages_handled"] += 1

    def _set_node_status(self, role_id: str, status: str):
        if role_id in self.dytopo_graph["nodes"]:
            self.dytopo_graph["nodes"][role_id]["status"] = status
        self._refresh_topology_view()

    def _reset_node_statuses(self):
        for role_id in self.dytopo_graph["nodes"]:
            self.dytopo_graph["nodes"][role_id]["status"] = "idle"
        self._refresh_topology_view()

    def _render_dytopo_svg(self) -> str:
        width, height = 760, 220
        cx, cy = width // 2, height // 2
        roles = list(ROLE_METADATA.keys())
        node_coords = {}
        mode = self.dytopo_graph.get("current_mode", "adaptive")

        if mode == "star":
            node_coords["architect"] = (cx, cy)
            outer = [r for r in roles if r != "architect"]
            for i, r in enumerate(outer):
                ang = 2 * math.pi * i / len(outer)
                node_coords[r] = (
                    cx + int(210 * math.cos(ang)),
                    cy + int(78 * math.sin(ang)),
                )
        elif mode == "chain":
            spacing = width // (len(roles) + 1)
            for i, r in enumerate(roles):
                node_coords[r] = (spacing * (i + 1), cy)
        elif mode == "tree":
            node_coords["architect"] = (cx, 38)
            node_coords["coder"] = (cx - 150, 110)
            node_coords["tester"] = (cx - 195, 180)
            node_coords["critic"] = (cx + 150, 110)
            node_coords["benchmarker"] = (cx + 195, 180)
        else:
            for i, r in enumerate(roles):
                ang = (2 * math.pi * i / len(roles)) - (math.pi / 2)
                node_coords[r] = (
                    cx + int(260 * math.cos(ang)),
                    cy + int(74 * math.sin(ang)),
                )

        svg_parts = []
        for key, count in self.dytopo_graph["edges"].items():
            try:
                src, dst = key.split("->")
                if src in node_coords and dst in node_coords:
                    x1, y1 = node_coords[src]
                    x2, y2 = node_coords[dst]
                    sw = min(1.5 + count * 1.2, 5.5)
                    svg_parts.append(
                        f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#6366f1' stroke-width='{sw}' stroke-opacity='0.7' stroke-dasharray='4,4'/>"
                    )
            except Exception:
                continue

        for r_id, (nx, ny) in node_coords.items():
            meta = ROLE_METADATA[r_id]
            st = self.dytopo_graph["nodes"][r_id]["status"]
            mc = self.dytopo_graph["nodes"][r_id]["messages_handled"]
            b_col = (
                "#38bdf8"
                if st == "active"
                else ("#10b981" if st == "completed" else meta["color"])
            )
            svg_parts.append(
                f"<g>"
                f"<circle cx='{nx}' cy='{ny}' r='22' fill='#1e293b' stroke='{b_col}' stroke-width='2.5'/>"
                f"<text x='{nx}' y='{ny + 4}' text-anchor='middle' font-size='14'>{meta['icon']}</text>"
                f"<text x='{nx}' y='{ny + 35}' text-anchor='middle' fill='#cbd5e1' font-size='10' font-weight='600'>{meta['title']} ({mc})</text>"
                f"</g>"
            )

        active_mode_title = next((t for t, k in DYTOPO_MODES if k == mode), mode)
        total_ex = sum(self.dytopo_graph["edges"].values())
        return f"""
        <div style="padding:10px; border-radius:10px; background:#090d16; border:1px solid #334155;">
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#94a3b8; margin-bottom:4px;">
                <span>Mode: <b style="color:#a5b4fc;">{active_mode_title}</b></span>
                <span>Edges Routed: <b style="color:#38bdf8;">{total_ex}</b></span>
            </div>
            <svg viewBox="0 0 {width} {height}" style="width:100%; height:185px; display:block;">
                {''.join(svg_parts)}
            </svg>
        </div>
        """

    def _refresh_topology_view(self):
        self.topology_graph_widget.value = self._render_dytopo_svg()

    # ==========================================
    # 5. LLM QUERYING & DYTOPO SWARM EXECUTION
    # ==========================================
    def _prompt_llm(
        self, model_name: str, prompt_text: str, temperature: float = 0.2
    ) -> str:
        if HAS_KBENCH and getattr(kbench, "llms", None) and model_name in kbench.llms:
            llm = kbench.llms[model_name]
            return str(llm.prompt(prompt_text, temperature=temperature))
        return (
            f"**[ArgonOS Offline Simulation • `{model_name}`]**\n\n"
            f"Processed prompt ({len(prompt_text):,} chars) at temperature `{temperature:.2f}`."
        )

    def _build_context(self, new_user_input: str, new_attachments: list) -> str:
        sys_prompt = self.sys_prompt_input.value.strip()
        parts = [f"System:\n{sys_prompt}\n"] if sys_prompt else []
        max_msgs = self.max_turns_slider.value * 2
        recent = self.chat_history[-max_msgs:] if max_msgs > 0 else self.chat_history

        for msg in recent:
            if msg["role"] == "user":
                r_str = "User"
                body = self._compose_prompt(
                    msg["text"], msg.get("attachments", [])
                )
            else:
                r_str = msg.get("agent_role") or "Assistant"
                body = msg["text"]
            parts.append(f"{r_str}:\n{body}\n")

        final_p = self._compose_prompt(new_user_input, new_attachments)
        parts.append(f"User:\n{final_p}\n\nAssistant:")
        return "\n".join(parts)

    def on_send_clicked(self, _):
        typed_text = self.textarea.value.strip()
        attachments = list(self.attachments)
        if not typed_text and not attachments:
            self._set_status("⚠ Enter a prompt or attach a file first.", "#f59e0b")
            return

        self._toggle_controls(disabled=True)
        self.set_led_state(True)
        chosen_model = self.model_dropdown.value
        self._set_status(
            f"⏳ Querying <b>{html.escape(chosen_model)}</b>...", "#38bdf8"
        )

        try:
            context = self._build_context(typed_text, attachments)
            response_text = self._prompt_llm(
                chosen_model, context, temperature=self.temp_slider.value
            )
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self.chat_history.append(
                {
                    "role": "user",
                    "text": typed_text,
                    "attachments": attachments,
                    "time": ts,
                }
            )
            self.chat_history.append(
                {
                    "role": "model",
                    "text": response_text,
                    "model_name": chosen_model,
                    "time": ts,
                }
            )

            self.textarea.value = ""
            self.attachments = []
            self._refresh_attachment_view()
            self.refresh_chat_view()
            self.save_chat_to_disk()
            self._set_status(f"✓ Response received at {ts}.", "#10b981")
        except Exception as exc:
            self._set_status(f"✕ Error: {html.escape(str(exc))}", "#ef4444")
        finally:
            self.set_led_state(False)
            self._toggle_controls(disabled=False)

    def on_stop_agent_clicked(self, _):
        self._agent_stop_requested = True
        proc = getattr(self, "_active_proc", None)
        if proc is not None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._set_status(
            "⏹ Stop requested. Halting DyTopo Swarm cleanly...", "#ef4444"
        )

    def _parse_dytopo_decision(self, raw_text: str) -> dict:
        cleaned = raw_text.strip()
        fenced = re.findall(
            rf"{CODE_FENCE}(?:json)?[ \t]*\r?\n(.*?){CODE_FENCE}",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        candidates = list(fenced)
        fb, lb = cleaned.find("{"), cleaned.rfind("}")
        if fb != -1 and lb > fb:
            candidates.append(cleaned[fb : lb + 1])

        for cand in candidates:
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict) and "action" in parsed:
                    return parsed
            except Exception:
                continue

        return {
            "thought": "Direct synthesis.",
            "action": "FINISH",
            "channel": "Consensus Gate",
            "next_agent": "architect",
            "final_answer": raw_text,
        }

    def _append_agent_card(
        self,
        agent_role: str,
        model_name: str,
        text: str,
        badge: str = "",
        channel: str = "",
    ):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.chat_history.append(
            {
                "role": "model",
                "agent_role": agent_role,
                "agent_badge": badge,
                "channel": channel,
                "model_name": model_name,
                "text": text,
                "time": ts,
            }
        )
        self.refresh_chat_view()

    def on_dytopo_launch_clicked(self, _):
        goal_text = self.textarea.value.strip()
        attachments = list(self.attachments)
        if not goal_text and not attachments:
            self._set_status(
                "⚠ Enter a mission goal or attach files first.", "#f59e0b"
            )
            return

        self._agent_stop_requested = False
        self._toggle_controls(disabled=True)
        self.btn_stop_agent.disabled = False
        self.set_led_state(True)

        topology_mode = self.dytopo_mode_dropdown.value
        self.dytopo_graph["current_mode"] = topology_mode
        self._reset_node_statuses()

        max_steps = self.agent_max_steps_slider.value
        allow_exec = self.agent_auto_exec_checkbox.value
        temp = self.temp_slider.value

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        composed_goal = self._compose_prompt(goal_text, attachments)

        self.chat_history.append(
            {
                "role": "user",
                "text": f"🚀 [DyTopo Swarm Mission • Mode: {topology_mode.upper()}]\n{goal_text}",
                "attachments": attachments,
                "time": ts,
            }
        )
        self.textarea.value = ""
        self.attachments = []
        self._refresh_attachment_view()
        self.refresh_chat_view()

        import threading

        threading.Thread(
            target=self._run_swarm_worker,
            args=(
                goal_text,
                attachments,
                composed_goal,
                topology_mode,
                max_steps,
                allow_exec,
                temp,
            ),
            daemon=True,
        ).start()

    def _run_swarm_worker(self, goal_text, attachments, composed_goal, topology_mode, max_steps, allow_exec, temp):
        swarm_memory = [
            f"MISSION GOAL:\n{composed_goal}\n",
            f"TOPOLOGY MODE: {topology_mode.upper()}\n",
            f"WORKSPACE: {self.work_dir}\n",
        ]
        active_agent = "architect"

        try:
            for step in range(1, max_steps + 1):
                if self._agent_stop_requested:
                    self._append_agent_card(
                        agent_role="Lead Architect",
                        model_name=self.role_selectors["architect"].value,
                        badge=f"Stopped at Step {step}/{max_steps}",
                        text="⏹ **DyTopo Swarm stopped by operator.**",
                    )
                    break

                self._set_node_status(active_agent, "active")
                arch_model = self.role_selectors["architect"].value
                self._set_status(
                    f"🕸️ <b>Step {step}/{max_steps}:</b> Lead Architect routing topology...",
                    "#6366f1",
                )

                router_prompt = (
                    "You are the DyTopo Master Router and Lead Architect.\n"
                    f"TOPOLOGY MODE: {topology_mode.upper()}\n"
                    f"WORKSPACE FILES: {self._list_workspace_files()}\n"
                    f"STEP: {step} of {max_steps}\n\n"
                    "RECENT LOGS:\n"
                    + "\n---\n".join(swarm_memory[-5:])
                    + "\n\nRespond ONLY with valid JSON:\n"
                    "{\n"
                    '  "thought": "Reasoning",\n'
                    '  "channel": "Direct Message" | "Team Broadcast" | "Reflection Loop" | "Consensus Gate",\n'
                    '  "action": "DELEGATE" | "EXECUTE_CODE" | "FINISH",\n'
                    '  "next_agent": "coder" | "tester" | "critic" | "benchmarker" | "architect",\n'
                    '  "target_file": "filename.py",\n'
                    '  "instruction": "Task instructions",\n'
                    '  "final_answer": "Summary if action is FINISH"\n'
                    "}"
                )

                arch_raw = self._prompt_llm(
                    arch_model, router_prompt, temperature=temp
                )
                decision = self._parse_dytopo_decision(arch_raw)

                thought = decision.get("thought", "")
                action = str(decision.get("action", "FINISH")).upper().strip()
                channel = str(
                    decision.get("channel", "Direct Message")
                ).strip()
                next_agent = (
                    str(decision.get("next_agent", "coder")).lower().strip()
                )
                target_file = (
                    os.path.basename(
                        str(decision.get("target_file") or "").strip()
                    )
                    or "solution.py"
                )
                instruction = str(decision.get("instruction", ""))
                final_answer = str(decision.get("final_answer", ""))

                if next_agent not in ROLE_METADATA:
                    next_agent = "coder"

                self._record_edge_interaction("architect", next_agent)
                self._record_topology_transition(
                    step=f"{step}/{max_steps}",
                    source="Architect",
                    action=f"{action} ➔ {ROLE_METADATA[next_agent]['title']}",
                    channel=channel,
                )

                if action == "FINISH":
                    self._set_node_status("architect", "completed")
                    self._append_agent_card(
                        agent_role="🏁 Lead Architect (Goal Accomplished)",
                        model_name=arch_model,
                        badge=f"Step {step}/{max_steps} • FINISHED",
                        channel="Consensus Gate",
                        text=f"**Thought:** {thought}\n\n{final_answer or instruction or arch_raw}",
                    )
                    self.save_chat_to_disk()
                    self._set_status(
                        f"✓ DyTopo Swarm finished in {step} steps.", "#10b981"
                    )
                    return

                self._append_agent_card(
                    agent_role="🏛️ Lead Architect (Router)",
                    model_name=arch_model,
                    badge=f"Step {step}/{max_steps} • {action}",
                    channel=channel,
                    text=(
                        f"**Thought:** {thought}\n"
                        f"- **Edge:** `Architect` ➔ `{ROLE_METADATA[next_agent]['title']}` (`{target_file}`)\n"
                        f"- **Instruction:** {instruction}"
                    ),
                )

                if action == "EXECUTE_CODE" and allow_exec:
                    ok, exec_log = self.execute_file(target_file)
                    status_badge = "PASS ✓" if ok else "FAIL ✕"
                    self._append_agent_card(
                        agent_role="⚡ Sandbox Executor",
                        model_name="IPython Kernel",
                        badge=f"Step {step}/{max_steps} • {status_badge}",
                        channel="Direct Message",
                        text=f"Executed `{target_file}` ➔ **{status_badge}**\n\n{CODE_FENCE}text\n{exec_log}\n{CODE_FENCE}",
                    )
                    swarm_memory.append(
                        f"STEP {step} [EXEC {target_file} -> {status_badge}]:\n{exec_log}"
                    )
                    continue

                self._set_node_status(next_agent, "active")
                agent_model = self.role_selectors[next_agent].value
                r_info = ROLE_METADATA[next_agent]
                self._set_status(
                    f"{r_info['icon']} <b>Step {step}/{max_steps}:</b> {r_info['title']} working...",
                    r_info["color"],
                )

                agent_prompt = (
                    f"You are the {r_info['title']} in ArgonOS v9.\n"
                    f"MISSION GOAL:\n{composed_goal}\n\n"
                    f"ARCHITECT INSTRUCTION:\n{instruction}\n\n"
                    f"If writing code, start your fenced block with `%%writefile {target_file}`.\\n"
                    f"When modifying existing workspace files, NEVER regenerate the entire file with `%%writefile`.\\n"
                    f"Use `%%patch {target_file}` with localized `<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` blocks, including 2–3 lines of surrounding context to ensure uniqueness."
                )
                agent_resp = self._prompt_llm(
                    agent_model, agent_prompt, temperature=temp
                )
                self._append_agent_card(
                    agent_role=f"{r_info['icon']} {r_info['title']}",
                    model_name=agent_model,
                    badge=f"Step {step}/{max_steps} • {target_file}",
                    channel=channel,
                    text=agent_resp,
                )

                saved = self._extract_and_save_blocks(
                    agent_resp, default_filename=target_file
                )
                self.save_chat_to_disk()
                swarm_memory.append(
                    f"STEP {step} [{r_info['title']}]: Saved={saved}\n{agent_resp[-2000:]}"
                )
                self._set_node_status(next_agent, "completed")
                active_agent = next_agent

        except Exception as exc:
            self._set_status(
                f"✕ Swarm Error: {html.escape(str(exc))}", "#ef4444"
            )
        finally:
            self.btn_stop_agent.disabled = True
            self.set_led_state(False)
            self._reset_node_statuses()
            self._toggle_controls(disabled=False)

    # ==========================================
    # 6. SANDBOX CODE EXECUTION & FILE PERSISTENCE
    # ==========================================
    def _extract_code_blocks(self, text: str) -> list:
        return re.findall(
            rf"{CODE_FENCE}([a-zA-Z0-9_+\-]*)[ \t]*\r?\n(.*?){CODE_FENCE}",
            text,
            flags=re.DOTALL,
        )

    def apply_patch(self, filename: str, patch_text: str) -> tuple:
        import re as _re

        safe_name = os.path.basename(filename.strip())
        target_path = self.work_dir / safe_name
        if not target_path.is_file():
            return False, f"File '{safe_name}' not found in {self.work_dir}."

        original = target_path.read_text(encoding="utf-8", errors="replace")
        original_backup = original

        hunk_pattern = _re.compile(
            r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
            _re.DOTALL,
        )
        hunks = hunk_pattern.findall(patch_text)
        if not hunks:
            return False, "No valid SEARCH/REPLACE hunks found."

        def normalize(s: str) -> str:
            return "\n".join(line.rstrip() for line in s.replace("\r\n", "\n").split("\n"))

        content = original
        applied = 0
        added = 0
        removed = 0
        for idx, (search, replace) in enumerate(hunks, start=1):
            search_norm = normalize(search)
            content_norm = normalize(content)
            if search_norm in content_norm:
                # map back by doing string replace on normalized? We'll operate on normalized content then reconstruct? Simpler: replace exact substring in original content if present else normalized line-based.
                if search in content:
                    content = content.replace(search, replace, 1)
                else:
                    # fallback: line-based normalized replacement
                    search_lines = search_norm.split("\n")
                    content_lines = content.split("\n")
                    # find start
                    norm_content_lines = [l.rstrip() for l in content_lines]
                    norm_search_lines = [l.rstrip() for l in search_lines]
                    start = None
                    for i in range(len(norm_content_lines) - len(norm_search_lines) + 1):
                        if norm_content_lines[i : i + len(norm_search_lines)] == norm_search_lines:
                            start = i
                            break
                    if start is None:
                        target_path.write_text(original_backup, encoding="utf-8")
                        return False, f"Hunk #{idx} failed: SEARCH block not found in target"
                    end = start + len(norm_search_lines)
                    content_lines[start:end] = replace.split("\n")
                    content = "\n".join(content_lines)
                applied += 1
                removed += len(search.splitlines())
                added += len(replace.splitlines())
            else:
                target_path.write_text(original_backup, encoding="utf-8")
                return False, f"Hunk #{idx} failed: SEARCH block not found in target"

        target_path.write_text(content, encoding="utf-8")
        return True, f"Applied {applied} hunks to {safe_name} (+{added} lines, -{removed} lines)"

    def _extract_and_save_blocks(
        self, text: str, default_filename: str = None
    ) -> list:
        blocks = self._extract_code_blocks(text)
        saved = []
        for idx, (lang, block) in enumerate(blocks):
            lang_l = lang.lower().strip()
            lines = block.strip().splitlines()
            if lines and lines[0].strip().startswith("%%patch"):
                raw_fn = lines[0].replace("%%patch", "", 1).strip()
                safe_fn = os.path.basename(raw_fn) or default_filename
                patch_body = "\n".join(lines[1:])
                ok, report = self.apply_patch(safe_fn, patch_body)
                if ok:
                    saved.append(safe_fn)
                    if safe_fn not in self._saved_files:
                        self._saved_files.append(safe_fn)
                else:
                    if hasattr(self, "_set_status"):
                        self._set_status(f"✕ Patch failed: {report}", "#ef4444")
                continue
            if lines and lines[0].strip().startswith("%%writefile"):
                raw_fn = lines[0].replace("%%writefile", "", 1).strip()
                safe_fn = (
                    os.path.basename(raw_fn)
                    or default_filename
                    or f"extracted_{idx + 1}.{EXTENSION_MAP.get(lang_l, 'txt')}"
                )
                content = "\n".join(lines[1:])
            elif default_filename and idx == 0:
                safe_fn = os.path.basename(default_filename)
                content = block
            else:
                safe_fn = f"argon_artifact_{idx + 1}.{EXTENSION_MAP.get(lang_l, 'txt')}"
                content = block

            (self.work_dir / safe_fn).write_text(content, encoding="utf-8")
            saved.append(safe_fn)
            if safe_fn not in self._saved_files:
                self._saved_files.append(safe_fn)

        if saved:
            self.refresh_files_explorer()
        return saved

    def save_chat_to_disk(self):
        if not self.chat_history:
            return
        last_answer = ""
        for entry in reversed(self.chat_history):
            if entry["role"] == "model":
                last_answer = entry["text"]
                break

        payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "kernel": "ArgonOS v9",
            "active_model": self.model_dropdown.value,
            "dytopo_mode": self.dytopo_graph.get("current_mode"),
            "last_response": last_answer,
            "full_history": self.chat_history,
        }
        (self.work_dir / "argon_output.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._extract_and_save_blocks(last_answer)
        self.refresh_files_explorer()

    def execute_file(self, filename: str) -> tuple:
        """Executes a .py or .sh script in an isolated subprocess and captures output in Sandbox Console."""
        import subprocess
        import threading

        safe_name = os.path.basename(filename.strip())
        file_path = self.work_dir / safe_name
        if not file_path.exists():
            msg = f"File '{safe_name}' not found in {self.work_dir}."
            self._set_status(f"⚠ {html.escape(msg)}", "#ef4444")
            return False, msg

        self.set_led_state(True)
        self._set_status(
            f"⚡ Executing <b>{html.escape(safe_name)}</b> in Sandbox...",
            "#f59e0b",
        )

        timeout = 120
        try:
            timeout = int(getattr(self, "exec_timeout_seconds", 120))
        except Exception:
            timeout = 120

        env = dict(os.environ)
        # Preserve standard paths; sanitize by keeping only expected keys
        # (already a copy; ensure PYTHONUNBUFFERED for live output)
        env["PYTHONUNBUFFERED"] = "1"

        if safe_name.endswith(".sh"):
            cmd = ["/bin/bash", str(file_path)]
        else:
            cmd = [sys.executable, str(file_path)]

        ok = True
        log_str = ""
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.work_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,
            )
            self._active_proc = proc
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except Exception:
                    stdout, stderr = "", ""
                ok = False
                timeout_banner = (
                    f"⏱ Execution timed out after {timeout} seconds. "
                    f"Child process terminated.\n"
                )
                parts = [timeout_banner]
                if stdout:
                    parts.append(f"STDOUT (partial):\n{stdout.strip()}")
                if stderr:
                    parts.append(f"STDERR (partial):\n{stderr.strip()}")
                log_str = "\n\n".join(parts) if parts else timeout_banner
            else:
                if proc.returncode != 0:
                    ok = False
                parts = []
                if stdout:
                    parts.append(f"STDOUT:\n{stdout.strip()}")
                if stderr:
                    parts.append(f"STDERR / TRACEBACK:\n{stderr.strip()}")
                log_str = (
                    "\n\n".join(parts)
                    if parts
                    else "Executed cleanly with no stdout/stderr output."
                )
        except Exception as exc:
            ok = False
            log_str = f"Execution error: {exc}"
            traceback.print_exc()
        finally:
            self.set_led_state(False)

        header_banner = (
            f"┌─ ArgonOS v9 Sandbox Terminal: {safe_name}\n"
            f"└─ Path: {file_path} | Status: {'PASS ✓' if ok else 'FAIL ✕'}\n\n"
        )
        self.console_log_html.value = self._format_console_terminal(
            header_banner + log_str,
            status_color="#10b981" if ok else "#ef4444",
        )

        (self.work_dir / "last_console.log").write_text(log_str, encoding="utf-8")
        self._active_proc = None
        self._last_raw_exec_output = log_str
        self._last_exec_log = (
            f"Execution log for `{safe_name}`:\n{CODE_FENCE}text\n{log_str}\n{CODE_FENCE}"
        )

        if ok:
            self._set_status(
                f"✓ Execution of <b>{html.escape(safe_name)}</b> succeeded.",
                "#10b981",
            )
        else:
            self._set_status(
                f"✕ Execution failed in <b>{html.escape(safe_name)}</b>.",
                "#ef4444",
            )

        self.refresh_files_explorer()
        return ok, log_str

    def _on_feed_output_clicked(self, _):
        if not self._last_exec_log:
            self._set_status(
                "⚠ No console output available to send yet.", "#f59e0b"
            )
            return
        curr = self.textarea.value.strip()
        prefix = f"{curr}\n\n" if curr else ""
        self.textarea.value = (
            f"{prefix}{self._last_exec_log}\n\nPlease analyze this execution log and fix or optimize the code:"
        )
        self.focus_app_window("talk")
        self._set_status(
            "↩️ Console output appended to TalkToArgon prompt.", "#38bdf8"
        )

    # ==========================================
    # 7. ATTACHMENTS, CLIPBOARD & CHAT TIMELINE
    # ==========================================
    def _list_workspace_files(self) -> list:
        try:
            return [
                p.name
                for p in sorted(self.work_dir.iterdir())
                if p.is_file() and not p.name.startswith(".")
            ]
        except Exception:
            return []

    @staticmethod
    def _fence_for(content: str) -> str:
        longest = max((len(r) for r in re.findall(r"`+", content)), default=0)
        return "`" * max(3, longest + 1)

    def _compose_prompt(self, text: str, attachments: list) -> str:
        if not attachments:
            return text
        blocks = []
        for item in attachments:
            fence = self._fence_for(item["content"])
            blocks.append(
                f"Attached file: {item['name']}\n{fence}{item['lang']}\n{item['content']}\n{fence}"
            )
        return (
            "\n\n".join(blocks)
            + "\n\n"
            + (text or "Analyze the attached file(s).")
        )

    def on_attach_existing_clicked(self, _):
        fn = self.attach_select.value
        if not fn:
            return
        path = self.work_dir / fn
        if not path.is_file():
            return
        content = path.read_text(encoding="utf-8", errors="replace")[
            :ATTACH_MAX_BYTES
        ]
        self.attachments = [a for a in self.attachments if a["name"] != fn]
        self.attachments.append(
            {
                "name": fn,
                "content": content,
                "lang": TEXT_LANG_BY_EXT.get(path.suffix.lower(), "text"),
            }
        )
        self._refresh_attachment_view()
        self._set_status(f"📎 Attached <b>{html.escape(fn)}</b>.", "#10b981")

    def on_clear_attachments_clicked(self, _):
        self.attachments = []
        self._refresh_attachment_view()

    def _refresh_attachment_view(self):
        if not self.attachments:
            self.attachment_list.value = (
                "<div style='color:#94a3b8; font-size:11.5px;'>No files attached.</div>"
            )
            self.btn_clear_attachments.disabled = True
            return
        chips = "".join(
            f"<span class='kwb-chip'>📎 {html.escape(a['name'])}</span>"
            for a in self.attachments
        )
        self.attachment_list.value = f"<div>{chips}</div>"
        self.btn_clear_attachments.disabled = False

    def _copy_workspace_file(self, filename: str, button=None):
        path = self.work_dir / filename
        if not path.is_file():
            return
        self._copy_text_to_clipboard(
            path.read_text(encoding="utf-8", errors="replace"),
            label=filename,
            button=button,
        )

    def _copy_text_to_clipboard(
        self,
        text: str,
        label: str = "Content",
        button: widgets.Button = None,
    ):
        payload = json.dumps(text)
        js_code = f"""
        (function() {{
            const text = {payload};
            function fallbackCopy(val) {{
                const ta = document.createElement('textarea');
                ta.value = val;
                ta.style.position = 'fixed';
                ta.style.top = '-9999px';
                ta.style.left = '-9999px';
                document.body.appendChild(ta);
                ta.focus();
                ta.select();
                ta.setSelectionRange(0, val.length);
                try {{ document.execCommand('copy'); }} catch(e) {{}}
                document.body.removeChild(ta);
            }}
            if (navigator.clipboard && window.isSecureContext) {{
                navigator.clipboard.writeText(text).catch(() => fallbackCopy(text));
            }} else {{
                fallbackCopy(text);
            }}
        }})();
        """
        self._run_js(js_code)
        if button is not None:
            button.description = "✓ Copied!"
        self._set_status(
            f"📋 Copied <b>{html.escape(label)}</b> to clipboard.", "#10b981"
        )

    def _format_model_text(self, text: str) -> str:
        if HAS_MARKDOWN:
            return markdown.markdown(text, extensions=["fenced_code", "tables"])
        return f"<div style='white-space:pre-wrap; font-family:monospace;'>{html.escape(text)}</div>"

    def _build_copy_action_bar(self, raw_text: str) -> widgets.HTML:
        """Constructs an instant client-side copy bar that executes in-gesture on mobile touch."""
        items = []
        safe_main = html.escape(raw_text, quote=True)
        items.append(
            f'<button class="kwb-copy-widget-btn" onclick="window.argonDirectCopy(this)" data-text="{safe_main}">📋 Copy Response</button>'
        )

        for idx, (lang, code) in enumerate(self._extract_code_blocks(raw_text)):
            lbl = f"📋 Copy {lang or 'code'} #{idx + 1}"
            safe_code = html.escape(code.strip(), quote=True)
            items.append(
                f'<button class="kwb-copy-widget-btn" onclick="window.argonDirectCopy(this)" data-text="{safe_code}">{html.escape(lbl)}</button>'
            )

        return widgets.HTML(
            value=f'<div style="display:flex; flex-flow:row wrap; gap:4px; align-items:center;">{"".join(items)}</div>'
        )

    def _build_user_card_widget(
        self, entry: dict, turn_num: int = None
    ) -> widgets.Widget:
        t_badge = (
            f"<span class='kwb-turn-badge'>Turn #{turn_num}</span>"
            if turn_num
            else ""
        )
        chips = "".join(
            f"<span class='kwb-chip'>📎 {html.escape(a['name'])}</span>"
            for a in entry.get("attachments", [])
        )
        header = widgets.HBox(
            [
                widgets.HTML(
                    value=f"<div class='kwb-role'>{t_badge}Operator <span style='font-size:11px; color:#94a3b8;'>({html.escape(entry.get('time', ''))})</span></div>"
                ),
                self._build_copy_action_bar(entry["text"]),
            ],
            layout=widgets.Layout(
                width="100%",
                justify_content="space-between",
                align_items="center",
            ),
        )
        body = widgets.HTML(
            value=f"{chips}<div class='kwb-user-body'>{html.escape(entry['text'])}</div>"
        )
        card = widgets.VBox(
            [header, body],
            layout=widgets.Layout(width="100%", margin="6px 0"),
        )
        card.add_class("kwb-chat-card")
        card.add_class("kwb-user-card")
        return card

    def _build_model_card_widget(
        self, entry: dict, turn_num: int = None
    ) -> widgets.Widget:
        t_badge = (
            f"<span class='kwb-turn-badge'>Turn #{turn_num}</span>"
            if turn_num
            else ""
        )
        role_title = entry.get("agent_role") or entry.get(
            "model_name", "Model"
        )
        badge = entry.get("agent_badge")
        badge_html = (
            f"<span class='kwb-role-badge'>{html.escape(badge)}</span>"
            if badge
            else ""
        )

        top_bar = self._build_copy_action_bar(entry["text"])
        bottom_bar = self._build_copy_action_bar(entry["text"])
        bottom_bar.add_class("kwb-card-footer")

        header = widgets.HBox(
            [
                widgets.HTML(
                    value=f"<div class='kwb-role'>{t_badge}{html.escape(role_title)}{badge_html} <span style='font-size:11px; color:#94a3b8;'>({html.escape(entry.get('time', ''))})</span></div>"
                ),
                top_bar,
            ],
            layout=widgets.Layout(
                width="100%",
                justify_content="space-between",
                align_items="center",
            ),
        )
        body = widgets.HTML(
            value=f"<div class='kwb-model-body'>{self._format_model_text(entry['text'])}</div>"
        )
        card = widgets.VBox(
            [header, body, bottom_bar],
            layout=widgets.Layout(width="100%", margin="6px 0"),
        )
        card.add_class("kwb-chat-card")
        card.add_class(
            "kwb-agent-card" if entry.get("agent_role") else "kwb-model-card"
        )
        return card

    def refresh_chat_view(self):
        if not self.chat_history:
            self.chat_container.children = [self.chat_empty_html]
            return

        turns, current = [], []
        for entry in self.chat_history:
            if entry["role"] == "user" and current:
                turns.append(current)
                current = [entry]
            else:
                current.append(entry)
        if current:
            turns.append(current)

        cards = []
        if self.sort_order_toggle.value == "newest":
            for t_idx in range(len(turns), 0, -1):
                items = turns[t_idx - 1]
                for e in reversed([m for m in items if m["role"] != "user"]):
                    cards.append(self._build_model_card_widget(e, t_idx))
                for e in [u for u in items if u["role"] == "user"]:
                    cards.append(self._build_user_card_widget(e, t_idx))
        else:
            for t_idx, items in enumerate(turns, start=1):
                for e in items:
                    cards.append(
                        self._build_user_card_widget(e, t_idx)
                        if e["role"] == "user"
                        else self._build_model_card_widget(e, t_idx)
                    )
        self.chat_container.children = cards

    def _get_empty_chat_html(self) -> str:
        return """
        <div style="padding:28px 16px; text-align:center; color:#94a3b8;">
            <div style="font-size:26px; margin-bottom:6px;">🤖 🕸️</div>
            <div style="font-size:13.5px; font-weight:700; color:#e2e8f0;">TalkToArgon & DyTopo Swarm Studio Ready</div>
            <div style="font-size:11.5px; margin-top:4px;">Newest responses appear at the top with header & footer copy bars.</div>
        </div>
        """

    def _set_status(self, html_msg: str, color: str = "#38bdf8"):
        self.status_label.value = (
            f"<div class='kwb-status' style='color:{color}; border-color:{color}55;'>{html_msg}</div>"
        )

    def _toggle_controls(self, disabled: bool):
        has_turns = len(self.chat_history) >= 2
        self.btn_send.disabled = disabled
        self.btn_superagent.disabled = disabled
        self.btn_save.disabled = disabled
        self.btn_reset.disabled = disabled
        self.btn_undo.disabled = disabled or not has_turns
        self.btn_copy_last.disabled = disabled or not has_turns

    def on_undo_clicked(self, _):
        while self.chat_history and self.chat_history[-1]["role"] == "model":
            self.chat_history.pop()
        if self.chat_history and self.chat_history[-1]["role"] == "user":
            last_u = self.chat_history.pop()
            self.textarea.value = last_u["text"]
        self.refresh_chat_view()

    def on_copy_last_clicked(self, button):
        for entry in reversed(self.chat_history):
            if entry["role"] == "model":
                self._copy_text_to_clipboard(
                    entry["text"], label="Latest Response", button=button
                )
                return

    def _on_copy_console_clicked(self, button):
        if self._last_raw_exec_output:
            self._copy_text_to_clipboard(
                self._last_raw_exec_output,
                label="Console Output",
                button=button,
            )

    def on_save_clicked(self, _):
        self.save_chat_to_disk()
        self._set_status(
            "💾 Exported conversation & code artifacts.", "#10b981"
        )

    def on_reset_clicked(self, _):
        self.chat_history.clear()
        self.attachments.clear()
        self.dytopo_graph["edges"].clear()
        self.dytopo_graph["transitions"].clear()
        self._reset_node_statuses()
        self._refresh_attachment_view()
        self.refresh_chat_view()
        self._set_status("🧹 Session and DyTopo graph cleared.", "#94a3b8")

    # ==========================================
    # 8. DESKTOP VIEWPORT & PORTAL RENDER ENGINE
    # ==========================================
    def render(self):
        """Renders the ArgonOS v9 Multi-Window Touch Desktop and mounts live ipywidgets into windows."""
        telemetry = self._get_system_telemetry()
        windows_html = []
        dock_buttons_html = []
        widget_staging_boxes = []

        for idx, (app_id, app) in enumerate(self.apps.items()):
            win_id = f"win_{app_id}_{self.uid}"
            portal_target_id = f"portal_target_{app_id}_{self.uid}"
            staging_class = f"argon-staging-{app_id}-{self.uid}"

            is_vis = app.visible
            disp_style = "flex" if is_vis else "none"
            active_cls = "active-win" if idx == 0 else ""
            max_cls = "is-maximized" if app.maximized else ""
            dock_active = "active" if is_vis else ""

            windows_html.append(
                f"""
                <div class="argon-win-{self.uid} {active_cls} {max_cls}" id="{win_id}"
                     style="top:{app.top}px; left:{app.left}px; width:{app.width}; height:{app.height}; display:{disp_style}; z-index:{120 - idx * 10};"
                     onpointerdown="window.argonBringToFront(this, '{self.uid}')">

                    <div class="win-header-{self.uid}" style="background:{app.header_color};"
                         onpointerdown="window.argonStartDrag(event, this)">
                        <span style="display:flex; align-items:center; gap:6px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                            {app.icon} {html.escape(app.title)}
                        </span>
                        <div style="display:flex; align-items:center; flex-shrink:0;">
                            <button class="win-btn-{self.uid}" title="Maximize / Restore" onclick="window.argonMaximizeWin(this)">□</button>
                            <button class="win-btn-{self.uid}" title="Close Window" style="background:#ef4444;"
                                    onclick="window.argonCloseWin('{win_id}', 'dock_{app_id}_{self.uid}')">✕</button>
                        </div>
                    </div>

                    <div class="win-body-{self.uid}" id="{portal_target_id}">
                        <div class="argon-static-fallback">{app.content_html}</div>
                    </div>
                </div>
                """
            )

            dock_buttons_html.append(
                f"""
                <button class="app-icon-btn-{self.uid} {dock_active}"
                        id="dock_{app_id}_{self.uid}"
                        onclick="window.argonToggleApp('{win_id}', this, '{self.uid}')"
                        title="{html.escape(app.title)}">
                    <span>{app.icon}</span>
                    <span class="dock-label-{self.uid}">{html.escape(app.app_id.upper())}</span>
                </button>
                """
            )

            if app.widget is not None:
                wrapper = widgets.Box(
                    [app.widget], layout=widgets.Layout(width="100%")
                )
                wrapper.add_class(staging_class)
                widget_staging_boxes.append(wrapper)

        desktop_html = f"""
        <style>
            .argon-desktop-{self.uid} {{
                position: relative;
                width: 100%;
                height: 680px;
                background: #060913;
                border-radius: 18px;
                overflow: hidden;
                user-select: none;
                -webkit-user-select: none;
                font-family: Inter, ui-sans-serif, system-ui, -apple-system, monospace;
                border: 2px solid #334155;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.65);
            }}

            .argon-desktop-{self.uid}.is-fullscreen {{
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                width: 100vw !important;
                height: 100vh !important;
                z-index: 2147483647 !important;
                border-radius: 0 !important;
                border: 0 !important;
            }}

            /* GPU-ACCELERATED RGB LED EDGE LOADING STRIP */
            .rgb-strip-{self.uid} {{
                position: absolute;
                inset: -100%;
                width: 300%;
                height: 300%;
                background: conic-gradient(from 0deg, #ff0055, #7928ca, #0070f3, #00dfd8, #00ff66, #ffea00, #ff0055);
                animation: rgbSpin-{self.uid} 2.4s linear infinite;
                opacity: 0;
                visibility: hidden;
                transition: opacity 0.3s ease;
                z-index: 1;
                will-change: transform, opacity;
            }}
            .argon-desktop-{self.uid}.rgb-active .rgb-strip-{self.uid} {{
                opacity: 1;
                visibility: visible;
            }}
            @keyframes rgbSpin-{self.uid} {{
                to {{ transform: rotate(360deg); }}
            }}

            .desk-inner-{self.uid} {{
                position: absolute;
                inset: 4px;
                background:
                    radial-gradient(circle at 82% 14%, rgba(99, 102, 241, 0.16), transparent 36%),
                    radial-gradient(circle at 18% 80%, rgba(14, 165, 233, 0.14), transparent 40%),
                    linear-gradient(180deg, #0b1120 0%, #060913 100%);
                border-radius: 14px;
                z-index: 2;
                overflow: hidden;
            }}
            .argon-desktop-{self.uid}.is-fullscreen .desk-inner-{self.uid} {{
                border-radius: 0;
                inset: 3px;
            }}

            /* FLOATING MULTI-WINDOWS */
            .argon-win-{self.uid} {{
                position: absolute;
                background: rgba(15, 23, 42, 0.96);
                border: 2px solid #475569;
                border-radius: 14px;
                box-shadow: 0 18px 42px rgba(0, 0, 0, 0.75);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                max-width: calc(100% - 16px);
                max-height: calc(100% - 68px);
                backdrop-filter: blur(12px);
                transition: border-color 0.15s ease, box-shadow 0.15s ease;
            }}
            .argon-win-{self.uid}.active-win {{
                border-color: #38bdf8 !important;
                box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.4), 0 20px 48px rgba(14, 165, 233, 0.32);
            }}
            .argon-win-{self.uid}.is-maximized {{
                top: 6px !important;
                left: 6px !important;
                width: calc(100% - 12px) !important;
                height: calc(100% - 64px) !important;
                max-width: none !important;
                max-height: none !important;
                border-radius: 12px !important;
            }}

            /* 100% TOUCH-SAFE POINTER DRAG HANDLE */
            .win-header-{self.uid} {{
                height: 42px;
                min-height: 42px;
                padding: 0 12px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                color: #ffffff;
                font-size: 12.5px;
                font-weight: 700;
                cursor: grab;
                touch-action: none;
                user-select: none;
            }}
            .win-header-{self.uid}:active {{
                cursor: grabbing;
            }}

            .win-btn-{self.uid} {{
                width: 30px;
                height: 30px;
                border-radius: 7px;
                background: rgba(255, 255, 255, 0.18);
                color: #ffffff;
                font-weight: bold;
                border: 0;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                margin-left: 5px;
                cursor: pointer;
                touch-action: manipulation;
            }}
            .win-btn-{self.uid}:hover {{
                background: rgba(255, 255, 255, 0.3);
            }}

            .win-body-{self.uid} {{
                flex: 1 1 auto;
                overflow-y: auto;
                overflow-x: hidden;
                padding: 6px;
                background: #0f172a;
                color: #f1f5f9;
                user-select: text;
                -webkit-user-select: text;
                touch-action: pan-y;
            }}

            /* THREE-SECTION BOTTOM TASKBAR */
            .taskbar-{self.uid} {{
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 54px;
                background: rgba(9, 13, 24, 0.98);
                border-top: 1px solid #334155;
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 12px;
                z-index: 999999;
                gap: 8px;
            }}

            .taskbar-telemetry-{self.uid} {{
                display: flex;
                align-items: center;
                gap: 6px;
                flex-wrap: wrap;
            }}

            .telemetry-chip-{self.uid} {{
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 7px;
                padding: 4px 8px;
                font-size: 10.5px;
                font-weight: 700;
                color: #10b981;
                white-space: nowrap;
            }}

            .app-icon-btn-{self.uid} {{
                height: 38px;
                padding: 0 10px;
                border-radius: 9px;
                background: #1e293b;
                border: 1px solid #475569;
                color: #f8fafc;
                font-size: 13px;
                font-weight: 700;
                display: inline-flex;
                align-items: center;
                gap: 5px;
                margin-left: 4px;
                cursor: pointer;
                touch-action: manipulation;
                transition: all 0.15s ease;
            }}
            .app-icon-btn-{self.uid}.active {{
                border-color: #38bdf8;
                background: linear-gradient(135deg, #1e1b4b, #1d4ed8);
                box-shadow: 0 0 10px rgba(56, 189, 248, 0.35);
            }}
            .dock-label-{self.uid} {{
                font-size: 10px;
                letter-spacing: 0.04em;
            }}

            /* WORKBENCH & EDITOR DARK THEME STYLING */
            .kwb-prompt textarea, .kwb-system-prompt textarea, .kwb-editor-textarea textarea {{
                background: #090d16 !important;
                color: #f1f5f9 !important;
                border: 1px solid #334155 !important;
                border-radius: 10px !important;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
                font-size: 12.5px !important;
                line-height: 1.55 !important;
                padding: 10px !important;
            }}
            .kwb-status {{
                display: inline-flex;
                align-items: center;
                padding: 4px 10px;
                border-radius: 999px;
                border: 1px solid #334155;
                background: #090d16;
                font-size: 11.5px;
                font-weight: 600;
            }}
            .kwb-chat-shell {{
                background: #090d16;
                border: 1px solid #1e293b;
                border-radius: 12px;
            }}
            .kwb-chat-card {{
                padding: 12px 14px;
                border-radius: 10px;
                border: 1px solid #334155;
                background: #1e293b;
            }}
            .kwb-user-card {{
                border-color: #1d4ed8;
                background: #172554;
            }}
            .kwb-agent-card {{
                border-color: #6366f1;
                background: #1e1b4b;
            }}
            .kwb-role {{
                color: #f8fafc;
                font-size: 13px;
                font-weight: 700;
            }}
            .kwb-turn-badge {{
                background: #334155;
                color: #38bdf8;
                padding: 2px 6px;
                border-radius: 5px;
                font-size: 10px;
                margin-right: 6px;
            }}
            .kwb-role-badge {{
                background: #312e81;
                color: #c7d2fe;
                padding: 2px 7px;
                border-radius: 999px;
                font-size: 10px;
                margin-left: 6px;
            }}
            .kwb-chip {{
                display: inline-block;
                background: #1e3a8a;
                color: #93c5fd;
                padding: 2px 8px;
                border-radius: 999px;
                font-size: 11px;
                margin-right: 4px;
            }}
            .kwb-user-body, .kwb-model-body {{
                color: #e2e8f0;
                font-size: 13px;
                line-height: 1.6;
                overflow-wrap: anywhere;
            }}
            .kwb-model-body pre {{
                background: #090d16;
                border: 1px solid #334155;
                padding: 10px;
                border-radius: 8px;
                overflow-x: auto;
                color: #38bdf8;
            }}
            .kwb-copy-widget-btn {{
                background: #1e1b4b !important;
                color: #c7d2fe !important;
                border: 1px solid #4f46e5 !important;
                font-size: 11px !important;
                font-weight: 700 !important;
                border-radius: 7px;
                padding: 4px 8px;
                cursor: pointer;
                touch-action: manipulation;
                margin: 2px;
            }}
            .kwb-copy-widget-btn:hover {{
                background: #312e81 !important;
            }}
        </style>

        <div class="argon-desktop-{self.uid}" id="desk_{self.uid}">
            <!-- GPU-ACCELERATED RGB LED PERIMETER LOADING STRIP -->
            <div class="rgb-strip-{self.uid}" id="rgb_strip_{self.uid}"></div>

            <div class="desk-inner-{self.uid}">
                <!-- REGISTERED MULTI-WINDOW APPLICATIONS -->
                {''.join(windows_html)}

                <!-- THREE-SECTION BOTTOM SYSTEM TASKBAR -->
                <div class="taskbar-{self.uid}">
                    <!-- LEFT: SYSTEM TELEMETRY SCREEN -->
                    <div class="taskbar-telemetry-{self.uid}">
                        <span class="telemetry-chip-{self.uid}">🟢 ARGON-OS v9</span>
                        <span class="telemetry-chip-{self.uid}" style="color:#38bdf8;">🧠 {html.escape(telemetry['ram'])}</span>
                        <span class="telemetry-chip-{self.uid}" style="color:#f59e0b;">💾 {html.escape(telemetry['disk'])}</span>
                    </div>

                    <!-- CENTER: HARDWARE FULLSCREEN BUTTON -->
                    <div>
                        <button onclick="window.argonToggleFS(this)"
                                style="background:linear-gradient(135deg, #f59e0b, #d97706); color:#ffffff; border:0; border-radius:9px; padding:8px 14px; font-weight:800; font-size:11.5px; cursor:pointer; box-shadow:0 4px 12px rgba(245, 158, 11, 0.35);">
                            ⛶ Fullscreen
                        </button>
                    </div>

                    <!-- RIGHT: DYNAMIC APP LAUNCHER DOCK -->
                    <div style="display:flex; align-items:center; overflow-x:auto;">
                        {''.join(dock_buttons_html)}
                    </div>
                </div>
            </div>
        </div>
        """

        js_kernel = f"""
        (function() {{
            let topZ = 200;

            function fallbackMobileCopy(text, btn) {{
                const ta = document.createElement("textarea");
                ta.value = text;
                ta.style.position = "fixed";
                ta.style.top = "-9999px";
                ta.style.left = "-9999px";
                document.body.appendChild(ta);
                ta.focus();
                ta.select();
                ta.setSelectionRange(0, text.length);
                try {{
                    document.execCommand("copy");
                }} catch(e) {{}}
                document.body.removeChild(ta);
                if (btn) {{
                    const oldText = btn.innerText;
                    btn.innerText = "✓ Copied!";
                    setTimeout(() => {{ btn.innerText = oldText; }}, 1800);
                }}
            }}

            window.argonDirectCopy = function(btn) {{
                const text = btn.getAttribute("data-text") || "";
                if (navigator.clipboard && window.isSecureContext) {{
                    navigator.clipboard.writeText(text).then(() => {{
                        const oldText = btn.innerText;
                        btn.innerText = "✓ Copied!";
                        setTimeout(() => {{ btn.innerText = oldText; }}, 1800);
                    }}).catch(() => fallbackMobileCopy(text, btn));
                }} else {{
                    fallbackMobileCopy(text, btn);
                }}
            }};

            window.argonBringToFront = function(win, uid) {{
                if (!win) return;
                topZ += 1;
                win.style.zIndex = topZ;
                const root = uid ? document.getElementById("desk_" + uid) : document;
                if (root) {{
                    root.querySelectorAll("[class*='argon-win-']").forEach(w => w.classList.remove("active-win"));
                }}
                win.classList.add("active-win");
            }};

            window.argonMaximizeWin = function(btn) {{
                const win = btn.closest("[class*='argon-win-']");
                if (!win) return;

                if (win.classList.contains("is-maximized")) {{
                    win.classList.remove("is-maximized");
                    btn.innerText = "□";
                }} else {{
                    win.classList.add("is-maximized");
                    btn.innerText = "🗗";
                }}
                window.argonBringToFront(win);
            }};

            window.argonCloseWin = function(winId, dockId) {{
                const win = document.getElementById(winId);
                if (win) win.style.display = "none";
                const dock = document.getElementById(dockId);
                if (dock) dock.classList.remove("active");
            }};

            window.argonToggleApp = function(winId, btn, uid) {{
                const win = document.getElementById(winId);
                if (!win) return;

                if (win.style.display === "none") {{
                    win.style.display = "flex";
                    window.argonBringToFront(win, uid);
                    if (btn) btn.classList.add("active");
                }} else {{
                    if (win.classList.contains("active-win")) {{
                        win.style.display = "none";
                        if (btn) btn.classList.remove("active");
                    }} else {{
                        window.argonBringToFront(win, uid);
                        if (btn) btn.classList.add("active");
                    }}
                }}
            }};

            window.argonOpenAppById = function(appId, uid) {{
                const win = document.getElementById("win_" + appId + "_" + uid);
                const btn = document.getElementById("dock_" + appId + "_" + uid);
                if (!win) return;
                win.style.display = "flex";
                window.argonBringToFront(win, uid);
                if (btn) btn.classList.add("active");
            }};

            window.argonStartDrag = function(e, header) {{
                if (e.target.tagName === "BUTTON" || e.target.closest("button")) return;
                const win = header.closest("[class*='argon-win-']");
                const desktop = header.closest("[class*='desk-inner-']");
                if (!win || !desktop || win.classList.contains("is-maximized")) return;

                window.argonBringToFront(win);

                const startX = e.clientX;
                const startY = e.clientY;
                const initLeft = win.offsetLeft;
                const initTop = win.offsetTop;

                try {{ header.setPointerCapture(e.pointerId); }} catch (err) {{}}

                function onMove(ev) {{
                    const dx = ev.clientX - startX;
                    const dy = ev.clientY - startY;
                    const maxL = Math.max(4, desktop.clientWidth - Math.min(win.clientWidth, 180));
                    const maxT = Math.max(4, desktop.clientHeight - 96);
                    const newL = Math.max(4, Math.min(initLeft + dx, maxL));
                    const newT = Math.max(4, Math.min(initTop + dy, maxT));
                    win.style.left = newL + "px";
                    win.style.top = newT + "px";
                    ev.preventDefault();
                }}

                function onUp(ev) {{
                    header.removeEventListener("pointermove", onMove);
                    header.removeEventListener("pointerup", onUp);
                    header.removeEventListener("pointercancel", onUp);
                    try {{ header.releasePointerCapture(ev.pointerId); }} catch (err) {{}}
                }}

                header.addEventListener("pointermove", onMove);
                header.addEventListener("pointerup", onUp);
                header.addEventListener("pointercancel", onUp);
                e.preventDefault();
            }};

            window.argonToggleFS = function(btn) {{
                const desktop = btn.closest("[class*='argon-desktop-']");
                if (!desktop) return;

                if (!desktop.classList.contains("is-fullscreen")) {{
                    desktop.classList.add("is-fullscreen");
                    btn.innerText = "🗗 Exit";
                    btn.style.background = "#ef4444";
                    if (desktop.requestFullscreen) desktop.requestFullscreen().catch(() => {{}});
                }} else {{
                    desktop.classList.remove("is-fullscreen");
                    btn.innerText = "⛶ Fullscreen";
                    btn.style.background = "linear-gradient(135deg, #f59e0b, #d97706)";
                    if (document.fullscreenElement && document.exitFullscreen) document.exitFullscreen().catch(() => {{}});
                }}
            }};

            window.argonSetLED = function(uid, active) {{
                const desktop = document.getElementById("desk_" + uid);
                if (!desktop) return;
                if (active) {{
                    desktop.classList.add("rgb-active");
                }} else {{
                    desktop.classList.remove("rgb-active");
                }}
            }};

            window.argonToggleLED = function(btn, uid) {{
                const desktop = document.getElementById("desk_" + uid);
                if (!desktop) return;
                const nextState = !desktop.classList.contains("rgb-active");
                window.argonSetLED(uid, nextState);
                if (btn) {{
                    btn.innerText = nextState ? "⚡ LED FX: ON" : "⚡ LED FX: OFF";
                }}
            }};

            // Mount live ipywidgets into their corresponding ArgonOS floating window portals
            const appIds = {json.dumps(list(self.apps.keys()))};
            const uid = "{self.uid}";
            let attempts = 0;
            const portalTimer = setInterval(function() {{
                attempts += 1;
                let mountedCount = 0;
                appIds.forEach(function(appId) {{
                    const target = document.getElementById("portal_target_" + appId + "_" + uid);
                    const staging = document.querySelector(".argon-staging-" + appId + "-" + uid);
                    if (target && staging) {{
                        const fallback = target.querySelector(".argon-static-fallback");
                        if (fallback) fallback.style.display = "none";
                        if (staging.parentElement !== target) {{
                            target.appendChild(staging);
                        }}
                        mountedCount += 1;
                    }}
                }});
                if (mountedCount === appIds.length || attempts > 40) {{
                    clearInterval(portalTimer);
                }}
            }}, 120);
        }})();
        """

        staging_container = widgets.VBox(
            widget_staging_boxes + [self._js_bridge],
            layout=widgets.Layout(width="100%"),
        )

        display(HTML(desktop_html))
        display(staging_container)
        display(Javascript(js_kernel))

    def verify_kernel_integrity(self) -> bool:
        """Runs a comprehensive boot & functional self-test on ArgonOS v9."""
        required_apps = {"talk", "files", "editor", "console", "sysinfo"}
        mounted_apps = set(self.apps.keys())
        missing = required_apps - mounted_apps
        if missing:
            raise RuntimeError(f"ArgonOS v9 missing core apps: {missing}")

        # Verify LED strip toggle API
        self.set_led_state(True)
        assert self.led_active is True
        self.set_led_state(False)
        assert self.led_active is False

        # Verify Editor & Files Explorer roundtrip
        self.refresh_files_explorer()
        self._show_file_info("argon_code_v91.py")

        # Verify file upload widgets & extraction helper
        assert hasattr(self, "file_upload")
        assert hasattr(self, "files_upload_widget")
        dummy_v8 = ({"name": "test_v8.txt", "content": b"v8 hello"},)
        extracted_v8 = self._extract_uploaded_files(dummy_v8)
        assert (
            len(extracted_v8) == 1
            and extracted_v8[0]["name"] == "test_v8.txt"
            and extracted_v8[0]["content"] == b"v8 hello"
        )
        dummy_v7 = {"test_v7.py": {"content": b"print('v7 hello')"}}
        extracted_v7 = self._extract_uploaded_files(dummy_v7)
        assert (
            len(extracted_v7) == 1
            and extracted_v7[0]["name"] == "test_v7.py"
            and extracted_v7[0]["content"] == b"print('v7 hello')"
        )

        sys.stdout.write(
            f"ARGON_OS_V9_BOOT: PASS | Registered Apps: {sorted(mounted_apps)} | "
            f"Default LLM: {self.default_model} | Workspace Files: {len(self._list_workspace_files())}\n"
        )
        sys.stdout.flush()
        return True


# ==========================================
# 9. BOOTSTRAP ARGON-OS v9 MICRO-KERNEL
# ==========================================
if __name__ == "__main__":
    os_kernel = ArgonOS()
    os_kernel.render()
    os_kernel.verify_kernel_integrity()
