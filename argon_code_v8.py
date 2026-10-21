import datetime
import html
import json
import math
import os
from pathlib import Path
import re
import shlex
import traceback

from IPython import get_ipython
from IPython.display import Javascript, display
from IPython.utils.capture import capture_output
import ipywidgets as widgets
import kaggle_benchmarks as kbench


# Optional Markdown support.
try:
    import markdown

    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False


# Builds a markdown code fence programmatically so the source
# contains no literal triple backticks.
CODE_FENCE = "`" * 3

# Attachment limits and language mapping.
ATTACH_MAX_BYTES = 200_000
ATTACH_MAX_FILES = 10
TEXT_LANG_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".html": "html",
    ".css": "css",
    ".sh": "bash",
    ".sql": "sql",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".cpp": "cpp",
    ".c": "c",
    ".java": "java",
    ".csv": "text",
    ".txt": "text",
}

EXTENSION_MAP = {
    "python": "py",
    "py": "py",
    "javascript": "js",
    "js": "js",
    "typescript": "ts",
    "ts": "ts",
    "html": "html",
    "css": "css",
    "bash": "sh",
    "shell": "sh",
    "sh": "sh",
    "sql": "sql",
    "json": "json",
    "yaml": "yaml",
    "yml": "yml",
    "cpp": "cpp",
    "c++": "cpp",
    "c": "c",
    "java": "java",
    "text": "txt",
    "txt": "txt",
}

# ==========================================
# DYTOPO ROLE & TOPOLOGY CONSTANTS
# ==========================================
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


class KaggleLLMWorkbench:
    def __init__(self, work_dir="/kaggle/working"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.chat_history = []
        self.attachments = []
        self._last_exec_log = ""
        self._last_raw_exec_output = ""
        self._saved_files = []
        self._agent_stop_requested = False

        # DyTopo interaction graph state
        self.dytopo_graph = {
            "nodes": {
                role_id: {"status": "idle", "messages_handled": 0}
                for role_id in ROLE_METADATA
            },
            "edges": {},  # (src, dst) -> count
            "transitions": [],
            "current_mode": "adaptive",
        }

        self.available_models = list(kbench.llms.keys())
        if not self.available_models:
            raise RuntimeError(
                "No language models were found in kaggle_benchmarks.llms."
            )

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

        self._init_ui()
        self._bind_events()

    def _pick_preferred_model(self, candidates: list) -> str:
        for candidate in candidates:
            if candidate in self.available_models:
                return candidate
        return self.available_models[0]

    # ==========================================
    # 1. UI INITIALIZATION & CSS
    # ==========================================
    def _init_ui(self):
        self._js_sink = widgets.Output(
            layout=widgets.Layout(
                width="0px",
                height="0px",
                overflow="hidden",
                margin="0",
                padding="0",
            )
        )

        self.theme = widgets.HTML(
            value="""
            <style>
                :root {
                    --kwb-surface: #ffffff;
                    --kwb-surface-soft: #f8fafc;
                    --kwb-border: #dce4ee;
                    --kwb-border-strong: #cbd5e1;
                    --kwb-text: #172033;
                    --kwb-muted: #64748b;
                    --kwb-primary: #4f46e5;
                    --kwb-code-bg: #0f172a;
                    --kwb-radius: 14px;
                    --kwb-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
                }

                .kwb-root {
                    width: 100%;
                    box-sizing: border-box;
                    color: var(--kwb-text);
                    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                }

                .kwb-shell { gap: 14px; }

                .kwb-header {
                    box-sizing: border-box;
                    position: relative;
                    overflow: hidden;
                    padding: 24px 26px;
                    border-radius: 18px;
                    background:
                        radial-gradient(circle at 88% 8%, rgba(255, 255, 255, 0.18), transparent 32%),
                        linear-gradient(135deg, #1e1b4b 0%, #4338ca 48%, #6366f1 100%);
                    box-shadow: 0 16px 38px rgba(30, 27, 75, 0.25);
                    color: #ffffff;
                }

                .kwb-header-row {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 18px;
                    flex-wrap: wrap;
                }

                .kwb-brand {
                    display: flex;
                    align-items: center;
                    gap: 14px;
                }

                .kwb-logo {
                    display: grid;
                    place-items: center;
                    width: 52px;
                    height: 52px;
                    flex: 0 0 52px;
                    border: 1px solid rgba(255, 255, 255, 0.28);
                    border-radius: 14px;
                    background: rgba(255, 255, 255, 0.14);
                    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2), 0 8px 22px rgba(15, 23, 42, 0.25);
                    font-size: 26px;
                }

                .kwb-title {
                    font-size: 22px;
                    line-height: 1.25;
                    font-weight: 780;
                    letter-spacing: -0.02em;
                }

                .kwb-subtitle {
                    max-width: 740px;
                    margin-top: 5px;
                    color: rgba(255, 255, 255, 0.88);
                    font-size: 12.5px;
                    line-height: 1.5;
                }

                .kwb-header-badges {
                    display: flex;
                    align-items: center;
                    gap: 7px;
                    flex-wrap: wrap;
                }

                .kwb-header-badge {
                    padding: 7px 12px;
                    border: 1px solid rgba(255, 255, 255, 0.24);
                    border-radius: 999px;
                    background: rgba(255, 255, 255, 0.14);
                    color: #ffffff;
                    font-size: 11.5px;
                    font-weight: 650;
                    backdrop-filter: blur(8px);
                }

                .kwb-panel {
                    box-sizing: border-box;
                    padding: 16px;
                    border: 1px solid var(--kwb-border);
                    border-radius: var(--kwb-radius);
                    background: var(--kwb-surface);
                    box-shadow: var(--kwb-shadow);
                }

                .kwb-section-title {
                    color: var(--kwb-text);
                    font-size: 15px;
                    font-weight: 750;
                }

                .kwb-section-subtitle {
                    margin-top: 3px;
                    color: var(--kwb-muted);
                    font-size: 12px;
                    line-height: 1.45;
                }

                .kwb-toolbar {
                    align-items: center;
                    flex-wrap: wrap;
                    gap: 8px;
                }

                .kwb-toolbar-label {
                    color: var(--kwb-muted);
                    font-size: 11px;
                    font-weight: 750;
                    letter-spacing: 0.08em;
                    text-transform: uppercase;
                }

                .kwb-root select,
                .kwb-root textarea,
                .kwb-root input[type="text"],
                .kwb-root input[type="number"] {
                    box-sizing: border-box;
                    border: 1px solid var(--kwb-border-strong) !important;
                    border-radius: 10px !important;
                    background: var(--kwb-surface) !important;
                    color: var(--kwb-text) !important;
                    outline: none;
                    transition: border-color 150ms ease, box-shadow 150ms ease;
                }

                .kwb-root select:focus,
                .kwb-root textarea:focus,
                .kwb-root input:focus {
                    border-color: var(--kwb-primary) !important;
                    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.16) !important;
                }

                .kwb-prompt textarea {
                    min-height: 165px;
                    padding: 15px 16px !important;
                    border-color: #cfd8e5 !important;
                    border-radius: 13px !important;
                    background: #ffffff !important;
                    color: #172033 !important;
                    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                    font-size: 13.5px;
                    line-height: 1.6;
                    resize: vertical;
                }

                .kwb-system-prompt textarea {
                    padding: 12px 13px !important;
                    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                    font-size: 12.5px;
                    line-height: 1.5;
                    resize: vertical;
                }

                .kwb-root .jupyter-widgets.widget-accordion {
                    overflow: hidden;
                    border: 1px solid var(--kwb-border);
                    border-radius: var(--kwb-radius);
                    background: var(--kwb-surface);
                    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
                }

                .kwb-settings-content {
                    padding: 8px 4px 6px;
                }

                .kwb-actionbar {
                    align-items: center;
                    flex-wrap: wrap;
                }

                .kwb-status-wrap {
                    min-width: 230px;
                    margin-left: auto;
                }

                .kwb-status {
                    display: inline-flex;
                    align-items: center;
                    min-height: 32px;
                    box-sizing: border-box;
                    padding: 6px 12px;
                    border: 1px solid var(--kwb-border);
                    border-radius: 999px;
                    background: var(--kwb-surface-soft);
                    font-size: 12.5px;
                    line-height: 1.35;
                    font-weight: 600;
                }

                .kwb-root .widget-button {
                    border: 0 !important;
                    border-radius: 9px !important;
                    font-weight: 700 !important;
                    box-shadow: 0 2px 5px rgba(15, 23, 42, 0.12);
                    transition: transform 130ms ease, box-shadow 130ms ease;
                }

                .kwb-root .widget-button:hover:not(:disabled) {
                    transform: translateY(-1px);
                    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.15);
                }

                .kwb-root .widget-button:active:not(:disabled) {
                    transform: translateY(0);
                }

                .kwb-copy-widget-btn {
                    background: #eef2ff !important;
                    color: #3730a3 !important;
                    border: 1px solid #a5b4fc !important;
                    font-size: 12px !important;
                    font-weight: 700 !important;
                    padding: 0 10px !important;
                }

                .kwb-console {
                    box-sizing: border-box;
                    padding: 16px;
                    border: 1px solid #334155;
                    border-radius: var(--kwb-radius);
                    background: #0f172a;
                    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.22);
                }

                .kwb-console-title {
                    color: #e5e7eb;
                    font-size: 12px;
                    font-weight: 750;
                    letter-spacing: 0.06em;
                    text-transform: uppercase;
                }

                .kwb-console-output {
                    box-sizing: border-box;
                    border: 1px solid #334155 !important;
                    border-radius: 10px;
                    background: #090e1a !important;
                    color: #e2e8f0;
                    font-size: 13.5px;
                    line-height: 1.55;
                }

                .kwb-chat-shell {
                    box-sizing: border-box;
                    border: 1px solid var(--kwb-border);
                    border-radius: 16px;
                    background: linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(241, 245, 249, 0.98));
                    box-shadow: var(--kwb-shadow);
                }

                .kwb-chat-card {
                    box-sizing: border-box;
                    width: 100% !important;
                    margin: 8px 0 !important;
                    border-radius: 14px;
                    animation: kwb-card-in 160ms ease-out;
                }

                @keyframes kwb-card-in {
                    from { opacity: 0; transform: translateY(3px); }
                    to { opacity: 1; transform: translateY(0); }
                }

                .kwb-user-card {
                    padding: 16px 20px;
                    border: 1px solid #bfdbfe;
                    background: linear-gradient(135deg, #eff6ff, #f8fafc);
                    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.06);
                }

                .kwb-model-card {
                    padding: 20px 24px;
                    border: 1px solid #cbd5e1;
                    background: #ffffff;
                    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.07);
                }

                .kwb-agent-card {
                    padding: 20px 24px;
                    border: 1px solid #a5b4fc;
                    background: linear-gradient(135deg, #ffffff, #f5f7ff);
                    box-shadow: 0 8px 22px rgba(67, 56, 202, 0.08);
                }

                .kwb-card-footer {
                    margin-top: 12px;
                    padding-top: 10px;
                    border-top: 1px dashed var(--kwb-border);
                }

                .kwb-card-identity {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    min-width: 0;
                }

                .kwb-avatar {
                    display: inline-grid;
                    place-items: center;
                    width: 36px;
                    height: 36px;
                    flex: 0 0 36px;
                    border-radius: 10px;
                    font-size: 17px;
                }

                .kwb-user-avatar { background: #dbeafe; color: #1d4ed8; }
                .kwb-model-avatar { background: #dcfce7; color: #15803d; }
                .kwb-agent-avatar { background: #e0e7ff; color: #4338ca; }

                .kwb-role {
                    color: var(--kwb-text);
                    font-size: 14.5px;
                    font-weight: 750;
                }

                .kwb-role-badge {
                    display: inline-block;
                    margin-left: 8px;
                    padding: 3px 9px;
                    border-radius: 999px;
                    background: #e0e7ff;
                    color: #3730a3;
                    font-size: 11px;
                    font-weight: 700;
                    vertical-align: middle;
                }

                .kwb-turn-badge {
                    display: inline-block;
                    margin-right: 6px;
                    padding: 2px 8px;
                    border-radius: 6px;
                    background: #e2e8f0;
                    color: #334155;
                    font-size: 11px;
                    font-weight: 750;
                }

                .kwb-channel-badge {
                    display: inline-block;
                    margin-left: 6px;
                    padding: 2px 8px;
                    border-radius: 6px;
                    background: #f1f5f9;
                    color: #475569;
                    border: 1px solid #cbd5e1;
                    font-size: 10.5px;
                    font-weight: 700;
                }

                .kwb-time { margin-top: 2px; color: #64748b; font-size: 11px; }

                .kwb-chip {
                    display: inline-block;
                    margin: 0 6px 8px 0;
                    padding: 4px 10px;
                    border-radius: 999px;
                    background: #dbeafe;
                    color: #1e40af;
                    font-size: 11.5px;
                    font-weight: 650;
                }

                .kwb-user-body {
                    white-space: pre-wrap;
                    overflow-wrap: anywhere;
                    color: #1e293b;
                    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                    font-size: 13.5px;
                    line-height: 1.65;
                }

                .kwb-model-body {
                    overflow-wrap: anywhere;
                    color: #1e293b;
                    font-size: 14.5px;
                    line-height: 1.72;
                }

                .kwb-model-body pre {
                    overflow-x: auto;
                    margin: 12px 0;
                    padding: 16px;
                    border: 1px solid #1e293b;
                    border-radius: 11px;
                    background: var(--kwb-code-bg);
                    color: #e2e8f0;
                    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                    font-size: 13.5px;
                    line-height: 1.6;
                }

                .kwb-model-body code {
                    padding: 2px 6px;
                    border-radius: 5px;
                    background: #f1f5f9;
                    color: #be123c;
                    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                    font-size: 0.92em;
                }

                .kwb-model-body pre code {
                    padding: 0;
                    background: transparent;
                    color: inherit;
                }

                /* DyTopo Visualizer styles */
                .dytopo-container {
                    padding: 14px;
                    border-radius: 14px;
                    background: #0f172a;
                    border: 1px solid #334155;
                    box-shadow: inset 0 2px 6px rgba(0,0,0,0.3);
                }

                .dytopo-meta-bar {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                    color: #94a3b8;
                    font-size: 12px;
                    font-weight: 600;
                }

                .dytopo-svg {
                    width: 100%;
                    height: 270px;
                    display: block;
                }

                @media (prefers-color-scheme: dark) {
                    :root {
                        --kwb-surface: #182235;
                        --kwb-surface-soft: #202c40;
                        --kwb-border: #334155;
                        --kwb-border-strong: #475569;
                        --kwb-text: #f1f5f9;
                        --kwb-muted: #94a3b8;
                        --kwb-shadow: 0 12px 30px rgba(0, 0, 0, 0.25);
                    }
                    .kwb-prompt textarea {
                        border-color: #475569 !important;
                        background: #0f172a !important;
                        color: #f1f5f9 !important;
                    }
                    .kwb-chat-shell {
                        background: linear-gradient(180deg, #111827, #0f172a);
                    }
                    .kwb-user-card {
                        border-color: #1d4ed8;
                        background: linear-gradient(135deg, #172554, #1e293b);
                    }
                    .kwb-model-card { border-color: #334155; background: #1e293b; }
                    .kwb-agent-card {
                        border-color: #4f46e5;
                        background: linear-gradient(135deg, #1e293b, #1e1b4b);
                    }
                    .kwb-role, .kwb-model-body, .kwb-user-body { color: #f1f5f9; }
                    .kwb-model-body code { background: #334155; color: #fda4af; }
                    .kwb-copy-widget-btn {
                        border-color: #6366f1 !important;
                        background: #312e81 !important;
                        color: #e0e7ff !important;
                    }
                }
            </style>
            """
        )

        safe_work_dir = html.escape(str(self.work_dir))
        model_count = len(self.available_models)
        markdown_badge = "On" if HAS_MARKDOWN else "Fallback"

        self.header = widgets.HTML(
            value=f"""
            <div class="kwb-header">
                <div class="kwb-header-row">
                    <div class="kwb-brand">
                        <div class="kwb-logo">⚡</div>
                        <div>
                            <div class="kwb-title">Kaggle LLM Workbench v8 (DyTopo Edition)</div>
                            <div class="kwb-subtitle">
                                Dynamic Agent Interaction Topology with live self-reconfiguring graph routing,
                                specialized multi-agent roles, code execution verification, and interactive visual telemetry.
                            </div>
                        </div>
                    </div>
                    <div class="kwb-header-badges">
                        <span class="kwb-header-badge">◉ {model_count} models</span>
                        <span class="kwb-header-badge">🕸️ DyTopo: Active</span>
                        <span class="kwb-header-badge">📁 {safe_work_dir}</span>
                        <span class="kwb-header-badge">Markdown: {markdown_badge}</span>
                    </div>
                </div>
            </div>
            """
        )

        # Single prompt model dropdown
        self.model_dropdown = widgets.Dropdown(
            options=self.available_models,
            value=self.default_model,
            description="Active Model",
            style={"description_width": "90px"},
            layout=widgets.Layout(width="470px", max_width="100%", min_width="280px"),
        )

        self.temp_slider = widgets.FloatSlider(
            value=0.2,
            min=0.0,
            max=1.5,
            step=0.05,
            description="Temperature",
            readout_format=".2f",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="330px", max_width="100%"),
        )

        self.max_turns_slider = widgets.IntSlider(
            value=10,
            min=1,
            max=50,
            step=1,
            description="Memory turns",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="330px", max_width="100%"),
        )

        self.sys_prompt_input = widgets.Textarea(
            value=(
                "You are an elite coding assistant and software architect. "
                "Analyze, review, and optimize code provided by the user. "
                "Maintain full context across the conversation."
            ),
            layout=widgets.Layout(width="100%", height="85px"),
        )
        self.sys_prompt_input.add_class("kwb-system-prompt")

        settings_controls = widgets.HBox(
            [self.temp_slider, self.max_turns_slider],
            layout=widgets.Layout(width="100%", flex_flow="row wrap"),
        )
        settings_controls.add_class("kwb-toolbar")

        settings_content = widgets.VBox(
            [
                settings_controls,
                widgets.HTML(
                    value="<div style='margin-top:6px; color:var(--kwb-muted); font-size:10.5px; font-weight:750; text-transform:uppercase;'>System instruction</div>"
                ),
                self.sys_prompt_input,
            ],
            layout=widgets.Layout(width="100%"),
        )
        settings_content.add_class("kwb-settings-content")

        self.settings_accordion = widgets.Accordion(
            children=[settings_content],
            layout=widgets.Layout(width="100%"),
        )
        self.settings_accordion.set_title(0, "⚙ Advanced Configuration & Memory")
        self.settings_accordion.selected_index = None

        # SuperAgent DyTopo UI Initialization
        self._init_dytopo_ui()

        # Prompt input & headers
        self.input_header = widgets.HTML(
            value="""
            <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:9px; flex-wrap:wrap;">
                <div>
                    <div class="kwb-section-title">Prompt / DyTopo Mission Goal</div>
                    <div class="kwb-section-subtitle">
                        Send an individual prompt, or launch the <b>Autonomous DyTopo Swarm</b> to solve complex tasks dynamically.
                    </div>
                </div>
                <div style="color:var(--kwb-muted); font-size:11px; font-weight:650;">
                    Supports fenced code, live graph updates & %%writefile
                </div>
            </div>
            """
        )

        self.textarea = widgets.Textarea(
            value="",
            placeholder=(
                "Describe a mission or paste your code...\n\n"
                "Example DyTopo Goal:\n"
                "Develop an optimized, vectorized Trie-based auto-complete engine in `trie_engine.py`, "
                "write a thorough pytest suite, execute tests, profile memory/latency, and refine."
            ),
            layout=widgets.Layout(width="100%", height="165px"),
        )
        self.textarea.add_class("kwb-prompt")

        # Action Buttons
        self.btn_send = widgets.Button(
            description="Send Prompt",
            button_style="primary",
            icon="paper-plane",
            tooltip="Send a single prompt to the active model",
            layout=widgets.Layout(width="145px", height="38px"),
        )

        self.btn_superagent = widgets.Button(
            description="Launch DyTopo Swarm",
            button_style="warning",
            icon="share-alt",
            tooltip="Run the autonomous multi-agent dynamic topology workflow",
            layout=widgets.Layout(width="195px", height="38px"),
        )

        self.btn_stop_agent = widgets.Button(
            description="Stop Swarm",
            button_style="danger",
            icon="stop",
            tooltip="Request the DyTopo agent swarm to stop cleanly after current step",
            layout=widgets.Layout(width="125px", height="38px"),
            disabled=True,
        )

        self.btn_undo = widgets.Button(
            description="Undo Last",
            button_style="info",
            icon="undo",
            tooltip="Remove the latest conversation turn",
            layout=widgets.Layout(width="115px", height="38px"),
            disabled=True,
        )

        self.btn_copy_last = widgets.Button(
            description="Copy Latest",
            button_style="info",
            icon="copy",
            tooltip="Copy the latest agent response to the clipboard",
            layout=widgets.Layout(width="125px", height="38px"),
            disabled=True,
        )

        self.btn_save = widgets.Button(
            description="Export Files",
            button_style="success",
            icon="download",
            tooltip="Export conversation JSON and extracted code",
            layout=widgets.Layout(width="135px", height="38px"),
        )

        self.btn_reset = widgets.Button(
            description="Clear All",
            button_style="danger",
            icon="trash",
            tooltip="Clear the conversation and reset the topology graph",
            layout=widgets.Layout(width="110px", height="38px"),
        )

        self.status_label = widgets.HTML(
            value="<div class='kwb-status' style='color:#64748b;'>● Ready for prompt or DyTopo Swarm</div>",
            layout=widgets.Layout(min_width="230px"),
        )
        self.status_label.add_class("kwb-status-wrap")

        self.download_links_box = widgets.VBox(layout=widgets.Layout(width="100%", margin="0"))
        self.exec_buttons_box = widgets.HBox(layout=widgets.Layout(width="100%", margin="0", flex_wrap="wrap"))
        self.exec_buttons_box.add_class("kwb-toolbar")

        # Execution Console
        self.exec_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                max_height="750px",
                overflow_y="auto",
                padding="16px",
            ),
        )
        self.exec_output.add_class("kwb-console-output")

        self.btn_copy_console = widgets.Button(
            description="📋 Copy Console Output",
            icon="copy",
            button_style="info",
            layout=widgets.Layout(width="205px", height="36px"),
        )

        self.btn_feed_output = widgets.Button(
            description="Send Output/Error to AI",
            icon="reply",
            button_style="warning",
            layout=widgets.Layout(width="215px", height="36px"),
        )

        self.btn_clear_console = widgets.Button(
            description="Hide Console",
            icon="eye-slash",
            layout=widgets.Layout(width="135px", height="36px"),
        )

        console_actions = widgets.HBox(
            [self.btn_copy_console, self.btn_feed_output, self.btn_clear_console],
            layout=widgets.Layout(margin="8px 0 0 0", flex_wrap="wrap"),
        )
        console_actions.add_class("kwb-toolbar")

        self.exec_container = widgets.VBox(
            [
                widgets.HTML(value="<div class='kwb-console-title'>⚡ Execution Sandbox Console</div>"),
                self.exec_output,
                console_actions,
            ],
            layout=widgets.Layout(display="none", width="100%"),
        )
        self.exec_container.add_class("kwb-console")

        # Timeline sort toggle
        self.sort_order_toggle = widgets.ToggleButtons(
            options=[
                ("⬇️ Newest First (Top)", "newest"),
                ("⬆️ Oldest First (Bottom)", "oldest"),
            ],
            value="newest",
            style={"button_width": "185px", "font_weight": "bold"},
        )

        self.chat_empty_html = widgets.HTML(value=self._get_empty_chat_html())
        self.chat_container = widgets.VBox([self.chat_empty_html], layout=widgets.Layout(width="100%", padding="16px"))
        self.chat_container.add_class("kwb-chat-shell")

        # Attachments UI
        self._init_attachment_ui()

    def _init_dytopo_ui(self):
        # Topology Mode Selector
        self.dytopo_mode_dropdown = widgets.Dropdown(
            options=DYTOPO_MODES,
            value="adaptive",
            description="Topology Mode",
            style={"description_width": "115px"},
            layout=widgets.Layout(width="360px", max_width="100%"),
        )

        # Dynamic Role Model Selectors
        self.role_selectors = {}
        for role_id, meta in ROLE_METADATA.items():
            preferred = (
                self.default_coder_model
                if role_id in ("coder", "tester")
                else (self.default_critic_model if role_id in ("critic", "benchmarker") else self.default_model)
            )
            self.role_selectors[role_id] = widgets.Dropdown(
                options=self.available_models,
                value=preferred,
                description=f"{meta['icon']} {meta['title']}",
                style={"description_width": "160px"},
                layout=widgets.Layout(width="380px", max_width="100%"),
            )

        self.agent_max_steps_slider = widgets.IntSlider(
            value=8,
            min=2,
            max=20,
            step=1,
            description="Max swarm steps",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="280px", max_width="100%"),
        )

        self.agent_auto_exec_checkbox = widgets.Checkbox(
            value=True,
            description="Allow autonomous execution & sandbox validation (.py / .sh)",
            indent=False,
            layout=widgets.Layout(width="auto"),
        )

        # Live SVG Topology Graph Widget
        self.topology_graph_widget = widgets.HTML(
            value=self._render_dytopo_svg(),
            layout=widgets.Layout(width="100%"),
        )

        # Topology Transition History
        self.topology_history_html = widgets.HTML(
            value="<div style='color:#64748b; font-size:12px;'>No topology transitions recorded yet.</div>",
            layout=widgets.Layout(width="100%"),
        )

        # Assembly of DyTopo control panels
        dytopo_header = widgets.HTML(
            value="""
            <div style="margin-bottom:8px;">
                <div class="kwb-section-title">🕸️ DyTopo: Dynamic Agent Interaction Topology & Dynamic Edge Routing</div>
                <div class="kwb-section-subtitle">
                    DyTopo dynamically shifts communication graphs (Adaptive, Star, Chain, Debate, Tree),
                    spawns specialized agents, and re-routes information flows (Direct Messages, Broadcasts, Reflection Loops, Consensus Gates).
                </div>
            </div>
            """
        )

        role_grid = widgets.VBox(
            [
                widgets.HBox([self.role_selectors["architect"], self.role_selectors["coder"]], layout=widgets.Layout(flex_flow="row wrap", gap="8px")),
                widgets.HBox([self.role_selectors["tester"], self.role_selectors["critic"]], layout=widgets.Layout(flex_flow="row wrap", gap="8px")),
                widgets.HBox([self.role_selectors["benchmarker"]], layout=widgets.Layout(flex_flow="row wrap", gap="8px")),
            ],
            layout=widgets.Layout(width="100%", gap="6px"),
        )

        config_bar = widgets.HBox(
            [self.dytopo_mode_dropdown, self.agent_max_steps_slider, self.agent_auto_exec_checkbox],
            layout=widgets.Layout(width="100%", flex_flow="row wrap", gap="14px", align_items="center"),
        )

        dytopo_panel = widgets.VBox(
            [
                dytopo_header,
                config_bar,
                widgets.HTML(value="<div class='kwb-toolbar-label' style='margin-top:6px;'>Specialist Role Allocations</div>"),
                role_grid,
                widgets.HTML(value="<div class='kwb-toolbar-label' style='margin-top:10px;'>Live Topology Interaction Graph</div>"),
                self.topology_graph_widget,
                widgets.HTML(value="<div class='kwb-toolbar-label' style='margin-top:10px;'>Topology Routing History</div>"),
                self.topology_history_html,
            ],
            layout=widgets.Layout(width="100%", gap="10px"),
        )
        dytopo_panel.add_class("kwb-settings-content")

        self.dytopo_accordion = widgets.Accordion(
            children=[dytopo_panel],
            layout=widgets.Layout(width="100%"),
        )
        self.dytopo_accordion.set_title(0, "🕸️ DyTopo Multi-Agent Swarm & Live Topology Visualizer")
        self.dytopo_accordion.selected_index = 0

    def _bind_events(self):
        self.btn_send.on_click(self.on_send_clicked)
        self.btn_superagent.on_click(self.on_dytopo_launch_clicked)
        self.btn_stop_agent.on_click(self.on_stop_agent_clicked)
        self.btn_undo.on_click(self.on_undo_clicked)
        self.btn_copy_last.on_click(self.on_copy_last_clicked)
        self.btn_save.on_click(self.on_save_clicked)
        self.btn_reset.on_click(self.on_reset_clicked)
        self.btn_copy_console.on_click(self._on_copy_console_clicked)
        self.btn_feed_output.on_click(self._on_feed_output_clicked)
        self.btn_clear_console.on_click(lambda _: setattr(self.exec_container.layout, "display", "none"))
        self.sort_order_toggle.observe(lambda _: self.refresh_chat_view(), names="value")
        self.dytopo_mode_dropdown.observe(self._on_topology_mode_changed, names="value")
        self._bind_attachment_events()

    # ==========================================
    # 2. DYTOPO TOPOLOGY & GRAPH VISUALIZATION
    # ==========================================
    def _on_topology_mode_changed(self, change):
        new_mode = change["new"]
        self.dytopo_graph["current_mode"] = new_mode
        self._record_topology_transition(
            step="Config",
            source="User",
            action=f"Switched topology mode to {new_mode.upper()}",
            channel="Orchestration",
        )
        self._refresh_topology_view()

    def _record_topology_transition(self, step, source, action, channel):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        record = {
            "step": step,
            "source": source,
            "action": action,
            "channel": channel,
            "time": ts,
        }
        self.dytopo_graph["transitions"].append(record)

        items_html = []
        for t in reversed(self.dytopo_graph["transitions"][-6:]):
            items_html.append(
                f"<div style='margin-bottom:4px; font-size:11.5px;'>"
                f"<span style='color:#a5b4fc; font-weight:700;'>[{t['time']}]</span> "
                f"<span style='color:#38bdf8; font-weight:700;'>{html.escape(str(t['source']))}</span> "
                f"→ <span style='color:#e2e8f0;'>{html.escape(str(t['action']))}</span> "
                f"<span style='color:#94a3b8;'>({html.escape(str(t['channel']))})</span>"
                f"</div>"
            )
        self.topology_history_html.value = (
            "<div style='max-height:100px; overflow-y:auto; padding:6px 10px; background:#090e1a; border-radius:8px; border:1px solid #1e293b;'>"
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
        """Renders an interactive SVG visualizer depicting nodes, edges, states, and message flow."""
        width = 860
        height = 270
        center_x = width // 2
        center_y = height // 2

        roles = list(ROLE_METADATA.keys())
        node_coords = {}
        total_roles = len(roles)

        mode = self.dytopo_graph.get("current_mode", "adaptive")

        # Layout positioning calculation based on topology mode
        if mode == "star":
            node_coords["architect"] = (center_x, center_y)
            outer_roles = [r for r in roles if r != "architect"]
            for i, r in enumerate(outer_roles):
                angle = 2 * math.pi * i / len(outer_roles)
                node_coords[r] = (
                    center_x + int(240 * math.cos(angle)),
                    center_y + int(95 * math.sin(angle)),
                )
        elif mode == "chain":
            spacing = width // (total_roles + 1)
            for i, r in enumerate(roles):
                node_coords[r] = (spacing * (i + 1), center_y)
        elif mode == "tree":
            node_coords["architect"] = (center_x, 45)
            node_coords["coder"] = (center_x - 170, 135)
            node_coords["tester"] = (center_x - 220, 225)
            node_coords["critic"] = (center_x + 170, 135)
            node_coords["benchmarker"] = (center_x + 220, 225)
        else:  # Adaptive & Debate circular equilibrium layout
            radius_x = 310
            radius_y = 90
            for i, r in enumerate(roles):
                angle = (2 * math.pi * i / total_roles) - (math.pi / 2)
                node_coords[r] = (
                    center_x + int(radius_x * math.cos(angle)),
                    center_y + int(radius_y * math.sin(angle)),
                )

        svg_elements = []

        # Draw edges
        for key, count in self.dytopo_graph["edges"].items():
            try:
                src, dst = key.split("->")
                if src in node_coords and dst in node_coords:
                    x1, y1 = node_coords[src]
                    x2, y2 = node_coords[dst]
                    stroke_w = min(1 + count * 1.5, 6.0)
                    svg_elements.append(
                        f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' "
                        f"stroke='#4f46e5' stroke-width='{stroke_w}' stroke-opacity='0.6' stroke-dasharray='4,4'/>"
                    )
                    # Edge weight badge
                    mid_x = (x1 + x2) // 2
                    mid_y = (y1 + y2) // 2
                    svg_elements.append(
                        f"<circle cx='{mid_x}' cy='{mid_y}' r='8' fill='#1e1b4b' stroke='#6366f1' stroke-width='1.5'/>"
                        f"<text x='{mid_x}' y='{mid_y + 3}' text-anchor='middle' fill='#e0e7ff' font-size='9' font-weight='bold'>{count}</text>"
                    )
            except Exception:
                continue

        # Draw nodes
        for r_id, (nx, ny) in node_coords.items():
            meta = ROLE_METADATA[r_id]
            node_state = self.dytopo_graph["nodes"][r_id]["status"]
            msg_count = self.dytopo_graph["nodes"][r_id]["messages_handled"]

            glow = ""
            border_color = meta["color"]
            if node_state == "active":
                border_color = "#38bdf8"
                glow = "filter='drop-shadow(0 0 10px #38bdf8)'"
            elif node_state == "reviewing":
                border_color = "#f59e0b"
                glow = "filter='drop-shadow(0 0 10px #f59e0b)'"
            elif node_state == "completed":
                border_color = "#10b981"

            svg_elements.append(
                f"<g {glow}>"
                f"<circle cx='{nx}' cy='{ny}' r='28' fill='#1e293b' stroke='{border_color}' stroke-width='2.5'/>"
                f"<text x='{nx}' y='{ny - 2}' text-anchor='middle' font-size='16'>{meta['icon']}</text>"
                f"<text x='{nx}' y='{ny + 14}' text-anchor='middle' fill='{border_color}' font-size='9' font-weight='bold'>{node_state.upper()}</text>"
                f"<text x='{nx}' y='{ny + 42}' text-anchor='middle' fill='#cbd5e1' font-size='11' font-weight='600'>{meta['title']}</text>"
                f"<rect x='{nx - 20}' y='{ny - 36}' width='40' height='14' rx='7' fill='#0f172a' stroke='{border_color}' stroke-width='1'/>"
                f"<text x='{nx}' y='{ny - 26}' text-anchor='middle' fill='#f8fafc' font-size='9' font-weight='bold'>msg: {msg_count}</text>"
                f"</g>"
            )

        active_mode_title = next((t for t, k in DYTOPO_MODES if k == mode), mode)
        total_exchanges = sum(self.dytopo_graph["edges"].values())

        return f"""
        <div class="dytopo-container">
            <div class="dytopo-meta-bar">
                <div><span>Topology Mode:</span> <strong style="color:#a5b4fc;">{active_mode_title}</strong></div>
                <div><span>Total Edge Interactions:</span> <strong style="color:#38bdf8;">{total_exchanges}</strong></div>
            </div>
            <svg viewBox="0 0 {width} {height}" class="dytopo-svg">
                {''.join(svg_elements)}
            </svg>
        </div>
        """

    def _refresh_topology_view(self):
        self.topology_graph_widget.value = self._render_dytopo_svg()

    # ==========================================
    # 3. PROMPT COMPOSITION & STANDARD QUERIES
    # ==========================================
    def _build_context(self, new_user_input: str, new_attachments: list) -> str:
        sys_prompt = self.sys_prompt_input.value.strip()
        context_parts = [f"System:\n{sys_prompt}\n"] if sys_prompt else []

        max_messages = self.max_turns_slider.value * 2
        recent_history = self.chat_history[-max_messages:] if max_messages > 0 else self.chat_history

        for message in recent_history:
            if message["role"] == "user":
                role_string = "User"
                body = self._compose_prompt(message["text"], message.get("attachments", []))
            else:
                role_string = message.get("agent_role") or "Assistant"
                body = message["text"]
            context_parts.append(f"{role_string}:\n{body}\n")

        final_prompt = self._compose_prompt(new_user_input, new_attachments)
        context_parts.append(f"User:\n{final_prompt}\n\nAssistant:")
        return "\n".join(context_parts)

    def on_send_clicked(self, _):
        typed_text = self.textarea.value.strip()
        attachments = list(self.attachments)

        if not typed_text and not attachments:
            self._set_status("⚠ Please enter a prompt or attach a file first.", "#d93025")
            return

        self._toggle_controls(disabled=True)
        chosen_model = self.model_dropdown.value
        safe_model = html.escape(str(chosen_model))
        temperature_text = f"{self.temp_slider.value:.2f}"

        self._set_status(f"⏳ Querying <b>{safe_model}</b> at temperature {temperature_text}...", "#d97706")

        try:
            context = self._build_context(typed_text, attachments)
            llm = kbench.llms[chosen_model]
            raw_response = llm.prompt(context, temperature=self.temp_slider.value)
            response_text = str(raw_response)

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.chat_history.append({"role": "user", "text": typed_text, "attachments": attachments, "time": timestamp})
            self.chat_history.append({"role": "model", "text": response_text, "model_name": chosen_model, "time": timestamp})

            self.textarea.value = ""
            self.attachments = []
            self._refresh_attachment_view()
            self.refresh_chat_view()
            self.save_chat_to_disk()

            self._set_status(f"✓ Response received at {timestamp}.", "#16805a")
        except Exception as exc:
            self._set_status(f"✕ Error: {html.escape(str(exc))}", "#c62828")
        finally:
            self._toggle_controls(disabled=False)

    # ==========================================
    # 4. AUTONOMOUS DYTOPO SWARM ENGINE
    # ==========================================
    def on_stop_agent_clicked(self, _):
        self._agent_stop_requested = True
        self._set_status("⏹ Stop requested. Halting DyTopo swarm cleanly...", "#d93025")

    def _parse_dytopo_decision(self, raw_text: str) -> dict:
        """Parses structured JSON routing instructions from the Lead Architect / Router."""
        cleaned = raw_text.strip()
        fenced_matches = re.findall(
            rf"{CODE_FENCE}(?:json)?[ \t]*\r?\n(.*?){CODE_FENCE}",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        candidates = list(fenced_matches)

        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            candidates.append(cleaned[first_brace : last_brace + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "action" in parsed:
                    return parsed
            except Exception:
                continue

        return {
            "thought": "Direct response without envelope.",
            "action": "FINISH",
            "channel": "Consensus Gate",
            "next_agent": "architect",
            "final_answer": raw_text,
        }

    def _append_agent_card(self, agent_role: str, model_name: str, text: str, badge: str = "", channel: str = ""):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.chat_history.append(
            {
                "role": "model",
                "agent_role": agent_role,
                "agent_badge": badge,
                "channel": channel,
                "model_name": model_name,
                "text": text,
                "time": timestamp,
            }
        )
        self.refresh_chat_view()

    def on_dytopo_launch_clicked(self, _):
        goal_text = self.textarea.value.strip()
        attachments = list(self.attachments)

        if not goal_text and not attachments:
            self._set_status("⚠ Please enter a mission goal or attach files first.", "#d93025")
            return

        self._agent_stop_requested = False
        self._toggle_controls(disabled=True)
        self.btn_stop_agent.disabled = False

        topology_mode = self.dytopo_mode_dropdown.value
        self.dytopo_graph["current_mode"] = topology_mode
        self._reset_node_statuses()

        max_steps = self.agent_max_steps_slider.value
        allow_exec = self.agent_auto_exec_checkbox.value
        temp = self.temp_slider.value

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        composed_goal = self._compose_prompt(goal_text, attachments)

        self.chat_history.append(
            {
                "role": "user",
                "text": f"🚀 [DyTopo Mission Goal • Mode: {topology_mode.upper()}]\n{goal_text}",
                "attachments": attachments,
                "time": timestamp,
            }
        )
        self.textarea.value = ""
        self.attachments = []
        self._refresh_attachment_view()
        self.refresh_chat_view()

        swarm_memory = [
            f"MISSION GOAL:\n{composed_goal}\n",
            f"ACTIVE TOPOLOGY MODE: {topology_mode.upper()}\n",
            f"WORKSPACE DIRECTORY: {self.work_dir}\n",
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
                self._set_status(f"🕸️ <b>Step {step}/{max_steps}:</b> Lead Architect dynamically routing graph...", "#4f46e5")

                workspace_files = self._list_workspace_files()
                recent_logs = "\n---\n".join(swarm_memory[-6:])

                router_prompt = (
                    "You are the Dynamic Interaction Topology Router (DyTopo Master) and Lead Architect.\n"
                    f"TOPOLOGY MODE: {topology_mode.upper()}\n"
                    f"WORKSPACE FILES: {workspace_files}\n"
                    f"STEP: {step} of {max_steps}\n"
                    f"LAST ACTIVE AGENT: {active_agent}\n\n"
                    f"RECENT EXECUTION & OBSERVATION LOGS:\n{recent_logs}\n\n"
                    "Select the next optimal routing edge and channel to solve the MISSION GOAL.\n"
                    "Allowed next_agent values: ['coder', 'tester', 'critic', 'benchmarker', 'architect']\n"
                    "Allowed action values: ['DELEGATE', 'EXECUTE_CODE', 'REFLECT', 'FINISH']\n"
                    "Allowed channel values: ['Direct Message', 'Team Broadcast', 'Reflection Loop', 'Consensus Gate']\n\n"
                    "Respond ONLY with a valid JSON object matching this schema:\n"
                    "{\n"
                    '  "thought": "Deep analytical reasoning about current state and next interaction edge",\n'
                    '  "topology_adaptation": "Note on how the dynamic graph is evolving",\n'
                    '  "channel": "Direct Message" | "Team Broadcast" | "Reflection Loop" | "Consensus Gate",\n'
                    '  "action": "DELEGATE" | "EXECUTE_CODE" | "REFLECT" | "FINISH",\n'
                    '  "next_agent": "coder" | "tester" | "critic" | "benchmarker" | "architect",\n'
                    '  "target_file": "filename.py (if applicable)",\n'
                    '  "instruction": "Detailed task instructions for the chosen agent",\n'
                    '  "final_answer": "Complete final summary for the operator (only if action is FINISH)"\n'
                    "}"
                )

                arch_raw = str(kbench.llms[arch_model].prompt(router_prompt, temperature=temp))
                decision = self._parse_dytopo_decision(arch_raw)

                thought = decision.get("thought", "")
                action = str(decision.get("action", "FINISH")).upper().strip()
                channel = str(decision.get("channel", "Direct Message")).strip()
                next_agent = str(decision.get("next_agent", "coder")).lower().strip()
                target_file = os.path.basename(str(decision.get("target_file") or "").strip()) or "solution.py"
                instruction = str(decision.get("instruction", ""))
                final_answer = str(decision.get("final_answer", ""))
                adaptation = str(decision.get("topology_adaptation", ""))

                if next_agent not in ROLE_METADATA:
                    next_agent = "coder"

                self._record_edge_interaction("architect", next_agent)
                self._record_topology_transition(
                    step=f"{step}/{max_steps}",
                    source="Architect",
                    action=f"{action} → {ROLE_METADATA[next_agent]['title']}",
                    channel=channel,
                )

                if action == "FINISH":
                    self._set_node_status("architect", "completed")
                    finish_md = f"**Architect Thought:** {thought}\n\n{final_answer or instruction or arch_raw}"
                    self._append_agent_card(
                        agent_role="🏁 Lead Architect (Goal Accomplished)",
                        model_name=arch_model,
                        badge=f"Step {step}/{max_steps} • FINISHED",
                        channel="Consensus Gate",
                        text=finish_md,
                    )
                    self.save_chat_to_disk()
                    self._set_status(f"✓ DyTopo mission successfully completed in {step} steps.", "#16805a")
                    return

                # Display the Architect's routing dispatch card
                routing_card_md = (
                    f"**Thought:** {thought}\n\n"
                    f"- **Dynamic Edge:** `Architect` ➔ `{ROLE_METADATA[next_agent]['title']}`\n"
                    f"- **Channel:** `{channel}` | **Action:** `{action}`\n"
                    + (f"- **Topology Adaptation:** {adaptation}\n" if adaptation else "")
                    + (f"- **Target File:** `{target_file}`\n" if target_file else "")
                    + (f"- **Instruction:** {instruction}" if instruction else "")
                )
                self._append_agent_card(
                    agent_role="🏛️ Lead Architect (Router)",
                    model_name=arch_model,
                    badge=f"Step {step}/{max_steps} • Routing",
                    channel=channel,
                    text=routing_card_md,
                )

                if self._agent_stop_requested:
                    break

                # Execute sandbox code if instructed
                if action == "EXECUTE_CODE":
                    if not allow_exec:
                        swarm_memory.append(f"STEP {step}: Execution skipped because sandbox execution is disabled.")
                        continue

                    self._set_status(f"⚡ <b>Step {step}/{max_steps}:</b> Sandbox executing <code>{html.escape(target_file)}</code>...", "#d97706")
                    success, exec_log = self.execute_file(target_file)
                    status_badge = "PASS ✓" if success else "FAIL ✕"

                    exec_md = (
                        f"Executed `{target_file}` via Sandbox ➔ **{status_badge}**\n\n"
                        f"{CODE_FENCE}text\n{exec_log[-3500:]}\n{CODE_FENCE}"
                    )
                    self._append_agent_card(
                        agent_role="⚡ Sandbox Executor",
                        model_name="Local IPython Kernel",
                        badge=f"Step {step}/{max_steps} • {status_badge}",
                        channel="Direct Message",
                        text=exec_md,
                    )
                    swarm_memory.append(f"STEP {step} [EXECUTION of {target_file} -> {status_badge}]:\n{exec_log[-3000:]}")
                    active_agent = "tester" if not success else "critic"
                    continue

                # Delegate to Specialist Role
                self._set_node_status(next_agent, "active")
                agent_model = self.role_selectors[next_agent].value
                role_info = ROLE_METADATA[next_agent]

                self._set_status(
                    f"{role_info['icon']} <b>Step {step}/{max_steps}:</b> {role_info['title']} ({html.escape(agent_model)}) working...",
                    role_info["color"],
                )

                existing_code = ""
                path = self.work_dir / target_file
                if path.is_file():
                    try:
                        content = self._read_workspace_file(target_file)
                        fence = self._fence_for(content)
                        existing_code = f"\nCURRENT WORKSPACE FILE `{target_file}`:\n{fence}python\n{content}\n{fence}\n"
                    except Exception:
                        pass

                agent_prompt = (
                    f"You are the {role_info['title']} in a dynamic multi-agent AI team.\n"
                    f"ROLE OBJECTIVE: {role_info['description']}\n\n"
                    f"MISSION GOAL:\n{composed_goal}\n\n"
                    f"INCOMING INSTRUCTION FROM ARCHITECT:\n{instruction}\n"
                    f"{existing_code}\n"
                    "RECENT CONTEXT & EXECUTION HISTORY:\n"
                    + "\n---\n".join(swarm_memory[-4:])
                    + "\n\nProvide your comprehensive, production-ready response. "
                    "If providing or revising code, include complete, self-contained implementations "
                    f"and start your fenced code block with `%%writefile {target_file}`."
                )

                agent_resp = str(kbench.llms[agent_model].prompt(agent_prompt, temperature=temp))
                self._append_agent_card(
                    agent_role=f"{role_info['icon']} {role_info['title']}",
                    model_name=agent_model,
                    badge=f"Step {step}/{max_steps} • {target_file}",
                    channel=channel,
                    text=agent_resp,
                )

                # Save generated code blocks automatically
                saved = self._extract_and_save_blocks(agent_resp, default_filename=target_file)
                self.save_chat_to_disk()

                swarm_memory.append(
                    f"STEP {step} [{role_info['title']} -> {agent_model}]:\n"
                    f"Saved: {saved}\n"
                    f"Response Summary:\n{agent_resp[-2500:]}"
                )

                self._set_node_status(next_agent, "completed")
                active_agent = next_agent

            if not self._agent_stop_requested:
                self._set_status("👑 Lead Architect synthesizing final mission report...", "#4f46e5")
                final_summary_prompt = (
                    "You are the Lead Architect. The maximum autonomous steps limit has been reached. "
                    "Synthesize a clear, definitive summary of everything engineered, tested, verified, and saved:\n\n"
                    + "\n---\n".join(swarm_memory[-8:])
                )
                final_resp = str(kbench.llms[arch_model].prompt(final_summary_prompt, temperature=temp))
                self._append_agent_card(
                    agent_role="🏁 Lead Architect (Mission Wrap-Up)",
                    model_name=arch_model,
                    badge="Max Steps Reached",
                    channel="Team Broadcast",
                    text=final_resp,
                )
                self.save_chat_to_disk()
                self._set_status("✓ DyTopo Swarm completed mission (max steps reached).", "#16805a")

        except Exception as exc:
            self._set_status(f"✕ DyTopo Swarm Error: {html.escape(str(exc))}", "#c62828")
        finally:
            self.btn_stop_agent.disabled = True
            self._reset_node_statuses()
            self._toggle_controls(disabled=False)

    # ==========================================
    # 5. EXECUTION & CODE PERSISTENCE
    # ==========================================
    def _extract_code_blocks(self, text: str) -> list:
        return re.findall(
            rf"{CODE_FENCE}([a-zA-Z0-9_+\-]*)[ \t]*\r?\n(.*?){CODE_FENCE}",
            text,
            flags=re.DOTALL,
        )

    def _extract_and_save_blocks(self, text: str, default_filename: str = None) -> list:
        code_blocks = self._extract_code_blocks(text)
        newly_saved = []

        for index, (language, block) in enumerate(code_blocks):
            lines = block.strip().splitlines()
            normalized_language = language.lower().strip()
            extension = EXTENSION_MAP.get(normalized_language, "txt")

            if lines and lines[0].strip().startswith("%%writefile"):
                raw_filename = lines[0].replace("%%writefile", "", 1).strip()
                safe_filename = os.path.basename(raw_filename)
                if not safe_filename:
                    safe_filename = default_filename or f"extracted_{index + 1}.{extension}"
                content = "\n".join(lines[1:])
            elif default_filename and index == 0:
                safe_filename = os.path.basename(default_filename)
                content = block
            else:
                safe_filename = f"argon_code_{index + 1}.{extension}"
                content = block

            file_path = self.work_dir / safe_filename
            with open(file_path, "w", encoding="utf-8") as output_file:
                output_file.write(content)

            newly_saved.append(safe_filename)
            if safe_filename not in self._saved_files:
                self._saved_files.append(safe_filename)

        return newly_saved

    def save_chat_to_disk(self):
        if not self.chat_history:
            return

        last_answer = ""
        for entry in reversed(self.chat_history):
            if entry["role"] == "model":
                last_answer = entry["text"]
                break

        code_blocks = self._extract_code_blocks(last_answer)

        export_payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "active_model": self.model_dropdown.value,
            "dytopo_mode": self.dytopo_graph.get("current_mode"),
            "dytopo_edges": self.dytopo_graph.get("edges"),
            "role_models": {r: self.role_selectors[r].value for r in ROLE_METADATA},
            "last_response": last_answer,
            "extracted_blocks_count": len(code_blocks),
            "full_history": self.chat_history,
        }

        json_path = self.work_dir / "argon_output.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, ensure_ascii=False, indent=2)

        self._extract_and_save_blocks(last_answer)

        saved_files = ["argon_output.json"] + [f for f in self._saved_files if f != "argon_output.json"]

        links = []
        for filename in saved_files:
            safe_href = html.escape(filename, quote=True)
            safe_label = html.escape(filename)
            links.append(f"<a href='{safe_href}' target='_blank'>📄 {safe_label}</a>")

        links_html = " <span style='opacity:.35;'>•</span> ".join(links)
        self.download_links_box.children = [
            widgets.HTML(
                value=(
                    "<div class='kwb-generated-files' style='padding:12px; border:1px solid #cfdcff; border-radius:11px; background:#f2f6ff; font-size:13px;'>"
                    "<strong style='color:#315db5;'>Workspace Artifacts</strong>"
                    f"<div style='margin-top:3px;'>{links_html}</div></div>"
                )
            )
        ]

        action_buttons = []
        for filename in saved_files:
            if filename.endswith((".py", ".sh")):
                run_btn = widgets.Button(
                    description=f"Run {filename}",
                    button_style="success",
                    icon="play",
                    layout=widgets.Layout(width="auto", height="34px"),
                )
                run_btn.on_click(lambda _, sf=filename: self.execute_file(sf))
                action_buttons.append(run_btn)

            if filename != "argon_output.json":
                copy_btn = widgets.Button(
                    description=f"📋 Copy {filename}",
                    icon="copy",
                    layout=widgets.Layout(width="auto", height="34px"),
                )
                copy_btn.add_class("kwb-copy-widget-btn")
                copy_btn.on_click(lambda btn, sf=filename: self._copy_workspace_file(sf, btn))
                action_buttons.append(copy_btn)

        self.exec_buttons_box.children = action_buttons
        self.attach_select.options = self._list_workspace_files()

    def _copy_workspace_file(self, filename: str, button=None):
        path = self.work_dir / filename
        if not path.is_file():
            self._set_status(f"⚠ File {html.escape(filename)} not found.", "#d93025")
            return
        self._copy_text_to_clipboard(path.read_text(encoding="utf-8"), label=filename, button=button)

    def execute_file(self, filename: str) -> tuple:
        file_path = self.work_dir / filename
        if not file_path.exists():
            msg = f"File '{filename}' was not found in {self.work_dir}."
            self._set_status(f"⚠ {html.escape(msg)}", "#d93025")
            return False, msg

        code = file_path.read_text(encoding="utf-8")
        self.exec_container.layout.display = "flex"
        self.exec_output.clear_output(wait=True)

        safe_filename = html.escape(filename)
        self._set_status(f"⚡ Sandbox executing <b>{safe_filename}</b>...", "#d97706")

        ipython = get_ipython()
        execution_successful = True
        captured = None

        with self.exec_output:
            print(f"┌─ DyTopo Sandbox: {filename}")
            print(f"└─ Working Dir: {file_path}\n")

            with capture_output(stdout=True, stderr=True, display=True) as captured:
                try:
                    if filename.endswith(".sh"):
                        if ipython is None:
                            raise RuntimeError("IPython session required to execute shell scripts.")
                        quoted_path = shlex.quote(str(file_path))
                        result = ipython.system(f"bash {quoted_path}")
                        if isinstance(result, int) and result != 0:
                            execution_successful = False
                    else:
                        if ipython is not None:
                            result = ipython.run_cell(code)
                            if result.error_in_exec is not None or result.error_before_exec is not None:
                                execution_successful = False
                        else:
                            execution_globals = {"__name__": "__main__", "__file__": str(file_path)}
                            exec(compile(code, str(file_path), "exec"), execution_globals, execution_globals)
                except Exception:
                    traceback.print_exc()
                    execution_successful = False

            captured.show()

        log_parts = []
        if captured is not None and captured.stdout:
            log_parts.append(f"STDOUT:\n{captured.stdout.strip()}")
        if captured is not None and captured.stderr:
            log_parts.append(f"STDERR / TRACEBACK:\n{captured.stderr.strip()}")

        log_string = "\n\n".join(log_parts) if log_parts else "Code executed cleanly with no stdout/stderr output."
        self._last_raw_exec_output = log_string
        self._last_exec_log = f"Execution log for `{filename}`:\n{CODE_FENCE}text\n{log_string[-4000:]}\n{CODE_FENCE}"

        if execution_successful:
            self._set_status(f"✓ Execution of <b>{safe_filename}</b> succeeded.", "#16805a")
        else:
            self._set_status(f"✕ Execution failed in <b>{safe_filename}</b>.", "#c62828")

        return execution_successful, log_string

    def _on_feed_output_clicked(self, _):
        if not self._last_exec_log:
            self._set_status("⚠ No execution output is available yet.", "#d97706")
            return
        curr = self.textarea.value.strip()
        prefix = f"{curr}\n\n" if curr else ""
        self.textarea.value = f"{prefix}{self._last_exec_log}\n\nPlease inspect the execution output and refine the solution:"
        self._set_status("Execution output appended to the prompt box.", "#2563b8")

    # ==========================================
    # 6. ATTACHMENTS & WORKSPACE FILES
    # ==========================================
    def _init_attachment_ui(self):
        self.attachments = []
        self.attach_select = widgets.Dropdown(
            options=self._list_workspace_files(),
            description="Workspace",
            style={"description_width": "75px"},
            layout=widgets.Layout(width="340px", max_width="100%"),
        )

        self.btn_refresh_files = widgets.Button(description="Refresh", icon="refresh", layout=widgets.Layout(width="105px", height="34px"))
        self.btn_attach_existing = widgets.Button(
            description="Attach file", button_style="info", icon="paperclip", layout=widgets.Layout(width="130px", height="34px")
        )
        self.upload_widget = widgets.FileUpload(accept=",".join(TEXT_LANG_BY_EXT.keys()), multiple=True, description="Upload", layout=widgets.Layout(width="auto"))
        self.btn_attach_upload = widgets.Button(
            description="Attach uploads", button_style="info", icon="upload", layout=widgets.Layout(width="140px", height="34px")
        )
        self.attachment_list = widgets.HTML()
        self.btn_clear_attachments = widgets.Button(
            description="Clear attachments", button_style="danger", icon="times", layout=widgets.Layout(width="150px", height="34px"), disabled=True
        )

        self.attachment_panel = widgets.VBox(
            [
                widgets.HTML(value="<div class='kwb-section-title'>Attach Context Files</div>"),
                widgets.HBox([self.attach_select, self.btn_refresh_files, self.btn_attach_existing], layout=widgets.Layout(flex_flow="row wrap", gap="8px")),
                widgets.HBox([self.upload_widget, self.btn_attach_upload], layout=widgets.Layout(flex_flow="row wrap", gap="8px")),
                self.attachment_list,
                self.btn_clear_attachments,
            ],
            layout=widgets.Layout(width="100%", gap="8px"),
        )
        self.attachment_panel.add_class("kwb-panel")
        self._refresh_attachment_view()

    def _bind_attachment_events(self):
        self.btn_refresh_files.on_click(self.on_refresh_files_clicked)
        self.btn_attach_existing.on_click(self.on_attach_existing_clicked)
        self.btn_attach_upload.on_click(self.on_attach_upload_clicked)
        self.btn_clear_attachments.on_click(self.on_clear_attachments_clicked)

    def _list_workspace_files(self) -> list:
        return [p.name for p in sorted(self.work_dir.iterdir()) if p.is_file() and p.suffix.lower() in TEXT_LANG_BY_EXT]

    def _decode_text(self, data: bytes, name: str) -> str:
        if len(data) > ATTACH_MAX_BYTES:
            raise ValueError(f"{name} is {len(data):,} bytes (limit {ATTACH_MAX_BYTES:,}).")
        if b"\x00" in data:
            raise ValueError(f"{name} appears to be binary.")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name} is not valid UTF-8 text.") from exc

    def _read_workspace_file(self, filename: str) -> str:
        path = self.work_dir / Path(filename).name
        if not path.is_file():
            raise ValueError(f"{filename} not found in workspace.")
        return self._decode_text(path.read_bytes(), path.name)

    @staticmethod
    def _normalize_uploads(value) -> list:
        if isinstance(value, dict):
            return [(name, bytes(info["content"])) for name, info in value.items()]
        return [(info["name"], bytes(info["content"])) for info in value]

    def _add_attachment(self, name: str, content: str, source: str):
        self.attachments = [it for it in self.attachments if it["name"] != name]
        if len(self.attachments) >= ATTACH_MAX_FILES:
            raise ValueError(f"Limit of {ATTACH_MAX_FILES} attachments reached.")
        self.attachments.append(
            {
                "name": name,
                "content": content,
                "source": source,
                "lang": TEXT_LANG_BY_EXT.get(Path(name).suffix.lower(), "text"),
            }
        )

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
            blocks.append(f"Attached file: {item['name']}\n{fence}{item['lang']}\n{item['content']}\n{fence}")
        instruction = text or "Please analyze the attached file(s)."
        return "\n\n".join(blocks) + "\n\n" + instruction

    def _refresh_attachment_view(self):
        if not self.attachments:
            self.attachment_list.value = "<div style='color:#7a8799; font-size:12px;'>No files attached.</div>"
            self.btn_clear_attachments.disabled = True
            return

        chips = []
        for item in self.attachments:
            lines = len(item["content"].splitlines())
            chips.append(f"<span class='kwb-chip'>📎 {html.escape(item['name'])} <span style='opacity:.7;'>({lines:,} lines)</span></span>")
        self.attachment_list.value = "<div>" + "".join(chips) + "</div>"
        self.btn_clear_attachments.disabled = False

    def on_refresh_files_clicked(self, _):
        files = self._list_workspace_files()
        self.attach_select.options = files
        self.attach_select.value = files[0] if files else None
        self._set_status(f"🔄 Scanned {len(files)} files in workspace.", "#2563b8")

    def on_attach_existing_clicked(self, _):
        fn = self.attach_select.value
        if not fn:
            self._set_status("⚠ No workspace file selected.", "#d97706")
            return
        try:
            content = self._read_workspace_file(fn)
            self._add_attachment(fn, content, "workspace")
        except ValueError as exc:
            self._set_status(f"✕ {html.escape(str(exc))}", "#c62828")
            return
        self._refresh_attachment_view()
        self._set_status(f"📎 Attached {html.escape(fn)}.", "#16805a")

    def on_attach_upload_clicked(self, _):
        uploads = self._normalize_uploads(self.upload_widget.value)
        if not uploads:
            self._set_status("⚠ Choose one or more files to upload first.", "#d97706")
            return
        attached, errors = [], []
        for name, data in uploads:
            try:
                content = self._decode_text(data, name)
                self._add_attachment(name, content, "upload")
                attached.append(name)
            except ValueError as exc:
                errors.append(str(exc))

        self.upload_widget.value = type(self.upload_widget.value)()
        self._refresh_attachment_view()
        if errors:
            self._set_status(f"✕ Skipped: {html.escape(' | '.join(errors))}", "#c62828")
        else:
            self._set_status(f"📎 Attached {len(attached)} uploaded file(s).", "#16805a")

    def on_clear_attachments_clicked(self, _):
        self.attachments = []
        self._refresh_attachment_view()
        self._set_status("Attachments cleared.", "#64748b")

    # ==========================================
    # 7. CLIPBOARD, FORMATTING & CHAT RENDERING
    # ==========================================
    def _copy_text_to_clipboard(self, text: str, label: str = "Content", button: widgets.Button = None):
        payload = json.dumps(text)
        js_code = f"""
        (function() {{
            const text = {payload};
            function fallbackCopy(val) {{
                const ta = document.createElement('textarea');
                ta.value = val;
                ta.setAttribute('readonly', '');
                ta.style.position = 'fixed';
                ta.style.top = '-9999px';
                ta.style.left = '-9999px';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.focus();
                ta.select();
                ta.setSelectionRange(0, ta.value.length);
                try {{ document.execCommand('copy'); }} catch(e) {{}}
                document.body.removeChild(ta);
            }}
            if (navigator.clipboard && window.isSecureContext) {{
                navigator.clipboard.writeText(text).catch(function() {{ fallbackCopy(text); }});
            }} else {{
                fallbackCopy(text);
            }}
        }})();
        """
        with self._js_sink:
            self._js_sink.clear_output(wait=True)
            display(Javascript(js_code))

        if button is not None:
            button.description = "✓ Copied!"
            button.icon = "check"

        self._set_status(f"📋 Copied <b>{html.escape(label)}</b> ({len(text):,} chars) to clipboard.", "#16805a")

    def _format_model_text(self, text: str) -> str:
        if HAS_MARKDOWN:
            return markdown.markdown(text, extensions=["fenced_code", "tables"])
        return f"<div style='white-space:pre-wrap; font-family:monospace;'>{html.escape(text)}</div>"

    def _build_copy_action_bar(self, raw_text: str) -> widgets.HBox:
        copy_md_btn = widgets.Button(
            description="📋 Copy Response",
            icon="copy",
            layout=widgets.Layout(width="auto", height="32px"),
        )
        copy_md_btn.add_class("kwb-copy-widget-btn")
        copy_md_btn.on_click(lambda btn, txt=raw_text: self._copy_text_to_clipboard(txt, label="Full response", button=btn))
        btn_list = [copy_md_btn]

        code_blocks = self._extract_code_blocks(raw_text)
        for idx, (lang, content) in enumerate(code_blocks):
            clean_code = content.strip()
            lbl = f"📋 Copy {lang or 'code'} #{idx + 1}" if len(code_blocks) > 1 else f"📋 Copy {lang or 'code'}"
            copy_c_btn = widgets.Button(
                description=lbl,
                icon="code",
                layout=widgets.Layout(width="auto", height="32px"),
            )
            copy_c_btn.add_class("kwb-copy-widget-btn")
            copy_c_btn.on_click(lambda btn, c=clean_code, l=lbl: self._copy_text_to_clipboard(c, label=l, button=btn))
            btn_list.append(copy_c_btn)

        return widgets.HBox(btn_list, layout=widgets.Layout(flex_flow="row wrap", gap="8px", align_items="center"))

    def _build_user_card_widget(self, entry: dict, turn_num: int = None) -> widgets.Widget:
        time_string = html.escape(entry.get("time", ""))
        raw_text = entry["text"]
        safe_text = html.escape(raw_text)
        turn_badge = f"<span class='kwb-turn-badge'>Turn #{turn_num}</span>" if turn_num is not None else ""
        chips = "".join(f"<span class='kwb-chip'>📎 {html.escape(it['name'])}</span>" for it in entry.get("attachments", []))

        identity = widgets.HTML(
            value=f"""
            <div class="kwb-card-identity">
                <div class="kwb-avatar kwb-user-avatar">◉</div>
                <div>
                    <div class="kwb-role">{turn_badge}Operator</div>
                    <div class="kwb-time">{time_string}</div>
                </div>
            </div>
            """
        )

        copy_btn = widgets.Button(description="📋 Copy Prompt", icon="copy", layout=widgets.Layout(width="auto", height="30px"))
        copy_btn.add_class("kwb-copy-widget-btn")
        copy_btn.on_click(lambda btn, t=raw_text: self._copy_text_to_clipboard(t, label="Prompt", button=btn))

        header_row = widgets.HBox([identity, copy_btn], layout=widgets.Layout(width="100%", justify_content="space-between", align_items="center"))
        body_html = widgets.HTML(value=f"{chips}<div class='kwb-user-body'>{safe_text}</div>", layout=widgets.Layout(width="100%", margin="0"))

        box = widgets.VBox([header_row, body_html], layout=widgets.Layout(width="100%", margin="8px 0"))
        box.add_class("kwb-chat-card")
        box.add_class("kwb-user-card")
        return box

    def _build_model_card_widget(self, entry: dict, turn_num: int = None) -> widgets.Widget:
        time_str = html.escape(entry.get("time", ""))
        model_name = html.escape(entry.get("model_name", "Model"))
        agent_role = entry.get("agent_role")
        agent_badge = entry.get("agent_badge")
        channel = entry.get("channel")
        raw_text = entry["text"]
        rendered_body = self._format_model_text(raw_text)

        turn_badge = f"<span class='kwb-turn-badge'>Turn #{turn_num}</span>" if turn_num is not None else ""
        channel_badge = f"<span class='kwb-channel-badge'>{html.escape(channel)}</span>" if channel else ""

        if agent_role:
            badge_html = f"<span class='kwb-role-badge'>{html.escape(agent_badge)}</span>" if agent_badge else ""
            title_html = f"{turn_badge}{html.escape(agent_role)} <span style='font-weight:500; opacity:.75;'>({model_name})</span>{badge_html}{channel_badge}"
            avatar_cls = "kwb-agent-avatar"
            avatar_icon = "🕸️"
            card_cls = "kwb-agent-card"
        else:
            title_html = f"{turn_badge}{model_name}"
            avatar_cls = "kwb-model-avatar"
            avatar_icon = "✦"
            card_cls = "kwb-model-card"

        identity = widgets.HTML(
            value=f"""
            <div class="kwb-card-identity">
                <div class="kwb-avatar {avatar_cls}">{avatar_icon}</div>
                <div style="min-width:0;">
                    <div class="kwb-role">{title_html}</div>
                    <div class="kwb-time">{time_str}</div>
                </div>
            </div>
            """
        )

        top_bar = self._build_copy_action_bar(raw_text)
        bottom_bar = self._build_copy_action_bar(raw_text)
        bottom_bar.add_class("kwb-card-footer")

        header_row = widgets.HBox([identity, top_bar], layout=widgets.Layout(width="100%", justify_content="space-between", align_items="center"))
        body_html = widgets.HTML(value=f"<div class='kwb-model-body'>{rendered_body}</div>", layout=widgets.Layout(width="100%", margin="0"))

        card = widgets.VBox([header_row, body_html, bottom_bar], layout=widgets.Layout(width="100%", margin="8px 0"))
        card.add_class("kwb-chat-card")
        card.add_class(card_cls)
        return card

    def refresh_chat_view(self):
        if not self.chat_history:
            self.chat_container.children = [self.chat_empty_html]
            return

        turns = []
        current = []
        for entry in self.chat_history:
            if entry["role"] == "user" and current:
                turns.append(current)
                current = [entry]
            else:
                current.append(entry)
        if current:
            turns.append(current)

        card_widgets = []
        newest_first = self.sort_order_toggle.value == "newest"

        if newest_first:
            for turn_idx in range(len(turns), 0, -1):
                turn_items = turns[turn_idx - 1]
                user_msgs = [e for e in turn_items if e["role"] == "user"]
                model_msgs = [e for e in turn_items if e["role"] != "user"]
                for e in reversed(model_msgs):
                    card_widgets.append(self._build_model_card_widget(e, turn_idx))
                for e in user_msgs:
                    card_widgets.append(self._build_user_card_widget(e, turn_idx))
        else:
            for turn_idx, turn_items in enumerate(turns, start=1):
                for e in turn_items:
                    if e["role"] == "user":
                        card_widgets.append(self._build_user_card_widget(e, turn_idx))
                    else:
                        card_widgets.append(self._build_model_card_widget(e, turn_idx))

        self.chat_container.children = card_widgets

    def _get_empty_chat_html(self) -> str:
        return """
        <div style="padding:60px 24px; text-align:center; color:#64748b;">
            <div style="font-size:32px; margin-bottom:10px;">🕸️</div>
            <div style="font-size:15px; font-weight:750; color:#1e293b;">DyTopo Multi-Agent Swarm Ready</div>
            <div style="margin-top:6px; font-size:13px;">Dynamic interaction edges & newest answers appear right at the top.</div>
        </div>
        """

    def _set_status(self, html_msg: str, color: str = "#202124"):
        self.status_label.value = f"<div class='kwb-status' style='color:{color}; border-color:{color}33;'>{html_msg}</div>"

    def _toggle_controls(self, disabled: bool):
        has_turns = len(self.chat_history) >= 2
        self.btn_send.disabled = disabled
        self.btn_superagent.disabled = disabled
        self.btn_save.disabled = disabled
        self.btn_reset.disabled = disabled
        self.btn_undo.disabled = disabled or not has_turns
        self.btn_copy_last.disabled = disabled or not has_turns

        self.btn_attach_existing.disabled = disabled
        self.btn_attach_upload.disabled = disabled
        self.btn_refresh_files.disabled = disabled
        self.btn_clear_attachments.disabled = disabled or not self.attachments

        self.model_dropdown.disabled = disabled
        self.dytopo_mode_dropdown.disabled = disabled
        for sel in self.role_selectors.values():
            sel.disabled = disabled
        self.agent_max_steps_slider.disabled = disabled
        self.agent_auto_exec_checkbox.disabled = disabled
        self.temp_slider.disabled = disabled
        self.max_turns_slider.disabled = disabled
        self.sys_prompt_input.disabled = disabled

    def on_undo_clicked(self, _):
        if not self.chat_history:
            return
        while self.chat_history and self.chat_history[-1]["role"] == "model":
            self.chat_history.pop()
        if self.chat_history and self.chat_history[-1]["role"] == "user":
            last_u = self.chat_history.pop()
            self.textarea.value = last_u["text"]
            self.attachments = list(last_u.get("attachments", []))
            self._refresh_attachment_view()

        self.refresh_chat_view()
        self._set_status("↩ Turn removed and restored into the prompt box.", "#2563b8")
        has_turns = len(self.chat_history) >= 2
        self.btn_undo.disabled = not has_turns
        self.btn_copy_last.disabled = not has_turns

    def on_copy_last_clicked(self, button):
        for entry in reversed(self.chat_history):
            if entry["role"] == "model":
                self._copy_text_to_clipboard(entry["text"], label="Latest response", button=button)
                return
        self._set_status("⚠ No response to copy yet.", "#d97706")

    def _on_copy_console_clicked(self, button):
        if not self._last_raw_exec_output:
            self._set_status("⚠ No console output to copy.", "#d97706")
            return
        self._copy_text_to_clipboard(self._last_raw_exec_output, label="Console output", button=button)

    def on_save_clicked(self, _):
        if not self.chat_history:
            self._set_status("⚠ Nothing to export yet.", "#d97706")
            return
        self.save_chat_to_disk()
        self._set_status(f"💾 Files and conversation exported to {html.escape(str(self.work_dir))}.", "#2563b8")

    def on_reset_clicked(self, _):
        self.chat_history.clear()
        self._last_exec_log = ""
        self._last_raw_exec_output = ""
        self._saved_files = []
        self.attachments = []
        self._refresh_attachment_view()

        # Reset DyTopo graph interactions
        self.dytopo_graph["edges"] = {}
        self.dytopo_graph["transitions"] = []
        for r_id in self.dytopo_graph["nodes"]:
            self.dytopo_graph["nodes"][r_id] = {"status": "idle", "messages_handled": 0}

        self._refresh_topology_view()
        self.topology_history_html.value = "<div style='color:#64748b; font-size:12px;'>No topology transitions recorded yet.</div>"

        self.refresh_chat_view()
        self.download_links_box.children = []
        self.exec_buttons_box.children = []
        self.exec_container.layout.display = "none"
        self.exec_output.clear_output(wait=True)
        self.btn_undo.disabled = True
        self.btn_copy_last.disabled = True
        self._set_status("Workbench session and DyTopo graph reset.", "#64748b")

    def display(self):
        top_bar = widgets.HBox(
            [widgets.HTML(value="<div class='kwb-toolbar-label'>Interactive Direct Model</div>"), self.model_dropdown],
            layout=widgets.Layout(width="100%", align_items="center", flex_flow="row wrap"),
        )
        top_bar.add_class("kwb-toolbar")
        toolbar_panel = widgets.VBox([top_bar], layout=widgets.Layout(width="100%"))
        toolbar_panel.add_class("kwb-panel")

        input_panel = widgets.VBox([self.input_header, self.textarea], layout=widgets.Layout(width="100%"))
        input_panel.add_class("kwb-panel")

        action_bar = widgets.HBox(
            [
                self.btn_send,
                self.btn_superagent,
                self.btn_stop_agent,
                self.btn_undo,
                self.btn_copy_last,
                self.btn_save,
                self.btn_reset,
                self.status_label,
            ],
            layout=widgets.Layout(width="100%", align_items="center", flex_flow="row wrap"),
        )
        action_bar.add_class("kwb-actionbar")
        action_bar.add_class("kwb-toolbar")

        files_area = widgets.VBox([self.download_links_box, self.exec_buttons_box], layout=widgets.Layout(width="100%"))

        chat_heading_bar = widgets.HBox(
            [
                widgets.HTML(
                    value="""
                    <div>
                        <div class="kwb-section-title">DyTopo Interaction & Swarm Timeline</div>
                        <div class="kwb-section-subtitle">
                            Newest responses at the top • Dynamic communication channels & one-click clipboard copy
                        </div>
                    </div>
                    """
                ),
                self.sort_order_toggle,
            ],
            layout=widgets.Layout(width="100%", justify_content="space-between", align_items="center", flex_flow="row wrap", margin="6px 0 2px 0"),
        )

        app_layout = widgets.VBox(
            [
                self.theme,
                self._js_sink,
                self.header,
                toolbar_panel,
                self.dytopo_accordion,
                self.settings_accordion,
                input_panel,
                self.attachment_panel,
                action_bar,
                files_area,
                self.exec_container,
                chat_heading_bar,
                self.chat_container,
            ],
            layout=widgets.Layout(width="100%"),
        )
        app_layout.add_class("kwb-root")
        app_layout.add_class("kwb-shell")
        display(app_layout)


# ==========================================
# 8. WORKBENCH BOOTSTRAP
# ==========================================
workbench = KaggleLLMWorkbench()
workbench.display()