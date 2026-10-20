import datetime
import html
import json
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


class KaggleLLMWorkbench:
    def __init__(self, work_dir="/kaggle/working"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.chat_history = []
        self.attachments = []
        self._last_exec_log = ""
        self._saved_files = []
        self._agent_stop_requested = False

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
    # 1. UI INITIALIZATION
    # ==========================================
    def _init_ui(self):
        # Hidden sink used to execute clipboard JavaScript reliably
        # without being stripped by ipywidgets HTML sanitization.
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
                    --kwb-primary: #5856d6;
                    --kwb-code-bg: #111827;
                    --kwb-radius: 14px;
                    --kwb-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
                }

                .kwb-root {
                    width: 100%;
                    box-sizing: border-box;
                    color: var(--kwb-text);
                    font-family:
                        Inter,
                        ui-sans-serif,
                        system-ui,
                        -apple-system,
                        BlinkMacSystemFont,
                        "Segoe UI",
                        sans-serif;
                }

                .kwb-shell {
                    gap: 13px;
                }

                .kwb-header {
                    box-sizing: border-box;
                    position: relative;
                    overflow: hidden;
                    padding: 23px 25px;
                    border-radius: 18px;
                    background:
                        radial-gradient(
                            circle at 88% 8%,
                            rgba(255, 255, 255, 0.18),
                            transparent 27%
                        ),
                        linear-gradient(
                            135deg,
                            #30327a 0%,
                            #5856d6 55%,
                            #7c3aed 100%
                        );
                    box-shadow: 0 16px 38px rgba(50, 51, 123, 0.22);
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
                    width: 49px;
                    height: 49px;
                    flex: 0 0 49px;
                    border: 1px solid rgba(255, 255, 255, 0.24);
                    border-radius: 14px;
                    background: rgba(255, 255, 255, 0.13);
                    box-shadow:
                        inset 0 1px 0 rgba(255, 255, 255, 0.18),
                        0 8px 20px rgba(24, 24, 70, 0.18);
                    font-size: 25px;
                }

                .kwb-title {
                    font-size: 22px;
                    line-height: 1.2;
                    font-weight: 760;
                    letter-spacing: -0.02em;
                }

                .kwb-subtitle {
                    max-width: 650px;
                    margin-top: 5px;
                    color: rgba(255, 255, 255, 0.80);
                    font-size: 12px;
                    line-height: 1.5;
                }

                .kwb-header-badges {
                    display: flex;
                    align-items: center;
                    gap: 7px;
                    flex-wrap: wrap;
                }

                .kwb-header-badge {
                    padding: 7px 10px;
                    border: 1px solid rgba(255, 255, 255, 0.20);
                    border-radius: 999px;
                    background: rgba(255, 255, 255, 0.12);
                    color: #ffffff;
                    font-size: 10.5px;
                    font-weight: 650;
                    backdrop-filter: blur(8px);
                }

                .kwb-panel {
                    box-sizing: border-box;
                    padding: 15px;
                    border: 1px solid var(--kwb-border);
                    border-radius: var(--kwb-radius);
                    background: var(--kwb-surface);
                    box-shadow: var(--kwb-shadow);
                }

                .kwb-superagent-panel {
                    border: 1px solid #c7d2fe;
                    background:
                        linear-gradient(
                            135deg,
                            rgba(238, 242, 255, 0.72),
                            rgba(245, 243, 255, 0.72)
                        );
                }

                .kwb-section-title {
                    color: var(--kwb-text);
                    font-size: 14px;
                    font-weight: 750;
                }

                .kwb-section-subtitle {
                    margin-top: 3px;
                    color: var(--kwb-muted);
                    font-size: 11px;
                    line-height: 1.45;
                }

                .kwb-toolbar {
                    align-items: center;
                    flex-wrap: wrap;
                    gap: 8px;
                }

                .kwb-toolbar-label {
                    color: var(--kwb-muted);
                    font-size: 10.5px;
                    font-weight: 750;
                    letter-spacing: 0.08em;
                    text-transform: uppercase;
                }

                .kwb-root .widget-label {
                    color: var(--kwb-text);
                    font-weight: 650;
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
                    transition:
                        border-color 150ms ease,
                        box-shadow 150ms ease;
                }

                .kwb-root select:focus,
                .kwb-root textarea:focus,
                .kwb-root input:focus {
                    border-color: var(--kwb-primary) !important;
                    box-shadow:
                        0 0 0 3px rgba(88, 86, 214, 0.14) !important;
                }

                .kwb-prompt textarea {
                    min-height: 160px;
                    padding: 15px 16px !important;
                    border-color: #cfd8e5 !important;
                    border-radius: 13px !important;
                    background: #ffffff !important;
                    color: #172033 !important;
                    font-family:
                        ui-monospace,
                        SFMono-Regular,
                        Menlo,
                        Monaco,
                        Consolas,
                        monospace;
                    font-size: 13px;
                    line-height: 1.55;
                    resize: vertical;
                    box-shadow:
                        inset 0 1px 2px rgba(15, 23, 42, 0.03);
                }

                .kwb-system-prompt textarea {
                    padding: 12px 13px !important;
                    font-family:
                        ui-monospace,
                        SFMono-Regular,
                        Menlo,
                        Monaco,
                        Consolas,
                        monospace;
                    font-size: 12px;
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
                    padding: 7px 4px 5px;
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
                    min-height: 31px;
                    box-sizing: border-box;
                    padding: 6px 11px;
                    border: 1px solid var(--kwb-border);
                    border-radius: 999px;
                    background: var(--kwb-surface-soft);
                    font-size: 12px;
                    line-height: 1.35;
                    font-weight: 600;
                }

                .kwb-root .widget-button {
                    border: 0 !important;
                    border-radius: 9px !important;
                    font-weight: 700 !important;
                    box-shadow: 0 2px 5px rgba(15, 23, 42, 0.12);
                    transition:
                        transform 130ms ease,
                        box-shadow 130ms ease,
                        filter 130ms ease;
                }

                .kwb-root .widget-button:hover:not(:disabled) {
                    transform: translateY(-1px);
                    filter: brightness(1.03);
                    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.15);
                }

                .kwb-root .widget-button:active:not(:disabled) {
                    transform: translateY(0);
                }

                .kwb-root .widget-button:disabled {
                    opacity: 0.55;
                    box-shadow: none;
                }

                .kwb-root .jupyter-button.mod-primary {
                    background:
                        linear-gradient(135deg, #5856d6, #7048d7);
                    color: #ffffff;
                }

                .kwb-root .jupyter-button.mod-success {
                    background:
                        linear-gradient(135deg, #12815b, #19a373);
                    color: #ffffff;
                }

                .kwb-root .jupyter-button.mod-danger {
                    background:
                        linear-gradient(135deg, #c73547, #e34e5c);
                    color: #ffffff;
                }

                .kwb-root .jupyter-button.mod-warning {
                    background:
                        linear-gradient(135deg, #dc811b, #f0a02d);
                    color: #ffffff;
                }

                .kwb-root .jupyter-button.mod-info {
                    background:
                        linear-gradient(135deg, #2871c7, #4293e3);
                    color: #ffffff;
                }

                .kwb-copy-widget-btn {
                    background: #f1f5f9 !important;
                    color: #334155 !important;
                    border: 1px solid #cbd5e1 !important;
                    font-size: 11.5px !important;
                    box-shadow: none !important;
                }

                .kwb-copy-widget-btn:hover:not(:disabled) {
                    background: #eef2ff !important;
                    color: #4338ca !important;
                    border-color: #818cf8 !important;
                }

                .kwb-generated-files {
                    padding: 11px 13px;
                    border: 1px solid #cfdcff;
                    border-radius: 11px;
                    background: #f2f6ff;
                    color: #334155;
                    font-size: 12px;
                    line-height: 1.75;
                }

                .kwb-generated-files strong {
                    color: #315db5;
                }

                .kwb-generated-files a {
                    color: #4057c8;
                    font-weight: 650;
                    text-decoration: none;
                }

                .kwb-generated-files a:hover {
                    text-decoration: underline;
                }

                .kwb-console {
                    box-sizing: border-box;
                    padding: 15px;
                    border: 1px solid #334155;
                    border-radius: var(--kwb-radius);
                    background: #111827;
                    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.2);
                }

                .kwb-console-title {
                    color: #e5e7eb;
                    font-size: 11px;
                    font-weight: 750;
                    letter-spacing: 0.06em;
                    text-transform: uppercase;
                }

                .kwb-console-output {
                    box-sizing: border-box;
                    border: 1px solid #374151 !important;
                    border-radius: 10px;
                    background: #0b1220 !important;
                    color: #d7e0ec;
                }

                .kwb-chat-shell {
                    box-sizing: border-box;
                    border: 1px solid var(--kwb-border);
                    border-radius: 16px;
                    background:
                        linear-gradient(
                            180deg,
                            rgba(248, 250, 252, 0.97),
                            rgba(243, 246, 251, 0.97)
                        );
                    box-shadow: var(--kwb-shadow);
                }

                .kwb-chat-scroll {
                    scrollbar-width: thin;
                    scrollbar-color: #aeb9c8 transparent;
                }

                .kwb-chat-scroll::-webkit-scrollbar {
                    width: 8px;
                }

                .kwb-chat-scroll::-webkit-scrollbar-thumb {
                    border-radius: 999px;
                    background: #aeb9c8;
                }

                .kwb-chat-scroll::-webkit-scrollbar-track {
                    background: transparent;
                }

                .kwb-chat-card {
                    box-sizing: border-box;
                    margin: 6px 2px;
                    border-radius: 13px;
                    animation: kwb-card-in 180ms ease-out;
                }

                @keyframes kwb-card-in {
                    from {
                        opacity: 0;
                        transform: translateY(4px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }

                .kwb-user-card {
                    margin-left: 7%;
                    padding: 13px 15px;
                    border: 1px solid #cddcff;
                    background:
                        linear-gradient(135deg, #edf4ff, #f7faff);
                    box-shadow:
                        0 5px 14px rgba(41, 112, 214, 0.07);
                }

                .kwb-model-card {
                    margin-right: 3%;
                    padding: 15px 17px;
                    border: 1px solid #dbe3ea;
                    background: #ffffff;
                    box-shadow:
                        0 7px 18px rgba(15, 23, 42, 0.07);
                }

                .kwb-agent-card {
                    margin-right: 2%;
                    padding: 14px 16px;
                    border: 1px solid #c7d2fe;
                    background:
                        linear-gradient(135deg, #ffffff, #f5f7ff);
                    box-shadow:
                        0 7px 18px rgba(67, 56, 202, 0.08);
                }

                .kwb-card-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 12px;
                    margin-bottom: 6px;
                }

                .kwb-card-identity {
                    display: flex;
                    align-items: center;
                    gap: 9px;
                    min-width: 0;
                }

                .kwb-avatar {
                    display: inline-grid;
                    place-items: center;
                    width: 30px;
                    height: 30px;
                    flex: 0 0 30px;
                    border-radius: 9px;
                    font-size: 15px;
                }

                .kwb-user-avatar {
                    background: #dbeafe;
                    color: #1d4ed8;
                }

                .kwb-model-avatar {
                    background: #dcfce7;
                    color: #15803d;
                }

                .kwb-agent-avatar {
                    background: #e0e7ff;
                    color: #4338ca;
                }

                .kwb-role {
                    color: var(--kwb-text);
                    font-size: 13px;
                    font-weight: 750;
                }

                .kwb-role-badge {
                    display: inline-block;
                    margin-left: 6px;
                    padding: 2px 7px;
                    border-radius: 999px;
                    background: #e0e7ff;
                    color: #3730a3;
                    font-size: 10px;
                    font-weight: 700;
                    vertical-align: middle;
                }

                .kwb-time {
                    margin-top: 1px;
                    color: #8a98aa;
                    font-size: 10px;
                }

                .kwb-chip {
                    display: inline-block;
                    margin: 0 6px 8px 0;
                    padding: 3px 8px;
                    border-radius: 999px;
                    background: #dbeafe;
                    color: #1e40af;
                    font-size: 11px;
                    font-weight: 650;
                }

                .kwb-user-body {
                    white-space: pre-wrap;
                    overflow-wrap: anywhere;
                    color: #26354a;
                    font-family:
                        ui-monospace,
                        SFMono-Regular,
                        Menlo,
                        Monaco,
                        Consolas,
                        monospace;
                    font-size: 12.5px;
                    line-height: 1.55;
                }

                .kwb-model-body {
                    overflow-wrap: anywhere;
                    color: #263142;
                    font-size: 13.5px;
                    line-height: 1.68;
                }

                .kwb-model-body h1,
                .kwb-model-body h2,
                .kwb-model-body h3,
                .kwb-model-body h4 {
                    margin-top: 1.25em;
                    margin-bottom: 0.55em;
                    color: #172033;
                    line-height: 1.3;
                }

                .kwb-model-body h1 {
                    font-size: 20px;
                }

                .kwb-model-body h2 {
                    font-size: 17px;
                }

                .kwb-model-body h3 {
                    font-size: 15px;
                }

                .kwb-model-body p {
                    margin: 0.65em 0;
                }

                .kwb-model-body ul,
                .kwb-model-body ol {
                    padding-left: 22px;
                }

                .kwb-model-body blockquote {
                    margin: 12px 0;
                    padding: 8px 13px;
                    border-left: 3px solid #8b8cec;
                    background: #f7f7ff;
                    color: #525f73;
                }

                .kwb-chat-card pre {
                    overflow-x: auto;
                    margin: 11px 0;
                    padding: 14px;
                    border: 1px solid #273449;
                    border-radius: 10px;
                    background: var(--kwb-code-bg);
                    color: #d8e1ee;
                    font-family:
                        ui-monospace,
                        SFMono-Regular,
                        Menlo,
                        Monaco,
                        Consolas,
                        monospace;
                    font-size: 12.5px;
                    line-height: 1.55;
                    box-shadow:
                        inset 0 1px 0 rgba(255, 255, 255, 0.04);
                }

                .kwb-chat-card code {
                    padding: 2px 5px;
                    border-radius: 4px;
                    background: #eef1f5;
                    color: #b42347;
                    font-family:
                        ui-monospace,
                        SFMono-Regular,
                        Menlo,
                        Monaco,
                        Consolas,
                        monospace;
                    font-size: 0.92em;
                }

                .kwb-chat-card pre code {
                    padding: 0;
                    background: transparent;
                    color: inherit;
                }

                .kwb-chat-card table {
                    width: 100%;
                    margin: 12px 0;
                    border-collapse: collapse;
                    border: 1px solid #dbe3ea;
                }

                .kwb-chat-card th,
                .kwb-chat-card td {
                    padding: 8px 10px;
                    border: 1px solid #dbe3ea;
                    text-align: left;
                }

                .kwb-chat-card th {
                    background: #f1f5f9;
                    font-weight: 700;
                }

                .kwb-empty {
                    padding: 55px 20px;
                    text-align: center;
                    color: #7a8799;
                }

                .kwb-empty-icon {
                    display: grid;
                    place-items: center;
                    width: 54px;
                    height: 54px;
                    margin: 0 auto 12px;
                    border: 1px solid #dbe3ea;
                    border-radius: 16px;
                    background: #ffffff;
                    box-shadow:
                        0 8px 22px rgba(15, 23, 42, 0.07);
                    font-size: 25px;
                }

                .kwb-empty-title {
                    color: #334155;
                    font-size: 14px;
                    font-weight: 750;
                }

                .kwb-empty-copy {
                    margin-top: 5px;
                    font-size: 12px;
                }

                @media (max-width: 780px) {
                    .kwb-header {
                        padding: 19px;
                    }

                    .kwb-user-card,
                    .kwb-model-card,
                    .kwb-agent-card {
                        margin-left: 0;
                        margin-right: 0;
                    }

                    .kwb-status-wrap {
                        width: 100%;
                        margin-left: 0;
                    }
                }

                @media (prefers-color-scheme: dark) {
                    :root {
                        --kwb-surface: #182235;
                        --kwb-surface-soft: #202c40;
                        --kwb-border: #334155;
                        --kwb-border-strong: #475569;
                        --kwb-text: #e5ebf4;
                        --kwb-muted: #9eacbd;
                        --kwb-shadow:
                            0 12px 30px rgba(0, 0, 0, 0.22);
                    }

                    .kwb-prompt textarea {
                        border-color: #42516a !important;
                        background: #131d2e !important;
                        color: #e5ebf4 !important;
                    }

                    .kwb-superagent-panel {
                        border-color: #3730a3;
                        background:
                            linear-gradient(
                                135deg,
                                rgba(30, 27, 75, 0.55),
                                rgba(24, 34, 53, 0.85)
                            );
                    }

                    .kwb-chat-shell {
                        background:
                            linear-gradient(180deg, #131d2e, #101827);
                    }

                    .kwb-user-card {
                        border-color: #2e4e7e;
                        background:
                            linear-gradient(135deg, #182a45, #172235);
                    }

                    .kwb-model-card {
                        border-color: #344257;
                        background: #182235;
                    }

                    .kwb-agent-card {
                        border-color: #4338ca;
                        background:
                            linear-gradient(135deg, #1b243b, #1e1b4b);
                    }

                    .kwb-role,
                    .kwb-model-body,
                    .kwb-model-body h1,
                    .kwb-model-body h2,
                    .kwb-model-body h3,
                    .kwb-model-body h4,
                    .kwb-user-body {
                        color: #e2e8f0;
                    }

                    .kwb-chat-card code {
                        background: #29364a;
                        color: #ff91ad;
                    }

                    .kwb-model-body blockquote {
                        background: #202b42;
                        color: #b7c2d1;
                    }

                    .kwb-chat-card th {
                        background: #202c40;
                    }

                    .kwb-chat-card table,
                    .kwb-chat-card th,
                    .kwb-chat-card td {
                        border-color: #3b4a60;
                    }

                    .kwb-copy-widget-btn {
                        border-color: #42516a !important;
                        background: #202c40 !important;
                        color: #cbd5e1 !important;
                    }

                    .kwb-empty-icon {
                        border-color: #334155;
                        background: #182235;
                    }

                    .kwb-empty-title {
                        color: #dbe4ef;
                    }

                    .kwb-generated-files {
                        border-color: #344f76;
                        background: #182a45;
                        color: #cbd5e1;
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
                        <div class="kwb-logo">✦</div>
                        <div>
                            <div class="kwb-title">
                                Kaggle LLM Workbench v6
                            </div>
                            <div class="kwb-subtitle">
                                Prompt, attach files, execute generated code, or
                                launch an autonomous Multi-Model SuperAgent team
                                directly inside your notebook workspace.
                            </div>
                        </div>
                    </div>

                    <div class="kwb-header-badges">
                        <span class="kwb-header-badge">
                            ◉ {model_count} models
                        </span>
                        <span class="kwb-header-badge">
                            🤖 SuperAgent v6
                        </span>
                        <span class="kwb-header-badge">
                            📁 {safe_work_dir}
                        </span>
                        <span class="kwb-header-badge">
                            Markdown: {markdown_badge}
                        </span>
                    </div>
                </div>
            </div>
            """
        )

        # Manual model selector.
        self.model_dropdown = widgets.Dropdown(
            options=self.available_models,
            value=self.default_model,
            description="Model",
            style={"description_width": "55px"},
            layout=widgets.Layout(
                width="470px",
                max_width="100%",
                min_width="280px",
            ),
        )

        # Temperature.
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

        # Conversation memory.
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

        # System prompt.
        self.sys_prompt_input = widgets.Textarea(
            value=(
                "You are an elite coding assistant and software architect. "
                "Analyze, review, and optimize code provided by the user. "
                "Maintain full context across the conversation."
            ),
            layout=widgets.Layout(width="100%", height="90px"),
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
                    value="""
                    <div style="
                        margin-top:6px;
                        color:var(--kwb-muted);
                        font-size:10.5px;
                        font-weight:750;
                        letter-spacing:.07em;
                        text-transform:uppercase;
                    ">
                        System instruction
                    </div>
                    """
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
        self.settings_accordion.set_title(
            0,
            "⚙ Advanced configuration and conversation memory",
        )
        self.settings_accordion.selected_index = None

        # SuperAgent Orchestrator UI.
        self._init_superagent_ui()

        # Main input.
        self.input_header = widgets.HTML(
            value="""
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                gap:12px;
                margin-bottom:9px;
                flex-wrap:wrap;
            ">
                <div>
                    <div class="kwb-section-title">
                        Prompt / SuperAgent Goal
                    </div>
                    <div class="kwb-section-subtitle">
                        Send a single prompt to the active model, or click
                        <b>Launch SuperAgent</b> to let the multi-model team
                        plan, code, execute, and verify autonomously.
                    </div>
                </div>

                <div style="
                    color:var(--kwb-muted);
                    font-size:10px;
                    font-weight:650;
                ">
                    Supports fenced code and %%writefile exports
                </div>
            </div>
            """
        )

        self.textarea = widgets.Textarea(
            value="",
            placeholder=(
                "Ask a question, paste code, or describe a complete "
                "SuperAgent goal...\n\n"
                "Example SuperAgent Goal:\n"
                "Build a Python module `prime_sieve.py` with a fast prime "
                "generator and unit tests, execute it to verify all tests "
                "pass, and benchmark it."
            ),
            layout=widgets.Layout(width="100%", height="165px"),
        )
        self.textarea.add_class("kwb-prompt")

        # Action buttons.
        self.btn_send = widgets.Button(
            description="Send Prompt",
            button_style="primary",
            icon="paper-plane",
            tooltip="Send a single prompt to the active model",
            layout=widgets.Layout(width="145px", height="38px"),
        )

        self.btn_superagent = widgets.Button(
            description="Launch SuperAgent",
            button_style="warning",
            icon="rocket",
            tooltip=(
                "Run the autonomous multi-model Orchestrator loop "
                "(Plan -> Code -> Execute -> Critic -> Finish)"
            ),
            layout=widgets.Layout(width="180px", height="38px"),
        )

        self.btn_stop_agent = widgets.Button(
            description="Stop Agent",
            button_style="danger",
            icon="stop",
            tooltip="Request the SuperAgent loop to stop after the current step",
            layout=widgets.Layout(width="120px", height="38px"),
            disabled=True,
        )

        self.btn_undo = widgets.Button(
            description="Undo Last",
            button_style="info",
            icon="undo",
            tooltip="Remove the latest conversation turn",
            layout=widgets.Layout(width="120px", height="38px"),
            disabled=True,
        )

        self.btn_copy_last = widgets.Button(
            description="Copy Last",
            button_style="info",
            icon="copy",
            tooltip="Copy the latest model/agent response to the clipboard",
            layout=widgets.Layout(width="120px", height="38px"),
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
            tooltip="Clear the conversation",
            layout=widgets.Layout(width="110px", height="38px"),
        )

        self.status_label = widgets.HTML(
            value=(
                "<div class='kwb-status' style='color:#64748b;'>"
                "● Ready for a prompt or SuperAgent goal"
                "</div>"
            ),
            layout=widgets.Layout(min_width="230px"),
        )
        self.status_label.add_class("kwb-status-wrap")

        # Generated file links.
        self.download_links_box = widgets.VBox(
            layout=widgets.Layout(width="100%", margin="0"),
        )

        # Execution buttons.
        self.exec_buttons_box = widgets.HBox(
            layout=widgets.Layout(
                width="100%",
                margin="0",
                flex_wrap="wrap",
            ),
        )
        self.exec_buttons_box.add_class("kwb-toolbar")

        # Execution console.
        self.exec_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                max_height="390px",
                overflow_y="auto",
                padding="12px",
            ),
        )
        self.exec_output.add_class("kwb-console-output")

        self.btn_feed_output = widgets.Button(
            description="Send Output/Error to AI",
            icon="reply",
            button_style="warning",
            tooltip="Append the latest execution log to the prompt",
            layout=widgets.Layout(width="220px", height="34px"),
        )

        self.btn_clear_console = widgets.Button(
            description="Hide Console",
            icon="eye-slash",
            tooltip="Hide the execution console",
            layout=widgets.Layout(width="135px", height="34px"),
        )

        console_actions = widgets.HBox(
            [self.btn_feed_output, self.btn_clear_console],
            layout=widgets.Layout(margin="4px 0 0 0", flex_wrap="wrap"),
        )
        console_actions.add_class("kwb-toolbar")

        self.exec_container = widgets.VBox(
            [
                widgets.HTML(
                    value="""
                    <div class="kwb-console-title">
                        ⚡ Execution console
                    </div>
                    """
                ),
                self.exec_output,
                console_actions,
            ],
            layout=widgets.Layout(display="none", width="100%"),
        )
        self.exec_container.add_class("kwb-console")

        # Chat container using native VBox so per-card ipywidgets Buttons work
        # without being stripped by HTML sanitization.
        self.chat_empty_html = widgets.HTML(
            value=self._get_empty_chat_html()
        )

        self.chat_container = widgets.VBox(
            [self.chat_empty_html],
            layout=widgets.Layout(
                width="100%",
                max_height="720px",
                overflow_y="auto",
                padding="13px",
            ),
        )
        self.chat_container.add_class("kwb-chat-shell")
        self.chat_container.add_class("kwb-chat-scroll")

        # File attachments.
        self._init_attachment_ui()

    def _init_superagent_ui(self):
        self.orchestrator_dropdown = widgets.Dropdown(
            options=self.available_models,
            value=self.default_model,
            description="👑 Orchestrator",
            style={"description_width": "110px"},
            layout=widgets.Layout(width="380px", max_width="100%"),
        )

        self.coder_dropdown = widgets.Dropdown(
            options=self.available_models,
            value=self.default_coder_model,
            description="🛠️ Coder",
            style={"description_width": "110px"},
            layout=widgets.Layout(width="380px", max_width="100%"),
        )

        self.critic_dropdown = widgets.Dropdown(
            options=self.available_models,
            value=self.default_critic_model,
            description="🧪 Critic / QA",
            style={"description_width": "110px"},
            layout=widgets.Layout(width="380px", max_width="100%"),
        )

        self.agent_max_steps_slider = widgets.IntSlider(
            value=6,
            min=2,
            max=15,
            step=1,
            description="Max steps",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="280px", max_width="100%"),
        )

        self.agent_auto_exec_checkbox = widgets.Checkbox(
            value=True,
            description="Allow autonomous code execution (.py / .sh)",
            indent=False,
            layout=widgets.Layout(width="auto"),
        )

        agent_header = widgets.HTML(
            value="""
            <div style="margin-bottom:6px;">
                <div class="kwb-section-title">
                    🤖 Autonomous SuperAgent Orchestrator (Multi-Model Swarm)
                </div>
                <div class="kwb-section-subtitle">
                    Assign specialized frontier models to collaborate. The
                    <b>Orchestrator</b> plans tasks, delegates coding to the
                    <b>Coder</b>, runs scripts in the live kernel, consults the
                    <b>Critic / QA</b> on failures, and verifies the final goal.
                </div>
            </div>
            """
        )

        role_row = widgets.HBox(
            [
                self.orchestrator_dropdown,
                self.coder_dropdown,
                self.critic_dropdown,
            ],
            layout=widgets.Layout(
                width="100%",
                flex_flow="row wrap",
                gap="8px",
            ),
        )

        options_row = widgets.HBox(
            [
                self.agent_max_steps_slider,
                self.agent_auto_exec_checkbox,
            ],
            layout=widgets.Layout(
                width="100%",
                flex_flow="row wrap",
                gap="14px",
                align_items="center",
            ),
        )

        superagent_box = widgets.VBox(
            [agent_header, role_row, options_row],
            layout=widgets.Layout(width="100%", gap="8px"),
        )
        superagent_box.add_class("kwb-settings-content")

        self.superagent_accordion = widgets.Accordion(
            children=[superagent_box],
            layout=widgets.Layout(width="100%"),
        )
        self.superagent_accordion.set_title(
            0,
            "🤖 SuperAgent Team Configuration (Orchestrator • Coder • Critic)",
        )
        self.superagent_accordion.selected_index = 0

    def _bind_events(self):
        self.btn_send.on_click(self.on_send_clicked)
        self.btn_superagent.on_click(self.on_superagent_clicked)
        self.btn_stop_agent.on_click(self.on_stop_agent_clicked)
        self.btn_undo.on_click(self.on_undo_clicked)
        self.btn_copy_last.on_click(self.on_copy_last_clicked)
        self.btn_save.on_click(self.on_save_clicked)
        self.btn_reset.on_click(self.on_reset_clicked)
        self.btn_feed_output.on_click(self._on_feed_output_clicked)
        self.btn_clear_console.on_click(
            lambda _: setattr(
                self.exec_container.layout,
                "display",
                "none",
            )
        )

        self._bind_attachment_events()

    # ==========================================
    # 2. CORE LOGIC AND MANUAL PROMPTING
    # ==========================================
    def _build_context(self, new_user_input: str, new_attachments: list) -> str:
        sys_prompt = self.sys_prompt_input.value.strip()

        context_parts = (
            [f"System:\n{sys_prompt}\n"]
            if sys_prompt
            else []
        )

        max_messages = self.max_turns_slider.value * 2

        recent_history = (
            self.chat_history[-max_messages:]
            if max_messages > 0
            else self.chat_history
        )

        for message in recent_history:
            if message["role"] == "user":
                role_string = "User"
                body = self._compose_prompt(
                    message["text"],
                    message.get("attachments", []),
                )
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
            self._set_status(
                "⚠ Please enter a prompt or attach a file first.",
                "#d93025",
            )
            return

        self._toggle_controls(disabled=True)

        chosen_model = self.model_dropdown.value
        safe_model = html.escape(str(chosen_model))
        temperature_text = f"{self.temp_slider.value:.2f}"

        self._set_status(
            f"⏳ Querying <b>{safe_model}</b> at temperature "
            f"{temperature_text}...",
            "#d97706",
        )

        try:
            context = self._build_context(typed_text, attachments)

            llm = kbench.llms[chosen_model]
            raw_response = llm.prompt(
                context,
                temperature=self.temp_slider.value,
            )
            response_text = str(raw_response)

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            self.chat_history.append(
                {
                    "role": "user",
                    "text": typed_text,
                    "attachments": attachments,
                    "time": timestamp,
                }
            )

            self.chat_history.append(
                {
                    "role": "model",
                    "text": response_text,
                    "model_name": chosen_model,
                    "time": timestamp,
                }
            )

            self.textarea.value = ""
            self.attachments = []
            self._refresh_attachment_view()

            self.refresh_chat_view()
            self.save_chat_to_disk()

            self._set_status(
                f"✓ Response received at {timestamp} and auto-saved.",
                "#16805a",
            )

        except Exception as exc:
            self._set_status(
                f"✕ Error: {html.escape(str(exc))}",
                "#c62828",
            )

        finally:
            self._toggle_controls(disabled=False)

    # ==========================================
    # 2B. AUTONOMOUS SUPERAGENT ORCHESTRATOR
    # ==========================================
    def on_stop_agent_clicked(self, _):
        self._agent_stop_requested = True
        self._set_status(
            "⏹ Stop requested. Halting SuperAgent after current step...",
            "#d93025",
        )

    def _parse_orchestrator_json(self, raw_text: str) -> dict:
        """Extracts a structured JSON decision from the Orchestrator's output."""
        cleaned = raw_text.strip()

        # 1. Try fenced JSON block first.
        fenced_matches = re.findall(
            rf"{CODE_FENCE}(?:json)?[ \t]*\r?\n(.*?){CODE_FENCE}",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        candidates = list(fenced_matches)

        # 2. Try locating outermost '{' ... '}' in raw text.
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

        # Fallback if the model replied in plain text instead of JSON.
        return {
            "thought": "Orchestrator responded directly without JSON envelope.",
            "action": "FINISH",
            "final_answer": raw_text,
        }

    def _append_agent_card(
        self,
        agent_role: str,
        model_name: str,
        text: str,
        badge: str = "",
    ):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.chat_history.append(
            {
                "role": "model",
                "agent_role": agent_role,
                "agent_badge": badge,
                "model_name": model_name,
                "text": text,
                "time": timestamp,
            }
        )
        self.refresh_chat_view()

    def on_superagent_clicked(self, _):
        goal_text = self.textarea.value.strip()
        attachments = list(self.attachments)

        if not goal_text and not attachments:
            self._set_status(
                "⚠ Please enter a SuperAgent goal or attach a file first.",
                "#d93025",
            )
            return

        self._agent_stop_requested = False
        self._toggle_controls(disabled=True)
        self.btn_stop_agent.disabled = False

        orch_model = self.orchestrator_dropdown.value
        coder_model = self.coder_dropdown.value
        critic_model = self.critic_dropdown.value
        max_steps = self.agent_max_steps_slider.value
        allow_exec = self.agent_auto_exec_checkbox.value
        temp = self.temp_slider.value

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        composed_goal = self._compose_prompt(goal_text, attachments)

        # Record user's initial goal card.
        self.chat_history.append(
            {
                "role": "user",
                "text": f"🚀 [SuperAgent Goal]\n{goal_text}",
                "attachments": attachments,
                "time": timestamp,
            }
        )
        self.textarea.value = ""
        self.attachments = []
        self._refresh_attachment_view()
        self.refresh_chat_view()

        agent_memory = [
            f"USER GOAL:\n{composed_goal}\n",
            f"WORKSPACE DIRECTORY: {self.work_dir}\n",
        ]

        try:
            for step in range(1, max_steps + 1):
                if self._agent_stop_requested:
                    self._append_agent_card(
                        agent_role="👑 Orchestrator",
                        model_name=orch_model,
                        badge=f"Stopped at Step {step}/{max_steps}",
                        text="⏹ **SuperAgent execution stopped by user.**",
                    )
                    break

                self._set_status(
                    f"👑 <b>Step {step}/{max_steps}:</b> Orchestrator "
                    f"(<b>{html.escape(orch_model)}</b>) is planning...",
                    "#5856d6",
                )

                workspace_files = self._list_workspace_files()
                exec_policy_note = (
                    "RUN_FILE is ENABLED. Prefer running generated Python/Bash "
                    "scripts to verify correctness before finishing."
                    if allow_exec
                    else "RUN_FILE is DISABLED by user. Use ASK_CRITIC to "
                    "verify code instead of executing it."
                )

                orch_prompt = (
                    "You are the Autonomous Master Orchestrator managing a "
                    "team of specialized AI agents inside a live Python/Linux "
                    "workspace.\n\n"
                    f"CURRENT WORKSPACE FILES: {workspace_files}\n"
                    f"EXECUTION POLICY: {exec_policy_note}\n"
                    f"CURRENT STEP: {step} of {max_steps}\n\n"
                    "HISTORY OF ACTIONS & OBSERVATIONS SO FAR:\n"
                    + "\n---\n".join(agent_memory[-10:])
                    + "\n\n"
                    "Decide the next single step to achieve the USER GOAL. "
                    "Respond ONLY with a valid JSON object matching this exact "
                    "schema:\n"
                    "{\n"
                    '  "thought": "Brief analysis of progress and what is needed next",\n'
                    '  "action": "DELEGATE_CODER" | "RUN_FILE" | "ASK_CRITIC" | "FINISH",\n'
                    '  "target_file": "filename.py (required for DELEGATE_CODER or RUN_FILE)",\n'
                    '  "instruction": "Detailed instructions for Coder or Critic",\n'
                    '  "final_answer": "Comprehensive final summary for the user (only when action is FINISH)"\n'
                    "}"
                )

                orch_raw = str(
                    kbench.llms[orch_model].prompt(
                        orch_prompt,
                        temperature=temp,
                    )
                )
                decision = self._parse_orchestrator_json(orch_raw)

                action = str(decision.get("action", "FINISH")).upper().strip()
                thought = decision.get("thought", "")
                target_file = (
                    os.path.basename(str(decision.get("target_file") or "").strip())
                    or "solution.py"
                )
                instruction = decision.get("instruction", "")
                final_answer = decision.get("final_answer", "")

                if action == "FINISH":
                    summary_md = (
                        f"**Thought:** {thought}\n\n"
                        f"{final_answer or instruction or orch_raw}"
                    )
                    self._append_agent_card(
                        agent_role="🏁 Orchestrator (Goal Complete)",
                        model_name=orch_model,
                        badge=f"Step {step}/{max_steps} • FINISH",
                        text=summary_md,
                    )
                    self.save_chat_to_disk()
                    self._set_status(
                        f"✓ SuperAgent completed goal in {step} step(s).",
                        "#16805a",
                    )
                    return

                # Show Orchestrator decision card.
                plan_md = (
                    f"**Thought:** {thought}\n\n"
                    f"- **Next Action:** `{action}`\n"
                    + (
                        f"- **Target File:** `{target_file}`\n"
                        if target_file
                        else ""
                    )
                    + (
                        f"- **Instruction:** {instruction}"
                        if instruction
                        else ""
                    )
                )
                self._append_agent_card(
                    agent_role="👑 Orchestrator",
                    model_name=orch_model,
                    badge=f"Step {step}/{max_steps} • {action}",
                    text=plan_md,
                )

                if self._agent_stop_requested:
                    break

                # ------------------------------------------
                # ACTION 1: DELEGATE TO LEAD CODER
                # ------------------------------------------
                if action == "DELEGATE_CODER":
                    self._set_status(
                        f"🛠️ <b>Step {step}/{max_steps}:</b> Lead Coder "
                        f"(<b>{html.escape(coder_model)}</b>) writing "
                        f"<code>{html.escape(target_file)}</code>...",
                        "#d97706",
                    )

                    existing_code_context = ""
                    existing_path = self.work_dir / target_file
                    if existing_path.is_file():
                        try:
                            existing_text = self._read_workspace_file(target_file)
                            fence = self._fence_for(existing_text)
                            existing_code_context = (
                                f"\nEXISTING CONTENT OF `{target_file}`:\n"
                                f"{fence}python\n{existing_text}\n{fence}\n"
                            )
                        except ValueError:
                            pass

                    coder_prompt = (
                        "You are the Lead Software Engineer in an autonomous "
                        "AI team.\n\n"
                        f"USER GOAL:\n{composed_goal}\n\n"
                        f"ORCHESTRATOR INSTRUCTION:\n{instruction}\n"
                        f"{existing_code_context}\n"
                        f"Write complete, self-contained, production-ready code. "
                        f"You MUST start your main code block with "
                        f"`%%writefile {target_file}` on the first line inside "
                        f"the fenced code block so it can be saved and executed "
                        f"automatically."
                    )

                    coder_response = str(
                        kbench.llms[coder_model].prompt(
                            coder_prompt,
                            temperature=temp,
                        )
                    )

                    self._append_agent_card(
                        agent_role="🛠️ Lead Coder",
                        model_name=coder_model,
                        badge=f"Step {step}/{max_steps} • {target_file}",
                        text=coder_response,
                    )

                    saved = self._extract_and_save_blocks(
                        coder_response,
                        default_filename=target_file,
                    )
                    self.save_chat_to_disk()

                    agent_memory.append(
                        f"STEP {step} [DELEGATE_CODER -> {coder_model}]:\n"
                        f"Instruction: {instruction}\n"
                        f"Saved files: {saved}\n"
                        f"Coder response:\n{coder_response[-2500:]}"
                    )

                # ------------------------------------------
                # ACTION 2: RUN FILE IN KERNEL SANDBOX
                # ------------------------------------------
                elif action == "RUN_FILE":
                    if not allow_exec:
                        obs = (
                            "Autonomous execution is disabled by user "
                            "checkbox. Use ASK_CRITIC or FINISH."
                        )
                        agent_memory.append(f"STEP {step} [RUN_FILE]: {obs}")
                        continue

                    self._set_status(
                        f"⚡ <b>Step {step}/{max_steps}:</b> Executing "
                        f"<code>{html.escape(target_file)}</code> in kernel...",
                        "#d97706",
                    )

                    success, exec_log = self.execute_file(target_file)
                    status_Badge = "PASS ✓" if success else "FAIL ✕"

                    exec_card_md = (
                        f"Executed `{target_file}` -> **{status_Badge}**\n\n"
                        f"{CODE_FENCE}text\n{exec_log[-3000:]}\n{CODE_FENCE}"
                    )
                    self._append_agent_card(
                        agent_role="⚡ Kernel Sandbox Executor",
                        model_name="Local IPython Kernel",
                        badge=f"Step {step}/{max_steps} • {status_Badge}",
                        text=exec_card_md,
                    )

                    agent_memory.append(
                        f"STEP {step} [RUN_FILE `{target_file}` -> {status_Badge}]:\n"
                        f"{exec_log[-3000:]}"
                    )

                # ------------------------------------------
                # ACTION 3: ASK CRITIC / QA REVIEWER
                # ------------------------------------------
                elif action == "ASK_CRITIC":
                    self._set_status(
                        f"🧪 <b>Step {step}/{max_steps}:</b> Critic/QA "
                        f"(<b>{html.escape(critic_model)}</b>) reviewing...",
                        "#2563b8",
                    )

                    file_snapshot = ""
                    if target_file and (self.work_dir / target_file).is_file():
                        try:
                            content = self._read_workspace_file(target_file)
                            fence = self._fence_for(content)
                            file_snapshot = (
                                f"\nCURRENT FILE `{target_file}`:\n"
                                f"{fence}\n{content}\n{fence}\n"
                            )
                        except ValueError:
                            pass

                    critic_prompt = (
                        "You are the Principal QA & Code Critic in an "
                        "autonomous AI engineering team.\n\n"
                        f"USER GOAL:\n{composed_goal}\n\n"
                        f"ORCHESTRATOR QUESTION / FOCUS:\n{instruction}\n"
                        f"{file_snapshot}\n"
                        "RECENT EXECUTION & TEAM LOGS:\n"
                        + "\n---\n".join(agent_memory[-4:])
                        + "\n\nProvide a concise, actionable root-cause analysis "
                        "and exact code fixes needed."
                    )

                    critic_response = str(
                        kbench.llms[critic_model].prompt(
                            critic_prompt,
                            temperature=temp,
                        )
                    )

                    self._append_agent_card(
                        agent_role="🧪 Critic / QA Reviewer",
                        model_name=critic_model,
                        badge=f"Step {step}/{max_steps} • Review",
                        text=critic_response,
                    )

                    agent_memory.append(
                        f"STEP {step} [ASK_CRITIC -> {critic_model}]:\n"
                        f"{critic_response[-2500:]}"
                    )

                else:
                    # Unknown action fallback: treat as finish.
                    self._append_agent_card(
                        agent_role="🏁 Orchestrator",
                        model_name=orch_model,
                        badge=f"Step {step}/{max_steps}",
                        text=orch_raw,
                    )
                    break

            # If max_steps reached without explicit FINISH, request final synthesis.
            if not self._agent_stop_requested:
                self._set_status(
                    "👑 Generating final SuperAgent summary...",
                    "#5856d6",
                )
                synth_prompt = (
                    "You are the Master Orchestrator. The maximum number of "
                    "autonomous steps has been reached. Summarize what was "
                    "built, tested, and verified for the user based on the "
                    "session log below:\n\n"
                    + "\n---\n".join(agent_memory[-8:])
                )
                final_summary = str(
                    kbench.llms[orch_model].prompt(
                        synth_prompt,
                        temperature=temp,
                    )
                )
                self._append_agent_card(
                    agent_role="🏁 Orchestrator (Final Summary)",
                    model_name=orch_model,
                    badge="Max Steps Reached",
                    text=final_summary,
                )
                self.save_chat_to_disk()
                self._set_status(
                    "✓ SuperAgent finished (max steps reached).",
                    "#16805a",
                )

        except Exception as exc:
            self._set_status(
                f"✕ SuperAgent Error: {html.escape(str(exc))}",
                "#c62828",
            )

        finally:
            self.btn_stop_agent.disabled = True
            self._toggle_controls(disabled=False)

    # ==========================================
    # 2C. UNDO, COPY, RESET
    # ==========================================
    def on_undo_clicked(self, _):
        if len(self.chat_history) >= 2:
            self.chat_history.pop()
            if (
                self.chat_history
                and self.chat_history[-1]["role"] == "user"
            ):
                last_user_message = self.chat_history.pop()
                self.textarea.value = last_user_message["text"]
                self.attachments = list(
                    last_user_message.get("attachments", [])
                )
                self._refresh_attachment_view()

            self.refresh_chat_view()

            self._set_status(
                "↩ Last turn removed.",
                "#2563b8",
            )

        has_turns = len(self.chat_history) >= 2
        self.btn_undo.disabled = not has_turns
        self.btn_copy_last.disabled = not has_turns

    def on_copy_last_clicked(self, button):
        for entry in reversed(self.chat_history):
            if entry["role"] == "model":
                self._copy_text_to_clipboard(
                    entry["text"],
                    label="Latest response",
                    button=button,
                )
                return

        self._set_status("⚠ No model response to copy yet.", "#d97706")

    def on_save_clicked(self, _):
        if not self.chat_history:
            self._set_status(
                "⚠ Nothing is available to export yet.",
                "#d97706",
            )
            return

        self.save_chat_to_disk()

        safe_directory = html.escape(str(self.work_dir))

        self._set_status(
            f"💾 Files exported to {safe_directory}.",
            "#2563b8",
        )

    def on_reset_clicked(self, _):
        self.chat_history.clear()
        self._last_exec_log = ""
        self._saved_files = []
        self.attachments = []
        self._refresh_attachment_view()

        self.refresh_chat_view()

        self.download_links_box.children = []
        self.exec_buttons_box.children = []
        self.exec_container.layout.display = "none"
        self.exec_output.clear_output(wait=True)

        self.btn_undo.disabled = True
        self.btn_copy_last.disabled = True

        self._set_status("Chat history cleared.", "#64748b")

    # ==========================================
    # 3. DISK EXPORT, EXTRACTION AND EXECUTION
    # ==========================================
    def _extract_code_blocks(self, text: str) -> list:
        # Matches fenced blocks with optional language tag.
        # Supports both \n and \r\n line endings.
        return re.findall(
            rf"{CODE_FENCE}([a-zA-Z0-9_+\-]*)[ \t]*\r?\n(.*?){CODE_FENCE}",
            text,
            flags=re.DOTALL,
        )

    def _extract_and_save_blocks(
        self,
        text: str,
        default_filename: str = None,
    ) -> list:
        code_blocks = self._extract_code_blocks(text)
        newly_saved = []

        for index, (language, block) in enumerate(code_blocks):
            lines = block.strip().splitlines()
            normalized_language = language.lower().strip()
            extension = EXTENSION_MAP.get(normalized_language, "txt")

            if lines and lines[0].strip().startswith("%%writefile"):
                raw_filename = (
                    lines[0]
                    .replace("%%writefile", "", 1)
                    .strip()
                )
                safe_filename = os.path.basename(raw_filename)
                if not safe_filename:
                    safe_filename = (
                        default_filename
                        or f"extracted_{index + 1}.{extension}"
                    )
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
            "model": self.model_dropdown.value,
            "orchestrator_model": self.orchestrator_dropdown.value,
            "coder_model": self.coder_dropdown.value,
            "critic_model": self.critic_dropdown.value,
            "system_prompt": self.sys_prompt_input.value,
            "temperature": self.temp_slider.value,
            "last_response": last_answer,
            "extracted_code_blocks": [
                {
                    "lang": language or "txt",
                    "code": code,
                }
                for language, code in code_blocks
            ],
            "full_history": self.chat_history,
        }

        json_path = self.work_dir / "argon_output.json"

        with open(json_path, "w", encoding="utf-8") as output_file:
            json.dump(
                export_payload,
                output_file,
                ensure_ascii=False,
                indent=2,
            )

        self._extract_and_save_blocks(last_answer)

        saved_files = ["argon_output.json"] + [
            f for f in self._saved_files if f != "argon_output.json"
        ]

        links = []

        for filename in saved_files:
            safe_href = html.escape(filename, quote=True)
            safe_label = html.escape(filename)

            links.append(
                f"<a href='{safe_href}' target='_blank'>"
                f"📄 {safe_label}</a>"
            )

        links_html = (
            " <span style='opacity:.35;'>•</span> "
        ).join(links)

        self.download_links_box.children = [
            widgets.HTML(
                value=(
                    "<div class='kwb-generated-files'>"
                    "<strong>Generated files</strong>"
                    "<div style='margin-top:3px;'>"
                    f"{links_html}"
                    "</div>"
                    "</div>"
                )
            )
        ]

        # Create execution and copy buttons for saved files.
        action_buttons = []

        for filename in saved_files:
            if filename.endswith((".py", ".sh")):
                run_button = widgets.Button(
                    description=f"Run {filename}",
                    button_style="success",
                    icon="play",
                    tooltip=(
                        f"Execute {filename} in the current "
                        "notebook session"
                    ),
                    layout=widgets.Layout(width="auto", height="34px"),
                )

                run_button.on_click(
                    lambda _, selected_file=filename:
                    self.execute_file(selected_file)
                )

                action_buttons.append(run_button)

            if filename != "argon_output.json":
                copy_file_btn = widgets.Button(
                    description=f"Copy {filename}",
                    icon="copy",
                    tooltip=f"Copy contents of {filename} to clipboard",
                    layout=widgets.Layout(width="auto", height="34px"),
                )
                copy_file_btn.add_class("kwb-copy-widget-btn")
                copy_file_btn.on_click(
                    lambda btn, selected_file=filename:
                    self._copy_workspace_file(selected_file, btn)
                )
                action_buttons.append(copy_file_btn)

        self.exec_buttons_box.children = action_buttons

        # Keep the workspace dropdown in sync with new exports.
        self.attach_select.options = self._list_workspace_files()

    def _copy_workspace_file(self, filename: str, button=None):
        file_path = self.work_dir / filename
        if not file_path.is_file():
            self._set_status(
                f"⚠ File {html.escape(filename)} not found.",
                "#d93025",
            )
            return

        content = file_path.read_text(encoding="utf-8")
        self._copy_text_to_clipboard(content, label=filename, button=button)

    def execute_file(self, filename: str) -> tuple:
        """Executes a workspace file and returns (success_bool, log_string)."""
        file_path = self.work_dir / filename

        if not file_path.exists():
            msg = f"File '{filename}' was not found in {self.work_dir}."
            self._set_status(f"⚠ {html.escape(msg)}", "#d93025")
            return False, msg

        code = file_path.read_text(encoding="utf-8")

        self.exec_container.layout.display = "flex"
        self.exec_output.clear_output(wait=True)

        safe_filename = html.escape(filename)

        self._set_status(
            f"⚡ Executing <b>{safe_filename}</b>...",
            "#d97706",
        )

        ipython = get_ipython()
        execution_successful = True
        captured = None

        with self.exec_output:
            print(f"┌─ Executing: {filename}")
            print(f"└─ Path: {file_path}\n")

            with capture_output(
                stdout=True,
                stderr=True,
                display=True,
            ) as captured:
                try:
                    if filename.endswith(".sh"):
                        if ipython is None:
                            raise RuntimeError(
                                "A live IPython session is required "
                                "to execute shell scripts."
                            )

                        quoted_path = shlex.quote(str(file_path))
                        result = ipython.system(f"bash {quoted_path}")

                        if isinstance(result, int) and result != 0:
                            execution_successful = False

                    else:
                        if ipython is not None:
                            result = ipython.run_cell(code)

                            if result.error_in_exec is not None:
                                execution_successful = False

                            if result.error_before_exec is not None:
                                execution_successful = False

                        else:
                            execution_globals = {
                                "__name__": "__main__",
                                "__file__": str(file_path),
                            }

                            exec(
                                compile(code, str(file_path), "exec"),
                                execution_globals,
                                execution_globals,
                            )

                except Exception:
                    traceback.print_exc()
                    execution_successful = False

            # Re-display captured text, tables and plots.
            captured.show()

        log_parts = []

        if captured is not None and captured.stdout:
            log_parts.append("STDOUT:\n" f"{captured.stdout.strip()}")

        if captured is not None and captured.stderr:
            log_parts.append("STDERR / TRACEBACK:\n" f"{captured.stderr.strip()}")

        if log_parts:
            log_string = "\n\n".join(log_parts)
        else:
            log_string = "Code executed silently with no stdout/stderr output."

        self._last_exec_log = (
            f"I executed `{filename}` and got the following output:\n"
            f"{CODE_FENCE}text\n{log_string[-4000:]}\n{CODE_FENCE}"
        )

        if execution_successful:
            self._set_status(
                f"✓ Execution of <b>{safe_filename}</b> finished.",
                "#16805a",
            )
        else:
            self._set_status(
                f"✕ Execution failed in <b>{safe_filename}</b>. "
                "Use “Send Output/Error to AI” to debug it.",
                "#c62828",
            )

        return execution_successful, log_string

    def _on_feed_output_clicked(self, _):
        if not self._last_exec_log:
            self._set_status(
                "⚠ No execution output is available yet.",
                "#d97706",
            )
            return

        current_text = self.textarea.value.strip()

        prefix = f"{current_text}\n\n" if current_text else ""

        self.textarea.value = (
            f"{prefix}"
            f"{self._last_exec_log}\n\n"
            "Please analyze the result or fix any errors:"
        )

        self._set_status(
            "Execution log appended to the prompt box.",
            "#2563b8",
        )

    # ==========================================
    # 3B. FILE ATTACHMENTS
    # ==========================================
    def _init_attachment_ui(self):
        self.attachments = []

        self.attach_select = widgets.Dropdown(
            options=self._list_workspace_files(),
            description="Workspace",
            style={"description_width": "75px"},
            layout=widgets.Layout(width="340px", max_width="100%"),
        )

        self.btn_refresh_files = widgets.Button(
            description="Refresh",
            icon="refresh",
            tooltip="Rescan the working directory",
            layout=widgets.Layout(width="105px", height="34px"),
        )

        self.btn_attach_existing = widgets.Button(
            description="Attach file",
            button_style="info",
            icon="paperclip",
            tooltip="Attach the selected workspace file",
            layout=widgets.Layout(width="130px", height="34px"),
        )

        self.upload_widget = widgets.FileUpload(
            accept=",".join(TEXT_LANG_BY_EXT.keys()),
            multiple=True,
            description="Upload",
            layout=widgets.Layout(width="auto"),
        )

        self.btn_attach_upload = widgets.Button(
            description="Attach uploads",
            button_style="info",
            icon="upload",
            tooltip="Attach the uploaded files",
            layout=widgets.Layout(width="140px", height="34px"),
        )

        self.attachment_list = widgets.HTML()

        self.btn_clear_attachments = widgets.Button(
            description="Clear attachments",
            button_style="danger",
            icon="times",
            layout=widgets.Layout(width="150px", height="34px"),
            disabled=True,
        )

        self.attachment_panel = widgets.VBox(
            [
                widgets.HTML(
                    value=(
                        "<div class='kwb-section-title'>Attach files</div>"
                        "<div class='kwb-section-subtitle'>"
                        "Attached files are included with your next prompt or "
                        "SuperAgent goal."
                        "</div>"
                    )
                ),
                widgets.HBox(
                    [
                        self.attach_select,
                        self.btn_refresh_files,
                        self.btn_attach_existing,
                    ],
                    layout=widgets.Layout(flex_flow="row wrap", gap="8px"),
                ),
                widgets.HBox(
                    [self.upload_widget, self.btn_attach_upload],
                    layout=widgets.Layout(flex_flow="row wrap", gap="8px"),
                ),
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
        return [
            path.name
            for path in sorted(self.work_dir.iterdir())
            if path.is_file()
            and path.suffix.lower() in TEXT_LANG_BY_EXT
        ]

    def _decode_text(self, data: bytes, name: str) -> str:
        if len(data) > ATTACH_MAX_BYTES:
            raise ValueError(
                f"{name} is {len(data):,} bytes. "
                f"The limit is {ATTACH_MAX_BYTES:,} bytes."
            )

        if b"\x00" in data:
            raise ValueError(f"{name} appears to be binary.")

        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name} is not valid UTF-8 text.") from exc

    def _read_workspace_file(self, filename: str) -> str:
        path = self.work_dir / Path(filename).name

        if not path.is_file():
            raise ValueError(f"{filename} was not found in the workspace.")

        return self._decode_text(path.read_bytes(), path.name)

    @staticmethod
    def _normalize_uploads(value) -> list:
        if isinstance(value, dict):
            return [
                (name, bytes(info["content"]))
                for name, info in value.items()
            ]

        return [
            (info["name"], bytes(info["content"]))
            for info in value
        ]

    def _add_attachment(self, name: str, content: str, source: str):
        self.attachments = [
            item for item in self.attachments if item["name"] != name
        ]

        if len(self.attachments) >= ATTACH_MAX_FILES:
            raise ValueError(
                f"You can attach up to {ATTACH_MAX_FILES} files per prompt."
            )

        self.attachments.append(
            {
                "name": name,
                "content": content,
                "source": source,
                "lang": TEXT_LANG_BY_EXT.get(
                    Path(name).suffix.lower(),
                    "text",
                ),
            }
        )

    @staticmethod
    def _fence_for(content: str) -> str:
        longest = max(
            (len(run) for run in re.findall(r"`+", content)),
            default=0,
        )
        return "`" * max(3, longest + 1)

    def _compose_prompt(self, text: str, attachments: list) -> str:
        if not attachments:
            return text

        blocks = []

        for item in attachments:
            fence = self._fence_for(item["content"])
            blocks.append(
                f"Attached file: {item['name']}\n"
                f"{fence}{item['lang']}\n"
                f"{item['content']}\n"
                f"{fence}"
            )

        instruction = text or "Please analyze the attached file(s)."
        return "\n\n".join(blocks) + "\n\n" + instruction

    def _refresh_attachment_view(self):
        if not self.attachments:
            self.attachment_list.value = (
                "<div style='color:#7a8799; font-size:12px;'>"
                "No files attached to the next prompt."
                "</div>"
            )
            self.btn_clear_attachments.disabled = True
            return

        chips = []

        for item in self.attachments:
            line_count = len(item["content"].splitlines())
            chips.append(
                "<span class='kwb-chip'>"
                f"📎 {html.escape(item['name'])}"
                f" <span style='opacity:.7;'>({line_count:,} lines)</span>"
                "</span>"
            )

        self.attachment_list.value = "<div>" + "".join(chips) + "</div>"
        self.btn_clear_attachments.disabled = False

    def on_refresh_files_clicked(self, _):
        files = self._list_workspace_files()
        self.attach_select.options = files
        self.attach_select.value = files[0] if files else None

        self._set_status(
            f"🔄 Found {len(files)} text file(s) in the workspace.",
            "#2563b8",
        )

    def on_attach_existing_clicked(self, _):
        filename = self.attach_select.value

        if not filename:
            self._set_status("⚠ No workspace file is selected.", "#d97706")
            return

        try:
            content = self._read_workspace_file(filename)
            self._add_attachment(filename, content, "workspace")
        except ValueError as exc:
            self._set_status(f"✕ {html.escape(str(exc))}", "#c62828")
            return

        self._refresh_attachment_view()
        self._set_status(
            f"📎 Attached {html.escape(filename)}.",
            "#16805a",
        )

    def on_attach_upload_clicked(self, _):
        uploads = self._normalize_uploads(self.upload_widget.value)

        if not uploads:
            self._set_status(
                "⚠ Choose one or more files to upload first.",
                "#d97706",
            )
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
            self._set_status(
                f"✕ Skipped: {html.escape(' | '.join(errors))}",
                "#c62828",
            )
        else:
            self._set_status(
                f"📎 Attached {len(attached)} uploaded file(s).",
                "#16805a",
            )

    def on_clear_attachments_clicked(self, _):
        self.attachments = []
        self._refresh_attachment_view()
        self._set_status("Attachments cleared.", "#64748b")

    # ==========================================
    # 4. CLIPBOARD AND VIEW RENDERING
    # ==========================================
    def _copy_text_to_clipboard(
        self,
        text: str,
        label: str = "Content",
        button: widgets.Button = None,
    ):
        """Copies text to clipboard via the Jupyter kernel comm channel.

        Uses navigator.clipboard first, and automatically falls back to an
        off-screen textarea + document.execCommand('copy') when running inside
        restricted Kaggle/Jupyter cross-origin iframes.
        """
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
                try {{
                    document.execCommand('copy');
                }} catch (err) {{
                    console.error('Fallback copy failed:', err);
                }}
                document.body.removeChild(ta);
            }}

            if (navigator.clipboard && window.isSecureContext) {{
                navigator.clipboard.writeText(text).catch(function() {{
                    fallbackCopy(text);
                }});
            }} else {{
                fallbackCopy(text);
            }}
        }})();
        """

        with self._js_sink:
            self._js_sink.clear_output(wait=True)
            display(Javascript(js_code))

        if button is not None:
            original_desc = button.description
            original_icon = button.icon
            if original_desc != "Copied!":
                button.description = "Copied!"
                button.icon = "check"

                def _restore():
                    button.description = original_desc
                    button.icon = original_icon

                button._restore_fn = _restore

        char_count = len(text)
        safe_label = html.escape(label)
        self._set_status(
            f"📋 Copied <b>{safe_label}</b> ({char_count:,} chars) to clipboard.",
            "#16805a",
        )

    def _format_model_text(self, text: str, msg_idx: int) -> str:
        del msg_idx  # Retained for signature compatibility.

        if HAS_MARKDOWN:
            return markdown.markdown(
                text,
                extensions=["fenced_code", "tables"],
            )

        return (
            "<div style='"
            "white-space:pre-wrap;"
            "font-family:ui-monospace, SFMono-Regular, "
            "Menlo, Monaco, Consolas, monospace;"
            "'>"
            f"{html.escape(text)}"
            "</div>"
        )

    def _build_user_card_widget(self, entry: dict) -> widgets.Widget:
        time_string = html.escape(entry.get("time", ""))
        safe_user_text = html.escape(entry["text"])

        chips = "".join(
            "<span class='kwb-chip'>"
            f"📎 {html.escape(item['name'])}</span>"
            for item in entry.get("attachments", [])
        )

        card_html = widgets.HTML(
            value=f"""
            <div class="kwb-card-header">
                <div class="kwb-card-identity">
                    <div class="kwb-avatar kwb-user-avatar">◉</div>
                    <div>
                        <div class="kwb-role">You</div>
                        <div class="kwb-time">{time_string}</div>
                    </div>
                </div>
            </div>
            {chips}
            <div class="kwb-user-body">{safe_user_text}</div>
            """,
            layout=widgets.Layout(width="100%", margin="0"),
        )

        box = widgets.VBox(
            [card_html],
            layout=widgets.Layout(width="93%", margin="6px 0 6px auto"),
        )
        box.add_class("kwb-chat-card")
        box.add_class("kwb-user-card")
        return box

    def _build_model_card_widget(
        self, entry: dict, index: int
    ) -> widgets.Widget:
        time_string = html.escape(entry.get("time", ""))
        selected_name = html.escape(entry.get("model_name", "Model"))
        agent_role = entry.get("agent_role")
        agent_badge = entry.get("agent_badge")
        raw_text = entry["text"]
        rendered_body = self._format_model_text(raw_text, index)

        if agent_role:
            safe_role = html.escape(agent_role)
            badge_html = (
                f"<span class='kwb-role-badge'>{html.escape(agent_badge)}</span>"
                if agent_badge
                else ""
            )
            title_html = (
                f"{safe_role} "
                f"<span style='font-weight:500; opacity:.75;'>({selected_name})</span>"
                f"{badge_html}"
            )
            avatar_cls = "kwb-agent-avatar"
            avatar_icon = "🤖"
            card_cls = "kwb-agent-card"
        else:
            title_html = selected_name
            avatar_cls = "kwb-model-avatar"
            avatar_icon = "✦"
            card_cls = "kwb-model-card"

        header_identity = widgets.HTML(
            value=f"""
            <div class="kwb-card-identity">
                <div class="kwb-avatar {avatar_cls}">{avatar_icon}</div>
                <div style="min-width:0;">
                    <div class="kwb-role">{title_html}</div>
                    <div class="kwb-time">{time_string}</div>
                </div>
            </div>
            """,
            layout=widgets.Layout(margin="0"),
        )

        # Native ipywidgets Buttons so Jupyter's HTML sanitizer cannot strip
        # the click handler.
        copy_md_btn = widgets.Button(
            description="Copy Markdown",
            icon="copy",
            tooltip="Copy the raw Markdown response to the clipboard",
            layout=widgets.Layout(width="auto", height="30px"),
        )
        copy_md_btn.add_class("kwb-copy-widget-btn")
        copy_md_btn.on_click(
            lambda btn, txt=raw_text: self._copy_text_to_clipboard(
                txt,
                label="Markdown response",
                button=btn,
            )
        )

        button_list = [copy_md_btn]

        code_blocks = self._extract_code_blocks(raw_text)
        for block_idx, (lang, code_content) in enumerate(code_blocks):
            clean_code = code_content.strip()
            lang_label = (lang or "code").strip()
            btn_label = (
                f"Copy {lang_label} #{block_idx + 1}"
                if len(code_blocks) > 1
                else f"Copy {lang_label}"
            )
            copy_code_btn = widgets.Button(
                description=btn_label,
                icon="code",
                tooltip=f"Copy code block #{block_idx + 1} to the clipboard",
                layout=widgets.Layout(width="auto", height="30px"),
            )
            copy_code_btn.add_class("kwb-copy-widget-btn")
            copy_code_btn.on_click(
                lambda btn, c=clean_code, lbl=btn_label: (
                    self._copy_text_to_clipboard(c, label=lbl, button=btn)
                )
            )
            button_list.append(copy_code_btn)

        actions_box = widgets.HBox(
            button_list,
            layout=widgets.Layout(
                flex_flow="row wrap",
                gap="6px",
                align_items="center",
            ),
        )

        header_row = widgets.HBox(
            [header_identity, actions_box],
            layout=widgets.Layout(
                width="100%",
                justify_content="space-between",
                align_items="center",
                flex_flow="row wrap",
                margin="0 0 8px 0",
            ),
        )

        body_html = widgets.HTML(
            value=f'<div class="kwb-model-body">{rendered_body}</div>',
            layout=widgets.Layout(width="100%", margin="0"),
        )

        card = widgets.VBox(
            [header_row, body_html],
            layout=widgets.Layout(width="97%", margin="6px auto 6px 0"),
        )
        card.add_class("kwb-chat-card")
        card.add_class(card_cls)
        return card

    def refresh_chat_view(self):
        if not self.chat_history:
            self.chat_container.children = [self.chat_empty_html]
            return

        card_widgets = []

        for index, entry in enumerate(self.chat_history):
            if entry["role"] == "user":
                card_widgets.append(self._build_user_card_widget(entry))
            else:
                card_widgets.append(
                    self._build_model_card_widget(entry, index)
                )

        self.chat_container.children = card_widgets

    def _get_empty_chat_html(self) -> str:
        return """
        <div class="kwb-empty">
            <div class="kwb-empty-icon">💬</div>
            <div class="kwb-empty-title">
                Your workspace & SuperAgent team are ready
            </div>
            <div class="kwb-empty-copy">
                Send a manual prompt or click <b>Launch SuperAgent</b> to
                watch your multi-model team plan, write, run, and verify code.
            </div>
        </div>
        """

    def _set_status(
        self,
        html_msg: str,
        color: str = "#202124",
    ):
        self.status_label.value = (
            "<div class='kwb-status' "
            f"style='color:{color}; border-color:{color}33;'>"
            f"{html_msg}"
            "</div>"
        )

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
        self.btn_clear_attachments.disabled = (
            disabled or not self.attachments
        )

        self.model_dropdown.disabled = disabled
        self.orchestrator_dropdown.disabled = disabled
        self.coder_dropdown.disabled = disabled
        self.critic_dropdown.disabled = disabled
        self.agent_max_steps_slider.disabled = disabled
        self.agent_auto_exec_checkbox.disabled = disabled
        self.temp_slider.disabled = disabled
        self.max_turns_slider.disabled = disabled
        self.sys_prompt_input.disabled = disabled

    def display(self):
        top_bar = widgets.HBox(
            [
                widgets.HTML(
                    value="""
                    <div class="kwb-toolbar-label">
                        Active model (Single Prompt)
                    </div>
                    """
                ),
                self.model_dropdown,
            ],
            layout=widgets.Layout(
                width="100%",
                align_items="center",
                flex_flow="row wrap",
            ),
        )
        top_bar.add_class("kwb-toolbar")

        toolbar_panel = widgets.VBox(
            [top_bar],
            layout=widgets.Layout(width="100%"),
        )
        toolbar_panel.add_class("kwb-panel")

        input_panel = widgets.VBox(
            [self.input_header, self.textarea],
            layout=widgets.Layout(width="100%"),
        )
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
            layout=widgets.Layout(
                width="100%",
                align_items="center",
                flex_flow="row wrap",
            ),
        )
        action_bar.add_class("kwb-actionbar")
        action_bar.add_class("kwb-toolbar")

        files_area = widgets.VBox(
            [self.download_links_box, self.exec_buttons_box],
            layout=widgets.Layout(width="100%"),
        )

        chat_heading = widgets.HTML(
            value="""
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                margin:4px 2px -3px;
                gap:12px;
                flex-wrap:wrap;
            ">
                <div class="kwb-section-title">
                    Conversation & SuperAgent Timeline
                </div>

                <div style="
                    color:var(--kwb-muted);
                    font-size:10px;
                ">
                    Every manual turn and autonomous SuperAgent step is logged below
                </div>
            </div>
            """
        )

        app_layout = widgets.VBox(
            [
                self.theme,
                self._js_sink,
                self.header,
                toolbar_panel,
                self.superagent_accordion,
                self.settings_accordion,
                input_panel,
                self.attachment_panel,
                action_bar,
                files_area,
                self.exec_container,
                chat_heading,
                self.chat_container,
            ],
            layout=widgets.Layout(width="100%"),
        )

        app_layout.add_class("kwb-root")
        app_layout.add_class("kwb-shell")

        display(app_layout)


# ==========================================
# 5. LAUNCH WORKBENCH
# ==========================================
workbench = KaggleLLMWorkbench()
workbench.display()