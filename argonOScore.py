from IPython.display import HTML, Javascript, display
import html
import os
from pathlib import Path
import time

# Unikátny token pre každé spustenie v notebooku (ochrana pred duplicate-ID)
UID = int(time.time() * 1000)
WORK_DIR = Path("/kaggle/working")
WORK_DIR.mkdir(parents=True, exist_ok=True)


class ArgonApp:
    """Základná trieda pre aplikáciu v ArgonOS."""

    def __init__(
        self,
        app_id: str,
        title: str,
        icon: str,
        header_color: str,
        initial_pos: tuple,
        content_html: str,
    ):
        self.app_id = app_id
        self.title = title
        self.icon = icon
        self.header_color = header_color
        self.top, self.left = initial_pos
        self.content_html = content_html


class ArgonOS:
    """Modulárne mikro-jadro a okenný manažér ArgonOS."""

    def __init__(self, uid=UID, work_dir=WORK_DIR):
        self.uid = uid
        self.work_dir = work_dir
        self.apps = {}
        self._init_core_apps()

    def register_app(self, app: ArgonApp):
        self.apps[app.app_id] = app

    def _init_core_apps(self):
        # 1. TESTOVACIA APP: ARGON FILES (Správca súborov)
        self.register_app(
            ArgonApp(
                app_id="files",
                title="Argon Files Explorer",
                icon="📁",
                header_color="linear-gradient(90deg, #059669, #10b981)",
                initial_pos=(65, 14),
                content_html=self._render_files_app(),
            )
        )

        # 2. TESTOVACIA APP: SYSTEM TELEMETRY (Ovládací panel / BIOS)
        self.register_app(
            ArgonApp(
                app_id="sysinfo",
                title="System Telemetry & FX",
                icon="⚙️",
                header_color="linear-gradient(90deg, #4338ca, #6366f1)",
                initial_pos=(120, 24),
                content_html=self._render_sysinfo_app(),
            )
        )

    def _render_files_app(self) -> str:
        """Vygeneruje živý zoznam súborov vo workspace s rýchlymi akciami."""
        try:
            files = [
                p
                for p in sorted(self.work_dir.iterdir())
                if p.is_file() and not p.name.startswith(".")
            ]
        except Exception:
            files = []

        file_items = []
        for f in files:
            size_kb = f.stat().st_size / 1024
            name_escaped = html.escape(f.name)
            is_py = f.suffix.lower() == ".py"
            badge_color = "#38bdf8" if is_py else "#a5b4fc"

            file_items.append(
                f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 8px 10px;
                    margin-bottom: 6px;
                    background: rgba(15, 23, 42, 0.6);
                    border: 1px solid #334155;
                    border-radius: 8px;
                ">
                    <div style="min-width:0; overflow:hidden; text-overflow:ellipsis;">
                        <span style="font-weight:bold; color:{badge_color};">📄 {name_escaped}</span>
                        <div style="font-size:10px; color:#94a3b8;">{size_kb:.1f} KB</div>
                    </div>
                    <div style="display:flex; gap:4px; flex-shrink:0;">
                        <button onclick="alert('Cesta: {html.escape(str(f))}')" 
                                style="background:#1e293b; color:#e2e8f0; border:1px solid #475569; border-radius:6px; padding:4px 8px; font-size:10px;">
                            👁️ Info
                        </button>
                    </div>
                </div>
            """
            )

        if not file_items:
            list_html = (
                "<div style='color:#64748b; font-size:12px; padding:12px; text-align:center;'>Priečinok /kaggle/working je zatiaľ prázdny.</div>"
            )
        else:
            list_html = "".join(file_items)

        return f"""
        <div style="padding: 12px; color: #f1f5f9; font-size: 12px; display:flex; flex-direction:column; height:100%;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <b style="color:#38bdf8;">/kaggle/working</b>
                <span style="background:#0f172a; padding:2px 8px; border-radius:6px; font-size:10px; color:#10b981;">{len(files)} súborov</span>
            </div>
            <div style="overflow-y:auto; flex-grow:1; max-height:210px;">
                {list_html}
            </div>
        </div>
        """

    def _render_sysinfo_app(self) -> str:
        """Ovládací panel a systémové telemetrie."""
        return f"""
        <div style="padding: 14px; color: #f1f5f9; font-size: 12px; line-height: 1.6;">
            <b style="color:#38bdf8;">ArgonOS Core v9 (Micro-Kernel)</b><br>
            <div style="margin: 8px 0; padding:8px; background:#090d16; border-radius:8px; border:1px solid #1e293b; font-size:11px;">
                <div>• Platform: <span style="color:#a5b4fc;">Kaggle Linux VM</span></div>
                <div>• Touch Engine: <span style="color:#10b981;">Active (PointerCapture)</span></div>
                <div>• Window Manager: <span style="color:#10b981;">Multi-Instance Z-Stack</span></div>
            </div>

            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px;">
                <span>🌈 Obvodový RGB LED pás:</span>
                <button onclick="window.argonToggleLED(this, '{self.uid}')"
                        id="btnLedSys_{self.uid}"
                        style="background: #1e293b; color: #38bdf8; border: 1px solid #0284c7; border-radius: 8px; padding: 6px 12px; font-weight:bold; font-size:11px;">
                    ⚡ LED: OFF
                </button>
            </div>
        </div>
        """

    def render(self):
        # 1. Vygenerovanie HTML pre jednotlivé okná
        windows_html = []
        dock_buttons_html = []

        for idx, (app_id, app) in enumerate(self.apps.items()):
            win_id = f"win_{app_id}_{self.uid}"
            active_class = "active-win" if idx == 0 else ""
            dock_active = "active" if idx == 0 else ""

            # HTML okna
            windows_html.append(
                f"""
            <div class="argon-win-{self.uid} {active_class}" id="{win_id}" 
                 style="top: {app.top}px; left: {app.left}px; z-index: {100 - idx * 10};" 
                 onpointerdown="window.argonBringToFront(this)">
                
                <div class="win-header-{self.uid}" style="background: {app.header_color};" onpointerdown="window.argonStartDrag(event, this)">
                    <span>{app.icon} {app.title}</span>
                    <div style="display:flex;">
                        <button class="win-btn-{self.uid}" onclick="window.argonMaximizeWin(this)">□</button>
                        <button class="win-btn-{self.uid}" style="background:#ef4444;" onclick="this.closest('.argon-win-{self.uid}').style.display='none';">✕</button>
                    </div>
                </div>

                <div style="flex-grow:1; overflow:hidden;">
                    {app.content_html}
                </div>
            </div>
            """
            )

            # Tlačidlo do pravého docku na taskbari
            dock_buttons_html.append(
                f"""
                <button class="app-icon-btn {dock_active}" 
                        id="dock_{app_id}_{self.uid}"
                        onclick="window.argonToggleApp('{win_id}', this)" 
                        title="{app.title}">
                    {app.icon}
                </button>
            """
            )

        html_body = f"""
        <style>
            .argon-desktop-{self.uid} {{
                position: relative;
                width: 100%;
                height: 560px;
                background: #060913;
                border-radius: 18px;
                overflow: hidden;
                user-select: none;
                -webkit-user-select: none;
                touch-action: none;
                font-family: monospace;
                transition: all 0.2s ease;
                border: 2px solid #334155;
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

            /* RGB LED SVETELNÝ PÁS */
            .rgb-strip-{self.uid} {{
                position: absolute;
                inset: -100%;
                width: 300%;
                height: 300%;
                background: conic-gradient(from 0deg, #ff0055, #7928ca, #0070f3, #00dfd8, #00ff66, #ffea00, #ff0055);
                animation: rgbSpin-{self.uid} 3s linear infinite;
                opacity: 0;
                visibility: hidden;
                transition: opacity 0.35s ease;
                z-index: 1;
            }}
            .argon-desktop-{self.uid}.rgb-active .rgb-strip-{self.uid} {{
                opacity: 1;
                visibility: visible;
            }}
            @keyframes rgbSpin-{self.uid} {{ to {{ transform: rotate(360deg); }} }}

            .desk-inner-{self.uid} {{
                position: absolute;
                inset: 3px;
                background: radial-gradient(circle at 50% 15%, #15182d 0%, #060913 100%);
                border-radius: 15px;
                z-index: 2;
                overflow: hidden;
            }}
            .argon-desktop-{self.uid}.is-fullscreen .desk-inner-{self.uid} {{
                border-radius: 0;
                inset: 0;
            }}

            /* OKNÁ */
            .argon-win-{self.uid} {{
                position: absolute;
                background: rgba(22, 30, 49, 0.96);
                border: 2px solid #475569;
                border-radius: 14px;
                box-shadow: 0 16px 35px rgba(0,0,0,0.7);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                width: 88%;
                max-width: 330px;
                touch-action: none;
                transition: border-color 0.15s;
            }}
            .argon-win-{self.uid}.active-win {{
                border-color: #6366f1 !important;
                box-shadow: 0 16px 40px rgba(99, 102, 241, 0.35);
            }}

            .argon-win-{self.uid}.is-maximized {{
                top: 6px !important;
                left: 6px !important;
                width: calc(100% - 12px) !important;
                height: calc(100% - 66px) !important;
                max-width: none !important;
                border-radius: 12px !important;
            }}

            .win-header-{self.uid} {{
                height: 44px;
                min-height: 44px;
                padding: 0 10px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                cursor: grab;
                touch-action: none;
            }}

            .win-btn-{self.uid} {{
                width: 32px;
                height: 32px;
                border-radius: 7px;
                background: rgba(255,255,255,0.18);
                color: white;
                font-weight: bold;
                border: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-left: 4px;
            }}
            .win-btn-{self.uid}:active {{ transform: scale(0.92); }}

            /* TROJDIELNY TASKBAR */
            .taskbar-{self.uid} {{
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 52px;
                background: rgba(10, 15, 29, 0.98);
                border-top: 1px solid #334155;
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 10px;
                z-index: 999999;
            }}

            .taskbar-chip {{
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 3px 6px;
                font-size: 10px;
                color: #10b981;
            }}

            .app-icon-btn {{
                width: 38px;
                height: 38px;
                border-radius: 9px;
                background: #1e293b;
                border: 1px solid #475569;
                color: white;
                font-size: 17px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                margin-left: 4px;
                touch-action: manipulation;
            }}
            .app-icon-btn.active {{
                border-color: #6366f1;
                background: #312e81;
            }}
        </style>

        <div class="argon-desktop-{self.uid}" id="desk_{self.uid}">
            <div class="rgb-strip-{self.uid}"></div>

            <div class="desk-inner-{self.uid}">
                <!-- VŠETKY REGISTROVANÉ APLIKÁCIE AKO OKNÁ -->
                {''.join(windows_html)}

                <!-- TROJDIELNY SYSTÉMOVÝ TASKBAR -->
                <div class="taskbar-{self.uid}">
                    <!-- VĽAVO: SYSTÉMOVÝ MONITOR -->
                    <div style="display:flex; flex-direction:column; gap:2px; min-width:85px;">
                        <span class="taskbar-chip">🟢 ARGON-OS</span>
                        <span style="font-size:9px; color:#94a3b8;">KERNEL: v9.1</span>
                    </div>

                    <!-- V STREDE: FULLSCREEN PREPÍNAČ -->
                    <div>
                        <button onclick="window.argonToggleFS(this)" 
                                style="background: linear-gradient(135deg, #f59e0b, #d97706); color: white; border: 0; border-radius: 9px; padding: 8px 14px; font-weight: bold; font-size: 11px;">
                            ⛶ Fullscreen
                        </button>
                    </div>

                    <!-- VPRAVO: DYNAMICKÝ DOCK APLIKÁCIÍ -->
                    <div style="display:flex; align-items:center;">
                        {''.join(dock_buttons_html)}
                    </div>
                </div>
            </div>
        </div>
        """

        js_script = f"""
        (function() {{
            let topZ = 150;

            window.argonBringToFront = function(win) {{
                if (!win) return;
                topZ += 1;
                win.style.zIndex = topZ;
                document.querySelectorAll("[class*='argon-win-']").forEach(w => w.classList.remove("active-win"));
                win.classList.add("active-win");
            }};

            window.argonMaximizeWin = function(btn) {{
                const win = btn.closest("[class*='argon-win-']");
                if (!win) return;

                if (win.classList.contains("is-maximized")) {{
                    win.classList.remove("is-maximized");
                    win.style.left = win.dataset.origL || "14px";
                    win.style.top = win.dataset.origT || "60px";
                    win.style.width = win.dataset.origW || "88%";
                    win.style.height = win.dataset.origH || "auto";
                    btn.innerText = "□";
                }} else {{
                    win.dataset.origL = win.style.left || "14px";
                    win.dataset.origT = win.style.top || "60px";
                    win.dataset.origW = win.style.width || "88%";
                    win.dataset.origH = win.style.height || "auto";
                    win.classList.add("is-maximized");
                    btn.innerText = "🗗";
                }}
                window.argonBringToFront(win);
            }};

            window.argonToggleApp = function(winId, btn) {{
                const win = document.getElementById(winId);
                if (!win) return;

                if (win.style.display === "none") {{
                    win.style.display = "flex";
                    window.argonBringToFront(win);
                    btn.classList.add("active");
                }} else {{
                    if (win.classList.contains("active-win")) {{
                        win.style.display = "none";
                        btn.classList.remove("active");
                    }} else {{
                        window.argonBringToFront(win);
                        btn.classList.add("active");
                    }}
                }}
            }};

            window.argonStartDrag = function(e, header) {{
                if (e.target.tagName === "BUTTON") return;
                const win = header.closest("[class*='argon-win-']");
                const desktop = header.closest("[class*='desk-inner-']");
                if (!win || !desktop || win.classList.contains("is-maximized")) return;

                window.argonBringToFront(win);

                const startX = e.clientX;
                const startY = e.clientY;
                const initLeft = win.offsetLeft;
                const initTop = win.offsetTop;

                header.setPointerCapture(e.pointerId);

                function onMove(ev) {{
                    const dx = ev.clientX - startX;
                    const dy = ev.clientY - startY;
                    let newL = Math.max(2, Math.min(initLeft + dx, desktop.clientWidth - win.clientWidth - 2));
                    let newT = Math.max(2, Math.min(initTop + dy, desktop.clientHeight - win.clientHeight - 56));
                    win.style.left = newL + "px";
                    win.style.top = newT + "px";
                    ev.preventDefault();
                }}

                function onUp(ev) {{
                    header.removeEventListener("pointermove", onMove);
                    header.removeEventListener("pointerup", onUp);
                    header.removeEventListener("pointercancel", onUp);
                    try {{ header.releasePointerCapture(ev.pointerId); }} catch(err) {{}}
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
                    btn.innerText = "🗗 Ukončiť";
                    btn.style.background = "#ef4444";
                    if (desktop.requestFullscreen) desktop.requestFullscreen().catch(() => {{}});
                }} else {{
                    desktop.classList.remove("is-fullscreen");
                    btn.innerText = "⛶ Fullscreen";
                    btn.style.background = "#f59e0b";
                    if (document.fullscreenElement && document.exitFullscreen) document.exitFullscreen().catch(() => {{}});
                }}
            }};

            window.argonToggleLED = function(btn, uid) {{
                const desktop = document.getElementById("desk_" + uid);
                if (!desktop) return;
                if (desktop.classList.contains("rgb-active")) {{
                    desktop.classList.remove("rgb-active");
                    btn.innerText = "⚡ LED: OFF";
                    btn.style.background = "#1e293b";
                    btn.style.borderColor = "#0284c7";
                }} else {{
                    desktop.classList.add("rgb-active");
                    btn.innerText = "⚡ LED: ACTIVE";
                    btn.style.background = "linear-gradient(135deg, #ec4899, #8b5cf6)";
                    btn.style.borderColor = "transparent";
                }}
            }};
        }})();
        """

        display(HTML(html_body))
        display(Javascript(js_script))


# Spustenie základného jadra ArgonOS
os_kernel = ArgonOS()
os_kernel.render()