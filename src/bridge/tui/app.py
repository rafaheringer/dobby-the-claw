"""Textual TUI dashboard for the Dobby bridge."""

from __future__ import annotations

import logging
import queue
from datetime import datetime
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Label, RichLog, Static

from bridge.tui.status import check_service_up, get_pi_stats

if TYPE_CHECKING:
    from bridge.config import BridgeConfig

# ---------------------------------------------------------------------------
# Log level colours (Rich markup)
# ---------------------------------------------------------------------------
_LEVEL_STYLE: dict[int, str] = {
    logging.DEBUG: "dim",
    logging.INFO: "white",
    logging.WARNING: "yellow",
    logging.ERROR: "bold red",
    logging.CRITICAL: "bold red reverse",
}


class StatusBar(Static):
    """Top panel — service health, Pi stats, mode label."""

    DEFAULT_CSS = """
    StatusBar {
        height: 4;
        background: $panel;
        border-bottom: solid $primary-darken-2;
        padding: 0 2;
        content-align: left middle;
    }
    """

    def __init__(self, config: "BridgeConfig", mode_name: str) -> None:
        super().__init__()
        self._config = config
        self._mode_name = mode_name
        self._ha_up: bool | None = None
        self._oc_up: bool | None = None
        self._pi: dict | None = None

    def render_status(self) -> str:
        def dot(up: bool | None) -> str:
            if up is None:
                return "[yellow]●[/]"
            return "[green]●[/]" if up else "[red]●[/]"

        def up_text(up: bool | None) -> str:
            if up is None:
                return "[yellow]…[/]"
            return "[green]UP[/]" if up else "[red]DOWN[/]"

        ha_part = ""
        if self._config.home_assistant_enabled:
            ha_part = f"  HA {dot(self._ha_up)} {up_text(self._ha_up)}"

        oc_part = ""
        if self._config.openclaw_enabled:
            oc_part = f"  OpenClaw {dot(self._oc_up)} {up_text(self._oc_up)}"

        pi_part = "  Pi [dim]—[/]"
        if self._pi:
            load = self._pi["cpu_load"]
            used = self._pi["mem_used_gb"]
            total = self._pi["mem_total_gb"]
            pi_part = f"  Pi CPU [cyan]{load:.1f}[/]  RAM [cyan]{used:.1f}[/]/[dim]{total:.1f}[/] GB"

        mode_part = f"  [dim]│[/]  mode [bold]{self._mode_name}[/]"

        now = datetime.now().strftime("%H:%M:%S")
        ts_part = f"  [dim]│  {now}[/]"

        return f"[bold magenta]dobby[/]{ha_part}{oc_part}{pi_part}{mode_part}{ts_part}"

    def update_ha(self, up: bool) -> None:
        self._ha_up = up
        self.update(self.render_status())

    def update_oc(self, up: bool) -> None:
        self._oc_up = up
        self.update(self.render_status())

    def update_pi(self, stats: dict | None) -> None:
        self._pi = stats
        self.update(self.render_status())

    def on_mount(self) -> None:
        self.update(self.render_status())


class DobbyTUI(App):
    """Bridge dashboard: status bar, live log panel, text input."""

    TITLE = "Dobby"
    CSS = """
    Screen {
        background: $background;
    }

    StatusBar {
        height: 4;
    }

    #log {
        border: none;
        padding: 0 1;
        background: $background;
        scrollbar-gutter: stable;
    }

    #input-row {
        height: 3;
        background: $panel;
        border-top: solid $primary-darken-2;
        align: left middle;
        padding: 0 1;
    }

    #prompt-label {
        width: auto;
        color: $primary;
        padding: 0 1 0 0;
    }

    #chat-input {
        width: 1fr;
        border: none;
        background: transparent;
        padding: 0;
        height: 1;
    }

    #chat-input:focus {
        border: none;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Sair", show=True),
    ]

    def __init__(
        self,
        config: "BridgeConfig",
        text_queue: "queue.Queue[str]",
        log_queue: "queue.Queue[logging.LogRecord]",
        mode_name: str,
    ) -> None:
        super().__init__()
        self._config = config
        self._text_queue = text_queue
        self._log_queue = log_queue
        self._mode_name = mode_name

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield StatusBar(self._config, self._mode_name)
        yield RichLog(id="log", auto_scroll=True, highlight=True, markup=True, wrap=True)
        with Horizontal(id="input-row"):
            yield Label("You ❯", id="prompt-label")
            yield Input(placeholder="Digite uma mensagem…", id="chat-input")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self.query_one("#chat-input", Input).focus()
        # Drain log queue 10× per second
        self.set_interval(0.1, self._drain_log_queue)
        # Refresh status every 5 s; first check immediately after mount
        self.set_interval(5.0, self._refresh_status)
        self.set_timer(0.2, self._refresh_status)

    def on_unmount(self) -> None:
        # Signal orchestrator to stop cleanly
        self._text_queue.put("/quit")

    # ------------------------------------------------------------------
    # Text input
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if text:
            self._text_queue.put(text)
            log = self.query_one("#log", RichLog)
            log.write(f"[bold cyan]You ❯[/] {text}")
        event.input.clear()

    # ------------------------------------------------------------------
    # Log draining
    # ------------------------------------------------------------------

    def _drain_log_queue(self) -> None:
        log = self.query_one("#log", RichLog)
        while True:
            try:
                record = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self._write_log_record(log, record)

    def _write_log_record(self, log: RichLog, record: logging.LogRecord) -> None:
        style = _LEVEL_STYLE.get(record.levelno, "white")
        level_name = record.levelname.ljust(8)
        name = record.name
        # Shorten common prefixes to save width
        for prefix in ("bridge.", "reachy.", "homeassistant."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        msg = record.getMessage()
        log.write(
            f"[dim]{ts}[/]  [{style}]{level_name}[/]  [dim]{name}[/]  [{style}]{msg}[/]"
        )

    # ------------------------------------------------------------------
    # Status refresh
    # ------------------------------------------------------------------

    async def _refresh_status(self) -> None:
        status_bar = self.query_one(StatusBar)

        tasks = []
        if self._config.home_assistant_enabled and self._config.home_assistant_ws_url:
            tasks.append(("ha", check_service_up(self._config.home_assistant_ws_url)))
        if self._config.openclaw_enabled and self._config.openclaw_ws_url:
            tasks.append(("oc", check_service_up(self._config.openclaw_ws_url)))
        tasks.append(("pi", get_pi_stats(
            host=self._config.pi_ssh_host,
            user=self._config.pi_ssh_user,
        )))

        import asyncio
        results = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)

        for (key, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                result = None
            if key == "ha":
                status_bar.update_ha(bool(result))
            elif key == "oc":
                status_bar.update_oc(bool(result))
            elif key == "pi":
                status_bar.update_pi(result if isinstance(result, dict) else None)

        # Always re-render timestamp
        status_bar.update(status_bar.render_status())
