"""Control one owned tab in the user's running Arc through native Apple Events.

Arc must allow JavaScript from Apple Events. No cookies are copied, browser
profiles are read, or debugging ports are opened. Arguments use stdin/argv,
never shell interpolation or dynamically constructed AppleScript.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ARC_APP = Path("/Applications/Arc.app")
SCRIPT = """on run argv
    set operation to item 1 of argv
    tell application "Arc"
        if operation is "open" then
            if (count of windows) is 0 then make new window
            set ownedWindowID to id of window 1
            set previousTabIDs to id of every tab of window 1
            tell window 1
                make new tab with properties {URL:item 2 of argv}
            end tell
            repeat with tabIndex from 1 to count of tabs of window 1
                set candidateID to id of tab tabIndex of window 1
                if previousTabIDs does not contain candidateID then
                    return ownedWindowID & linefeed & candidateID
                end if
            end repeat
            error "Arc did not expose the new tab"
        end if
        set windowID to item 2 of argv
        set tabID to item 3 of argv
        tell window id windowID
            if operation is "navigate" then
                set URL of tab id tabID to item 4 of argv
                return ""
            else if operation is "evaluate" then
                return execute tab id tabID javascript (item 4 of argv)
            else if operation is "loading" then
                return loading of tab id tabID
            else if operation is "close" then
                close tab id tabID
                return ""
            end if
        end tell
    end tell
    error "Unknown Arc operation"
end run"""


class ArcTimeoutError(RuntimeError):
    pass


def _arc_command(*arguments: str) -> str:
    if sys.platform != "darwin" or not ARC_APP.is_dir():
        raise RuntimeError(
            "Arc scraping requires Arc installed in /Applications on macOS."
        )
    try:
        result = subprocess.run(
            ["osascript", "-", *arguments],
            input=SCRIPT,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ArcTimeoutError(
            f"Arc did not respond to browser automation ({arguments[0]})."
        ) from error
    if result.returncode:
        detail = result.stderr.strip()[:500]
        raise RuntimeError(
            f"Arc automation failed: {detail}. Allow your terminal to control Arc in macOS "
            "Automation settings, and enable Arc > View > Developer > Allow "
            "JavaScript from Apple Events. Then run python setup_x.py."
        )
    return result.stdout.strip()


class ArcTab:
    """Small page interface shared with the collector's browser fixture tests."""

    def __init__(self, window_id: str, tab_id: str) -> None:
        self.window_id = window_id
        self.tab_id = tab_id

    @classmethod
    def open(cls, url: str) -> ArcTab:
        identifiers = _arc_command("open", url).splitlines()
        if len(identifiers) != 2 or not all(identifiers):
            raise RuntimeError("Arc did not return the new scraper tab's identifiers.")
        return cls(*identifiers)

    def evaluate(self, expression: str) -> Any:
        javascript = (
            "(() => { const expression = (" + expression + "); "
            "const value = typeof expression === 'function' ? expression() : expression; "
            "return JSON.stringify({value: value === undefined ? null : value}); })()"
        )
        output = _arc_command("evaluate", self.window_id, self.tab_id, javascript)
        try:
            envelope = json.loads(output)
            # Arc serializes JavaScript string results once more for Apple Events.
            if isinstance(envelope, str):
                envelope = json.loads(envelope)
            if not isinstance(envelope, dict) or set(envelope) != {"value"}:
                raise ValueError("Invalid Arc result envelope")
            return envelope["value"]
        except ValueError as error:
            raise RuntimeError("Arc returned an invalid browser response.") from error

    def goto(
        self, url: str, *, wait_until: str = "domcontentloaded", timeout: int = 30_000
    ) -> None:
        _arc_command("navigate", self.window_id, self.tab_id, url)
        deadline = time.monotonic() + timeout / 1000
        # Arc can stall JavaScript Apple Events while its new renderer loads.
        while _arc_command("loading", self.window_id, self.tab_id).lower() == "true":
            if time.monotonic() >= deadline:
                raise RuntimeError("Arc search page did not finish loading.")
            time.sleep(0.5)
        self.wait_for_function(
            f"() => location.href === {json.dumps(url)} && "
            "document.readyState !== 'loading'",
            timeout=max(1, int((deadline - time.monotonic()) * 1000)),
        )

    def wait_for_function(self, expression: str, *, timeout: int = 20_000) -> None:
        deadline = time.monotonic() + timeout / 1000
        predicate = (
            "() => { const predicate = (" + expression + "); "
            "return Boolean(typeof predicate === 'function' ? predicate() : predicate); }"
        )
        while time.monotonic() < deadline:
            try:
                if self.evaluate(predicate):
                    return
            except ArcTimeoutError:
                if time.monotonic() >= deadline:
                    raise
            time.sleep(0.5)
        raise RuntimeError("Arc page did not become ready before the timeout.")

    def wait_for_timeout(self, milliseconds: int) -> None:
        time.sleep(milliseconds / 1000)

    def close(self) -> None:
        try:
            _arc_command("close", self.window_id, self.tab_id)
        except RuntimeError:
            # The user may already have closed our tab or its window.
            pass
