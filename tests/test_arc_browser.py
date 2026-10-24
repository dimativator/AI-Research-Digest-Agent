import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from research_digest.retrieval.arc_browser import (
    ArcTab,
    ArcTimeoutError,
    SCRIPT,
    _arc_command,
)


class ArcBrowserTests(unittest.TestCase):
    def test_dom_element_readiness_uses_javascript_truthiness(self):
        from playwright.sync_api import sync_playwright

        with sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True, channel="chromium")
            try:
                page = browser.new_page()
                page.set_content('<article data-testid="tweet">A real post</article>')
                tab = ArcTab("window", "tab")

                def arc_evaluate(*arguments):
                    return json.dumps(page.evaluate(arguments[3]))

                with patch(
                    "research_digest.retrieval.arc_browser._arc_command",
                    side_effect=arc_evaluate,
                ):
                    tab.wait_for_function(
                        "() => document.querySelector('article[data-testid=\"tweet\"]')",
                        timeout=100,
                    )
            finally:
                browser.close()

    def test_wait_retries_transient_renderer_timeouts(self):
        tab = ArcTab("window", "tab")
        tab.evaluate = Mock(side_effect=[ArcTimeoutError("Renderer loading"), True])
        with patch("time.sleep"):
            tab.wait_for_function("() => true", timeout=1000)
        self.assertEqual(tab.evaluate.call_count, 2)

    def test_navigation_waits_for_native_loading_before_javascript(self):
        tab = ArcTab("window", "tab")
        tab.evaluate = Mock(return_value=True)
        with (
            patch(
                "research_digest.retrieval.arc_browser._arc_command",
                side_effect=["", "true", "false"],
            ) as command,
            patch("time.sleep"),
        ):
            tab.goto("https://x.com/home")
        self.assertEqual(
            [call.args[0] for call in command.call_args_list],
            ["navigate", "loading", "loading"],
        )

    def test_arc_double_serialized_values_preserve_types(self):
        tab = ArcTab("window", "tab")
        for value in (False, True, None, "false", {"ready": True}, [{"id": "123"}]):
            with (
                self.subTest(value=value),
                patch(
                    "research_digest.retrieval.arc_browser._arc_command",
                    return_value=json.dumps(json.dumps({"value": value})),
                ),
            ):
                self.assertEqual(tab.evaluate("() => null"), value)

    def test_commands_pass_data_as_arguments_without_shell_interpolation(self):
        url = 'https://x.com/search?q="test"&text=$(touch /tmp/not-executed)'
        with (
            patch("research_digest.retrieval.arc_browser.sys.platform", "darwin"),
            patch("pathlib.Path.is_dir", return_value=True),
            patch(
                "subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0, stdout="window-id\ntab-id\n"
                ),
            ) as run,
        ):
            tab = ArcTab.open(url)
        arguments = run.call_args.args[0]
        self.assertEqual(arguments, ["osascript", "-", "open", url])
        self.assertEqual(run.call_args.kwargs["input"], SCRIPT)
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual((tab.window_id, tab.tab_id), ("window-id", "tab-id"))

    def test_evaluation_and_close_target_only_the_owned_tab(self):
        tab = ArcTab("owned-window", "owned-tab")
        with patch(
            "research_digest.retrieval.arc_browser._arc_command",
            return_value=json.dumps({"value": {"ready": True}}),
        ) as command:
            self.assertEqual(tab.evaluate("() => ({ready: true})"), {"ready": True})
            self.assertEqual(
                command.call_args.args[:3], ("evaluate", "owned-window", "owned-tab")
            )
            tab.close()
            command.assert_called_with("close", "owned-window", "owned-tab")

    def test_permission_errors_have_actionable_instructions(self):
        with (
            patch("research_digest.retrieval.arc_browser.sys.platform", "darwin"),
            patch("pathlib.Path.is_dir", return_value=True),
            patch(
                "subprocess.run",
                return_value=SimpleNamespace(
                    returncode=1, stdout="", stderr="Apple Events denied (-1743)"
                ),
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Allow JavaScript from Apple Events"
            ):
                _arc_command("evaluate", "window", "tab", "1")

    def test_waits_are_bounded_and_closed_tabs_are_harmless(self):
        tab = ArcTab("window", "tab")
        tab.evaluate = Mock(return_value=False)
        with self.assertRaisesRegex(RuntimeError, "did not become ready"):
            tab.wait_for_function("() => false", timeout=0)
        with patch(
            "research_digest.retrieval.arc_browser._arc_command",
            side_effect=RuntimeError("Already closed"),
        ):
            tab.close()


if __name__ == "__main__":
    unittest.main()
