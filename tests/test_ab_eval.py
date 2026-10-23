"""Tests for the evaluation harness itself — a grader that lies is worse than none."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ab_eval", ROOT / "scripts" / "ab_eval.py")
ab_eval = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ab_eval)


def test_normalise_collapses_whitespace_and_case():
    assert ab_eval.normalise("  Hello   World\n") == "hello world"


def test_exact_verifier():
    item = {"expect": "391"}
    assert ab_eval.verify("391", item, "exact", None) is True
    assert ab_eval.verify(" 391 ", item, "exact", None) is True
    assert ab_eval.verify("391.0", item, "exact", None) is False


def test_exact_without_expectation_is_unscored():
    assert ab_eval.verify("anything", {}, "exact", None) is None


def test_contains_verifier():
    item = {"expect": "git restore --staged"}
    assert ab_eval.verify("Use `git restore --staged file.py`", item, "contains", None) is True
    assert ab_eval.verify("git reset HEAD", item, "contains", None) is False


def test_cmd_verifier_uses_the_answer_file():
    item = {"messages": []}
    # the command receives the answer as a file; exit code decides pass/fail
    assert ab_eval.verify("good", item, "cmd", "grep -q good {answer}") is True
    assert ab_eval.verify("bad", item, "cmd", "grep -q good {answer}") is False


def test_cmd_verifier_timeout_is_a_failure():
    item = {"messages": []}
    # a command that genuinely hangs; the extra file argument is not consumed
    assert (
        ab_eval.verify(
            "x", item, "cmd", "python -c \"import time; time.sleep(5)\" {answer}", timeout=0.5
        )
        is False
    )


def test_no_verifier_reports_unscored():
    assert ab_eval.verify("whatever", {}, "none", None) is None


def test_example_prompts_are_valid():
    items = ab_eval.load_prompts(str(ROOT / "prompts.example.jsonl"))
    assert len(items) >= 6
    assert all("messages" in item and "id" in item for item in items)
    assert any("expect" in item for item in items)


def test_dead_tiers_are_not_reported_as_parity(tmp_path):
    """Both endpoints down must be an invalid run (exit 2), never gap 0."""
    out = tmp_path / "report.json"
    code = ab_eval.main(
        [
            str(ROOT / "prompts.example.jsonl"),
            "--local",
            "http://127.0.0.1:9/v1",
            "--upstream",
            "http://127.0.0.1:9/v1",
            "--verify",
            "exact",
            "--timeout",
            "3",
            "--concurrency",
            "8",
            "--out",
            str(out),
        ]
    )
    assert code == 2
    report = __import__("json").loads(out.read_text())
    assert report["local_errors"] == report["prompts"]
    assert report["agreement"] == 0.0  # two empty answers are not agreement
