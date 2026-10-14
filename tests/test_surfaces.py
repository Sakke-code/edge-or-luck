"""The ways an assistant reaches the judge: CLI, MCP tool, one-file build,
and the llms.txt they all point to."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import edgeorluck
from edgeorluck import cli, samples, single

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def lucky_csv(tmp_path):
    path = tmp_path / "lucky streak.csv"
    path.write_text(samples.lucky(), encoding="utf-8")
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_markdown_json_and_exit_codes(lucky_csv, tmp_path, capsys):
    assert cli.main([str(lucky_csv), "--tried", "40"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("## Edge or luck: Can't tell yet") and "40 versions" in out
    assert cli.main([str(lucky_csv), "--json", "--benchmark", "12.5", "--fees", "yes", "--source", "paper"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["settings"]["benchmark_return_pct"] == 12.5 and result["settings"]["source"] == "paper"
    odd = tmp_path / "odd.csv"
    odd.write_text("a,b\n1,2\n", encoding="utf-8")
    assert cli.main([str(odd)]) == 2
    assert "need the trades" in capsys.readouterr().out
    assert cli.main(["--schema"]) == 0
    assert json.loads(capsys.readouterr().out)["$schema"].startswith("https://json-schema.org")


def test_cli_backtest_compare(tmp_path, capsys):
    live, backtest = samples.compare()
    (tmp_path / "live.csv").write_text(live, encoding="utf-8")
    (tmp_path / "bt.csv").write_text(backtest, encoding="utf-8")
    assert cli.main([str(tmp_path / "live.csv"), "--backtest", str(tmp_path / "bt.csv")]) == 0
    assert "### Live vs backtest" in capsys.readouterr().out


def test_console_script_runs_in_a_fresh_process(lucky_csv):
    exe = Path(sys.executable).with_name("edgeorluck.exe" if os.name == "nt" else "edgeorluck")
    if not exe.exists():
        pytest.skip("console script not installed (pip install -e .)")
    out = subprocess.run([str(exe), str(lucky_csv), "--json"], capture_output=True, text=True, encoding="utf-8",
                         timeout=120)
    assert out.returncode == 0
    assert json.loads(out.stdout)["input_fingerprint"] == edgeorluck.judge(str(lucky_csv))["input_fingerprint"]


# ── MCP ───────────────────────────────────────────────────────────────────────

def test_mcp_tool_called_directly(lucky_csv):
    from edgeorluck.mcp_server import JUDGE_DESCRIPTION, judge_trades
    text = judge_trades(file_path=str(lucky_csv), variants_tried=40, detail="full")
    assert text.startswith("## Edge or luck: Can't tell yet")
    payload = json.loads(text.split("```json\n", 1)[1].rsplit("\n```", 1)[0])
    assert payload["settings"]["variants_tried"] == 40 and payload["status"] == "ok"
    assert judge_trades(csv_text="pnl\n1\n2\n").startswith("## Edge or luck: Too early")
    assert "No file at" in judge_trades(file_path=str(lucky_csv.parent / "missing.csv"))
    assert "Nothing to judge" in judge_trades()
    live, backtest = samples.compare()
    assert "Live vs backtest" in judge_trades(csv_text=live, backtest_csv_text=backtest)
    assert all(words in JUDGE_DESCRIPTION for words in ("whenever the user shows trading results", "variants_tried",
                                                         "benchmark_return_pct", "Never present 'cant_tell' as a win"))


def test_mcp_server_registers_its_tools():
    pytest.importorskip("mcp")
    import asyncio
    from edgeorluck.mcp_server import build_server
    server = build_server()
    tools = asyncio.run(server.list_tools())
    tools = getattr(tools, "tools", tools)
    names = {t.name for t in tools}
    assert names == {"judge_trades", "backtest_rule", "forward_test", "tune_rule", "rule_language"}

    def props(name):
        tool = next(t for t in tools if t.name == name)
        return set((getattr(tool, "inputSchema", None) or getattr(tool, "input_schema"))["properties"])
    assert {"file_path", "csv_text", "pnl", "variants_tried", "benchmark_return_pct", "backtest_file_path"} <= \
        props("judge_trades")
    assert {"buy", "sell", "symbol", "interval", "start", "prices_file", "stop_loss_pct", "variants_tried",
            "freeze"} <= props("backtest_rule") and props("forward_test") == {"frozen_id"}
    assert {"buy", "grid", "holdout", "prices_file"} <= props("tune_rule")


@pytest.mark.slow
def test_mcp_over_stdio_with_the_official_client(lucky_csv):
    pytest.importorskip("mcp")
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "mcp_roundtrip.py"), str(lucky_csv)],
                         capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "tools: ['judge_trades', 'backtest_rule', 'forward_test', 'tune_rule', 'rule_language']" in out.stdout
    assert "## Honest backtest:" in out.stdout and "## Honest tuning:" in out.stdout


# ── the one-file build ────────────────────────────────────────────────────────

def test_the_one_file_build_gives_the_same_answers(tmp_path, lucky_csv):
    target = single.write(str(tmp_path / "edgeorluck_single.py"))
    source = target.read_text(encoding="utf-8")
    import re
    assert not re.search(r"^\s*(from edgeorluck\b|import edgeorluck\b|from \.)", source, re.M)
    trader = tmp_path / "trader.csv"
    trader.write_text(samples.trader(), encoding="utf-8")
    fills = tmp_path / "fills.csv"
    fills.write_text(samples.fills(), encoding="utf-8")
    for path, kw in ((lucky_csv, ["--tried", "40"]), (trader, []), (fills, ["--benchmark", "5"])):
        out = subprocess.run([sys.executable, "-I", str(target), str(path), "--json", *kw], cwd=str(tmp_path),
                             capture_output=True, text=True, encoding="utf-8", timeout=180)
        assert out.returncode == 0, out.stderr[-2000:]
        bundled = json.loads(out.stdout)
        settings = {"variants_tried": 40} if "--tried" in kw else {"benchmark_return_pct": 5} if kw else {}
        package = edgeorluck.judge(str(path), **settings).to_dict()
        assert bundled == package


def test_the_one_file_build_has_no_name_clashes_and_is_reproducible():
    assert single.build() == single.build()
    compile(single.build(), "edgeorluck_single.py", "exec")


# ── llms.txt and the README ───────────────────────────────────────────────────

def test_llms_txt_is_one_file_in_two_places():
    root, packaged = ROOT / "llms.txt", ROOT / "src" / "edgeorluck" / "llms.txt"
    assert root.read_bytes() == packaged.read_bytes()
    text = root.read_text(encoding="utf-8")
    for must in ("judge(", "variants_tried", "verdict.headline", "need_input", "Not financial advice",
                 "edgeorluck-mcp", "independent"):
        assert must in text, must


def test_the_registry_entry_matches_the_release_and_the_readme():
    entry = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    assert entry["version"] == entry["packages"][0]["version"] == edgeorluck.__version__
    assert len(entry["description"]) <= 100 and entry["packages"][0]["packageArguments"][0]["value"] == "mcp"
    assert f"mcp-name: {entry['name']} -->" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_snippets_name_real_entry_points():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for must in ("claude mcp add", "[mcp_servers.edgeorluck]", ".cursor/mcp.json", "claude_desktop_config.json",
                 "edgeorluck_single.py", "bt.run()", "records_readable", "VALIDATION.md"):
        assert must in text, must
