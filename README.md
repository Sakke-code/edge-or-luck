# edge or luck

**Is your trading strategy a real edge, or just luck?** Drop in your trades and get an honest answer in plain words: what the record shows, how sure that is, what would change it, and one next step.

Most strategies that look good are luck: the best of many tries, a rising market, a backtest that peeked at the future, or a few big trades. This judge checks for each of those. The same file always gets the same answer, and a "real edge" verdict is wrong rarely. That rate is measured, not promised (see [VALIDATION.md](VALIDATION.md)).

Open source (MIT). It runs on your machine and keeps nothing. Not financial advice: a statistical check of past trades.

```text
$ edgeorluck "BTC breakout.csv" --tried 40
## Edge or luck: Can't tell yet
**Can't tell yet: +0.458% per trade so far, but luck often makes records like this when you tried 40 versions.**
...
**Next step:** Keep it on paper until about 630 more trades, then check again.
```

## Install

```bash
pip install edgeorluck            # the judge (numpy + pandas)
pip install "edgeorluck[mcp]"     # + the MCP server for AI assistants
pip install "edgeorluck[web]"     # + the web page and API
pip install "edgeorluck[xlsx]"    # + Excel files
```

The latest code, straight from GitHub: `pip install "edgeorluck[all] @ git+https://github.com/Sakke-code/edge-or-luck"`.

## Use it

```python
import edgeorluck

r = edgeorluck.judge("trades.csv", variants_tried=40, benchmark_return_pct=34.5)
print(r["verdict"]["headline"])   # one sentence, safe to quote
print(r.to_markdown())            # the whole answer, ready to paste
```

What it reads (it never crashes on messy input; when it can't judge, it says exactly what it needs):

| You have | Give it |
|---|---|
| TradingView strategy tester | Strategy Tester, then **List of trades**, then export (CSV or XLSX), old or new column names |
| An exchange account | the spot **trade history (fills)**: time, pair, side, price, quantity, fee. Paired into round trips first-in-first-out, net of fees |
| [backtesting.py](https://kernc.github.io/backtesting.py/) | `stats = bt.run(); print(edgeorluck.judge(stats).to_markdown())` (also uses its Buy & Hold return) |
| [vectorbt](https://vectorbt.dev/) | `edgeorluck.judge(pf)` or `edgeorluck.judge(pf.trades.records_readable)`; each column is a version |
| [freqtrade](https://www.freqtrade.io/) | `edgeorluck user_data/backtest_results/backtest-result-....json` (or the `.zip`) |
| A bot's own log | any CSV with a P&L column (money or %), or `edgeorluck.judge([{"pnl": 12.5, "closed_at": "2025-01-03"}, ...])` |
| Just numbers | `edgeorluck.judge([12.5, -3, 4.1, ...])` or the text `"+12.5%, -3%, 4.1%"` |

Exchange fills: a fee in the base coin is valued at the fill price; a fee in a third coin (BNB, BGB) is reported, not guessed. Sells with no earlier buy, and positions still open at the end, are named in the warnings.

**About the export formats:** we built the readers from public descriptions of each export's columns (Binance spot trade history old and new, Bitget spot fills and order history, TradingView's list of trades), not from real accounts. If yours fails, please [open an issue](https://github.com/Sakke-code/edge-or-luck/issues) with the header row and a few anonymised rows.

### Make the verdict more honest

| Parameter | Why it matters |
|---|---|
| `variants_tried=N` | If you tried 40 versions and kept the best, one of them looks good by luck alone. The bar rises to match. |
| `benchmark_return_pct=34.5` | A strategy that only buys looks good whenever prices rise. Did it beat simply holding? (Derived automatically from TradingView and backtesting.py results when possible.) |
| `fees_included=True/False` | A small edge before fees is often a loss after them. |
| `source="backtest" / "paper" / "live"` | Backtests are usually better than live trading. |
| `backtest=...` | Pass the backtest next to the live trades: is live keeping the backtest's promise? |

### The command line

```bash
edgeorluck trades.csv [--tried 50] [--benchmark 34.5] [--fees yes] [--source live] [--json | --md]
edgeorluck live.csv --backtest backtest.csv
edgeorluck --schema
edgeorluck --serve --port 5078        # the web page, served by waitress
edgeorluck --single-file              # write edgeorluck_single.py
```

## What you get back

A versioned JSON result (`schema_version` 1.0; `edgeorluck.schema()` or `edgeorluck --schema` for the JSON Schema) built so an assistant can answer a non-coder in three sentences without computing anything else:

- `verdict`: `code` (`real_edge`, `cant_tell`, `no_edge`, `losing`, `faded`, `too_early`), `headline` (one sentence, safe to quote), `plain` (2-4 short sentences) and `why`.
- `next_step`: one concrete step about testing, never buy, sell or size advice, with its `why`.
- `ask_the_user`: the questions whose answers would change the verdict, each with `why` and the parameter to set (`answer_with`).
- `what_would_change_it`: for example, "About 140 more trades at this average would settle it."
- `checks`: pass / warn / fail, each with a fix and a plain `why`. Money comes first: when the average percent is positive but the money is negative, it says so.
- "Is it broken, or just a bad streak?": the odds of the latest stretch if nothing changed, e.g. "a drop that deep within 30 trades happens about 1 in 12 times for this strategy".
- Red flags for records that look too good (a very high win rate, no losing streaks, a huge profit factor, almost no drawdown, many small wins with rare huge losses) and an `ai_audit_prompt` to paste to the AI that wrote the bot.
- `numbers`, `compare` (live vs backtest), `input_fingerprint`, `how_we_know` (the method, with references), `limitations`.

## Use it from your AI assistant

The MCP server runs on your machine, so the assistant can pass the path of your export. Its tool `judge_trades` takes `file_path`, `csv_text` or `pnl`, plus `variants_tried`, `benchmark_return_pct`, `fees_included`, `source`, `unit` and the `backtest_...` fields.

These use [uv](https://docs.astral.sh/uv/)'s `uvx`, which fetches the package from PyPI on first use:

Claude Code:

```bash
claude mcp add edgeorluck -- uvx --from "edgeorluck[mcp]" edgeorluck-mcp
```

Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.edgeorluck]
command = "uvx"
args = ["--from", "edgeorluck[mcp]", "edgeorluck-mcp"]
```

Cursor (`.cursor/mcp.json`) and Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "edgeorluck": {
      "command": "uvx",
      "args": ["--from", "edgeorluck[mcp]", "edgeorluck-mcp"]
    }
  }
}
```

**Without uv** (after `pip install "edgeorluck[mcp]"` in a virtual environment), point the same entries at that environment's Python:

```bash
claude mcp add edgeorluck -- /path/to/edge-or-luck/.venv/bin/python -m edgeorluck.mcp_server
```

```toml
[mcp_servers.edgeorluck]
command = "/path/to/edge-or-luck/.venv/bin/python"
args = ["-m", "edgeorluck.mcp_server"]
```

On Windows the Python is `C:\\path\\to\\edge-or-luck\\.venv\\Scripts\\python.exe` (double the backslashes in JSON and TOML).

**ChatGPT and other chats without installs:** download `edgeorluck_single.py` (from the web page, or `edgeorluck --single-file`), upload it together with your trades, and paste:

> I uploaded edgeorluck_single.py and my trades. Run `python edgeorluck_single.py <my file>` (add `--tried N` if I tell you how many versions I tried). Then answer me from its output only: quote the verdict's headline word for word, explain it in short simple sentences, tell me the next step, and ask me its questions. Don't add your own statistics. It is not financial advice.

**One prompt for any AI** (for a trader who can't code):

> Before you tell me my strategy is good, check it with edge or luck (https://github.com/Sakke-code/edge-or-luck): run `edgeorluck.judge()` on my trade export, tell me how many versions I tried (ask me if you don't know), quote its headline word for word, and give me its next step. Don't round "can't tell" up to "good".

[llms.txt](llms.txt) describes all of this for assistants in one page.

## How it judges (the short version)

- **Luck is the default.** A record has to beat what luck does with the same trades. The judge flips each trade's sign at random 1,000 times ("coin-flip copies") to show what luck looks like.
- **You can check any time.** The 95% range is anytime-valid: from 30 trades on (no range is shown before) it stays honest however often you look. A normal confidence interval checked every day is wrong far more often than it says.
- **Every version you tried counts.** All strategies shown and versions tried form one family, and the bar rises with its size.
- **The market is taken out**, roughly, when the input shows prices or a buy-and-hold return.
- **Skewed records need more.** Many small wins with rare big losses must clear a higher bar, and "real edge" needs at least five losing trades.

The method, with references, is in `how_we_know` and [VALIDATION.md](VALIDATION.md): Waudby-Smith, Arbour, Sinha, Kennedy & Ramdas (2021) and Howard, Ramdas, McAuliffe & Sekhon (2021) for anytime-valid ranges; Wang & Ramdas (2022) for e-BH; Newey & West (1987) for autocorrelation; Hall (1987) for the skew correction.

## How often it's wrong

Measured on simulated records whose truth is known ([VALIDATION.md](VALIDATION.md), regenerated by `python tools/validate.py`):

- A strategy with **no edge** is called a "real edge" at most 0.8% of the time, across normal, heavy-tailed, autocorrelated, skewed and martingale-shaped trades at 30 to 1,000 trades (target: 2.5%).
- **Checked every 10 trades** from 30 to 300, a no-edge record is ever called real 0.5% of the time (1.9% when neighbouring trades move together).
- **The best of 100 versions** of a no-edge strategy passes 8.8% of the time when the judge isn't told, and 0% with `variants_tried=100`.
- **The price of that honesty:** a small edge (0.2 standard deviations per trade) is found 60% of the time at 300 trades and 99.7% at 1,000. Below that, the judge says "can't tell yet" and how many more trades would settle it.
- **Same file, same answer:** identical results in a fresh process, every time.

## What it can't tell you

- What happens next. It judges past trades; markets change.
- Whether the backtest itself is honest. It can see signs of a bug (records that look too good), not the bug.
- Anything exact about small samples: the maths is a normal approximation that gets better as trades accumulate.
- Trades are treated as independent draws. Trades that move together (open at the same time, the same market move) count more than they should. It warns when trades overlap.
- The buy-and-hold comparison is rough: it assumes the price moved at a steady pace.

## The web page

`edgeorluck --serve` runs the page and a small JSON API with waitress: `POST /api/judge` (multipart `files`, optional `backtest`, fields `variants_tried`, `benchmark_return_pct`, `fees_included`, `source`, `unit`; or JSON `{sample}` / `{csv_text | pnl | rows, ...}`), `GET /api/schema`, `/api/samples`, `/api/engine`, `/api/track-record`, `/api/validation`, `/llms.txt`, `/healthz`, `/edgeorluck_single.py`. Uploads are read in memory and never stored. The page's own forecasts ("will the next 30 trades end positive?") are kept as one-way fingerprints in a hash-chained ledger (`EDGEORLUCK_DATA_DIR`) and graded when a trader comes back. The library itself writes nothing.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
python tools/validate.py          # VALIDATION.md and src/edgeorluck/data/validation.json
python tools/write_examples.py    # docs/examples/*.json
```

License: MIT.
