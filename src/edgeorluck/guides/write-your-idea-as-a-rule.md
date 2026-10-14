---
title: Write your idea as a rule
summary: Turn a trading idea into a rule the honest backtester can run, with common setups already translated and every function in one table.
flip: Say it in my words, and I can test it without letting it cheat.
mood: happy
minutes: 3
group: backtest
order: 3
---

## The shape of a rule

A rule has a way in and a way out:

- **buy**: a condition, checked at the close of every bar, like `rsi(14) < 30`.
- **A way out**, at least one of:
  - **sell**, a condition like `rsi(14) > 70`;
  - **stop_loss_pct**, a loss in percent from your entry;
  - **take_profit_pct**, a gain in percent;
  - **max_hold_bars**, the most bars to stay in.

Every decision is filled at the next bar's open, with fees and slippage. Stops and targets are checked at closes too, because inside one bar nobody can know whether the high or the low came first. Add `"side": "short"` to sell first.

## Common ideas, translated

| The idea | buy | way out |
|---|---|---|
| Oversold, then overbought | `rsi(14) < 30` | sell `rsi(14) > 70` |
| Golden cross | `crosses_above(sma(50), sma(200))` | sell `crosses_below(sma(50), sma(200))` |
| MACD cross | `crosses_above(ema(12) - ema(26), ema(ema(12) - ema(26), 9))` | sell `crosses_below(ema(12) - ema(26), ema(ema(12) - ema(26), 9))` |
| Breakout on volume | `close > prev(highest(high, 20)) and volume > 2 * sma(volume, 20)` | stop 5%, hold 48 bars |
| Buy a sharp dip | `pct(24) < -8` | take profit 6%, hold 72 bars |
| Bollinger band bounce | `close < sma(20) - 2 * std(20)` | sell `close > sma(20)` |
| Big candle | `change(1) > 2 * atr(14)` | hold 12 bars |
| Only in an uptrend | add `and close > sma(200)` to any buy | |

## Every function

| Function | What it gives |
|---|---|
| `sma(n)`, `ema(n)` | Moving averages. `ema` matches TradingView's `ta.ema`. |
| `rsi(n)` | Wilder's RSI, 0 to 100, as TradingView's `ta.rsi`. |
| `highest(n)`, `lowest(n)` | The highest or lowest of the last n bars, this one included. |
| `std(n)`, `atr(n)` | Standard deviation and average true range, as TradingView's `ta.stdev` and `ta.atr`. |
| `change(n)`, `pct(n)` | The difference from n bars ago, or the change in percent. |
| `prev(x)`, `prev(x, n)` | The value one bar ago, or n bars ago. |
| `crosses_above(a, b)`, `crosses_below(a, b)` | True on the bar where a crossed b. |
| `abs(x)`, `min(a, b)`, `max(a, b)` | Absolute value, the smaller, the larger. |

The prices are `open`, `high`, `low`, `close` and `volume`. A function given just a number works on the close, so `sma(20)` means `sma(close, 20)`. Combine conditions with `and`, `or` and `not`.

:::flip happy
`highest(high, 20)` includes this bar, so a close can never be above it. For a breakout, compare with the bar before: `close > prev(highest(high, 20))`.
:::

## Why a language at all?

A rule written this way **can only look back**: every function uses the bars up to now and nothing after. It is also safe to run from a web page or an AI, because nothing else is allowed in it: no attributes, no indexing, no imports, no code.

Need something it can't say? Write a Python function that takes the price table and returns 1 (long), 0 (flat) or -1 (short) for each bar. The backtester will check it for look-ahead before judging it.

:::try
```python
import edgeorluck as eol
px = eol.prices("ETHUSDT", "4h", start="2023-01-01")
rule = {"buy": "close > prev(highest(high, 20)) and volume > 2 * sma(volume, 20)",
        "stop_loss_pct": 5, "max_hold_bars": 48}
print(eol.backtest(px, rule).to_markdown())
```
:::
