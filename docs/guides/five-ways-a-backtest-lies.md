<!-- Generated from src/edgeorluck/guides by tools/write_guides.py: edit the source, not this. -->
# Five ways a backtest lies

Quick backtests, especially ones an AI writes in a hurry, are usually too kind. The five common leaks, how to spot each one, and how to close it.

## 1. It trades at a price it couldn't have had

A signal made from a bar's close only exists once that bar has closed. The earliest you can trade on it is the next bar. Many backtests quietly trade at the same close the signal used, or even earn the move that created the signal.

The classic version in pandas:

```python
pos = (rsi < 30).astype(int)
too_kind = pos * close.pct_change()            # today's signal earns today's move
honest   = pos.shift(1) * close.pct_change()   # trade after the signal, not before
```

The first line looks innocent and can turn a losing rule into a winning one.

**Close it:** fill every decision at the next bar's open. The edge or luck backtester always does.

## 2. It peeks at the future

Some tools look at later bars without saying so:

- `shift(-1)`, or any label made from later prices;
- centred moving averages, and pivots or zigzags that confirm a bar only later;
- TradingView scripts that repaint, or `request.security` with lookahead on;
- scaling with the mean, max or standard deviation of the whole data set.

**The test:** delete the bars after some day. Does the rule's decision on that day change? If it does, the rule is using the future. `edgeorluck.backtest` runs exactly this test on a Python rule, at 20 spread-out bars, and refuses to judge a rule that fails.

:::flip worried
If deleting tomorrow changes today's decision, today's decision knew about tomorrow.
:::

## 3. It forgets fees and slippage

A typical spot fee is 0.1% on the buy and 0.1% on the sell, plus a little slippage on each fill. That is about 0.3% per round trip. A strategy that makes 0.2% a trade before costs **loses** money after them, and fast strategies feel it most.

**Close it:** charge a fee and slippage on every fill. The backtester does (0.1% and 0.05% by default, so set your exchange's real numbers). When you bring your own results, tell the judge whether fees are included.

## 4. It was tuned until it worked

Try enough settings and one will fit the noise of the past perfectly. It won't fit the future. This is the most common way a good-looking backtest fails live.

**Close it:** tune on one part of the history and judge on a part the tuning never saw, and count every version you tried. See [Tune settings without finding luck](tune-without-finding-luck.md) and [Count your versions](count-your-versions.md).

## 5. It only tested coins that survived

Testing on today's top coins means testing on winners. Coins that crashed, were delisted or dropped out of the top 50 aren't in today's list, so the past looks rosier than it was.

**Close it:** pick the coins as they were at the start of the test, or fix the list in advance and keep the losers in. This is the one leak edge or luck can't check for you.

## The quick check

Ask your AI to run the idea through the honest backtester instead of writing its own:

```python
import edgeorluck as eol
px = eol.prices("BTCUSDT", "1h", start="2024-01-01")
print(eol.backtest(px, {"buy": "rsi(14) < 30", "sell": "rsi(14) > 70"}).to_markdown())
```

It closes leaks 1 to 3 by construction. It also runs everything again with fills one bar later, which catches edges that live on instant fills. It compares the result with simply holding and hands the trades to the same strict judge.

:::try
`pip install edgeorluck`, then `edgeorluck backtest --buy "rsi(14) < 30" --sell "rsi(14) > 70" --symbol BTCUSDT`.
:::
