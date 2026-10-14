---
title: Is live keeping the backtest's promise?
summary: Live results usually come in below the backtest. How to tell a normal gap from a backtest that was wrong.
flip: Backtests are optimists. Let's see how far reality lags.
mood: wait
minutes: 2
group: backtest
order: 4
---

## Expect some gap

Live trading pays costs a backtest often skips: real slippage, fills a little late, an order that didn't fill at all. On top of that, the backtest was picked because it looked good, so part of its result was luck that won't repeat. A small gap is normal. The question is whether the gap is bigger than luck explains.

## How the judge checks

Give it both sets of trades:

```bash
edgeorluck live.csv --backtest backtest.csv
```

- From 10 live trades on, it compares the live average per trade with the backtest's.
- It says **worse than the backtest** only when the whole range for the difference is below zero.
- That range stays valid however often you check, so checking after every trade is fine.
- Next to it come plain odds: if the live trades were drawn from the backtest's own trades, how often would a live record this weak turn up?

Measured on made-up strategies:

- When live really is worse, the judge catches it by 200 live trades {{compare_caught}} of the time.
- When live is as good as the backtest, it wrongly says "worse" {{compare_false}} of the time, even when checked every 10 trades.

## What to do with the answer

- **Consistent with the backtest:** keep going. As live trades pile up, judge them on their own too.
- **Worse than the backtest:** stop and find the gap. The usual suspects, most likely first:
  1. fills (slippage, or trading the signal's own close in the backtest);
  2. fees missing in the backtest;
  3. a look-ahead in the backtest (run it through the honest backtester and look at the one-bar-later check);
  4. a change in the market.

:::flip wait
A backtest that loses most of its edge one bar later, and a live record that's worse than its backtest, are often the same bug seen twice.
:::

:::try
`edgeorluck live.csv --backtest backtest.csv`, or drop both files on the [front page](/).
:::
