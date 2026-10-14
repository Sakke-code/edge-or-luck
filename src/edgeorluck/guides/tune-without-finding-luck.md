---
title: Tune settings without finding luck
summary: Trying settings and keeping the best finds the luckiest one. Hold part of the history back, choose on the rest, and judge once.
flip: The best setting on old data is often just the luckiest one.
mood: confused
minutes: 3
group: backtest
order: 2
---

## What goes wrong

Here is a real run, on hourly BTCUSDT prices from January 2024 to October 2026. The rule was "buy when RSI(14) is below *low*, sell when it is above *high*", with 16 settings tried.

- The winning setting averaged **+2.9% per trade** on the part of the history that picked it.
- On the part it never saw, the same setting averaged **-3.4% per trade**.
- It ranked 15th of the 16 there, and only 2 of the 16 settings made money at all.

Nothing was wrong with the code. The winner was simply the setting that fit that stretch of noise best.

## Hold some history back

1. Split the prices: the first 70% to choose with, the last 30% kept unseen.
2. Try every setting on the first part only.
3. Judge the chosen setting once on the unseen part.
4. Don't go back and tune again after seeing that result. If you do, the unseen part becomes tuning data too, and you need fresh data to judge.

`edgeorluck.tune` does all four:

```python
import edgeorluck as eol
px = eol.prices("BTCUSDT", "1h", start="2024-01-01", end="2026-10-01")
tuned = eol.tune(px, {"buy": "rsi(14) < {low}", "sell": "rsi(14) > {high}"},
                 {"low": [20, 25, 30, 35], "high": [60, 65, 70, 75]})
print(tuned.to_markdown())
```

## Reading the result

- **The verdict on the unseen part is the honest one.** Quote that, not the first part.
- **How the other settings did there.** If most settings make money on unseen data, the idea itself may have something, and the exact setting matters less. If only a few do, the winner was mostly luck.
- **The first part** is shown too, with the bar raised for every setting tried, so it can't pass easily.
- **Too few unseen trades?** Use a longer history or a bigger holdout (`holdout=0.4`). Don't fall back on the first part.

:::flip confused
Twenty settings next to each other, like 20, 21, 22 and so on, mostly repeat one another. Five far apart tell you more.
:::

## Fewer, wider settings

Every extra setting raises the bar a little. Try a few settings far apart (20, 30, 40) rather than many close together. If the idea only works at exactly 27, it is probably not an idea.

:::try
`edgeorluck tune --buy "rsi(14) < {low}" --sell "rsi(14) > {high}" --grid "low=20,25,30 high=65,70,75"`
:::
