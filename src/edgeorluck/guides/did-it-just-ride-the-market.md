---
title: Did it just ride the market?
summary: In a rising market, any strategy that buys makes money. How the judge takes the market out, and how rough that is.
flip: Up 1.5% while the coin doubled? That's not an edge, that's a passenger.
mood: confused
minutes: 2
group: verdict
order: 3
---

## Every buyer wins in a bull market

From January 2024 to October 2026, simply holding BTC returned about **+97%**. Over the same dates, a classic rule ("buy when RSI(14) is below 30 and the price is above its 200-bar average, sell when RSI is above 70") made **+1.5%** on hourly bars.

It made money, but it was far worse than doing nothing. A record like that says nothing about skill until the market is taken out.

## What the judge does

1. **It finds out what holding returned** over the same dates. That comes from your export (TradingView and backtesting.py results include it), from the trades' own prices, or from you (`--benchmark 34.5` means holding made +34.5%).
2. **It estimates the market's part of every trade.** When it knows each trade's dates, it works out what holding would have made over that trade's time, and takes that out.
3. **It judges what's left.** If nothing positive is left, the verdict is "no edge", even when the strategy made money.

For short trades it works the other way round: a rising market works against them, and the judge credits that.

## How rough is it?

The estimate assumes the market rose at a steady pace over the whole period. If the price jumped while your strategy was out of the market, the estimate is off. That's why the result calls it a rough check.

:::flip confused
The real test: does it make money when prices fall or go sideways? An edge should survive those too.
:::

:::try
`edgeorluck trades.csv --benchmark 34.5`. A backtest from `edgeorluck.backtest` brings its own benchmark.
:::
