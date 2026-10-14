---
title: Count your versions
summary: The best of many tries looks good by luck alone. How to count what you tried, and how the judge raises the bar to match.
flip: Tell me how many you tried. I promise I won't laugh.
mood: idle
minutes: 2
group: verdict
order: 2
---

## The best of many is lucky by design

Test 100 settings of a strategy that has no edge, and the best one will still look like a winner. Measured on made-up strategies with no edge:

| Versions tried | Called a "real edge" if you don't say | ...if you do say |
|---|---|---|
| 10 | {{best10_without}} | {{best10_with}} |
| 100 | {{best100_without}} | {{best100_with}} |
| 1,000 | {{best1000_without}} | {{best1000_with}} |

The judge can't see how many versions you tried. It only knows if you tell it.

## What counts as a version

Count every version whose results you looked at, not just the ones you kept:

- every setting you tried (RSI below 30, 25 and 20 is three versions);
- every change to the rule ("added a trend filter" is a new version);
- every coin or timeframe you tried it on and dropped;
- versions your AI tried for you along the way.

A rough number is fine. If you're unsure, round up. Too high costs you a little; too low can fool you.

## How the bar rises

One version needs evidence of 40. When you tried N versions, the best one needs 40 × N. The versions you didn't show still count, as if they had no evidence at all, so hiding them doesn't help.

:::flip idle
Used tune()? It counts the settings for you, and judges the winner on data none of them saw.
:::

:::try
`edgeorluck trades.csv --tried 50`, or `edgeorluck.judge("trades.csv", variants_tried=50)`. The front page asks you too.
:::
