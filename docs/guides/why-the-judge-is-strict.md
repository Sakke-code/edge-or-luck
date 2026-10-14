<!-- Generated from src/edgeorluck/guides by tools/write_guides.py: edit the source, not this. -->
# Why the judge is so strict

Most good-looking records are luck. Why "can't tell yet" is the honest answer so often, what it takes to get past it, and why a plain t-test is too easy to fool.

## Luck makes beautiful records

Toss a fair coin 100 times and you will very likely see a run of five or more heads somewhere. A strategy with no edge behaves the same way: some stretches look brilliant. Three habits then make luck look like skill:

- **Trying many versions** and keeping the best. The best of 20 coins looks great.
- **Checking often** and getting excited on a good day. You end up reading the luckiest moment.
- **A rising market.** When prices go up, anything that buys makes money.

So the judge starts from "this is luck". It only moves when the record is too strong for luck to explain, after accounting for all three.

## How strong is too strong?

The judge turns the record into one number, **the evidence**. With no edge, the evidence averages exactly 1 (that is proved, see [How it checks](../../README.md#how-it-judges-the-short-version)). A "real edge" needs evidence of at least **40**, and 40 × N when you tried N versions.

That bar is what keeps it honest. On thousands of made-up strategies with no edge, it called a real edge 0.15% of the time. Checked again and again as the trades came in, it was fooled at most 1.9% of the time.

## The price: it's slow

Strict means slow. Take a small but real edge, where the average trade is a fifth of a typical trade's swing up or down. The judge finds it:

- 12% of the time at 100 trades;
- 60% of the time at 300 trades;
- 99.7% of the time at 1,000 trades.

Until then it says **"can't tell yet"**, and when it can, it estimates how many more trades would settle it if the average holds.

:::flip shrug
"Can't tell yet" isn't "no". It means: good so far, but luck makes records like this too often to be sure.
:::

## What to do with "can't tell yet"

1. Keep it small, or on paper.
2. Don't change the strategy while you wait. A changed strategy is a new version, and its count starts again.
3. Check as often as you like. The range is built to stay honest however often you look, so you don't need to wait for a magic number of trades.

## Why not just a t-test or a Sharpe ratio?

A plain t-test assumes you looked once, at a number of trades you fixed in advance, and that you tried one version. A trader does none of those things. Checking often, trying versions and riding a rising market each make a plain test look better than it should. The judge corrects for all three, and needs at least 30 trades and 5 losing trades before it will say "real edge".

:::try
Drop your trades on the [front page](../../README.md#use-it), or run `edgeorluck trades.csv --tried 20`.
:::
