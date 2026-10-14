<!-- Generated from src/edgeorluck/guides by tools/write_guides.py: edit the source, not this. -->
# Broken, or just a bad streak?

Every strategy has losing runs. How to tell a normal one from a strategy that has stopped working.

## Losing runs are normal

Take a strategy that wins 55% of its trades, a good one. Over 200 trades, the chance of at least one run of **five losses in a row** is about 88%. The chance of six in a row is about 60%. Losing runs aren't a sign that something broke. They're what winning 55% of the time looks like.

## What the judge checks

**Broken, or a bad streak?** Once there are 60 trades or more, the judge compares your latest 30 trades with your own earlier ones. How often would a stretch this weak come out of your own history? You get the odds in plain words, like "a drop that deep within 30 trades happens about 1 in 12 times for this strategy".

**Faded.** A strategy that had an edge can lose it. The judge says "faded" when:

- it had an edge before;
- its latest 30 or 60 trades lose money;
- and that drop is so unlikely from its own history that luck explains it less than once in 400 times.

Measured on made-up strategies, a real fade is caught 62% of the time. A steady edge is wrongly called faded 1.4% of the time.

:::flip worried
The classic mistake is changing the strategy in the middle of a normal losing run. Then you've tuned it to the bad streak.
:::

## What to do

- **Normal for this strategy:** keep going, and don't change it now.
- **Rare:** pause it and look for what changed: the market, the costs, or the code.
- **Faded:** stop it. Whatever it had may be gone. If you rebuild it, that's a new version and its count starts again.

:::try
Check your latest trades on the [front page](../../README.md#use-it), or run `edgeorluck trades.csv`.
:::
