# The judge's maths, machine-checked

Every exact mathematical claim edge or luck makes is proved here in [Lean 4](https://lean-lang.org) with
[Mathlib](https://github.com/leanprover-community/mathlib4). A claim that is only approximate on real trades
(the estimated spread, non-normal or dependent trades) is not dressed up as a proof: it is measured in
[VALIDATION.md](../VALIDATION.md) and labelled as measured on the page's "How Flip checks" page.

| File | What it proves |
|---|---|
| `EdgeOrLuck/Mixture.lean` | The e-value's closed form (the code's `log_e_standardized`), that it averages exactly 1 when there is no edge (at most 1 when the edge is negative), that it rises with the evidence, that the code's bisection finds its boundary, and that "the 95% range is above zero" and "real edge" are the same statement. |
| `EdgeOrLuck/Anytime.lean` | Ville's inequality for nonnegative supermartingales, that the Gaussian likelihood-ratio mixture is one, and so that the 95% range holds at every trade count at once: checking again and again does not fool it. |
| `EdgeOrLuck/Selection.lean` | e-BH controls false discoveries under any dependence, the code's procedure is self-consistent, versions not shown never get selected, and one strategy with N versions needs e ≥ 40N. Also: Brier scoring is proper, the money-vs-percent sentence, Wilson, and the live-vs-backtest range. |
| `EdgeOrLuck/Counting.lean` | The losing-streak odds are exact, the coin-flip test's counting lemma and its validity, the serial-correlation factor, FIFO bookkeeping of exchange fills, the drawdown, and the time in the market (the length of the union of the trades' spans). |

Each main theorem's docstring names the Python function or the page sentence it backs, and the Python side is
tied to the theorems by the `tests/test_bridge_*.py` tests (the code satisfies the theorems' hypotheses and
conclusions on many random inputs).

## Check them yourself

```bash
cd proofs
lake exe cache get   # Mathlib's prebuilt files (a few GB)
lake build           # every file; zero errors means every proof checks
```

Each file ends with `#print axioms` for its main theorems: only Lean's standard `propext`, `Classical.choice`
and `Quot.sound` appear. No `sorry`, no new axioms, no `native_decide`.
