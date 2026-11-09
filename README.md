# More Determinate and Less Settled

**David G. Kamper**
University of California, Los Angeles
University of Chicago Law School

Replication package for a law review article giving Mitchell Berman's principled
positivism its first systematic computational test. The manuscript itself is under
review and is not included here; it is available from the author.

---

## The argument

Berman's principled positivism holds that legal content is fixed by a weighted balance
of legal principles, each contributing force in proportion to its weight and to how
strongly the facts of a case activate it. Formally, `F = Σ wᵢ · aᵢ`.

The Article reaches two findings.

**The formal apparatus is sound.** Given faithful weights, the weighted sum returns a
single determinate verdict in exactly the cases where a Hartian validation rule
contradicts itself. It tolerates rough weights, weighs magnitude rather than ranking by
fixed priority, and leaves less law unsettled than a tiered Hartian hierarchy. Several
of these properties hold as theorems rather than as artifacts of a run.

**That soundness rests on an appeal to practice the theory leaves underspecified.** Four
mechanisms by which practice might confer weight all clear every requirement the theory
imposes, yet they ground the principles into three distinct weight structures, and those
structures decide about a third of cases differently.

Principled positivism therefore does not resolve the indeterminacy it inherited. It
relocates it, from how principles combine, which it handles well, to which practice
grounds the weights, and to the prior question of how legal grounds themselves change.

---

## The four requirements

Every mechanism and every aggregation rule is scored against the same four requirements,
each drawn from something Berman claims or commits to.

| Requirement | Operational test |
| --- | --- |
| Well-defined | At least two principles carry weight above a small cutoff |
| Noise-tolerant | Critical uncertainty σ\* ≥ 0.07 |
| Magnitude-weighing | On the magnitude probe, the weighted sum changes its verdict more than five times as often as a lexicographic rule |
| More law than Hart | A positive determinacy advantage over the hierarchical Hart baseline |

Two properties earlier work associated with the theory, weight gradation and correlation
benefit, are measured but are **not** requirements. Berman's account permits principles
to carry equal weight.

## The four mechanisms

| Mechanism | Weight is conferred by |
| --- | --- |
| Uptake | How often the bench invokes a principle |
| Reliance | Embedding in the holdings that go on relying on it |
| Strategic | Judges adapting to which arguments move their colleagues |
| Appointment | Bench turnover, read out through surviving holding-support |

## Three Harts, kept distinct

The Article runs three separate Hartian comparisons and never conflates them.

- **A Hartian validation rule** is the contradiction foil. It asks of each standard only
  whether it is valid and which way it points, so on a genuine conflict it certifies both
  sides at once.
- **The hierarchical Hart** is the determinacy baseline. It sorts the principles into
  ranked tiers and decides at the highest tier holding an applicable principle, and only
  where enough of that tier's principles point the same way.
- **A consensus Hart** is a flat share bar, deciding when a fixed share of the principles
  agree.

---

## Repository structure

```
computational/
  berman_core.py            shared engine: principles, activations, aggregation, baselines
  sa_toolkit.py             global sensitivity analysis helpers
  notebooks/                0 through 9, listed below
  data/notebook0..9/        saved outputs, CSV plus a summary JSON per notebook
  figures/                  supplement figures
  figures/pub/              F1 to F5 as published

index.html                  public landing page, served at the Pages URL
assets/                     figures for that page
```

The project site is at **https://dgkamper.github.io/berman-modeling/**, which serves the
paper, the notebooks, and the data whether or not the repository itself is public.

### Notebooks

Run in dependency order. Notebooks 0 and 7 establish the static results; 1, 2, 3 and 9
are the four mechanisms; 4, 5 and 6 are the robustness Parts; 8 rebuilds every figure.

| Notebook | What it establishes | Part |
| --- | --- | --- |
| `0_static_properties` | Static model: determinacy, noise tolerance, the Hart comparisons, the magnitude probe | IV |
| `1_abm_emergence` | Uptake: weights emerging from a simulated bench, the live/dead cut, the cluster gap | V |
| `2_reliance_mechanism` | Reliance: weight from embedding in holdings | VI |
| `3_strategic_mechanism` | Strategic adaptation, trained by REINFORCE | VI |
| `4_threshold` | The determinacy threshold swept from 0 to 0.30 | VIII |
| `5_aggregation_functions` | Ten aggregation rules across cardinal, ordinal, and lexical classes | VII |
| `6_global_sensitivity` | Sobol variance decomposition over four inputs | VIII |
| `7_analytical_results` | The five theorems and the scope each is proved under | IV.D |
| `8_figures` | Rebuilds F1 to F5, the theorems table, and every caption | all |
| `9_appointment_embedding` | Appointment, and the decoupling test against reliance | VI |

Notebooks were developed and verified in Google Colab. `berman_core.py` is duplicated
inside `notebooks/` so a Colab session can import it without a path fixup; the two copies
are byte-identical.

### Reproducing

```bash
pip install -r requirements.txt
jupyter lab computational/notebooks
```

Each notebook writes to `data/notebookN/` and carries its own self-checks. Run 0 and 7
before 8, which reads their summary JSON.

---

## Headline figures

All numbers are computed on synthetic cases with independent activations, the
conservative baseline. Correlation only improves the model's performance, so every figure
below should be read as a floor. The one exception is the margin over Hart, which
reverses under category-correlated activations and is reported as such in Part IV.

- Determinacy 86.2% in general, 79.2% on conflict cases.
- A validation rule contradicts itself on 100% of directly opposed cases; the weighted
  sum on 0%.
- Critical uncertainty mean 0.090, rising from 0.069 to 0.156 under category correlation.
- Margin over the hierarchical Hart +0.272, positive on every seed.
- The weighted sum responds to a magnitude-only perturbation about 9.7 times as often as
  a lexical rule.
- With equal weights the sum and a vote-count disagree on 15.4% of cases, reproduced at
  15.3% on a separate implementation.
- Four mechanisms, three structures: reliance and appointment correlate at +0.93, while
  uptake opposes both. The four reach different verdicts on 36.1% of cases and opposite
  determinate verdicts on 27.2%.

---

## Status

The manuscript is a working draft of August 2026. Two citations are pending permission
from Professor Berman: his book in progress, *To Say What the Law Is*, which is marked
"Do Not Quote or Cite Without Permission," and correspondence of March 2026. Neither
should be quoted onward.

Parts VIII and IX of earlier drafts framed the argument as an impossibility result
against five criteria. That framing is retired: it tested a target stronger than Berman
commits to. The retired sections are excluded from the build and from this
repository.

## License

MIT, see `LICENSE`. The manuscript text is not covered by that license and remains the
author's.
