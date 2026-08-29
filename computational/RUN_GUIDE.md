# Reproducibility run guide

Purpose: after this run, **every number in the manuscript comes out of a cell you executed.**
Before these changes, several figures in Parts VI, VII and VIII were computed outside the notebooks
and could not be regenerated. That gap is now closed.

## What changed

Only three notebooks were edited. The other seven are untouched.

| Notebook | Change |
| --- | --- |
| `5_aggregation_functions.ipynb` | New cell **E2b**: the equal-count closeness deciles Part VII prints, the closeness statistics, and verdict agreement restricted to the cases the gated rule decides. Plus 7 new self-check guards. |
| `8_figures.ipynb` | F1's left bar relabelled from "Hartian hierarchy" to "Hart (validation rule)". The old label contradicts the manuscript in three places. |
| `9_appointment_embedding.ipynb` | New cell **Exp G**: the two worked cases, the perturbation band, the mapping-robustness sweep, the tau = 0 check, and the tier-balance test. Plus 9 new self-check guards. |

Backups of the originals sit beside each file as `*.ipynb.bak`.

## Run order

Run all ten in order. Notebooks 0 through 7 and 9 write the data; notebook 8 reads what they wrote,
so it must come after. Each ends in a self-check that raises if anything drifts, so a green run is
the reproducibility claim.

1. `0_static_properties.ipynb`
2. `1_abm_emergence.ipynb`
3. `2_reliance_mechanism.ipynb`
4. `3_strategic_mechanism.ipynb`
5. `4_threshold.ipynb`
6. `5_aggregation_functions.ipynb`  **(changed)**
7. `6_global_sensitivity.ipynb`
8. `7_analytical_results.ipynb`
9. `9_appointment_embedding.ipynb`  **(changed)**
10. `8_figures.ipynb`  **(changed, and must run last)**

## What to check as you go

Each notebook prints `ALL CHECKS PASSED` at the end and then asserts. If a notebook stops on
`self-check failed`, send me the failing check name and stop there rather than continuing.

Two specifically worth eyeballing:

- **Notebook 5** should print a decile profile beginning `0.449` and ending `0.0`, and the closeness
  line `largest |F| on a disagreement 1.61 against an observed maximum of 4.44`.
- **Notebook 9** should print `TVA_Burger` with all four mechanisms positive, `Obergefell` with
  uptake negative and reliance and appointment positive, and a tier-balance line showing the
  divergence falling from about `0.361` to about `0.303` when the tiers are balanced but not
  collapsing.
- **Notebook 8** should render F1 with its left bar reading **Hart (validation rule)**. If it still
  says "Hartian hierarchy", the edit did not take and the figure will contradict the text.

## After the run

Tell me it is green and I will re-verify the manuscript against the freshly written
`data/notebook*/` outputs, so any figure that shifted in the last digit gets synced to what your run
actually produced.
