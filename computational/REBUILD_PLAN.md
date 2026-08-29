# Rebuild Plan — "Latest Run" notebooks

Goal: rebuild the computational pipeline so it tests what *Berman's actual theory*
claims (see `CONCEPTUAL_AUDIT_vs_Berman.md`), with every code defect fixed
(`../../Github Repository/BermanModeling/REVIEW_NOTES.md`). Go notebook by
notebook; keep/copy what is faithful, rewrite what is not, build what is missing.

---

## 0. What we are now testing (the reframed goal)

> Can principled positivism's formal structure — *nonlexical* aggregation of
> *practice-grounded* weights — bear the weight Berman places on it?

Reframed around Berman's actual commitments:
- **Faithful & kept:** stipulated nonlexical SAW works (tolerates rough weights,
  handles conflict, delivers more law than Hart, exploits correlation).
- **The real open problem:** the *mechanism* of practice-grounding. Berman grounds
  weights in **uptake + embeddedness/reliance** (JESP 321–22) — not fitness
  selection (ABM) or reward optimization (MARL). The theory-faithful mechanism has
  never been implemented. **This becomes the centerpiece.**
- **Retired/relabeled:** the sweeping "impossibility." C5 (graded weights) is not a
  Berman requirement (he permits equal weights, 323–24); Lexico/Cascade are the
  *lexical* foil the theory rejects (311–13, 333), not in-theory options.

---

## 1. New folder + shared-core architecture

```
Experiment/Run_2026-06/                 # clean path (no trailing-space parent)
├── berman_core.py                      # ONE canonical core — ends verbatim-copy risk
├── notebooks/
│   ├── 0_static_properties.ipynb       # Part IV  (REWRITE-LIGHT)
│   ├── 1_abm_emergence.ipynb           # Part V   (REWRITE)
│   ├── 1b_threshold.ipynb              # Part IX threshold (REWRITE; drop τ artifact)
│   ├── 1c_reliance_embedding.ipynb     # NEW — Berman's actual mechanism
│   ├── 2_mechanism_comparison.ipynb    # Part VI  (REWRITE + add reliance)
│   ├── 3_aggregation_functions.ipynb   # Part VII (REWRITE; reclassify lexical foil)
│   ├── 4_robustness_search.ipynb       # Part VIII(REWRITE; reframe from "impossibility")
│   └── 5_publication_figures.ipynb     # REWRITE to new results
├── data/                               # outputs, one subfolder per notebook
└── REBUILD_PLAN.md                     # (copy of this file)
```

**Shared core (kills the verbatim-copy bug, finding #47/#49):** all model code
(`Principle`, `Case`, `SAWModel`, the ABM, MARL, the six aggregation functions,
the criteria battery, a *seeded* `bootstrap_ci`, a single consistent determinacy
cutoff) lives in `berman_core.py`. Each notebook's first cell `%%writefile`s the
canonical core (so Colab is self-contained) then `from berman_core import *`. One
source of truth; fix once, fixed everywhere.

**Output paths:** one `OUTDIR` constant per notebook (env-overridable), no
hardcoded Drive paths.

---

## 2. Criteria redesign (the intellectual core)

Split "what Berman *requires*" (pass/fail) from "what's *interesting to measure*"
(reported descriptively, never used to fail the theory).

| Old | New | Status | Why |
|-----|-----|--------|-----|
| C1 stability | **R1 well-definedness** | requirement (pass/fail) | weights nondegenerate — fair |
| C3 tolerance ≥±0.07 | **R2 rough weights suffice** | requirement | Berman's explicit Perry response (328–29) |
| *(none)* | **R3 nonlexicality** | requirement **(NEW)** | THE structural claim: outcomes track activation *magnitudes* & multiple principles jointly, not just signs/ranks (operationalized as sensitivity to magnitude-only perturbation). Directly tests lexical-vs-nonlexical (311–13, 333). |
| C2 Hart adv. ≥+19.1% | **R4 more law than Hart** | requirement, **as a margin/curve** | Berman's claim is qualitative (answer too-little-law), not a fixed %. Report the advantage curve vs Hart variants; positive margin = pass. |
| C5 graded differentiation | **D1 weight gradation** | *descriptive only* | Berman permits equal weights (323–24). Measure & report; never fail the theory on it. |
| C4 correlation benefit | **D2 correlation benefit** | *descriptive only* | A robustness observation (independent baseline is conservative), not a theory requirement. |

Consequence: the "two incompatible pairs" that generated the impossibility
(C2/C4, C3/C5) are dismantled — C4 and C5 are no longer pass/fail. The honest
question becomes whether each *mechanism* yields weights satisfying R1–R4.

---

## 3. Notebook-by-notebook

| # | Notebook | Action | What it tests (Berman-faithful) | Key fixes |
|---|----------|--------|---------------------------------|-----------|
| core | `berman_core.py` | **BUILD** | — | seeded bootstrap; one determinacy cutoff; correct category map (federalism≠moral); no shuffle of position-dependent inputs |
| 0 | Static Properties | **REWRITE-LIGHT** | R2 (rough weights), R4 (Hart margin), D2 (correlation), R3 (nonlexicality via cardinal-vs-ordinal, recast as "magnitude matters"); add a 5-bin *coarse-cardinal* condition matching Berman's "slight/moderate/weighty" | 30 seeds (not 20); fix Exp 2b variance; keep TVA v. Hill + add Obergefell worked example (Berman Fig 11) |
| 1 | ABM Emergence | **REWRITE** | Does fitness-ABM produce well-defined, tolerant, nonlexical weights? Bimodality reframed as the *legal/sub-legal threshold* (Bermanian, good), not a C5 "failure" | actually call `simulate()` in the correlation exp; consistent cutoff |
| 1b | Threshold | **REWRITE** | τ is practically-implied: flip-rate robustness threshold (≈0.137) is THE threshold | fix robustness direction (sustained crossing); **drop the endogenous-MARL-τ artifact** |
| 1c | **Reliance/Embedding** | **NEW (centerpiece)** | Berman's *actual* mechanism: weight ∝ dispositive uptake embedded in precedents that remain good law. Distinguish dispositive vs incidental invocation; precedent graph w/ reliance + overruling | the experiment the paper never did |
| 2 | Mechanism Comparison | **REWRITE** | Compare ABM / MARL / **reliance** on R1–R4 (+ D1/D2 descriptive). Reframe "mechanism dependence" as "two proxies + the faithful one" | fix MARL/Hybrid C3 force-log; fair Hart tiers; verify torch/GPU actually used |
| 3 | Aggregation Functions | **REWRITE** | Among **nonlexical** functions (SAW, Nonlinear, Multi-Level): does choice matter? Lexico/Cascade run **explicitly as the lexical foil** ("even the mode Berman rejects doesn't dominate") | **remove the `rng.shuffle(a)` bug**; align criterion labels to R1–R4 |
| 4 | Robustness Search | **REWRITE** | Reframed: "is the Berman-faithful combination (nonlexical × reliance × robustness-τ) defensible on R1–R4?" + robustness across blends/variants | drop τ artifact; honest result, not "impossibility" |
| 5 | Publication Figures | **REWRITE** | regenerate from new data | reproducible from committed CSVs |

"Pipe/copy the ones that are fine": no notebook is fully clean, but Notebook 0's
**experiment logic** (conflict types, Hart variants, correlation) is faithful and
is largely *copied* into the rewrite — the changes are the shared core, seeds,
and reframing, not the experiments themselves.

---

## 4. Execution order & per-notebook workflow (the step-by-step)

We work strictly top-to-bottom; each notebook is finished and reviewed before the
next, so a problem surfaces early.

**Per notebook, every time:**
1. **Spec** — I write a 1-paragraph "what this tests and why (Berman cite)" at the top.
2. **Build** — I write/rewrite the notebook against `berman_core.py`.
3. **Run** — I execute it (locally where possible; see decisions) and capture data.
4. **Verify** — I check the numbers are sane and self-consistent; I report them to you.
5. **Review gate** — you eyeball the result; we adjust or move on.

**Global order:**
1. `berman_core.py` (build + unit-sanity-check).
2. NB 0 → establishes the reference σ\* and Hart-margin values the criteria use.
3. NB 1 (ABM) → emergence + bimodality, reframed.
4. NB 1c (**reliance** — the new science) → before NB 2, since NB 2 compares to it.
5. NB 1b (threshold) → robustness-τ ≈0.137 as the single threshold.
6. NB 2 (mechanism comparison: ABM vs MARL vs reliance on R1–R4).
7. NB 3 (aggregation functions, reclassified).
8. NB 4 (robustness search, reframed).
9. NB 5 (figures) → regenerate everything.

After the run: update the manuscript Parts V–VIII to match, and refresh the public
repo + README from the verified outputs.

---

## 5. Decisions needed before building
1. **Where to run** — local-verified vs Colab-targeted vs hybrid.
2. **Criteria/framing** — adopt the full R1–R4 + D1/D2 redesign, or a lighter relabel.
3. **Reliance mechanism** — build NB 1c as the centerpiece, or keep smaller.
4. **Folder name/location** — `Experiment/Run_2026-06/` or your preference.

---

## UPDATE (org decisions, confirmed)

**Clean sequential numbering** (legacy 1b/1c letters dropped):

| # | file | does | Part | status |
|---|------|------|------|--------|
| 0 | `0_static_properties` | SAW given weights | IV | done |
| 1 | `1_abm_emergence` | simple uptake mechanism; bimodal split | V | done |
| 2 | `2_reliance_mechanism` | **Berman's actual mechanism** (reliance + winning conflicts) | V-VI | next (centerpiece) |
| 3 | `3_mechanism_comparison` | uptake vs reliance vs strategic (MARL), across R1-R4 | VI | |
| 4 | `4_threshold` | is tau required? robustness threshold | IX | |
| 5 | `5_aggregation_functions` | SAW vs lexical foil etc. | VII | |
| 6 | `6_robustness_search` | reframed "impossibility" | VIII | |
| 7 | `7_figures` | publication figures from all CSVs | - | |

**Data + figures.** Every notebook writes `data/notebookN/*.csv` + `notebookN_summary.json` (the durable, citable results). Each notebook also saves 2-3 inline diagnostic PNGs to `figures/`; `7_figures` reads all CSVs to produce the polished paper figures (PDF+PNG, one style). NB0/NB1 to be retrofitted with their key plots.
