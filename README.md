# Computational Jurisprudence: Testing Berman's Principled Positivism

**David G. Kamper**  
University of California, Los Angeles  
[davidgkamper@ucla.edu](mailto:davidgkamper@ucla.edu)

This repository contains the complete code, data, and figures for "Computational Jurisprudence: Testing Berman's Principled Positivism Through Agent-Based Modeling" (2026).

---

## Overview

Mitchell Berman's principled positivism proposes that legal principles aggregate through Simple Additive Weighting (SAW), with each principle contributing force proportional to its weight and activation. This project subjects that theory to the first systematic computational analysis, proceeding in three stages:

1. **Static Properties** (Notebook 0): Tests the SAW formula with stipulated parameters across thousands of synthetic cases. The model tolerates ±0.165 weight uncertainty, produces 76.8% determinacy in balanced-conflict cases, and delivers 19–87 percentage points more determinacy than Hart's consensus-based alternative.

2. **Emergent Parameters** (Notebook 1): An agent-based model simulates 100 judges with four interpretive methodologies deciding cases over time. Bimodal weight differentiation emerges under Berman fitness (dip test rejects unimodality in 20/20 seeds, *p* < 0.05), and emergent behavior conforms to SAW almost perfectly (*r* = 0.9994).

3. **Mechanism Dependence** (Notebook 2): Different practice mechanisms produce incompatible weight structures. Evolutionary selection achieves differentiation but not gradation; strategic optimization (MARL) achieves gradation but not differentiation. No single mechanism satisfies all five theoretical criteria simultaneously.

## Repository Structure

```
BermanModeling/
├── BermanNotebook0_Static_Properties.ipynb    # Stage 1: Formal properties
├── BermanNotebook1_ABM_Emergence.ipynb        # Stage 2: ABM emergence
├── BermanNotebook2_Mechanism_Dependence.ipynb  # Stage 3: Mechanism comparison
├── BermanPublicationFigures.ipynb             # Generates all publication figures
├── data/
│   ├── notebook0/          # 20 CSVs + 1 JSON from Notebook 0
│   ├── notebook1/          # 7 CSVs + 1 JSON from Notebook 1
│   └── notebook2/          # 9 CSVs + 1 JSON from Notebook 2
├── figures/                # 8 main + 3 supplementary figures (PDF + PNG)
├── README.md
├── LICENSE
└── requirements.txt
```

## Notebooks

| Notebook | Experiments | Runtime | Hardware |
|----------|-------------|---------|----------|
| **Notebook 0**: Static Properties | 9 experiments + 4 supplements | ~20 min | CPU |
| **Notebook 1**: ABM Emergence | 5 experiments | ~45 min | CPU (High-RAM) |
| **Notebook 2**: Mechanism Dependence | 4 experiments | ~90 min | T4 GPU |
| **Publication Figures** | Reads all exported CSVs | ~5 min | CPU |

All notebooks are designed to run in [Google Colab](https://colab.research.google.com/). Upload to Colab and run all cells sequentially.

### Notebook 0: Static Properties

Tests Berman's three claims with stipulated parameters:
- **Exp 1**: Weight uncertainty tolerance across 45 configurations (3 distributions × 3 principle counts × 5 thresholds)
- **Exp 2a–c**: Activation assessment (pluralist disagreement, objectivist error, interval ranges)
- **Exp 2b Supplement**: Methodology-specific systematic error
- **Exp 3**: Conflict case types (aligned, opposed, balanced, random)
- **Exp 4**: Berman vs. Hart (6 Hart variants across 30 seeds)
- **Exp 5**: Uncertainty decomposition (epistemic vs. aleatory)
- **Exp 6**: Threshold map (determinacy–stability trade-off)
- **Exp 7**: Principle count and three-way consistency
- **Exp 8**: Correlated activations (independent, category-correlated, fully correlated)
- **Exp 9**: Cardinal vs. ordinal aggregation

### Notebook 1: ABM Emergence

Tests whether parameters can emerge from simulated practice:
- **Exp 1**: Three-condition emergence (consensus vs. Berman fitness, ± median threshold). 20 seeds, 5000 cases each.
- **Exp 2**: Sensitivity analyses (agent count, invocation spread, selection cutoff)
- **Exp 3**: Emergent activation correlation structure
- **Exp 4**: Cross-validation against five theoretical criteria (10 seeds)
- **Exp 5**: SAW fidelity (correlation between ABM force and SAW-predicted force)

### Notebook 2: Mechanism Dependence

Compares three practice mechanisms:
- **Exp 1**: ABM vs. MARL vs. Hybrid side by side (10 seeds each)
- **Exp 2**: MARL diagnostics (parsimony sweep, architecture sensitivity)
- **Exp 3**: Cross-validation of all mechanisms against five criteria
- **Exp 3b**: Full Berman-vs-Hart curve with emergent weights

## Data

Each notebook exports its results as CSV files to the corresponding `data/notebookN/` directory. These CSVs are read by the Publication Figures notebook to generate all figures in the paper.

Summary JSON files (`notebook0_summary.json`, etc.) contain top-level results for quick reference.

### Key Data Files

| File | Contents |
|------|----------|
| `notebook0/exp1_weight_uncertainty.csv` | Critical σ* across 45 configurations |
| `notebook0/exp4_berman_vs_hart.csv` | Determinacy rates for Berman and 6 Hart variants |
| `notebook0/exp8_correlated.csv` | Robustness under three correlation structures |
| `notebook1/exp1_emergence.csv` | Bimodality metrics across 3 conditions × 20 seeds |
| `notebook1/exp1_condition_c_weights.csv` | Emergent weights from Condition C (20 seeds × 9 principles) |
| `notebook2/exp1_summary.csv` | Weight CV and survivors for ABM, MARL, Hybrid |
| `notebook2/exp3_pass_rates.csv` | C1–C5 pass rates by mechanism |
| `notebook2/exp3b_hart_curve.csv` | Berman advantage across Hart variants with emergent weights |

## Figures

All figures are at 600 DPI in both PDF (vector) and PNG (raster) formats.

### Main Figures

| Figure | Description | Paper Section |
|--------|-------------|---------------|
| `fig1_weight_uncertainty` | Weight tolerance across parameter grid; threshold dominance | Part IV |
| `fig2_berman_vs_hart_stipulated` | Berman's advantage over Hart variants | Part IV |
| `fig3_abm_emergence` | Bimodal differentiation under three ABM conditions | Part V |
| `fig4_abm_crossvalidation` | Cross-validation: 3/5 criteria pass, 2/5 fail | Part V |
| `fig5_mechanism_heatmap` | Pass rates by mechanism (complementary failure pattern) | Part VI |
| `fig6_hart_emergent_weights` | Hart curve with emergent weights from three mechanisms | Part VI |
| `fig7_complementarity` | Venn diagram + pass/fail table | Part VI |
| `fig8_complete_argument` | Three-stage argument summary | Part VII |

### Supplementary Figures

| Figure | Description |
|--------|-------------|
| `figS1_correlated_activations` | Category correlation doubles robustness |
| `figS2_methodology_error` | Homogeneous vs. mixed panel accuracy |
| `figS3_sensitivity` | ABM sensitivity (agent count, invocation spread, selection cutoff) |

## Requirements

```
python>=3.10
numpy>=1.24
scipy>=1.10
pandas>=1.5
matplotlib>=3.7
torch>=2.0        # Notebook 2 only
```

Install with:
```bash
pip install -r requirements.txt
```

Or simply run in Google Colab, which has all dependencies pre-installed.

## Replication

To reproduce all results from scratch:

1. Open each notebook in Google Colab
2. Run all cells sequentially
3. Notebooks export CSVs to their respective `data/` directories
4. Open `BermanPublicationFigures.ipynb` and run all cells to regenerate figures

All random seeds are fixed and reported. Results are deterministic given the same seed.

To reproduce using pre-computed data (without re-running experiments):

1. Open `BermanPublicationFigures.ipynb`
2. Point the data paths to the `data/` directories in this repository
3. Run all cells to regenerate figures from the exported CSVs

## Key Results

| Finding | Value | Source |
|---------|-------|--------|
| Mean weight tolerance (σ*) | ±0.165 | Notebook 0, Exp 1 |
| Berman vs. hierarchical Hart | +19.1% | Notebook 0, Exp 4 |
| Berman vs. consensus Hart (50%) | +33.1% | Notebook 0, Exp 4 |
| Balanced-conflict determinacy | 76.8% | Notebook 0, Exp 3 |
| Cardinal vs. ordinal advantage | 3.3× | Notebook 0, Exp 9 |
| Correlated activation improvement | 2.0× | Notebook 0, Exp 8 |
| Dip test (Berman fitness) | 20/20 reject | Notebook 1, Exp 1 |
| SAW fidelity | *r* = 0.9994 | Notebook 1, Exp 5 |
| ABM weight CV | 0.895 | Notebook 2, Exp 1 |
| MARL weight CV | 0.344 | Notebook 2, Exp 1 |
| C2 pass rate (all mechanisms) | 0% | Notebook 2, Exp 3 |

## Citation

```bibtex
@article{kamper2026computational,
  author  = {Kamper, David G.},
  title   = {Computational Jurisprudence: Testing Berman's Principled
             Positivism Through Agent-Based Modeling},
  year    = {2026},
  note    = {Manuscript}
}
```

## Acknowledgments

I am grateful to Mitchell Berman for extensive discussions of his theory and direct feedback on the computational framework, and to Amin Ebrahimi Afrouzi for insights on constitutional precedent analysis.

## License

MIT License. See [LICENSE](LICENSE) for details.
