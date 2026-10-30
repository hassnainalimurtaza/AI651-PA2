# Question 2 outputs: what each file is for

Hardware for all timings: Tesla T4 (Kaggle). Learned models: mean (seed 0, seed 1).
Original Kaggle downloads (with checkpoints): `../kaggle_downloads/nbX_*_outputs.zip`.

## Notebook 1: `01_setup_table1_figure1.ipynb` → `nb1_setup/`
| File | Report item | Used for |
|---|---|---|
| `results/table1.md` | **Table 1** (all checks pass) | Setup |
| `figures/figure1.png` | **Figure 1** (caption: median speed 3.32 px/frame, 103,050 pixels kept) | Q2.1, Q2.2 |
| `results/figure1_stats.json` | direction fractions per bin (0.026–0.083, uniform = 0.056), speed percentiles (10%: 2.14, 90%: 4.12) | Q2.1, Q2.3, Q2.5, Q2.9 |

## Notebook 2: `02_train_model_A.ipynb` → `nb2_modelA/`
| File | Report item | Used for |
|---|---|---|
| `results/table2.md` | **Table 2** (model A vs Blank vs Copy-last) | Q2.4 |
| `figures/figure2.png` | **Figure 2** (test videos 9007, 9012 clean; 9000 overlapping) | Q2.4 |
| `figures/modelA_curves.png` | training curves (optional appendix) | — |
| `results/eval_A_baselines.json` | exact per-seed numbers | backing data |

## Notebook 3: `03_test_baseline.ipynb` → `nb3_test_baseline/`
| File | Report item | Used for |
|---|---|---|
| `results/table3.md` | **Table 3** (C_M, E_M, C_R, E_R) | Q2.6, Q2.8 |
| `figures/figure3_M1_video9347.png`, `figure3_M2_video9928.png`, `figure3_R1_video9279.png`, `figure3_R2_video9049.png` | **Figure 3** (top-2 C_M and top-2 C_R videos) | Q2.7, Q2.8 |
| `gifs/figure3_*.gif` | GIFs for Figure 3: **push to GitHub, give paths in the caption** | Figure 3 caption |
| `figures/figure4.png` | **Figure 4** (per-video C_M, C_R histograms) | Q2.6–Q2.8 |
| `results/table4.md` | **Table 4** (speeds 1×, 2×, 3×) | Q2.9, Q2.10 |
| `figures/figure5.png` | **Figure 5** (test video 9017 at 1×, 2×, 3×) | Q2.9 |
| `results/selected_videos.json` | which videos Figures 3/5 use (reused by Figure 6) | backing data |

## Notebook 4: `04_train_models_BCD.ipynb` (latest rerun) → `nb4_models_BCD/`
| File | Report item | Used for |
|---|---|---|
| `results/table5.md` | **Table 5** (models A–D) | Q2.12, Q2.13, Q2.14 |
| `results/d_choice.json` | model D = `D_p8_ln` (8×8 patches + per-pixel gate normalisation, 595,520 params, reach 16 px) | Q2.12 |
| `results/eval_ABCD.json` | exact per-seed numbers | backing data |

## Notebook 5: `05_final_model.ipynb` → `nb5_final_model/`
| File | Report item | Used for |
|---|---|---|
| `results/table6.md` | **Table 6** (symmetrised B, symmetrised D = final model) | Q2.15 |
| `figures/figure6a_*.png` (same 4 videos as Figure 3) | **Figure 6(a)** | Q2.16 |
| `gifs/figure6a_*.gif` | GIFs for Figure 6(a): **push to GitHub, give paths in the caption** | Figure 6 caption |
| `figures/figure6b.png` (same video 9017 as Figure 5) | **Figure 6(b)** | Q2.16, Q2.17 |
| `results/eval_symmetrised.json` | exact per-seed numbers | backing data |

## Key numbers for the answers
- Model A: E(5) 0.0223; MSE_h 0.0172 → 0.0262; Blank 0.0426; Copy-last 0.0641. Reach 8 px.
- Model A: C_M 0.00187 / E_M 0.0229; C_R 0.00317 / E_R 0.0239.
- Model A, E(2) at 1×/2×/3×: 0.0189 / 0.045 / 0.0527 (Copy-last 0.0548 / 0.0661 / 0.0703).
- B: C_R 0.00168, E_R 0.0222 (rotation gap roughly halved).
- C: E(2) 2×/3× 0.0288 / 0.0333, but 1× E(5) worse (0.0267).
- D: E(5) 1× 0.0212; E(2) 2×/3× 0.0224 / 0.026; but C_M 0.00267, C_R 0.00423 (less consistent than A).
- Symmetrised D (final): E(5) 0.0209, C_M ≈ 8e-17, C_R ≈ 7e-17, E(2) 3× 0.0255; inference 2.06 s vs 0.252 s unwrapped.
- Symmetrised B: E(5) 0.0216 vs B 0.0223; inference 3.3 s vs 0.414 s.
