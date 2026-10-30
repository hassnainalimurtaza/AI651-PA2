# AI651 PA2: Deep Learning for Space, Time and Graphs

Roll number: 25280069

- **Question 1** (river flood forecasting design) is a written answer: see `report/main.tex`.
- **Question 2** (Motion or Memory? ConvLSTM on Moving MNIST) is in `Question2/`.

## Question 2: notebooks (run on Kaggle, Tesla T4 GPU)

These are the executed notebooks, with their outputs saved. Run them in order:

| # | Notebook | Produces |
|---|----------|----------|
| 1 | `01_setup_table1_figure1.ipynb` | Table 1, Figure 1 |
| 2 | `02_train_model_A.ipynb` | model A (seeds 0 and 1), Table 2, Figure 2 |
| 3 | `03_test_baseline.ipynb` | Tables 3 and 4, Figures 3 to 5, GIFs |
| 4 | `04_train_models_BCD.ipynb` | models B, C and D (seeds 0 and 1), Table 5 |
| 5 | `05_final_model.ipynb` | Table 6, Figure 6, GIFs |

- Notebooks 3 and 4 take the output of notebook 2 as input. Notebook 5 takes the outputs of notebooks 3 and 4.
- `q2lib.py` is the shared code: models, transforms, metrics and plots. Each notebook embeds it in its first cell.
- `build_notebooks.py` regenerates the notebooks from `q2lib.py`.
- `outputs/` holds every table, figure and GIF used in the report. `outputs/OUTPUTS.md` maps each file to its report item.

## GIFs (referenced in the captions of Figures 3 and 6)

- `Question2/outputs/nb3_test_baseline/gifs/`: baseline model A
- `Question2/outputs/nb5_final_model/gifs/`: final model (symmetrised D)
