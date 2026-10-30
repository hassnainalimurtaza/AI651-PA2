"""Builds the five Kaggle notebooks for Question 2. Each notebook embeds q2lib.py in its
first code cell (via %%writefile) so it runs standalone on Kaggle.

Run:  python build_notebooks.py      (only writes .ipynb JSON files, runs no model code)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = open(os.path.join(HERE, "q2lib.py")).read()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n")}


def write(name, cells):
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}}}
    path = os.path.join(HERE, name)
    with open(path, "w") as fh:
        json.dump(nb, fh, indent=1)
    print("wrote", path)


def header(title, purpose, kaggle_inputs, gpu=True):
    acc = ("**GPU T4** (required: this notebook trains or evaluates models, and must not run on CPU)"
           if gpu else "**None needed** (optical flow and tensor checks only; GPU is fine too)")
    return [
        md(f"""
# {title}

{purpose}

**Kaggle settings (right-hand panel → Session options):**
- Accelerator: {acc}
- Internet: **On** (torchvision downloads Moving MNIST, about 800 MB, into `/tmp`)
- Inputs to attach: {kaggle_inputs}

Everything this notebook saves goes into `/kaggle/working/` (`ckpt/`, `results/`, `figures/`, `gifs/`).
After **Save Version → Save & Run All**, that output can be attached as an input to the next notebook.

Lecture abbreviations in the comments: **L4** = *4. Baseline DL Architectures*,
**L5** = *5. Transformers for Time Series*, **L6** = *6. Spatiotemporal modeling* (page numbers as printed in the notes).
"""),
        md("""
## Cell 1: Write the shared library `q2lib.py`

**What happens:** `%%writefile` saves the cell's text as a Python file next to the notebook, so the next cell can `import` it.
All notebooks embed the *same* library, which keeps the data handling, models and metrics identical across every experiment.
Read the docstrings: each function says which part of the handout it implements and which lecture it comes from.
"""),
        code("%%writefile q2lib.py\n" + LIB),
        md("""
## Cell 2: Imports

**What happens:** we (re)load the library and report which device PyTorch will use.
`importlib.reload` makes sure that, if you edit and re-run Cell 1, the new version is used.
"""),
        code("""
import importlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import q2lib
importlib.reload(q2lib)
from q2lib import *

print("PyTorch", torch.__version__, "| device:", DEVICE,
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
"""),
    ] + ([
        md("""
## Cell 3: GPU guard

**What happens:** stops the notebook immediately if no GPU is attached, so training never silently falls back to the CPU.
Report the hardware in your timing tables (the handout asks for it): it is printed here.
"""),
        code("""
assert DEVICE.type == "cuda", "No GPU: set Accelerator = GPU T4 in the Kaggle session options."
HARDWARE = torch.cuda.get_device_name(0)
print("Hardware used for all timings:", HARDWARE)
"""),
    ] if gpu else []) + [
        md("""
## Load Moving MNIST and split by video

**What happens:** all 10,000 videos (20 frames of 64×64 each) are loaded with `split=None`, then split **by video index**:
0–7,999 train, 8,000–8,999 validation, 9,000–9,999 test.
We do **not** use torchvision's `split="train"/"test"`, because that splits the *frames* of each video, not the videos (handout, Setup → Data).
Frames stay as `uint8` in memory and are divided by 255 to land in [0, 1] batch by batch.

*Lecture link:* careful train/test generation from sequences, L4 p.4. The test split is only for evaluation, never for choosing designs.
"""),
        code("""
data = load_moving_mnist()
for k, v in data.items():
    print(f"{k:5s}", tuple(v.shape), v.dtype)
"""),
    ]


# ---------------------------------------------------------------------------------------
# Notebook 1: setup checks (Table 1) and optical flow (Figure 1)
# ---------------------------------------------------------------------------------------
nb1 = header(
    "PA2 Q2 · Notebook 1: Setup checks (Table 1) and training-data motion (Figure 1)",
    "Verifies the transformation and window functions (Table 1), measures the motion in the training videos "
    "with optical flow (Figure 1, used for Q2.1 and Q2.2), and checks model A's parameter count.",
    "none", gpu=False)
nb1 += [
    md("""
## Table 1: Check the transformation and window functions

**What happens:** two synthetic videos of shape (20, 1, 64, 64):
- a **random video** `x` (`torch.manual_seed(0); torch.rand(...)`);
- a **marker video** `p`: all zeros except one bright pixel at frame 7, row 10, column 20.

Checks (a)–(c) must be **exactly 0**, because mirror and rotation only *move* pixels and never change values:
- (a) mirroring twice gives back the original;
- (b) four quarter-turns give back the original;
- (c) M R M = R⁻¹ (a group identity).

Checks (d)–(e) catch mistakes that (a)–(c) cannot detect, such as flipping the wrong axis or rotating clockwise: we read off where the bright pixel ends up.

Checks (f)–(h) run the window function with t = 1 on a video whose frame *f* is filled with the value *f*. The values we read back are the frame numbers used.

*Lecture link:* building a known structure into the pipeline instead of hoping the model learns it, L4 p.5 (translation invariance) and L4 p.8 (inductive biases).
"""),
    code("""
torch.manual_seed(0)
x = torch.rand(20, 1, 64, 64)                      # random video
p = torch.zeros(20, 1, 64, 64); p[7, 0, 10, 20] = 1  # marker video

def max_abs(a, b):
    return (a - b).abs().max().item()

def bright(v):
    f, _, r, c = (v == v.max()).nonzero()[0].tolist()
    return (f, r, c)

rows = []
a = mirror(mirror(x));                         rows.append(("(a) max|M(M(x)) - x|", "0", max_abs(a, x), torch.equal(a, x)))
b = rotate(rotate(rotate(rotate(x))));         rows.append(("(b) max|R^4(x) - x|", "0", max_abs(b, x), torch.equal(b, x)))
c = mirror(rotate(mirror(x)))
rinv = rotate(x, -1);                          rows.append(("(c) max|MRM(x) - R^-1(x)|", "0", max_abs(c, rinv), torch.equal(c, rinv)))
rows.append(("(d) bright pixel of M(p) (frame,row,col)", "(7, 10, 43)", bright(mirror(p)), None))
rows.append(("(e) bright pixel of R(p) (frame,row,col)", "(7, 43, 10)", bright(rotate(p)), None))

# Video whose frame f (1-based) is filled with the value f -> window values = frame numbers.
numbered = torch.arange(1, 21, dtype=torch.float32).view(1, 20, 1, 1, 1).expand(1, 20, 1, 64, 64)
expected = {1: "1-5 / 6-10", 2: "1, 3, 5, 7, 9 / 11, 13, 15, 17, 19", 3: "1, 4, 7, 10, 13 / 16, 19"}
for k, tag in zip((1, 2, 3), "fgh"):
    X, Y, mask = make_windows(numbered, t=1, k=k)
    ins = [int(v) for v in X[0, :, 0, 0, 0]]
    tgs = [int(v) for v in Y[0, mask, 0, 0, 0]]
    rows.append((f"({tag}) {k}x input / target frames", expected[k], f"{ins} / {tgs}", None))

table1 = pd.DataFrame(rows, columns=["Check", "Expected", "Yours", "torch.equal"]).set_index("Check")
display(table1)
save_table(table1, "table1")

# Hard stop if anything is wrong: every later result depends on these functions.
assert table1.iloc[0, 1] == 0 and table1.iloc[1, 1] == 0 and table1.iloc[2, 1] == 0
assert bright(mirror(p)) == (7, 10, 43) and bright(rotate(p)) == (7, 43, 10)
print("Table 1: all checks pass")
"""),
    md("""
**Also check the allowed start frames.** With speed augmentation the windows need all 5 inputs and at least 2 targets, which the handout says gives t ∈ {1..14} at 1×, {1..8} at 2× and {1, 2} at 3×.
"""),
    code("""
for k in (1, 2, 3):
    s = allowed_starts(k, speed_aug=True)
    print(f"{k}x speed-augmented starts: {s[0]}..{s[-1]} ({len(s)} options)")
print("normal-speed starts:", allowed_starts(1, speed_aug=False))
"""),
    md("""
## Figure 1: Is the training data already symmetric?

**What happens:** **optical flow** estimates, for every pixel, how far it moved between two consecutive frames, as a displacement (dx, dy).
We use OpenCV's Farneback method with exactly the handout's settings on the **first 50 training videos** × 19 frame pairs = 950 pairs.
We keep only pixels that belong to a digit in both frames (intensity > 0.2) and that actually move (more than 0.5 px).

- **Direction** θ = atan2(−dy, dx), with the minus sign because image rows grow *downward*. 0° means right and 90° means up.
- **Speed** = √(dx² + dy²) pixels per frame.

**How to read it for Q2.1:**
- If the direction histogram is roughly flat (close to the 1/18 line), every direction is already common in training. A mirrored or rotated *trajectory* is then nothing new; only the mirrored or rotated *digit shapes* are.
- The median speed tells you what 1× means in pixels; 2× and 3× are twice and three times that. Compare these with model A's *reach* (Q2.3).

*Lecture link:* L6 p.3–4 chooses a kernel size from the physics ("rain moving at a given speed covers about 4 cells per time lag"). Here we measure the "speed of the rain" for Moving MNIST.
"""),
    code("""
angles, speeds = flow_statistics(data["train"], n_videos=50)
med = float(np.median(speeds))
n_kept = int(len(speeds))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
w = np.full(n_kept, 1.0 / n_kept)                    # fractions, not raw counts
ax[0].hist(angles, bins=np.arange(0, 361, 20), weights=w, color="tab:blue", edgecolor="white")
ax[0].axhline(1 / 18, color="red", ls="--", label="uniform (1/18)")
ax[0].set(xlabel="flow direction θ (degrees; 0 = right, 90 = up)", ylabel="fraction of kept pixels",
          title="(a) direction", xticks=range(0, 361, 60)); ax[0].legend()
ax[1].hist(speeds, bins=60, weights=w, color="tab:green", edgecolor="white")
ax[1].axvline(med, color="red", ls="--", label=f"median = {med:.3g} px/frame")
ax[1].set(xlabel="flow speed (pixels per frame)", ylabel="fraction of kept pixels", title="(b) speed"); ax[1].legend()
fig.suptitle(f"Figure 1: optical flow on the first 50 training videos (950 frame pairs), {n_kept:,} pixels kept")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/figure1.png", dpi=150); plt.show()

dir_frac = np.histogram(angles, bins=np.arange(0, 361, 20))[0] / n_kept
print(f"Caption facts -> median speed {med:.3g} px/frame; pixels kept {n_kept:,}")
print(f"2x ~ {2 * med:.3g} px/frame, 3x ~ {3 * med:.3g} px/frame (median)")
print("direction fractions per 20-degree bin:", np.round(dir_frac, 4))
print(f"min/max bin fraction: {dir_frac.min():.4f} / {dir_frac.max():.4f} (uniform = {1/18:.4f})")
save_json({"median_speed": med, "n_pixels": n_kept, "dir_fractions": dir_frac.tolist(),
           "speed_percentiles": {q: float(np.percentile(speeds, q)) for q in (10, 25, 50, 75, 90)}},
          "figure1_stats.json")
"""),
    md("""
## Model A sanity check: parameter count and reach

**What happens:** we build model A *without training it* and count trainable parameters. The handout says it must be **exactly 480,784**. If it matches, the architecture is right.

By hand:
- ConvLSTM layer 1: (16 + 64) input channels × 256 gate channels × 3 × 3 + 256 biases = 184,576
- ConvLSTM layer 2: (64 + 64) × 256 × 9 + 256 = 295,168
- 1 × 1 head: 64 × 16 + 16 = 1,040
- Total: 480,784

`reach_pixels` traces the longest path information can take in one time step (see its docstring and Q2.3). Your report must explain the reasoning; the function only confirms the arithmetic.

*Lecture link:* ConvLSTM = LSTM gates (L4 pp.15–18) with convolutions (L6 p.11); receptive field growth with depth (L4 p.7).
"""),
    code("""
model_A = ConvLSTMForecaster(**BASELINE_CFG)
print("model A trainable parameters:", f"{count_params(model_A):,}")
assert count_params(model_A) == 480_784
print("model A reach (pixels per time step):", reach_pixels(BASELINE_CFG))
out = model_A(torch.rand(2, 5, 1, 64, 64))
print("forward-pass output shape:", tuple(out.shape), "(batch, 5 forecasts, 1, 64, 64)")
"""),
    md("""
## Next steps

1. Put **Table 1** and **Figure 1** (`/kaggle/working/figures/figure1.png`) into the report, with the median speed and pixel count in the caption.
2. Answer **Q2.1** and **Q2.2**. For Q2.1, mirroring maps direction θ to 180° − θ and a 90° rotation maps θ to θ + 90°. Would the histogram look any different after that?
3. Run **Notebook 2** (train model A).
"""),
]
write("01_setup_table1_figure1.ipynb", nb1)


# ---------------------------------------------------------------------------------------
# Notebook 2: train model A, Table 2, Figure 2
# ---------------------------------------------------------------------------------------
nb2 = header(
    "PA2 Q2 · Notebook 2: Baseline ConvLSTM, model A (Table 2, Figure 2)",
    "Trains model A with seeds 0 and 1 (30 epochs each, about 10–15 min per seed on a T4), compares it with the "
    "Blank and Copy-last reference predictors (Table 2), and shows example forecasts (Figure 2).",
    "none")
nb2 += [
    md("""
## Train model A (seeds 0 and 1)

**What happens, step by step, per batch (all inside `train_run`):**
1. Shuffle the 8,000 training videos and take 32.
2. Cut **one random window** per video: at normal speed, start frame t ∈ {1..11}, 5 input frames and 5 target frames.
3. **Roll out 9 steps:** steps 1–5 read the true frames, steps 6–9 read the model's *own previous prediction*. No teacher forcing, so gradients flow back through the fed-back predictions.
4. Loss = squared error averaged over pixels and available target frames per video, then over the batch.
5. Adam with learning rate 10⁻³, no schedule. After 30 epochs the final weights are saved to `ckpt/A_s{seed}.pt`.

The seed fixes initialisation, shuffling and window sampling, so runs are reproducible.
The per-epoch wall-clock time (validation excluded) is stored for the timing columns of Table 5.

*Lecture links:*
- LSTM gates and cell state: L4 pp.15–18.
- ConvLSTM ("Recurrent" space–time composition): L6 p.1 and p.11.
- Teacher forcing: L4 p.51.
- Autoregressive error accumulation: L5 pp.10–11.
- MSE loss: L5 p.12.

If a checkpoint already exists (for example, you re-run after a crash), it is loaded instead of retrained.
"""),
    code("""
models_A, hists_A = {}, {}
for seed in (0, 1):
    if find_file(f"A_s{seed}.pt"):
        models_A[seed], ck = load_checkpoint("A", seed); hists_A[seed] = ck["history"]
        print(f"loaded existing A_s{seed}.pt")
    else:
        models_A[seed], hists_A[seed] = train_run("A", seed, data)
"""),
    md("""
## Training curves (for your own monitoring)

**What happens:** training loss and validation E⁽⁵⁾ per epoch. Validation is only for *monitoring*; it never feeds back into training.
Both curves should fall and flatten. If validation rises while training keeps falling, the model is overfitting.
"""),
    code("""
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for s, h in hists_A.items():
    ax[0].plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], label=f"seed {s}")
    ax[1].plot(range(1, len(h["val_E5"]) + 1), h["val_E5"], label=f"seed {s}")
ax[0].set(title="model A training loss", xlabel="epoch", ylabel="masked MSE"); ax[0].legend()
ax[1].set(title="model A validation E(5) at 1x", xlabel="epoch"); ax[1].legend()
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/modelA_curves.png", dpi=130); plt.show()
for s, h in hists_A.items():
    print(f"seed {s}: mean epoch time {np.mean(h['epoch_time']):.1f}s")
"""),
    md("""
## Table 2: Model A vs Blank vs Copy-last (test split, 1×, t = 1)

**What happens:** for each test video and forecast horizon h = 1..5 we compute MSE_h (mean over the 64×64 pixels), then average over the 1,000 test videos.
E⁽⁵⁾ is the mean of the five horizons (handout eq. 5).
Learned models are reported as **mean (seed 0, seed 1)**. Blank and Copy-last are deterministic, so they get one value.

How to read it:
- **Blank** = the error of doing nothing.
- **Copy-last** = the error of assuming nothing moves.

A model that has learned motion should beat Copy-last, especially at later horizons. Model A's error should *grow* with h, because errors compound in the rollout.

*Lecture link:* always compare with a naive repeat-last-value baseline, L5 p.26.
"""),
    code("""
res_A = {s: evaluate(models_A[s], data["test"], return_preds=(s == 0)) for s in (0, 1)}
res_blank = evaluate(Blank(), data["test"])
res_copy = evaluate(CopyLast(), data["test"])

keys = [f"MSE_{h}" for h in range(1, 6)] + ["E5_1x"]
table2 = pd.DataFrame({
    "Model A": {**metrics_row([res_A[0], res_A[1]], keys), "params": f"{count_params(models_A[0]):,}"},
    "Blank": {**metrics_row([res_blank], keys), "params": "0"},
    "Copy-last": {**metrics_row([res_copy], keys), "params": "0"},
}).T.rename(columns={"E5_1x": "E(5)"})
display(table2)
save_table(table2, "table2")
save_json({"A_s0": scalars(res_A[0]), "A_s1": scalars(res_A[1]),
           "Blank": scalars(res_blank), "CopyLast": scalars(res_copy)}, "eval_A_baselines.json")
"""),
    md("""
## Figure 2: Input, prediction and ground truth for three test videos

**What happens:** we pick three test videos, at least one where the digits **overlap** in the target frames.
`overlapping()` uses a simple heuristic: two separate digits form at least 2 bright blobs, and overlapping digits merge into 1. **Look at the picture** to confirm, and change `chosen` if needed.

What to look for (Q2.4):
- Does the prediction keep the digit's shape and position over h = 1..5, or does it blur?
- When digits overlap, the model can't tell which digit went where. An MSE-trained model then predicts roughly the **average** of the possibilities, which shows up as a smeared or ghosted blob.

*Lecture link:* MSE is minimised by the mean (companion notebook, "Think about 1"); MSE vs MAE, L5 p.12.
"""),
    code("""
X, Y, P = res_A[0]["preds"]["1x"]
overlap = [i for i in range(len(Y)) if overlapping(Y[i, :, 0])]
clean = [i for i in range(len(Y)) if not overlapping(torch.cat([X[i, :, 0], Y[i, :, 0]]))]
print(f"{len(overlap)} test videos look overlapping in the targets; {len(clean)} look clean")

chosen = [clean[0], clean[1], overlap[0]]   # change these indices after looking, if needed
rows, labels = [], []
for i in chosen:
    tag = " (overlap)" if i in overlap else ""
    rows += [[X[i, j, 0] for j in range(5)], [P[i, j, 0] for j in range(5)], [Y[i, j, 0] for j in range(5)]]
    labels += [f"video {9000 + i}{tag}\\ninput 1-5", "prediction 6-10", "truth 6-10"]
plot_grid(rows, labels, [f"col {c}" for c in range(1, 6)],
          title="Figure 2: model A (seed 0), input / prediction / ground truth, 1x, t = 1",
          path=f"{FIG_DIR}/figure2.png")
save_json({"figure2_videos": chosen}, "figure2_videos.json")
"""),
    md("""
## Before you open Notebook 3: write Q2.5 and commit it

The handout requires your **Q2.5 prediction** (will model A pass the mirror test? the rotation test? at what speed does it break?) to be written **before** running any test, and **committed to your GitHub repository** (commit times may be checked).

You now have what you need for it:
- the median speed from Figure 1 (`results/figure1_stats.json`);
- model A's reach (Q2.3);
- Table 2 and Figure 2.

Next: answer Q2.3 and Q2.4, write and commit Q2.5, then run Notebook 3 with **this notebook's output attached as input**, since it needs `ckpt/A_s0.pt` and `ckpt/A_s1.pt`.
"""),
]
write("02_train_model_A.ipynb", nb2)


# ---------------------------------------------------------------------------------------
# Notebook 3: testing the baseline (Tables 3-4, Figures 3-5)
# ---------------------------------------------------------------------------------------
nb3 = header(
    "PA2 Q2 · Notebook 3: Testing the baseline (Tables 3–4, Figures 3–5)",
    "Stress-tests model A under mirror and rotation (consistency gap C_T and transformed error E_T) and at "
    "2× and 3× speed. **Run this only after your Q2.5 answer is committed.**",
    "output of Notebook 2 (for `ckpt/A_s0.pt`, `ckpt/A_s1.pt`)")
nb3 += [
    md("""
## Load model A (both seeds) and evaluate everything once

**What happens:** `evaluate` computes, on the 1,000 test videos at t = 1:
- E⁽⁵⁾ at 1×;
- the mirror and rotation metrics;
- E⁽²⁾ at 1×, 2× and 3×, and E⁽⁵⁾ at 2×.

**The commuting-diagram test (handout eq. 6).** For a transformation T (mirror M or rotation R):
- **Consistency gap** C_T = MSE( f(T X), T f(X) ). It asks: does predicting on the transformed video equal transforming the normal prediction?
- **Transformed error** E_T = MSE( f(T X), T Y ). It asks: is the prediction on the transformed video actually correct?

Always report both. Blank and Copy-last have C_T = 0 (mirroring a black frame changes nothing) but are useless forecasters (Q2.6).
"""),
    code("""
models_A = {s: load_checkpoint("A", s)[0] for s in (0, 1)}
res_A = {s: evaluate(models_A[s], data["test"], return_preds=(s == 0)) for s in (0, 1)}
res_blank = evaluate(Blank(), data["test"])
res_copy = evaluate(CopyLast(), data["test"])
print("evaluation done")
"""),
    md("""
## Table 3: Mirror and rotation

*Lecture link:* a convolution is shift-consistent by construction (weight sharing, L4 p.8 and L6 p.4). Nothing in model A forces the same for mirror or rotation, so any consistency it has must be *learned* from the data.
"""),
    code("""
keys = ["E5_1x", "C_M", "E_M", "C_R", "E_R"]
table3 = pd.DataFrame({
    "Model A": metrics_row([res_A[0], res_A[1]], keys),
    "Blank": metrics_row([res_blank], keys),
    "Copy-last": metrics_row([res_copy], keys),
}).T.rename(columns={"E5_1x": "E(5)"})
display(table3)
save_table(table3, "table3")
"""),
    md("""
## Figure 3: The worst videos for consistency (seed-0 model A)

**What happens:** we rank test videos by their *per-video* C_M and C_R and take the top two for each.
For each, we draw four rows × ten frames (inputs in columns 1–5, future in 6–10):

| Row | Columns 1–5 | Columns 6–10 |
|---|---|---|
| (1) | X | f(X) |
| (2) | T(X) | T(Y) (the truth for the transformed video) |
| (3) | T(X) | f(T(X)) (what the model predicts when shown the transformed video) |
| (4) | T(X) | T(f(X)) (the transformed normal prediction) |

**Rows 3 and 4 disagreeing = inconsistency.** Compare both with row 2 to see which one is right.
For Q2.7, describe *how* the motion changes: wrong direction, lost digit, early or missing bounce, or blur.
Each panel is also saved as an animated GIF stepping through h = 1..5. Put the GIF paths in the caption and push the GIFs to your repository.
"""),
    code("""
pv = res_A[0]["per_video"]
top_M = [int(i) for i in np.argsort(-pv["C_M"])[:2]]
top_R = [int(i) for i in np.argsort(-pv["C_R"])[:2]]
print("top-2 C_M videos (test index):", top_M, [f"{pv['C_M'][i]:.3g}" for i in top_M])
print("top-2 C_R videos (test index):", top_R, [f"{pv['C_R'][i]:.3g}" for i in top_R])

for tname, T, vids in (("M", mirror, top_M), ("R", rotate, top_R)):
    for rank, vid in enumerate(vids, 1):
        figure3_panel(models_A[0], data["test"], vid, T, tname, "Figure 3, model A seed 0",
                      fig_path=f"{FIG_DIR}/figure3_{tname}{rank}_video{9000 + vid}.png",
                      gif_path=f"{GIF_DIR}/figure3_{tname}{rank}_video{9000 + vid}.gif")
"""),
    md("""
## Figure 4: How is inconsistency spread over the test set?

**What happens:** histograms of per-video C_M and C_R for seed-0 model A, with the Figure 3 videos marked.
A log x-axis is used because the values can span orders of magnitude.

Questions to ask:
- Is inconsistency spread evenly, or concentrated in a few videos? (Overlap cases are a good guess.)
- Is C_R generally larger than C_M (Q2.8)?
"""),
    code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for a, tname, vids in ((ax[0], "C_M", top_M), (ax[1], "C_R", top_R)):
    vals = pv[tname]
    bins = np.logspace(np.log10(max(vals.min(), 1e-8)), np.log10(vals.max()), 50)
    a.hist(vals, bins=bins, color="tab:blue", edgecolor="white")
    a.set_xscale("log")
    for v in vids:
        a.axvline(vals[v], color="red", ls="--")
        a.text(vals[v], a.get_ylim()[1] * 0.9, f" {9000 + v}", color="red", fontsize=7, rotation=90)
    a.set(title=f"per-video {tname} (model A, seed 0)", xlabel=tname, ylabel="number of test videos")
fig.suptitle("Figure 4: per-video consistency gaps; red = videos shown in Figure 3")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/figure4.png", dpi=150); plt.show()
"""),
    md("""
## Table 4: Faster motion

**What happens:** the same test videos played at 2× and 3× (keep every k-th frame, t = 1).
At 3× only two target frames exist (16 and 19), so **comparisons across speeds use E⁽²⁾** (first two horizons only). E⁽⁵⁾ is also reported at 1× and 2×.

What to look for (Q2.9): at which speed does model A's error jump, and does it get worse than Copy-last?
Compare the per-frame displacement (median speed × k, from Figure 1) with model A's reach.

*Lecture link:* L6 p.3–4, where the kernel size must cover how far the pattern moves per time step.
"""),
    code("""
keys = ["E2_1x", "E2_2x", "E2_3x", "E5_1x", "E5_2x"]
table4 = pd.DataFrame({
    "Model A": metrics_row([res_A[0], res_A[1]], keys),
    "Copy-last": metrics_row([res_copy], keys),
}).T.rename(columns={"E2_1x": "E(2) 1x", "E2_2x": "E(2) 2x", "E2_3x": "E(2) 3x",
                     "E5_1x": "E(5) 1x", "E5_2x": "E(5) 2x"})
display(table4)
save_table(table4, "table4")
print("model A reach (px/step):", reach_pixels(BASELINE_CFG))
try:
    med = load_json("figure1_stats.json")["median_speed"]
    print(f"median digit speed: 1x {med:.3g}, 2x {2*med:.3g}, 3x {3*med:.3g} px per model step")
except FileNotFoundError:
    print("(attach Notebook 1 output to print the Figure 1 speeds here)")
"""),
    md("""
## Figure 5: One video at 1×, 2×, 3×

**What happens:** we need a test video whose digits **do not overlap** in any displayed true frame at any speed, so that failures come from speed and not from overlap.
We search automatically with the blob heuristic and show the first match. Check it visually, and change `fig5_vid` if needed.

Failure vocabulary for Q2.9:
- **Blur:** the model is uncertain and averages possibilities.
- **Stalling:** the model can't link the new position to the old one, so it keeps the digit near where it was.
- **Duplicate:** the old and new positions are both kept.
- **Vanishing:** the digit is lost entirely.
"""),
    code("""
def clean_at_all_speeds(i):
    for k in (1, 2, 3):
        X, Y, m = make_windows(data["test"][i:i + 1], 1, k)
        if overlapping(to_unit(Y[0, :2, 0])):
            return False
    return True

fig5_vid = next(i for i in range(len(data["test"])) if clean_at_all_speeds(i))
print("Figure 5 video:", 9000 + fig5_vid, "(test index", fig5_vid, ")")
figure5_panel(models_A[0], data["test"], fig5_vid, "Figure 5, model A seed 0",
              fig_path=f"{FIG_DIR}/figure5.png")
save_json({"top_M": top_M, "top_R": top_R, "figure5": fig5_vid}, "selected_videos.json")
"""),
    md("""
## Before you open Notebook 4: write Q2.11 and commit it

Answer Q2.6–Q2.10 from these results. Then write your **Q2.11 prediction** (at most three sentences: which of B, C, D fixes which failure) and **commit it before training B–D**.

Next: Notebook 4 with **Notebook 2's output attached** (for model A's checkpoints). Notebook 5 will also need **this notebook's output** (`results/selected_videos.json`).
"""),
]
write("03_test_baseline.ipynb", nb3)


# ---------------------------------------------------------------------------------------
# Notebook 4: models B, C, D (Table 5)
# ---------------------------------------------------------------------------------------
nb4 = header(
    "PA2 Q2 · Notebook 4: What can data fix? Models B, C, D (Table 5)",
    "Trains B (symmetry augmentation), C (speed augmentation) and D (larger reach + speed augmentation), "
    "two seeds each (about 1–1.5 h on a T4), then builds Table 5. **Run only after Q2.11 is committed.**",
    "output of Notebook 2 (model A checkpoints)")
nb4 += [
    md("""
## Why these three models? (a controlled experiment)

Each model changes **one thing**, so a fix can be blamed on that one thing:

| Model | Change | If it fixes a failure, the failure was… |
|---|---|---|
| B | A + random mirror/rotation of every training window | about **appearance** (unfamiliar digit shapes) |
| C | A + random speed k ∈ {1, 2, 3} | about **unfamiliar motion** |
| D | C + larger-reach architecture | about **architectural reach**: data can't create a path that doesn't exist |

*Lecture link:* inductive bias vs learning from data, L6 p.2 ("rely on domain knowledge…") and L6 pp.3–4.
"""),
    md("""
## Choosing model D

Model D must:
- have **at most 721,176** parameters (1.5 × model A);
- have **strictly larger one-step reach** than A;
- use **no attention**;
- keep everything else as model A: input/output shapes, 9-step rollout without teacher forcing, training protocol.

Three candidates are prepared. Each roughly doubles the reach of A in a different way:

| Candidate | Idea | Lecture |
|---|---|---|
| `D_k5` | 5×5 kernels (each conv moves 2 cells), hidden width cut to 46 to fit the budget | bigger receptive field per layer, L4 p.7, L6 p.3 |
| `D_dil2` | 3×3 kernels with dilation 2 (taps 2 cells apart), same parameter count as A | dilation, L4 p.7 |
| `D_p8` | fold 8×8 patches (one cell = 8 px) | patching, L6 pp.6–7, L5 pp.4–5 |

The cell below prints parameters and reach for each. Q2.12 asks you to describe D and derive its reach by the same reasoning as Q2.3.
"""),
    code("""
rows = {}
for name, cfg in {"A (baseline)": BASELINE_CFG, **D_CANDIDATES}.items():
    n = count_params(ConvLSTMForecaster(**cfg))
    rows[name] = {**cfg, "params": f"{n:,}", "<= 721,176": n <= MAX_D_PARAMS, "reach (px/step)": reach_pixels(cfg)}
display(pd.DataFrame(rows).T)
"""),
    md("""
### How model D was chosen (report this in Q2.12)

1. **`D_k5`** (5×5 kernels, 46 hidden channels) **collapsed on seed 0**. From epoch 2 its training loss stayed at **0.0402**, which is the loss of predicting the same average grey frame for every video. Seed 1 stalled at that level for 3 epochs before slowly recovering.
2. We then compared `D_dil2` and `D_p8` on the **validation** split (seed 0, 8 epochs) and picked `D_p8` (8×8 patches). Its full 30-epoch seed-0 run **collapsed the same way** (final loss 0.0402), while seed 1 trained normally.
3. Model C (same training data and seed, baseline architecture) never collapsed. So the wider-reach models get stuck in the "predict the average frame" solution under the fixed protocol (Adam 10⁻³, no schedule), most likely because their gates saturate and gradients stop flowing.
4. **Fix: `D_p8_ln`** = `D_p8` + a per-pixel normalisation of each gate's pre-activations (a layer norm over channels at every position, adding 1,024 parameters).
   - The handout allows normalisation in D; it only forbids it in A–C.
   - The statistics are computed per pixel and never pooled over space, so **the reach is unchanged (16 px)**.
   - This is the standard remedy for saturated recurrent gates.

The cell below records the choice. `version` is a timestamp, so Notebook 5 always picks the newest choice even when several Notebook 4 outputs are attached.
"""),
    code("""
import time
D_CHOICE = "D_p8_ln"
register_D(D_CANDIDATES[D_CHOICE])
save_json({"D_choice": D_CHOICE, "cfg": D_CANDIDATES[D_CHOICE], "version": time.time()}, "d_choice.json")
print("model D =", D_CHOICE, D_CANDIDATES[D_CHOICE],
      "| params", f"{count_params(ConvLSTMForecaster(**D_CANDIDATES[D_CHOICE])):,}",
      "| reach", reach_pixels(D_CANDIDATES[D_CHOICE]), "px")
"""),
    md("""
## Train B, C, D (two seeds each)

**What happens:** the same `train_run` as model A, with the augmentation flags from the model table:
- **B:** for each window, draw g uniformly from the 8 symmetries (including the identity) and apply it to input **and** target.
- **C, D:** for each window, draw k uniformly from {1, 2, 3}, then t uniformly from the allowed starts for that k. Windows with only 2 or 3 real targets are masked in the loss.

Existing checkpoints are reused **only if they were trained with the current design**. An old `D_s0.pt` from a collapsed design (`D_k5`, `D_p8`) is detected by its stored configuration and retrained.
If the Kaggle session stops, save the version and re-run with its output attached, and finished runs are skipped.
"""),
    code("""
models, hists = {}, {}
for name in ("A", "B", "C", "D"):
    for seed in (0, 1):
        try:   # reuse a checkpoint only if it has the current design (collapsed D_k5 is skipped)
            models[name, seed], ck = load_checkpoint(name, seed, cfg=RUNS[name][0])
            hists[name, seed] = ck["history"]
        except FileNotFoundError:
            models[name, seed], hists[name, seed] = train_run(name, seed, data)
"""),
    md("""
## Table 5: Models A–D

**What happens:** every model and seed is evaluated on the test split. Columns:
- E⁽⁵⁾ at 1×; C_M, E_M, C_R, E_R; E⁽²⁾ at 1×, 2×, 3×;
- parameter count;
- training time per epoch (mean over 30 epochs);
- inference time for all 1,000 test videos at batch 32.

For Q2.13, label each failure from Notebook 3 using the row that fixed it:
- If **C still fails at 3× but D does well**, more data could not help, because the information path did not exist (reach).
- If **B** fixes the mirror/rotation errors, the problem was appearance.

For Q2.14, compare the E⁽⁵⁾ 1× column: did anything get worse on the original clips?
"""),
    code("""
results = {}
for (name, seed), m in models.items():
    r = evaluate(m, data["test"])
    r["params"] = count_params(m)
    r["train_time"] = float(np.mean(hists[name, seed]["epoch_time"]))
    r["infer_time"] = inference_time(m, data["test"])
    results[name, seed] = r

keys = ["E5_1x", "C_M", "E_M", "C_R", "E_R", "E2_1x", "E2_2x", "E2_3x", "train_time", "infer_time"]
table5 = pd.DataFrame({
    name: {**metrics_row([results[name, 0], results[name, 1]], keys), "params": f"{results[name, 0]['params']:,}"}
    for name in ("A", "B", "C", "D")
}).T.rename(columns={"E5_1x": "E(5) 1x", "E2_1x": "E(2) 1x", "E2_2x": "E(2) 2x", "E2_3x": "E(2) 3x",
                     "train_time": "train s/epoch", "infer_time": "inference s"})
display(table5)
print("Hardware:", HARDWARE)
save_table(table5, "table5")
save_json({f"{n}_s{s}": scalars(r) for (n, s), r in results.items()}, "eval_ABCD.json")
"""),
    md("""
## Next

Answer Q2.12–Q2.14. Then run Notebook 5 with the outputs of **Notebook 3** (selected videos) and **this notebook** (checkpoints for B and D, plus `d_choice.json`) attached.
"""),
]
write("04_train_models_BCD.ipynb", nb4)


# ---------------------------------------------------------------------------------------
# Notebook 5: the final (symmetrised) model, Table 6, Figure 6
# ---------------------------------------------------------------------------------------
nb5 = header(
    "PA2 Q2 · Notebook 5: The final model, built-in symmetry (Table 6, Figure 6)",
    "Wraps trained models in the symmetrised forecaster (no retraining), measures it, optionally trains D+B, "
    "and repeats Figures 3 and 5 with the final model.",
    "outputs of Notebook 3 (`selected_videos.json`) and Notebook 4 (checkpoints B, D and `d_choice.json`)")
nb5 += [
    md("""
## The symmetrised forecaster (handout eq. 7)

f̄(X) = (1/8) Σ_{g ∈ G} g⁻¹( f( g(X) ) )

In words: transform the input by each of the 8 symmetries of the square, predict, undo the transformation, and average.

**Why C_M = C_R = 0 exactly (the short proof for your report).** Take any h in G. Then
f̄(hX) = (1/8) Σ_g g⁻¹ f(g h X).
Substitute g′ = g h. As g runs over all of G, so does g′, because multiplying by h only re-orders the group. Since g⁻¹ = h g′⁻¹:
f̄(hX) = h · (1/8) Σ_{g′} g′⁻¹ f(g′ X) = h f̄(X).
Mirror and rotation are both in G, so the consistency gap is 0. Only floating-point rounding remains, which is why the values are ~1e-15 rather than exactly 0.

**Learned vs built-in (Q2.15):**
- Augmentation (model B) *encourages* consistency, but nothing guarantees it.
- Symmetrisation *guarantees* it by construction, at 8× inference cost.
- Averaging 8 predictions that disagree can **blur** the forecast.

*Lecture link:* building a symmetry into the architecture instead of learning it, as weight sharing does for shifts (L4 p.8, L6 p.4; handout "Looking ahead").
"""),
    md("""
**What happens:** we find model D's design. `d_choice.json` (written by Notebook 4) is used if it is attached.
Otherwise the design is read from the D checkpoint itself, since every checkpoint stores the configuration it was trained with.
The listing shows which checkpoints and result files this notebook can see. If B or D are missing, attach **Notebook 4's saved output** (Add Input → Notebook Output).
"""),
    code("""
import os, glob, json
print("checkpoints visible:", sorted({os.path.basename(p) for p in glob.glob("/kaggle/input/**/*.pt", recursive=True)}
                                     | {os.path.basename(p) for p in glob.glob(f"{CKPT_DIR}/*.pt")}))
print("json visible:", sorted({os.path.basename(p) for p in glob.glob("/kaggle/input/**/*.json", recursive=True)}))

# Several Notebook 4 outputs may be attached (older ones hold B, C and collapsed D designs).
# Use the NEWEST design choice: only the latest Notebook 4 writes a "version" timestamp.
choices = [json.load(open(f)) for f in find_files("d_choice.json")]
if not choices:
    raise FileNotFoundError("No d_choice.json: attach the latest Notebook 4's saved output.")
latest = max(choices, key=lambda c: c.get("version", 0))
d_name, d_cfg = latest["D_choice"], latest["cfg"]
register_D(d_cfg)
print("model D design:", d_name, d_cfg, "| reach", reach_pixels(d_cfg), "px")

models, hists = {}, {}
for name in ("B", "D"):
    for seed in (0, 1):
        models[name, seed], ck = load_checkpoint(name, seed, cfg=RUNS[name][0])   # design must match
        hists[name, seed] = ck["history"]
        print(f"{name}_s{seed}: {ck['cfg']}, final train loss {ck['history']['train_loss'][-1]:.5f}")
"""),
    md("""
### Optional: train D+B (D architecture + speed and symmetry augmentation)

Set `TRAIN_DB = True` only if Table 5 suggests that symmetry augmentation helps on top of D. Say in the report which final model you chose and why.
Takes about the same time as training D (two seeds).
"""),
    code("""
TRAIN_DB = False
if TRAIN_DB:
    for seed in (0, 1):
        if find_file(f"DB_s{seed}.pt"):
            models["DB", seed], ck = load_checkpoint("DB", seed); hists["DB", seed] = ck["history"]
        else:
            models["DB", seed], hists["DB", seed] = train_run("DB", seed, data)
FINAL = "DB" if TRAIN_DB else "D"
print("final model = symmetrised", FINAL)
"""),
    md("""
## Table 6: Symmetrised B, symmetrised D (and symmetrised D+B)

**What happens:** each trained model is wrapped in `Symmetrised(...)` and evaluated exactly like Table 5.
C_M and C_R should be around 1e-15 or 1e-16 (floating-point noise) and are shown in scientific notation.
- **Training time** = the underlying learned model's (symmetrisation needs no training).
- **Inference time** includes all 8 transformations, forward passes, inverse transformations and the averaging.
"""),
    code("""
results = {}
for (name, seed), m in models.items():
    sym = Symmetrised(m)
    r = evaluate(sym, data["test"])
    r["params"] = count_params(m)
    r["train_time"] = float(np.mean(hists[name, seed]["epoch_time"]))
    r["infer_time"] = inference_time(sym, data["test"])
    results[name, seed] = r

keys = ["E5_1x", "C_M", "E_M", "C_R", "E_R", "E2_1x", "E2_2x", "E2_3x", "train_time", "infer_time"]
names = ["B", "D"] + (["DB"] if TRAIN_DB else [])
table6 = pd.DataFrame({
    f"sym. {n}": {**metrics_row([results[n, 0], results[n, 1]], keys), "params": f"{results[n, 0]['params']:,}"}
    for n in names
}).T.rename(columns={"E5_1x": "E(5) 1x", "E2_1x": "E(2) 1x", "E2_2x": "E(2) 2x", "E2_3x": "E(2) 3x",
                     "train_time": "train s/epoch", "infer_time": "inference s"})
display(table6)
print("Hardware:", HARDWARE)
for n in names:
    print(f"sym. {n}: max C_M {max(results[n, s]['C_M'] for s in (0, 1)):.2e}, "
          f"max C_R {max(results[n, s]['C_R'] for s in (0, 1)):.2e}  (should be ~0)")
save_table(table6, "table6")
save_json({f"sym_{n}_s{s}": scalars(r) for (n, s), r in results.items()}, "eval_symmetrised.json")
"""),
    md("""
## Figure 6: Figures 3 and 5 repeated with the final model

**What happens:**
- (a) the same four Figure 3 panels (same test videos, same transformations) with the final symmetrised model (seed 0), plus GIFs;
- (b) the same Figure 5 video at 1×, 2×, 3×.

For Q2.16, go back to the video you described in Q2.7. Do rows 3 and 4 now agree (they must, by construction)? Do they now match row 2 (the truth)? Which change fixed which part:
- larger reach for the speed and tracking failure;
- symmetrisation for the mirror and rotation inconsistency.
"""),
    code("""
sel = load_json("selected_videos.json")
final = Symmetrised(models[FINAL, 0])
label = f"Figure 6(a), symmetrised {FINAL} seed 0"
for tname, T, vids in (("M", mirror, sel["top_M"]), ("R", rotate, sel["top_R"])):
    for rank, vid in enumerate(vids, 1):
        figure3_panel(final, data["test"], vid, T, tname, label,
                      fig_path=f"{FIG_DIR}/figure6a_{tname}{rank}_video{9000 + vid}.png",
                      gif_path=f"{GIF_DIR}/figure6a_{tname}{rank}_video{9000 + vid}.gif")
figure5_panel(final, data["test"], sel["figure5"], f"Figure 6(b), symmetrised {FINAL} seed 0",
              fig_path=f"{FIG_DIR}/figure6b.png")
"""),
    md("""
## Finish

Answer Q2.15–Q2.17. Q2.17 is the 150-word final verdict: did the final model learn motion? What evidence would convince a skeptic, and what test has it still not passed (for example 4× speed, three digits, or a new digit style)?

Download `/kaggle/working/` from every notebook. The tables are in `results/*.csv|md`, the figures in `figures/`, the GIFs in `gifs/`. Push the notebooks, `q2lib.py` and the GIFs to the public GitHub repository.
"""),
]
write("05_final_model.ipynb", nb5)
