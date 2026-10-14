# nmrid: compound identification from ¹H NMR spectra

This is the code for the senior thesis *Machine Learning for NMR Spectroscopy: Compound Identification from ¹H NMR Spectra*. It benchmarks a Random Forest on hand-crafted features, a 1D CNN and a Siamese CNN on nmrshiftdb2 spectra, with a no-learning cosine library search as the reference point. It runs the four planned experiments:

| Research question | Command | Main outputs |
|---|---|---|
| RQ1: which architecture identifies compounds best? | `--rq 1` | `rq1_results.csv`, `rq1_table.tex`, `rq1_comparison.png`, confusion matrices |
| RQ2: how robust are the models to noise and shift drift? | `--rq 2` | `rq2_noise.png`, `rq2_shift.png` |
| RQ3: do hand-crafted features compete with deep learning? | `--rq 3` | `rq3_feature_engineering.png`, RF feature-group importance |
| RQ4: how does accuracy grow with training data? | `--rq 4` | `rq4_learning_curves.png` |
| Extension: can the model find compounds it never saw in training? | `--rq openset` | `openset_results.csv` |

The pipeline was written for this project. It takes design ideas from the pSCNN code of Wei et al. (2022), but no code is copied from it (see [Relation to pSCNN](#relation-to-the-pscnn-code-wei-et-al-2022)).

---

## 1. Pipeline at a glance

```
nmrshiftdb2 SD file ──► parse ¹H peak lists (RDKit proton counts, NMReDATA multiplets)
                             │
                             ▼
        K library compounds, each with one reference peak list
                             │
     ┌───────────────────────┴───────────────────────────┐
     ▼                                                   ▼
reference spectrum                    R simulated "re-measurements" per compound
(Lorentzian, 2,000 bins, 0–12 ppm)     (shift drift, 300–600 MHz, line width, integral
                                         jitter, missing peaks, solvent peaks, noise)
     │                                                   │ split 70/15/15 per compound
     ▼                                                   ▼
 ┌───────────────┬──────────────────────┬────────────────────────┬─────────────────────┐
 │ cosine search │ Random Forest        │ 1D CNN                 │ Siamese CNN          │
 │ (baseline)    │ 145 features         │ 4 conv blocks → K-way  │ shared encoder →     │
 │               │ 500 trees            │ softmax                │ P(same compound)     │
 └───────────────┴──────────────────────┴────────────────────────┴─────────────────────┘
                             │  every model → score matrix (queries × K compounds)
                             ▼
     top-1 / top-5 accuracy, macro-F1, confusion analysis, ms per spectrum
```

## 2. Install

Python 3.9 or newer. A GPU is optional.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                     # about 2 minutes; everything should pass
```

With conda you can instead run `conda env create -f environment.yml` and then `conda activate nmrid`.

## 3. Quick start with synthetic data (no download)

```bash
python -m nmrid prepare    -c configs/demo.yaml
python -m nmrid experiment -c configs/demo.yaml --rq all
```

This builds 60 synthetic compounds and runs every experiment in about 10–20 minutes on a laptop CPU. Results go to `results/demo_synthetic/`. **The numbers are a smoke test of the code and cannot be used as thesis results.**

## 4. Running the thesis experiments on nmrshiftdb2

1. **Download the data.** Get `nmrshiftdb2withsignals.sd` from <https://sourceforge.net/projects/nmrshiftdb2/files/data/> and save it as `data/raw/nmrshiftdb2withsignals.sd`. The reader also accepts `.sd.gz` files and files that use NMReDATA tags.

2. **Check the parser before anything else.**
   ```bash
   python -m nmrid inspect --sdf data/raw/nmrshiftdb2withsignals.sd -n 5
   ```
   This prints each record's fields and the resonances parsed from them (shift, number of protons, couplings). Compare two or three records against the nmrshiftdb2 web page for the same molecule. The parser follows the documented `shift;intensity;atom|…` format, but it was written without access to the full file. If the output looks wrong, adjust `parse_classic_1h` in `nmrid/io_nmrshiftdb.py` before going further.

3. **Build the dataset.** This takes a few minutes and prints statistics, including how many compounds have more than one real ¹H spectrum.
   ```bash
   python -m nmrid prepare -c configs/default.yaml
   ```

4. **Run the experiments.** You can run them all together or one at a time. Trained models are cached, so RQ2 and RQ3 reuse the RQ1 models.
   ```bash
   python -m nmrid experiment -c configs/default.yaml --rq 1
   python -m nmrid experiment -c configs/default.yaml --rq 2 3
   python -m nmrid experiment -c configs/default.yaml --rq 4 openset
   ```

5. **Final RQ1 table with error bars.** Set `experiments.seeds: [0, 1, 2]` so that `rq1_results.csv` and `rq1_table.tex` report mean ± standard deviation over three trainings.

To change any setting without editing YAML, use `--set`, for example `--set data.n_compounds=500 --set cnn.epochs=30`.

**Expected cost.** These are estimates scaled from measurements on a 2-core machine, for K = 1,000 compounds (15,000 training spectra):

| Step | Time on an 8-core laptop CPU |
|---|---|
| Random Forest, 500 trees | a few minutes |
| 1D CNN, up to 50 epochs | 20–40 min |
| Siamese CNN, up to 40 epochs | 1–2 h |
| RQ1–RQ4 plus open-set | roughly 8–12 h |

A free Colab GPU cuts the neural-network times by about 5–10×. You can also start with `n_compounds: 500` to check timings first.

## 5. Evaluation protocol

nmrshiftdb2 usually has **one** ¹H spectrum per compound. A classifier can only predict classes it has seen during training. So if whole compounds are held out, a test compound is a class the RF and CNN have never seen, and they cannot identify it. The pipeline therefore uses two protocols.

**Closed-set identification (RQ1–RQ4, all models).** The task is: "Which of the K library compounds produced this spectrum?"
- Each compound gets a clean reference spectrum and R = 20 simulated re-measurements.
- The re-measurements of *each compound* are split 14 / 3 / 3 (70/15/15) into train, validation and test.
- Every class therefore appears in every split, but no test spectrum is ever used for training.
- This is the same logic Wei et al. (2022) used: their test set came from augmenting the same 24 standards.
- Compounds that have a second real nmrshiftdb2 spectrum add a **real re-measurement** test set with no simulation at all. It is reported as `real_top1` and `real_top5`, and it plays the role of Wei et al.'s experimental-mixture validation.

**Open-set retrieval (extension).** Here the *compounds* are split 70/15/15, as in the original plan.
- The Siamese network trains on 70% of the compounds and stops early based on unseen validation compounds.
- It is then tested on the remaining 15%: queries from compounds it never saw are searched against all K references.
- Only similarity-based methods can take part: the Siamese CNN, CNN embeddings with nearest-neighbour search, and cosine search.
- This measures the main practical advantage of metric learning: new compounds can be added to the library without retraining.

The closed-set design also changes one sentence in the Definitions chapter. Section 2.2 currently says that spectra are split 70/15/15 by compound identity. That sentence now needs to describe the two protocols above.

## 6. Design decisions to report in the thesis

- **Spectrum representation.** The grid has 2,000 bins over 0–12 ppm (0.006 ppm per bin). Each resonance is a Lorentzian with half-width 0.012 ppm. The area inside each bin is computed exactly from the Lorentzian's cumulative distribution (arctan) rather than sampled at the bin centre, so the integrals are conserved. When coupling constants are available (NMReDATA), multiplets follow the first-order n + 1 rule. Every spectrum is normalised to a maximum of 1.

- **Measurement simulator** (`nmrid/simulate.py`). It adds:
  - a global referencing offset (σ = 0.005 ppm);
  - per-resonance drift (σ = 0.01 ppm), comparable to the shift variation in Wei et al. (2022, Fig. 6a);
  - a random spectrometer frequency (300–600 MHz), which changes multiplet widths in ppm;
  - line-width (×0.7–1.5) and integral jitter;
  - a 5% chance that a resonance is missing;
  - residual solvent and water peaks (shift values from Fulmer et al., 2010);
  - Gaussian noise.

  All parameters are in the `simulation:` block of the config.

- **CNN pooling keeps position.** A methyl singlet at 1.2 ppm and one at 3.7 ppm have the same local shape but different chemistry. Global average pooling throws away *where* a pattern occurred. In the synthetic tests, a CNN with global average pooling stayed near chance level (1–7% top-1), while the same network pooling to 16 ordered segments of 0.75 ppm reached 77–82%. The encoder therefore uses `AdaptiveAvgPool1d(16)` after a 1×1 bottleneck to 32 channels. RQ3 includes the global-average-pooling version as an ablation, so this finding is reproduced on the real data.

- **Parameter counts.** The encoder has 274k parameters. The CNN has about 290k for K = 60, about 400k for K = 500 and about 530k for K = 1,000, because the output layer grows with K. The Siamese CNN has 340k. The run records the exact counts in `rq1_results.csv`.

- **145 hand-crafted features.** These are:
  - 100 histogram windows of 0.12 ppm;
  - the positions of the 20 tallest peaks, sorted by ppm (−1 means no peak);
  - their heights;
  - the peak count;
  - four moments of the spectrum treated as a distribution over ppm (mean, standard deviation, skewness, kurtosis).

  The Definitions chapter currently says 143, so that number needs updating.

- **Memory-compact Random Forest** (`nmrid/rf.py`). scikit-learn stores a 1,000-class probability vector at every node of every tree. That came to 514 MB for 40 trees here, or about 6.4 GB for 500 trees. The compact forest trains the same scikit-learn decision trees (bootstrap weights, √d features per split, Gini impurity) but keeps only the splits and the sparse leaf distributions. It reached the same accuracy in the test (0.939 vs 0.939) at about 2 MB per 40 trees. Set `rf.backend: sklearn` to use the stock class.

- **Siamese training.** Each pair is a replicate plus its compound's reference or another replicate (positive), or another compound's spectrum (negative). Half of the negatives are hard negatives: the 20 most similar references by cosine similarity. The head sees |a − b| and a ⊙ b, so the score does not depend on input order. Early stopping uses retrieval top-1 on validation queries, which is the quantity actually reported. The Siamese pair ACC, TPR and FPR (the metrics in Wei et al.) are also written to `rq1_siamese_pair_metrics.csv`.

- **Fair training budget.** Small training sets (low learning-curve fractions) are cycled until each epoch has at least 50 batches. This way the models at 10% data are not simply under-trained.

- **The cosine baseline is essential.** It is what classical library-search software does. If a learned model cannot beat it, that is a finding worth reporting, not something to hide.

## 7. Identifying a measured spectrum

```bash
python -m nmrid search -c configs/default.yaml --query path/to/bruker/sample --model siamese --top 10
python -m nmrid search -c configs/default.yaml --query spectrum.csv --model cnn --exclude 7.20 7.32
```

A Bruker folder (processed `pdata/1`, or a raw FID if no processed data exists) or a two-column CSV (ppm, intensity) is binned onto the same grid. The command prints the ten best-matching library compounds. `--exclude` blanks solvent windows. This replaces the query-against-all-standards loop in the `__main__` block of `pSCNN.py`.

## 8. Outputs

```
results/<run_name>/
  config.json                 exact settings used
  dataset_info.json           parsing statistics and split sizes
  models/                     *.pt (weights + history), *.joblib (RF), *.json (training time, parameters)
  tables/                     every number behind every figure (CSV) and rq1_table.tex
  figures/                    300-dpi PNGs: RQ figures, training curves, confusion matrices, example spectra
```

## 9. Relation to the pSCNN code (Wei et al., 2022)

| | pSCNN (provided source) | this pipeline |
|---|---|---|
| Framework | TensorFlow 2.5 / Keras | PyTorch |
| Task | binary: is compound X in this mixture? | K-way identification of a pure compound, plus open-set retrieval |
| Branch weights | separate ("pseudo-Siamese") | shared (Siamese); `siamese.shared_weights: false` gives the pseudo-Siamese variant for comparison |
| Encoder | 6 × Conv1D(32, k=5) + max-pool, flatten | 4 blocks, 32→256 filters, k = 7/5/3/3, BatchNorm, position-preserving pooling |
| Pair head | concatenate → Dense(100) → sigmoid | [\|a − b\|, a ⊙ b] → MLP → sigmoid, or contrastive loss |
| Input | full-resolution Bruker spectrum, divided by its maximum | 2,000-bin spectrum built from nmrshiftdb2 peak lists; Bruker spectra are binned onto the same grid |
| Augmentation | superposition of standards at random ratios + noise | simulated re-measurements + on-the-fly noise, shift and scaling |
| Metrics | ACC, TPR, FPR | top-1, top-5, macro precision / recall / F1, confusion analysis, ms per spectrum; pair ACC / TPR / FPR for comparison |
| Baselines | none | cosine library search, Random Forest |

Things noticed in the provided pSCNN source, which may be worth a sentence in the related-work discussion:

- In `pSCNN.py`, `create_input_layers` never appends to its list, so it returns no inputs. The copy in `train_pSCNN.py` fixes this.
- `model.fit(..., validation_split=0.1)` takes the *last* 10% of the arrays before any shuffling. This is only safe if the augmented pairs are stored in random order.
- The loss/accuracy figure uses two y-axes on one panel, and all paths are hard-coded Windows paths.

## 10. Limitations

- Most test spectra are simulated re-measurements of the reference peak list. The real re-measurement set is the check against reality, and its size depends on how many compounds in nmrshiftdb2 have more than one ¹H spectrum.
- In the classic nmrshiftdb2 format, spectra are peak lists without coupling constants, so most resonances are rendered as singlets. Multiplet structure is only used when NMReDATA J values are present.
- The simulator parameters are reasonable choices, not measured instrument statistics. RQ2 probes sensitivity to them.

## 11. Tests

`python -m pytest -q` checks:
- rendering (integral conservation and the n + 1 multiplet rule);
- the SD parser on hand-built classic and NMReDATA records, including proton counting with implicit and explicit hydrogens;
- the Bruker reader's ppm axis, on a processed spectrum written in the pSCNN folder layout;
- feature layout and model shapes;
- that the Siamese and pseudo-Siamese parameter counts differ by exactly one encoder;
- a complete tiny end-to-end run, including library search.

## References used in the code

- Wei, W., Liao, Y., Wang, Y., Wang, S., Du, W., Lu, H., Kong, B., Yang, H., Zhang, Z. (2022). Deep learning-based method for compound identification in NMR spectra of mixtures. *Molecules* 27(12), 3653. doi:10.3390/molecules27123653. Code: <https://github.com/yuxuanliao/pSCNN>
- Kuhn, S. et al. (2024). Twenty years of nmrshiftdb2: a case study of an open database for analytical chemistry. *Magnetic Resonance in Chemistry*. doi:10.1002/mrc.5418
- Fulmer, G. R. et al. (2010). NMR chemical shifts of trace impurities. *Organometallics* 29, 2176–2179.
- Breiman, L. (2001). Random forests. *Machine Learning* 45, 5–32.
- Hadsell, R., Chopra, S., LeCun, Y. (2006). Dimensionality reduction by learning an invariant mapping. *CVPR*.
