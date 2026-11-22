# Quantum classification heads for dermoscopic skin-lesion diagnosis

Code, data partition and run-level results for the paper

> Sánchez-Medel N, Altamirano-Robles L, Díaz-Hernández R, Ramírez-Gutiérrez KA, Zapotecas-Martínez S, Velázquez-Arreola JJ.
> *Do Quantum Classification Heads Help Dermoscopic Skin-Lesion Diagnosis? A Parameter-Matched, Lesion-Level Benchmark with Held-Out and Cross-Dataset Evaluation.*
> Submitted to *Quantum Machine Intelligence* (2026).

**Version history.** v1.0: code, partition and results of the main, robustness and encoding analyses. v1.1: sensitivity analyses requested in internal review (new script 05, training options `--mlp-lado`, `--mlp-angulo`, `--peso-lesion`, `--qk-escalas`, `--rbf-gammas`, and their results). The Zenodo record of v1.1 also archives the per-image predictions of the main design.

The study compares quantum classification heads with classical heads of equal capacity on identical, frozen image features:

- a **variational quantum classifier (VQC)** against a **parameter-matched multilayer perceptron (MLP)**;
- an **SVM with an IQP quantum fidelity kernel** against an **SVM with a Gaussian (RBF) kernel**;
- a logistic regression on the same inputs, and a logistic regression on all backbone features as a reference.

Features come from EfficientNet-B1 and ViT-B/32 (ImageNet weights, frozen) and are reduced by PCA to *d* = 4, 6 or 8 dimensions, one per qubit. Heads are trained on 25, 50, 100, 200 or all lesions per class with five seeds, and evaluated on a lesion-disjoint test set, a reserved HAM10000 cohort and the ISIC 2018 test set, with paired lesion-level bootstrap inference.

Quantum circuits are simulated exactly (state vector) in PyTorch. The simulator is checked against PennyLane at start-up.

## Repository contents

```
scripts/
  01_extraer_caracteristicas.py   feature extraction (frozen backbones) and manifest of all evaluation sets
  02_entrenar_cabezas.py          training and evaluation of all heads (quantum circuits simulated in PyTorch)
  03_analizar.py                  summary tables, paired lesion-level bootstrap with Holm correction, figures
  04_tablas_latex.py              LaTeX tables and numeric macros used in the manuscript
  05_analisis_revision.py         sensitivity analyses on stored predictions (lesion-level AUC, hierarchical
                                  bootstrap, without akiec, encoding angles, sampled-kernel spectrum, data checks)
data/
  split_ledger.csv                lesion-level partition of the 2,000-image HAM10000 subset
results/
  main/                           800 runs of the main design (+ paired bootstrap comparisons)
  encoding_pi/                    VQC with the angle encoding pi*tanh(z) (encoding ablation)
  robustness/                     d = 6, 8: learning rates, depth (L = 4, 6, 8), data re-uploading, 1,024 shots;
                                  all d: MLP controls, lesion-weighted training, extended kernel grids (v1.1)
```

`data/split_ledger.csv` has SHA-256 `8ecf5ac82a7b34520f9f3c1a95d9948108363bbc44d02ae8a18975141df83b37`. Each lesion (`lesion_id`) belongs to exactly one partition (`train`, `val` or `test`).

The images are **not** included. Download them from their original sources:

- HAM10000 (images and `HAM10000_metadata.csv`): https://doi.org/10.7910/DVN/DBW86T
- ISIC 2018 Task 3 test images and ground truth: https://challenge.isic-archive.com/data/

The extracted features (`feats_*.npz`) and the per-image predictions of every run (`preds/*.npy`) are archived on Zenodo (see the DOI badge or the release notes).

## Installation

Python 3.11 and a CUDA GPU were used (any GPU or the CPU also works; the heads are small).

```bash
pip install -r requirements.txt
```

## Reproducing the study

Commands are shown for a Unix shell; on Windows PowerShell use the same arguments with Windows paths.

**1. Feature extraction**

```bash
python scripts/01_extraer_caracteristicas.py \
  --ledger data/split_ledger.csv \
  --ham-meta /path/HAM10000_metadata.csv --ham-dir /path/HAM10000_images \
  --isic-gt /path/ISIC2018_Task3_Test_GroundTruth.csv --isic-dir /path/ISIC2018_Task3_Test_Input \
  --salida features
```

This writes `manifiesto.csv`, `feats_effb1.npz`, `feats_vitb32.npz` and `resumen_extraccion.json` (software versions and checks). The reserved cohort is every HAM10000 image whose lesion is not in the ledger.

**2. Main design (800 runs)**

```bash
python scripts/02_entrenar_cabezas.py --features features --salida runs/main
```

Defaults: both backbones, *d* = 4, 6, 8, *L* = 2 layers, *k* = 25, 50, 100, 200, all lesions per class, seeds 0–4, all six heads, Adam with learning rate 0.01, up to 3,000 steps, patience 200, angle encoding (π/2)·tanh(z). Runs are resumable: finished runs are skipped.

**3. Robustness and encoding analyses**

```bash
python scripts/02_entrenar_cabezas.py --features features --salida runs/lr0003   --qubits 6,8 --cabezas mlp,vqc --lr 0.003
python scripts/02_entrenar_cabezas.py --features features --salida runs/lr003    --qubits 6,8 --cabezas mlp,vqc --lr 0.03
python scripts/02_entrenar_cabezas.py --features features --salida runs/layers4  --qubits 6,8 --cabezas mlp,vqc --capas 4
python scripts/02_entrenar_cabezas.py --features features --salida runs/layers6  --qubits 6,8 --cabezas mlp,vqc --capas 6
python scripts/02_entrenar_cabezas.py --features features --salida runs/layers8  --qubits 6,8 --cabezas mlp,vqc --capas 8
python scripts/02_entrenar_cabezas.py --features features --salida runs/reupload --qubits 6,8 --cabezas mlp,vqc --resubir
python scripts/02_entrenar_cabezas.py --features features --salida runs/shots    --qubits 6,8 --cabezas mlp,vqc,svm_rbf,svm_cuantico --shots 1024
python scripts/02_entrenar_cabezas.py --features features --salida runs/encoding_pi --cabezas vqc --escala-angulo 1.0
# Sensitivity analyses added in v1.1 (all d)
python scripts/02_entrenar_cabezas.py --features features --salida runs/mlp_other_side --cabezas mlp --mlp-lado otro
python scripts/02_entrenar_cabezas.py --features features --salida runs/mlp_bounded_input --cabezas mlp --mlp-angulo
python scripts/02_entrenar_cabezas.py --features features --salida runs/lesion_weighted --cabezas logreg,mlp,vqc,svm_rbf,svm_cuantico --peso-lesion
python scripts/02_entrenar_cabezas.py --features features --salida runs/extended_grid --cabezas svm_rbf,svm_cuantico \
  --qk-escalas 0.00625,0.0125,0.025,0.05,0.1,0.2,0.4 --rbf-gammas 0.003,0.01,0.03,0.1,0.3,1,3
python scripts/05_analisis_revision.py --features features --principal runs/main --shots runs/shots --salida analysis/revision
```

With more layers the MLP is enlarged automatically to keep the number of parameters matched.

**4. Statistics, figures and LaTeX tables**

```bash
python scripts/03_analizar.py --cabezas runs/main --salida analysis
python scripts/04_tablas_latex.py --resumen runs/main/resumen_corridas.csv --salida tables \
  --comparaciones analysis/comparaciones_pareadas.csv \
  --codificacion runs/encoding_pi/resumen_corridas.csv \
  --robustez "B@lr \$0.003\$=runs/lr0003/resumen_corridas.csv" "C@lr \$0.03\$=runs/lr003/resumen_corridas.csv" \
             "D@\$L=4\$ layers=runs/layers4/resumen_corridas.csv" "G@\$L=6\$ layers=runs/layers6/resumen_corridas.csv" \
             "H@\$L=8\$ layers=runs/layers8/resumen_corridas.csv" "E@Data re-uploading=runs/reupload/resumen_corridas.csv" \
             "F@1,024 shots=runs/shots/resumen_corridas.csv"
```

The bootstrap step needs the per-image predictions (`runs/main/preds/`). To regenerate only the tables and figures from the files in `results/`, replace `runs/...` with `results/...`; `03_analizar.py --solo-figuras` reuses `comparaciones_pareadas.csv` and skips the bootstrap.

## Result files

Code comments and column names are in Spanish. Each row of `resumen_corridas.csv` is one run:

| Column | Meaning |
|---|---|
| `run_id` | `backbone\|head\|d\|k\|seed` |
| `backbone` | `effb1` (EfficientNet-B1) or `vitb32` (ViT-B/32) |
| `cabeza` | head: `vqc`, `mlp`, `logreg` (*d* inputs), `logreg_completa` (all features), `svm_rbf`, `svm_cuantico` (quantum kernel) |
| `qubits_d` | input dimension *d* (number of qubits); 0 for the full-feature reference |
| `capas` | circuit layers *L* (0 for classical heads) |
| `k_por_clase` | training lesions per class (`todas` = all) |
| `semilla` | seed |
| `n_train_img`, `n_train_lesiones` | training images and lesions |
| `parametros` | trainable parameters (SVMs: support vectors + 1) |
| `pasos` | optimisation steps until the best validation loss |
| `hiper` | selected hyperparameters; for variants, `shots=…`, `reupload` or `angle=…` |
| `auc_*`, `sens_*`, `esp_*` | AUC, sensitivity and specificity on `train`, `val`, `test` (internal test), `cohorte` (reserved HAM10000 cohort) and `isic2018` |
| `umbral_val` | decision threshold (Youden index on validation) |
| `segundos` | run time |

`comparaciones_pareadas.csv` contains, for each comparison (`vqc - mlp`, `svm_cuantico - svm_rbf`, `vqc - logreg`), evaluation set, backbone, *d* and *k*, the paired difference in AUC averaged over seeds (`delta_auc`), its 95% lesion-level bootstrap interval (`ic95_inf`, `ic95_sup`; B = 2,000), the number of seeds won by the quantum head (`semillas_gana_cuantico`), and the bootstrap and Holm-adjusted *p* values (`p_bootstrap`, `p_holm`).

## Citation

If you use this code or the partition, please cite the paper (see `CITATION.cff`).

## License

Code: MIT (see `LICENSE`). HAM10000 and ISIC 2018 are subject to their own licences (CC BY-NC 4.0).
