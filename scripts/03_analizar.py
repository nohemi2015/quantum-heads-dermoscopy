#!/usr/bin/env python3
"""
03_analizar.py
==============
Paso 3 del estudio cuantico: analisis estadistico, tablas y figuras a partir de las
salidas del paso 2.

Que calcula:
  1. tabla_por_cabeza.csv
       media y desviacion estandar entre semillas de AUC, sensibilidad y especificidad
       (umbral fijado en validacion) para cada backbone x cabeza x d x k.
  2. comparaciones_pareadas.csv
       Delta AUC cuantica - clasica (vqc - mlp, svm_cuantico - svm_rbf, vqc - logreg),
       promediada sobre las semillas, con IC 95 % por bootstrap a nivel de LESION
       (se remuestrean lesiones, no imagenes, porque las imagenes de una misma lesion
       no son independientes). Los dos modelos de cada par se evaluan sobre el MISMO
       remuestreo (comparacion pareada). Incluye p bootstrap bilateral y correccion de
       Holm dentro de cada familia (comparacion x conjunto de evaluacion).
  3. robustez.csv  (si se dan carpetas con --robustez)
       Delta AUC vqc - mlp en cada variante (capas, tasa de aprendizaje).
  4. Figuras (PNG 300 dpi y PDF):
       fig1_auc_vs_k_<conjunto>.png  AUC frente a lesiones por clase, por backbone y d
       fig2_delta_<conjunto>.png     Delta AUC cuantica - clasica con IC 95 %, frente a k

Uso (CMD o PowerShell):
  python 03_analizar.py --cabezas C:\\proyecto_cuantico\\cabezas --salida C:\\proyecto_cuantico\\analisis
      --robustez C:\\proyecto_cuantico\\robustez_capas4 C:\\proyecto_cuantico\\robustez_lr003 C:\\proyecto_cuantico\\robustez_lr03

Opciones: --B 2000 (replicas bootstrap; 2000 es lo recomendado para el articulo).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CONJUNTOS = ["test", "cohorte", "isic2018"]
NOMBRE_CONJ = {"test": "Test interno (299 img.)", "cohorte": "Cohorte reservada (7,124 img.)",
               "isic2018": "ISIC 2018 test (1,511 img.)"}
K_ORDEN = ["25", "50", "100", "200", "todas"]
PARES = [("vqc", "mlp"), ("svm_cuantico", "svm_rbf"), ("vqc", "logreg")]
# Paleta categorica validada (orden fijo). Cuantico = linea discontinua; cada cabeza con su marcador.
ESTILO = {
    "mlp": dict(color="#2a78d6", ls="-", marker="o", label="MLP (mismos parámetros)"),
    "vqc": dict(color="#eb6834", ls="--", marker="s", label="Circuito variacional (VQC)"),
    "svm_rbf": dict(color="#1baf7a", ls="-", marker="^", label="SVM kernel RBF"),
    "svm_cuantico": dict(color="#eda100", ls="--", marker="D", label="SVM kernel cuántico"),
    "logreg": dict(color="#e87ba4", ls="-", marker="v", label="Regresión logística (d entradas)"),
}
TINTA, TINTA2, REJILLA = "#1f1f1e", "#5f5e58", "#e6e5e0"

# Textos de las figuras en espanol (es) o ingles (en, para el manuscrito)
TXT = {
    "es": {"conj": NOMBRE_CONJ, "k": "Lesiones por clase en entrenamiento", "todas": "todas", "auc": "AUC",
           "ref": "Referencia: todas las características", "t1": "AUC frente al tamaño de entrenamiento (media ± DE entre {n} semillas)",
           "dy": "Δ AUC (cuántico − clásico)\n↑ favorece lo cuántico", "qb": "qubits",
           "t2": "Diferencia pareada con IC 95 % (bootstrap por lesión) · {c}",
           "par": {"vqc - mlp": "Circuito variacional − MLP con mismos parámetros", "svm_cuantico - svm_rbf": "Kernel cuántico − kernel RBF"},
           "cab": {k: v["label"] for k, v in ESTILO.items()}},
    "en": {"conj": {"test": "Internal test (299 images)", "cohorte": "Reserved cohort (7,124 images)", "isic2018": "ISIC 2018 test (1,511 images)"},
           "k": "Training lesions per class", "todas": "all", "auc": "AUC",
           "ref": "Reference: all backbone features", "t1": "{c}: AUC versus training size (mean ± SD over {n} seeds)",
           "dy": "Δ AUC (quantum − classical)\n↑ favours quantum", "qb": "qubits",
           "t2": "Paired difference with 95% lesion-level bootstrap CI · {c}",
           "par": {"vqc - mlp": "VQC − parameter-matched MLP", "svm_cuantico - svm_rbf": "Quantum kernel − RBF kernel"},
           "cab": {"mlp": "MLP (matched parameters)", "vqc": "Variational quantum classifier (VQC)", "svm_rbf": "SVM, RBF kernel",
                   "svm_cuantico": "SVM, quantum kernel", "logreg": "Logistic regression (d inputs)"}},
}
L = TXT["es"]


# ----------------------------------------------------------------------------------------
# AUC ponderada vectorizada (pesos = conteos del bootstrap por lesion)
# ----------------------------------------------------------------------------------------
def auc_ponderada(score, y, W):
    """AUC para cada fila de W (B x n) de pesos enteros. Maneja empates como 1/2."""
    orden = np.argsort(score, kind="mergesort")
    s, yy, Ws = score[orden], y[orden], W[:, orden].astype(np.float64)
    # Agrupar empates: bloques de scores identicos
    nuevo = np.r_[True, s[1:] != s[:-1]]
    bloque = np.cumsum(nuevo) - 1
    nb = bloque[-1] + 1
    pos = np.zeros((W.shape[0], nb))
    neg = np.zeros((W.shape[0], nb))
    np.add.at(pos.T, bloque[yy == 1], Ws[:, yy == 1].T)
    np.add.at(neg.T, bloque[yy == 0], Ws[:, yy == 0].T)
    neg_antes = np.cumsum(neg, axis=1) - neg
    num = (pos * (neg_antes + 0.5 * neg)).sum(1)
    return num / (pos.sum(1) * neg.sum(1))


def pesos_lesion(lesiones, B, rng):
    """Matriz B x n: cuantas veces entra cada imagen al remuestrear lesiones con reemplazo."""
    codigos, inv = np.unique(lesiones, return_inverse=True)
    L = len(codigos)
    W = np.empty((B, len(lesiones)), dtype=np.int32)
    for b in range(B):
        conteo = np.bincount(rng.integers(0, L, L), minlength=L)
        W[b] = conteo[inv]
    return W


def holm(p):
    p = np.asarray(p, float)
    orden = np.argsort(p)
    ajust = np.empty_like(p)
    acum = 0.0
    for i, j in enumerate(orden):
        acum = max(acum, min(1.0, (len(p) - i) * p[j]))
        ajust[j] = acum
    return ajust


# ----------------------------------------------------------------------------------------
def cargar(carpeta):
    r = pd.read_csv(Path(carpeta) / "resumen_corridas.csv")
    r["k"] = r["k_por_clase"].astype(str)
    return r


def tabla_por_cabeza(r):
    met = [f"{m}_{c}" for c in ["val"] + CONJUNTOS for m in ("auc", "sens", "esp")]
    g = r.groupby(["backbone", "cabeza", "qubits_d", "k"])
    t = g[met].agg(["mean", "std"])
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t["parametros"] = g["parametros"].median()
    t["n_semillas"] = g.size()
    t = t.reset_index()
    t["k"] = pd.Categorical(t["k"], K_ORDEN, ordered=True)
    return t.sort_values(["backbone", "qubits_d", "k", "cabeza"])


def comparaciones(r, carpeta, B, semilla):
    ids = pd.read_csv(Path(carpeta) / "eval_ids.csv")
    rng = np.random.default_rng(semilla)
    pesos = {c: pesos_lesion(ids.loc[ids["conjunto"] == c, "lesion_id"].to_numpy(), B, rng) for c in CONJUNTOS}
    mascara = {c: (ids["conjunto"] == c).to_numpy() for c in CONJUNTOS}
    y = ids["label"].to_numpy()
    cache = {}

    def preds(run_id):
        if run_id not in cache:
            cache[run_id] = np.load(Path(carpeta) / "preds" / f"{run_id.replace('|', '_')}.npy").astype(np.float64)
        return cache[run_id]

    filas = []
    combos = r[["backbone", "qubits_d", "k"]].drop_duplicates()
    combos = combos[combos["qubits_d"] > 0]
    total = len(combos) * len(PARES)
    hecho = 0
    for _, cb in combos.iterrows():
        sub = r[(r["backbone"] == cb.backbone) & (r["qubits_d"] == cb.qubits_d) & (r["k"] == cb.k)]
        for a, b in PARES:
            hecho += 1
            A = sub[sub["cabeza"] == a].set_index("semilla")["run_id"]
            Bm = sub[sub["cabeza"] == b].set_index("semilla")["run_id"]
            sem = sorted(set(A.index) & set(Bm.index))
            if not sem:
                continue
            for c in CONJUNTOS:
                m, W, yc = mascara[c], pesos[c], y[mascara[c]]
                dif_boot = np.zeros(B)
                dif_obs = []
                for s in sem:
                    pa, pb = preds(A[s])[m], preds(Bm[s])[m]
                    uno = np.ones((1, len(yc)), dtype=np.int32)
                    dif_obs.append(auc_ponderada(pa, yc, uno)[0] - auc_ponderada(pb, yc, uno)[0])
                    dif_boot += auc_ponderada(pa, yc, W) - auc_ponderada(pb, yc, W)
                dif_boot /= len(sem)
                d_obs = float(np.mean(dif_obs))
                p = min(1.0, 2 * min((dif_boot <= 0).mean(), (dif_boot >= 0).mean()))
                filas.append({"backbone": cb.backbone, "qubits_d": cb.qubits_d, "k": cb.k, "comparacion": f"{a} - {b}",
                              "conjunto": c, "n_semillas": len(sem), "delta_auc": d_obs,
                              "ic95_inf": np.percentile(dif_boot, 2.5), "ic95_sup": np.percentile(dif_boot, 97.5),
                              "p_bootstrap": max(p, 1 / B), "semillas_gana_cuantico": int(np.sum(np.array(dif_obs) > 0))})
            print(f"  comparaciones {hecho}/{total}", end="\r", flush=True)
    print()
    c = pd.DataFrame(filas)
    c["p_holm"] = c.groupby(["comparacion", "conjunto"])["p_bootstrap"].transform(holm)
    c["k"] = pd.Categorical(c["k"], K_ORDEN, ordered=True)
    return c.sort_values(["comparacion", "conjunto", "backbone", "qubits_d", "k"])


def robustez(carpetas, base):
    filas = []
    variantes = [("base (capas 2, lr 0.01)", base)] + [(Path(p).name, cargar(p)) for p in carpetas]
    for nombre, r in variantes:
        key = ["backbone", "qubits_d", "k", "semilla"]
        A = r[r["cabeza"] == "vqc"].set_index(key)
        Bm = r[r["cabeza"] == "mlp"].set_index(key)
        comun = A.index.intersection(Bm.index)
        for c in CONJUNTOS:
            d = (A.loc[comun, f"auc_{c}"] - Bm.loc[comun, f"auc_{c}"]).groupby(["backbone", "qubits_d", "k"]).agg(["mean", "std", "size"])
            gana = (A.loc[comun, f"auc_{c}"] > Bm.loc[comun, f"auc_{c}"]).groupby(["backbone", "qubits_d", "k"]).sum()
            d = d.join(gana.rename("semillas_gana_vqc")).reset_index()
            d["variante"], d["conjunto"] = nombre, c
            d["parametros_vqc"] = d.apply(lambda f: int(A.loc[(f.backbone, f.qubits_d, f.k)]["parametros"].iloc[0]), axis=1)
            filas.append(d)
    out = pd.concat(filas).rename(columns={"mean": "delta_auc_media", "std": "delta_auc_de", "size": "n_semillas"})
    d_rob = set().union(*[set(r.loc[r["cabeza"] == "vqc", "qubits_d"]) for _, r in variantes[1:]])
    return out[out["qubits_d"].isin(d_rob)] if d_rob else out


# ----------------------------------------------------------------------------------------
def estilo_ejes(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA2)
    ax.tick_params(colors=TINTA2, labelsize=8)
    ax.grid(axis="y", color=REJILLA, lw=0.8)
    ax.set_axisbelow(True)


def fig_auc(t, conjunto, salida):
    bbs = sorted(t["backbone"].unique())
    ds = sorted(d for d in t["qubits_d"].unique() if d > 0)
    fig, axes = plt.subplots(len(bbs), len(ds), figsize=(3.1 * len(ds), 2.7 * len(bbs)), sharex=True, sharey=True, squeeze=False)
    x = np.arange(len(K_ORDEN))
    for i, bb in enumerate(bbs):
        ref = t[(t["backbone"] == bb) & (t["cabeza"] == "logreg_completa")].set_index("k").reindex(K_ORDEN)
        for j, d in enumerate(ds):
            ax = axes[i, j]
            estilo_ejes(ax)
            if len(ref) and ref[f"auc_{conjunto}_mean"].notna().any():
                ax.plot(x, ref[f"auc_{conjunto}_mean"], color="#9a9890", lw=1.5, ls=":", label=L["ref"])
            for cab, es in ESTILO.items():
                s = t[(t["backbone"] == bb) & (t["qubits_d"] == d) & (t["cabeza"] == cab)].set_index("k").reindex(K_ORDEN)
                if s[f"auc_{conjunto}_mean"].isna().all():
                    continue
                m, sd = s[f"auc_{conjunto}_mean"].to_numpy(), s[f"auc_{conjunto}_std"].fillna(0).to_numpy()
                ax.fill_between(x, m - sd, m + sd, color=es["color"], alpha=0.12, lw=0)
                ax.plot(x, m, color=es["color"], ls=es["ls"], lw=2, marker=es["marker"], ms=5,
                        markeredgecolor="white", markeredgewidth=1, label=L["cab"][cab])
            ax.set_title(f"{'EfficientNet-B1' if bb == 'effb1' else 'ViT-B/32'}  ·  d = {d} {L['qb']}", fontsize=9, color=TINTA)
            ax.set_xticks(x, [L["todas"] if k == "todas" else k for k in K_ORDEN])
            if i == len(bbs) - 1:
                ax.set_xlabel(L["k"], fontsize=8, color=TINTA2)
            if j == 0:
                ax.set_ylabel(L["auc"], fontsize=8, color=TINTA2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(L["t1"].format(c=L["conj"][conjunto], n=int(t["n_semillas"].max())), fontsize=10, color=TINTA, y=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(Path(salida) / f"fig1_auc_vs_k_{conjunto}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_delta(c, conjunto, salida):
    pares = [f"{a} - {b}" for a, b in PARES[:2]]
    titulos = L["par"]
    bbs = sorted(c["backbone"].unique())
    colores_d = {4: "#86b6ef", 6: "#2a78d6", 8: "#184f95"}  # rampa ordinal de un solo tono
    fig, axes = plt.subplots(len(bbs), 2, figsize=(9, 2.8 * len(bbs)), sharex=True, sharey=True, squeeze=False)
    x = np.arange(len(K_ORDEN))
    for i, bb in enumerate(bbs):
        for j, par in enumerate(pares):
            ax = axes[i, j]
            estilo_ejes(ax)
            ax.axhline(0, color=TINTA, lw=1)
            s = c[(c["backbone"] == bb) & (c["comparacion"] == par) & (c["conjunto"] == conjunto)]
            for k_d, (d, col) in enumerate(colores_d.items()):
                sd = s[s["qubits_d"] == d].set_index("k").reindex(K_ORDEN)
                if sd["delta_auc"].isna().all():
                    continue
                xx = x + (k_d - 1) * 0.18
                ax.errorbar(xx, sd["delta_auc"], yerr=[sd["delta_auc"] - sd["ic95_inf"], sd["ic95_sup"] - sd["delta_auc"]],
                            fmt="o", color=col, ms=5, lw=1.5, capsize=0, markeredgecolor="white", markeredgewidth=1,
                            label=f"d = {d} {L['qb']}")
            ax.set_title(f"{'EfficientNet-B1' if bb == 'effb1' else 'ViT-B/32'}  ·  {titulos[par]}", fontsize=9, color=TINTA)
            ax.set_xticks(x, [L["todas"] if k == "todas" else k for k in K_ORDEN])
            if i == len(bbs) - 1:
                ax.set_xlabel(L["k"], fontsize=8, color=TINTA2)
            if j == 0:
                ax.set_ylabel(L["dy"], fontsize=8, color=TINTA2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(L["t2"].format(c=L["conj"][conjunto]), fontsize=10, color=TINTA, y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(Path(salida) / f"fig2_delta_{conjunto}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cabezas", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--robustez", nargs="*", default=[])
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--semilla", type=int, default=2026)
    ap.add_argument("--idioma", choices=["es", "en"], default="en", help="idioma de las figuras (en por defecto; es solo para revisar)")
    ap.add_argument("--solo-figuras", action="store_true", help="reusar comparaciones_pareadas.csv y solo redibujar")
    a = ap.parse_args()
    global L
    L = TXT[a.idioma]
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)

    r = cargar(a.cabezas)
    print(f"Corridas: {len(r)}")
    t = tabla_por_cabeza(r)
    t.to_csv(sal / "tabla_por_cabeza.csv", index=False)
    if a.solo_figuras and (sal / "comparaciones_pareadas.csv").exists():
        c = pd.read_csv(sal / "comparaciones_pareadas.csv")
        c["k"] = pd.Categorical(c["k"].astype(str), K_ORDEN, ordered=True)
    else:
        print(f"Bootstrap por lesion con B = {a.B} replicas...")
        c = comparaciones(r, a.cabezas, a.B, a.semilla)
        c.to_csv(sal / "comparaciones_pareadas.csv", index=False)
    if a.robustez:
        rb = robustez(a.robustez, r)
        rb.to_csv(sal / "robustez.csv", index=False)
        print("\nRobustez (cohorte, Delta AUC vqc - mlp, media entre k):")
        print(rb[rb["conjunto"] == "cohorte"].groupby(["variante", "backbone", "qubits_d"])["delta_auc_media"].mean().unstack().round(3).to_string())
    for conj in CONJUNTOS:
        fig_auc(t, conj, sal)
        fig_delta(c, conj, sal)

    # Resumen en consola
    print("\nResumen (Delta AUC cuantico - clasico; IC 95 % bootstrap por lesion):")
    for (comp, conj), g in c.groupby(["comparacion", "conjunto"], observed=True):
        sig_neg = int(((g["ic95_sup"] < 0) & (g["p_holm"] < 0.05)).sum())
        sig_pos = int(((g["ic95_inf"] > 0) & (g["p_holm"] < 0.05)).sum())
        print(f"  {comp:24s} {conj:9s} media {g['delta_auc'].mean():+.3f} | rango [{g['delta_auc'].min():+.3f}, {g['delta_auc'].max():+.3f}] | "
              f"significativas (Holm): a favor clasico {sig_neg}/{len(g)}, a favor cuantico {sig_pos}/{len(g)}")
    (sal / "parametros_analisis.json").write_text(json.dumps({"B": a.B, "semilla": a.semilla, "cabezas": a.cabezas,
                                                              "robustez": a.robustez}, indent=2))
    print(f"\nListo. Archivos en {sal}")


if __name__ == "__main__":
    main()
