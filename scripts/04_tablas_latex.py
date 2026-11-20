#!/usr/bin/env python3
"""
04_tablas_latex.py
==================
Genera las tablas del manuscrito (archivos .tex en manuscrito/tables/) directamente
desde resumen_corridas.csv, para que ninguna cifra se copie a mano.

Uso:
  python 04_tablas_latex.py --resumen C:\\proyecto_cuantico\\cabezas\\resumen_corridas.csv --salida C:\\proyecto_cuantico\\manuscrito\\tables
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BB = {"effb1": "EfficientNet-B1", "vitb32": "ViT-B/32"}
CAB = {"logreg": "LogReg ($d$ inputs)", "mlp": "MLP (matched)", "vqc": "VQC", "svm_rbf": "SVM-RBF",
       "svm_cuantico": "SVM-QK (quantum kernel)"}
K = ["25", "50", "100", "200", "todas"]
K_TEX = {"25": "25", "50": "50", "100": "100", "200": "200", "todas": "all"}


def fmt(m, s):
    return f"{m:.3f} $\\pm$ {s:.3f}"


def tabla_parametros(r, sal):
    t = r[r["cabeza"].isin(["logreg", "mlp", "vqc"])].groupby(["cabeza", "qubits_d"])["parametros"].median().unstack()
    filas = [f"{CAB[c]} & " + " & ".join(str(int(t.loc[c, d])) for d in t.columns) + r" \\" for c in ["logreg", "mlp", "vqc"]]
    sv = r[r["cabeza"].isin(["svm_rbf", "svm_cuantico"]) & (r["k"] == "todas")].groupby(["cabeza", "qubits_d"])["parametros"].median().unstack()
    filas += [f"{CAB[c]}$^\\dagger$ & " + " & ".join(str(int(sv.loc[c, d])) for d in sv.columns) + r" \\" for c in ["svm_rbf", "svm_cuantico"]]
    cuerpo = "\n".join(filas)
    (sal / "tab_parameters.tex").write_text(
        "\\begin{tabular}{lrrr}\n\\toprule\nHead & $d=4$ & $d=6$ & $d=8$ \\\\\n\\midrule\n" + cuerpo +
        "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def tabla_principal(r, sal):
    full = r[r["k"] == "todas"]
    lineas = []
    for bb in ["effb1", "vitb32"]:
        ref = full[(full["backbone"] == bb) & (full["cabeza"] == "logreg_completa")]
        lineas.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{BB[bb]}}}}} \\\\")
        for d in [4, 6, 8]:
            for c in ["logreg", "mlp", "vqc", "svm_rbf", "svm_cuantico"]:
                s = full[(full["backbone"] == bb) & (full["qubits_d"] == d) & (full["cabeza"] == c)]
                celdas = []
                for conj in ["test", "cohorte", "isic2018"]:
                    v = s[f"auc_{conj}"]
                    celdas.append(fmt(v.mean(), v.std(ddof=1)))
                pref = f"$d={d}$" if c == "logreg" else ""
                lineas.append(f"{pref} & {CAB[c]} & " + " & ".join(celdas) + r" \\")
            lineas.append("\\addlinespace[2pt]")
        celdas = [fmt(ref[f"auc_{conj}"].mean(), ref[f"auc_{conj}"].std(ddof=1)) for conj in ["test", "cohorte", "isic2018"]]
        lineas.append(f"\\multicolumn{{2}}{{l}}{{All features (LogReg, {int(ref['parametros'].iloc[0])} params.)}} & " + " & ".join(celdas) + r" \\")
        lineas.append("\\midrule")
    lineas[-1] = "\\bottomrule"
    (sal / "tab_main.tex").write_text(
        "\\begin{tabular}{llccc}\n\\toprule\n & Head & Internal test & Reserved cohort & ISIC 2018 test \\\\\n"
        "\\midrule\n" + "\n".join(lineas) + "\n\\end{tabular}\n", encoding="utf-8")


def tabla_delta(r, sal):
    key = ["backbone", "qubits_d", "k", "semilla"]
    bloques = []
    for a, b, nombre in [("vqc", "mlp", "VQC $-$ matched MLP"), ("svm_cuantico", "svm_rbf", "SVM-QK $-$ SVM-RBF")]:
        A = r[r["cabeza"] == a].set_index(key)
        B = r[r["cabeza"] == b].set_index(key)
        dif = (A["auc_cohorte"] - B["auc_cohorte"]).groupby(["qubits_d", "k"])
        media, gana, n = dif.mean().unstack(), (dif.apply(lambda x: (x > 0).sum())).unstack(), dif.size().unstack()
        bloques.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{nombre}}}}} \\\\")
        for d in [4, 6, 8]:
            bloques.append(f"$d={d}$ & " + " & ".join(f"{media.loc[d, k]:+.3f} ({int(gana.loc[d, k])}/{int(n.loc[d, k])})".replace("-", "$-$").replace("+", "$+$") for k in K) + r" \\")
        bloques.append("\\midrule")
    bloques[-1] = "\\bottomrule"
    (sal / "tab_delta.tex").write_text(
        "\\begin{tabular}{lccccc}\n\\toprule\n & \\multicolumn{5}{c}{Training lesions per class} \\\\\n\\cmidrule(lr){2-6}\n"
        " & " + " & ".join(K_TEX[k] for k in K) + " \\\\\n\\midrule\n" + "\n".join(bloques) + "\n\\end{tabular}\n", encoding="utf-8")


def cifras(r, sal):
    """Macros con las cifras que se citan en el texto (se actualizan solas al regenerar)."""
    key = ["backbone", "qubits_d", "k", "semilla"]
    m = {}
    for a, b, tag in [("vqc", "mlp", "Vqc"), ("svm_cuantico", "svm_rbf", "Qk")]:
        A, B = r[r["cabeza"] == a].set_index(key), r[r["cabeza"] == b].set_index(key)
        for conj, ctag in [("cohorte", "Coh"), ("test", "Test"), ("isic2018", "Isic")]:
            dd = A[f"auc_{conj}"] - B[f"auc_{conj}"]
            m[f"d{tag}{ctag}"] = f"{dd.mean():.3f}"
            m[f"win{tag}{ctag}"] = f"{int((dd > 0).sum())}"
        m[f"n{tag}"] = str(len(dd))
        dc = (A["auc_cohorte"] - B["auc_cohorte"]).groupby(["qubits_d", "k"]).mean()
        for d, dtag in [(4, "Four"), (6, "Six"), (8, "Eight")]:
            m[f"d{tag}Full{dtag}"] = f"{dc.loc[(d, 'todas')]:.3f}"
        m[f"d{tag}WorstK"] = f"{dc.min():.3f}"
    sd = r[(r["k"] == "todas") & (r["qubits_d"] == 8)].groupby(["backbone", "cabeza"])["auc_cohorte"].std().groupby("cabeza").mean()
    m["sdVqcEight"], m["sdMlpEight"] = f"{sd['vqc']:.3f}", f"{sd['mlp']:.3f}"
    ref = r[(r["cabeza"] == "logreg_completa") & (r["k"] == "todas")].groupby("backbone")["auc_cohorte"].mean()
    m["refEff"], m["refVit"] = f"{ref['effb1']:.3f}", f"{ref['vitb32']:.3f}"
    m["nRuns"] = str(len(r))
    texto = "\n".join(f"\\newcommand{{\\{k}}}{{{v.replace('-', '$-$') if v.startswith('-') else v}}}" for k, v in m.items())
    (sal / "numbers.tex").write_text("% Generado por 04_tablas_latex.py; no editar a mano.\n" + texto + "\n", encoding="utf-8")


def tabla_robustez(base, variantes, sal):
    """Delta AUC VQC - MLP (cohorte) por variante, d y k. variantes: lista de (etiqueta, DataFrame) o
    (etiqueta, DataFrame, sufijo_macro). El sufijo fija el nombre de las macros (p. ej. robDE) aunque cambie
    el orden de las filas; sin sufijo se usa la letra que corresponde a la posicion."""
    etiquetas = ["A"] + [v[2] if len(v) > 2 and v[2] else "BCDEFGHIJ"[j] for j, v in enumerate(variantes)]
    variantes = [v[:2] for v in variantes]
    key = ["backbone", "qubits_d", "k", "semilla"]
    filas, macros = [], {}
    todas = [("Base: $L=2$, lr $0.01$", base[base["qubits_d"].isin([6, 8])])] + variantes
    for i, (nombre, r) in enumerate(todas):
        A, B = r[r["cabeza"] == "vqc"].set_index(key), r[r["cabeza"] == "mlp"].set_index(key)
        dif = (A["auc_cohorte"] - B["auc_cohorte"])
        g = dif.groupby(["qubits_d", "k"]).mean().unstack()
        pv = A.groupby("qubits_d")["parametros"].median()
        for d in [6, 8]:
            celdas = " & ".join(f"{g.loc[d, k]:+.3f}".replace("-", "$-$").replace("+", "$+$") for k in K)
            pref = nombre if d == 6 else ""
            filas.append(f"{pref} & {d} & {int(pv.loc[d])} & {celdas} & {int((dif.xs(d, level='qubits_d') > 0).sum())}/{int(dif.xs(d, level='qubits_d').size)} \\\\")
        filas.append("\\addlinespace[2pt]")
        etiqueta = etiquetas[i]
        macros[f"robD{etiqueta}"] = f"{dif.mean():.3f}"
        macros[f"robW{etiqueta}"] = f"{int((dif > 0).sum())}"
        macros[f"robN{etiqueta}"] = f"{int(dif.size)}"
        dtr = (A["auc_train"] - B["auc_train"]).mean()
        macros[f"robTrainD{etiqueta}"] = "0.000" if abs(dtr) < 0.0005 else f"{dtr:+.3f}"
        if i > 0:
            vb = base[base["qubits_d"].isin([6, 8])]
            for cab in ("vqc", "mlp"):
                a = r[r["cabeza"] == cab].groupby(["backbone", "qubits_d", "k"])["auc_cohorte"].mean()
                b = vb[vb["cabeza"] == cab].groupby(["backbone", "qubits_d", "k"])["auc_cohorte"].mean()
                g_ = (a - b).mean()
                macros[f"robGain{cab.capitalize()}{etiqueta}"] = "0.000" if abs(g_) < 0.0005 else f"{g_:+.3f}"
            a = r[r["cabeza"] == "vqc"].groupby(["backbone", "qubits_d", "k"])["auc_train"].mean()
            b = vb[vb["cabeza"] == "vqc"].groupby(["backbone", "qubits_d", "k"])["auc_train"].mean()
            g_ = (a - b).mean()
            macros[f"robTrainGainVqc{etiqueta}"] = "0.000" if abs(g_) < 0.0005 else f"{g_:+.3f}"
    filas[-1] = "\\bottomrule"
    (sal / "tab_robustness.tex").write_text(
        "\\begin{tabular}{llr" + "c" * len(K) + "c}\n\\toprule\n & & & \\multicolumn{" + str(len(K)) +
        "}{c}{Training lesions per class} & \\\\\n\\cmidrule(lr){4-" + str(3 + len(K)) + "}\n"
        "Variant & $d$ & VQC params. & " + " & ".join(K_TEX[k] for k in K) + " & VQC wins \\\\\n\\midrule\n" +
        "\n".join(filas) + "\n\\end{tabular}\n", encoding="utf-8")
    texto = "\n".join(f"\\newcommand{{\\{k}}}{{{v.replace('-', '$-$').replace('+', '$+$')}}}" for k, v in macros.items())
    (sal / "numbers_robustness.tex").write_text("% Generado por 04_tablas_latex.py\n" + texto + "\n", encoding="utf-8")


def cifras_bootstrap(ruta, sal):
    """Macros con los conteos de diferencias significativas (Holm) de comparaciones_pareadas.csv."""
    c = pd.read_csv(ruta)
    m = {}
    for comp, tag in [("vqc - mlp", "Vqc"), ("svm_cuantico - svm_rbf", "Qk"), ("vqc - logreg", "Lr")]:
        for conj, ctag in [("cohorte", "Coh"), ("test", "Test"), ("isic2018", "Isic")]:
            g = c[(c["comparacion"] == comp) & (c["conjunto"] == conj)]
            m[f"sig{tag}{ctag}"] = str(int(((g["p_holm"] < 0.05) & (g["ic95_sup"] < 0)).sum()))
            m[f"pos{tag}{ctag}"] = str(int(((g["p_holm"] < 0.05) & (g["ic95_inf"] > 0)).sum()))
            m[f"nb{tag}{ctag}"] = str(len(g))
    m["posAll"] = str(int(((c["p_holm"] < 0.05) & (c["ic95_inf"] > 0)).sum()))
    m["maxDelta"] = f"{c['delta_auc'].max():.3f}"
    m["nCompAll"] = str(len(c))
    texto = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items())
    (sal / "numbers_bootstrap.tex").write_text("% Generado por 04_tablas_latex.py\n" + texto + "\n", encoding="utf-8")


def cifras_codificacion(base, pi, sal):
    """Ablacion de codificacion: VQC con angulo pi*tanh(z) frente a la base (pi/2)*tanh(z),
    emparejados por run_id (mismo backbone, d, k y semilla)."""
    b = base[base["cabeza"] == "vqc"]
    q = pi[pi["cabeza"] == "vqc"]
    col = [c for c in b.columns if c.startswith("auc_")]
    m = b[["run_id"] + col].merge(q[["run_id"] + col], on="run_id", suffixes=("_half", "_pi"))
    mlp = base[base["cabeza"] == "mlp"].assign(run_id=lambda x: x["run_id"].str.replace("|mlp|", "|vqc|", regex=False))
    m = m.merge(mlp[["run_id", "auc_cohorte"]].rename(columns={"auc_cohorte": "auc_cohorte_mlp"}), on="run_id")
    d = lambda c: m[f"auc_{c}_pi"] - m[f"auc_{c}_half"]
    f3 = lambda x: "0.000" if abs(x) < 0.0005 else f"{x:.3f}".replace("-", "$-$")
    mac = {
        "encN": str(len(m)),
        "encWins": str(int((d("cohorte") > 0).sum())),
        "encDCoh": f3(d("cohorte").mean()), "encDTest": f3(d("test").mean()), "encDIsic": f3(d("isic2018").mean()),
        "encPiCoh": f"{m['auc_cohorte_pi'].mean():.3f}", "encHalfCoh": f"{m['auc_cohorte_half'].mean():.3f}",
        "encPiTrain": f"{m['auc_train_pi'].mean():.3f}", "encHalfTrain": f"{m['auc_train_half'].mean():.3f}",
        "encNearPi": f"{100 * (m['auc_cohorte_pi'] < 0.55).mean():.0f}",
        "encNearHalf": f"{100 * (m['auc_cohorte_half'] < 0.55).mean():.0f}",
        "encPiMlp": f3((m["auc_cohorte_pi"] - m["auc_cohorte_mlp"]).mean()),
    }
    texto = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items())
    (sal / "numbers_encoding.tex").write_text("% Generado por 04_tablas_latex.py\n" + texto + "\n", encoding="utf-8")


def cifras_shots(base, sh, sal):
    """Macros del analisis con ruido de medicion: kernel cuantico frente al RBF y frente a su version exacta."""
    def dif(r, q, c):
        a = r[r["cabeza"] == q].assign(key=lambda x: x["run_id"].str.replace(f"|{q}|", "|X|", regex=False))
        b = r[r["cabeza"] == c].assign(key=lambda x: x["run_id"].str.replace(f"|{c}|", "|X|", regex=False))
        j = a.merge(b, on="key", suffixes=("_q", "_c"))
        return j["auc_cohorte_q"] - j["auc_cohorte_c"]
    f3 = lambda x: ("0.000" if abs(x) < 0.0005 else f"{x:.3f}").replace("-", "$-$")
    b68 = base[base["qubits_d"].isin(sh["qubits_d"].unique())]
    d = dif(sh, "svm_cuantico", "svm_rbf")
    mac = {"qkShotsD": f3(d.mean()), "qkShotsW": str(int((d > 0).sum())), "qkShotsN": str(len(d)),
           "qkExactD": f3(dif(b68, "svm_cuantico", "svm_rbf").mean())}
    for cab, tag in [("svm_cuantico", "Qk"), ("vqc", "Vqc")]:
        k = sh[sh["cabeza"] == cab].merge(base[base["cabeza"] == cab][["run_id", "auc_cohorte", "auc_train"]],
                                          on="run_id", suffixes=("", "_ex"))
        mac[f"shotLoss{tag}"] = f3((k["auc_cohorte"] - k["auc_cohorte_ex"]).mean())
        mac[f"shotLossTrain{tag}"] = f3((k["auc_train"] - k["auc_train_ex"]).mean())
    texto = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items())
    (sal / "numbers_shots.tex").write_text("% Generado por 04_tablas_latex.py\n" + texto + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resumen", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--robustez", nargs="*", default=[],
                    help='pares "etiqueta=ruta_a_resumen_corridas.csv", en el orden en que deben aparecer')
    ap.add_argument("--comparaciones", help="comparaciones_pareadas.csv del paso 3")
    ap.add_argument("--codificacion", help="resumen_corridas.csv de la corrida con --escala-angulo 1.0")
    a = ap.parse_args()
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    r = pd.read_csv(a.resumen)
    r["k"] = r["k_por_clase"].astype(str)
    tabla_parametros(r, sal)
    tabla_principal(r, sal)
    tabla_delta(r, sal)
    cifras(r, sal)
    if a.comparaciones:
        cifras_bootstrap(a.comparaciones, sal)
    if a.codificacion:
        cifras_codificacion(r, pd.read_csv(a.codificacion), sal)
    if a.robustez:
        variantes = []
        for par in a.robustez:
            etiqueta, ruta = par.rsplit("=", 1)
            sufijo = ""
            if "@" in etiqueta:  # "E@Etiqueta=ruta": E fija el sufijo de las macros
                sufijo, etiqueta = etiqueta.split("@", 1)
            v = pd.read_csv(ruta)
            v["k"] = v["k_por_clase"].astype(str)
            variantes.append((etiqueta, v, sufijo))
            if "svm_cuantico" in set(v["cabeza"]) and v["hiper"].astype(str).str.contains("shots").any():
                cifras_shots(r, v, sal)
        tabla_robustez(r, variantes, sal)
    print(f"Tablas escritas en {sal}")


if __name__ == "__main__":
    main()
