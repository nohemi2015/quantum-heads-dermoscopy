#!/usr/bin/env python3
"""
05_analisis_revision.py
=======================
Analisis adicionales pedidos en la revision interna del manuscrito. No entrena nada:
usa las caracteristicas (paso 1) y las predicciones guardadas de la corrida principal (paso 2).

  A  AUC agregado por LESION (puntaje de la lesion = promedio de los logits de sus imagenes),
     con bootstrap por lesion y Holm, para las tres comparaciones.
  B  Bootstrap JERARQUICO: en cada replica se remuestrean las semillas (entrenamientos) y
     las lesiones de evaluacion, para incluir la variabilidad del entrenamiento en el IC.
  C  Sensibilidad excluyendo akiec de los conjuntos de evaluacion.
  D  Distribucion de los angulos de codificacion: fraccion de componentes en la zona donde,
     con theta = pi*tanh(z), las colas opuestas se parecen mas de 50 % (|tanh z| > 0.75).
  E  Kernel cuantico con shots: simetria, diagonal y autovalores negativos de la matriz
     de entrenamiento (d = 6, 8; todas las lesiones).
  F  Comprobaciones de datos: image_id repetidos entre conjuntos, lesiones compartidas,
     diagnosticos por conjunto (tabla de correspondencia a benigno/maligno).
  G  Lesiones e imagenes de entrenamiento de cada corrida (k, semilla), para reproducibilidad.

Uso (Windows PowerShell, una sola linea):
  python 05_analisis_revision.py --features C:\\proyecto_cuantico\\features --principal C:\\proyecto_cuantico\\cabezas
      --shots C:\\proyecto_cuantico\\robustez_shots --salida C:\\proyecto_cuantico\\revision

Tarda unos minutos (el bootstrap usa B = 2000 como el paso 3).
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent


def importar(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, AQUI / archivo)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [archivo]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


an = importar("analizar", "03_analizar.py")      # auc_ponderada, pesos_lesion, holm
en = importar("entrenar", "02_entrenar_cabezas.py")  # submuestra, kernel_cuantico

CONJ = ["cohorte", "test", "isic2018"]
PARES = [("vqc", "mlp"), ("svm_cuantico", "svm_rbf"), ("vqc", "logreg")]


def cargar_preds(carpeta, run_id):
    return np.load(Path(carpeta) / "preds" / f"{run_id.replace('|', '_')}.npy").astype(np.float64)


def resumen_familia(df, col_p="p"):
    df = df.copy()
    df["p_holm"] = df.groupby(["comparacion", "conjunto"])[col_p].transform(an.holm)
    return df


def contar(df):
    g = df.groupby(["comparacion", "conjunto"])
    return pd.DataFrame({
        "n": g.size(),
        "delta_medio": g["delta"].mean(),
        "sig_a_favor_clasico": g.apply(lambda x: int(((x["ic_sup"] < 0) & (x["p_holm"] < 0.05)).sum())),
        "sig_a_favor_cuantico": g.apply(lambda x: int(((x["ic_inf"] > 0) & (x["p_holm"] < 0.05)).sum())),
        "gana_cuantico_punto": g.apply(lambda x: int((x["delta"] > 0).sum())),
    }).reset_index()


def comparar(r, carpeta, ids, B, rng, modo):
    """modo: 'lesion' (AUC por lesion), 'jerarquico' (semillas + lesiones), 'sin_akiec'."""
    y_all = ids["label"].to_numpy()
    filas = []
    prep = {}
    for c in CONJ:
        m = (ids["conjunto"] == c).to_numpy()
        if modo == "sin_akiec":
            m = m & (ids["dx"] != "akiec").to_numpy()
        les = ids.loc[m, "lesion_id"].to_numpy()
        if modo == "lesion":
            codigos, inv = np.unique(les, return_inverse=True)
            y_les = pd.Series(y_all[m]).groupby(inv).max().to_numpy()
            W = np.stack([np.bincount(rng.integers(0, len(codigos), len(codigos)), minlength=len(codigos)) for _ in range(B)])
            prep[c] = (m, inv, y_les, W)
        else:
            prep[c] = (m, None, y_all[m], an.pesos_lesion(les, B, rng))
    combos = r[["backbone", "qubits_d", "k"]].drop_duplicates()
    combos = combos[combos["qubits_d"] > 0]
    for _, cb in combos.iterrows():
        sub = r[(r["backbone"] == cb.backbone) & (r["qubits_d"] == cb.qubits_d) & (r["k"] == cb.k)]
        for a, b in PARES:
            A = sub[sub["cabeza"] == a].set_index("semilla")["run_id"]
            Bm = sub[sub["cabeza"] == b].set_index("semilla")["run_id"]
            sem = sorted(set(A.index) & set(Bm.index))
            if not sem:
                continue
            for c in CONJ:
                m, inv, yc, W = prep[c]
                uno = np.ones((1, len(yc)), dtype=np.int32)
                obs, boot = [], []
                for s in sem:
                    pa, pb = cargar_preds(carpeta, A[s])[m], cargar_preds(carpeta, Bm[s])[m]
                    if modo == "lesion":
                        pa = pd.Series(pa).groupby(inv).mean().to_numpy()
                        pb = pd.Series(pb).groupby(inv).mean().to_numpy()
                    obs.append(an.auc_ponderada(pa, yc, uno)[0] - an.auc_ponderada(pb, yc, uno)[0])
                    boot.append(an.auc_ponderada(pa, yc, W) - an.auc_ponderada(pb, yc, W))
                boot = np.array(boot)  # semillas x B
                if modo == "jerarquico":
                    elige = rng.integers(0, len(sem), size=(B, len(sem)))
                    dist = boot[elige, np.arange(B)[:, None]].mean(1)
                else:
                    dist = boot.mean(0)
                d = float(np.mean(obs))
                p = max(min(1.0, 2 * min((dist <= 0).mean(), (dist >= 0).mean())), 1 / B)
                filas.append({"backbone": cb.backbone, "qubits_d": cb.qubits_d, "k": cb.k,
                              "comparacion": f"{a} - {b}", "conjunto": c, "delta": d,
                              "ic_inf": np.percentile(dist, 2.5), "ic_sup": np.percentile(dist, 97.5), "p": p,
                              "sd_semillas": float(np.std(obs, ddof=1)) if len(obs) > 1 else np.nan})
    return resumen_familia(pd.DataFrame(filas))


def angulos(man, feats, qubits):
    filas = []
    tr = man.index[man["conjunto"] == "train"].to_numpy()
    for bb, X in feats.items():
        for d in qubits:
            sc1 = en.StandardScaler().fit(X[tr])
            pca = en.PCA(n_components=d, random_state=0).fit(sc1.transform(X[tr]))
            sc2 = en.StandardScaler().fit(pca.transform(sc1.transform(X[tr])))
            for c in ["train"] + CONJ:
                idx = man.index[man["conjunto"] == c].to_numpy()
                t = np.tanh(sc2.transform(pca.transform(sc1.transform(X[idx]))))
                a = np.abs(t)
                filas.append({"backbone": bb, "d": d, "conjunto": c,
                              "frac_componentes_|tanh|>0.75": float((a > 0.75).mean()),
                              "frac_imagenes_alguna_componente": float((a > 0.75).any(1).mean()),
                              "mediana_|tanh|": float(np.median(a)), "p90_|tanh|": float(np.quantile(a, 0.9)),
                              "F_media_pi": float(np.mean(np.cos(np.pi * t) ** 2)),
                              "F_media_pi_medios": float(np.mean(np.cos(np.pi / 2 * t) ** 2))})
    return pd.DataFrame(filas)


def kernel_shots(man, feats, carpeta_shots):
    rs = pd.read_csv(Path(carpeta_shots) / "resumen_corridas.csv")
    rs = rs[(rs["cabeza"] == "svm_cuantico") & (rs["k_por_clase"].astype(str) == "todas") & (rs["semilla"] == 0)]
    tr = man.index[man["conjunto"] == "train"].to_numpy()
    filas = []
    for _, f in rs.iterrows():
        d = int(f["qubits_d"])
        escala = float(str(f["hiper"]).split("param=")[1].split(",")[0])
        X = feats[f["backbone"]]
        sc1 = en.StandardScaler().fit(X[tr])
        pca = en.PCA(n_components=d, random_state=0).fit(sc1.transform(X[tr]))
        sc2 = en.StandardScaler().fit(pca.transform(sc1.transform(X[tr])))
        Z = sc2.transform(pca.transform(sc1.transform(X[tr])))
        K = en.kernel_cuantico(Z, Z, escala)
        Ks = en.kernel_cuantico(Z, Z, escala, 1024, np.random.default_rng(1000 * 0 + d), simetrico=True)
        ev, evs = np.linalg.eigvalsh(K), np.linalg.eigvalsh(Ks)
        filas.append({"backbone": f["backbone"], "d": d, "escala_s": escala,
                      "simetrica": bool(np.allclose(Ks, Ks.T)), "diagonal_1": bool(np.allclose(np.diag(Ks), 1)),
                      "min_autovalor_exacto": float(ev.min()), "min_autovalor_shots": float(evs.min()),
                      "n_autovalores_negativos_shots": int((evs < -1e-10).sum()),
                      "masa_negativa_relativa": float(-evs[evs < 0].sum() / np.abs(evs).sum()),
                      "error_frobenius_relativo": float(np.linalg.norm(Ks - K) / np.linalg.norm(K))})
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--principal", required=True, help="carpeta de la corrida principal (preds/, eval_ids.csv)")
    ap.add_argument("--shots", help="carpeta de la corrida con --shots 1024 (para el analisis E)")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)

    man = pd.read_csv(Path(a.features) / "manifiesto.csv")
    r = pd.read_csv(Path(a.principal) / "resumen_corridas.csv")
    r["k"] = r["k_por_clase"].astype(str)
    ids = pd.read_csv(Path(a.principal) / "eval_ids.csv")
    feats = {}
    for bb in ["effb1", "vitb32"]:
        z = np.load(Path(a.features) / f"feats_{bb}.npz", allow_pickle=True)
        assert (z["image_id"] == man["image_id"].to_numpy()).all()
        feats[bb] = z["X"].astype(np.float64)

    # Versiones del entorno (para la seccion de reproducibilidad)
    import platform, sklearn, torch
    vers = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__}
    try:
        import pennylane
        vers["pennylane"] = pennylane.__version__
    except ImportError:
        pass
    json.dump(vers, open(sal / "versiones.json", "w"), indent=2)
    print("Versiones:", vers)

    # F: comprobaciones de datos
    print("F  comprobaciones de datos")
    dup = man[man.duplicated("image_id", keep=False)]
    les = man.groupby("lesion_id")["conjunto"].nunique()
    compartidas = man[man["lesion_id"].isin(les[les > 1].index)]
    dx = pd.crosstab(man["dx"], man["conjunto"])
    dx["etiqueta"] = man.groupby("dx")["label"].first()
    dx.to_csv(sal / "F_diagnosticos_por_conjunto.csv")
    json.dump({"image_id_repetidos": int(len(dup)), "lesiones_en_mas_de_un_conjunto": int(compartidas["lesion_id"].nunique()),
               "detalle_lesiones_compartidas": compartidas.groupby("lesion_id")["conjunto"].apply(list).astype(str).to_dict()},
              open(sal / "F_comprobaciones.json", "w"), indent=2)
    print(dx)

    # G: lesiones de entrenamiento por corrida
    print("G  lesiones de entrenamiento por corrida")
    tr = man[man["conjunto"] == "train"]
    filas = []
    for k in [25, 50, 100, 200, 0]:
        for s in range(5):
            idx = en.submuestra(tr, k, s)
            for i in idx:
                filas.append({"k_por_clase": k if k else "todas", "semilla": s, "image_id": man.at[i, "image_id"],
                              "lesion_id": man.at[i, "lesion_id"], "label": int(man.at[i, "label"])})
    pd.DataFrame(filas).to_csv(sal / "G_entrenamiento_por_corrida.csv", index=False)

    # D: angulos
    print("D  distribucion de angulos")
    angulos(man, feats, [4, 6, 8]).to_csv(sal / "D_angulos.csv", index=False, float_format="%.4f")

    # E: kernel con shots
    if a.shots:
        print("E  kernel cuantico con shots")
        kernel_shots(man, feats, a.shots).to_csv(sal / "E_kernel_shots.csv", index=False, float_format="%.6g")

    # A, B, C: comparaciones
    for modo, nombre in [("lesion", "A_por_lesion"), ("jerarquico", "B_jerarquico"), ("sin_akiec", "C_sin_akiec")]:
        print(f"{nombre}: bootstrap B = {a.B} ...")
        c = comparar(r, a.principal, ids, a.B, rng, modo)
        c.to_csv(sal / f"{nombre}.csv", index=False, float_format="%.6g")
        cont = contar(c)
        cont.to_csv(sal / f"{nombre}_resumen.csv", index=False, float_format="%.4f")
        print(cont.to_string(index=False))
    print(f"\nListo. Resultados en {sal}")


if __name__ == "__main__":
    main()
