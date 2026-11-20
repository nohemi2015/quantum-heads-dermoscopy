#!/usr/bin/env python3
"""
02_entrenar_cabezas.py
======================
Paso 2 del estudio cuantico: entrenar cabezas cuanticas y clasicas sobre las MISMAS
caracteristicas congeladas (salida del paso 1) y guardar sus predicciones.

Pregunta: con la misma entrada de d dimensiones, ¿una cabeza cuantica discrimina
maligno/benigno mejor que una clasica de capacidad equivalente? ¿Cambia la respuesta
con pocos datos o fuera de distribucion (cohorte reservada, ISIC 2018)?

Cabezas (todas reciben los mismos d componentes principales, d = numero de qubits):
  logreg         regresion logistica sobre d entradas                 (d + 1 parametros)
  mlp            perceptron d -> h -> 1, con h elegido para igualar
                 los parametros del circuito variacional               (~ los del vqc)
  vqc            circuito variacional: AngleEmbedding(RY, angulo = (pi/2)*tanh(z))
                 + StronglyEntanglingLayers,
                 se miden <Z> de todos los qubits y una capa lineal da el logit
                                                                        (3*d*capas + d + 1)
  svm_rbf        SVM con kernel gaussiano clasico (C y gamma elegidos en validacion)
  svm_cuantico   SVM con kernel cuantico de fidelidad, mapa tipo IQP (Havlicek 2019),
                 dos repeticiones (C y escala elegidos en validacion; misma rejilla de 16)
                 En los SVM, "parametros" = vectores de soporte + 1 (no es comparable 1:1).
  logreg_completa  referencia: regresion logistica sobre TODAS las caracteristicas

Reglas para que la comparacion sea justa:
  - Estandarizacion y PCA se ajustan SOLO con las imagenes de entrenamiento de cada corrida.
  - Validacion (val) se usa para detener el entrenamiento, elegir hiperparametros y fijar
    el umbral de Youden. Test, cohorte e ISIC 2018 nunca se usan para decidir nada.
  - vqc, mlp y logreg usan el mismo entrenador (Adam, lote completo, lr 0.01, parada
    temprana por perdida de validacion). Los dos SVM usan rejillas del mismo tamano.
  - Pocos datos: se eligen k lesiones por clase (no imagenes) con cada semilla; todas las
    imagenes de esas lesiones entran juntas. k = 25, 50, 100, 200 y "todas".

Opciones de robustez: --shots N simula el ruido de medicion con N mediciones por circuito
(vqc y svm_cuantico); --resubir repite la codificacion antes de cada capa del vqc.

Simulacion: los circuitos se simulan en PyTorch (vector de estado exacto, sin ruido) y los
gradientes se obtienen por retropropagacion, que es exacta y equivalente a parameter-shift
en simulacion sin ruido. Antes de entrenar, el script compara el simulador contra PennyLane
(si esta instalado) y contra una construccion matricial independiente del kernel.

Salidas en --salida:
  resumen_corridas.csv   una fila por corrida: configuracion, parametros, AUC, sens/esp...
  preds/<run_id>.npy     puntaje por imagen de val+test+cohorte+isic2018 (orden de eval_ids.csv)
  eval_ids.csv           image_id, lesion_id, conjunto y etiqueta de esas imagenes
  verificacion.json      resultado de las pruebas del simulador

El script se puede interrumpir y volver a correr: salta las corridas ya terminadas.

Uso:
  python 02_entrenar_cabezas.py --features C:\\proyecto_cuantico\\features --salida C:\\proyecto_cuantico\\cabezas
  Prueba rapida (minutos): agrega --rapido
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

EVAL = ["val", "test", "cohorte", "isic2018"]
COLUMNAS = (["run_id", "backbone", "cabeza", "qubits_d", "capas", "k_por_clase", "semilla", "n_train_img",
             "n_train_lesiones", "parametros", "pasos", "hiper", "n_vectores_soporte", "auc_train", "umbral_val"]
            + [f"{m}_{c}" for c in EVAL for m in ("auc", "sens", "esp")] + ["segundos"])


# ----------------------------------------------------------------------------------------
# Simulador de vector de estado (qubit 0 = bit mas significativo, igual que PennyLane)
# ----------------------------------------------------------------------------------------
def aplicar_1q(psi, U, w, n):
    """Aplica una compuerta de 1 qubit. psi: (B, 2^n). U: (2,2) comun o (B,2,2) por muestra."""
    B = psi.shape[0]
    p = psi.reshape(B, 2 ** w, 2, 2 ** (n - w - 1))
    if U.dim() == 2:
        p = torch.einsum("ji,baic->bajc", U, p)
    else:
        p = torch.einsum("bji,baic->bajc", U, p)
    return p.reshape(B, 2 ** n)


def perm_cnot(c, t, n):
    idx = np.arange(2 ** n)
    bit_c = (idx >> (n - 1 - c)) & 1
    return torch.as_tensor(np.where(bit_c == 1, idx ^ (1 << (n - 1 - t)), idx))


def signos_z(n):
    idx = np.arange(2 ** n)[:, None]
    return torch.as_tensor(1 - 2 * ((idx >> (n - 1 - np.arange(n))) & 1), dtype=torch.float32)


def rot(phi, theta, omega):
    """Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi), definicion de PennyLane."""
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    e = lambda a: torch.exp(1j * a)
    return torch.stack([torch.stack([e(-(phi + omega) / 2) * c, -e((phi - omega) / 2) * s]),
                        torch.stack([e(-(phi - omega) / 2) * s, e((phi + omega) / 2) * c])])


class CircuitoVariacional(torch.nn.Module):
    """AngleEmbedding(rotacion Y) + StronglyEntanglingLayers + <Z_i> de cada qubit + lineal."""

    def __init__(self, n, capas, gen, shots=0, resubir=False):
        super().__init__()
        self.n, self.capas, self.shots, self.resubir = n, capas, shots, resubir
        self.pesos = torch.nn.Parameter(torch.rand(capas, n, 3, generator=gen) * 2 * np.pi)
        self.salida = torch.nn.Linear(n, 1)
        rangos = [(l % (n - 1)) + 1 if n > 1 else 0 for l in range(capas)]
        self.register_buffer("perms", torch.stack([
            torch.stack([perm_cnot(i, (i + r) % n, n) for i in range(n)]) for r in rangos]))
        self.register_buffer("signos", signos_z(n))

    def esperanzas(self, x):
        B, n = x.shape[0], self.n
        psi = torch.zeros(B, 2 ** n, dtype=torch.complex64, device=x.device)
        psi[:, 0] = 1
        c, s = torch.cos(x / 2), torch.sin(x / 2)
        ry = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex64)
        for l in range(self.capas):
            # Codificacion: una vez al inicio, o antes de cada capa si --resubir (data re-uploading)
            if l == 0 or self.resubir:
                for w in range(n):
                    psi = aplicar_1q(psi, ry[:, w], w, n)
            for w in range(n):
                psi = aplicar_1q(psi, rot(*self.pesos[l, w]).to(torch.complex64), w, n)
            if n > 1:
                for w in range(n):
                    psi = psi[:, self.perms[l, w]]
        probs = psi.abs() ** 2
        exacto = probs @ self.signos
        if self.shots <= 0:
            return exacto
        # Ruido de medicion: se muestrean `shots` cadenas de bits completas por imagen y se
        # estiman los <Z_i> con ellas. El gradiente se toma del valor exacto (estimador
        # "straight-through"), lo que es optimista para el circuito: en hardware el gradiente
        # tambien tendria ruido de muestreo.
        p = (probs / probs.sum(-1, keepdim=True)).float().clamp_min(0)
        conteos = torch.distributions.Multinomial(total_count=self.shots, probs=p.detach()).sample()
        estimado = conteos @ self.signos / self.shots
        return exacto + (estimado - exacto).detach()

    def forward(self, x):
        return self.salida(self.esperanzas(x)).squeeze(-1)


def estados_iqp(x, escala, reps=2):
    """Estado del mapa IQP: (U_phi(x) H^n)^reps |0>, con fase
    phi(b) = escala * (sum_i x_i z_i + sum_{i<j} x_i x_j z_i z_j), z = +-1."""
    x = torch.as_tensor(x, dtype=torch.float64)
    n = x.shape[1]
    z = 1.0 - 2.0 * ((torch.arange(2 ** n)[:, None] >> (n - 1 - torch.arange(n))) & 1).double()
    pares = list(itertools.combinations(range(n), 2))
    fase = x @ z.T
    if pares:
        i, j = map(list, zip(*pares))
        fase = fase + (x[:, i] * x[:, j]) @ (z[:, i] * z[:, j]).T
    fase = escala * fase
    H = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64) / np.sqrt(2)
    Hn = H
    for _ in range(n - 1):
        Hn = torch.kron(Hn, H)
    Hn = Hn.to(torch.complex128)
    psi = torch.full((x.shape[0], 2 ** n), 2 ** (-n / 2), dtype=torch.complex128)
    for r in range(reps):
        psi = psi * torch.exp(1j * fase)
        if r < reps - 1:
            psi = psi @ Hn.T
    return psi


def kernel_cuantico(xa, xb, escala, shots=0, rng=None, simetrico=False):
    """Kernel de fidelidad. Con shots > 0, cada entrada se estima con `shots` repeticiones del
    circuito de compute-uncompute: k_hat = Binomial(shots, k) / shots."""
    Sa, Sb = estados_iqp(xa, escala), estados_iqp(xb, escala)
    K = ((Sa.conj() @ Sb.T).abs() ** 2).numpy()
    if shots <= 0:
        return K
    K = rng.binomial(shots, np.clip(K, 0.0, 1.0)) / shots
    if simetrico:  # matriz de entrenamiento: una sola estimacion por par y diagonal exacta (= 1)
        K = np.triu(K, 1)
        K = K + K.T + np.eye(len(K))
    return K


# ----------------------------------------------------------------------------------------
# Verificacion del simulador
# ----------------------------------------------------------------------------------------
def verificar():
    res = {}
    gen = torch.Generator().manual_seed(0)
    try:
        import pennylane as qml
        for n, capas in [(4, 2), (6, 3)]:
            m = CircuitoVariacional(n, capas, gen)
            x = torch.rand(5, n, generator=gen) * 2 * np.pi - np.pi
            dev = qml.device("default.qubit", wires=n)

            @qml.qnode(dev)
            def c(xi, w):
                qml.AngleEmbedding(xi, wires=range(n), rotation="Y")
                qml.StronglyEntanglingLayers(w, wires=range(n))
                return [qml.expval(qml.PauliZ(i)) for i in range(n)]

            w = m.pesos.detach().double().numpy()
            ref = np.array([[float(v) for v in c(xi.double().numpy(), w)] for xi in x])
            res[f"vqc_vs_pennylane_n{n}_L{capas}"] = float(np.abs(ref - m.esperanzas(x).detach().numpy()).max())
    except ImportError:
        res["vqc_vs_pennylane"] = "PennyLane no instalado: verificacion omitida"

    # Kernel: comparar contra construccion matricial explicita (compuertas como matrices densas).
    n = 3
    x = np.random.default_rng(1).normal(size=(4, n))
    Hs = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    Hn = Hs
    for _ in range(n - 1):
        Hn = np.kron(Hn, Hs)
    Z = np.diag([1, -1])
    I = np.eye(2)

    def op(ops):
        M = np.array([[1.0]])
        for o in ops:
            M = np.kron(M, o)
        return M

    def U(xi):
        G = sum(xi[k] * op([Z if q == k else I for q in range(n)]) for k in range(n))
        G = G + sum(xi[i] * xi[j] * op([Z if q in (i, j) else I for q in range(n)])
                    for i, j in itertools.combinations(range(n), 2))
        return np.diag(np.exp(1j * 0.7 * np.diag(G)))

    e0 = np.zeros(2 ** n)
    e0[0] = 1
    S = np.array([U(xi) @ Hn @ U(xi) @ Hn @ e0 for xi in x])
    K_ref = np.abs(S.conj() @ S.T) ** 2
    K = kernel_cuantico(x, x, 0.7)
    res["kernel_vs_matrices"] = float(np.abs(K - K_ref).max())
    res["kernel_diagonal_uno"] = float(np.abs(np.diag(K) - 1).max())
    return res


# ----------------------------------------------------------------------------------------
# Entrenamiento y evaluacion
# ----------------------------------------------------------------------------------------
class MLP(torch.nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.red = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.Tanh(), torch.nn.Linear(h, 1))

    def forward(self, x):
        return self.red(x).squeeze(-1)


class Lineal(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.capa = torch.nn.Linear(d, 1)

    def forward(self, x):
        return self.capa(x).squeeze(-1)


def contar(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def entrenar_torch(modelo, Xtr, ytr, Xva, yva, a, dispositivo):
    """Adam con lote completo; se guarda el estado con menor perdida de validacion."""
    modelo = modelo.to(dispositivo)
    Xtr, ytr, Xva, yva = (torch.as_tensor(v, dtype=torch.float32, device=dispositivo) for v in (Xtr, ytr, Xva, yva))
    opt = torch.optim.Adam(modelo.parameters(), lr=a.lr)
    perdida = torch.nn.BCEWithLogitsLoss()
    mejor, estado, paso_mejor = np.inf, None, 0
    for paso in range(a.pasos):
        modelo.train()
        opt.zero_grad()
        perdida(modelo(Xtr), ytr).backward()
        opt.step()
        modelo.eval()
        with torch.no_grad():
            pv = perdida(modelo(Xva), yva).item()
        if pv < mejor - 1e-6:
            mejor, paso_mejor = pv, paso
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        elif paso - paso_mejor >= a.paciencia:
            break
    modelo.load_state_dict(estado)
    modelo.eval()
    return modelo, paso_mejor + 1


def predecir_torch(modelo, X, dispositivo, lote=4096):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), lote):
            out.append(modelo(torch.as_tensor(X[i:i + lote], dtype=torch.float32, device=dispositivo)).cpu().numpy())
    return np.concatenate(out)


def h_igual(d, objetivo):
    """Ancho oculto del MLP cuyo numero de parametros h*(d+2)+1 queda mas cerca del objetivo."""
    return max(1, int(round((objetivo - 1) / (d + 2))))


def elegir_svm(Ktr_fn, Kva_fn, ytr, yva, rejilla_kernel):
    """Elige (C, parametro del kernel) por AUC en validacion. Devuelve el mejor ajuste."""
    mejor = None
    for pk in rejilla_kernel:
        Ktr, Kva = Ktr_fn(pk), Kva_fn(pk)
        for C in (0.1, 1.0, 10.0, 100.0):
            svm = SVC(C=C, kernel="precomputed").fit(Ktr, ytr)
            auc = roc_auc_score(yva, svm.decision_function(Kva))
            if mejor is None or auc > mejor[0]:
                mejor = (auc, C, pk, svm)
    return mejor


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr - fpr)]


def metricas(y_por, s_por, umbral):
    fila = {}
    for c, (y, s) in ((c, (y_por[c], s_por[c])) for c in y_por):
        fila[f"auc_{c}"] = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
        p = s >= umbral
        fila[f"sens_{c}"] = p[y == 1].mean()
        fila[f"esp_{c}"] = (~p[y == 0]).mean()
    return fila


def submuestra(man_tr, k, semilla):
    """k lesiones por clase (benigna/maligna), con todas sus imagenes. k=0 significa todas."""
    if k == 0:
        return man_tr.index.to_numpy()
    rng = np.random.default_rng(semilla)
    les = man_tr.groupby("lesion_id")["label"].first()
    elegidas = np.concatenate([rng.choice(les[les == c].index.to_numpy(), size=min(k, int((les == c).sum())), replace=False)
                               for c in (0, 1)])
    return man_tr.index[man_tr["lesion_id"].isin(elegidas)].to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--backbones", default="effb1,vitb32")
    ap.add_argument("--qubits", default="4,6,8")
    ap.add_argument("--capas", type=int, default=2)
    ap.add_argument("--k", default="25,50,100,200,0", help="lesiones por clase; 0 = todas")
    ap.add_argument("--semillas", default="0,1,2,3,4")
    ap.add_argument("--cabezas", default="logreg_completa,logreg,mlp,vqc,svm_rbf,svm_cuantico")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--escala-angulo", type=float, default=0.5,
                    help="angulo = escala * pi * tanh(z). 0.5 (por defecto) = pi/2; 1.0 = pi (solo para el experimento de codificacion)")
    ap.add_argument("--shots", type=int, default=0,
                    help="mediciones por circuito (0 = simulacion exacta). Afecta a vqc y svm_cuantico")
    ap.add_argument("--resubir", action="store_true",
                    help="data re-uploading: repetir la codificacion antes de cada capa del vqc")
    ap.add_argument("--pasos", type=int, default=3000)
    ap.add_argument("--paciencia", type=int, default=200)
    ap.add_argument("--dispositivo", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rapido", action="store_true", help="prueba corta: 1 semilla, k=50 y todas, 4 qubits")
    a = ap.parse_args()
    if a.rapido:
        a.semillas, a.k, a.qubits, a.pasos = "0", "50,0", "4", 600
    sal = Path(a.salida)
    (sal / "preds").mkdir(parents=True, exist_ok=True)
    disp = torch.device(a.dispositivo)
    torch.backends.cuda.matmul.allow_tf32 = False

    ver = verificar()
    (sal / "verificacion.json").write_text(json.dumps(ver, indent=2))
    print("Verificacion del simulador (diferencias maximas, deben ser < 1e-4):")
    for k, v in ver.items():
        print(f"  {k}: {v}")
    malos = [k for k, v in ver.items() if isinstance(v, float) and v > 1e-4]
    if malos:
        raise SystemExit(f"El simulador no coincide con la referencia: {malos}")

    man = pd.read_csv(Path(a.features) / "manifiesto.csv")
    ev = man[man["conjunto"].isin(EVAL)]
    ev[["image_id", "lesion_id", "conjunto", "dx", "label"]].to_csv(sal / "eval_ids.csv", index=False)
    archivo = sal / "resumen_corridas.csv"
    hechas = set(pd.read_csv(archivo)["run_id"]) if archivo.exists() else set()
    cabezas = a.cabezas.split(",")
    qubits = [int(q) for q in a.qubits.split(",")]

    for bb in a.backbones.split(","):
        z = np.load(Path(a.features) / f"feats_{bb}.npz", allow_pickle=True)
        assert (z["image_id"] == man["image_id"].to_numpy()).all(), "El orden de las caracteristicas no coincide con el manifiesto"
        X = z["X"].astype(np.float64)
        tr_all = man[man["conjunto"] == "train"]
        idx_va = man.index[man["conjunto"] == "val"].to_numpy()
        idx_ev = ev.index.to_numpy()
        y = man["label"].to_numpy()
        conj_ev = ev["conjunto"].to_numpy()

        for k, semilla in itertools.product([int(v) for v in a.k.split(",")], [int(v) for v in a.semillas.split(",")]):
            idx_tr = submuestra(tr_all, k, semilla)
            ytr, yva = y[idx_tr], y[idx_va]
            y_por = {c: y[idx_ev][conj_ev == c] for c in EVAL}

            def guardar(cabeza, d, params, s_tr, s_ev, extra, t0):
                run_id = f"{bb}|{cabeza}|d{d}|k{k}|s{semilla}"
                np.save(sal / "preds" / f"{run_id.replace('|', '_')}.npy", s_ev.astype(np.float32))
                s_por = {c: s_ev[conj_ev == c] for c in EVAL}
                fila = {"run_id": run_id, "backbone": bb, "cabeza": cabeza, "qubits_d": d, "capas": a.capas if cabeza == "vqc" else 0,
                        "k_por_clase": k if k else "todas", "semilla": semilla, "n_train_img": len(idx_tr),
                        "n_train_lesiones": tr_all.loc[idx_tr, "lesion_id"].nunique(), "parametros": params,
                        "auc_train": roc_auc_score(ytr, s_tr), "umbral_val": youden(y_por["val"], s_por["val"]),
                        "segundos": round(time.time() - t0, 1), **extra}
                fila.update(metricas(y_por, s_por, fila["umbral_val"]))
                pd.DataFrame([fila]).reindex(columns=COLUMNAS).to_csv(archivo, mode="a", header=not archivo.exists(), index=False)
                print(f"  {run_id:42s} params={params:6d}  AUC val={fila['auc_val']:.3f} test={fila['auc_test']:.3f} "
                      f"cohorte={fila['auc_cohorte']:.3f} isic={fila['auc_isic2018']:.3f}  ({fila['segundos']} s)", flush=True)

            # Referencia: todas las caracteristicas
            if "logreg_completa" in cabezas and f"{bb}|logreg_completa|d0|k{k}|s{semilla}" not in hechas:
                t0 = time.time()
                sc = StandardScaler().fit(X[idx_tr])
                mejor = max(((roc_auc_score(yva, m.decision_function(sc.transform(X[idx_va]))), C, m) for C in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
                             for m in [LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[idx_tr]), ytr)]), key=lambda t: t[0])
                m = mejor[2]
                guardar("logreg_completa", 0, X.shape[1] + 1, m.decision_function(sc.transform(X[idx_tr])),
                        m.decision_function(sc.transform(X[idx_ev])), {"hiper": f"C={mejor[1]}"}, t0)

            for d in qubits:
                # Entrada comun: estandarizar -> PCA(d) -> estandarizar componentes (todo ajustado con train)
                sc1 = StandardScaler().fit(X[idx_tr])
                pca = PCA(n_components=d, random_state=0).fit(sc1.transform(X[idx_tr]))
                sc2 = StandardScaler().fit(pca.transform(sc1.transform(X[idx_tr])))
                Z = lambda idx: sc2.transform(pca.transform(sc1.transform(X[idx])))
                Ztr, Zva, Zev = Z(idx_tr), Z(idx_va), Z(idx_ev)
                # Codificacion: angulo = (pi/2) * tanh(z), acotado a (-pi/2, pi/2).
                # NO usar pi * tanh(z): RY(+pi)|0> y RY(-pi)|0> son el mismo estado fisico
                # (difieren solo en una fase global), asi que las dos colas de cada
                # componente se pliegan sobre el mismo punto y se pierde la informacion
                # mas discriminativa. Con pi/2 la codificacion es inyectiva y las colas
                # quedan en extremos opuestos de la esfera de Bloch.
                angulo = lambda M: a.escala_angulo * np.pi * np.tanh(M)
                gen = lambda: torch.Generator().manual_seed(1000 * semilla + d)
                n_vqc = 3 * d * a.capas + d + 1

                for cabeza in [c for c in ("logreg", "mlp", "vqc") if c in cabezas]:
                    if f"{bb}|{cabeza}|d{d}|k{k}|s{semilla}" in hechas:
                        continue
                    t0 = time.time()
                    torch.manual_seed(1000 * semilla + d)
                    if cabeza == "logreg":
                        m, A = Lineal(d), (lambda M: M)
                    elif cabeza == "mlp":
                        m, A = MLP(d, h_igual(d, n_vqc)), (lambda M: M)
                    else:
                        m, A = CircuitoVariacional(d, a.capas, gen(), a.shots, a.resubir), angulo
                    m, pasos = entrenar_torch(m, A(Ztr), ytr, A(Zva), yva, a, disp)
                    hiper = ";".join(t for t in ([f"shots={a.shots}"] if a.shots else []) +
                                     (["reupload"] if a.resubir else []) +
                                     ([f"angle={a.escala_angulo}pi"] if a.escala_angulo != 0.5 else [])) if cabeza == "vqc" else ""
                    guardar(cabeza, d, contar(m), predecir_torch(m, A(Ztr), disp), predecir_torch(m, A(Zev), disp),
                            {"pasos": pasos, "hiper": hiper}, t0)

                for cabeza in [c for c in ("svm_rbf", "svm_cuantico") if c in cabezas]:
                    if f"{bb}|{cabeza}|d{d}|k{k}|s{semilla}" in hechas:
                        continue
                    t0 = time.time()
                    if cabeza == "svm_rbf":
                        kern = lambda A_, B_, g: np.exp(-g * ((A_[:, None, :] - B_[None, :, :]) ** 2).sum(-1))
                        rejilla = [g / d for g in (0.03, 0.1, 0.3, 1.0)]
                        f = lambda A_, B_, g: kern(A_, B_, g)
                    else:
                        rejilla = (0.05, 0.1, 0.2, 0.4)
                        rng_k = np.random.default_rng(1000 * semilla + d)
                        f = lambda A_, B_, e: kernel_cuantico(A_, B_, e, a.shots, rng_k, simetrico=A_ is B_)
                    auc_va, C, pk, svm = elegir_svm(lambda p: f(Ztr, Ztr, p), lambda p: f(Zva, Ztr, p), ytr, yva, rejilla)
                    s_ev = np.concatenate([svm.decision_function(f(Zev[i:i + 2000], Ztr, pk)) for i in range(0, len(Zev), 2000)])
                    guardar(cabeza, d, int(len(svm.support_)) + 1, svm.decision_function(f(Ztr, Ztr, pk)), s_ev,
                            {"hiper": f"C={C}, param={pk:.4g}" + (f", shots={a.shots}" if a.shots and cabeza == "svm_cuantico" else ""),
                             "n_vectores_soporte": int(len(svm.support_))}, t0)

    print(f"\nListo. Resultados en {archivo}")


if __name__ == "__main__":
    main()
