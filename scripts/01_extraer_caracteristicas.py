#!/usr/bin/env python3
"""
01_extraer_caracteristicas.py
=============================
Paso 1 del estudio cuantico: extraer UNA sola vez las caracteristicas de cada imagen
con backbones congelados, para que todas las cabezas (cuanticas y clasicas) reciban
exactamente las mismas entradas.

Que produce (en la carpeta --salida):
  manifiesto.csv                    una fila por imagen: conjunto, lesion, diagnostico, etiqueta
  feats_<backbone>.npz              matriz de caracteristicas (n_imagenes x dimension) + image_id
  resumen_extraccion.json           conteos, versiones y huellas (hash) para reproducibilidad

Conjuntos que se arman:
  train / val / test   las 2,000 imagenes del split_ledger (articulo CNN/ViT/CapsNet)
  cohorte              todas las demas imagenes de HAM10000 cuya lesion NO esta en el ledger
                       (el articulo reporta 7,124 imagenes de 5,740 lesiones)
  isic2018             test oficial de ISIC 2018 (opcional; 1,511 imagenes)

Por que asi:
  - El backbone esta congelado y en modo evaluacion: sin aumentacion, sin dropout,
    sin gradientes. Las caracteristicas son deterministas y se calculan una vez.
  - Las imagenes se redimensionan directo a 224x224 (sin recorte), igual que en tu
    articulo, y se normalizan con la media/desviacion propia de cada backbone.
  - Nada se ajusta aqui con datos de validacion o prueba: solo se "leen" imagenes.
    El PCA y todo lo que aprende va en el paso 2, ajustado SOLO con train.

Uso (PowerShell, una sola linea; cambia las rutas):
  python 01_extraer_caracteristicas.py --ledger "C:\\...\\split_ledger.csv"
      --ham-meta "C:\\...\\HAM10000_metadata.csv" --ham-dir "C:\\...\\carpeta_con_imagenes_HAM"
      --isic-gt "C:\\...\\ISIC2018_Task3_Test_GroundTruth.csv" --isic-dir "C:\\...\\carpeta_ISIC2018_test"
      --salida "C:\\proyecto_cuantico\\features"

  --ham-dir y --isic-dir se buscan de forma recursiva: basta con dar la carpeta padre.
  Para una prueba rapida sin descargar pesos: agrega --prueba (usa 64 imagenes y pesos aleatorios).

Requisitos: pip install torch torchvision timm pandas pillow numpy
"""
import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Backbones: nombre corto -> modelo de timm.
#  - EfficientNet-B1 preentrenado en ImageNet-1k (familia usada en tu articulo).
#  - ViT-B/32 con los pesos originales de Google en ImageNet-21k
#    (equivalente a google/vit-base-patch32-224-in21k que usaste con Hugging Face).
BACKBONES = {
    "effb1": "efficientnet_b1.ft_in1k",
    "vitb32": "vit_base_patch32_224.orig_in21k",
}
MALIGNAS = {"mel", "bcc", "akiec"}  # mismo criterio binario del articulo
EXT = {".jpg", ".jpeg", ".png"}


def huella(ruta):
    """SHA-256 de un archivo: permite citar exactamente que ledger se uso."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def indexar_imagenes(carpeta, salida):
    """Recorre la carpeta (recursivo) y devuelve {image_id: ruta}.

    Solo acepta archivos cuyo nombre es EXACTAMENTE el ID (ISIC_0012345.jpg). Los archivos con
    sufijo (por ejemplo ISIC_0012345_fototipoVI.jpg, versiones de Brown Skin) se ignoran, para no
    mezclar imagenes sinteticas con las originales. Si el mismo nombre exacto aparece en varias
    subcarpetas, se comprueba que sean el mismo archivo; si difieren, se detiene.
    """
    indice, huellas, ignorados, repetidos, conflictos = {}, {}, [], 0, []
    for p in sorted(Path(carpeta).rglob("*")):
        if p.suffix.lower() not in EXT or not p.stem.upper().startswith("ISIC_"):
            continue
        if not re.fullmatch(r"ISIC_\d{7}", p.stem, flags=re.I):
            ignorados.append(str(p))
            continue
        iid = p.stem.upper()
        if iid in indice:
            repetidos += 1
            if iid not in huellas:
                huellas[iid] = huella(indice[iid])
            if huella(p) != huellas[iid]:
                conflictos.append((iid, str(indice[iid]), str(p)))
            continue
        indice[iid] = p
    nombre = Path(carpeta).name or "raiz"
    if ignorados:
        pd.Series(ignorados, name="ruta").to_csv(Path(salida) / f"ignorados_{nombre}.csv", index=False)
    if conflictos:
        pd.DataFrame(conflictos, columns=["image_id", "ruta_1", "ruta_2"]).to_csv(
            Path(salida) / f"conflictos_{nombre}.csv", index=False)
        sys.exit(f"{len(conflictos)} IDs tienen archivos DISTINTOS con el mismo nombre en {carpeta}. "
                 f"Revisa conflictos_{nombre}.csv")
    print(f"  {carpeta}: {len(indice)} imagenes indexadas | {repetidos} copias identicas ignoradas | "
          f"{len(ignorados)} archivos con sufijo ignorados")
    return indice


def leer_isic2018(ruta_gt):
    """Acepta el formato con columna dx o el formato one-hot oficial (MEL, NV, ...)."""
    gt = pd.read_csv(ruta_gt)
    idc = "image_id" if "image_id" in gt else "image"
    if "dx" in gt:
        dx = gt["dx"].str.lower()
    else:
        dx = gt[["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]].idxmax(axis=1).str.lower()
    lesion = gt["lesion_id"] if "lesion_id" in gt else gt[idc]  # sin lesion_id: cada imagen es su propia lesion
    return pd.DataFrame({"image_id": gt[idc].str.upper(), "lesion_id": lesion, "dx": dx, "conjunto": "isic2018"})


def armar_manifiesto(a):
    print("Armando manifiesto...")
    led = pd.read_csv(a.ledger)
    meta = pd.read_csv(a.ham_meta)
    meta["image_id"] = meta["image_id"].str.upper()

    # Comprobaciones del ledger (deben pasar; si no, algo cambio respecto al articulo).
    assert led["image_id"].is_unique, "Hay imagenes repetidas en el ledger"
    assert (led.groupby("lesion_id")["partition"].nunique() == 1).all(), "Una lesion aparece en dos particiones"
    assert (led["label"] == led["dx_official"].isin(MALIGNAS).astype(int)).all(), "Etiquetas distintas al diagnostico"

    sub = led[["image_id", "lesion_id", "dx_official", "partition"]].rename(
        columns={"dx_official": "dx", "partition": "conjunto"})
    lesiones_sub = set(led["lesion_id"])
    coh = meta[~meta["lesion_id"].isin(lesiones_sub)][["image_id", "lesion_id", "dx"]].assign(conjunto="cohorte")
    partes = [sub, coh]
    if a.isic_gt:
        isic = leer_isic2018(a.isic_gt)
        compartidas = set(isic["lesion_id"]) & set(meta["lesion_id"])
        assert not compartidas, f"{len(compartidas)} lesiones de ISIC 2018 aparecen en HAM10000"
        partes.append(isic)
    man = pd.concat(partes, ignore_index=True)
    man["label"] = man["dx"].isin(MALIGNAS).astype(int)
    assert man["image_id"].is_unique, "Una imagen quedo en dos conjuntos"

    # Localizar archivos.
    indice = indexar_imagenes(a.ham_dir, a.salida)
    if a.isic_dir:
        indice.update({k: v for k, v in indexar_imagenes(a.isic_dir, a.salida).items() if k not in indice})
    man["ruta"] = man["image_id"].map(lambda i: str(indice[i]) if i in indice else "")
    faltan = man[man["ruta"] == ""]
    if len(faltan):
        faltan.to_csv(Path(a.salida) / "imagenes_faltantes.csv", index=False)
        print(f"  AVISO: {len(faltan)} imagenes no se encontraron (ver imagenes_faltantes.csv)")
        print(faltan.groupby("conjunto").size().to_string())
    man = man[man["ruta"] != ""].reset_index(drop=True)

    print("\nImagenes por conjunto (lesiones):")
    for c, g in man.groupby("conjunto", sort=False):
        print(f"  {c:9s} {len(g):6d} imagenes  {g['lesion_id'].nunique():6d} lesiones  "
              f"{int(g['label'].sum()):5d} malignas ({100 * g['label'].mean():.1f} %)")
    print("  Esperado del articulo: train 1404, val 297, test 299, cohorte 7124 (5740 lesiones)")
    return man


class Imagenes(torch.utils.data.Dataset):
    def __init__(self, rutas, tam, media, desv):
        self.rutas, self.tam = rutas, tam
        self.media = torch.tensor(media).view(3, 1, 1)
        self.desv = torch.tensor(desv).view(3, 1, 1)

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        im = Image.open(self.rutas[i]).convert("RGB").resize((self.tam, self.tam), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (x - self.media) / self.desv


@torch.inference_mode()
def extraer(nombre, modelo_timm, man, a, dispositivo):
    import timm
    print(f"\n=== {nombre} ({modelo_timm}) ===")
    modelo = timm.create_model(modelo_timm, pretrained=not a.prueba, num_classes=0).eval().to(dispositivo)
    cfg = modelo.pretrained_cfg
    tam = cfg["input_size"][-1]
    ds = Imagenes(man["ruta"].tolist(), tam, cfg["mean"], cfg["std"])
    dl = torch.utils.data.DataLoader(ds, batch_size=a.lote, num_workers=a.workers, pin_memory=dispositivo.type == "cuda")
    salidas, t0 = [], time.time()
    for k, x in enumerate(dl):
        salidas.append(modelo(x.to(dispositivo, non_blocking=True)).float().cpu())  # float32, sin AMP
        if k % 20 == 0:
            print(f"  lote {k + 1}/{len(dl)}  ({time.time() - t0:.0f} s)", flush=True)
    F = torch.cat(salidas).numpy()
    assert np.isfinite(F).all(), "Hay NaN o infinitos en las caracteristicas"

    # Chequeo de determinismo: recalcular las primeras imagenes debe dar lo mismo.
    x0 = torch.stack([ds[i] for i in range(min(8, len(ds)))]).to(dispositivo)
    dif = float(np.abs(modelo(x0).float().cpu().numpy() - F[: len(x0)]).max())
    print(f"  {F.shape[0]} imagenes x {F.shape[1]} dimensiones | diferencia al recalcular: {dif:.2e}")
    np.savez_compressed(Path(a.salida) / f"feats_{nombre}.npz", X=F.astype(np.float32),
                        image_id=man["image_id"].to_numpy())
    return {"modelo_timm": modelo_timm, "dimension": int(F.shape[1]), "entrada": tam,
            "media": list(cfg["mean"]), "desv": list(cfg["std"]), "dif_recalculo": dif,
            "segundos": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--ham-meta", required=True)
    ap.add_argument("--ham-dir", required=True)
    ap.add_argument("--isic-gt")
    ap.add_argument("--isic-dir")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--backbones", default="effb1,vitb32")
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0, help="0 es lo mas seguro en Windows")
    ap.add_argument("--prueba", action="store_true", help="64 imagenes y pesos aleatorios, solo para probar")
    a = ap.parse_args()
    if bool(a.isic_gt) != bool(a.isic_dir):
        sys.exit("Para ISIC 2018 indica ambos: --isic-gt y --isic-dir")
    Path(a.salida).mkdir(parents=True, exist_ok=True)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo}" + (f" ({torch.cuda.get_device_name(0)})" if dispositivo.type == "cuda" else ""))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False  # float32 exacto: mas reproducible
    torch.backends.cudnn.allow_tf32 = False

    man = armar_manifiesto(a)
    if a.prueba:
        man = man.groupby("conjunto", group_keys=False).head(16).reset_index(drop=True)
        print(f"\nMODO PRUEBA: {len(man)} imagenes, pesos aleatorios (no uses estos resultados)")
    man.to_csv(Path(a.salida) / "manifiesto.csv", index=False)

    import timm
    resumen = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "prueba": a.prueba,
               "ledger_sha256": huella(a.ledger), "n_imagenes": len(man),
               "por_conjunto": man["conjunto"].value_counts().to_dict(),
               "versiones": {"python": platform.python_version(), "torch": torch.__version__,
                             "timm": timm.__version__, "dispositivo": str(dispositivo)},
               "backbones": {}}
    for nombre in a.backbones.split(","):
        resumen["backbones"][nombre] = extraer(nombre, BACKBONES[nombre], man, a, dispositivo)
    (Path(a.salida) / "resumen_extraccion.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nListo. Archivos en: {a.salida}")


if __name__ == "__main__":
    main()
