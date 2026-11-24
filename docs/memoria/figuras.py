"""Figuras y cifras de la memoria (docs/memoria/memoria.tex), desde las corridas y sus predicciones guardadas.

    python docs/memoria/figuras.py [carpeta de runs]     # por defecto runs/final

Escribe docs/memoria/fig/pareto.pdf y confusion.pdf, e imprime las tablas de las que salen las cifras del texto.
No usa fotos: solo test.csv, probs_*.npy, metrics_*.json, dispositivo.csv y data/convenios.csv.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tesis.config import CLASSES, THRESHOLDS
from tesis.evidencia import tabla
from tesis.metrics import IDX, wilson

RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/final")
FIG = Path(__file__).parent / "fig"
NOMBRE = {"A_resnet50": "ResNet-50", "A_efficientnetb0": "EfficientNet-B0", "A_mobilenetv3": "MobileNetV3-Large"}
FORMATO = {"model": "float32", "model_fp16": "float16", "model_dyn": "int8 dinámico", "model_int8": "int8",
           "model_int8bal": "int8 balanceado"}
CLASE = ["chiclera", "lateral", "check out", "cabecera", "exhibidor\nde piso", "captura\ninválida", "contenido\najeno"]
plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})


def resultados():
    t = tabla(RUNS, pd.read_csv("data/convenios.csv"))
    t = t[t.run.isin(NOMBRE)].copy()
    t["MB"] = [json.load(open(RUNS / r / f"metrics_{f}.json"))["size_mb"] for r, f in zip(t.run, t.formato)]
    t["caida"] = t.run.map(t[t.formato == "model"].set_index("run").f1_macro) - t.f1_macro
    d = pd.read_csv(RUNS / "dispositivo.csv")
    return t.merge(d[["run", "formato", "lat_p50_ms", "lat_p95_ms", "mem_mb"]], on=["run", "formato"], how="left")


def pareto(t):
    """F1-macro frente a la latencia p95 del modelo solo en el Honor X6b, con el frente de Pareto."""
    fig, a = plt.subplots(figsize=(3.45, 2.5))
    for (run, nombre), m in zip(NOMBRE.items(), "s^o"):
        d = t[t.run == run]
        a.plot(d.lat_p95_ms, d.f1_macro, m, ms=4, label=nombre)
    s = t.dropna(subset=["lat_p95_ms"]).sort_values(["lat_p95_ms", "f1_macro"], ascending=[True, False])
    frente = s[s.f1_macro > s.f1_macro.cummax().shift(fill_value=-1)]
    a.step(frente.lat_p95_ms, frente.f1_macro, "k-", where="post", lw=.8, label="frente de Pareto")
    a.axhline(THRESHOLDS["f1_macro"], color="gray", ls=":", lw=.8)
    a.text(t.lat_p95_ms.max(), THRESHOLDS["f1_macro"] + .004, "umbral de F1-macro", ha="right", fontsize=7, color="gray")
    a.set_xscale("log")
    a.set_xticks([20, 30, 50, 100, 200], ["20", "30", "50", "100", "200"])
    a.minorticks_off()
    a.set_xlabel("latencia p95 en el Honor X6b (ms)")
    a.set_ylabel("F1-macro (7 clases)")
    a.grid(alpha=.3, lw=.4)
    a.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(pad=.3)
    fig.savefig(FIG / "pareto.pdf")
    return frente


def confusion(run="A_mobilenetv3", fmt="model_fp16"):
    """Matriz de confusión normalizada por fila del modelo que lleva el teléfono."""
    test = pd.read_csv(RUNS / run / "test.csv")
    y, p = test.label.map(IDX).values, np.load(RUNS / run / f"probs_{fmt}.npy").argmax(1)
    c = confusion_matrix(y, p, labels=range(len(CLASSES)), normalize="true")
    fig, a = plt.subplots(figsize=(3.45, 3.1))
    a.imshow(c, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if c[i, j] >= .005:
                a.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if c[i, j] > .5 else "black")
    a.set_xticks(range(len(CLASSES)), CLASE, rotation=90, fontsize=6.5)
    a.set_yticks(range(len(CLASSES)), CLASE, fontsize=6.5)
    a.set_xlabel("clase predicha")
    a.set_ylabel("clase verdadera")
    fig.tight_layout(pad=.3)
    fig.savefig(FIG / "confusion.pdf")
    pr, rc, f1, n = precision_recall_fscore_support(y, p, labels=range(len(CLASSES)), zero_division=0)
    return pd.DataFrame({"clase": CLASSES, "fotos": n, "precisión": pr, "recall": rc, "F1": f1,
                         "IC 95 % del recall": ["[{:.3f}, {:.3f}]".format(*wilson(round(r * k), int(k))) for r, k in zip(rc, n)]})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    FIG.mkdir(exist_ok=True)
    t = resultados()
    cols = ["run", "formato", "MB", "f1_macro", "recall_macro", "reject_recall", "captura_invalida_recall",
            "contenido_ajeno_recall", "false_alarm", "caida", "lat_p50_ms", "lat_p95_ms", "mem_mb"]
    print(t[cols].assign(run=t.run.map(NOMBRE), formato=t.formato.map(FORMATO)).round(3).to_string(index=False))
    print("\nFrente de Pareto:")
    print(pareto(t)[["run", "formato", "f1_macro", "lat_p95_ms"]].round(3).to_string(index=False))
    print("\nPor clase, MobileNetV3-Large en float16:")
    print(confusion().round(3).to_string(index=False))
    print(f"\nfiguras en {FIG}")
