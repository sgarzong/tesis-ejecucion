"""Evalúa un run sobre su conjunto de prueba (Keras o TFLite) y escribe métricas en JSON.

Uso:
    python -m tesis.evaluate runs/A_mobilenetv3 model_int8.tflite
    python -m tesis.evaluate runs/A_mobilenetv3 model_fp16.tflite --val   # predicciones sobre la validación
Guarda las probabilidades (probs_<formato>.npy, filas en el orden de test.csv). La verificación contra convenio y los
intervalos de confianza se calculan después, desde esas predicciones (tesis.evidencia.tabla).
"""
import argparse
import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import CLASSES
from .data import dataset
from .metrics import report


def predict(model_path, df):
    ds = dataset(df, batch=1 if model_path.suffix == ".tflite" else 64)
    if model_path.suffix != ".tflite":
        return keras.models.load_model(model_path).predict(ds)
    try:
        it = tf.lite.Interpreter(str(model_path))
        it.allocate_tensors()
    except RuntimeError:  # XNNPACK no prepara el MobileNetV3 int8: kernels estándar, mismos resultados
        it = tf.lite.Interpreter(str(model_path), experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES))
        it.allocate_tensors()
    inp, out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    probs = []
    for x, _ in ds:
        it.set_tensor(inp, x.numpy())
        it.invoke()
        probs.append(it.get_tensor(out)[0])
    return np.array(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("model", nargs="?", default="model.keras")
    ap.add_argument("--val", action="store_true", help="solo predice la validación: val_probs_<formato>.npy")
    args = ap.parse_args()

    run = Path(args.run)
    if args.val:  # con qué calibrar el umbral del verificador (tesis.metrics.calibrar_umbral) sin tocar la prueba
        val = pd.read_csv(run / "val.csv")
        np.save(run / f"val_probs_{Path(args.model).stem}.npy", np.concatenate(  # por partes: dataset() las deja en RAM
            [predict(run / args.model, val.iloc[i:i + 1000]) for i in range(0, len(val), 1000)]))
        return
    test = pd.read_csv(run / "test.csv")
    probs = predict(run / args.model, test)
    m = report(test.label.map(CLASSES.index).values, probs)
    m["size_mb"] = (run / args.model).stat().st_size / 2**20

    stem = Path(args.model).stem
    np.save(run / f"probs_{stem}.npy", probs)  # para bootstrap entre configuraciones
    json.dump(m, open(run / f"metrics_{stem}.json", "w"), indent=2, default=float)
    print(json.dumps({k: m[k] for k in ("f1_macro", "recall_macro", "reject_recall", "size_mb")}, indent=2, default=float))


if __name__ == "__main__":
    main()
