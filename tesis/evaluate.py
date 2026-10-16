"""Evalúa un run sobre su conjunto de prueba (Keras o TFLite) y escribe métricas en JSON.

Uso:
    python -m tesis.evaluate runs/A_mobilenetv3 model_int8.tflite --contracts data/contracts.csv
contracts.csv (opcional): store, label   -> una fila por convenio vigente del local.
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
from .metrics import report, verification


def predict(model_path, df):
    ds = dataset(df, batch=1 if model_path.suffix == ".tflite" else 64)
    if model_path.suffix != ".tflite":
        return keras.models.load_model(model_path).predict(ds)
    it = tf.lite.Interpreter(str(model_path))
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
    ap.add_argument("--contracts")
    args = ap.parse_args()

    run = Path(args.run)
    test = pd.read_csv(run / "test.csv")
    probs = predict(run / args.model, test)
    y, stores = test.label.map(CLASSES.index).values, test.store.values

    contracts = None
    if args.contracts:
        contracts = pd.read_csv(args.contracts).groupby("store").label.agg(set).to_dict()
    m = report(y, probs, stores, contracts)
    if contracts:
        m.update(verification(probs, y, stores, contracts))
    m["size_mb"] = (run / args.model).stat().st_size / 2**20

    stem = Path(args.model).stem
    np.save(run / f"probs_{stem}.npy", probs)  # para bootstrap entre configuraciones
    json.dump(m, open(run / f"metrics_{stem}.json", "w"), indent=2, default=float)
    print(json.dumps({k: m[k] for k in ("f1_macro", "recall_macro", "reject_recall", "size_mb")}, indent=2, default=float))


if __name__ == "__main__":
    main()
