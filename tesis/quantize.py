"""Conversión a TFLite float32 e int8 post-entrenamiento (sección 5.5, procedimiento estándar de TFLite).

Uso:
    python -m tesis.quantize runs/A_mobilenetv3
Genera model_fp32.tflite y model_int8.tflite en la carpeta del run. La calibración usa
imágenes de entrenamiento sin aumentos.
"""
import sys
from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf

from .data import dataset

N_CALIB = 300


def convert(run):
    run = Path(run)
    saved = run / "saved_model"
    keras.models.load_model(run / "model.keras").export(saved)

    (run / "model_fp32.tflite").write_bytes(tf.lite.TFLiteConverter.from_saved_model(str(saved)).convert())

    calib = pd.read_csv(run / "train.csv").sample(N_CALIB, random_state=0)

    def representative():
        for x, _ in dataset(calib, batch=1).take(N_CALIB):
            yield [x]

    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Entrada y salida quedan en float32: la app entrega RGB [0, 255] sin conocer la escala int8.
    (run / "model_int8.tflite").write_bytes(conv.convert())

    for f in ("model_fp32.tflite", "model_int8.tflite"):
        print(f, f"{(run / f).stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    convert(sys.argv[1])
