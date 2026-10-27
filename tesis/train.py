"""Entrenamiento por transferencia con protocolo común (sección 5.5).

Uso:
    python -m tesis.train --model mobilenetv3 --exp A
    python -m tesis.train --model resnet50 --exp A --no-mask   # control del atajo de la superposición
"""
import argparse
import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from .config import CLASSES, IMG_SIZE, MASK_H, MASK_W, MODELS, SEED
from .data import AUGMENT, dataset
from .split import experiment


def overlay_mask():
    m = np.ones((IMG_SIZE, IMG_SIZE, 1), dtype="float32")
    m[-round(IMG_SIZE * MASK_H):, -round(IMG_SIZE * MASK_W):] = 0
    return m


def build_model(name, mask=True):
    """Entrada: RGB float32 [0, 255] de IMG_SIZE². Máscara y preprocesamiento incluidos en el grafo."""
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = inp * overlay_mask() if mask else inp
    shape = (IMG_SIZE, IMG_SIZE, 3)
    if name == "resnet50":
        x = keras.applications.resnet50.preprocess_input(x)
        base = keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=shape, pooling="avg")
    elif name == "mobilenetv3":
        base = keras.applications.MobileNetV3Large(include_top=False, weights="imagenet", input_shape=shape, pooling="avg")
    elif name == "efficientnetb0":
        base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=shape, pooling="avg")
    else:
        raise ValueError(f"modelo desconocido: {name}")
    x = keras.layers.Dropout(0.2)(base(x))
    return keras.Model(inp, keras.layers.Dense(len(CLASSES), activation="softmax")(x), name=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--exp", choices=["A", "B"], default="A")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--stores", default="data/stores_split.csv")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    df = pd.read_csv(args.manifest)
    parts = experiment(df[df.label.notna()], pd.read_csv(args.stores), args.exp)
    run = Path("runs") / f"{args.exp}_{args.model}{'_nomask' if args.no_mask else ''}"
    run.mkdir(parents=True, exist_ok=True)

    y = parts["train"].label.map(CLASSES.index).values
    present = np.unique(y)
    weights = dict(zip(present.tolist(), compute_class_weight("balanced", classes=present, y=y)))

    model = build_model(args.model, mask=not args.no_mask)
    # Los aumentos envuelven al modelo solo para entrenar (corren en la GPU); se guarda `model`, sin ellos.
    trainer = keras.Sequential([keras.Input((IMG_SIZE, IMG_SIZE, 3)), AUGMENT, model])
    trainer.compile(keras.optimizers.Adam(args.lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    trainer.fit(
        dataset(parts["train"], args.batch, train=True),
        validation_data=dataset(parts["val"], args.batch),
        epochs=args.epochs,
        class_weight=weights,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.3),
            keras.callbacks.CSVLogger(run / "history.csv"),
        ],
    )
    model.save(run / "model.keras")
    for k, v in parts.items():
        v.to_csv(run / f"{k}.csv", index=False)
    json.dump(vars(args), open(run / "args.json", "w"), indent=2)
    print(f"guardado en {run}")


if __name__ == "__main__":
    main()
