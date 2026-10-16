"""Carga de imágenes con tf.data. El enmascaramiento de la superposición vive dentro del modelo
(ver train.build_model), así la versión TFLite lo aplica igual en el dispositivo."""
import keras
import tensorflow as tf

from .config import CLASSES, IMG_SIZE, SEED

AUGMENT = keras.Sequential([
    keras.layers.RandomRotation(0.03, fill_mode="constant"),
    keras.layers.RandomZoom((-0.15, 0.0)),  # recorte aleatorio
    keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant"),
    keras.layers.RandomBrightness(0.25, value_range=(0, 255)),
    keras.layers.RandomContrast(0.25),
])
# ponytail: sin capa de perspectiva nativa en Keras; RandomZoom+Translation la aproximan.


def _load(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return tf.image.resize(img, (IMG_SIZE, IMG_SIZE)), label


def _jpeg(img, label):
    img = tf.image.random_jpeg_quality(tf.cast(img, tf.uint8), 40, 95)  # compresión con pérdida
    return tf.cast(img, tf.float32), label


def dataset(df, batch=32, train=False):
    """df con columnas path y label (nombre de clase). Imágenes float32 en [0, 255]."""
    labels = df.label.map(CLASSES.index).values
    ds = tf.data.Dataset.from_tensor_slices((df.path.values, labels))
    if train:
        ds = ds.shuffle(len(df), seed=SEED)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        ds = ds.map(_jpeg, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch).map(lambda x, y: (AUGMENT(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch)
    return ds.prefetch(tf.data.AUTOTUNE)
