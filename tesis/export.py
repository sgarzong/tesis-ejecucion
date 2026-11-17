"""Copia de entrenamiento: fotos derechas (EXIF aplicado) y reducidas, más el paquete para Colab.

    python -m tesis.export            # data/img320/ + data/manifest_320.csv + data/tesis_colab.zip

Por qué derechas: el 84 % de las fotos viene acostada con la etiqueta EXIF de rotación, y tf.io.decode_image
la ignora. Sin esto el modelo entrena con fotos de lado y la máscara de la superposición cae en otra esquina.
El teléfono entrega cuadros derechos, así que la copia corrige también el desajuste con el despliegue.
Por qué reducidas: data._load lleva todo a 224x224; 320 px de lado mayor conservan ese detalle y bajan el
paquete de ~48 GB a ~1.5 GB.
"""
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from . import evidencia

SIDE = 320
OUT = Path("data/img320")


def upright(job):
    src, dst = job
    if Path(dst).exists():
        return
    img = Image.open(src)
    img.draft("RGB", (SIDE * 2, SIDE * 2))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((SIDE, SIDE), Image.Resampling.LANCZOS)
    img.save(dst, "JPEG", quality=92)  # sin EXIF: los píxeles ya están derechos


def main():
    m = pd.read_csv("data/manifest.csv")
    dst = [(OUT / lab / Path(p).with_suffix(".jpg").name).as_posix() for p, lab in zip(m.path, m.label)]
    for lab in m.label.unique():
        (OUT / lab).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as ex:
        list(ex.map(upright, zip(m.path, dst), chunksize=32))
    m.assign(path=dst).to_csv("data/manifest_320.csv", index=False)
    # La partición congelada (tesis.split), con las rutas de la copia reducida: es la que lee tesis.train
    part = pd.read_csv("data/particion.csv", keep_default_na=False)
    assert set(part.path) == set(m.path), "data/particion.csv no corresponde al manifiesto: corre tesis.split"
    part.assign(path=part.path.map(dict(zip(m.path, dst)))).to_csv("data/particion_320.csv", index=False)

    # Con el código viajan los convenios y lo demás que el paquete de evidencias toma de data/
    extra = [f for f in evidencia.datos() if f.name not in ("manifest.csv", "particion.csv")]
    with zipfile.ZipFile("data/tesis_colab.zip", "w", zipfile.ZIP_STORED) as z:  # JPEG ya viene comprimido
        for f in [*Path("tesis").glob("*.py"), Path("requirements.txt"), *extra]:
            z.write(f)
        z.write("data/manifest_320.csv", "data/manifest.csv")
        z.write("data/particion_320.csv", "data/particion.csv")
        for f in dst:
            z.write(f)
    print(f"{len(m)} fotos -> {OUT} · paquete data/tesis_colab.zip "
          f"{Path('data/tesis_colab.zip').stat().st_size / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
