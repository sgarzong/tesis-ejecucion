"""Paquete de evidencias para la revisión del tutor: lo que respalda los resultados y el repositorio no lleva.

    python -m tesis.evidencia [carpeta de runs] [salida.zip]    # por defecto runs/odk y evidencia_tesis.zip
    python -m tesis.evidencia --tabla <paquete descomprimido>    # recalcula los resultados desde las predicciones

No lleva fotos. Lleva:
  LEEME.md         qué hay, de qué versión del código salió y cómo comprobarlo
  datos/           manifiesto (una fila por foto: clase, local, fecha, fuente, verificada), partición congelada,
                   convenios con su emparejamiento, locales y las bitácoras de revisión manual
  corridas/<run>/  modelos, args.json (configuración), history.csv (registro de entrenamiento), train/val/test.csv,
                   probs_<formato>.npy (predicciones sobre la prueba) y metrics_<formato>.json
  dispositivo/     mediciones originales del Honor X6b (dispositivo*.csv del modelo solo, fotos_app_* del prototipo)
  codigo/          el paquete tesis, los cuadernos, las pruebas y la planificación con que se generó todo

En Colab lo arma la última celda de entrenar_tesis con los runs de Drive; lo que sale de data/ viaja en
tesis_colab.zip (tesis.export). Las bitácoras van sin la ruta de origen de cada foto, que nombra al mercaderista:
queda el identificador con que la foto aparece en el manifiesto.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import IDX, evaluate_run
from .split import fingerprint

DATOS = ("manifest.csv", "particion.csv", "stores_split.csv", "convenios.csv", "convenios_emparejamiento.csv",
         "locales.csv", "revisiones.csv", "auditoria.csv", "auditoria_concordancia.csv")
CODIGO = ("tesis/*.py", "tests/*.py", "notebooks/*.ipynb", "docs/*.md", "android/README.md", "README.md",
          "requirements.txt")


def datos(d=Path("data")):
    """Archivos de data/ que van al paquete. Con las bitácoras de tesis.sort_gui a mano, escribe antes
    revisiones.csv: revisión, foto, clase sugerida y clase asignada (manda la última fila de cada foto)."""
    logs = sorted(d.glob("clasificadas*/revision.csv"))
    if logs:
        from .sort_gui import copy_name  # tkinter: solo donde se revisó
        pd.concat([pd.read_csv(f, encoding="utf-8-sig").assign(revision=f.parent.name, foto=lambda r: r.path.map(copy_name))
                   for f in logs])[["revision", "foto", "sugerida", "label"]].to_csv(d / "revisiones.csv", index=False)
    return [d / f for f in DATOS if (d / f).exists()]


def tabla(runs, conv):
    """Resultados de cada run y formato, recalculados desde su test.csv y sus predicciones. El experimento B promedia
    sobre las clases que su prueba tiene."""
    rows = []
    for t in sorted(Path(runs).glob("*/test.csv")):
        test = pd.read_csv(t)
        labels = sorted(test.label.map(IDX).unique()) if t.parent.name.startswith("B_") else None
        for p in sorted(t.parent.glob("probs_*.npy")):
            rows.append({"run": t.parent.name, "formato": p.stem[6:], "clases": len(labels or IDX),
                         **evaluate_run(test, np.load(p), conv, labels)})
    return pd.DataFrame(rows)


def leeme(files, runs):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if subprocess.run(["git", "status", "--porcelain", "tesis"], capture_output=True, text=True).stdout.strip():
            commit += " (con cambios sin confirmar en tesis/)"
    except OSError:
        commit = ""
    out = ["# Paquete de evidencias", "",
           "Tesis: clasificación de muebles de exhibición en dispositivos de gama baja (Sebastián Garzón, USFQ).",
           f"Generado el {pd.Timestamp.now():%Y-%m-%d %H:%M} con `python -m tesis.evidencia`."
           + (f" Código: commit `{commit}`." if commit else ""), "",
           "No incluye fotografías: son de Industrias Ales. Cada foto se identifica por su nombre de archivo en "
           "`datos/manifest.csv`.", ""]
    part = next((f for f in files if f.name == "particion.csv"), None)
    if part:
        p = pd.read_csv(part, keep_default_na=False)
        out += ["## Partición congelada", "",
                f"`datos/particion.csv`, huella `{fingerprint(p)}` (`tesis.split.fingerprint`; es la que imprime el "
                "cuaderno antes de entrenar y la que registra el README del repositorio). Una fila por foto: "
                "`split` es la partición de su local; `A` y `B`, su papel en cada experimento (vacío: no se usa). "
                "Validación y prueba solo tienen fotos verificadas a mano.", "",
                "Fotos del experimento A por partición y clase:", "", "```",
                p[p.A != ""].groupby(["A", "label"]).size().unstack(fill_value=0).to_string(), "```", "",
                "Fotos del experimento B (temporal) por partición y clase:", "", "```",
                p[p.B != ""].groupby(["B", "label"]).size().unstack(fill_value=0).to_string(), "```", "",
                f"Fotos de entrenamiento de A sin verificación manual: {((p.A == 'train') & (p.verified == 0)).sum():,} "
                f"de {(p.A == 'train').sum():,}, por fuente "
                f"{p[(p.A == 'train') & (p.verified == 0)].source.value_counts().to_dict()} "
                "(`sint`: capturas inválidas sintéticas; `odk`: fotos de ODK con etiqueta automática).", ""]
    rev = next((f for f in files if f.name == "revisiones.csv"), None)
    if rev:
        r = pd.read_csv(rev).drop_duplicates(["revision", "foto"], keep="last")
        t = r.assign(resultado=np.select([r.label == "descartada", r.label == r.sugerida],
                                         ["descartada", "aceptada como se sugirió"], "corregida"))
        out += ["## Revisión manual de etiquetas", "", "`datos/revisiones.csv`: cada decisión de la herramienta de "
                "revisión (`tesis.sort_gui`), con la clase sugerida y la asignada.", "", "```",
                t.groupby(["revision", "resultado"]).size().unstack(fill_value=0).assign(
                    revisadas=lambda d: d.sum(axis=1)).to_string(), "```", ""]
    aud = next((f for f in files if f.name == "auditoria_concordancia.csv"), None)
    if aud:
        out += ["## Auditoría ciega de las etiquetas", "", "Una segunda persona clasificó, sin ver la etiqueta, una "
                "muestra de la validación y la prueba del experimento A (`tesis.auditoria`). `datos/auditoria.csv` trae "
                "cada foto con las dos clases. La auditoría mide las etiquetas; no las cambió.", "", "```",
                pd.read_csv(aud).round(3).to_string(index=False), "```", ""]
    names = sorted({f.parent.name for f in Path(runs).glob("*/*") if f.is_file()})
    out += ["## Corridas y mediciones", "",
            ("`corridas/`: " + ", ".join(f"`{n}`" for n in names) + ".") if names else "Sin corridas en este paquete.",
            "Cada una trae su configuración (`args.json`), el registro de entrenamiento (`history.csv`), las listas "
            "de fotos que usó (`train.csv`, `val.csv`, `test.csv`), los modelos y, por formato, las predicciones "
            "sobre la prueba (`probs_<formato>.npy`, filas en el orden de `test.csv`, columnas en el orden de "
            "`tesis.config.CLASSES`) y sus métricas.", "",
            "`dispositivo/`: salidas originales de `python -m tesis.device` en el Honor X6b.", "",
            "## Cómo comprobar los resultados sin las fotos", "", "```",
            "pip install numpy pandas scikit-learn", "cd codigo",
            "python -m tesis.evidencia --tabla ..    # recalcula cada fila desde corridas/*/probs_*.npy", "```", "",
            "La tabla usa `datos/convenios.csv`, que sale de las maestras de muebles de Industrias Ales "
            "(`tesis.convenios`), no de las etiquetas de las fotos.", ""]
    return "\n".join(out)


def paquete(runs=Path("runs/odk"), out=Path("evidencia_tesis.zip")):
    files, runs = datos(), Path(runs)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("LEEME.md", leeme(files, runs))
        for f in files:
            z.write(f, f"datos/{f.name}")
        for f in sorted(runs.rglob("*")):
            rel = f.relative_to(runs)
            if f.is_file() and "saved_model" not in rel.parts:  # paso intermedio de la conversión: ya va model.keras
                z.write(f, ("dispositivo/" if len(rel.parts) == 1 else "corridas/") + rel.as_posix())
        for pattern in CODIGO:
            for f in sorted(Path().glob(pattern)):
                z.write(f, f"codigo/{f.as_posix()}")
    print(f"{out}: {out.stat().st_size / 2**20:.1f} MB")
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--tabla"]:
        base = Path(sys.argv[2])
        pd.set_option("display.width", 250, "display.max_columns", 40)
        print(tabla(base / "corridas", pd.read_csv(base / "datos/convenios.csv")).round(4).T.to_string())
    else:
        paquete(*map(Path, sys.argv[1:3]))
