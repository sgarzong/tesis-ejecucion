"""Paquete de evidencias para la revisión del tutor: lo que respalda los resultados y el repositorio no lleva.

    python -m tesis.evidencia [carpeta de runs] [salida.zip]    # por defecto runs/final y evidencia_tesis.zip
    python -m tesis.evidencia --reducido [carpeta de runs] [salida.zip]   # sin los modelos: evidencia_tesis_reducido.zip
    python -m tesis.evidencia --tabla <paquete descomprimido>    # recalcula los resultados y los compara

No lleva fotos. Lleva:
  LEEME.md         qué hay, de qué versión del código salió y cómo comprobarlo
  resultados.csv   la tabla de resultados con que se armó el paquete: contra ella compara --tabla
  datos/           manifiesto (una fila por foto: clase, local, fecha, fuente, verificada), partición congelada,
                   convenios con su emparejamiento, locales y las bitácoras de revisión manual
  corridas/<run>/  modelos (no en el reducido), args.json (configuración), history.csv (registro de entrenamiento),
                   train/val/test.csv, probs_<formato>.npy (predicciones sobre la prueba), val_probs_<formato>.npy
                   (sobre la validación: calibran el umbral del verificador) y metrics_<formato>.json
  dispositivo/     mediciones originales del Honor X6b (dispositivo*.csv del modelo solo, fotos_* del prototipo)
  codigo/          el paquete tesis, los cuadernos, las pruebas, la decisión del prototipo (Clasificador.java) y la
                   planificación con que se generó todo

En Colab lo arma la última celda de entrenar_tesis con los runs de Drive; lo que sale de data/ viaja en
tesis_colab.zip (tesis.export). Las bitácoras van sin la ruta de origen de cada foto, que nombra al mercaderista:
queda el identificador con que la foto aparece en el manifiesto.
"""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import IDX, calibrar_umbral, contract_keys, contracts, evaluate_run

DATOS = ("manifest.csv", "particion.csv", "stores_split.csv", "convenios.csv", "convenios_emparejamiento.csv",
         "locales.csv", "revisiones.csv", "auditoria.csv", "auditoria_concordancia.csv")
CODIGO = ("tesis/*.py", "tests/*.py", "notebooks/*.ipynb", "docs/*.md", "android/README.md",
          "android/app/src/main/java/ec/usfq/tesis/Clasificador.java", "README.md",
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


def umbral(run, conv):
    """Umbral de la decisión del verificador (tesis.metrics.decide) de un run, calibrado con su validación: val.csv y
    val_probs_<formato>.npy, que deja `python -m tesis.evaluate <run> model_fp16.tflite --val` (float32 en el control
    sin máscara, que no tiene float16). Vale para todos los formatos del run. Sin esas predicciones, inf: la decisión
    restringida al convenio."""
    p = next(iter(sorted(Path(run).glob("val_probs_*.npy"))), None)
    if p is None:
        return float("inf")
    val = pd.read_csv(Path(run) / "val.csv")
    return calibrar_umbral(np.load(p), val.label.map(IDX).values, contract_keys(val, conv), contracts(conv))


def tabla(runs, conv):
    """Resultados de cada run y formato, recalculados desde su test.csv y sus predicciones. El experimento B promedia
    sobre las clases que su prueba tiene."""
    rows = []
    for t in sorted(Path(runs).glob("*/test.csv")):
        test, tau = pd.read_csv(t), umbral(t.parent, conv)
        labels = sorted(test.label.map(IDX).unique()) if t.parent.name.startswith("B_") else None
        for p in sorted(t.parent.glob("probs_*.npy")):
            rows.append({"run": t.parent.name, "formato": p.stem[6:], "clases": len(labels or IDX),
                         **evaluate_run(test, np.load(p), conv, labels, tau)})
    return pd.DataFrame(rows)


def comprobar(t, base):
    """Compara la tabla recalculada `t` con lo que el paquete trae como reportado: resultados.csv (la tabla con que
    se armó el paquete) y, para las métricas predictivas, el metrics_<formato>.json que escribió cada evaluación.
    Devuelve las líneas del informe."""
    out = []
    ref = Path(base) / "resultados.csv"
    if ref.exists():
        r = pd.read_csv(ref)
        num = [c for c in t.columns if c in r.columns and pd.api.types.is_numeric_dtype(t[c])]
        a, b = t[num].to_numpy(float), r[num].to_numpy(float)
        igual = np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=True) | (np.isinf(a) & np.isinf(b))
        out.append(f"resultados.csv: coinciden {int(igual.sum()):,} de {igual.size:,} cifras "
                   f"({len(t)} filas, {len(num)} columnas)")
        out += [f"  distinta: {t.run[i]} {t.formato[i]} {num[j]}: {a[i, j]} recalculada, {b[i, j]} reportada"
                for i, j in zip(*np.nonzero(~igual))][:20]
    dif = []
    for x in t[t.clases == len(IDX)].itertuples():  # las de B promedian aquí solo las clases que su prueba tiene
        m = Path(base) / "corridas" / x.run / f"metrics_{x.formato}.json"
        if m.exists():
            rep = json.loads(m.read_text())
            dif += [abs(getattr(x, k) - rep[k]) for k in ("f1_macro", "recall_macro")]
    if dif:
        out.append(f"metrics_<formato>.json de cada corrida: {len(dif)} cifras de F1-macro y recall macro, "
                   f"mayor diferencia {max(dif):.1e}")
    return out


def leeme(files, runs, reducido=False):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if subprocess.run(["git", "status", "--porcelain", "tesis"], capture_output=True, text=True).stdout.strip():
            commit += " (con cambios sin confirmar en tesis/)"
    except OSError:
        commit = ""
    out = ["# Paquete de evidencias" + (" (reducido)" if reducido else ""), "",
           "Tesis: clasificación de muebles de exhibición en dispositivos de gama baja (Sebastián Garzón, USFQ).",
           f"Generado el {pd.Timestamp.now():%Y-%m-%d %H:%M} con `python -m tesis.evidencia`."
           + (f" Código: commit `{commit}`." if commit else ""), "",
           "No incluye fotografías: son de Industrias Ales. Cada foto se identifica por su nombre de archivo en "
           "`datos/manifest.csv`.", ""]
    if reducido:
        out += ["Es el paquete reducido: tampoco lleva los modelos entrenados (`model.keras` y los `.tflite`), que "
                "están en el paquete completo. Con lo que trae se recalculan todos los resultados, porque salen de "
                "las predicciones guardadas y no de los modelos.", ""]
    part = next((f for f in files if f.name == "particion.csv"), None)
    if part:
        from .split import fingerprint  # aquí y no arriba: split pide Pillow y --tabla debe correr sin él
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
            "de fotos que usó (`train.csv`, `val.csv`, `test.csv`)" + ("" if reducido else ", los modelos") + " y, por "
            "formato, las predicciones sobre la prueba (`probs_<formato>.npy`, filas en el orden de `test.csv`, "
            "columnas en el orden de `tesis.config.CLASSES`) y sus métricas. Las del experimento A traen además:", "",
            "- `val_probs_<formato>.npy`: predicciones sobre la validación (orden de `val.csv`), con las que se "
            "calibra el umbral de la decisión del verificador (`tesis.metrics.calibrar_umbral`). La prueba no "
            "interviene en ese umbral.",
            "- `telefono_emulado_<formato>.npy`, donde se corrió `python -m tesis.fotos_app --emular <run>`: la prueba "
            "con el preprocesamiento del teléfono emulado en Python (orden de `test.csv`). No es una medición en el "
            "teléfono.",
            "- `int8_diagnostico.csv` (cada variante de cuantización) e `int8_capas.csv` (error por tensor), donde se "
            "corrió `python -m tesis.quantize <run> --diagnostico`. Ese diagnóstico se midió sobre fotos de la prueba "
            "y solo describe: no se eligió ninguna configuración con él.", "",
            "`dispositivo/`: salidas originales de `python -m tesis.device` en el Honor X6b. `dispositivo*.csv` es el "
            "modelo solo; `fotos_app_*` y `fotos_camara_*`, el prototipo de extremo a extremo con fotos como las guarda "
            "ODK y a resolución de cámara. Cada ráfaga registra si el teléfono estaba sin conexión. "
            "`fotos_diferencias.csv` (`python -m tesis.fotos_app --diferencias`) compara, foto por foto, lo que dio el "
            "teléfono con la evaluación y con cuatro variantes del preprocesamiento.", "",
            "## Cómo comprobar los resultados sin las fotos", "", "```",
            "pip install numpy pandas scikit-learn", "cd codigo",
            "python -m tesis.evidencia --tabla ..    # recalcula cada fila desde corridas/*/probs_*.npy", "```", "",
            "El comando recalcula la tabla completa: calibra el umbral de cada corrida con su validación, evalúa cada "
            "formato sobre la prueba y escribe `resultados_recalculados.csv` junto a este archivo. Al final compara "
            "cifra por cifra con `resultados.csv`, la tabla con que se armó el paquete y de la que salen las cifras "
            "del README, y con el `metrics_<formato>.json` que escribió cada evaluación al terminar de entrenar.", "",
            "La tabla usa `datos/convenios.csv`, que sale de las maestras de muebles de Industrias Ales "
            "(`tesis.convenios`), no de las etiquetas de las fotos. Cada fila es una corrida y un formato; las columnas "
            "están descritas en `tesis.metrics.evaluate_run` y `tesis.metrics.verification`.", "",
            "## Pruebas unitarias", "", "```", "pip install pytest pillow", "cd codigo", "python -m pytest tests", "```",
            "", "Una de ellas compila `Clasificador.decidir`, la decisión del prototipo Android "
            "(`codigo/android/app/src/main/java/ec/usfq/tesis/Clasificador.java`), y comprueba que decide igual que "
            "`tesis.metrics.decide`. Necesita un JDK (`javac` en el PATH o en `JAVA_HOME`); sin él, esa prueba se "
            "salta.", ""]
    return "\n".join(out)


MODELOS = (".keras", ".tflite")


def paquete(runs=Path("runs/final"), out=None, reducido=False):
    """`reducido`: sin los modelos entrenados, que son casi todo el peso; lo demás es igual."""
    files, runs = datos(), Path(runs)
    out = Path(out or ("evidencia_tesis_reducido.zip" if reducido else "evidencia_tesis.zip"))
    conv = next((f for f in files if f.name == "convenios.csv"), None)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("LEEME.md", leeme(files, runs, reducido))
        if conv:  # lo reportado, para que --tabla compare contra ello
            z.writestr("resultados.csv", tabla(runs, pd.read_csv(conv)).to_csv(index=False))
        for f in files:
            z.write(f, f"datos/{f.name}")
        for f in sorted(runs.rglob("*")):
            rel = f.relative_to(runs)
            # saved_model es un paso intermedio de la conversión (ya va model.keras); benchmark_model, la herramienta
            if (f.is_file() and "saved_model" not in rel.parts and f.name != "benchmark_model"
                    and not (reducido and f.suffix in MODELOS)):
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
        t = tabla(base / "corridas", pd.read_csv(base / "datos/convenios.csv"))
        t.to_csv(base / "resultados_recalculados.csv", index=False)
        print(t.round(4).T.to_string())
        print("\n" + "\n".join(comprobar(t, base)))
    else:
        args = [a for a in sys.argv[1:] if a != "--reducido"]
        paquete(*map(Path, args[:2]), reducido="--reducido" in sys.argv)
