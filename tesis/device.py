"""Latencia y memoria en el Honor X6b (sección 5.6, protocolo de carga sostenida de la 5.7), con benchmark_model, la
herramienta oficial de TensorFlow Lite, por adb.

Uso: baja de MyDrive/tesis/runs_odk las carpetas A_* (bastan los .tflite) a runs/odk/, conecta el teléfono por USB
con la depuración USB activada, la pantalla siempre encendida (Opciones de desarrollador → Pantalla activa) y ODK
Collect abierto, y corre
    python -m tesis.device runs/odk               # mide runs/odk/A_*/*.tflite (≈ 30 min)
    python -m tesis.device runs/odk --hilos 2
Necesita adb: winget install Google.PlatformTools.

Cada formato corre RAFAGAS ráfagas de N inferencias seguidas (4 × 30 = 120 ≥ 100 ejecuciones), cada ráfaga en un proceso
nuevo: carga el modelo y la primera inferencia es en frío, como al abrir la app. Las ráfagas se intercalan entre
formatos en orden aleatorio, con una pausa entre una y otra, para que el calentamiento del teléfono no favorezca a
ninguno. Por formato:
    lat_p50_ms   mediana de las medianas de las ráfagas
    lat_p95_ms   el mayor p95 de las ráfagas: cota superior del p95 de las 120, porque en cada ráfaga solo la más
                 lenta lo supera (4 de 120 < 5 %)
    mem_mb       la mayor memoria pico del proceso
    primera_ms, trigesima_ms   promedio sobre las ráfagas: el estrangulamiento térmico
Mide solo la inferencia, con entrada aleatoria y XNNPACK en la CPU. La decodificación y el redimensionamiento de la foto
y el aumento del paquete se miden con la app (A8). Escribe <carpeta>/dispositivo_rafagas.csv y <carpeta>/dispositivo.csv:
este último se sube a runs_odk en Drive y entrenar_tesis lo lleva a la tabla 5.7.
"""
import argparse
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

URL = ("https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/"
       "android_aarch64_benchmark_model")
BINARIO = Path("data/benchmark_model")
REMOTO = "/data/local/tmp/tesis"
ODK = "org.odk.collect.android"
RAFAGAS, N = 4, 30


def adb(*args, check=True):
    r = subprocess.run(["adb", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode:
        raise RuntimeError(f"adb {' '.join(args)}:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return r


def parse(salida):
    """Una ráfaga en ms, de la salida de benchmark_model (tiempos en µs); None si no corrió."""
    stats = [dict(re.findall(r"(\w+)=([-+.\de]+)", l)) for l in salida.splitlines() if "median=" in l]
    stats = [s for s in stats if s.get("count") == str(N)]
    pico = re.search(r"peak memory footprint \(MB\) via periodic monitoring: ([\d.]+)", salida)
    init = re.search(r"Init: ([-+.\de]+)", salida)
    if not (stats and pico and init):
        return None
    s = {k: float(v) / 1000 for k, v in stats[-1].items()}
    return {"primera_ms": s["first"], "trigesima_ms": s["curr"], "mediana_ms": s["median"], "p95_ms": s["p95"],
            "max_ms": s["max"], "init_ms": float(init[1]) / 1000, "mem_mb": float(pico[1])}


def rafaga(modelo, hilos, xnnpack=True):
    r = adb("shell", f"{REMOTO}/benchmark_model", f"--graph={modelo}", f"--num_threads={hilos}", f"--num_runs={N}",
            "--min_secs=0", "--warmup_runs=0", "--warmup_min_secs=0", f"--use_xnnpack={str(xnnpack).lower()}",
            "--report_peak_memory_footprint=true", check=False)
    salida = r.stdout + r.stderr
    return (None if r.returncode else parse(salida)), salida


def resumen(raf):
    g = raf.groupby(["run", "formato"])
    return pd.DataFrame({"lat_p50_ms": g.mediana_ms.median(), "lat_p95_ms": g.p95_ms.max(), "mem_mb": g.mem_mb.max(),
                         "primera_ms": g.primera_ms.mean(), "trigesima_ms": g.trigesima_ms.mean(),
                         "init_ms": g.init_ms.mean(), "rafagas": g.size(), "xnnpack": g.xnnpack.all(),
                         "odk_activa": g.odk_activa.all(), "pantalla": g.pantalla.all()}).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("carpeta", type=Path, help="runs bajados de Drive: <carpeta>/A_*/*.tflite")
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--pausa", type=float, default=15, help="segundos entre ráfagas")
    a = ap.parse_args()
    modelos = sorted(a.carpeta.glob("A_*/*.tflite"))
    if not modelos:
        sys.exit(f"No hay {a.carpeta}/A_*/*.tflite")
    if not shutil.which("adb"):
        sys.exit("Falta adb: winget install Google.PlatformTools, y abre otra terminal")
    if adb("get-state", check=False).stdout.strip() != "device":
        sys.exit("No hay teléfono: conéctalo por USB, activa la depuración USB y acepta este PC en el teléfono")
    telefono = adb("shell", "getprop", "ro.product.model").stdout.strip()
    if not BINARIO.exists():
        urllib.request.urlretrieve(URL, BINARIO)
    adb("shell", "mkdir", "-p", REMOTO)
    adb("push", str(BINARIO), f"{REMOTO}/benchmark_model")
    adb("shell", "chmod", "755", f"{REMOTO}/benchmark_model")
    remoto = {m: f"{REMOTO}/{m.parent.name}__{m.name}" for m in modelos}
    for m in modelos:
        adb("push", str(m), remoto[m])
    print(f"{telefono}: {len(modelos)} formatos × {RAFAGAS} ráfagas de {N}, {a.hilos} hilos")

    filas = []
    for k in range(RAFAGAS):
        for m in random.Random(k).sample(modelos, len(modelos)):
            odk = bool(adb("shell", "pidof", ODK, check=False).stdout.strip())
            pantalla = "mWakefulness=Awake" in adb("shell", "dumpsys", "power").stdout
            avisos = [t for t, ok in (("ODK Collect cerrado", odk), ("pantalla apagada", pantalla)) if not ok]
            if avisos:
                print("  ojo:", ", ".join(avisos))
            xnnpack = True
            r, salida = rafaga(remoto[m], a.hilos)
            if r is None:  # XNNPACK no prepara el MobileNetV3 int8 (como en evaluate.predict): kernels estándar
                xnnpack = False
                r, salida = rafaga(remoto[m], a.hilos, xnnpack=False)
            if r is None:
                raise RuntimeError(f"{m}: benchmark_model falló\n{salida[-3000:]}")
            # La fila float32 de entrenar_tesis se evalúa con model.keras; al teléfono va model_fp32.tflite.
            filas.append({"run": m.parent.name, "formato": "model" if m.stem == "model_fp32" else m.stem,
                          "rafaga": k + 1, **r, "xnnpack": xnnpack, "odk_activa": odk, "pantalla": pantalla,
                          "hilos": a.hilos, "telefono": telefono})
            print(f"ráfaga {k + 1}/{RAFAGAS} {m.parent.name}/{m.stem}: mediana {r['mediana_ms']:.0f} ms, "
                  f"p95 {r['p95_ms']:.0f} ms, {r['mem_mb']:.0f} MB{'' if xnnpack else ' (sin XNNPACK)'}")
            time.sleep(a.pausa)
    raf = pd.DataFrame(filas)
    raf.to_csv(a.carpeta / "dispositivo_rafagas.csv", index=False)
    res = resumen(raf)
    res.to_csv(a.carpeta / "dispositivo.csv", index=False)
    adb("shell", "rm", "-r", REMOTO)
    print(res.round(1).to_string(index=False))
    print(f"\nSube {a.carpeta / 'dispositivo.csv'} a MyDrive/tesis/runs_odk y vuelve a correr entrenar_tesis.")


if __name__ == "__main__":
    main()
