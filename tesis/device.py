"""Latencia y memoria en el Honor X6b (sección 5.6, protocolo de carga sostenida de la 5.7), con benchmark_model, la
herramienta oficial de TensorFlow Lite, por adb.

Es un solo archivo que solo usa la biblioteca estándar: en otro computador basta Python 3.8+ y este archivo.
1. Baja de MyDrive/tesis/runs_odk las carpetas A_* (bastan los .tflite) y descomprímelas en una carpeta, p. ej. modelos/.
2. En el teléfono: Opciones de desarrollador → Depuración USB y Pantalla activa. Conéctalo por USB, acepta este
   computador y abre ODK Collect. (En Windows, si el teléfono no aparece, instala el controlador USB de Honor, HiSuite.)
3. python device.py modelos              (en el repo: python -m tesis.device runs/odk)  ≈ 30 min
   python device.py modelos --hilos 2
Si adb no está instalado, lo baja de Google a <carpeta>/platform-tools; benchmark_model se guarda en <carpeta>.

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
import csv
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/"
       "android_aarch64_benchmark_model")
URL_ADB = "https://dl.google.com/android/repository/platform-tools-latest-{}.zip"
REMOTO = "/data/local/tmp/tesis"
ODK = "org.odk.collect.android"
RAFAGAS, N = 4, 30
ADB = "adb"


def adb(*args, check=True):
    r = subprocess.run([ADB, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode:
        raise RuntimeError(f"adb {' '.join(args)}:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return r


def adb_local(carpeta):
    """adb del sistema o, si no hay, el de las platform-tools de Google bajadas a <carpeta>."""
    if shutil.which("adb"):
        return "adb"
    exe = carpeta / "platform-tools" / ("adb.exe" if os.name == "nt" else "adb")
    if not exe.exists():
        so = {"win32": "windows", "darwin": "darwin"}.get(sys.platform, "linux")
        print(f"Bajando adb (platform-tools-latest-{so})…")
        zip_ = carpeta / "platform-tools.zip"
        urllib.request.urlretrieve(URL_ADB.format(so), zip_)
        with zipfile.ZipFile(zip_) as z:
            z.extractall(carpeta)
        zip_.unlink()
        exe.chmod(0o755)  # zipfile no conserva el permiso de ejecución en macOS y Linux
    return str(exe)


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
    """Una fila por formato a partir de las filas de ráfagas."""
    grupos = {}
    for r in raf:
        grupos.setdefault((r["run"], r["formato"]), []).append(r)
    return [{"run": run, "formato": fmt,
             "lat_p50_ms": statistics.median(r["mediana_ms"] for r in g), "lat_p95_ms": max(r["p95_ms"] for r in g),
             "mem_mb": max(r["mem_mb"] for r in g), "primera_ms": statistics.mean(r["primera_ms"] for r in g),
             "trigesima_ms": statistics.mean(r["trigesima_ms"] for r in g),
             "init_ms": statistics.mean(r["init_ms"] for r in g), "rafagas": len(g),
             "xnnpack": all(r["xnnpack"] for r in g), "odk_activa": all(r["odk_activa"] for r in g),
             "pantalla": all(r["pantalla"] for r in g)}
            for (run, fmt), g in sorted(grupos.items())]


def escribir(filas, ruta):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def main():
    global ADB
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("carpeta", type=Path, help="carpeta con los runs bajados de Drive (A_*/*.tflite, a cualquier nivel)")
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--pausa", type=float, default=15, help="segundos entre ráfagas")
    a = ap.parse_args()
    # A cualquier nivel: descomprimir el zip de Drive suele dejar una carpeta de más.
    modelos = sorted(m for m in a.carpeta.rglob("*.tflite") if m.parent.name.startswith("A_"))
    if not modelos:
        sys.exit(f"No hay A_*/*.tflite dentro de {a.carpeta}")
    ADB = adb_local(a.carpeta)
    if adb("get-state", check=False).stdout.strip() != "device":
        sys.exit("No hay teléfono: conéctalo por USB, activa la depuración USB y acepta este computador en el teléfono")
    telefono = adb("shell", "getprop", "ro.product.model").stdout.strip()
    binario = a.carpeta / "benchmark_model"
    if not binario.exists():
        urllib.request.urlretrieve(URL, binario)
    adb("shell", "mkdir", "-p", REMOTO)
    adb("push", str(binario), f"{REMOTO}/benchmark_model")
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
    escribir(filas, a.carpeta / "dispositivo_rafagas.csv")
    res = resumen(filas)
    escribir(res, a.carpeta / "dispositivo.csv")
    adb("shell", "rm", "-r", REMOTO)
    for r in res:
        print(f"{r['run']:>22} {r['formato']:<14} p50 {r['lat_p50_ms']:7.1f} ms  p95 {r['lat_p95_ms']:7.1f} ms  "
              f"{r['mem_mb']:6.1f} MB{'' if r['xnnpack'] else '  sin XNNPACK'}")
    print(f"\nSube {a.carpeta / 'dispositivo.csv'} a MyDrive/tesis/runs_odk y vuelve a correr entrenar_tesis.")


if __name__ == "__main__":
    main()
