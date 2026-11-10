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
Mide solo la inferencia, con entrada aleatoria y XNNPACK en la CPU. Escribe <carpeta>/dispositivo_rafagas.csv y
<carpeta>/dispositivo.csv: este último se sube a runs_odk en Drive y entrenar_tesis lo lleva a la tabla 5.7.

De extremo a extremo, con el prototipo (A8, android/):
    python device.py modelos --app app-release.apk --fotos fotos_app
Instala el APK y repite el protocolo con fotos reales en lugar de entrada aleatoria: cada ráfaga, en un proceso nuevo
del prototipo (Rafaga.java), lee N fotos de disco, las endereza y reduce como el entrenamiento y las clasifica. El p95
de cada ráfaga es el de rango más cercano (en 30, solo la más lenta lo supera) y la memoria, el pico del proceso de la
app (VmHWM). Escribe app_rafagas.csv, app_fotos.csv (tiempos y probabilidades por foto, para tesis.fotos_app) y
dispositivo_app.csv, con el preprocesamiento, el tamaño del modelo y del runtime en el paquete y el del APK de ODK
Collect instalado. Las fotos se borran del teléfono al terminar.
"""
import argparse
import csv
import math
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
PAQUETE = "ec.usfq.tesis"  # el prototipo
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


def p95(t):
    return sorted(t)[math.ceil(0.95 * len(t)) - 1]  # rango más cercano


def parse_app(salida):
    """Una ráfaga del prototipo (rafaga.csv de Rafaga.java): sus tiempos como los de parse y las filas por foto; None
    si no corrió."""
    lineas = salida.splitlines()
    pie = dict(re.findall(r"(\w+)=([^,]+)", next((l for l in lineas if l.startswith("#")), "")))
    fotos = list(csv.DictReader(l for l in lineas if not l.startswith("#")))
    if "error" in pie or "carga_ms" not in pie or len(fotos) != N:
        return None, fotos
    t = [float(f["total_ms"]) for f in fotos]
    return {"primera_ms": t[0], "trigesima_ms": t[-1], "mediana_ms": statistics.median(t), "p95_ms": p95(t),
            "max_ms": max(t), "init_ms": float(pie["carga_ms"]), "mem_mb": int(pie["pico_kb"]) / 1024,
            "preproc_ms": statistics.median(float(f["leer_ms"]) + float(f["reducir_ms"]) for f in fotos)}, fotos


def rafaga_app(remoto, desde, n, hilos):
    """Rafaga.java en un proceso nuevo del prototipo; devuelve su rafaga.csv."""
    adb("shell", "am", "force-stop", PAQUETE)
    adb("shell", "rm", "-f", f"{remoto}/rafaga.csv")
    adb("shell", "am", "start", "-n", f"{PAQUETE}/.Rafaga", "--ei", "desde", str(desde), "--ei", "n", str(n),
        "--ei", "hilos", str(hilos))
    for _ in range(600):
        r = adb("shell", "cat", f"{remoto}/rafaga.csv", check=False)
        if r.returncode == 0:
            return r.stdout
        time.sleep(1)
    raise RuntimeError("el prototipo no terminó la ráfaga en 10 min (¿pantalla bloqueada?)")


def condiciones():
    """ODK Collect abierto y pantalla encendida, como en campo (sección 5.7); avisa si no."""
    odk = bool(adb("shell", "pidof", ODK, check=False).stdout.strip())
    pantalla = "mWakefulness=Awake" in adb("shell", "dumpsys", "power").stdout
    avisos = [t for t, ok in (("ODK Collect cerrado", odk), ("pantalla apagada", pantalla)) if not ok]
    if avisos:
        print("  ojo:", ", ".join(avisos))
    return odk, pantalla


def formato(m):  # la fila float32 de entrenar_tesis se evalúa con model.keras; al teléfono va model_fp32.tflite
    return "model" if m.stem == "model_fp32" else m.stem


def medir_app(a, telefono, modelos):
    with zipfile.ZipFile(a.app) as z:
        modelo = z.read("assets/modelo.tflite")
        runtime = sum(i.compress_size for i in z.infolist() if i.filename.startswith("lib/"))
    run, fmt = next(((m.parent.name, formato(m)) for m in modelos
                     if m.stat().st_size == len(modelo) and m.read_bytes() == modelo), ("?", "?"))
    fotos = sorted(f for f in a.fotos.iterdir() if f.suffix.lower() in (".jpg", ".jpeg"))
    if not fotos:
        sys.exit(f"No hay fotos .jpg en {a.fotos}")
    adb("install", "-r", str(a.app))
    remoto = f"/sdcard/Android/data/{PAQUETE}/files"
    rafaga_app(remoto, 0, 0, a.hilos)  # sin fotos no mide nada, pero crea la carpeta de la app
    adb("shell", "rm", "-rf", f"{remoto}/fotos")
    adb("shell", "mkdir", "-p", f"{remoto}/fotos")
    for f in fotos:
        adb("push", str(f), f"{remoto}/fotos/{f.name}")
    print(f"{telefono}: {run}/{fmt} en el prototipo, {len(fotos)} fotos, {RAFAGAS} ráfagas de {N}, {a.hilos} hilos")

    filas, por_foto = [], []
    for k in range(RAFAGAS):
        odk, pantalla = condiciones()
        salida = rafaga_app(remoto, k * N, N, a.hilos)
        r, fs = parse_app(salida)
        if r is None:
            raise RuntimeError(f"ráfaga {k + 1}: el prototipo falló\n{salida[-3000:]}")
        filas.append({"run": run, "formato": fmt, "rafaga": k + 1, **r, "xnnpack": True, "odk_activa": odk,
                      "pantalla": pantalla, "hilos": a.hilos, "telefono": telefono})
        por_foto += [{"rafaga": k + 1, **f} for f in fs]
        print(f"ráfaga {k + 1}/{RAFAGAS}: mediana {r['mediana_ms']:.0f} ms (preprocesamiento {r['preproc_ms']:.0f}), "
              f"p95 {r['p95_ms']:.0f} ms, {r['mem_mb']:.0f} MB")
        time.sleep(a.pausa)
    adb("shell", "rm", "-rf", f"{remoto}/fotos")  # las fotos de los locales no se quedan en el teléfono
    odk_b = sum(int(adb("shell", "stat", "-c", "%s", p[len("package:"):]).stdout)  # base y divisiones del APK
                for p in adb("shell", "pm", "path", ODK, check=False).stdout.split())
    [res] = resumen(filas)
    res.update({"preproc_ms": statistics.median(float(f["leer_ms"]) + float(f["reducir_ms"]) for f in por_foto),
                "modelo_mb": len(modelo) / 2**20, "runtime_mb": runtime / 2**20,
                "apk_mb": a.app.stat().st_size / 2**20, "odk_apk_mb": odk_b / 2**20})
    escribir(filas, a.carpeta / "app_rafagas.csv")
    escribir(por_foto, a.carpeta / "app_fotos.csv")
    escribir([res], a.carpeta / "dispositivo_app.csv")
    print(f"\n{run}/{fmt} de extremo a extremo: p50 {res['lat_p50_ms']:.0f} ms, p95 {res['lat_p95_ms']:.0f} ms "
          f"(preprocesamiento {res['preproc_ms']:.0f} ms), {res['mem_mb']:.0f} MB. Paquete: modelo "
          f"{res['modelo_mb']:.1f} MB + runtime {res['runtime_mb']:.1f} MB"
          + (f" = {(len(modelo) + runtime) / odk_b:.0%} del APK de ODK Collect ({res['odk_apk_mb']:.0f} MB)" if odk_b else ""))
    print(f"Sube {a.carpeta / 'dispositivo_app.csv'} y app_fotos.csv a MyDrive/tesis/runs_odk.")


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
    ap.add_argument("--app", type=Path, help="APK del prototipo: de extremo a extremo, en lugar de benchmark_model")
    ap.add_argument("--fotos", type=Path, help="con --app: carpeta de fotos .jpg (python -m tesis.fotos_app)")
    a = ap.parse_args()
    if a.app and not a.fotos:
        ap.error("--app necesita --fotos")
    # A cualquier nivel: descomprimir el zip de Drive suele dejar una carpeta de más.
    modelos = sorted(m for m in a.carpeta.rglob("*.tflite") if m.parent.name.startswith("A_"))
    if not modelos and not a.app:  # con --app solo sirven para saber qué formato lleva el APK
        sys.exit(f"No hay A_*/*.tflite dentro de {a.carpeta}")
    ADB = adb_local(a.carpeta)
    if adb("get-state", check=False).stdout.strip() != "device":
        sys.exit("No hay teléfono: conéctalo por USB, activa la depuración USB y acepta este computador en el teléfono")
    telefono = adb("shell", "getprop", "ro.product.model").stdout.strip()
    if a.app:
        return medir_app(a, telefono, modelos)
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
            odk, pantalla = condiciones()
            xnnpack = True
            r, salida = rafaga(remoto[m], a.hilos)
            if r is None:  # XNNPACK no prepara el MobileNetV3 int8 (como en evaluate.predict): kernels estándar
                xnnpack = False
                r, salida = rafaga(remoto[m], a.hilos, xnnpack=False)
            if r is None:
                raise RuntimeError(f"{m}: benchmark_model falló\n{salida[-3000:]}")
            filas.append({"run": m.parent.name, "formato": formato(m),
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
