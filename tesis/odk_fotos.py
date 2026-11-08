"""Descarga de fotos de ODK Central (solo lectura): candidatas reales a capturas inválidas (sección 5.3).

    python -m tesis.odk_fotos promociones_mercaderista sos_participacion_percha_moderno sos_visibilidad_plan_ninja

Credenciales: ODK_URL, ODK_USER y ODK_PASSWORD del entorno o, si faltan, del .env del pipeline (ENV). No se
copian a ningún lado. Salida: data/odk/<formulario>/<envío>/<foto> y data/odk/<formulario>/envios.json con los
datos de cada envío y data/odk/<formulario>/fotos.csv con una fila por foto (campo, cadena, local, fecha), para
saber qué muestra y ubicarla en la partición por local. Reanudable: lo ya bajado se salta, porque los adjuntos de
Central no cambian.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ENV = Path(r"C:\Users\sebas\Desktop\Nueva carpeta\Prueba\.env")
OUT = Path("data/odk")
PROJECT = 1  # Industrias Ales
IMG = (".jpg", ".jpeg", ".png", ".webp")
STORE_FIELDS = {  # formulario -> (campo de cadena, campo de local) en sus datos
    "promociones_mercaderista": ("grp_local/cadena_label", "grp_local/local_nombre"),
    "sos_participacion_percha_moderno": ("local_datos/cadena_grupo", "local_datos/local_nombre"),
    "sos_visibilidad_plan_ninja": ("datos_local/local_cadena", "datos_local/local_nombre"),
}


def credentials():
    env = {}
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("\"'")
    get = lambda k: os.getenv(k) or env.get(k)
    return get("ODK_URL").rstrip("/") + "/v1", get("ODK_USER"), get("ODK_PASSWORD")


def session():
    base, user, password = credentials()
    r = requests.post(f"{base}/sessions", json={"email": user, "password": password}, timeout=30)
    r.raise_for_status()
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {r.json()['token']}"
    return base, s


def download_form(base, s, form):
    url = f"{base}/projects/{PROJECT}/forms/{form}"
    out = OUT / form
    out.mkdir(parents=True, exist_ok=True)
    data = s.get(f"{url}.svc/Submissions", timeout=120).json()["value"]  # OData: los datos de cada envío
    (out / "envios.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def one(iid):
        got = 0
        for a in s.get(f"{url}/submissions/{iid}/attachments", timeout=60).json():
            dst = out / iid.replace("uuid:", "") / a["name"]
            if not a.get("exists") or dst.exists() or not a["name"].lower().endswith(IMG):
                continue  # también salta audit.csv, el registro de pantallas que ODK adjunta a cada envío
            r = s.get(f"{url}/submissions/{iid}/attachments/{a['name']}", timeout=120)
            r.raise_for_status()
            dst.parent.mkdir(exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.replace(dst)  # un corte a mitad no deja una foto truncada que parezca completa
            got += 1
        return got

    with ThreadPoolExecutor(8) as ex:
        new = sum(ex.map(one, [d["__id"] for d in data]))
    total = sum(1 for f in out.rglob("*") if f.suffix.lower() in IMG)
    print(f"{form}: {len(data)} envíos · {new} fotos nuevas · {total} en {out}", flush=True)
    photo_table(base, s, form, data)


def flat(x, p=""):
    for k, v in x.items():
        yield from flat(v, f"{p}{k}/") if isinstance(v, dict) else [(p + k, v)]


def photo_table(base, s, form, data):
    """data/odk/<form>/fotos.csv: una fila por foto con el campo del formulario que la pidió (dice qué muestra)
    y el local, la cadena y la fecha del envío. Los grupos repetidos viven en tablas OData aparte."""
    url = f"{base}/projects/{PROJECT}/forms/{form}.svc"
    chain_f, store_f = STORE_FIELDS[form]
    sub = {d["__id"]: dict(flat(d)) for d in data}
    rows = []
    for es in (e["url"] for e in s.get(url, timeout=60).json()["value"]):
        for row in s.get(f"{url}/{es}", timeout=120).json()["value"]:
            f = dict(flat(row))
            iid = f.get("__Submissions-id") or f["__id"]  # en una tabla de repetición, el envío padre
            for field, v in f.items():
                if isinstance(v, str) and v.lower().endswith(IMG):
                    meta = sub[iid]
                    rows.append({"path": (OUT / form / iid.replace("uuid:", "") / v).as_posix(), "form": form,
                                 "campo": f"{es.split('.')[-1]}:{field}", "cadena": meta.get(chain_f) or "",
                                 "local": meta.get(store_f) or "", "fecha": meta.get("fecha") or ""})
    pd.DataFrame(rows).to_csv(OUT / form / "fotos.csv", index=False)
    print(f"  {len(rows)} fotos con su campo -> {OUT / form / 'fotos.csv'}")


if __name__ == "__main__":
    base, s = session()
    for form in sys.argv[1:]:
        download_form(base, s, form)
