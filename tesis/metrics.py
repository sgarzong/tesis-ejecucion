"""Métricas predictivas, verificación contra convenio y viabilidad (secciones 5.6–5.9).

`probs`: matriz (n, 7) en el orden de config.CLASSES. `y`: índices de clase verdadera.
`contracts`: dict clave -> set de nombres de clase con convenio vigente; la clave de cada foto la da contract_keys.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, recall_score

from .config import CLASSES, REJECT, THRESHOLDS, VENTANA_CONVENIO

IDX = {c: i for i, c in enumerate(CLASSES)}
REJECT_IDX = [IDX[c] for c in REJECT]
MUEBLES = [c for c in CLASSES if c not in REJECT]


def contracts(conv):
    """Convenio de cada (local, fuente) según data/convenios.csv (tesis.convenios): sale de las maestras de
    Industrias Ales, no de las etiquetas de las fotos."""
    return conv.groupby(["store", "fuente"]).clase.agg(set).to_dict()


def contract_keys(df, conv):
    """Clave (local, fuente) del convenio que rige cada foto de `df`: la fuente fechada a menos de VENTANA_CONVENIO
    días de la foto. None si no hay ninguna o la foto no tiene fecha: esa foto queda fuera de la verificación."""
    dates = pd.to_datetime(df.date, errors="coerce")
    keys = np.full(len(df), None, dtype=object)
    for fuente, fecha in conv.drop_duplicates("fuente")[["fuente", "fecha"]].itertuples(index=False):
        for i in np.flatnonzero(((dates - pd.Timestamp(fecha)).abs() <= pd.Timedelta(days=VENTANA_CONVENIO)).values):
            keys[i] = (df.store.iloc[i], fuente)
    return keys


def wilson(k, n, z=1.96):
    """Intervalo de Wilson al 95 % de la proporción k/n: sirve con n chico y cerca de 0 o 1."""
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float((p + z * z / (2 * n)) / d - half), float((p + z * z / (2 * n)) / d + half)


def conditional_pred(probs, stores, contracts):
    """Argmax restringido a los muebles del local más las dos categorías de rechazo."""
    mask = np.zeros_like(probs, dtype=bool)
    for i, s in enumerate(stores):
        mask[i, [IDX[c] for c in contracts.get(s, CLASSES)] + REJECT_IDX] = True
    return np.where(mask, probs, -np.inf).argmax(1)


def decide(probs, stores, contracts, tau=np.inf):
    """Decisión del verificador. La clase más probable se conserva si es un rechazo, un mueble del convenio o un
    mueble que el convenio no lista con probabilidad ≥ `tau`: hay evidencia de que la foto muestra otro mueble. Si
    no, se decide dentro del convenio. `tau` sale de la validación (calibrar_umbral). Con tau=inf es
    conditional_pred, que no puede ver un mueble fuera del convenio y lo hace pasar por uno contratado; con tau=0,
    el argmax sin restringir."""
    free, cond = probs.argmax(1), conditional_pred(probs, stores, contracts)
    return np.where((free != cond) & (probs.max(1) >= tau), free, cond)


def pares_fuera_convenio(y, stores, contracts):
    """Pares (imagen, mueble declarado) que la restricción al convenio no deja ver: la foto muestra un mueble que el
    convenio de su local no lista y se declara cada uno de los que sí lista. Todos son discrepantes. Incluye los
    locales con un solo mueble en el convenio, que verification_pairs deja fuera."""
    img, declared = [], []
    for i, (label, s) in enumerate(zip(y, stores)):
        own = contracts.get(s)
        if own is None or label in REJECT_IDX or CLASSES[label] in own:
            continue
        for c in own:
            img.append(i)
            declared.append(IDX[c])
    return np.array(img, dtype=int), np.array(declared, dtype=int)


def calibrar_umbral(probs, y, stores, contracts):
    """Umbral de `decide`, con fotos de validación: el que maximiza la detección de un mueble fuera del convenio
    (pares_fuera_convenio) menos las falsas alarmas (fotos que muestran un mueble de su convenio, declarado tal
    cual), es decir, la J de Youden, sobre una rejilla de centésimas. En empate, el mayor. inf si no hay con qué."""
    ok = np.array([s in contracts and CLASSES[c] in contracts[s] for c, s in zip(y, stores)])
    img, dec = pares_fuera_convenio(y, stores, contracts)
    if not ok.any() or not len(img):
        return float("inf")
    free, cond, pmax = probs.argmax(1), conditional_pred(probs, stores, contracts), probs.max(1)
    grid = np.arange(101) / 100
    j = []
    for tau in grid:
        d = np.where((free != cond) & (pmax >= tau), free, cond)
        j.append((d[img] != dec).mean() - (d[ok] != y[ok]).mean())
    return float(grid[np.flatnonzero(np.array(j) >= max(j) - 1e-12)[-1]])


def report(y, probs, stores=None, contracts=None, labels=None):
    """`labels`: clases sobre las que promedian F1-macro y recall-macro (todas). El experimento B pasa las que su
    prueba tiene: una clase sin fotos aporta un cero al promedio que no dice nada del modelo."""
    pred = probs.argmax(1)
    every = list(range(len(CLASSES)))
    labels = every if labels is None else list(labels)
    p, r, _, n = precision_recall_fscore_support(y, pred, labels=every, zero_division=0)
    out = {
        "f1_macro": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, labels=labels, average="macro", zero_division=0),
        "per_class": {c: {"precision": p[i], "recall": r[i], "support": int(n[i]),
                          "recall_ic95": wilson(round(r[i] * n[i]), int(n[i]))} for i, c in enumerate(CLASSES)},
        # El umbral se exige a cada rechazo; sin fotos de alguno no hay con qué medirlo (NaN incumple)
        "reject_recall": min(r[i] if n[i] else float("nan") for i in REJECT_IDX),
        "confusion": confusion_matrix(y, pred, labels=every, normalize="true").round(3).tolist(),
    }
    if contracts is not None:
        # Lo que el formulario puede recibir en un local con convenio fechado: uno de sus muebles o un rechazo.
        # Los muebles que la maestra no lista (los ganados en negociación) no tienen respuesta correcta posible.
        own = [contracts.get(s) for s in stores]
        con = np.array([o is not None and (CLASSES[c] in o or c in REJECT_IDX) for c, o in zip(y, own)])
        out["conditional_accuracy"] = float((conditional_pred(probs, stores, contracts)[con] == y[con]).mean())
        out["accuracy_con_convenio"] = float((pred[con] == y[con]).mean())  # mismas fotos, sin restringir
        out["conditional_n"] = int(con.sum())
    return out


def bootstrap_f1_diff(y, pred_a, pred_b, stores, n=2000, seed=0, labels=None):
    """IC 95 % de F1-macro(a) - F1-macro(b) con remuestreo por local. `labels`: clases que promedia (todas)."""
    rng = np.random.default_rng(seed)
    stores = np.asarray(stores)
    groups = [np.flatnonzero(stores == s) for s in np.unique(stores)]
    labels = list(range(len(CLASSES))) if labels is None else labels
    diffs = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        diffs.append(f1_score(y[idx], pred_a[idx], labels=labels, average="macro", zero_division=0)
                     - f1_score(y[idx], pred_b[idx], labels=labels, average="macro", zero_division=0))
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def verification_pairs(y, stores, contracts):
    """Pares (imagen, mueble declarado) por intercambio dentro del local (sección 5.9).

    Solo locales con 2+ convenios e imágenes de mueble con convenio en ese local.
    Devuelve listas paralelas: índice de imagen, índice declarado, es_correcto.
    """
    img, declared, correct = [], [], []
    for i, (label, s) in enumerate(zip(y, stores)):
        own = contracts.get(s, set())
        if len(own) < 2 or CLASSES[label] not in own:
            continue
        for c in own:
            img.append(i)
            declared.append(IDX[c])
            correct.append(IDX[c] == label)
    return np.array(img), np.array(declared), np.array(correct)


def verification(probs, y, stores, contracts, tau=np.inf):
    """Tasa de detección sobre pares discrepantes y de falsas alarmas sobre pares correctos, con la decisión de
    `decide` y su umbral `tau`. Las `_restringida` y `_libre` son las mismas con la predicción restringida al
    convenio y sin restringir. `fuera_*` mide lo que la restricción no deja ver (pares_fuera_convenio): un mueble
    que el convenio no lista, declarado como uno que sí."""
    img, declared, correct = verification_pairs(y, stores, contracts)
    if not len(img):  # ninguna foto con convenio fechado (NaN incumple el criterio de falsas alarmas)
        return {"detection_rate": float("nan"), "false_alarm": float("nan"), "n_mismatch": 0, "n_correct": 0}
    out_img, out_dec = pares_fuera_convenio(y, stores, contracts)
    out = {"n_mismatch": int((~correct).sum()), "n_correct": int(correct.sum()),
           "fuera_n": len(out_img), "fuera_fotos": len(set(out_img))}
    for sufijo, t in (("", tau), ("_restringida", np.inf), ("_libre", 0)):
        d = decide(probs, stores, contracts, t)
        alert = d[img] != declared  # cualquier alerta: discrepancia o rechazo
        out |= {("detection_rate" if not sufijo else "detection" + sufijo): float(alert[~correct].mean()),
                "false_alarm" + sufijo: float(alert[correct].mean()),
                "fuera_detection" + sufijo: float((d[out_img] != out_dec).mean()) if len(out_img) else float("nan")}
    return out


def evaluate_run(test, probs, conv, labels=None, tau=np.inf):
    """Fila de resultados de un run y formato sobre su prueba (`test`: filas del manifiesto, `probs` alineadas):
    métricas predictivas con el n y el IC 95 % de cada rechazo, y verificación contra el convenio de `conv`
    (data/convenios.csv). `tau` es el umbral de `decide`, calibrado con la validación del run; con inf, la decisión
    es la restringida al convenio.

    La verificación se mide dos veces (sección 5.9) y son experimentos distintos: pares por intercambio sobre el
    histórico (discrepancias construidas) y lo que el mercaderista declaró en ODK (errores reales de campo). De ODK,
    `odk_*` son las fotos cuyo mueble declarado está en el convenio del local, con la predicción restringida a él;
    `odk_libre_*`, `odk_todas_restringida_*` y `odk_umbral_*`, todas las que declaran un mueble, con cada decisión
    (las maestras solo listan los muebles alquilados; los otros formularios declaran muebles que no están en ellas).
    `convenio_*` es la exactitud sobre todas las fotos de un local con convenio fechado, también las que muestran un
    mueble que el convenio no lista: en esas la decisión restringida falla por construcción."""
    y = test.label.map(IDX).values
    con, keys = contracts(conv), contract_keys(test, conv)
    r = report(y, probs, keys, con, labels)
    out = {"n": len(y), "f1_macro": r["f1_macro"], "recall_macro": r["recall_macro"],
           "reject_recall": r["reject_recall"]}
    for c in sorted(REJECT):
        pc = r["per_class"][c]
        out |= {f"{c}_n": pc["support"], f"{c}_recall": pc["recall"] if pc["support"] else float("nan"),
                f"{c}_ic95": "[{:.3f}, {:.3f}]".format(*pc["recall_ic95"])}
    out |= {k: r[k] for k in ("conditional_accuracy", "accuracy_con_convenio", "conditional_n")}
    reglas = (("", tau), ("_restringida", np.inf), ("_libre", 0))
    fechado = np.array([k in con for k in keys])
    out |= {"umbral": tau, "convenio_n": int(fechado.sum())}
    out |= {"convenio_accuracy" + s: float((decide(probs, keys, con, t)[fechado] == y[fechado]).mean())
            if fechado.any() else float("nan") for s, t in reglas}
    hist = (test.source == "hist").values
    out |= verification(probs[hist], y[hist], keys[hist], con, tau)
    odk = np.array([s == "odk" and d in con.get(k, ()) for s, d, k in zip(test.source, test.declarado, keys)])
    dec = test.declarado[odk].map(IDX).values
    alert, ok = conditional_pred(probs[odk], keys[odk], con) != dec, y[odk] == dec  # ok: la foto muestra lo declarado
    out |= {"odk_false_alarm": float(alert[ok].mean()) if ok.any() else float("nan"), "odk_n_ok": int(ok.sum()),
            "odk_detected": int(alert[~ok].sum()), "odk_n_err": int((~ok).sum())}
    todas = ((test.source == "odk") & test.declarado.isin(MUEBLES)).values
    dec = test.declarado[todas].map(IDX).values
    ok = y[todas] == dec
    out |= {"odk_libre_n_ok": int(ok.sum()), "odk_libre_n_err": int((~ok).sum())}
    for s, t in (("libre", 0), ("todas_restringida", np.inf), ("umbral", tau)):
        alert = decide(probs[todas], keys[todas], con, t) != dec
        out |= {f"odk_{s}_false_alarm": float(alert[ok].mean()) if ok.any() else float("nan"),
                f"odk_{s}_detected": int(alert[~ok].sum())}
    return out


def failed_criteria(m):
    """Criterios de la tabla 5.7 que `m` incumple. Lista vacía = configuración viable.
    `m` puede traer: f1_macro, recall_macro, reject_recall, false_alarm, quant_drop,
    lat_p50_ms, lat_p95_ms, mem_mb, size_mb. Las claves ausentes se reportan como pendientes."""
    higher = {"f1_macro", "recall_macro", "reject_recall"}
    return [k for k, t in THRESHOLDS.items()
            if k not in m or not (m[k] >= t if k in higher else m[k] <= t)]  # NaN incumple
