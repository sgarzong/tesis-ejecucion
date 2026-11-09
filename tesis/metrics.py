"""Métricas predictivas, verificación contra convenio y viabilidad (secciones 5.6–5.9).

`probs`: matriz (n, 7) en el orden de config.CLASSES. `y`: índices de clase verdadera.
`contracts`: dict store -> set de nombres de clase con convenio vigente.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, recall_score

from .config import CLASSES, REJECT, THRESHOLDS

IDX = {c: i for i, c in enumerate(CLASSES)}
REJECT_IDX = [IDX[c] for c in REJECT]
MUEBLES = [c for c in CLASSES if c not in REJECT]


def contracts(manifest):
    """Muebles con convenio por local, a falta de la maestra de convenios: los que aparecen en las fotos verificadas
    del local y los que ODK declara para él (el formulario solo despliega los contratados). Es un superconjunto del
    convenio: restringe la predicción menos que la maestra, no más."""
    out = {}
    for s, label, declared in manifest[manifest.verified == 1][["store", "label", "declarado"]].itertuples(index=False):
        for c in {label, declared} & set(MUEBLES):
            out.setdefault(s, set()).add(c)
    return out


def conditional_pred(probs, stores, contracts):
    """Argmax restringido a los muebles del local más las dos categorías de rechazo."""
    mask = np.zeros_like(probs, dtype=bool)
    for i, s in enumerate(stores):
        mask[i, [IDX[c] for c in contracts.get(s, CLASSES)] + REJECT_IDX] = True
    return np.where(mask, probs, -np.inf).argmax(1)


def report(y, probs, stores=None, contracts=None):
    pred = probs.argmax(1)
    labels = list(range(len(CLASSES)))
    p, r, _, n = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    out = {
        "f1_macro": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, labels=labels, average="macro", zero_division=0),
        "per_class": {c: {"precision": p[i], "recall": r[i], "support": int(n[i])} for i, c in enumerate(CLASSES)},
        "reject_recall": min(r[i] for i in REJECT_IDX),
        "confusion": confusion_matrix(y, pred, labels=labels, normalize="true").round(3).tolist(),
    }
    if contracts is not None:
        out["conditional_accuracy"] = float((conditional_pred(probs, stores, contracts) == y).mean())
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


def verification(probs, y, stores, contracts):
    """Tasa de detección sobre pares discrepantes y de falsas alarmas sobre pares correctos."""
    img, declared, correct = verification_pairs(y, stores, contracts)
    pred = conditional_pred(probs, stores, contracts)[img]
    alert = pred != declared  # cualquier alerta: discrepancia o rechazo
    return {"detection_rate": float(alert[~correct].mean()), "false_alarm": float(alert[correct].mean()),
            "n_mismatch": int((~correct).sum()), "n_correct": int(correct.sum())}


def failed_criteria(m):
    """Criterios de la tabla 5.7 que `m` incumple. Lista vacía = configuración viable.
    `m` puede traer: f1_macro, recall_macro, reject_recall, false_alarm, quant_drop,
    lat_p50_ms, lat_p95_ms, mem_mb, size_mb. Las claves ausentes se reportan como pendientes."""
    higher = {"f1_macro", "recall_macro", "reject_recall"}
    return [k for k, t in THRESHOLDS.items()
            if k not in m or (m[k] < t if k in higher else m[k] > t)]
