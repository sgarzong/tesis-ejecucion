"""Avance de un entrenamiento en curso (tesis.train) a partir de su log y su history.csv.

    python -m tesis.progress runs/A_mobilenetv3            # una vez
    python -m tesis.progress runs/A_mobilenetv3 --watch    # se actualiza cada minuto (Ctrl+C para salir)

La parada temprana hace que el final no se conozca de antemano: se informa el avance contra el tope de
épocas y contra el final más temprano posible (PATIENCE épocas sin mejorar val_loss desde la mejor).
"""
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

MAX_EPOCHS, PATIENCE = 40, 6  # los de tesis.train


def report(run):
    log = (run / "train.log").read_text(encoding="utf-8", errors="ignore")
    steps = re.findall(r"(\d+)/(\d+)\D", re.sub(r"\x1b\[[0-9;]*m", "", log)[-2000:])
    step, total = map(int, steps[-1]) if steps else (0, 1)
    hist = pd.read_csv(run / "history.csv") if (run / "history.csv").exists() else pd.DataFrame()
    done = len(hist)
    if (run / "model.keras").exists():
        return f"Terminado: {done} épocas · mejor val_accuracy {hist.val_accuracy.max():.3f} · {run / 'model.keras'}"
    now = done + step / total
    elapsed = datetime.now() - datetime.fromtimestamp((run / "train.log").stat().st_ctime)
    per_epoch = elapsed / max(now, 1e-6)
    since_best = done - 1 - int(hist.val_loss.idxmin()) if done else 0
    earliest = min(MAX_EPOCHS, done + PATIENCE - since_best) if done else PATIENCE + 1
    eta = lambda n: (datetime.now() + per_epoch * (n - now)).strftime("%a %H:%M")
    bar = lambda f: "#" * round(f * 30) + "-" * (30 - round(f * 30))
    lines = [f"Época {done + 1}, paso {step}/{total} ({step / total:.0%}) · {per_epoch.seconds // 60} min por época"]
    if done:
        b = hist.loc[hist.val_loss.idxmin()]
        lines.append(f"Mejor hasta ahora: época {int(b.epoch) + 1}, val_accuracy {b.val_accuracy:.3f}"
                     f" · última {hist.val_accuracy.iloc[-1]:.3f} · {since_best} épocas sin mejorar")
    lines += [f"[{bar(now / earliest)}] {now / earliest:5.1%}  si deja de mejorar ya: termina en la época {earliest}"
              f" (~{eta(earliest)})",
              f"[{bar(now / MAX_EPOCHS)}] {now / MAX_EPOCHS:5.1%}  si mejora hasta el tope: época {MAX_EPOCHS}"
              f" (~{eta(MAX_EPOCHS)})"]
    return "\n".join(lines)


if __name__ == "__main__":
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/A_mobilenetv3")
    while True:
        print(f"\n{datetime.now():%H:%M} · {run}\n{report(run)}", flush=True)
        if "--watch" not in sys.argv or (run / "model.keras").exists():
            break
        time.sleep(60)
