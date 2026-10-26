"""Revisión rápida de la preetiqueta (tesis.prelabel): cada página muestra una sola clase sugerida.

    python -m tesis.sort_gui data/prelabel.csv data/clasificadas

Clic sobre una foto = seleccionarla (otro clic la quita); arrastrar = seleccionar varias. Luego un
botón de clase o su tecla aplica a todas las seleccionadas (Esc limpia la selección). Sin selección,
la tecla aplica a la foto bajo el mouse:
    1-7 = cambiar su clase    a = 2ª opción del modelo    z = volver a la sugerida
    x   = descartar (rota o duplicada: queda fuera del dataset)
Espacio (o Enter) confirma la página: las no tocadas quedan con la clase sugerida.
b o ← = página anterior (para corregir lo ya confirmado), → = siguiente sin confirmar.
Clic derecho sostenido = foto en grande.

Las confirmadas se copian a <salida>/<clase>/ (los originales no se tocan) y cada decisión se agrega a
<salida>/revision.csv (path, label, sugerida; manda la última fila de cada path), que permite reanudar.
"""
import collections
import csv
import hashlib
import itertools
import queue
import shutil
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageOps, ImageTk

from .config import CLASSES, TO_TESIS

DISCARD = "descartada"
COLS, ROWS = 8, 3
BUCKETS = (0.95, 0.7)  # dentro de cada clase: seguras primero, dudosas al final
BG, NEUTRAL, SELECTED = "#111111", "#2b2b2b", "#facc15"
COLOR = dict(zip(CLASSES, ["#2563eb", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#737373", "#a16207"]))
COLOR[DISCARD] = "#dc2626"


def copy_name(path):
    """Nombre de la copia: prefijo por ruta para que fotos homónimas de distintos locales no choquen."""
    return f"{hashlib.sha1(path.encode()).hexdigest()[:8]}_{Path(path).name}"


def place(out, path, old, label):
    """Mueve la copia de `path` de <out>/<old>/ a <out>/<label>/ (ninguna si se descarta)."""
    if old == label:
        return
    if old and old != DISCARD:
        (out / old / copy_name(path)).unlink(missing_ok=True)
    if label != DISCARD:
        (out / label).mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out / label / copy_name(path))


def paginate(rows, n):
    """Páginas de hasta n filas que nunca mezclan clases sugeridas (rows ya ordenadas por label)."""
    pages = []
    for _, g in itertools.groupby(rows, key=lambda r: r["label"]):
        g = list(g)
        pages += [g[i:i + n] for i in range(0, len(g), n)]
    return pages


def open_img(path, size):
    img = Image.open(path)
    img.draft("RGB", size)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail(size)
    return img


class App:
    def __init__(self, root, prelabel, out):
        self.root, self.out, self.log = root, out, out / "revision.csv"
        with prelabel.open(encoding="utf-8-sig") as f:
            self.rows = list(csv.DictReader(f))
        self.sugg = {r["path"]: r["label"] for r in self.rows}
        self.decided = {}
        if self.log.exists():
            with self.log.open(encoding="utf-8-sig") as f:
                self.decided = {r["path"]: r["label"] for r in csv.DictReader(f)}
        self.edit, self.cache, self.pages, self.page = {}, {}, [[]], 0
        self.hover, self.zoomwin, self.ready_at, self.armed = None, None, 0.0, True
        self.sel, self.drag_on = set(), True  # índices seleccionados en la página actual
        self.history = collections.deque(maxlen=10)  # (hora, revisadas) de las últimas páginas
        self.jobs = queue.Queue()  # copias en segundo plano: confirmar no espera al disco
        threading.Thread(target=self.copier, daemon=True).start()

        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.bh = (sh - 215) // ROWS - 36
        self.bw = min(self.bh * 3 // 4, (sw - 40) // COLS - 16)  # fotos verticales: caja 3:4
        root.configure(bg=BG)
        bar = tk.Frame(root, bg=BG)
        bar.pack(fill="x", padx=8, pady=(6, 0))
        self.banner = tk.Label(bar, font=("Segoe UI", 16, "bold"), fg="white", padx=14)
        self.banner.pack(side="left")
        tk.Label(bar, text="  Sugerida:", bg=BG, fg="#ddd").pack(side="left")
        self.cls = ttk.Combobox(bar, state="readonly", width=16, values=["(todas)", *CLASSES])
        self.cls.current(0)
        self.cls.bind("<<ComboboxSelected>>", lambda e: self.refilter())
        self.cls.pack(side="left", padx=4)
        self.pending = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="solo pendientes", variable=self.pending, command=self.refilter, takefocus=0,
                       bg=BG, fg="#ddd", selectcolor="#333", activebackground=BG).pack(side="left")
        self.status = tk.Label(bar, bg=BG, fg="#ddd", font=("Segoe UI", 10))
        self.status.pack(side="right")
        btns = tk.Frame(root, bg=BG)
        btns.pack(fill="x", padx=8, pady=(4, 0))
        hint = {"captura_invalida": " (foto mala)", "contenido_ajeno": " (no es mueble)"}
        actions = [(f"{i} {c}{hint.get(c, '')}", COLOR[c], lambda c=c: self.set_label(c))
                   for i, c in enumerate(CLASSES, 1)]
        actions += [("x descartar (rota/duplicada)", COLOR[DISCARD], lambda: self.set_label(DISCARD)),
                    ("a 2ª opción", NEUTRAL, self.second), ("z sugerida", NEUTRAL, self.revert)]
        for text, color, fn in actions:
            tk.Button(btns, text=text, command=fn, bg=color, fg="white", activebackground=color,
                      activeforeground="white", relief="flat", padx=8, takefocus=0,
                      font=("Segoe UI", 10, "bold")).pack(side="left", padx=2)
        tk.Label(root, bg=BG, fg="#aaa", anchor="w", padx=10, text=(
            "Clic o arrastre = seleccionar fotos, luego botón o tecla = asignarles la clase · sin selección, la tecla "
            "aplica a la foto bajo el mouse · Esc = limpiar selección · Espacio = confirmar página · b = atrás · "
            "clic derecho sostenido = ampliar")).pack(fill="x")

        grid = tk.Frame(root, bg=BG)
        grid.pack(pady=4)
        self.tiles = []
        for k in range(COLS * ROWS):
            t = tk.Label(grid, compound="top", width=self.bw, height=self.bh + 22, fg="white",
                         font=("Segoe UI", 10, "bold"), highlightthickness=6)
            t.grid(row=k // COLS, column=k % COLS, padx=2, pady=2)
            t.bind("<Enter>", lambda e, k=k: setattr(self, "hover", k))
            t.bind("<Leave>", lambda e: setattr(self, "hover", None))
            t.bind("<ButtonPress-1>", lambda e, k=k: self.press(k))
            t.bind("<B1-Motion>", self.drag)
            t.bind("<ButtonPress-3>", lambda e, k=k: self.zoom(k))
            t.bind("<ButtonRelease-3>", lambda e: self.unzoom())
            self.tiles.append(t)

        for i, c in enumerate(CLASSES, 1):
            root.bind(str(i), lambda e, c=c: self.set_label(c))
        for key, fn in {"x": lambda: self.set_label(DISCARD), "a": self.second, "z": self.revert,
                        "b": lambda: self.go(-1)}.items():
            root.bind(key, lambda e, fn=fn: fn())
            root.bind(key.upper(), lambda e, fn=fn: fn())
        for key in ("0", "<Delete>"):
            root.bind(key, lambda e: self.set_label(DISCARD))
        for key in ("space", "Return"):
            root.bind(f"<{key}>", lambda e: self.confirm())
            # Mantener Espacio apretado no debe confirmar páginas a ciegas (auto-repetición).
            root.bind(f"<KeyRelease-{key}>", lambda e: setattr(self, "armed", True))
        root.bind("<Escape>", lambda e: self.clear_sel())
        root.bind("<Left>", lambda e: self.go(-1))
        root.bind("<Right>", lambda e: self.go(1))
        self.refilter()

    def label_of(self, path):
        return self.edit.get(path) or self.decided.get(path) or self.sugg[path]

    def refilter(self):
        c = self.cls.get()
        q = [r for r in self.rows if c == "(todas)" or r["label"] == c]
        if self.pending.get():
            q = [r for r in q if r["path"] not in self.decided]
        # Por path dentro de cada tramo de confianza: fotos del mismo local y visita quedan juntas.
        q.sort(key=lambda r: (r["label"], -sum(float(r["conf"]) >= b for b in BUCKETS), r["path"]))
        self.pages, self.page = paginate(q, COLS * ROWS) or [[]], 0
        self.root.focus_set()  # que Espacio no lo capture el combo
        self.show()

    def load(self, path):
        try:
            return open_img(path, (self.bw, self.bh))
        except Exception:
            return Image.new("RGB", (self.bw, self.bh))  # ilegible: cuadro negro

    def show(self):
        rows = self.pages[self.page]
        self.sel.clear()
        for k, t in enumerate(self.tiles):
            if k < len(rows):
                t.pil = self.cache.get(rows[k]["path"]) or self.load(rows[k]["path"])
                self.paint(k)
                t.grid()
            else:
                t.grid_remove()
        c = rows[0]["label"] if rows else None
        self.banner.configure(text=f"{CLASSES.index(c) + 1}  {c}" if c else "sin fotos", bg=COLOR.get(c, NEUTRAL))
        self.ready_at = time.monotonic() + 0.15  # una tecla en vuelo no cae sobre la página nueva
        self.prefetch()
        self.update_status()

    def prefetch(self):
        """Decodifica la página siguiente en otro hilo: el cambio de página es instantáneo."""
        self.cache = cache = {}
        if self.page + 1 < len(self.pages):
            paths = [r["path"] for r in self.pages[self.page + 1]]
            threading.Thread(target=lambda: [cache.setdefault(p, self.load(p)) for p in paths], daemon=True).start()

    def paint(self, k):
        r, t = self.pages[self.page][k], self.tiles[k]
        p, conf = r["path"], float(r["conf"])
        label = self.label_of(p)
        img = t.pil
        if label == DISCARD:
            text, color = "✕ DESCARTADA", COLOR[DISCARD]
            img = Image.blend(img, Image.new("RGB", img.size), 0.6)
        elif label != self.sugg[p]:  # solo las correcciones resaltan
            text, color = f"→ {CLASSES.index(label) + 1} {label}", COLOR[label]
        else:
            text, color = ("✓ " if p in self.decided and p not in self.edit else "") + f"{conf:.0%}", NEUTRAL
            alt = TO_TESIS.get(r["top2"])
            if conf < BUCKETS[1] and alt and alt != label:
                text += f" · a: {alt}?"
        frame = SELECTED if k in self.sel else color
        if k in self.sel:
            text = "● " + text
        t.photo = ImageTk.PhotoImage(img)
        t.configure(image=t.photo, text=text, bg=color, highlightbackground=frame, highlightcolor=frame)

    def press(self, k):
        if k < len(self.pages[self.page]):
            self.drag_on = k not in self.sel  # arrastrar repite lo que hizo el primer clic
            self.toggle(k, self.drag_on)

    def drag(self, e):
        w = self.root.winfo_containing(e.x_root, e.y_root)
        if w in self.tiles and (k := self.tiles.index(w)) < len(self.pages[self.page]):
            self.toggle(k, self.drag_on)

    def toggle(self, k, on):
        if on != (k in self.sel):
            (self.sel.add if on else self.sel.discard)(k)
            self.paint(k)
            self.update_status()

    def clear_sel(self):
        ks, self.sel = self.sel, set()
        for k in ks:
            self.paint(k)
        self.update_status()

    def apply(self, label_for):
        """Aplica label_for(fila) a las seleccionadas o, si no hay, a la foto bajo el mouse."""
        rows = self.pages[self.page]
        if time.monotonic() < self.ready_at:
            return
        ks = sorted(self.sel) or ([self.hover] if self.hover is not None and self.hover < len(rows) else [])
        self.sel = set()
        for k in ks:
            if label := label_for(rows[k]):
                self.edit[rows[k]["path"]] = label
            self.paint(k)
        self.update_status()

    def set_label(self, label):
        self.apply(lambda r: label)

    def second(self):
        self.apply(lambda r: TO_TESIS.get(r["top2"]))

    def revert(self):
        self.apply(lambda r: self.sugg[r["path"]])

    def confirm(self):
        if not self.armed:
            return
        self.armed = False
        new = []
        for r in self.pages[self.page]:
            p = r["path"]
            label, old = self.label_of(p), self.decided.get(p)
            self.edit.pop(p, None)
            if label != old:
                self.decided[p] = label
                self.jobs.put((p, old, label))
                new.append((p, label, self.sugg[p]))
        if new:
            self.out.mkdir(parents=True, exist_ok=True)
            first = not self.log.exists()
            with self.log.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if first:
                    w.writerow(["path", "label", "sugerida"])
                w.writerows(new)
        self.history.append((time.monotonic(), len(self.decided)))
        self.go(1)

    def copier(self):
        while True:
            p, old, label = self.jobs.get()
            try:
                place(self.out, p, old, label)
            except OSError as e:
                print("no se pudo copiar:", p, e, flush=True)
            self.jobs.task_done()

    def go(self, step):
        self.page = min(max(0, self.page + step), len(self.pages) - 1)
        self.show()

    def update_status(self):
        n, total = len(self.decided), len(self.rows)
        ok = sum(lab == self.sugg.get(p) for p, lab in self.decided.items())
        s = f"Página {self.page + 1}/{len(self.pages)}  ·  revisadas {n:,}/{total:,} ({n / max(total, 1):.0%})"
        if n:
            s += f"  ·  modelo acierta {ok / n:.0%}"
        (t0, n0), (t1, n1) = (self.history[0], self.history[-1]) if self.history else ((0, 0), (0, 0))
        if t1 > t0 and n1 > n0:
            rate = (n1 - n0) / (t1 - t0) * 3600
            s += f"  ·  {rate:,.0f} fotos/h  ·  faltan {(total - n) / rate:.1f} h"
        changes = sum(self.label_of(r["path"]) != self.sugg[r["path"]] for r in self.pages[self.page])
        sel = f"  ·  {len(self.sel)} seleccionadas" if self.sel else ""
        self.status.configure(text=f"{s}  ·  esta página: {changes} cambios{sel}")

    def zoom(self, k):
        self.unzoom()
        if k >= len(self.pages[self.page]):
            return
        path = self.pages[self.page][k]["path"]
        top = self.zoomwin = tk.Toplevel(self.root)
        top.title(path)
        size = (int(self.root.winfo_screenwidth() * .9), int(self.root.winfo_screenheight() * .85))
        top.photo = ImageTk.PhotoImage(open_img(path, size))
        tk.Label(top, image=top.photo).pack()
        for ev in ("<Button-1>", "<Escape>"):
            top.bind(ev, lambda e: self.unzoom())
        top.focus_set()

    def unzoom(self):
        if self.zoomwin:
            self.zoomwin.destroy()
            self.zoomwin = None
            self.root.focus_set()

    def close(self):
        self.status.configure(text="terminando copias…")
        self.root.update()
        self.jobs.join()
        if self.log.exists():  # compacta: una fila por foto
            tmp = self.log.with_suffix(".tmp")
            with tmp.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["path", "label", "sugerida"])
                w.writerows((p, lab, self.sugg.get(p, "")) for p, lab in self.decided.items())
            tmp.replace(self.log)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Revisión de muebles")
    root.state("zoomed")
    app = App(root, Path(sys.argv[1]), Path(sys.argv[2]))
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
