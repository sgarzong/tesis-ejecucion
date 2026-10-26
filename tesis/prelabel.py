"""Preetiqueta fotos con el ResNet-18 de muebles del proyecto de explicabilidad (versión reentrenada,
invariante a producto) y traduce sus 6 clases a las 7 de config.CLASSES. Se revisa con tesis.sort_gui.

Corre con el venv del clasificador (torch + torchvision), no con el de la tesis:
    <venv_clasificador>\\Scripts\\python -m tesis.prelabel <carpeta_fotos> data/prelabel.csv [carpeta_modelo]
Reanudable: si el CSV ya existe, salta las rutas que ya tiene.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .config import CLASSES, TO_TESIS

MODEL_DIR = Path(r"C:\Users\sebas\Downloads\proyecto-20260627T133347Z-3-001\proyecto"
                 r"\clasificacion_ales\retrain\model_noised")
EXT = {".jpg", ".jpeg", ".png", ".webp"}
assert set(TO_TESIS.values()) <= set(CLASSES)

# El modelo no conoce captura_invalida: se sugiere por brillo (gris 0-255) sobre la foto reducida.
# ponytail: solo detecta oscuras y planas; desenfoque/movidas necesitan varianza del laplaciano. Calibrar.
DARK_MEAN, FLAT_STD = 35, 12

TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class Fotos(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i])
            img.draft("RGB", (512, 512))  # decodifica el JPEG ya reducido: varias veces más rápido
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception:
            return torch.zeros(3, 224, 224), -1.0, 0.0, i  # ilegible -> captura_invalida
        g = np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32)
        return TF(img), float(g.mean()), float(g.std()), i


def load(model_dir):
    classes = json.loads((model_dir / "model_spec.json").read_text(encoding="utf-8"))["classes"]
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(classes))
    net.load_state_dict(torch.load(model_dir / "weights.pt", map_location="cpu"))
    return net.eval(), classes


def main(fotos, out, model_dir=MODEL_DIR):
    net, classes = load(model_dir)
    paths = sorted(str(p) for p in fotos.rglob("*") if p.suffix.lower() in EXT and p.is_file())
    done = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            done = {r["path"] for r in csv.DictReader(f)}
    todo = [p for p in paths if p not in done]
    print(f"{len(paths)} fotos, {len(done)} ya clasificadas, {len(todo)} por clasificar", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f, torch.no_grad():
        w = csv.writer(f)
        if new:
            w.writerow(["path", "label", "model_class", "conf", "top2", "top2_conf"])
        for n, (x, mean, std, idx) in enumerate(DataLoader(Fotos(todo), batch_size=64, num_workers=6)):
            top = net(x).softmax(1).topk(2, 1)
            for j, i in enumerate(idx.tolist()):
                c1, c2 = classes[top.indices[j, 0]], classes[top.indices[j, 1]]
                dark = mean[j] < DARK_MEAN or std[j] < FLAT_STD
                w.writerow([todo[i], "captura_invalida" if dark else TO_TESIS[c1], c1,
                            round(top.values[j, 0].item(), 4), c2, round(top.values[j, 1].item(), 4)])
            f.flush()
            if n % 20 == 0:
                print(f"  {min((n + 1) * 64, len(todo))}/{len(todo)}", flush=True)
    print("listo ->", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), *(Path(a) for a in sys.argv[3:]))
