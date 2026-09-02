"""Baseline MobileNetV3 appearance embedder: trained to classify SKUs from objects
composited onto random backgrounds, so the 960-d feature ignores background.
Also houses the shared cutout/augmentation helpers (load_cutouts, composite,
CropDS, IMSZ) reused by train_recognition.py and gallery.py.

  conda run -n cartgate python train_embed.py    # -> mbv3_embed_ft.onnx
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader

from cartgate.synth import make_cart_background, random_degrade

CUTOUTS = "out/cut_rembg"
IMSZ = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_cutouts():
    cut = {}
    for p in sorted(Path(CUTOUTS).glob("*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is not None and im.ndim == 3 and im.shape[2] == 4:
            cut.setdefault(p.stem.split("__")[0], []).append(im)
    return cut


def rand_bg(rng):
    """Varied backgrounds so the net can't rely on any one -> background invariance."""
    r = rng.random()
    if r < 0.4:
        return cv2.resize(make_cart_background(320, 320, rng), (IMSZ, IMSZ))
    if r < 0.7:
        base = np.full((IMSZ, IMSZ, 3), rng.integers(0, 255, 3), np.uint8)
        return np.clip(base + rng.normal(0, 25, (IMSZ, IMSZ, 3)), 0, 255).astype(np.uint8)
    # gradient + clutter
    a, b = rng.integers(0, 255, 3), rng.integers(0, 255, 3)
    t = np.linspace(0, 1, IMSZ)[:, None, None]
    g = (a * (1 - t) + b * t).astype(np.uint8)
    g = np.repeat(g, IMSZ, axis=1) if g.shape[1] == 1 else cv2.resize(g, (IMSZ, IMSZ))
    return g.reshape(IMSZ, IMSZ, 3) if g.shape != (IMSZ, IMSZ, 3) else g


def composite(cutout, rng):
    """One training crop: augmented object on a random bg, framed like a detected crop."""
    rgba = cutout.copy()
    # rotate
    ang = rng.uniform(0, 360)
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    rgba = cv2.warpAffine(rgba, M, (w, h), borderValue=(0, 0, 0, 0))
    ys, xs = np.where(rgba[:, :, 3] > 0)
    if len(ys) < 5:
        rgba = cutout.copy(); ys, xs = np.where(rgba[:, :, 3] > 0)
    rgba = rgba[ys.min():ys.max()+1, xs.min():xs.max()+1]
    # scale object to fill 50-95% of the crop
    fill = rng.uniform(0.5, 0.95)
    s = fill * IMSZ / max(rgba.shape[:2])
    rgba = cv2.resize(rgba, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    # lighting / colour jitter
    f = rng.uniform(0.6, 1.3); col = rng.uniform(0.85, 1.15, 3)
    rgba[:, :, :3] = np.clip(rgba[:, :, :3].astype(np.float32) * f * col, 0, 255).astype(np.uint8)
    canvas = rand_bg(rng)
    oh, ow = rgba.shape[:2]
    y0 = int(rng.uniform(0, IMSZ - oh)) if IMSZ > oh else 0
    x0 = int(rng.uniform(0, IMSZ - ow)) if IMSZ > ow else 0
    oh, ow = min(oh, IMSZ - y0), min(ow, IMSZ - x0)
    patch = rgba[:oh, :ow]
    a = patch[:, :, 3:4].astype(np.float32) / 255.0
    canvas[y0:y0+oh, x0:x0+ow] = (a * patch[:, :, :3] +
                                  (1 - a) * canvas[y0:y0+oh, x0:x0+ow]).astype(np.uint8)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)
    return random_degrade(canvas, rng)             # low-quality-camera robustness


class CropDS(Dataset):
    def __init__(self, cutouts, skus, n=6000):
        self.cut, self.skus, self.n = cutouts, skus, n
        self.idx = {s: i for i, s in enumerate(skus)}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i * 2654435761 % (2**32))
        sku = self.skus[rng.integers(len(self.skus))]
        cut = self.cut[sku][rng.integers(len(self.cut[sku]))]
        img = composite(cut, rng)
        rgb = cv2.cvtColor(cv2.resize(img, (IMSZ, IMSZ)), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)), self.idx[sku]


class EmbedNet(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        m = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2")
        self.features, self.avgpool = m.features, m.avgpool
        self.head = nn.Linear(960, n_cls)

    def embed(self, x):
        return torch.flatten(self.avgpool(self.features(x)), 1)   # 960-d

    def forward(self, x):
        return self.head(self.embed(x))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cut = load_cutouts()
    skus = sorted(cut.keys())
    print(f"device={dev}  SKUs={len(skus)}  cutouts={sum(len(v) for v in cut.values())}")
    dl = DataLoader(CropDS(cut, skus, n=6000), batch_size=64, shuffle=True, num_workers=8, drop_last=True)
    net = EmbedNet(len(skus)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    EPOCHS = 12
    for ep in range(1, EPOCHS + 1):
        net.train(); tot = correct = 0; lsum = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = net(x); loss = lossf(out, y)
            loss.backward(); opt.step()
            lsum += loss.item() * len(y); tot += len(y)
            correct += (out.argmax(1) == y).sum().item()
        print(f"  ep{ep:2d} loss={lsum/tot:.3f} train_acc={correct/tot:.3f}")

    # export the 960-d embedding to ONNX (same interface as mbv3_embed.onnx)
    net.eval()
    class EmbOnly(nn.Module):
        def __init__(self, n): super().__init__(); self.n = n
        def forward(self, x):
            v = self.n.embed(x)
            return v / v.norm(dim=1, keepdim=True).clamp_min(1e-6)
    torch.onnx.export(EmbOnly(net).to("cpu").eval(), torch.randn(1, 3, IMSZ, IMSZ),
                      "mbv3_embed_ft.onnx", input_names=["input"], output_names=["emb"],
                      opset_version=17, dynamic_axes={"input": {0: "b"}, "emb": {0: "b"}})
    print("exported mbv3_embed_ft.onnx")


if __name__ == "__main__":
    main()
