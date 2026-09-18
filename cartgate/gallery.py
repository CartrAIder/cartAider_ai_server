"""Build the SKU embedding gallery from dataset/<sku>/<images>. Each image is
expanded into rotation/flip variants so one studio shot still matches field crops
seen at arbitrary rotation."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from cartgate.segment import remove_background, crop_to_object

ROTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _fingerprint(dataset_dir: str, cutouts_dir: str, params: dict) -> str:
    """Identity of a gallery build: every source image (path/mtime/size) plus the
    build parameters. Any edited, added or removed photo changes it."""
    h = hashlib.sha256()
    h.update(json.dumps(params, sort_keys=True).encode())
    for root in (dataset_dir, cutouts_dir):
        p = Path(root)
        if not p.exists():
            continue
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                st = f.stat()
                h.update(f"{f.relative_to(p)}|{st.st_mtime_ns}|{st.st_size}".encode())
    return h.hexdigest()


def rotate_rgba(rgba: np.ndarray, deg: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    side = int(np.ceil(np.hypot(h, w)))
    canvas = np.zeros((side, side, 4), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = rgba
    M = cv2.getRotationMatrix2D((side / 2, side / 2), deg, 1.0)
    return cv2.warpAffine(canvas, M, (side, side), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def save_gallery_npz(gallery: dict, path: str = "out/gallery.npz") -> str:
    """Write a NumPy-version-portable gallery using arrays and JSON only."""
    skus = sorted(gallery)
    vectors = (
        np.concatenate([np.asarray(gallery[sku]["vectors"], dtype=np.float32) for sku in skus])
        if skus
        else np.zeros((0, 0), dtype=np.float32)
    )
    np.savez_compressed(
        path,
        sku_ids=np.array(skus),
        counts=np.array([len(gallery[sku]["vectors"]) for sku in skus], dtype=np.int32),
        vectors=vectors,
        views_json=np.array(json.dumps({sku: gallery[sku].get("views", []) for sku in skus})),
    )
    return path


def _load_npz(path: Path) -> dict:
    with np.load(str(path), allow_pickle=False) as archive:
        skus = [str(sku) for sku in archive["sku_ids"]]
        counts = archive["counts"].astype(int)
        vectors = archive["vectors"]
        views = json.loads(str(archive["views_json"]))

    gallery, offset = {}, 0
    for sku, count in zip(skus, counts):
        gallery[sku] = {
            "vectors": vectors[offset:offset + count],
            "views": views.get(sku, []),
        }
        offset += int(count)
    return gallery


def _load_pkl(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            return pickle.load(stream)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{path} was pickled by an incompatible NumPy version; "
            "deploy the sibling .npz gallery instead"
        ) from exc


def load_gallery(path: str = "out/gallery.pkl") -> dict:
    """Load a prebuilt .npz or .pkl gallery for serving.

    build_gallery() needs the product photos; a deployed service does not. Ship
    out/gallery.npz (a few MB) instead of the photo set and load it here: the
    embeddings are all the recognizer ever touches. Rebuild only when the
    product photos or the embedding model change.

    If the requested file is absent, the sibling with the other supported suffix
    is used. This keeps older serving code compatible with NPZ-only bundles.
    """
    gallery_path = Path(path)
    if not gallery_path.exists():
        alternate_suffix = ".npz" if gallery_path.suffix == ".pkl" else ".pkl"
        alternate = gallery_path.with_suffix(alternate_suffix)
        if alternate.exists():
            gallery_path = alternate

    gallery = _load_npz(gallery_path) if gallery_path.suffix == ".npz" else _load_pkl(gallery_path)
    if not gallery:
        raise ValueError(f"{gallery_path} holds no SKUs")
    return gallery


def build_gallery(dataset_dir: str, embedder, out_dir: str = "out",
                  save_cutouts: bool = True, remove_bg: bool = False,
                  enrich_synth: int = 0, cutouts_dir: str = "out/cut_rembg",
                  enrich_seed: int = 7, cache: bool = True) -> dict:
    """Returns {sku: {"vectors": [N,D], "views": [...]}}.

    remove_bg=False (default) embeds the full studio image with 90-deg rotations +
    flip (detected crops are embedded raw the same way); remove_bg=True uses cutouts.
    enrich_synth>0 also appends that many synthetic composite views per cutout
    (object on varied backgrounds), which makes the gallery cover the field-crop
    distribution rather than only clean studio shots.

    cache=True reuses out/gallery.pkl when the source photos and the build
    parameters are unchanged (rebuilding embeds thousands of views and costs
    minutes); any edited/added/removed photo invalidates it automatically.
    """
    dataset = Path(dataset_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if save_cutouts and remove_bg:
        (out / "cutouts").mkdir(parents=True, exist_ok=True)

    key_file = out / "gallery.key.json"
    pkl_file = out / "gallery.pkl"
    key = _fingerprint(dataset_dir, cutouts_dir if enrich_synth > 0 else "", {
        "remove_bg": remove_bg, "enrich_synth": enrich_synth, "enrich_seed": enrich_seed,
        # model_id carries the weight file's mtime/size: swapping dino_arc.onnx for a
        # retrained one must invalidate the cache, or the gallery silently describes
        # the OLD embedding space while queries come from the new one.
        "embedder": getattr(embedder, "model_id", None) or getattr(embedder, "name", type(embedder).__name__),
        "rotations": ROTATIONS,
    })
    if cache and key_file.exists() and pkl_file.exists():
        try:
            if json.loads(key_file.read_text()).get("key") == key:
                with open(pkl_file, "rb") as f:
                    gallery = pickle.load(f)
                npz_file = out / "gallery.npz"
                if not npz_file.exists():
                    save_gallery_npz(gallery, str(npz_file))
                print(f"  gallery cache hit ({len(gallery)} SKUs, "
                      f"{sum(v['vectors'].shape[0] for v in gallery.values())} vectors)")
                return gallery
        except Exception as exc:                     # corrupt cache -> just rebuild
            print(f"  gallery cache unusable ({exc}); rebuilding")

    syn_cut = {}
    if enrich_synth > 0:
        import cv2 as _cv2
        for p in sorted(Path(cutouts_dir).glob("*.png")):
            im = _cv2.imread(str(p), _cv2.IMREAD_UNCHANGED)
            if im is not None and im.ndim == 3 and im.shape[2] == 4:
                syn_cut.setdefault(p.stem.split("__")[0], []).append(im)

    gallery: dict = {}

    for sku_dir in sorted(p for p in dataset.iterdir() if p.is_dir()):
        # key by sku_id (folder prefix before "__"), as receipts/products.csv do
        sku = sku_dir.name.split("__")[0]
        vecs, views = [], []
        for img_path in sorted(sku_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            if remove_bg:
                rgba, _ = remove_background(bgr)
                rgba = crop_to_object(rgba)
                if save_cutouts:
                    cv2.imwrite(str(out / "cutouts" / f"{sku}__{img_path.stem}.png"), rgba)
                for deg in ROTATIONS:
                    rot = rotate_rgba(rgba, deg) if deg else rgba
                    vecs.append(embedder.embed(rot[:, :, :3], rot[:, :, 3]))
                    views.append({"src": img_path.name, "rot": deg, "flip": False})
                flipped = cv2.flip(rgba, 1)
                vecs.append(embedder.embed(flipped[:, :, :3], flipped[:, :, 3]))
                views.append({"src": img_path.name, "rot": 0, "flip": True})
            else:
                # no background removal: embed the whole image + clean 90-deg rotations
                for deg, rot in ((0, bgr), (90, cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)),
                                 (180, cv2.rotate(bgr, cv2.ROTATE_180)),
                                 (270, cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE))):
                    vecs.append(embedder.embed(rot, None))
                    views.append({"src": img_path.name, "rot": deg, "flip": False})
                vecs.append(embedder.embed(cv2.flip(bgr, 1), None))
                views.append({"src": img_path.name, "rot": 0, "flip": True})
        if enrich_synth > 0 and sku in syn_cut:
            from cartgate.train_embed import composite
            rng = np.random.default_rng(enrich_seed + hash(sku) % 10000)
            for co in syn_cut[sku]:
                for _ in range(enrich_synth):
                    comp = composite(co, rng)                     # 224 bgr, object on varied bg + degrade
                    vecs.append(embedder.embed(comp, None))
                    views.append({"src": f"synth:{sku}", "rot": 0, "flip": False})

        if vecs:
            gallery[sku] = {"vectors": np.stack(vecs), "views": views}
            print(f"  {sku}: {sum(1 for v in views if str(v['src']).startswith('synth'))} synth + "
                  f"{sum(1 for v in views if not str(v['src']).startswith('synth'))} studio "
                  f"-> {len(vecs)} gallery vectors")

    with open(pkl_file, "wb") as f:
        pickle.dump(gallery, f)
    save_gallery_npz(gallery, str(out / "gallery.npz"))
    with open(out / "gallery_meta.json", "w") as f:
        json.dump({k: {"n_vectors": int(v["vectors"].shape[0])} for k, v in gallery.items()}, f, indent=2)
    key_file.write_text(json.dumps({
        "key": key, "n_skus": len(gallery),
        "n_vectors": int(sum(v["vectors"].shape[0] for v in gallery.values())),
    }, indent=2))
    return gallery
