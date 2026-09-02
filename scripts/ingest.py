"""Unify the raw multi-collector dataset into gallery-ready SKU folders.

Labeling here means sorting photos into SKU folders, not drawing boxes. Two phases:
  scan  -- convert HEIC, drop exact dups, EXIF time-gap cluster camera dumps, and
           emit review.html + labels.csv + manifest.json for a human to name groups.
  build -- read the filled labels.csv and materialize dataset/<sku_id>__<name>/ and
           products.csv. Groups sharing a name merge into one SKU; blank names are
           skipped. Originals are only ever copied/converted, never modified.

  python3 ingest.py scan  [--src cjs_dataset] [--out .] [--gap 20]
  python3 ingest.py build [--out .]
"""
import argparse
import base64
import csv
import datetime as dt
import hashlib
import io
import json
import re
from pathlib import Path

import pillow_heif
from PIL import Image, ImageOps, ExifTags

pillow_heif.register_heif_opener()

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
_DTO, _DT = 36867, 306  # EXIF DateTimeOriginal / DateTime
# camera auto-generated file names => photo carries no human label
_CAMERA_RE = re.compile(r"^(img|dsc|dscf|pxl|photo|kakaotalk|screenshot|image)[-_ ]?\d+", re.I)
_TRAILING_NUM_RE = re.compile(r"[-_ ]?\d+$")


# --------------------------------------------------------------------------- utils
def content_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def capture_time(path: Path):
    """EXIF DateTimeOriginal as datetime, else None."""
    try:
        ex = Image.open(path).getexif()
        v = ex.get(_DTO) or ex.get(_DT)
        try:
            v = ex.get_ifd(ExifTags.IFD.Exif).get(_DTO) or v
        except Exception:
            pass
        if v:
            return dt.datetime.strptime(str(v).strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def load_rgb(path: Path) -> Image.Image:
    """Open any supported image, apply EXIF orientation, return RGB PIL image."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def thumb_b64(path: Path, box: int = 220) -> str:
    im = load_rgb(path)
    im.thumbnail((box, box))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def is_camera_name(stem: str) -> bool:
    return bool(_CAMERA_RE.match(stem))


def label_from_prefix(stem: str) -> str:
    """`mouse_3` -> `mouse`, `Aloe_2` -> `aloe`, `apple` -> `apple` (lowercased)."""
    return _TRAILING_NUM_RE.sub("", stem).strip().lower()


def safe_dirname(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = re.sub(r"\s+", "_", name)
    return re.sub(r"[^0-9A-Za-z가-힣_\-]", "", name) or "unnamed"


# --------------------------------------------------------------------------- scan
class Item:
    __slots__ = ("path", "collector", "hash", "time", "stem", "ext")

    def __init__(self, path: Path, collector: str, h: str, t, stem: str, ext: str):
        self.path, self.collector, self.hash = path, collector, h
        self.time, self.stem, self.ext = t, stem, ext


def _sort_key(it: "Item"):
    return (it.time or dt.datetime.fromtimestamp(it.path.stat().st_mtime), it.stem)


def collect_items(src: Path):
    """Return (items, groups). groups = list of dicts describing candidate SKUs."""
    seen_hash = set()
    groups = []  # {collector, source, suggested_name, items:[Item]}

    def add_unique(path, collector):
        try:
            h = content_hash(path)
        except Exception:
            return None
        if h in seen_hash:
            return None
        seen_hash.add(h)
        return Item(path, collector, h, capture_time(path), path.stem, path.suffix.lower())

    # 1) top-level loose files -> labeled by filename prefix
    loose = {}
    for p in sorted(src.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            it = add_unique(p, "loose")
            if it:
                loose.setdefault(label_from_prefix(p.stem), []).append(it)
    for name, its in sorted(loose.items()):
        groups.append(dict(collector="loose", source="filename", suggested_name=name, items=its))

    # 2) per-collector folders
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        subdirs = [s for s in sorted(d.iterdir()) if s.is_dir()]
        flat = [f for f in sorted(d.iterdir()) if f.is_file() and f.suffix.lower() in IMG_EXTS]

        # 2a) product-name subfolders -> already labeled
        for s in subdirs:
            its = [it for f in sorted(s.iterdir())
                   if f.is_file() and f.suffix.lower() in IMG_EXTS
                   for it in [add_unique(f, d.name)] if it]
            if its:
                groups.append(dict(collector=d.name, source="folder",
                                   suggested_name=s.name, items=its))

        if not flat:
            continue

        flat_items = [it for f in flat for it in [add_unique(f, d.name)] if it]
        camera = sum(is_camera_name(it.stem) for it in flat_items)

        if flat_items and camera >= 0.6 * len(flat_items):
            # 2b) unlabeled camera dump -> EXIF time-gap clustering (done later)
            groups.append(dict(collector=d.name, source="cluster",
                               suggested_name="", items=flat_items, _needs_cluster=True))
        else:
            # 2c) flat files labeled by their own filename (e.g. bandage.png)
            for it in flat_items:
                groups.append(dict(collector=d.name, source="filename",
                                   suggested_name=label_from_prefix(it.stem), items=[it]))
    return groups


def cluster_by_time(items, gap_sec: int):
    """Split a time-sorted list wherever the capture-time gap exceeds gap_sec."""
    items = sorted(items, key=_sort_key)
    clusters, cur, prev = [], [], None
    for it in items:
        t = _sort_key(it)[0]
        if prev is not None and (t - prev).total_seconds() > gap_sec:
            clusters.append(cur); cur = []
        cur.append(it); prev = t
    if cur:
        clusters.append(cur)
    return clusters


def scan(src: Path, out: Path, gap: int):
    raw_groups = collect_items(src)

    # expand cluster groups
    groups = []
    for g in raw_groups:
        if g.get("_needs_cluster"):
            for i, cl in enumerate(cluster_by_time(g["items"], gap), 1):
                groups.append(dict(collector=g["collector"], source="cluster",
                                   suggested_name="", items=cl))
        else:
            groups.append(g)

    # order: unlabeled clusters first (need attention), then labeled, both stable
    def order_key(g):
        t = min(_sort_key(it)[0] for it in g["items"])
        return (0 if g["source"] == "cluster" else 1, g["collector"], g["suggested_name"], t)
    groups.sort(key=order_key)

    for i, g in enumerate(groups, 1):
        g["group_id"] = f"G{i:04d}"
        g["items"].sort(key=_sort_key)

    out.mkdir(parents=True, exist_ok=True)
    _write_manifest(groups, out, gap)
    _write_labels_csv(groups, out)
    _write_review_html(groups, out, gap)

    n_img = sum(len(g["items"]) for g in groups)
    n_cluster = sum(1 for g in groups if g["source"] == "cluster")
    n_need = sum(len(g["items"]) for g in groups if g["source"] == "cluster")
    print(f"\n  scanned  : {n_img} unique images  ->  {len(groups)} candidate groups")
    print(f"  labeled  : {len(groups) - n_cluster} groups (name pre-filled)")
    print(f"  to name  : {n_cluster} clustered groups  ({n_need} images) need a product name")
    print(f"\n  review   : {out/'review.html'}   (open in a browser, type names, download labels.csv)")
    print(f"  or edit  : {out/'labels.csv'}      (fill the `name` column in any editor)")
    print(f"  then run : python3 ingest.py build --out {out}\n")


def _write_manifest(groups, out: Path, gap: int):
    data = {"gap_sec": gap, "groups": []}
    for g in groups:
        data["groups"].append(dict(
            group_id=g["group_id"], collector=g["collector"], source=g["source"],
            suggested_name=g["suggested_name"],
            items=[dict(path=str(it.path), hash=it.hash, ext=it.ext,
                        time=it.time.isoformat() if it.time else None) for it in g["items"]],
        ))
    (out / "manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _write_labels_csv(groups, out: Path):
    with open(out / "labels.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "collector", "source", "n_images", "time_start",
                    "suggested_name", "name"])
        for g in groups:
            t0 = min((it.time for it in g["items"] if it.time), default=None)
            w.writerow([g["group_id"], g["collector"], g["source"], len(g["items"]),
                        t0.isoformat(sep=" ") if t0 else "",
                        g["suggested_name"], g["suggested_name"]])


def _write_review_html(groups, out: Path, gap: int):
    cards, meta = [], []
    for g in groups:
        t0 = min((it.time for it in g["items"] if it.time), default=None)
        thumbs = "".join(
            f'<img class="{ "rep" if i==0 else "th" }" src="{thumb_b64(it.path)}" '
            f'alt="{it.path.name}" title="{it.path.name}" loading="lazy">'
            for i, it in enumerate(g["items"]))
        need = g["source"] == "cluster"
        badge = {"cluster": "이름 필요", "folder": "폴더 라벨", "filename": "파일명 라벨"}[g["source"]]
        val = g["suggested_name"].replace('"', "&quot;")
        cards.append(f'''
        <article class="card {'need' if need else 'done'}" data-need="{int(need)}">
          <div class="hd"><span class="gid">{g["group_id"]}</span>
            <span class="badge {g['source']}">{badge}</span>
            <span class="muted">{g["collector"]} · {len(g["items"])}장 · {t0.strftime("%m-%d %H:%M") if t0 else "시각없음"}</span>
          </div>
          <div class="thumbs">{thumbs}</div>
          <input id="inp_{g['group_id']}" value="{val}" placeholder="상품명 입력…"
                 aria-label="{g['group_id']} 상품명" oninput="mark(this)">
        </article>''')
        meta.append({"id": g["group_id"], "collector": g["collector"],
                     "source": g["source"], "n": len(g["items"]),
                     "t0": t0.isoformat(sep=" ") if t0 else "",
                     "sug": g["suggested_name"]})

    n_need = sum(1 for g in groups if g["source"] == "cluster")
    n_lab = len(groups) - n_need
    inner = f'''<style>
 :root{{--bg:#f6f7f9;--surface:#ffffff;--surface2:#fbfcfd;--line:#e3e7ee;--ink:#1a2130;
   --muted:#5f6b7e;--acc:#2563eb;--acc-ink:#fff;--need:#b54708;--need-bg:#fff7ed;
   --ok:#15803d;--chip:#eef2f8;
   --font:system-ui,-apple-system,'Apple SD Gothic Neo','Malgun Gothic',sans-serif;
   --mono:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;}}
 @media (prefers-color-scheme:dark){{:root{{--bg:#0f1115;--surface:#171a21;--surface2:#1b1f27;
   --line:#2a2f3a;--ink:#e6e8ec;--muted:#98a2b3;--acc:#4b8cf7;--need:#f0a35e;--need-bg:#241a10;
   --ok:#5fce8a;--chip:#232833;}}}}
 :root[data-theme="light"]{{--bg:#f6f7f9;--surface:#ffffff;--surface2:#fbfcfd;--line:#e3e7ee;
   --ink:#1a2130;--muted:#5f6b7e;--acc:#2563eb;--need:#b54708;--need-bg:#fff7ed;--ok:#15803d;--chip:#eef2f8;}}
 :root[data-theme="dark"]{{--bg:#0f1115;--surface:#171a21;--surface2:#1b1f27;--line:#2a2f3a;
   --ink:#e6e8ec;--muted:#98a2b3;--acc:#4b8cf7;--need:#f0a35e;--need-bg:#241a10;--ok:#5fce8a;--chip:#232833;}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);line-height:1.45}}
 header{{position:sticky;top:0;z-index:5;background:var(--surface);border-bottom:1px solid var(--line);
   padding:16px 22px}}
 .eyebrow{{font:600 11px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--acc);
   margin-bottom:6px}}
 h1{{margin:0;font-size:18px;font-weight:650;letter-spacing:-.01em}}
 .lead{{color:var(--muted);font-size:12.5px;margin-top:5px;max-width:70ch}}
 .lead b{{color:var(--ink);font-weight:600}}
 .bar{{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}}
 button{{font:600 13px var(--font);border:1px solid transparent;border-radius:8px;padding:8px 14px;
   cursor:pointer;background:var(--acc);color:var(--acc-ink)}}
 button.sec{{background:transparent;border-color:var(--line);color:var(--ink)}}
 button:focus-visible{{outline:2px solid var(--acc);outline-offset:2px}}
 .prog{{margin-left:auto;font:600 12px var(--mono);color:var(--muted);font-variant-numeric:tabular-nums}}
 .track{{height:5px;border-radius:3px;background:var(--chip);overflow:hidden;width:140px;display:inline-block;
   vertical-align:middle;margin-left:8px}}
 .track>i{{display:block;height:100%;width:0;background:var(--ok);transition:width .2s}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:14px;padding:18px 22px}}
 .card{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px 12px 14px;
   display:flex;flex-direction:column;gap:9px}}
 .card.need{{border-left:3px solid var(--need);background:linear-gradient(var(--need-bg),var(--surface) 60%)}}
 .card.done{{opacity:.9}}
 .hd{{display:flex;gap:7px;align-items:center;flex-wrap:wrap}}
 .gid{{font:600 12px var(--mono);color:var(--ink)}}
 .muted{{color:var(--muted);font-size:11.5px}}
 .badge{{font:600 10px var(--mono);letter-spacing:.03em;padding:2px 8px;border-radius:20px;
   border:1px solid var(--line);color:var(--muted);background:var(--chip)}}
 .badge.cluster{{color:var(--need);border-color:color-mix(in srgb,var(--need) 40%,transparent)}}
 .thumbs{{display:flex;flex-wrap:wrap;gap:5px;align-items:flex-start}}
 .thumbs img{{border-radius:7px;object-fit:cover;background:var(--chip);border:1px solid var(--line)}}
 img.rep{{width:120px;height:120px;border:2px solid var(--acc)}}
 img.th{{width:56px;height:56px}}
 input{{width:100%;padding:9px 11px;border-radius:8px;border:1px solid var(--line);
   background:var(--surface2);color:var(--ink);font-size:14px}}
 input::placeholder{{color:var(--muted)}}
 input:focus-visible{{outline:2px solid var(--acc);outline-offset:1px;border-color:var(--acc)}}
 .card.filled input{{border-color:var(--ok);background:color-mix(in srgb,var(--ok) 8%,var(--surface2))}}
 @media (prefers-reduced-motion:reduce){{*{{transition:none!important}}}}
</style>
<header>
 <div class="eyebrow">CartGate · 데이터 라벨링</div>
 <h1>그룹별 상품명 입력</h1>
 <p class="lead">그룹당 상품명 하나씩 적으세요. <b>두 그룹에 같은 이름</b>을 적으면 하나의 SKU로 병합됩니다(과분할 교정).
   비워 두면 그 그룹은 제외됩니다. 갭 임계값 {gap}s · 라벨 완료 {n_lab}개 · 이름 필요 {n_need}개.</p>
 <div class="bar">
   <button onclick="dl()">labels.csv 내려받기</button>
   <button class="sec" onclick="tog(this)">이름 필요만 보기</button>
   <span class="prog"><span id="stat">0 / {len(groups)}</span><span class="track"><i id="pbar"></i></span></span>
 </div>
</header>
<main class="grid" id="grid">{''.join(cards)}</main>
<script>
 const META={json.dumps(meta, ensure_ascii=False)};
 let onlyNeed=false;
 function mark(el){{el.closest('.card').classList.toggle('filled', el.value.trim().length>0); stat();}}
 function stat(){{let f=0;META.forEach(m=>{{if(document.getElementById('inp_'+m.id).value.trim())f++}});
   document.getElementById('stat').textContent=f+' / '+META.length;
   document.getElementById('pbar').style.width=(100*f/META.length)+'%';}}
 function tog(btn){{onlyNeed=!onlyNeed;btn.classList.toggle('sec',!onlyNeed);
   document.querySelectorAll('.card').forEach(c=>{{
     c.style.display=(onlyNeed && c.dataset.need==='0')?'none':'';}});}}
 function csvcell(s){{s=(s==null?'':''+s);return /[",\\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}}
 function dl(){{let rows=[['group_id','collector','source','n_images','time_start','suggested_name','name']];
   META.forEach(m=>rows.push([m.id,m.collector,m.source,m.n,m.t0,m.sug,
     document.getElementById('inp_'+m.id).value.trim()]));
   let csv='\\ufeff'+rows.map(r=>r.map(csvcell).join(',')).join('\\n');
   let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));
   a.download='labels.csv';a.click();}}
 document.querySelectorAll('.card input').forEach(mark);stat();
</script>'''
    doc = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           '<title>CartGate 데이터 라벨링 검토</title></head><body>'
           + inner + '</body></html>')
    (out / "review.html").write_text(doc, encoding="utf-8")
    (out / "review_artifact.html").write_text(inner, encoding="utf-8")


# --------------------------------------------------------------------------- build
def build(out: Path):
    man = json.loads((out / "manifest.json").read_text())
    gmap = {g["group_id"]: g for g in man["groups"]}

    names = {}
    with open(out / "labels.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            nm = (row.get("name") or "").strip()
            if nm:
                names[row["group_id"]] = nm

    # merge groups that share a name (case-insensitive)
    by_name = {}
    for gid, nm in names.items():
        by_name.setdefault(nm.lower(), {"name": nm, "gids": []})["gids"].append(gid)

    dataset = out / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    products, sku_i, total = [], 0, 0
    for key in sorted(by_name):
        sku_i += 1
        sku_id = f"S{sku_i:04d}"
        name = by_name[key]["name"]
        dst = dataset / f"{sku_id}__{safe_dirname(name)}"
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for gid in by_name[key]["gids"]:
            for it in gmap[gid]["items"]:
                src = Path(it["path"])
                try:
                    img = load_rgb(src)
                except Exception as e:
                    print(f"    ! skip {src}: {e}"); continue
                stem = safe_dirname(f"{gmap[gid]['collector']}_{src.stem}")
                img.save(dst / f"{stem}.jpg", format="JPEG", quality=92)
                n += 1
        products.append((sku_id, name, ""))
        total += n
        print(f"  {sku_id}  {name:<22} {n} imgs   ({', '.join(by_name[key]['gids'])})")

    with open(out / "products.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sku_id", "name", "barcode"])
        w.writerows(products)

    print(f"\n  built {sku_i} SKUs / {total} images -> {dataset}")
    print(f"  products.csv -> {out/'products.csv'}  (fill in barcodes when available)")
    print(f"  next: python3 run_demo.py --dataset {dataset} --onnx mbv3_embed.onnx\n")


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="unify + cluster + emit review.html/labels.csv")
    s.add_argument("--src", default="cjs_dataset")
    s.add_argument("--out", default=".")
    s.add_argument("--gap", type=int, default=20, help="new-group time gap (sec) for camera dumps")
    b = sub.add_parser("build", help="materialize dataset/<sku_id>__<name>/ from filled labels.csv")
    b.add_argument("--out", default=".")
    a = ap.parse_args()
    if a.cmd == "scan":
        scan(Path(a.src), Path(a.out), a.gap)
    else:
        build(Path(a.out))


if __name__ == "__main__":
    main()
