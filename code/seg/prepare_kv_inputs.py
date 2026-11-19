"""Resize the retained KartaView photos (kv/valid_ids_v3.csv) to a short side of 512 px, the inference size of the trained models,
so that they can be moved to the GPU server over a slow link.  Writes <cache>/us/kv/img512/<id>.jpg (JPEG quality 95) and img512_manifest.csv.

  python prepare_kv_inputs.py
"""
import sys
from pathlib import Path
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE_US                                   # noqa: E402

KV = CACHE_US / "kv"; OUT = KV / "img512"; OUT.mkdir(exist_ok=True)
ids = pd.read_csv(KV / "valid_ids_v3.csv", dtype={"id": str}).id.drop_duplicates()
rows = []
for i in ids:
    with Image.open(KV / "img" / f"{i}.jpg") as im:
        im = im.convert("RGB"); s = 512 / min(im.size); w, h = im.size
        im.resize((int(w * s + 0.5), int(h * s + 0.5)), Image.BILINEAR).save(OUT / f"{i}.jpg", quality=95)
        rows.append(dict(id=i, w=w, h=h))
pd.DataFrame(rows).to_csv(KV / "img512_manifest.csv", index=False); print(len(rows))
