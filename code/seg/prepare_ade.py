"""Prepare the street-scene vegetation-structure dataset from ADE20K (SceneParsing, 150 classes; Zhou et al., CVPR 2017 / IJCV 2019).

Taxonomy (6 classes; ADE20K "unlabeled" -> 255, ignored):
  0 tree            ADE tree, palm
  1 low vegetation  ADE grass, plant, flower, field
  2 sky             ADE sky
  3 ground          ADE road, sidewalk, earth, path, sand, runway, dirt track, land
  4 built           ADE wall, building, house, skyscraper, fence, bridge, tower, railing, column, awning, booth, hovel
  5 other           every other ADE class
Hierarchy used by the HVH head: vegetation = {tree, low vegetation}; the other four classes are their own super-classes.

Outdoor flag (fixed before any training): the image contains sky (>= 1% of labeled pixels) and no ceiling.
Splits: all ADE training images with labeled pixels -> train (about 20k), except a seeded random 500 outdoor images -> select
        (checkpoint choice, matched to the street-level target domain); ADE validation images with the outdoor flag -> test
        (reported once); the remaining validation images -> val_other (used only for the all-scene mIoU).

  python prepare_ade.py --root /path/ADEChallengeData2016 --out /path/vegstruct
Writes <out>/labels/{train,val}/<name>.png (uint8, 0-5, 255), <out>/splits.csv, <out>/label_map.csv, <out>/summary.json
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

TREE = {"tree", "palm"}
LOW = {"grass", "plant", "flower", "field"}
SKY = {"sky"}
GROUND = {"road", "sidewalk", "earth", "path", "sand", "runway", "dirt track", "land"}
BUILT = {"wall", "building", "house", "skyscraper", "fence", "bridge", "tower", "railing", "column", "awning", "booth", "hovel"}
NAMES = ["tree", "low_vegetation", "sky", "ground", "built", "other"]


def load_objectinfo(root):
    """objectInfo150: Idx, Ratio, Train, Val, Name ("tree", "plant, flora, plant life", ...); the first name is the class name"""
    f = root / "objectInfo150.txt"
    rows = [l.rstrip("\n").split("\t") for l in f.read_text(encoding="utf-8").splitlines()[1:] if l.strip()]
    return {int(r[0]): r[-1].replace(";", ",").split(",")[0].strip() for r in rows}


def build_lut(info):
    lut = np.full(256, 255, np.uint8)
    for idx, name in info.items():
        lut[idx] = 0 if name in TREE else 1 if name in LOW else 2 if name in SKY else 3 if name in GROUND else 4 if name in BUILT else 5
    return lut


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n_select", type=int, default=500); ap.add_argument("--seed", type=int, default=20260929)
    a = ap.parse_args(); root, out = Path(a.root), Path(a.out)
    info = load_objectinfo(root); lut = build_lut(info)
    ceiling = [i for i, n in info.items() if n == "ceiling"]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(ade_index=i, ade_name=n, class_id=int(lut[i]), class_name=NAMES[lut[i]]) for i, n in sorted(info.items())]).to_csv(out / "label_map.csv", index=False)
    rows = []
    for ade_split, our in (("training", "train"), ("validation", "val")):
        (out / "labels" / our).mkdir(parents=True, exist_ok=True)
        for f in sorted((root / "annotations" / ade_split).glob("*.png")):
            a_ = np.asarray(Image.open(f)); lab = a_ > 0
            if lab.sum() == 0: continue
            sky = float((a_ == [i for i, n in info.items() if n == "sky"][0]).sum()) / lab.sum()
            outdoor = bool(sky >= 0.01 and not np.isin(a_, ceiling).any())
            m = lut[a_]; Image.fromarray(m).save(out / "labels" / our / f.name)
            cnt = np.bincount(m[m < 255].ravel(), minlength=6) / max((m < 255).sum(), 1)
            rows.append(dict(name=f.stem, ade_split=ade_split, split=our, image=str(root / "images" / ade_split / (f.stem + ".jpg")), label=str(out / "labels" / our / f.name),
                             h=a_.shape[0], w=a_.shape[1], outdoor=outdoor, **{f"share_{n}": float(c) for n, c in zip(NAMES, cnt)}))
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(a.seed); tr = df.index[(df.split == "train") & df.outdoor].to_numpy(); sel = rng.choice(tr, a.n_select, replace=False)
    df.loc[sel, "split"] = "select"; df.loc[(df.split == "val") & df.outdoor, "split"] = "test"; df.loc[df.split == "val", "split"] = "val_other"
    df.to_csv(out / "splits.csv", index=False)
    summ = dict(n=len(df), **{f"n_{s}": int((df.split == s).sum()) for s in ("train", "select", "test", "val_other")}, n_train_outdoor=int(((df.split == "train") & df.outdoor).sum()),
                seed=a.seed, mean_share_test={n: float(df.loc[df.split == "test", f"share_{n}"].mean()) for n in NAMES},
                mean_share_train={n: float(df.loc[df.split == "train", f"share_{n}"].mean()) for n in NAMES}, outdoor_flag="sky >= 1% of labeled pixels and no ceiling")
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
