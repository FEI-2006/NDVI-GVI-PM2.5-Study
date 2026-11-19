"""Apply a trained vegetation-structure model to the retained KartaView photos (resized by prepare_kv_inputs.py).

Per image: the share of all pixels predicted as each class, so tree_gvi = tree / T, low_gvi = low vegetation / T, and
gvi_vs = (tree + low vegetation) / T, with T the number of pixels (the denominator of the GVI in the paper).

  python apply_kv.py --ckpt runs/vch/best.pt --variant vch --img data/kv512 --out runs/vch/kv_shares.csv [--gpu 1]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_seg import build_model, predict, MEAN, STD          # noqa: E402

NAMES = ["tree", "low", "sky", "ground", "built", "other"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--variant", required=True)
    ap.add_argument("--img", required=True); ap.add_argument("--out", required=True); ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args(); dev = torch.device(f"cuda:{a.gpu}")
    model, _, _ = build_model(a.variant); model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True)["model"]); model.to(dev).eval()
    rows = []
    for f in sorted(Path(a.img).glob("*.jpg")):
        im = Image.open(f).convert("RGB")
        x = torch.from_numpy(((np.asarray(im, np.float32) - MEAN) / STD).transpose(2, 0, 1).copy()).float()[None].to(dev)
        pred = predict(model, x, (im.height, im.width)).argmax(0).cpu().numpy()
        cnt = np.bincount(pred.ravel(), minlength=6) / pred.size
        rows.append(dict(id=f.stem, **{f"{n}": float(c) for n, c in zip(NAMES, cnt)}))
    df = pd.DataFrame(rows); df["gvi_vs"] = df.tree + df.low; df.to_csv(a.out, index=False)
    print(len(df), df[["tree", "low", "gvi_vs"]].mean().round(4).to_dict())


if __name__ == "__main__":
    main()
