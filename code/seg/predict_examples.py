"""Label maps for the figures: (1) five test images drawn at random (seed 20260929) among those whose true vegetation share is at least 15%,
predicted by every trained model; (2) six retained KartaView photos drawn at random (seed 20260926), predicted by the headline VCH-SegFormer.
Images are stored with a short side of 512 px.

  python predict_examples.py --data data/vegstruct --runs runs [runs_new ...] --kv data/kv/img512 --out runs/examples [--models ...] [--kv-model vch_b5] [--gpu 1]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_seg import build_model, load_eval, predict, MEAN, STD          # noqa: E402

MODELS = ["unet", "pspnet", "deeplab", "upcnxt", "upswint", "base", "vch", "base_b5", "vch_b5"]


def small(im, resample):
    s = 512 / min(im.size); return im.resize((int(im.width * s + 0.5), int(im.height * s + 0.5)), resample)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--runs", required=True, nargs="+", help="folders with <variant>/best.pt (searched in this order)"); ap.add_argument("--kv", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--models", default=",".join(MODELS)); ap.add_argument("--kv-model", default="vch_b5")
    a = ap.parse_args(); dev = torch.device(f"cuda:{a.gpu}"); out = Path(a.out); (out / "ade").mkdir(parents=True, exist_ok=True); (out / "kv").mkdir(exist_ok=True)
    df = pd.read_csv(Path(a.data) / "splits.csv"); te = df[(df.split == "test") & (df.share_tree + df.share_low_vegetation >= 0.15)]
    ex = te.sample(5, random_state=20260929); ex[["name", "share_tree", "share_low_vegetation"]].to_csv(out / "ade_examples.csv", index=False)
    for r in ex.itertuples():
        small(Image.open(r.image).convert("RGB"), Image.BILINEAR).save(out / "ade" / f"{r.name}_img.jpg", quality=92)
        small(Image.open(r.label), Image.NEAREST).save(out / "ade" / f"{r.name}_gt.png")
    kv = sorted(p.stem for p in Path(a.kv).glob("*.jpg")); kv_ids = list(np.random.default_rng(20260926).choice(kv, 6, replace=False))
    pd.DataFrame(dict(id=kv_ids)).to_csv(out / "kv_examples.csv", index=False)
    for v in a.models.split(","):
        ck = next((Path(r) / v / "best.pt" for r in a.runs if (Path(r) / v / "best.pt").exists()), None)
        if ck is None: print("no checkpoint for", v); continue
        m, _, _ = build_model(v); m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)["model"]); m.to(dev).eval()
        for r in ex.itertuples():
            x, lab = load_eval(r); p = predict(m, x.to(dev), lab.shape).argmax(0).cpu().numpy().astype(np.uint8)
            small(Image.fromarray(p), Image.NEAREST).save(out / "ade" / f"{r.name}_{v}.png")
        if v == a.kv_model:
            for i in kv_ids:
                im = Image.open(Path(a.kv) / f"{i}.jpg").convert("RGB")
                x = torch.from_numpy(((np.asarray(im, np.float32) - MEAN) / STD).transpose(2, 0, 1).copy()).float()[None].to(dev)
                Image.fromarray(predict(m, x, (im.height, im.width)).argmax(0).cpu().numpy().astype(np.uint8)).save(out / "kv" / f"{i}_vch.png")
        del m; torch.cuda.empty_cache()
    for i in kv_ids:
        Image.open(Path(a.kv) / f"{i}.jpg").convert("RGB").save(out / "kv" / f"{i}_img.jpg", quality=92)
    print("done", len(ex), len(kv_ids))


if __name__ == "__main__":
    main()
