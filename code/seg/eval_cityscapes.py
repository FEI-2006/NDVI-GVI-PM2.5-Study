"""Zero-shot evaluation of the trained six-class networks on the 500 Cityscapes validation street views: an independent street-view check, with the labels
of a public data set and no new annotation. The networks are not trained or tuned on Cityscapes.

Cityscapes trainIds are mapped to the six classes as follows: vegetation (8) -> tree, terrain (9) -> low vegetation, sky (10), road and sidewalk (0, 1) -> ground,
building, wall, fence (2, 3, 4) -> built, every other class (5-7, 11-18) -> other; unlabeled (255) is ignored. The two vegetation classes differ in definition
from the ADE20K ones (Cityscapes counts shrubs and hedges as vegetation and soil as terrain), so the networks are scored on the merged classes
  vegetation (prediction: tree + low vegetation; label: vegetation + terrain), sky, ground, built, other   (five-class mIoU),
and on the per-image vegetation share (GVI; label: vegetation + terrain, and vegetation alone), which is the quantity the application needs.
Images are resized to a short side of 512 px (1024 x 512), labels with nearest-neighbour resampling.

  python eval_cityscapes.py --root <cityscapes_full> --runs runs [runs_new ...] --models unet,pspnet,deeplab,base,vch --out cityscapes_zeroshot.csv [--gpu 0]
(<root>/images/val/<name>.jpg and <root>/labels/val/<name>.png; run folders <runs>/<model>/best.pt; model names as in train_seg.py, a _s<seed> suffix is a further seed)
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_seg import build_model, predict, MEAN, STD          # noqa: E402

LUT = np.full(256, 255, np.uint8)
for tid, c in {8: 0, 9: 1, 10: 2, 0: 3, 1: 3, 2: 4, 3: 4, 4: 4}.items():
    LUT[tid] = c
for tid in (5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18):
    LUT[tid] = 5
MERGE = np.array([0, 0, 1, 2, 3, 4])                           # six classes -> vegetation, sky, ground, built, other
NAMES5 = ["vegetation", "sky", "ground", "built", "other"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--runs", required=True, nargs="+"); ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); dev = torch.device(f"cuda:{a.gpu}"); root = Path(a.root)
    names = sorted(p.stem for p in (root / "images" / "val").glob("*.jpg"))
    if a.limit: names = names[:a.limit]
    imgs = []                                                    # decode once, reuse for every model
    for n in names:
        im = Image.open(root / "images" / "val" / f"{n}.jpg").convert("RGB"); s = 512 / min(im.size)
        im = im.resize((int(im.width * s + 0.5), int(im.height * s + 0.5)), Image.BILINEAR)
        lab = Image.open(root / "labels" / "val" / f"{n}.png").resize(im.size, Image.NEAREST)
        imgs.append((torch.from_numpy(((np.asarray(im, np.float32) - MEAN) / STD).transpose(2, 0, 1).copy()).float()[None], LUT[np.asarray(lab)]))
    rows = []
    for v in a.models.split(","):
        ck = next((Path(r) / v / "best.pt" for r in a.runs if (Path(r) / v / "best.pt").exists()), None)
        if ck is None:
            print("no checkpoint for", v); continue
        m, _, _ = build_model(v.split("_s2026")[0]); m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)["model"]); m.to(dev).eval()
        conf = np.zeros((6, 6), np.int64); share = []
        for x, lab in imgs:
            pred = predict(m, x.to(dev), lab.shape).argmax(0).cpu().numpy(); ok = lab != 255
            conf += np.bincount(lab[ok].astype(np.int64) * 6 + pred[ok], minlength=36).reshape(6, 6)
            n = ok.sum(); share.append((float(((pred <= 1) & ok).sum() / n), float(((lab <= 1) & ok).sum() / n), float(((lab == 0) & ok).sum() / n)))
        c5 = np.zeros((5, 5), np.int64)
        for i in range(6):
            for j in range(6):
                c5[MERGE[i], MERGE[j]] += conf[i, j]
        tp = np.diag(c5).astype(float); iou = tp / np.maximum(c5.sum(0) + c5.sum(1) - tp, 1)
        sh = np.array(share); pv, tv, tt = sh[:, 0], sh[:, 1], sh[:, 2]
        # vegetation alone as the label (terrain counted as not vegetation): IoU of tree + low prediction against Cityscapes vegetation (8)
        tp_v = conf[0, :2].sum(); fn_v = conf[0, 2:].sum(); fp_v = conf[1:, :2].sum()
        rows.append(dict(model=v, miou5=iou.mean(), oa=tp.sum() / c5.sum(), **{f"iou_{k}": float(x) for k, x in zip(NAMES5, iou)}, veg_iou_vegonly=float(tp_v / max(tp_v + fn_v + fp_v, 1)),
                         gvi_mae=float(np.abs(pv - tv).mean()), gvi_r=float(np.corrcoef(pv, tv)[0, 1]), gvi_bias=float((pv - tv).mean()),
                         gvi_mae_vegonly=float(np.abs(pv - tt).mean()), gvi_r_vegonly=float(np.corrcoef(pv, tt)[0, 1]), gvi_bias_vegonly=float((pv - tt).mean()), n_images=len(names)))
        print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in rows[-1].items()}, flush=True)
        del m; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
