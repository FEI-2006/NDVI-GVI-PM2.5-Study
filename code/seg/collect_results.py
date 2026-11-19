"""Collect the segmentation experiments (copied from the server into results/seg/<variant>/) into result tables.

Reads results/seg/<variant>/test_metrics.json and log.jsonl and results/seg/complexity.csv.
Writes results/tables/seg_accuracy.csv (one row per model), seg_curves.csv (selection-split vegetation IoU during training).

  python collect_results.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, TABLES as T                          # noqa: E402

S = ROOT / "results" / "seg"
ORDER = [("unet", "U-Net"), ("pspnet", "PSPNet"), ("deeplab", "DeepLabV3+"), ("upcnxt", "UperNet-ConvNeXt-T"), ("upswint", "UperNet-Swin-T"),
         ("base", "SegFormer-B2"), ("vcg", "SegFormer-B2 + VCG"), ("hvh", "SegFormer-B2 + HVH"), ("vch", "VCH-SegFormer-B2"),
         ("base_b5", "SegFormer-B5"), ("vch_b5", "VCH-SegFormer-B5")]
cx = pd.read_csv(S / "complexity.csv").set_index("variant") if (S / "complexity.csv").exists() else None
rows, curves = [], []
for v, name in ORDER:
    f = S / v / "test_metrics.json"
    if not f.exists(): print("missing", v); continue
    m = json.loads(f.read_text())
    r = dict(variant=v, model=name, vcg=v in ("vcg", "vch"), hvh=v in ("hvh", "vch"), miou=m["miou"], oa=m["oa"], veg_iou=m["veg_iou"], veg_precision=m["veg_precision"],
             veg_recall=m["veg_recall"], veg_f1=m["veg_f1"], veg_oa=m["veg_oa"], gvi_mae=m["gvi_mae"], gvi_r=m["gvi_r"], tree_mae=m["tree_mae"], low_mae=m["low_mae"],
             allval_miou=m.get("allval_miou"), best_it=m["best_it"], select_veg_iou=m["select_veg_iou"], n_train=m["n_train"], n_select=m["n_select"], n_test=m["n_test"],
             n_allval=m.get("n_allval"), train_hours=m["train_hours"], peak_mem_gb=m.get("peak_mem_gb"), **{f"iou_{k}": x for k, x in m["iou"].items()})
    if cx is not None and v in cx.index: r.update(params_m=cx.loc[v, "params_m"], gflops=cx.loc[v, "gflops"], ms_per_image=cx.loc[v, "ms_per_image"])
    rows.append(r)
    for l in (S / v / "log.jsonl").read_text().splitlines():
        d = json.loads(l)
        if "select_veg_iou" in d: curves.append(dict(variant=v, it=d["it"], select_veg_iou=d["select_veg_iou"], select_miou=d["select_miou"]))
# three-run means (checkpoint rule) of the SegFormer-B2 baseline and VCH-SegFormer: the rows the paper compares; the single runs above stay available
for v, name in (("base", "SegFormer-B2"), ("vch", "VCH-SegFormer-B2"), ("base_b5", "SegFormer-B5"), ("vch_b5", "VCH-SegFormer-B5")):
    ms = [json.loads((d / "test_metrics.json").read_text()) for d in (S / v, S / f"{v}_s20260930", S / f"{v}_s20260931") if (d / "test_metrics.json").exists()]
    if len(ms) < 2: continue
    flat = [dict(miou=m["miou"], oa=m["oa"], veg_iou=m["veg_iou"], veg_precision=m["veg_precision"], veg_recall=m["veg_recall"], veg_f1=m["veg_f1"], gvi_mae=m["gvi_mae"],
                 tree_mae=m["tree_mae"], low_mae=m["low_mae"], iou_tree=m["iou"]["tree"], iou_low=m["iou"]["low"]) for m in ms]
    fd = pd.DataFrame(flat); r = dict(variant=f"{v}_mean", model=name, n_runs=len(ms), n_train=ms[0]["n_train"], n_test=ms[0]["n_test"])
    r.update({c: fd[c].mean() for c in fd}); r.update({c + "_sd": fd[c].std(ddof=1) for c in fd})
    if cx is not None and v in cx.index: r.update(params_m=cx.loc[v, "params_m"], gflops=cx.loc[v, "gflops"], ms_per_image=cx.loc[v, "ms_per_image"])
    rows.append(r)
pd.DataFrame(rows).to_csv(T / "seg_accuracy.csv", index=False); pd.DataFrame(curves).to_csv(T / "seg_curves.csv", index=False)

# where the errors of the tree and low-vegetation classes go (pixels pooled over the three runs, checkpoint rule): the premise of HVH
NAMES = ["tree", "low", "sky", "ground", "built", "other"]; err_rows = []
for v in ("base", "vch", "base_b5", "vch_b5"):
    fs = [S / d / "test_confusion.npz" for d in (v, f"{v}_s20260930", f"{v}_s20260931") if (S / d / "test_confusion.npz").exists()]
    if not fs: continue
    c = sum(np.load(f)["conf"].sum(0) for f in fs).astype(float)
    for k in (0, 1):
        e = c[k].copy(); e[k] = 0
        err_rows += [dict(variant=v, true_class=NAMES[k], error_rate=e.sum() / c[k].sum(), predicted_as=NAMES[j], share_of_errors=e[j] / e.sum()) for j in range(6) if j != k]
if err_rows: pd.DataFrame(err_rows).to_csv(T / "seg_confusion_errors.csv", index=False)

# robustness of the ablation: every SegFormer run (first seed in runs/<v>, second seed in runs/<v>_s20260930), checkpoint chosen by the rule
# ("rule", test_metrics.json) and weights at the last iteration ("final", test_metrics_final.json; for the first seed only where the rule chose 16,000)
seeds = []
for v in ("base", "vcg", "hvh", "vch", "base_b5", "vch_b5"):
    for d, seed in ((S / v, 20260929), (S / f"{v}_s20260930", 20260930), (S / f"{v}_s20260931", 20260931)):
        for ck, fn in (("rule", "test_metrics.json"), ("final", "test_metrics_final.json")):
            f = d / fn
            if not f.exists() and ck == "final" and (d / "test_metrics.json").exists() and json.loads((d / "test_metrics.json").read_text())["best_it"] == 16000:
                f = d / "test_metrics.json"                    # the rule chose the last iteration, so the final weights are the chosen ones
            if not f.exists(): continue
            m = json.loads(f.read_text())
            seeds.append(dict(variant=v, seed=seed, checkpoint=ck, it=m["best_it"], miou=m["miou"], veg_iou=m["veg_iou"], veg_f1=m["veg_f1"], iou_tree=m["iou"]["tree"],
                              iou_low=m["iou"]["low"], gvi_mae=m["gvi_mae"], tree_mae=m["tree_mae"], low_mae=m["low_mae"], allval_miou=m.get("allval_miou")))
if seeds:
    pd.DataFrame(seeds).to_csv(T / "seg_seeds.csv", index=False); print(pd.DataFrame(seeds).round(4).to_string())


def pooled(c):
    """c: 6 x 6 confusion (truth x prediction) -> mIoU, vegetation IoU, tree IoU, low-vegetation IoU"""
    tp = np.diag(c).astype(float); iou = tp / np.maximum(c.sum(0) + c.sum(1) - tp, 1)
    vt, vp, vtp = c[:2].sum(), c[:, :2].sum(), c[:2, :2].sum()
    return np.array([iou.mean(), vtp / max(vt + vp - vtp, 1), iou[0], iou[1]])


def full(cs):
    """cs: n_images x 6 x 6 per-image confusion -> the metrics of test_metrics.json (pooled over pixels; GVI MAE over images)"""
    c = cs.sum(0); tp = np.diag(c).astype(float); iou = tp / np.maximum(c.sum(0) + c.sum(1) - tp, 1)
    vt, vp, vtp = c[:2].sum(), c[:, :2].sum(), c[:2, :2].sum(); prec, rec = vtp / max(vp, 1), vtp / max(vt, 1)
    n = cs.sum((1, 2)).astype(float); gt, gp = cs[:, :2].sum((1, 2)) / n, cs[:, :, :2].sum((1, 2)) / n
    return dict(miou=iou.mean(), oa=tp.sum() / c.sum(), veg_iou=vtp / max(vt + vp - vtp, 1), veg_precision=prec, veg_recall=rec, veg_f1=2 * prec * rec / max(prec + rec, 1e-9),
                iou_tree=iou[0], iou_low=iou[1], gvi_mae=float(np.abs(gp - gt).mean()), gvi_r=float(np.corrcoef(gp, gt)[0, 1]))


# both decoding rules (code/seg/eval_perimage.py): argmax (primary, = test_metrics.json) and vegetation-first (secondary)
conf = {v: np.load(S / v / "test_confusion.npz") for v, _ in ORDER if (S / v / "test_confusion.npz").exists()}
dec = [dict(variant=v, decoding=d, **full(conf[v][k])) for v in conf for d, k in (("argmax", "conf"), ("vegetation-first", "conf_vf")) if k in conf[v].files]
if dec:
    pd.DataFrame(dec).to_csv(T / "seg_decoding.csv", index=False); print(pd.DataFrame(dec).round(4).to_string())
    for v in conf:                                     # the per-image matrices must reproduce the metrics written at the end of training
        a_ = next(r for r in rows if r["variant"] == v); b_ = next(d for d in dec if d["variant"] == v and d["decoding"] == "argmax")
        assert abs(a_["veg_iou"] - b_["veg_iou"]) < 1e-6 and abs(a_["miou"] - b_["miou"]) < 1e-6, v

# paired bootstrap over test images (2,000 resamples, seed 20260929) of the difference between two models; training-seed variation is not included
boot = []
if conf:
    rng = np.random.default_rng(20260929); n = len(next(iter(conf.values()))["names"]); idx = [rng.integers(0, n, n) for _ in range(2000)]
    for d, key in (("argmax", "conf"), ("vegetation-first", "conf_vf")):
        for a, b in (("vcg", "base"), ("hvh", "base"), ("vch", "base"), ("vch", "vcg"), ("vch", "hvh"), ("vch", "deeplab"), ("vch", "pspnet"), ("vch", "unet"), ("base", "deeplab"),
                     ("vch_b5", "base_b5"), ("vch_b5", "vch"), ("vch_b5", "upcnxt"), ("vch_b5", "upswint"), ("vch_b5", "deeplab"), ("vch", "upcnxt"), ("vch", "upswint"),
                     ("base_b5", "base"), ("base_b5", "upcnxt"), ("base_b5", "upswint")):
            if a not in conf or b not in conf or key not in conf[a].files: continue
            ca, cb = conf[a][key], conf[b][key]; assert (conf[a]["names"] == conf[b]["names"]).all()
            d0 = pooled(ca.sum(0)) - pooled(cb.sum(0)); ds = np.array([pooled(ca[i].sum(0)) - pooled(cb[i].sum(0)) for i in idx])
            for j, mname in enumerate(("miou", "veg_iou", "iou_tree", "iou_low")):
                boot.append(dict(decoding=d, a=a, b=b, metric=mname, diff=d0[j], lo=np.percentile(ds[:, j], 2.5), hi=np.percentile(ds[:, j], 97.5),
                                 share_positive=float((ds[:, j] > 0).mean())))
    # further seeds: <v><n> = checkpoint chosen by the rule, <v><n>f = weights at the last iteration (the same file where the rule chose the last iteration)
    c2 = {}
    for n, suf in (("2", "_s20260930"), ("3", "_s20260931")):
        for v in ("base", "vcg", "hvh", "vch", "base_b5", "vch_b5"):
            d_ = S / f"{v}{suf}"
            if (d_ / "test_confusion.npz").exists():
                c2[v + n] = np.load(d_ / "test_confusion.npz")
                it_ = json.loads((d_ / "test_metrics.json").read_text())["best_it"]
                c2[v + n + "f"] = np.load(d_ / "test_confusion_final.npz") if (d_ / "test_confusion_final.npz").exists() else (c2[v + n] if it_ == 16000 else None)
    for n in ("2", "3"):
        for a, b in ((f"vcg{n}", f"base{n}"), (f"hvh{n}", f"base{n}"), (f"vch{n}", f"base{n}"), (f"vcg{n}f", f"base{n}f"), (f"hvh{n}f", f"base{n}f"), (f"vch{n}f", f"base{n}f"),
                     (f"vch_b5{n}", f"base_b5{n}"), (f"vch_b5{n}f", f"base_b5{n}f")):
            if c2.get(a) is None or c2.get(b) is None: continue
            ca, cb = c2[a]["conf"], c2[b]["conf"]; assert (c2[a]["names"] == c2[b]["names"]).all()
            d0 = pooled(ca.sum(0)) - pooled(cb.sum(0)); ds = np.array([pooled(ca[i].sum(0)) - pooled(cb[i].sum(0)) for i in idx])
            for j, mname in enumerate(("miou", "veg_iou", "iou_tree", "iou_low")):
                boot.append(dict(decoding="argmax", a=a, b=b, metric=mname, diff=d0[j], lo=np.percentile(ds[:, j], 2.5), hi=np.percentile(ds[:, j], 97.5),
                                 share_positive=float((ds[:, j] > 0).mean())))
    pd.DataFrame(boot).to_csv(T / "seg_bootstrap.csv", index=False); print(pd.DataFrame(boot).round(4).to_string())
print(pd.DataFrame(rows)[["model", "miou", "oa", "veg_iou", "veg_f1", "iou_tree", "iou_low", "gvi_mae"]].round(4).to_string())
