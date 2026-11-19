"""Train and evaluate the vegetation-structure segmentation models (VCH-SegFormer ablation variants and a DeepLabV3+ baseline).

All variants use the same data, crop size, schedule, iterations and seed; checkpoints are chosen on the selection split only,
and the test split (ADE20K validation images passing the outdoor filter) is evaluated once, with the chosen checkpoint.

  python train_seg.py --variant vch --data <vegstruct dir> --out <run dir> [--iters 16000 --bs 8 --gpu 0]
variants: base | vcg | hvh | vch (VCH-SegFormer) | deeplab (DeepLabV3+, Chen et al., ECCV 2018) | unet (U-Net, Ronneberger et al., MICCAI 2015)
          | pspnet (PSPNet, Zhao et al., CVPR 2017); the three comparison models use an ImageNet ResNet-50 encoder
          | base_b3 / base_b4 / base_b5, vch_b3 / vch_b4 / vch_b5, ... (the SegFormer variants with the larger MiT-B3..B5 encoders, ImageNet-1k)
          | upcnxt / upcnxs / upcnxb (UperNet with a ConvNeXt-T / S / B encoder) and upswint / upswins / upswinb (UperNet with a Swin-T / S / B encoder), ImageNet-1k
Outputs in <out>: log.jsonl, best.pt, final.pt, test_metrics.json (checkpoint chosen on the selection split; test = outdoor validation images;
allval_* = all 2,000 validation images), test_metrics_final.json (weights at the last iteration), test_images[_final].csv (per-image shares)
"""
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from vch_segformer import build, encoder_source, loss_fn, N_CLASSES          # noqa: E402

MEAN, STD = np.array([0.485, 0.456, 0.406]) * 255, np.array([0.229, 0.224, 0.225]) * 255
CROP = 512


class TrainSet(Dataset):
    def __init__(self, df, seed):
        self.df = df.reset_index(drop=True); self.seed = seed

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]; img = Image.open(r.image).convert("RGB"); lab = Image.open(r.label)
        s = random.uniform(0.5, 2.0) * CROP / min(img.size); size = (max(int(img.width * s + 0.5), 1), max(int(img.height * s + 0.5), 1))
        img, lab = img.resize(size, Image.BILINEAR), lab.resize(size, Image.NEAREST)
        x, y = np.asarray(img, np.float32), np.asarray(lab, np.int64)
        ph, pw = max(CROP - y.shape[0], 0), max(CROP - y.shape[1], 0)
        if ph or pw:
            x = np.pad(x, ((0, ph), (0, pw), (0, 0)), constant_values=0); x[y.shape[0]:, :] = MEAN; x[:, y.shape[1]:] = MEAN
            y = np.pad(y, ((0, ph), (0, pw)), constant_values=255)
        t, l = random.randint(0, x.shape[0] - CROP), random.randint(0, x.shape[1] - CROP)
        x, y = x[t:t + CROP, l:l + CROP], y[t:t + CROP, l:l + CROP]
        if random.random() < 0.5: x, y = x[:, ::-1], y[:, ::-1]
        x = x * random.uniform(0.9, 1.1) + random.uniform(-10, 10)                      # mild photometric jitter
        x = (x - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float(), torch.from_numpy(np.ascontiguousarray(y))


def load_eval(r):
    img = Image.open(r.image).convert("RGB"); lab = np.asarray(Image.open(r.label), np.int64)
    s = CROP / min(img.size); img = img.resize((int(img.width * s + 0.5), int(img.height * s + 0.5)), Image.BILINEAR)
    x = (np.asarray(img, np.float32) - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).float()[None], lab


@torch.no_grad()
def predict(model, x, out_hw):
    H, W = x.shape[-2:]; ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
    xp = F.pad(x, (0, pw, 0, ph))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lp = model(xp)["logp"].float()
    lp = F.interpolate(lp, size=xp.shape[-2:], mode="bilinear", align_corners=False)[..., :H, :W]
    return F.interpolate(lp, size=out_hw, mode="bilinear", align_corners=False)[0]


def evaluate(model, df, dev, per_image=False):
    model.eval(); conf = np.zeros((N_CLASSES, N_CLASSES), np.int64); rows = []
    for r in df.itertuples():
        x, lab = load_eval(r); lp = predict(model, x.to(dev), lab.shape)
        pred = lp.argmax(0).cpu().numpy(); m = lab != 255
        conf += np.bincount(lab[m] * N_CLASSES + pred[m], minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
        if per_image:
            n = m.sum()
            rows.append(dict(name=r.name, true_tree=float((lab[m] == 0).sum() / n), true_low=float((lab[m] == 1).sum() / n),
                             pred_tree=float((pred[m] == 0).sum() / n), pred_low=float((pred[m] == 1).sum() / n)))
    model.train()
    return metrics(conf), pd.DataFrame(rows)


def metrics(conf):
    tp = np.diag(conf).astype(float); iou = tp / np.maximum(conf.sum(0) + conf.sum(1) - tp, 1)
    veg_t, veg_p = conf[:2].sum(1).sum(), conf[:, :2].sum(0).sum(); veg_tp = conf[:2, :2].sum()
    prec, rec = veg_tp / max(veg_p, 1), veg_tp / max(veg_t, 1)
    tot = conf.sum(); veg_tn = tot - veg_t - veg_p + veg_tp
    return dict(miou=float(iou.mean()), oa=float(tp.sum() / tot), iou={c: float(v) for c, v in zip(["tree", "low", "sky", "ground", "built", "other"], iou)},
                veg_iou=float(veg_tp / max(veg_t + veg_p - veg_tp, 1)), veg_precision=float(prec), veg_recall=float(rec),
                veg_f1=float(2 * prec * rec / max(prec + rec, 1e-9)), veg_oa=float((veg_tp + veg_tn) / tot), confusion=conf.tolist())


CNN = {"deeplab": "DeepLabV3Plus", "unet": "Unet", "pspnet": "PSPNet"}     # comparison models, all with an ImageNet ResNet-50 encoder (timm)
UPERNET = {"upcnxt": ("convnext", "facebook/convnext-tiny-224"), "upcnxs": ("convnext", "facebook/convnext-small-224"),     # UperNet (Xiao et al., ECCV 2018) with
           "upcnxb": ("convnext", "facebook/convnext-base-224"), "upswint": ("swin", "microsoft/swin-tiny-patch4-window7-224"),   # an ImageNet-1k ConvNeXt (Liu et al., CVPR 2022)
           "upswins": ("swin", "microsoft/swin-small-patch4-window7-224"), "upswinb": ("swin", "microsoft/swin-base-patch4-window7-224")}  # or Swin (Liu et al., ICCV 2021) encoder


def build_upernet(variant):
    from transformers import AutoBackbone, ConvNextConfig, SwinConfig, UperNetConfig, UperNetForSemanticSegmentation
    kind, ck = UPERNET[variant]; ck = encoder_source(ck); outs = ["stage1", "stage2", "stage3", "stage4"]
    bcfg = (ConvNextConfig if kind == "convnext" else SwinConfig).from_pretrained(ck, out_features=outs)
    net = UperNetForSemanticSegmentation(UperNetConfig(backbone_config=bcfg, num_labels=N_CLASSES, use_auxiliary_head=False))
    net.backbone.load_state_dict(AutoBackbone.from_pretrained(ck, out_features=outs).state_dict())

    class UP(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.net = net

        def forward(self, x):
            return {"logp": F.log_softmax(self.net(pixel_values=x).logits.float(), 1)}
    return UP()


def build_model(variant):
    if variant in UPERNET:
        m = build_upernet(variant); enc = list(m.net.backbone.parameters()); ids = {id(p) for p in enc}
        return m, enc, [p for p in m.parameters() if id(p) not in ids]
    if variant in CNN:
        import segmentation_models_pytorch as smp

        class DL(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.net = getattr(smp, CNN[variant])("tu-resnet50", encoder_weights="imagenet", classes=N_CLASSES)

            def forward(self, x):
                return {"logp": F.log_softmax(self.net(x).float(), 1)}
        m = DL(); enc = list(m.net.encoder.parameters()); ids = {id(p) for p in enc}
        return m, enc, [p for p in m.parameters() if id(p) not in ids]
    m = build(variant); enc = list(m.encoder.parameters()); ids = {id(p) for p in enc}
    return m, enc, [p for p in m.parameters() if id(p) not in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=16000); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--eval_every", type=int, default=2000); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit_eval", type=int, default=0, help="smoke tests only: evaluate on the first N images")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev)
    df = pd.read_csv(Path(a.data) / "splits.csv")
    tr, sel, te, allval = df[df.split == "train"], df[df.split == "select"], df[df.split == "test"], df[df.split.isin(["test", "val_other"])]
    if a.limit_eval: sel, te, allval = sel[:a.limit_eval], te[:a.limit_eval], allval[:a.limit_eval]
    model, enc, dec = build_model(a.variant); model.to(dev)
    lr_dec = a.lr * 10                                   # one protocol for all models: new layers at 10x the encoder rate
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr}, {"params": dec, "lr": lr_dec}], lr=a.lr, weight_decay=0.01)
    base_lrs = [a.lr, lr_dec]; warm = 1000
    dl = DataLoader(TrainSet(tr, a.seed), batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=True,
                    worker_init_fn=lambda w: (random.seed(a.seed + w), np.random.seed(a.seed + w)))
    it, best, t0, log = 0, -1.0, time.time(), open(out / "log.jsonl", "a")
    if (out / "last.pt").exists():                     # resume after an interruption (model, optimizer, iteration, best score); the data order after the resume point differs
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=True)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); it, best = ck["it"], ck["best"]
        random.seed(a.seed + it); np.random.seed(a.seed + it); torch.manual_seed(a.seed + it)
        log.write(json.dumps(dict(resumed_at=it, best=best)) + "\n"); log.flush()
    model.train()
    while it < a.iters:
        for x, y in dl:
            f = min(1.0, (it + 1) / warm) * (1 - it / a.iters) ** 1.0
            for g, b in zip(opt.param_groups, base_lrs): g["lr"] = b * f
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(x)
            loss = loss_fn(o, y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); it += 1       # HVH, batch without vegetation: NaN fine-head term, no gradient from it
            if it % 100 == 0:
                log.write(json.dumps(dict(it=it, loss=float(loss), lr=opt.param_groups[0]["lr"], sec=round(time.time() - t0, 1))) + "\n"); log.flush()
            if it % a.eval_every == 0 or it == a.iters:
                m, _ = evaluate(model, sel, dev)
                log.write(json.dumps(dict(it=it, select_veg_iou=m["veg_iou"], select_miou=m["miou"], sec=round(time.time() - t0, 1))) + "\n"); log.flush()
                if m["veg_iou"] > best:
                    best = m["veg_iou"]; torch.save(dict(model=model.state_dict(), it=it, select=m), out / "best.pt")
            if it % 1000 == 0 and it < a.iters:
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), it=it, best=best), out / "last.tmp"); os.replace(out / "last.tmp", out / "last.pt")
            if it >= a.iters: break
    torch.save(dict(model=model.state_dict(), it=it), out / "final.pt")        # weights at the last iteration (fixed-iteration comparison)
    for tag, f in (("", "best.pt"), ("_final", "final.pt")):                    # primary: checkpoint chosen on the selection split; secondary: last iteration
        ck = torch.load(out / f, map_location=dev, weights_only=True); model.load_state_dict(ck["model"])
        m, per = evaluate(model, te, dev, per_image=True)
        per.to_csv(out / f"test_images{tag}.csv", index=False)
        ma, _ = evaluate(model, allval, dev); m.update(allval_miou=ma["miou"], allval_iou=ma["iou"], n_allval=len(allval))
        gvi_t, gvi_p = per.true_tree + per.true_low, per.pred_tree + per.pred_low
        m.update(variant=a.variant, seed=a.seed, best_it=ck["it"], select_veg_iou=ck["select"]["veg_iou"] if "select" in ck else None, n_train=len(tr), n_select=len(sel),
                 n_test=len(te), iters=a.iters, bs=a.bs, gvi_mae=float((gvi_p - gvi_t).abs().mean()), gvi_r=float(np.corrcoef(gvi_p, gvi_t)[0, 1]),
                 tree_mae=float((per.pred_tree - per.true_tree).abs().mean()), low_mae=float((per.pred_low - per.true_low).abs().mean()),
                 params_m=sum(p.numel() for p in model.parameters()) / 1e6, train_hours=(time.time() - t0) / 3600, peak_mem_gb=torch.cuda.max_memory_allocated(dev) / 1e9)
        if tag == "":
            model.eval(); xb = torch.randn(1, 3, CROP, CROP, device=dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(10): model(xb)
                torch.cuda.synchronize(); t1 = time.time()
                for _ in range(50): model(xb)
                torch.cuda.synchronize()
            m["fps_512"] = 50 / (time.time() - t1)
        (out / f"test_metrics{tag}.json").write_text(json.dumps(m, indent=1))
        print(tag or "best", json.dumps({k: v for k, v in m.items() if k not in ("confusion", "allval_iou")}))
    (out / "last.pt").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
