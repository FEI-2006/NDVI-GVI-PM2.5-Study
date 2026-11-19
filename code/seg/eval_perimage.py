"""Per-image confusion matrices on the test split for every trained model (for paired bootstrap intervals of metric differences over test images).

Two decoding rules, applied to the six class log-probabilities of every model:
  argmax            the class with the highest probability (primary; used for checkpoint selection and test_metrics.json);
  vegetation-first  the super-class with the highest probability, with p(vegetation) = p(tree) + p(low vegetation), and within vegetation
                    the larger of p(tree) and p(low vegetation) (secondary; the top-down rule that matches the vegetation-first design of HVH).

  python eval_perimage.py --data data/vegstruct --runs runs [--gpu 1]
Writes runs/<variant>/test_confusion.npz (test_confusion_final.npz with --ckpt final): names (test image names), conf and conf_vf (n_images x 6 x 6 pixel counts, rows = truth, columns = prediction).
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_seg import build_model, load_eval, predict           # noqa: E402
from vch_segformer import N_CLASSES                              # noqa: E402


def vegetation_first(lp):
    """lp: 6 x H x W log-probabilities -> labels decoded top-down (vegetation versus the other super-classes, then tree versus low)"""
    p = lp.exp(); sup = torch.cat([p[:2].sum(0, keepdim=True), p[2:]], 0).argmax(0)
    return torch.where(sup == 0, (p[1] > p[0]).long(), sup + 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--variants", default="unet,pspnet,deeplab,upcnxt,upswint,base,vcg,hvh,vch,base_b5,vch_b5", help="run directories; <variant>_s<seed> for other seeds")
    ap.add_argument("--ckpt", default="best", choices=["best", "final"], help="best.pt (checkpoint rule) or final.pt (last iteration)")
    a = ap.parse_args(); dev = torch.device(f"cuda:{a.gpu}")
    te = pd.read_csv(Path(a.data) / "splits.csv"); te = te[te.split == "test"]
    for v in a.variants.split(","):
        ck = Path(a.runs) / v / f"{a.ckpt}.pt"
        if not ck.exists(): print("skip", v); continue
        m, _, _ = build_model(v.split("_s")[0]); m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)["model"]); m.to(dev).eval()
        conf, conf_vf = (np.zeros((len(te), N_CLASSES, N_CLASSES), np.int64) for _ in range(2))
        for k, r in enumerate(te.itertuples()):
            x, lab = load_eval(r); lp = predict(m, x.to(dev), lab.shape); ok = lab != 255
            for c, pred in ((conf, lp.argmax(0)), (conf_vf, vegetation_first(lp))):
                c[k] = np.bincount(lab[ok] * N_CLASSES + pred.cpu().numpy()[ok], minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
        np.savez_compressed(Path(a.runs) / v / ("test_confusion.npz" if a.ckpt == "best" else "test_confusion_final.npz"), names=np.array([str(x) for x in te.name], dtype="U"), conf=conf, conf_vf=conf_vf)
        print(v, conf.sum(0).trace() / conf.sum(), conf_vf.sum(0).trace() / conf_vf.sum(), flush=True)
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
