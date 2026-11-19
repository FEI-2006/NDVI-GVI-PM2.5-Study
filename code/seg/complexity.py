"""Complexity of every trained model under the same conditions: parameters, FLOPs, and mean inference time for one 512 x 512 image.

FLOPs are counted with torch.utils.flop_counter (multiply-add = 2 FLOPs; convolutions, matrix products and attention).  Time is the
mean over 100 forward passes after 20 warm-up passes (batch 1, bf16 autocast, one GPU; the server is shared, so times are indicative).

  python complexity.py --out runs/complexity.csv [--gpu 1]
"""
import argparse, sys, time
from pathlib import Path
import pandas as pd, torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_seg import build_model                              # noqa: E402

VARIANTS = ["unet", "pspnet", "deeplab", "upcnxt", "upswint", "base", "vcg", "hvh", "vch", "base_b5", "vch_b5"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args(); dev = torch.device(f"cuda:{a.gpu}"); rows = []
    x = torch.randn(1, 3, 512, 512, device=dev)
    for v in VARIANTS:
        m = build_model(v)[0].to(dev).eval()
        with torch.no_grad():
            fc = FlopCounterMode(display=False)
            with fc: m(x)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(20): m(x)
                torch.cuda.synchronize(); t = time.time()
                for _ in range(100): m(x)
                torch.cuda.synchronize()
        rows.append(dict(variant=v, params_m=sum(p.numel() for p in m.parameters()) / 1e6, gflops=fc.get_total_flops() / 1e9, ms_per_image=(time.time() - t) * 10))
        print(rows[-1], flush=True); del m; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
