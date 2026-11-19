"""Export trained weights for release: <out>/<name>.safetensors (state dict, no pickle) and <name>.json (classes, preprocessing, provenance).

  python export_weights.py --run runs/vch --variant vch --name vch_segformer_b2_ade6 --out release
Load:  from vch_segformer import build; from safetensors.torch import load_file
       m = build("vch"); m.load_state_dict(load_file("vch_segformer_b2_ade6.safetensors")); m.eval()
       (build() fetches only the MiT-B2 architecture and ImageNet weights of nvidia/mit-b2, which the loaded state dict then replaces)
"""
import argparse, hashlib, json
from pathlib import Path
import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--variant", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--ckpt", default="best.pt")
    a = ap.parse_args(); run, out = Path(a.run), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(run / a.ckpt, map_location="cpu", weights_only=True)
    sd = {k: v.contiguous() for k, v in ck["model"].items()}
    f = out / f"{a.name}.safetensors"; save_file(sd, str(f))
    tm = json.loads((run / "test_metrics.json").read_text())
    meta = dict(name=a.name, variant=a.variant, architecture="SegFormer-B2 (MiT-B2 encoder, all-MLP decoder, 768 channels)"
                + {"vch": " + Vertical Context Gate + Hierarchical Vegetation Head", "vcg": " + Vertical Context Gate", "hvh": " + Hierarchical Vegetation Head", "base": ""}.get(a.variant, ""),
                code="code/seg/vch_segformer.py (build(variant)); training code/seg/train_seg.py", classes=["tree", "low vegetation", "sky", "ground", "built", "other"],
                ignore_index=255, output="log-probabilities of the six classes at 1/4 resolution (key 'logp'); upsample, then argmax",
                preprocessing=dict(rgb_mean=[123.675, 116.28, 103.53], rgb_std=[58.395, 57.12, 57.375], resize="short side 512 px", pad_to_multiple_of=32),
                training=dict(data="ADE20K SceneParsing, 19,696 training images, 6-class vegetation-structure labels", iterations=tm["iters"], batch_size=tm["bs"],
                              seed=tm.get("seed", 20260929), checkpoint_iteration=ck["it"], selection="highest vegetation IoU on 500 held-out outdoor training images"),
                test=dict(images=tm["n_test"], miou=tm["miou"], veg_iou=tm["veg_iou"], iou=tm["iou"]),
                sha256=hashlib.sha256(f.read_bytes()).hexdigest(), parameters=int(sum(v.numel() for v in sd.values())),
                license_note="Research and non-commercial use: the encoder was initialized from nvidia/mit-b2 (NVIDIA Source Code License for SegFormer) and trained on ADE20K (research use).")
    (out / f"{a.name}.json").write_text(json.dumps(meta, indent=1)); print(json.dumps({k: meta[k] for k in ("name", "sha256", "parameters")}))


if __name__ == "__main__":
    main()
