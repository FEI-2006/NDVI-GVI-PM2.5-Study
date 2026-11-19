"""VCH-SegFormer: SegFormer-B2 with a Vertical Context Gate (VCG) and a Hierarchical Vegetation Head (HVH) for vegetation-structure
segmentation of street-level images. Both modules can be switched off, which gives the ablation variants:
  base      SegFormer-B2 encoder (MiT-B2, ImageNet-1k) + all-MLP decoder + flat 6-class classifier   (Xie et al., NeurIPS 2021)
  +VCG      the fused decoder feature is re-weighted per image row by a gate computed from width-pooled context along the image height
            (motivated by height-driven attention, Choi et al., CVPR 2020; here a multiplicative gate initialized to identity,
            with a learned position table along the height and dilated 1-D convolutions)
  +HVH      the classifier predicts super-classes {vegetation, sky, ground, built, other} and, within vegetation, {tree, low vegetation};
            class probabilities are composed as p(tree) = p(veg) p(tree | veg), trained with the composed likelihood plus both levels
            (motivated by hierarchical semantic segmentation, Li et al., CVPR 2022)
  VCH       both modules
Classes: 0 tree, 1 low vegetation, 2 sky, 3 ground, 4 built, 5 other; 255 ignored.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 6
COARSE_OF = [0, 0, 1, 2, 3, 4]   # class -> super-class (0 = vegetation)


class MLP(nn.Module):
    def __init__(self, cin, cout):
        super().__init__(); self.proj = nn.Linear(cin, cout)

    def forward(self, x):                     # B,C,H,W -> B,cout,H,W
        B, C, H, W = x.shape
        return self.proj(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, -1, H, W)


class VCG(nn.Module):
    """Vertical Context Gate: g(row, channel) from width-pooled features along the image height; output = x * 2 g (identity at init)"""
    def __init__(self, c, r=8, npos=32):
        super().__init__()
        self.reduce = nn.Conv1d(c, c // r, 3, padding=1)
        self.pos = nn.Parameter(torch.zeros(1, c // r, npos))
        self.mix = nn.Sequential(nn.GELU(), nn.Conv1d(c // r, c // r, 3, padding=2, dilation=2), nn.GELU(), nn.Conv1d(c // r, c, 1))
        nn.init.zeros_(self.mix[-1].weight); nn.init.zeros_(self.mix[-1].bias)

    def forward(self, x):
        B, C, H, W = x.shape
        z = self.reduce(x.mean(3)) + F.interpolate(self.pos, size=H, mode="linear", align_corners=False)
        g = torch.sigmoid(self.mix(z))                        # B,C,H in (0,1); 0.5 at init
        return x * (2 * g).unsqueeze(-1)


class VCHSegFormer(nn.Module):
    def __init__(self, encoder, dims=(64, 128, 320, 512), emb=768, vcg=False, hvh=False):
        super().__init__()
        self.encoder, self.hvh = encoder, hvh
        self.lin = nn.ModuleList([MLP(d, emb) for d in dims])
        self.fuse = nn.Sequential(nn.Conv2d(4 * emb, emb, 1, bias=False), nn.BatchNorm2d(emb), nn.ReLU(inplace=True))
        self.vcg = VCG(emb) if vcg else nn.Identity()
        self.drop = nn.Dropout2d(0.1)
        if hvh:
            self.coarse = nn.Conv2d(emb, 5, 1)
            self.fine = nn.Sequential(nn.Conv2d(emb, emb // 4, 3, padding=1, bias=False), nn.BatchNorm2d(emb // 4), nn.ReLU(inplace=True), nn.Conv2d(emb // 4, 2, 1))
        else:
            self.cls = nn.Conv2d(emb, N_CLASSES, 1)

    def features(self, x):
        hs = self.encoder(pixel_values=x, output_hidden_states=True).hidden_states       # 4 stages, 1/4 .. 1/32
        size = hs[0].shape[-2:]
        f = [F.interpolate(l(h), size=size, mode="bilinear", align_corners=False) for l, h in zip(self.lin, hs)]
        return self.drop(self.vcg(self.fuse(torch.cat(f[::-1], 1))))

    def forward(self, x):
        """returns a dict with 'logp' (B,6,H/4,W/4 log-probabilities) and, for HVH, the logits of both levels"""
        f = self.features(x)
        if not self.hvh:
            return {"logp": F.log_softmax(self.cls(f).float(), 1)}
        c, fi = self.coarse(f).float(), self.fine(f).float()
        lc, lf = F.log_softmax(c, 1), F.log_softmax(fi, 1)
        logp = torch.cat([lc[:, :1] + lf, lc[:, 1:]], 1)               # tree, low, sky, ground, built, other
        return {"logp": logp, "coarse": c, "fine": fi}


def loss_fn(out, y, lam_coarse=0.4, lam_fine=0.4):
    """y: B,H,W labels at the input resolution (255 ignored); predictions are upsampled to it"""
    size = y.shape[-2:]
    up = lambda t: F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    loss = F.nll_loss(up(out["logp"]), y, ignore_index=255)
    if "coarse" in out:
        lut = torch.full((256,), 255, dtype=torch.long, device=y.device); lut[:6] = torch.tensor(COARSE_OF, device=y.device)
        yc = lut[y]
        yf = torch.where(y <= 1, y, torch.full_like(y, 255))
        loss = loss + lam_coarse * F.cross_entropy(up(out["coarse"]), yc, ignore_index=255) + lam_fine * F.cross_entropy(up(out["fine"]), yf, ignore_index=255)
    return loss


def encoder_source(repo):
    """Hugging Face repo id, or the local folder <NDVI_GVI_MODELS>/<name> when that variable is set and the folder exists (for machines without access to the hub)"""
    root = os.environ.get("NDVI_GVI_MODELS"); local = os.path.join(root, repo.split("/")[-1]) if root else None
    return local if local and os.path.isdir(local) else repo


def build(variant, pretrained=None):
    """variant: base | vcg | hvh | vch, optionally followed by the size of the MiT encoder (base_b5, vch_b5; default b2).
    All MiT-B1..B5 encoders have the stage widths (64, 128, 320, 512) and the decoder width 768 (nvidia/mit-b<n>, ImageNet-1k)."""
    from transformers import SegformerModel
    v, _, size = variant.partition("_b")
    enc = SegformerModel.from_pretrained(encoder_source(pretrained or f"nvidia/mit-b{size or 2}"))
    return VCHSegFormer(enc, dims=tuple(enc.config.hidden_sizes), emb=enc.config.decoder_hidden_size, vcg=v in ("vcg", "vch"), hvh=v in ("hvh", "vch"))
