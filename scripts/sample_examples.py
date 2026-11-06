"""Example generations beside their real held-out windows: EMA weights, PLMS 50 steps, classifier-free guidance 2 (the authors' sampling, as scripts/score_fid.py).
For each chosen magnification, `windows` windows of the fixed validation reference (runs/fid_ref, scripts/fid_reference.py; validation donors), spread across the reference,
each generated `seeds` times from freshly drawn random noise seeds, conditioned on the window's real UNI grid. Out, in --out:
  examples_<mag>.png: one row per window, columns real | generation per seed;
  examples.json: checkpoint step, and per image the reference index, magnification and noise seed (no slide or donor id).
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=.:scripts python scripts/sample_examples.py --ckpt runs/<name>/checkpoints/best.ckpt --run runs/<name> --out runs/<name>/examples"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from score_fid import LABEL, MAGS, reference
from train_gtex_uni import GTExLDM

from ldm.models.diffusion.plms import PLMSSampler


@torch.no_grad()
def generate(model, sampler, cond, m, seeds, steps, cfg):
    c = [torch.from_numpy(x).cuda().float() for x in cond]
    x_t = torch.stack([torch.randn((3, 56, 56), generator=torch.Generator().manual_seed(int(s))) for s in seeds]).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cc = model.get_learned_conditioning({"ssl_feat": c, "mag": torch.full((len(c),), m, dtype=torch.long, device="cuda")})
        z, _ = sampler.sample(S=steps, conditioning=cc, batch_size=len(c), shape=[3, 56, 56], verbose=False, unconditional_guidance_scale=cfg, unconditional_conditioning=torch.zeros_like(cc), x_T=x_t)
    img = model.decode_first_stage(z.float())
    return (255 * torch.clamp((img + 1) / 2, 0, 1)).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def grid(rows, header, gap=6, top=22):
    """rows: list of lists of uint8 (224, 224, 3) images; header: column titles."""
    h, w = rows[0][0].shape[:2]
    out = Image.new("RGB", (len(rows[0]) * (w + gap) - gap, top + len(rows) * (h + gap) - gap), "white")
    draw = ImageDraw.Draw(out)
    for j, t in enumerate(header):
        draw.text((j * (w + gap) + 4, 5), t, fill="black")
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            out.paste(Image.fromarray(im), (j * (w + gap), top + i * (h + gap)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", required=True, help="run directory with config.yaml")
    ap.add_argument("--ref", default="runs/fid_ref")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mags", type=int, nargs="+", default=[0, 2, 4, 7], help="0 = 20x ... 7 = 0.15625x")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=2.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(Path(a.run) / "config.yaml")
    model = GTExLDM(**cfg.model.params)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["state_dict"], strict=True)
    step = int(sd["global_step"])
    del sd
    model.cuda().eval()
    model.cond_stage_model.p_uncond = 0.0
    sampler = PLMSSampler(model)
    rng = np.random.default_rng()  # fresh seeds every run; recorded in examples.json
    record = {"checkpoint": str(a.ckpt), "step": step, "sampling": f"EMA weights, PLMS {a.steps} steps, guidance {a.cfg}", "images": []}
    with model.ema_scope():
        for m in a.mags:
            images, cond, idx = (reference(Path(a.ref), m, k, 10000) for k in ("image", "cond", "idx"))
            pos = np.linspace(0, len(idx) - 1, a.windows).round().astype(int)  # spread over the reference: different slides
            seeds = rng.integers(0, 2**31 - 1, size=(a.windows, a.seeds))
            gen = generate(model, sampler, np.repeat(cond[pos], a.seeds, axis=0), m, seeds.ravel(), a.steps, a.cfg).reshape(a.windows, a.seeds, 224, 224, 3)
            rows = [[images[p], *gen[i]] for i, p in enumerate(pos)]
            grid(rows, ["real", *[f"generated, seed {k + 1}" for k in range(a.seeds)]]).save(out / f"examples_{MAGS[m]}.png")
            for i, p in enumerate(pos):
                record["images"].append({"magnification": LABEL[MAGS[m]], "row": i, "reference_index": int(idx[p]), "seeds": [int(s) for s in seeds[i]]})
            print(f"{LABEL[MAGS[m]]}: {a.windows} windows x {a.seeds} seeds -> {out / f'examples_{MAGS[m]}.png'}", flush=True)
    with open(out / "examples.json", "w") as f:
        json.dump(record, f, indent=1)


if __name__ == "__main__":
    main()
