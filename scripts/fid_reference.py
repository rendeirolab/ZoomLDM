"""Fixed held-out reference for patch FID: per magnification m (0 = 20x ... 7 = 0.15625x), `n` windows of validation donors (the data.validation split of the config,
drawn by GTExDataset with seed 2000, runs of 8 windows per slide), each with its real image and its real conditioning:
  image: uint8 (224, 224, 3), the 2^m x 2^m tiles of the 226-px lattice read from the slide (<svs-dir>/<slide>.svs) at the pyramid level closest to 224 px, pixels outside
    the slide set to the slide's glass colour (prepB bg), resized to 224 px with antialiased bilinear interpolation, as the latents were rendered;
  cond: float16 (1024, h, h), the UNI grid of the same window (GTExDataset.grid, h = min(2^m, 8)).
One task builds one chunk: task t -> magnification t // chunks, chunk t % chunks; out <ref>/m<m>_<chunk>.npz with image, cond, idx (dataset indices only, no slide id):
  for t in $(seq 0 79); do PYTHONPATH=. python scripts/fid_reference.py --svs-dir <directory of the .svs files> --task $t; done   # or one SLURM array task each"""
import argparse
from pathlib import Path

import numpy as np
import openslide
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ldm.data.brca import MAG_DICT
from ldm.data.gtex import GTExDataset, latent_path

STEP = 226  # px of level 0 per lattice tile (the UNI and latent shards' lattice)


def resize224(u8):
    """uint8 (h, w, 3) -> uint8 (224, 224, 3): antialiased bilinear, rounded."""
    x = torch.as_tensor(np.ascontiguousarray(u8)).permute(2, 0, 1)[None].float()
    return F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


def window(slide, i0, j0, scale, bg):
    """Image of the 2^scale x 2^scale tiles from tile (i0, j0) (corner) of an openslide slide."""
    span = 2**scale * STEP
    lvl = slide.get_best_level_for_downsample(span / 224 + 1e-6)
    w = round(span / slide.level_downsamples[lvl])
    a = np.asarray(slide.read_region((int(i0) * STEP, int(j0) * STEP), lvl, (w, w)))
    return resize224(np.where(a[..., 3:] > 127, a[..., :3], np.asarray(bg, np.uint8)))


def dataset(config, m, n, seed=2000):
    c = OmegaConf.to_container(OmegaConf.load(config).data.validation.params.config)
    c.update(mag=list(MAG_DICT)[m], length=n, seed=seed, run=8)
    return GTExDataset(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/zoomldm_gtex_uni.yaml")
    ap.add_argument("--svs-dir", required=True, help="directory holding <slide>.svs for every slide")
    ap.add_argument("--out", default="runs/fid_ref")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--task", type=int, required=True)
    a = ap.parse_args()
    m, chunk = divmod(a.task, a.chunks)
    ds = dataset(a.config, m, a.n)
    size = a.n // a.chunks
    idx = np.arange(chunk * size, (chunk + 1) * size)
    images = np.zeros((len(idx), 224, 224, 3), np.uint8)
    h = min(2**m, 8)
    cond = np.zeros((len(idx), 1024, h, h), np.float16)
    open_slides = {}
    for k, i in enumerate(idx):
        s, _, r = ds.pick(int(i))
        uni, lat = ds.files(s)
        i0, j0 = (int(v) for v in lat[f"m{m}/xy"][r])
        cond[k] = ds.grid(uni, i0, j0, m)
        if s not in open_slides:
            if len(open_slides) > 16:
                open_slides.clear()
            name = ds.slides[s]
            bg = np.load(latent_path(ds.root / f"{name}.uni.h5").with_name(f"{name}.prepB.npz"))["bg"]
            open_slides[s] = (openslide.OpenSlide(str(Path(a.svs_dir) / f"{name}.svs")), bg)
        sl, bg = open_slides[s]
        images[k] = window(sl, i0, j0, m, bg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"m{m}_{chunk}.npz", image=images, cond=cond, idx=idx)
    print(f"magnification {m}, chunk {chunk}: {len(idx)} windows from {len({ds.pick(int(i))[0] for i in idx})} slides", flush=True)


if __name__ == "__main__":
    main()
