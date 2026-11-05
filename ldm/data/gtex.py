"""GTEx windows for ZoomLDM: the batch of ldm.data.brca.TCGADataset (image placeholder, vae_feat, ssl_feat, idx, mag) read from two h5 files per slide.
UNI: <root>/<slide>.uni.h5 (data/gtex-uni, symlinks; or a staged copy on the node's local disk, scripts/stage_slides.py): feat (n, 1024) float16 of the 224-px 20x tiles,
idx (rows, cols) -> row of feat or -1, bg (1024,) = UNI of a blank tile. Latents: next to the symlink target (or in the same directory), <slide>.latB.h5: groups m0..m7
(m = MAG_DICT value: 20x, 10x, ..., 0.15625x), lat (n, 3, 56, 56) float16 and xy (n, 2) = (col, row) of the window's top-left 20x tile; a scale-m window spans 2^m x 2^m tiles.
ssl_feat (1024, h, h), h = min(2^m, 8): the UNI rows inside the window (bg past the slide's edge or off tissue), mean-pooled to h x h, standardised over channels per cell
when normalize_ssl. vae_feat (3, 56, 56) float32. Sampling: samples come in runs of `run` consecutive indices from one slide (drawn with probability ~ its window count), so a
worker reads many windows per pair of open files; within a run each sample draws a scale uniformly among the slide's scales (or `mag`) and a window uniformly. Sample i of an
epoch is fixed by (seed, i). Window counts per slide and scale come from `counts` (scripts/count_windows.py). Open h5 handles are kept per worker (an LRU of `cache` slides).
Nothing here prints a slide id, a donor id or an age."""
import csv
import os
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ldm.data.brca import MAG_DICT


def latent_path(uni):
    p = Path(os.path.realpath(uni))
    return p.with_name(p.name[: -len(".uni.h5")] + ".latB.h5")


def read_counts(path, slides=None, split=None):
    """counts npz (scripts/count_windows.py): slide (S,) ids, n (S, 8) windows per scale -> (slides, n), restricted to one donor split of the manifest (columns slide_id, split) when given."""
    z = np.load(path, allow_pickle=True)
    s = list(z["slide"])
    n = z["n"].astype(np.int64)
    if split:
        with open(slides) as f:
            keep = {r["slide_id"] for r in csv.DictReader(f) if r["split"] == split}
        sel = [i for i, x in enumerate(s) if x in keep]
        s = [s[i] for i in sel]
        n = n[sel]
    return s, n


class GTExDataset(Dataset):
    def __init__(self, config=None):
        self.root = Path(config.get("root", "data/gtex-uni"))
        self.mag = config.get("mag", None)
        self.m = MAG_DICT[self.mag] if self.mag else None
        self.feat_target_size = config.get("feat_target_size", 8)
        self.normalize_ssl = config.get("normalize_ssl", True)
        self.seed = config.get("seed", 0)
        self.cache = config.get("cache", 64)
        self.run = config.get("run", 64)
        self.slides, self.n = read_counts(
            config.get("counts", "data/gtex_counts.npz"),
            config.get("slides"),
            config.get("split"),
        )
        if self.mag:
            w = self.n[:, self.m]
        else:
            w = self.n.sum(1)
        self.p = w / w.sum()
        self.length = config.get("length") or int(w.sum())
        self.h = OrderedDict()
        self.last = None
        print(
            f"GTExDataset: {len(self.slides)} slides, windows per scale {self.n.sum(0).tolist()}, length {self.length}, runs of {self.run}"
        )

    def __len__(self):
        return self.length

    def files(self, a):
        """(uni h5, latent h5) of slide a, kept open in a bounded LRU (per worker)."""
        if a in self.h:
            self.h.move_to_end(a)
            return self.h[a]
        if len(self.h) >= self.cache:
            _, (uni, lat) = self.h.popitem(last=False)
            uni.close()
            lat.close()
        s = self.slides[a]
        uni = h5py.File(self.root / f"{s}.uni.h5", "r")
        self.h[a] = (uni, h5py.File(latent_path(uni.filename), "r"))
        return self.h[a]

    def grid(self, u, i0, j0, m):
        """ssl_feat (1024, h, h) float32 of the window at (col i0, row j0), scale m."""
        w = 2 ** m
        h = min(2 ** m, 8)
        ix = np.full((w, w), -1, np.int64)
        sub = u["idx"][j0 : j0 + w, i0 : i0 + w]
        ix[: sub.shape[0], : sub.shape[1]] = sub
        v = ix[ix >= 0]
        x = np.broadcast_to(u["bg"][:], (w, w, 1024)).copy()
        if v.size:
            lo = int(v.min())
            x[ix >= 0] = u["feat"][lo : int(v.max()) + 1][v - lo]
        if h < w:
            x = x.reshape(h, w // h, h, w // h, -1).mean((1, 3), dtype=np.float32)
        else:
            x = x.astype(np.float32)
        if self.normalize_ssl:
            x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)
        return np.ascontiguousarray(x.transpose(2, 0, 1))

    def slide(self, idx):
        """Slide of the run that holds sample idx (one draw per run, remembered)."""
        r = idx // self.run
        if self.last is None or self.last[0] != r:
            choice = int(np.random.default_rng([self.seed, r]).choice(len(self.slides), p=self.p))
            self.last = (r, choice)
        return self.last[1]

    def __getitem__(self, idx):
        a = self.slide(idx)
        rng = np.random.default_rng([self.seed, int(idx)])
        if self.mag:
            m = self.m
        else:
            m = int(rng.choice(np.flatnonzero(self.n[a])))
        r = int(rng.integers(self.n[a, m]))
        uni, lat = self.files(a)
        vae = torch.from_numpy(lat[f"m{m}/lat"][r].astype(np.float32))
        i0, j0 = (int(v) for v in lat[f"m{m}/xy"][r])
        return {
            "image": np.ones((1, 1, 1, 3), dtype=np.float16),
            "vae_feat": vae,
            "ssl_feat": torch.from_numpy(self.grid(uni, i0, j0, m)),
            "idx": idx,
            "mag": m,
        }
