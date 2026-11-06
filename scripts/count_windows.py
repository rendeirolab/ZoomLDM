"""Window counts per slide and scale of the GTEx latents (ldm.data.gtex samples with them): a SLURM array counts files[task::ntasks] of data/gtex-uni into
runs/gtex_counts/part_<task>.npz (slide (S,), n (S, 8)); --merge joins the parts into data/gtex_counts.npz and checks that every symlink has exactly one row. Prints counts only.
  for t in $(seq 0 99); do PYTHONPATH=. python scripts/count_windows.py --task $t --ntasks 100; done   # or one SLURM array task each
  PYTHONPATH=. python scripts/count_windows.py --merge --ntasks 100"""
import argparse
from pathlib import Path

import h5py
import numpy as np

from ldm.data.gtex import latent_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/gtex-uni")
    ap.add_argument("--parts", default="runs/gtex_counts")
    ap.add_argument("--out", default="data/gtex_counts.npz")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    files = sorted(Path(a.root).glob("*.uni.h5"))
    parts = Path(a.parts)
    parts.mkdir(parents=True, exist_ok=True)
    if a.merge:
        z = [np.load(parts / f"part_{t}.npz") for t in range(a.ntasks)]
        slide = np.concatenate([p["slide"] for p in z])
        n = np.concatenate([p["n"] for p in z])
        names = [f.name[:-len(".uni.h5")] for f in files]
        assert sorted(slide) == sorted(names) and len(set(slide)) == len(names), "the parts do not cover every symlink exactly once"
        o = np.argsort(slide)
        np.savez(a.out, slide=slide[o], n=n[o])
        print(f"{len(slide)} slides, {(n.sum(1) > 0).sum()} with latents, windows per scale {n.sum(0).tolist()}")
        return
    mine = files[a.task::a.ntasks]
    n = np.zeros((len(mine), 8), np.int64)
    for i, f in enumerate(mine):
        p = latent_path(f)
        if p.exists():
            with h5py.File(p, "r") as h:
                n[i] = [h[f"m{m}/xy"].shape[0] for m in range(8)]
    slide = np.array([f.name[:-len(".uni.h5")] for f in mine])
    np.savez(parts / f"part_{a.task}.npz", slide=slide, n=n)
    print(f"task {a.task}: {len(mine)} slides counted")


if __name__ == "__main__":
    main()
