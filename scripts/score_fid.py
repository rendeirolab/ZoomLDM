"""Patch FID of a training run's checkpoints, scored in a loop on one GPU beside the training job (never on its GPUs).
Protocol of ZoomLDM's sample_patches notebooks: EMA weights, no null-token dropout, PLMS 50 steps, classifier-free guidance 2 against the all-zero context, images decoded by
the VQ-f4 first stage (224 px here; the authors' TCGA-BRCA images are 256 px). Per magnification, `n` images are generated from the real conditioning of the fixed held-out
windows in runs/fid_ref (scripts/fid_reference.py, validation donors), each from a fixed noise seed, and compared with those windows' real images: Frechet distance of
pytorch-fid's Inception pool features (2048-d); the real statistics are computed once and cached beside the reference.
Loop: take the newest full checkpoint of the run (checkpoints/last.ckpt or best.ckpt, whichever is newer than the last one scored; last.ckpt only once unchanged for a minute),
copy it to fid/candidate.ckpt (training keeps writing its own files), score the copy, and for every magnification whose FID improved keep the copy as
checkpoints/best_fid_<mag>.ckpt (hard links: one copy can be the best for several magnifications). A magnification has stopped improving after `patience` scored checkpoints
in a row without a new best; its best_fid checkpoint is then the one to keep. Out, in the run directory: fid_table.md (ours beside the authors'), fid/scores.jsonl, fid/best.json.
Exits once no SLURM job named --train-job is queued and the newest checkpoint has been scored (--once: score the newest checkpoint and exit). Example:
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=.:scripts python scripts/score_fid.py --run runs/<name> --train-job <training job name> --batch 500"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from train_gtex_uni import GTExLDM

from ldm.data.brca import MAG_DICT
from ldm.models.diffusion.plms import PLMSSampler

MAGS = list(MAG_DICT)
LABEL = {"20x": "20x", "10x": "10x", "5x": "5x", "2_5x": "2.5x", "1_25x": "1.25x", "0_625x": "0.625x", "0_3125x": "0.3125x", "0_15625x": "0.15625x"}
AUTHORS = {"20x": 6.77, "10x": 7.60, "5x": 7.98, "2_5x": 10.73, "1_25x": 8.74, "0_625x": 7.99, "0_3125x": 8.34, "0_15625x": 13.42}  # ZoomLDM paper, TCGA-BRCA, 256 px


def reference(ref, m, key, n):
    files = sorted(ref.glob(f"m{m}_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    out = np.concatenate([np.load(f)[key] for f in files])
    assert len(out) >= n, f"magnification {m}: {len(out)} reference windows, {n} needed"
    return out[:n]


@torch.no_grad()
def features(net, u8, bs=200):
    out = []
    for a in range(0, len(u8), bs):
        x = torch.from_numpy(u8[a : a + bs]).cuda().permute(0, 3, 1, 2).float() / 255
        out.append(net(x)[0].flatten(1).double().cpu())
    return torch.cat(out).numpy()


def real_stats(net, ref, m, n):
    p = ref / f"stats_m{m}_n{n}.npz"
    if not p.exists():
        f = features(net, reference(ref, m, "image", n))
        np.savez(p, mu=f.mean(0), sigma=np.cov(f, rowvar=False))
    z = np.load(p)
    return z["mu"], z["sigma"]


@torch.no_grad()
def generate(model, sampler, cond, idx, m, steps, cfg, bs):
    """uint8 (n, 224, 224, 3): one image per conditioning grid, from the noise seeded by (window index, magnification)."""
    out = np.zeros((len(cond), 224, 224, 3), np.uint8)
    for a in range(0, len(cond), bs):
        c = [torch.from_numpy(x).cuda().float() for x in cond[a : a + bs]]
        b = len(c)
        x_t = torch.stack([torch.randn((3, 56, 56), generator=torch.Generator().manual_seed(int(i) * 8 + m)) for i in idx[a : a + bs]]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cc = model.get_learned_conditioning({"ssl_feat": c, "mag": torch.full((b,), m, dtype=torch.long, device="cuda")})
            z, _ = sampler.sample(S=steps, conditioning=cc, batch_size=b, shape=[3, 56, 56], verbose=False, unconditional_guidance_scale=cfg, unconditional_conditioning=torch.zeros_like(cc), x_T=x_t)
        for k in range(0, b, 50):  # the VQ decoder in fp32 and in chunks, as the notebooks decode
            img = model.decode_first_stage(z[k : k + 50].float())
            out[a + k : a + k + len(img)] = (255 * torch.clamp((img + 1) / 2, 0, 1)).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return out


def newest(run, done):
    """(checkpoint to score or None, newest mtime among the run's full checkpoints)."""
    cands = [p for p in (run / "checkpoints/last.ckpt", run / "checkpoints/best.ckpt") if p.exists()]
    if not cands:
        return None, 0.0
    p = max(cands, key=lambda q: q.stat().st_mtime)
    t = p.stat().st_mtime
    if t <= done or (p.name == "last.ckpt" and time.time() - t < 60):
        return None, t
    return p, t


def training_queued(name):
    try:
        r = subprocess.run(["squeue", "-h", "-u", os.environ["USER"], "-n", name, "-o", "%i"], capture_output=True, text=True, timeout=60)
        return r.returncode != 0 or bool(r.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return True  # unknown: keep scoring


def table(out, best, last, scored, patience, n):
    rows = ["| magnification | authors: TCGA-BRCA, 256 px | ours, best: GTEx, 224 px | best at step | ours, latest | status |", "|---|---|---|---|---|---|"]
    for mag in MAGS:
        b = best.get(mag)
        if b is None:
            rows.append(f"| {LABEL[mag]} | {AUTHORS[mag]:.2f} | | | | not scored |")
            continue
        status = f"stopped: no new best in {b['since']} scored checkpoints; keep checkpoints/best_fid_{mag}.ckpt" if b["since"] >= patience else ("improving" if b["since"] == 0 else f"no new best in {b['since']}")
        rows.append(f"| {LABEL[mag]} | {AUTHORS[mag]:.2f} | {b['fid']:.2f} | {b['step']} | {last['fid'][mag]:.2f} | {status} |")
    text = "\n".join([
        f"# Patch FID, {n} images per magnification, real conditioning",
        "",
        "Authors: ZoomLDM on TCGA-BRCA at 256 px (paper). Ours: this run on GTEx at 224 px, windows of held-out validation donors, generated from each window's real UNI conditioning and compared with",
        "that window's real image; EMA weights, PLMS 50 steps, guidance 2, pytorch-fid Inception features. The data sets differ, so the authors' numbers are a comparison, not a target.",
        f"A magnification's best checkpoint is final once {patience} scored checkpoints in a row bring no new best.",
        "",
        *rows,
        "",
        f"Scored checkpoints (step): {', '.join(str(s) for s in scored)}. Latest scored: step {last['step']}.",
        "",
    ])
    tmp = out / "fid_table.md.tmp"
    tmp.write_text(text)
    tmp.replace(out / "fid_table.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<name>: config.yaml and checkpoints/ of the training run")
    ap.add_argument("--out", default=None, help="where the table, fid/ and best_fid checkpoints go (default: the run directory)")
    ap.add_argument("--ref", default="runs/fid_ref")
    ap.add_argument("--train-job", default="gtex-uni-hgx")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--once", action="store_true", help="score the newest checkpoint once and exit")
    a = ap.parse_args()
    run, ref = Path(a.run), Path(a.ref)
    out = Path(a.out) if a.out else run
    (out / "fid").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda().eval()
    real = {m: real_stats(net, ref, m, a.n) for m in range(len(MAGS))}
    print(f"real statistics ready for {len(real)} magnifications ({a.n} windows each)", flush=True)
    state_file = out / "fid/state.json"
    state = json.load(open(state_file)) if state_file.exists() else {"mtime": 0.0, "step": None}
    scores = [json.loads(line) for line in open(out / "fid/scores.jsonl")] if (out / "fid/scores.jsonl").exists() else []
    run_cfg = OmegaConf.load(run / "config.yaml")
    while True:
        p, t = newest(run, state["mtime"])
        if p is None:
            if t <= state["mtime"] and not training_queued(a.train_job):
                print("training is no longer queued and its newest checkpoint is scored: done", flush=True)
                return
            time.sleep(a.poll)
            continue
        copy = out / "fid/candidate.ckpt"
        shutil.copyfile(p, copy)
        if p.stat().st_mtime != t:
            print(f"{p.name} changed while copying; retrying", flush=True)
            continue
        sd = torch.load(copy, map_location="cpu", weights_only=False)
        step = int(sd["global_step"])
        if step == state["step"]:
            state["mtime"] = t
            json.dump(state, open(state_file, "w"))
            copy.unlink()
            continue
        t0 = time.time()
        model = GTExLDM(**run_cfg.model.params)
        model.load_state_dict(sd["state_dict"], strict=True)
        del sd
        model.cuda().eval()
        model.cond_stage_model.p_uncond = 0.0
        sampler = PLMSSampler(model)
        fid = {}
        with model.ema_scope():
            for m, mag in enumerate(MAGS):
                t1 = time.time()
                imgs = generate(model, sampler, reference(ref, m, "cond", a.n), reference(ref, m, "idx", a.n), m, a.steps, a.cfg, a.batch)
                f = features(net, imgs)
                fid[mag] = float(calculate_frechet_distance(f.mean(0), np.cov(f, rowvar=False), *real[m]))
                print(f"step {step}: {LABEL[mag]} FID {fid[mag]:.2f} (authors {AUTHORS[mag]:.2f}), {(time.time() - t1) / 60:.1f} min", flush=True)
        del model, sampler
        torch.cuda.empty_cache()
        last = {"step": step, "fid": fid, "minutes": (time.time() - t0) / 60, "source": p.name}
        scores.append(last)
        with open(out / "fid/scores.jsonl", "a") as fh:
            fh.write(json.dumps(last) + "\n")
        bj = out / "fid/best.json"
        best = json.load(open(bj)) if bj.exists() else {}
        for mag, v in fid.items():
            if mag not in best or v < best[mag]["fid"]:
                tmp = out / f"checkpoints/.best_fid_{mag}.tmp"
                tmp.unlink(missing_ok=True)
                os.link(copy, tmp)
                tmp.replace(out / f"checkpoints/best_fid_{mag}.ckpt")
                best[mag] = {"fid": v, "step": step, "since": 0}
            else:
                best[mag]["since"] += 1
        json.dump(best, open(bj, "w"), indent=1)
        copy.unlink()
        table(out, best, last, [s["step"] for s in scores], a.patience, a.n)
        state = {"mtime": t, "step": step}
        json.dump(state, open(state_file, "w"))
        improved = [LABEL[k] for k, v in best.items() if v["step"] == step]
        print(f"step {step} scored in {last['minutes']:.0f} min; new best for: {', '.join(improved) or 'none'}", flush=True)
        if a.once:
            return


if __name__ == "__main__":
    main()
