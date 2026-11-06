"""Gradient diagnostics of a checkpoint on one GPU: gradients of `k` independent training batches (train mode: conditioning dropped 10% of the time, bf16 autocast as in
training), per parameter group: mean gradient norm, mean pairwise cosine between batch gradients, and the simple noise scale B = batch * tr(cov) / |mean|^2 (the batch size
at which gradient noise equals signal; McCandlish et al. 2018). Also the EMA: its update count and the relative distance between EMA and online UNet weights. No slide id is printed.
  srun ... data/venvs/zoomldm-uni/bin/python scripts/grad_stats.py --ckpt runs/exp/start.ckpt --run runs/<name> [--root <a local copy of the slides>]"""
import argparse
import itertools
from pathlib import Path

import torch
from omegaconf import OmegaConf
from train_gtex_uni import GTExLDM, collate

from ldm.util import instantiate_from_config

GROUPS = ["cond_stage_model", "model.diffusion_model.time_embed", "model.diffusion_model.input_blocks", "model.diffusion_model.middle_block", "model.diffusion_model.output_blocks", "model.diffusion_model.out."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--batch", type=int, default=160)
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args()
    cfg = OmegaConf.load(Path(a.run) / "config.yaml")
    model = GTExLDM(**cfg.model.params)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["state_dict"], strict=True)
    print(f"checkpoint step {sd.get('global_step')}", flush=True)
    del sd
    model.cuda().train()
    ema = dict(model.model_ema.named_buffers())
    num = float(ema.pop("num_updates"))
    on = {n: p for n, p in model.model.named_parameters() if p.requires_grad}
    d = sum(float((ema[model.model_ema.m_name2s_name[n]] - p).pow(2).sum()) for n, p in on.items()) ** 0.5
    w = sum(float(p.pow(2).sum()) for p in on.values()) ** 0.5
    print(f"EMA: num_updates {num:.0f}, decay used {min(model.model_ema.decay, (1 + num) / (10 + num)):.6f}, |EMA - online| / |online| = {d / w:.2e}", flush=True)
    d_cfg = cfg.data.train
    if a.root:
        d_cfg = OmegaConf.merge(d_cfg, {"params": {"config": {"root": a.root, "seed": 12345}}})
    ds = instantiate_from_config(d_cfg)
    names = {g: [n for n, _ in model.named_parameters() if n.startswith(g)] for g in GROUPS}
    params = dict(model.named_parameters())
    grads = {g: [] for g in GROUPS}
    for b in range(a.k):
        batch = collate([ds[i] for i in range(b * 100000, b * 100000 + a.batch)])  # far-apart index ranges: different slides
        batch = {k: ([x.cuda() for x in v] if isinstance(v, list) else v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model.shared_step(batch)
        loss.backward()
        for g in GROUPS:
            grads[g].append(torch.cat([params[n].grad.flatten().float() for n in names[g] if params[n].grad is not None]))
        print(f"batch {b}: loss {float(loss):.4f}", flush=True)
    for g in GROUPS:
        G = torch.stack(grads[g])
        mean = G.mean(0)
        cos = [float(torch.dot(G[i], G[j]) / (G[i].norm() * G[j].norm())) for i, j in itertools.combinations(range(len(G)), 2)]
        noise = float((G - mean).pow(2).sum(1).mean()) * len(G) / (len(G) - 1)  # trace of the per-batch gradient covariance, unbiased
        b_simple = a.batch * noise / max(float(mean.pow(2).sum()), 1e-30)
        print(f"{g:36s} {G.shape[1] / 1e6:6.1f} M  |grad| {float(G.norm(dim=1).mean()):.3e}  cosine between batches {sum(cos) / len(cos):+.3f}  noise scale B {b_simple:9.0f}", flush=True)


if __name__ == "__main__":
    main()
