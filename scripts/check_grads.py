"""Gradients of one training batch (192 windows, fixed timesteps and noise, conditioning kept) from the released checkpoint in three precisions:
fp32 (reference, in chunks of 48, no TF32), bf16 autocast with upstream's float16 autocast around the UNet (what the first two runs trained with),
and bf16 autocast with GTExLDM.apply_model (the UNet in bf16). Prints the loss and, for the UNet and the conditioner, the share of exactly-zero
gradient entries, the gradient norm and the cosine to the fp32 reference. Runs on one GPU; no slide id is printed."""
import argparse

import torch
from omegaconf import OmegaConf
from train_gtex_uni import ROOT, GTExLDM, collate, load_matching

from ldm.util import instantiate_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/zoomldm_gtex_uni.yaml")
    ap.add_argument("--root", default=None)
    ap.add_argument("--batch", type=int, default=192)
    a = ap.parse_args()
    cfg = OmegaConf.load(ROOT / a.config)
    model = GTExLDM(**cfg.model.params)
    load_matching(model, cfg.init_ckpt)
    model.cuda()
    model.cond_stage_model.p_uncond = 0.0
    d = cfg.data.train
    if a.root:
        d = OmegaConf.merge(d, {"params": {"config": {"root": a.root}}})
    batch = collate([instantiate_from_config(d)[i] for i in range(a.batch)])
    x = batch["vae_feat"].cuda()
    gens = [torch.Generator().manual_seed(int(i)) for i in batch["idx"]]
    t = torch.stack([torch.randint(0, model.num_timesteps, (), generator=g) for g in gens]).cuda()
    noise = torch.stack([torch.randn(x.shape[1:], generator=g) for g in gens]).cuda()

    def grads(mode, chunk):
        model.zero_grad(set_to_none=True)
        model.upstream_autocast = mode == "bf16 + upstream fp16 UNet"
        torch.backends.cuda.matmul.allow_tf32 = mode != "fp32"
        torch.backends.cudnn.allow_tf32 = mode != "fp32"
        total = 0.0
        for k in range(0, len(x), chunk):
            sl = slice(k, k + chunk)
            c = {"ssl_feat": [f.cuda() for f in batch["ssl_feat"][sl]], "mag": batch["mag"][sl].cuda()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mode != "fp32"):
                out = model.apply_model(model.q_sample(x[sl], t[sl], noise[sl]), t[sl], model.get_learned_conditioning(c))
                loss = (out.float() - noise[sl]).pow(2).mean() * (len(x[sl]) / len(x))
            loss.backward()
            total += float(loss)
        g = {}
        for name, mod in (("unet", model.model), ("conditioner", model.cond_stage_model)):
            g[name] = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().float() for p in mod.parameters()])
        return total, g

    ref_loss, ref = grads("fp32", 48)
    print(f"batch {len(x)}, fp32 reference loss {ref_loss:.5f}", flush=True)
    for mode in ("fp32", "bf16 + upstream fp16 UNet", "bf16 UNet (GTExLDM)"):
        loss, g = (ref_loss, ref) if mode == "fp32" else grads(mode, len(x))
        for name, v in g.items():
            r = ref[name]
            cos = float(torch.dot(v, r) / (v.norm() * r.norm() + 1e-30))
            zero = float((v == 0).float().mean())
            print(f"{mode:28s} loss {loss:.5f} {name:11s}: zero grads {zero:6.1%}, norm {float(v.norm()):.3e} (fp32 {float(r.norm()):.3e}), cosine to fp32 {cos:.4f}", flush=True)


if __name__ == "__main__":
    main()
