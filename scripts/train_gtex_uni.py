"""ZoomLDM with UNI conditioning on GTEx windows (configs/zoomldm_gtex_uni.yaml), initialised from the released ZoomLDM-BRCA checkpoint, on every GPU of the job's nodes (DDP, one task per GPU; NCCL across nodes).
Out: runs/<name>/ (config.yaml, batch.json, tb/ event files, checkpoints/last.ckpt every ckpt_minutes and weights-only {step}.ckpt every keep_minutes); a resubmitted job resumes from last.ckpt.
TensorBoard (tensorboard --logdir runs): train/loss_simple_step, lr, time/step_s, time/steps_per_h, val/loss, val/loss_ema, time/val_s; the SLURM log gets the loss, the learning
rate and steps/h every log_every_n_steps, and the held-out loss after each validation.
Held-out loss: data.validation, a fixed set of windows of validation donors only, every trainer.val_check_interval steps; each window has a fixed timestep and noise (seeded by its
index) and the conditioning is never dropped, so the loss of two checkpoints is compared on identical inputs. checkpoints/best.ckpt (full Lightning checkpoint: weights, EMA
weights, optimizer) is written only when val/loss_ema improves, with best.json (step, loss) and the resolved config.yaml beside it; the best loss survives resubmission (best.json).
(lr is read from the optimizer; ddpm's own lr_abs is stale after a resume: Lightning's optimizer wrapper keeps the param groups from before the optimizer state was restored.)
Precision: forward pass under bf16 autocast (Trainer precision "bf16"; GTExLDM.apply_model keeps the UNet in bf16, upstream forces float16 there); weights, gradients and optimizer
state in fp32. Learning-rate search: lr_range runs an exponential range test; judge.enabled stops a constant-rate run whose loss climbs (Judge). Batch size: data.batch_size, or the largest that fits (probed on
rank 0 at the first start, dummy batches of 8 x 8 grids: the token count does not depend on the scale). Launch one task per GPU on each node (SLURM srun, or torchrun-style
environment variables), with TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (resuming loads full Lightning checkpoints), e.g.
  srun --ntasks-per-node=<GPUs> python scripts/train_gtex_uni.py --name <run> [--root <a local copy of the slides>] [--set key=value ...]
GpuSample samples nvidia-smi on every node after the second logged step; StopFile stops cleanly when runs/<name>/STOP appears."""
import argparse
import datetime
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, default_collate

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import get_obj_from_str, instantiate_from_config

ROOT = Path(__file__).resolve().parents[1]


def collate(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if k == "ssl_feat":
            out[k] = vals
        else:
            out[k] = default_collate(vals)
    return out


def resolve(path):
    """hf://<owner>/<repo>/<file> -> that file from the Hugging Face hub (downloaded once, then cached); any other path as it is."""
    if not str(path).startswith("hf://"):
        return path
    parts = str(path)[len("hf://") :].split("/")
    return hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]))


def load_matching(model, path):
    """Copy every tensor of the checkpoint whose name and shape match the model; return (copied, mismatched, missing, unexpected) name lists (mismatched/missing tensors keep their init)."""
    sd = torch.load(resolve(path), map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    own = model.state_dict()
    copied, mismatched = [], []
    for k, v in own.items():
        if k not in sd:
            continue
        if tuple(sd[k].shape) == tuple(v.shape):
            v.copy_(sd[k])
            copied.append(k)
        else:
            mismatched.append(f"{k}: checkpoint {tuple(sd[k].shape)}, model {tuple(v.shape)}")
    missing = [k for k in own if k not in sd]
    unexpected = [k for k in sd if k not in own]
    print(f"{path}: copied {len(copied)} tensors; left out {len(mismatched)} mismatched, {len(missing)} missing; {len(unexpected)} unexpected", flush=True)
    for label, names in (("mismatched", mismatched), ("missing", missing), ("unexpected", unexpected)):
        if names:
            print(f"  {label}: " + ", ".join(names), flush=True)
    return copied, mismatched, missing, unexpected


def probe(model, sizes=(768, 640, 512, 448, 384, 320, 256, 192, 160, 128, 96, 64, 48, 32), margin=0.9):
    """Largest batch whose two training steps (bf16 autocast, fp32 AdamW) peak under margin x GPU memory; weights unchanged (lr 0)."""
    dev = torch.device("cuda", 0)
    total = torch.cuda.get_device_properties(dev).total_memory
    model.to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0)
    chosen = sizes[-1]
    for bs in sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        try:
            for _ in range(2):
                batch = {
                    "vae_feat": torch.randn(bs, 3, 56, 56, device=dev),
                    "ssl_feat": [torch.randn(1024, 8, 8, device=dev) for _ in range(bs)],
                    "mag": torch.randint(0, 8, (bs,), device=dev),
                }
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = model.shared_step(batch)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(dev)
            print(f"probe: batch {bs}: peak {peak / 2**30:.1f} of {total / 2**30:.0f} GiB", flush=True)
            if peak < margin * total:
                chosen = bs
                break
        except torch.cuda.OutOfMemoryError:
            print(f"probe: batch {bs}: out of memory", flush=True)
            opt.zero_grad(set_to_none=True)
    del opt
    model.zero_grad(set_to_none=True)
    model.cpu()
    torch.cuda.empty_cache()
    return chosen


class GTExLDM(LatentDiffusion):
    """LatentDiffusion with a deterministic held-out loss: eps MSE per window at a fixed timestep and noise (seeded by the window index), conditioning never dropped,
    for the online and the EMA weights."""

    upstream_autocast = False  # True: the UNet under upstream's torch.cuda.amp.autocast() (float16), only for scripts/check_grads.py
    lr_range = None  # (start, end, steps): exponential learning-rate range test instead of the configured schedule

    def apply_model(self, x_noisy, t, cond, return_ids=False, **kwargs):
        """The UNet under the caller's autocast (bf16 in training). Upstream wraps it in torch.cuda.amp.autocast(), whose default dtype is float16; without loss scaling
        the UNet's gradients underflow (the gradient of the mean loss is ~1e-7 per output element at batch 192)."""
        if self.upstream_autocast or hasattr(self, "split_input_params"):
            return super().apply_model(x_noisy, t, cond, return_ids=return_ids, **kwargs)
        if not isinstance(cond, dict):
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond if isinstance(cond, list) else [cond]}
        out = self.model(x_noisy, t, **cond, **kwargs)
        if isinstance(out, tuple) and not return_ids:
            return out[0]
        return out

    def configure_optimizers(self):
        if not self.lr_range:
            return super().configure_optimizers()
        start, end, steps = self.lr_range
        opt = torch.optim.AdamW(list(self.model.parameters()) + list(self.cond_stage_model.parameters()), lr=end)
        sched = LambdaLR(opt, lambda s: (start / end) * (end / start) ** min(s / steps, 1.0))
        return [opt], [{"scheduler": sched, "interval": "step", "frequency": 1}]

    def validation_step(self, batch, batch_idx):
        x, c = self.get_input(batch, self.first_stage_key)
        gens = [torch.Generator().manual_seed(int(i)) for i in batch["idx"]]
        t = torch.stack([torch.randint(0, self.num_timesteps, (), generator=g) for g in gens]).to(self.device)
        noise = torch.stack([torch.randn(x.shape[1:], generator=g) for g in gens]).to(self.device)
        p_uncond = self.cond_stage_model.p_uncond
        self.cond_stage_model.p_uncond = 0.0

        def mse():
            out = self.apply_model(self.q_sample(x, t, noise), t, self.get_learned_conditioning(c))
            return (out.float() - noise).pow(2).mean()

        try:
            loss = mse()
            with self.ema_scope():
                torch.clear_autocast_cache()  # autocast caches bf16 copies of the weights; without this the EMA pass reuses the online weights' copies
                loss_ema = mse()
            torch.clear_autocast_cache()
        finally:
            self.cond_stage_model.p_uncond = p_uncond
        self.log_dict({"val/loss": loss, "val/loss_ema": loss_ema}, on_step=False, on_epoch=True, sync_dist=True, batch_size=len(x))


class Best(Callback):
    """checkpoints/best.ckpt (full checkpoint) whenever val/loss_ema improves, with best.json (step, loss) and the resolved config.yaml beside it.
    Every rank calls save_checkpoint (rank 0 writes a temporary file, then renames it); the best loss is restored from best.json and from last.ckpt."""

    def __init__(self, out, cfg):
        self.dir = out / "checkpoints"
        self.cfg = cfg
        self.best = float("inf")
        self.t0 = None
        j = self.dir / "best.json"
        if j.exists():
            with open(j) as f:
                self.best = json.load(f)["loss"]

    def state_dict(self):
        return {"best": self.best}

    def load_state_dict(self, state):
        self.best = min(self.best, state["best"])

    def on_validation_start(self, trainer, module):
        self.t0 = time.time()

    def on_validation_end(self, trainer, module):
        m = trainer.callback_metrics
        if trainer.sanity_checking or "val/loss_ema" not in m:
            return
        v = float(m["val/loss_ema"])
        s = trainer.global_step
        dt = time.time() - self.t0
        better = v < self.best
        if better:
            self.best = v
            tmp = self.dir / "best.ckpt.tmp"
            trainer.save_checkpoint(tmp)
            if trainer.is_global_zero:
                tmp.rename(self.dir / "best.ckpt")
                OmegaConf.save(self.cfg, self.dir / "config.yaml", resolve=True)
                with open(self.dir / "best.json", "w") as f:
                    json.dump({"step": s, "loss": v}, f)
        if trainer.is_global_zero:
            trainer.logger.log_metrics({"time/val_s": dt}, step=s)
            note = ", best: saved best.ckpt" if better else ""
            print(f"step {s}: val loss {float(m['val/loss']):.4f} ema {v:.4f}{note} ({dt:.0f} s)", flush=True)


def theil_sen(x, y):
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = dx != 0
    return float(np.median((y[j] - y[i])[ok] / dx[ok]))


class Judge(Callback):
    """On-curve verdict at constant learning rate. Every `every` steps once `min_steps` constant-rate steps have passed, the per-step training loss (mean over ranks) of the
    last `window` constant-rate steps is cut into `block`-step medians (single-step spikes drop out); the Theil-Sen slope of the medians and its 95% block-bootstrap interval
    decide: easing (interval below 0), climbing (interval above 0; training stops), flat (neither after `flat_after` constant steps; training stops if it never eased,
    or always with stop_on_flat). Verdicts: runs/<name>/verdict.json, the SLURM log, TensorBoard judge/slope, judge/lo, judge/hi (per 1000 steps)."""

    def __init__(self, out, warm_up, block=100, every=250, min_steps=600, window=1500, flat_after=3000, stop_on_flat=False):
        self.out = out
        self.warm_up = warm_up
        self.block = block
        self.every = every
        self.min_steps = min_steps
        self.window = window
        self.flat_after = flat_after
        self.stop_on_flat = stop_on_flat
        self.loss = {}
        self.buf = []  # (step, loss on the GPU) since the last verdict: read back once per verdict, so training never waits on a per-step sync
        self.eased = False
        self.rng = np.random.default_rng(0)

    def judge(self, s, n):
        steps = np.array(sorted(k for k in self.loss if k > s - self.window))
        nb = len(steps) // self.block
        x = steps[-nb * self.block :].reshape(nb, self.block).mean(1)
        m = np.median(np.array([self.loss[k] for k in steps[-nb * self.block :]]).reshape(nb, self.block), 1)
        slope = theil_sen(x, m) * 1000
        boots = [theil_sen(x[i], m[i]) * 1000 for i in (self.rng.integers(nb, size=nb) for _ in range(1000))]
        lo, hi = (float(v) for v in np.nanpercentile(boots, [2.5, 97.5]))
        if hi < 0:
            verdict = "easing"
        elif lo > 0:
            verdict = "climbing"
        elif n >= self.flat_after:
            verdict = "flat"
        else:
            verdict = "undecided"
        return {"step": s, "verdict": verdict, "slope": slope, "lo": lo, "hi": hi, "level": float(m[-1])}

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        s = trainer.global_step
        n = s - self.warm_up
        if n > 0:
            self.buf.append((s, outputs["loss"].detach().float()))
        if n < self.min_steps or n % self.every:
            return
        steps = [k for k, _ in self.buf]
        vals = trainer.strategy.reduce(torch.stack([v for _, v in self.buf]), reduce_op="mean").tolist()  # mean over ranks, per step
        self.loss.update(zip(steps, vals))
        self.loss = {k: v for k, v in self.loss.items() if k > s - self.window}
        self.buf = []
        r = trainer.strategy.broadcast(self.judge(s, n) if trainer.is_global_zero else None)
        self.eased = self.eased or r["verdict"] == "easing"
        r["eased"] = self.eased
        r["lr"] = trainer.optimizers[0].param_groups[0]["lr"]
        stop = r["verdict"] == "climbing" or (r["verdict"] == "flat" and (self.stop_on_flat or not self.eased))
        if trainer.is_global_zero:
            trainer.logger.log_metrics({"judge/slope": r["slope"], "judge/lo": r["lo"], "judge/hi": r["hi"]}, step=s)
            with open(self.out / "verdict.json", "w") as f:
                json.dump(r, f)
            print(f"judge step {s}: {r['verdict']}, block-median loss {r['level']:.4f}, slope {r['slope']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] per 1000 steps over the last {min(n, self.window)} constant-rate steps{', stopping' if stop else ''}", flush=True)
        if stop:
            trainer.should_stop = True


class RangeLog(Callback):
    """Range test: the per-step training loss (mean over ranks) and learning rate to TensorBoard (range/loss, range/lr); stops once the smoothed loss (EMA 0.98) exceeds
    `stop` x its minimum after 300 steps (the loss has clearly diverged)."""

    def __init__(self, stop=1.5):
        self.stop = stop
        self.ema = None
        self.low = float("inf")

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        s = trainer.global_step
        v = float(trainer.strategy.reduce(outputs["loss"].detach().float(), reduce_op="mean"))
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        self.ema = v if self.ema is None else 0.98 * self.ema + 0.02 * v
        if s > 300:
            self.low = min(self.low, self.ema)
        if trainer.is_global_zero:
            trainer.logger.log_metrics({"range/loss": v, "range/lr": lr}, step=s)
        if s > 300 and self.ema > self.stop * self.low:
            if trainer.is_global_zero:
                print(f"range test: smoothed loss {self.ema:.4f} > {self.stop} x minimum {self.low:.4f} at lr {lr:.2e}, stopping", flush=True)
            trainer.should_stop = True


class GpuSample(Callback):
    """After the second logged step, 30 s of nvidia-smi (utilisation, memory) on every node, started by the node's first task: runs/<name>/nvidia_smi_<job>_<node>.txt."""

    def __init__(self, out):
        self.out = out
        self.logged = 0

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.logged >= 2 or os.environ.get("SLURM_LOCALID", "0") != "0" or trainer.global_step % trainer.log_every_n_steps:
            return
        self.logged += 1
        if self.logged == 2:
            f = self.out / f"nvidia_smi_{os.environ.get('SLURM_JOB_ID', 'local')}_{socket.gethostname().split('.')[0]}.txt"
            subprocess.Popen(f"timeout 30 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 1 > {f}", shell=True)


class StopFile(Callback):
    """Clean stop on request: every `every` steps rank 0 looks for runs/<name>/STOP; if it is there, every rank saves checkpoints/last.ckpt, the file is removed, and fit
    returns, so the process exits 0 (a script that chains several training commands can then start the next one)."""

    def __init__(self, out, every=50):
        self.out = out
        self.every = every

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if trainer.global_step % self.every:
            return
        if not trainer.strategy.broadcast((self.out / "STOP").exists() if trainer.is_global_zero else None):
            return
        trainer.save_checkpoint(self.out / "checkpoints/last.ckpt")
        if trainer.is_global_zero:
            (self.out / "STOP").unlink(missing_ok=True)
            print(f"step {trainer.global_step}: STOP file found, saved last.ckpt, stopping", flush=True)
        trainer.should_stop = True


class EmaControl(Callback):
    """EMA of the UNet, applied after any checkpoint restore (on_train_start): `decay` replaces the EMA decay (stored in the checkpoint from then on); `reset` copies the
    online weights into the EMA once per run (marker runs/<name>/ema_reset.json), so the average no longer carries the weights it started from."""

    def __init__(self, out, decay=None, reset=False):
        self.out = out
        self.decay = decay
        self.reset = reset

    def on_train_start(self, trainer, module):
        ema = module.model_ema
        if self.decay is not None:
            ema.decay.fill_(self.decay)
        marker = self.out / "ema_reset.json"
        if self.reset and not marker.exists():
            shadow = dict(ema.named_buffers())
            for n, p in module.model.named_parameters():
                if p.requires_grad:
                    shadow[ema.m_name2s_name[n]].copy_(p.data)
            trainer.strategy.barrier()
            if trainer.is_global_zero:
                with open(marker, "w") as f:
                    json.dump({"step": trainer.global_step}, f)
        if trainer.is_global_zero:
            print(f"EMA: decay {float(ema.decay):.6f}, updates {int(ema.num_updates)}, reset at step {json.load(open(marker))['step'] if marker.exists() else None}", flush=True)


class Snapshots(ModelCheckpoint):
    """Weights-only snapshots beside last.ckpt (its own class: Lightning wants one state key per ModelCheckpoint)."""


class Speed(Callback):
    """Learning rate, step time and steps per hour to TensorBoard; loss, lr and steps/h to stdout every log_every_n_steps."""

    def __init__(self):
        self.t0 = None
        self.dts = []

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        self.t0 = time.time()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.dts.append(time.time() - self.t0)
        s = trainer.global_step
        if s % trainer.log_every_n_steps or not self.dts:
            return
        dt = float(np.mean(self.dts[-trainer.log_every_n_steps:]))
        self.dts = self.dts[-trainer.log_every_n_steps:]
        m = trainer.callback_metrics
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        if trainer.is_global_zero:
            trainer.logger.log_metrics({"lr": lr, "time/step_s": dt, "time/steps_per_h": 3600 / dt}, step=s)
            print(
                f"step {s}: loss {float(m.get('train/loss_simple_step', np.nan)):.4f} lr {lr:.2e} {dt:.3f} s/step {3600 / dt:.0f} steps/h",
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/zoomldm_gtex_uni.yaml")
    ap.add_argument("--name", default="gtex_uni")
    ap.add_argument("--probe", action="store_true", help="probe the batch size and exit")
    ap.add_argument("--root", default=None, help="read the slides from this directory (a staged copy); the saved configs keep the repo path")
    ap.add_argument("--set", nargs="*", default=[], help="dotlist overrides, e.g. trainer.max_steps=40 data.batch_size=16")
    a = ap.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / a.config), OmegaConf.from_dotlist(a.set))
    if cfg.get("schedule"):  # replace the configured scheduler: ldm.gtex_schedules.WarmupHoldCosine with these parameters
        cfg.model.params.scheduler_config = OmegaConf.create({"target": "ldm.gtex_schedules.WarmupHoldCosine", "params": cfg.schedule})
    out = ROOT / "runs" / a.name
    rank = int(os.environ.get("SLURM_PROCID", 0))
    t = cfg.trainer
    out.mkdir(parents=True, exist_ok=True)
    last = out / "checkpoints/last.ckpt"
    if rank == 0:
        OmegaConf.save(cfg, out / "config.yaml")
    if last.exists():
        step = torch.load(last, map_location="cpu", mmap=True, weights_only=False)["global_step"]
        cfg.data.train.params.config.seed += step
    if get_obj_from_str(cfg.model.target) is LatentDiffusion:
        model = GTExLDM(**cfg.model.params)
    else:
        model = instantiate_from_config(cfg.model)
    model.learning_rate = cfg.model.base_learning_rate
    model.lr_range = tuple(cfg.lr_range) if cfg.get("lr_range") else None
    for prefix in cfg.get("freeze") or []:  # e.g. cond_stage_model: those parameters get no gradient (the UNet's EMA covers only the UNet)
        frozen = [p.requires_grad_(False) for n, p in model.named_parameters() if n.startswith(prefix)]
        print(f"frozen: {prefix} ({sum(p.numel() for p in frozen) / 1e6:.1f} M parameters)", flush=True)
    if not last.exists():
        load_matching(model, cfg.init_ckpt)
    bfile = out / "batch.json"
    bs = cfg.data.batch_size
    if not bs and rank == 0 and not bfile.exists():
        bs = probe(model)
        tmp = bfile.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"batch_size": bs}, f)
        tmp.rename(bfile)
    while not bs:
        if bfile.exists():
            with open(bfile) as f:
                bs = json.load(f)["batch_size"]
        else:
            time.sleep(5)
    print(f"batch size {bs} per GPU", flush=True)
    if a.probe:
        return
    def loader(d, workers, **kw):
        if a.root:
            d = OmegaConf.merge(d, {"params": {"config": {"root": a.root}}})
        return DataLoader(
            instantiate_from_config(d),
            batch_size=bs,
            num_workers=workers,
            collate_fn=collate,
            pin_memory=True,
            persistent_workers=True,
            **kw,
        )

    dl = loader(cfg.data.train, cfg.data.num_workers, prefetch_factor=cfg.data.prefetch_factor, drop_last=True)
    vl = loader(cfg.data.validation, cfg.data.val_workers)
    cks = [
        ModelCheckpoint(
            dirpath=out / "checkpoints",
            save_last=True,
            save_top_k=0,
            train_time_interval=datetime.timedelta(minutes=t.ckpt_minutes),
        ),
        Snapshots(
            dirpath=out / "checkpoints",
            filename="{step:07d}",
            save_top_k=-1,
            save_weights_only=True,
            train_time_interval=datetime.timedelta(minutes=t.keep_minutes),
        ),
    ]
    n = torch.cuda.device_count() or int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))  # per node; CPU: one process per task (the CPU smoke test)
    nodes = int(os.environ.get("SLURM_NNODES", 1))  # srun starts one task per GPU on every node of the job
    if n * nodes > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = None
    extra = []
    if cfg.get("lr_range"):
        extra.append(RangeLog())
    if cfg.get("ema") and (cfg.ema.get("decay") is not None or cfg.ema.get("reset")):
        extra.append(EmaControl(out, cfg.ema.get("decay"), bool(cfg.ema.get("reset"))))
    if cfg.get("judge") and cfg.judge.enabled:
        j = {k: v for k, v in cfg.judge.items() if k != "enabled"}
        extra.append(Judge(out, cfg.model.params.scheduler_config.params.warm_up_steps[0], **j))
    tr = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=n,
        num_nodes=nodes,
        strategy=strategy,
        precision="bf16",
        max_time=t.max_time,
        max_steps=t.max_steps,
        log_every_n_steps=t.log_every_n_steps,
        logger=TensorBoardLogger(save_dir=out, name="", version="tb"),
        callbacks=[*cks, Best(out, cfg), Speed(), GpuSample(out), StopFile(out), *extra],
        val_check_interval=t.val_check_interval,
        num_sanity_val_steps=0,
        benchmark=True,
        enable_progress_bar=False,
    )
    tr.fit(model, dl, vl, ckpt_path=str(last) if last.exists() else None)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
