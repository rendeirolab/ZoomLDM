"""ZoomLDM with UNI conditioning on GTEx windows (configs/zoomldm_gtex_uni.yaml), initialised from the released ZoomLDM-BRCA checkpoint, on every GPU of one node (DDP, one task per GPU).
Out: runs/<name>/ (config.yaml, batch.json, tb/ event files, checkpoints/last.ckpt every ckpt_minutes and weights-only {step}.ckpt every keep_minutes); a resubmitted job resumes from last.ckpt.
TensorBoard (tensorboard --logdir runs): train/loss_simple_step, lr, time/step_s, time/steps_per_h; the SLURM log gets the loss, the learning rate and steps/h every log_every_n_steps.
(lr is read from the optimizer; ddpm's own lr_abs is stale after a resume: Lightning's optimizer wrapper keeps the param groups from before the optimizer state was restored.)
Precision: forward pass under bf16 autocast (Trainer precision "bf16"); weights, gradients and optimizer state in fp32. Batch size: data.batch_size, or the largest that fits (probed on
rank 0 at the first start, dummy batches of 8 x 8 grids: the token count does not depend on the scale). Launch: scripts/train_gtex_uni.sbatch (stages the slides to the node's
local disk, sets the environment variables, samples nvidia-smi after the first logged steps into runs/<name>/nvidia_smi_<job>.txt)."""
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, default_collate

from ldm.util import instantiate_from_config

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


def load_matching(model, path):
    """Copy every tensor of the checkpoint whose name and shape match the model; return (copied, mismatched, missing, unexpected) name lists (mismatched/missing tensors keep their init)."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
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
    ap.add_argument("--set", nargs="*", default=[], help="dotlist overrides, e.g. trainer.max_steps=40 data.batch_size=16")
    a = ap.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / a.config), OmegaConf.from_dotlist(a.set))
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
    model = instantiate_from_config(cfg.model)
    model.learning_rate = cfg.model.base_learning_rate
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
    dl = DataLoader(
        instantiate_from_config(cfg.data.train),
        batch_size=bs,
        num_workers=cfg.data.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=cfg.data.prefetch_factor,
        drop_last=True,
    )
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
    n = torch.cuda.device_count()
    if n > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = None
    tr = pl.Trainer(
        accelerator="gpu",
        devices=n,
        num_nodes=1,
        strategy=strategy,
        precision="bf16",
        max_time=t.max_time,
        max_steps=t.max_steps,
        log_every_n_steps=t.log_every_n_steps,
        logger=TensorBoardLogger(save_dir=out, name="", version="tb"),
        callbacks=[*cks, Speed()],
        limit_val_batches=0,
        num_sanity_val_steps=0,
        benchmark=True,
        enable_progress_bar=False,
    )
    tr.fit(model, dl, ckpt_path=str(last) if last.exists() else None)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
