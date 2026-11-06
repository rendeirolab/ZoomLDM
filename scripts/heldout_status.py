"""Held-out loss history of a run and whether it is finished (read from runs/<name>/tb, val/loss_ema; on a step logged twice after a resume the newest value wins).
Finished: `patience` validations in a row without a new best (a new best beats the previous one by at least `min_delta`; repeated evaluations of the same weights
differ by ~1e-5 under bf16), or `climb` validations in a row each above the one before and above the best. Prints the history, the best, both counts and the verdict:
  data/venvs/zoomldm-uni/bin/python scripts/heldout_status.py --run runs/gtex_uni_hgx_lr4.05e-05"""
import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--climb", type=int, default=3)
    ap.add_argument("--min_delta", type=float, default=2e-5)
    a = ap.parse_args()
    ea = EventAccumulator(str(Path(a.run) / "tb"), size_guidance={"scalars": 0})
    ea.Reload()
    latest = {}
    for e in ea.Scalars("val/loss_ema"):
        if e.step not in latest or e.wall_time > latest[e.step][0]:
            latest[e.step] = (e.wall_time, e.value)
    steps = sorted(latest)
    vals = [latest[s][1] for s in steps]
    best, best_step, since, rising = float("inf"), None, 0, 0
    for i, (s, v) in enumerate(zip(steps, vals)):
        if v < best - a.min_delta:
            best, best_step, since = v, s, 0
        else:
            since += 1
        rising = rising + 1 if i and v > vals[i - 1] and v > best else 0
    print("held-out loss (EMA weights): " + ", ".join(f"{s}: {v:.4f}" for s, v in zip(steps, vals)))
    print(f"best {best:.4f} at step {best_step}; {since} validation(s) since without a new best (min delta {a.min_delta}); {rising} rising in a row")
    finished = since >= a.patience or rising >= a.climb
    print(f"verdict: {'finished' if finished else 'continue'} (finished after {a.patience} without a new best or {a.climb} rising in a row)")


if __name__ == "__main__":
    main()
