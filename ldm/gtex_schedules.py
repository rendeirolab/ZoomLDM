"""Learning-rate factor for the GTEx fine-tune experiments (scheduler_config target, used through LambdaLR like ldm.lr_scheduler):
linear warm-up from f_start to 1 over `warm_up` steps, constant until step `hold`, then a cosine from 1 to f_min over `decay` steps, then f_min (decay 0: constant)."""
import math


class WarmupHoldCosine:
    def __init__(self, warm_up=200, f_start=1e-6, hold=0, decay=0, f_min=0.0):
        self.warm_up, self.f_start, self.hold, self.decay, self.f_min = warm_up, f_start, hold, decay, f_min

    def schedule(self, n, **kwargs):
        if n < self.warm_up:
            return self.f_start + (1 - self.f_start) * n / self.warm_up
        if not self.decay or n < max(self.hold, self.warm_up):
            return 1.0
        t = min((n - max(self.hold, self.warm_up)) / self.decay, 1.0)
        return self.f_min + 0.5 * (1 - self.f_min) * (1 + math.cos(math.pi * t))

    def __call__(self, n, **kwargs):
        return self.schedule(n)
