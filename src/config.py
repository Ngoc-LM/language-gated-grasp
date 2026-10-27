"""Experiment defaults. Large caches and transient training runs stay outside working/."""
from dataclasses import dataclass, field, asdict

@dataclass
class Config:
    variants: list[str] = field(default_factory=lambda: ['A', 'B'])
    variant: str = 'A'
    img: int = 224
    batch: int = 32
    lr: float = 1e-3
    wd: float = 1e-4
    epochs: int = 6
    max_hours: float = 3.5
    warmup_steps: int = 500
    amp: bool = True
    seed: int = 0
    num_workers: int = 2
    width_norm: float = 150.0
    min_rect_px: float = 2.0
    lambda_gate: float = 0.5
    val_subset: int = 1000
    smoke: bool = False
    pack_root: str = '/tmp/ga_pack'
    cache: str = '/tmp/ga_cache'
    out: str = '/tmp/ga_runs/A'
    log_every: int = 200

    def validate(self):
        if self.variant not in 'ABCDE' or len(self.variant) != 1:
            raise ValueError('variant must be A, B, C, D or E (E = C0: text-free CLIP gate)')
        if self.img != 224:
            raise ValueError('This architecture/cache contract requires img=224')
        if self.batch < 1 or self.epochs < 1 or self.max_hours <= 0:
            raise ValueError('batch, epochs and max_hours must be positive')
        return self

    def to_dict(self):
        return asdict(self)
