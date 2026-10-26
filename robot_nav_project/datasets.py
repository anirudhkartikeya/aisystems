"""datasets.py - shared helpers for the data-collection scripts."""
import config


def split_for_seed(seed):
    """train / val / test for one episode. Whole episodes stay in one split,
    so no near-identical frames or tracks leak from training into testing."""
    r = seed % 20
    for name, rng in config.SPLIT_OF_20.items():
        if r in rng:
            return name
    return "train"


def vary_episode(rng, layout):
    """Give one episode its own crowd size, traffic and behaviour mix, so the
    data covers quiet streets, busy ones, careful and careless people.
    Returns a dict of the config values changed (to restore later)."""
    base = {"town": (18, 40, 5, 14), "street": (6, 18, 1, 6)}.get(layout, (3, 6, 0, 0))
    changes = {
        "N_PEDS": {layout: int(rng.integers(base[0], base[1] + 1))},
        "N_CARS": {layout: int(rng.integers(base[2], base[3] + 1))},
        "PED_AWARE_SHARE": float(rng.uniform(0.6, 0.9)),
        "PED_PHONE_SHARE": float(rng.uniform(0.05, 0.25)),
        "PED_JAYWALKER_SHARE": float(rng.uniform(0.0, 0.25)),
        "PED_PAUSE_RATE": float(rng.uniform(0.01, 0.04)),
        "PED_TURN_BACK_PROB": float(rng.uniform(0.0, 0.08)),
    }
    old = {k: getattr(config, k) for k in changes}
    for k, v in changes.items():
        setattr(config, k, v)
    return old


def restore(old):
    for k, v in old.items():
        setattr(config, k, v)
