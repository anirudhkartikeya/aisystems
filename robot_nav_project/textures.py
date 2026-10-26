"""textures.py - surface pictures for the ground: asphalt, sidewalk slabs,
grass. Made with numpy when the simulator starts (no image files needed).

Every texture tiles seamlessly: the noise is built in frequency space, so
its left edge matches its right edge and its top matches its bottom.
"""
import os

import numpy as np


def _periodic_noise(size, rng, smooth):
    """Seamless random noise in 0..1; bigger `smooth` = bigger blotches."""
    white = rng.normal(0, 1, (size, size))
    f = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(f, f)
    radius = np.sqrt(fx * fx + fy * fy)
    filt = np.exp(-(radius * smooth) ** 2)
    noise = np.real(np.fft.ifft2(np.fft.fft2(white) * filt))
    noise -= noise.min()
    return noise / max(noise.max(), 1e-9)


def _rgb(gray, tint):
    return np.clip(gray[:, :, None] * np.array(tint)[None, None, :] * 255, 0, 255).astype(np.uint8)


def asphalt(rng, size=256, base=0.25, tint=(1.0, 1.0, 1.04)):
    g = base + 0.06 * (_periodic_noise(size, rng, 40) - 0.5) + 0.05 * (_periodic_noise(size, rng, 3) - 0.5)
    g += (rng.random((size, size)) < 0.02) * 0.05                       # light stones in the mix
    # a few thin dark cracks / tar seams
    for _ in range(3):
        x, y = rng.integers(0, size, 2)
        for _ in range(int(rng.integers(30, 90))):
            g[y % size, x % size] -= 0.06
            x += int(rng.integers(-1, 2))
            y += int(rng.integers(0, 2))
    return _rgb(g, tint)


def sidewalk(rng, size=256, slabs=4, base=0.76):
    g = base + 0.05 * (_periodic_noise(size, rng, 6) - 0.5) + 0.03 * (_periodic_noise(size, rng, 30) - 0.5)
    step = size // slabs
    for k in range(slabs):                                              # slab joints
        g[k * step:k * step + 2, :] -= 0.12
        g[:, k * step:k * step + 2] -= 0.12
        for j in range(slabs):                                          # each slab a slightly different shade
            g[k * step + 2:(k + 1) * step, j * step + 2:(j + 1) * step] += rng.uniform(-0.025, 0.025)
    return _rgb(g, (1.0, 0.985, 0.95))


def grass(rng, size=256, park=False):
    g = 0.5 + 0.35 * (_periodic_noise(size, rng, 25) - 0.5) + 0.25 * (_periodic_noise(size, rng, 4) - 0.5)
    base = np.array((0.36, 0.60, 0.30) if park else (0.42, 0.58, 0.33))
    img = (g[:, :, None] * 0.6 + 0.7) * base[None, None, :]
    img += (rng.random((size, size, 1)) - 0.5) * 0.05                   # blades
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def make_all(folder, seed=7):
    """Write every texture as a .png into `folder`; returns {name: path}."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    images = {"asphalt": asphalt(rng), "parking": asphalt(rng, base=0.29, tint=(1.0, 1.0, 1.0)),
              "sidewalk": sidewalk(rng), "grass": grass(rng), "park": grass(rng, park=True)}
    paths = {}
    for name, img in images.items():
        path = os.path.join(folder, f"tex_{name}.png")
        Image.fromarray(img).save(path)
        paths[name] = path
    return paths
