#!/usr/bin/env python3
"""Fine-tune the FSRCNNX 8-0-4-1 network on synthetic game frames (CPU, JAX).

Usage:
  train.py --init <FSRCNNX.glsl> --textures <dir> --test <gametest dir> --out <out.glsl>
           [--samples 12000] [--steps 6000] [--batch 32] [--lr 2e-4]
"""
import argparse
import glob
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
import model as M  # noqa: E402

BORDER = 8  # HR pixels ignored in the loss (edge padding effects)


def load_test(folder):
    xs, ys = [], []
    for lp in sorted(glob.glob(os.path.join(folder, "lr_*.png"))):
        lr = np.asarray(Image.open(lp).convert("RGB"), np.float32) / 255.0
        hr = np.asarray(Image.open(lp.replace("lr_", "hr_")).convert("RGB"), np.float32) / 255.0
        xs.append(data.luma(lr)[None, :, :, None])
        ys.append(data.luma(hr)[None, :, :, None])
    return xs, ys


def psnr_y(P, xs, ys):
    vals = []
    for x, y in zip(xs, ys):
        p = np.clip(np.asarray(M.forward(P, x)), 0, 1)
        d = (p - y)[:, 4:-4, 4:-4]
        vals.append(10 * np.log10(1.0 / max(float(np.mean(d * d)), 1e-12)))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--textures", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=12000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--photo-share", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    X, Y = data.make_dataset(a.textures, a.samples, photo_share=a.photo_share, seed=a.seed)
    print("dataset %d samples in %.0fs" % (len(X), time.time() - t0), flush=True)
    tx, ty = load_test(a.test)

    P = {k: jnp.asarray(v) for k, v in M.params_from_mpv(open(a.init).read()).items()}
    print("init test PSNR (game frames): %.2f dB" % psnr_y(P, tx, ty), flush=True)

    sched = optax.cosine_decay_schedule(a.lr, a.steps, alpha=0.05)
    opt = optax.adam(sched)
    state = opt.init(P)

    def loss_fn(P, x, y):
        p = M.forward(P, x)
        d = (p - y)[:, BORDER:-BORDER, BORDER:-BORDER]
        return jnp.mean(jnp.sqrt(d * d + 1e-6))  # Charbonnier (smooth L1)

    @jax.jit
    def step(P, state, x, y):
        l, g = jax.value_and_grad(loss_fn)(P, x, y)
        u, state = opt.update(g, state, P)
        return optax.apply_updates(P, u), state, l

    rng = np.random.default_rng(a.seed)
    best, best_P = -1.0, P
    t0 = time.time()
    run = 0.0
    for i in range(1, a.steps + 1):
        idx = rng.integers(0, len(X), a.batch)
        x, y = X[idx], Y[idx]
        if rng.random() < 0.5:  # augment: transpose / flips (keeps the 2x grid aligned)
            x, y = x.transpose(0, 2, 1, 3), y.transpose(0, 2, 1, 3)
        if rng.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, :, ::-1]
        P, state, l = step(P, state, jnp.asarray(np.ascontiguousarray(x)), jnp.asarray(np.ascontiguousarray(y)))
        run = 0.98 * run + 0.02 * float(l) if i > 1 else float(l)
        if i % 500 == 0 or i == a.steps:
            score = psnr_y(P, tx, ty)
            if score > best:
                best, best_P = score, P
                open(a.out, "w").write(M.to_mpv({k: np.asarray(v) for k, v in best_P.items()}))
            print("step %5d  loss %.5f  test %.2f dB  best %.2f  (%.0fs)" % (i, run, score, best, time.time() - t0),
                  flush=True)
    print("done, best %.2f dB -> %s" % (best, a.out), flush=True)


if __name__ == "__main__":
    main()
