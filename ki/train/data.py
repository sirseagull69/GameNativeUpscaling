"""Synthetic "game frame" pairs for training/evaluating the KI upscaler.

A low-res game frame without anti-aliasing is a point sample of the scene at each
low-res pixel centre. We emulate that exactly: shapes and text are rasterised into a
4x-supersampled mask of the high-res frame, then
  * LR mask = mask sampled at LR pixel centres  (hard, aliased edges)
  * HR target = 4x4 box average of the mask      (clean, antialiased edges at 2x)
Textures (photo crops) are filtered in LR like mipmapped texture sampling.
"""
import glob
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
SS = 4  # supersampling factor of the HR frame for masks

_FONTS = [f for f in glob.glob("/usr/share/fonts/truetype/**/*.ttf", recursive=True)
          if "Bold" in f or "Regular" in f or "Sans" in f][:40]
_WORDS = ["HP", "MP", "LEVEL", "Score", "Repair", "Start", "Options", "Quest", "x12", "Gold",
          "Inventory", "Pause", "Continue", "LOADING", "Fix it!", "Battery", "Screen", "OK",
          "Save", "Back", "1080", "Time 03:21", "Coins 450", "Lv. 27", "ReStory"]


class Textures:
    def __init__(self, folder):
        self.images = [np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
                       for p in sorted(glob.glob(os.path.join(folder, "*")))]

    def crop(self, rng, h, w):
        """Random HR crop of size (h, w); resamples if the photo is too small."""
        img = self.images[rng.randrange(len(self.images))]
        scale = rng.uniform(0.5, 1.5)
        H, W = img.shape[:2]
        ch, cw = int(h / scale), int(w / scale)
        if ch > H or cw > W:
            scale = max(h / H, w / W) * 1.05
            ch, cw = int(h / scale), int(w / scale)
        y = rng.randrange(0, H - ch + 1)
        x = rng.randrange(0, W - cw + 1)
        c = img[y:y + ch, x:x + cw]
        if (ch, cw) != (h, w):
            c = np.asarray(Image.fromarray((c * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
                           np.float32) / 255.0
        if rng.random() < 0.5:
            c = c[:, ::-1]
        return np.ascontiguousarray(c)


def box_down(img, f):
    h, w = img.shape[:2]
    return img.reshape(h // f, f, w // f, f, *img.shape[2:]).mean(axis=(1, 3))


def _random_color(rng):
    return np.array([rng.random(), rng.random(), rng.random()], np.float32)


def _draw_shape(draw, rng, W, H):
    kind = rng.random()
    if kind < 0.45:
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.uniform(0.08, 0.6) * min(W, H)
        n = rng.randint(3, 7)
        ang = sorted(rng.uniform(0, 2 * np.pi) for _ in range(n))
        pts = [(cx + r * rng.uniform(0.4, 1.0) * np.cos(a), cy + r * rng.uniform(0.4, 1.0) * np.sin(a)) for a in ang]
        draw.polygon(pts, fill=255)
    elif kind < 0.65:
        x0, y0 = rng.uniform(-0.2, 1.0) * W, rng.uniform(-0.2, 1.0) * H
        x1, y1 = x0 + rng.uniform(0.1, 0.7) * W, y0 + rng.uniform(0.1, 0.7) * H
        draw.ellipse([x0, y0, x1, y1], fill=255)
    elif kind < 0.85:
        for _ in range(rng.randint(1, 4)):
            pts = [(rng.uniform(0, W), rng.uniform(0, H)) for _ in range(2)]
            draw.line(pts, fill=255, width=max(1, int(rng.uniform(0.25, 3.0) * SS * 2)))
    else:
        font_path = rng.choice(_FONTS) if _FONTS else None
        size = int(rng.uniform(6, 28) * SS * 2)  # in supersampled HR px (LR glyphs ~6-28 px)
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default(size=size)
        draw.text((rng.uniform(-0.1, 0.8) * W, rng.uniform(-0.1, 0.9) * H), rng.choice(_WORDS), fill=255, font=font)


def game_frame(tex, rng, lr_h, lr_w, n_shapes=None):
    """Returns (lr_rgb, hr_rgb) float32 in [0,1]; HR is exactly 2x LR."""
    hr_h, hr_w = 2 * lr_h, 2 * lr_w
    # Background texture: HR detail, LR is the filtered (mipmapped) version.
    bg_hr = tex.crop(rng, hr_h, hr_w)
    if rng.random() < 0.25:  # flat/gradient sky-like areas are common in games
        a, b = _random_color(rng), _random_color(rng)
        t = np.linspace(0, 1, hr_h, dtype=np.float32)[:, None, None]
        bg_hr = (1 - t) * a + t * b + np.zeros((hr_h, hr_w, 3), np.float32)
    hr = bg_hr.copy()
    lr = box_down(bg_hr, 2)

    ssW, ssH = hr_w * SS, hr_h * SS
    for _ in range(n_shapes if n_shapes is not None else rng.randint(2, 9)):
        mask_img = Image.new("L", (ssW, ssH), 0)
        draw = ImageDraw.Draw(mask_img)
        draw.fontmode = "1"  # aliased rasterisation; AA comes only from our own sampling
        _draw_shape(draw, rng, ssW, ssH)
        mask = np.asarray(mask_img, np.float32) / 255.0
        m_hr = box_down(mask, SS)[..., None]                              # antialiased coverage at HR
        o = SS * 2 // 2                                                   # LR pixel centre offset in ss px
        m_lr = mask[o::SS * 2, o::SS * 2][:lr_h, :lr_w][..., None]        # point sample at LR centres
        if rng.random() < 0.6:
            fill_hr = np.broadcast_to(_random_color(rng), (hr_h, hr_w, 3))
            if rng.random() < 0.4:  # simple shading gradient
                g = np.linspace(rng.uniform(0.5, 1.0), rng.uniform(0.5, 1.0), hr_w, dtype=np.float32)[None, :, None]
                fill_hr = np.clip(fill_hr * g, 0, 1)
            fill_lr = box_down(np.ascontiguousarray(fill_hr), 2)
        else:
            fill_hr = tex.crop(rng, hr_h, hr_w)
            fill_lr = box_down(fill_hr, 2)
        hr = hr * (1 - m_hr) + fill_hr * m_hr
        lr = lr * (1 - m_lr) + fill_lr * m_lr
    return np.clip(lr, 0, 1).astype(np.float32), np.clip(hr, 0, 1).astype(np.float32)


def photo_pair(tex, rng, lr_h, lr_w):
    """AA/supersampled game look: plain filtered downscale of a photo crop."""
    hr = tex.crop(rng, 2 * lr_h, 2 * lr_w)
    lr = np.asarray(Image.fromarray((hr * 255).astype(np.uint8)).resize((lr_w, lr_h), Image.LANCZOS),
                    np.float32) / 255.0
    return lr, hr


def luma(rgb):
    return rgb @ LUMA


def make_dataset(tex_folder, n, lr_size=48, photo_share=0.3, seed=0):
    tex = Textures(tex_folder)
    rng = random.Random(seed)
    X = np.zeros((n, lr_size, lr_size, 1), np.float32)
    Y = np.zeros((n, 2 * lr_size, 2 * lr_size, 1), np.float32)
    for i in range(n):
        if rng.random() < photo_share:
            lr, hr = photo_pair(tex, rng, lr_size, lr_size)
        else:
            lr, hr = game_frame(tex, rng, lr_size, lr_size)
        X[i, ..., 0] = luma(lr)
        Y[i, ..., 0] = luma(hr)
    return X, Y


def write_test_frames(tex_folder, out_dir, n=12, lr_w=480, lr_h=270, seed=1234):
    """Full test frames (LR + HR PNG pairs), e.g. 480x270 -> 960x540 (a quarter of the Thor's
    960x540 -> 1920x1080 case, to keep the software-GPU test fast)."""
    os.makedirs(out_dir, exist_ok=True)
    tex = Textures(tex_folder)
    rng = random.Random(seed)
    for i in range(n):
        lr, hr = game_frame(tex, rng, lr_h, lr_w, n_shapes=rng.randint(6, 14))
        Image.fromarray((lr * 255 + 0.5).astype(np.uint8)).save(os.path.join(out_dir, "lr_%02d.png" % i))
        Image.fromarray((hr * 255 + 0.5).astype(np.uint8)).save(os.path.join(out_dir, "hr_%02d.png" % i))
