#!/usr/bin/env python3
"""Compare upscalers on Mesa (OpenGL ES): bilinear, bicubic, SGSR v1, and one or more
KI networks (dumped shader dirs from ShaderDump.java). Reports Y-PSNR vs the original
for two kinds of low-res input:

  aliased  - every 2nd pixel (a game rendering at half resolution without anti-aliasing)
  lanczos  - antialiased downscale (a game with good AA / supersampling)

Usage:
  compare.py --sgsr <sgsr1_shader_mobile.frag> --net NAME=<shader_dir> [--net ...]
             [--strength NAME=float] [--save <dir>] <image> [<image> ...]
"""
import argparse
import ctypes
import importlib.util
import os
import re

import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("run_ki", os.path.join(HERE, "run_ki.py"))
rk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rk)
gl = rk.gl

SGSR_VS = """#version 310 es
layout(location = 0) in vec2 position;
layout(location = 0) out highp vec4 in_TEXCOORD0;
void main() {
    in_TEXCOORD0 = vec4(position, 0.0, 0.0);
    gl_Position = vec4(2.0 * position - 1.0, 0.0, 1.0);
}"""


class Sgsr:
    """Runs Qualcomm's SGSR v1 mobile shader exactly as published (RGBA mode).
    The image is uploaded top row first, matching the shader's top-left UV convention."""

    def __init__(self, frag_path, pipe):
        src = open(frag_path, encoding="utf-8").read()
        src = src.replace("#version 300 es", "#version 310 es")  # textureGather
        # Adreno accepts precision qualifiers inside expressions ("+ highp vec2(...)"); Mesa does not.
        src = re.sub(r"([+(,])\s*highp\s+vec2\(", r"\1 vec2(", src)
        # textureGather's component must be a constant expression on Mesa.
        src = re.sub(r"(textureGather\([^;]*?),\s*mode\)", r"\1, OperationMode)", src)
        self.prog = rk.compile_program(SGSR_VS, src)
        self.pipe = pipe

    def upscale(self, lr, out_w, out_h):
        h, w = lr.shape[:2]
        rgba = np.ascontiguousarray(np.dstack([lr, np.full((h, w), 255, np.uint8)]))
        src = rk.make_texture(w, h, gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, rgba, gl.GL_LINEAR)
        out_t = rk.make_texture(out_w, out_h, gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None, gl.GL_NEAREST)
        fbo = rk.make_fbo(out_t)
        gl.glViewport(0, 0, out_w, out_h)
        gl.glUseProgram(self.prog)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, src)
        gl.glUniform1i(gl.glGetUniformLocation(self.prog, "ps0"), 0)
        gl.glUniform4f(gl.glGetUniformLocation(self.prog, "ViewportInfo"), 1.0 / w, 1.0 / h, w, h)
        self.pipe.draw()
        buf = np.zeros((out_h, out_w, 4), np.uint8)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glReadPixels(0, 0, out_w, out_h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, buf)
        gl.glDeleteFramebuffers(1, [fbo])
        gl.glDeleteTextures(2, [src, out_t])
        return np.ascontiguousarray(buf[:, :, :3])


def downscale(hr, mode):
    h, w = hr.shape[:2]
    if mode == "aliased":
        return np.ascontiguousarray(hr[0::2, 0::2])
    return np.asarray(Image.fromarray(hr).resize((w // 2, h // 2), Image.LANCZOS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgsr")
    ap.add_argument("--net", action="append", default=[])
    ap.add_argument("--strength", action="append", default=[])
    ap.add_argument("--save")
    ap.add_argument("--modes", default="aliased,lanczos")
    ap.add_argument("--pairs", help="directory with lr_XX.png / hr_XX.png test frames")
    ap.add_argument("images", nargs="*")
    a = ap.parse_args()

    rk.make_context()
    nets = []
    for spec_ in a.net:
        name, d = spec_.split("=", 1)
        nets.append((name, rk.Pipeline(d)))
    strengths = dict((s.split("=")[0], float(s.split("=")[1])) for s in a.strength)
    base_pipe = nets[0][1] if nets else None
    sgsr = Sgsr(a.sgsr, base_pipe) if a.sgsr else None

    images = []
    for path in a.images:
        hr = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = (hr.shape[0] // 2) * 2, (hr.shape[1] // 2) * 2
        images.append((os.path.basename(path), np.ascontiguousarray(hr[:h, :w])))

    if a.pairs:
        import glob as _g
        pairs = []
        for lp in sorted(_g.glob(os.path.join(a.pairs, "lr_*.png"))):
            hp = lp.replace("lr_", "hr_")
            rd = lambda q: np.ascontiguousarray(cv2.cvtColor(cv2.imread(q, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            pairs.append((os.path.basename(lp), rd(lp), rd(hp)))
        scores = {}
        for name, lr, hr in pairs:
            h, w = hr.shape[:2]
            outs = {"bilinear": cv2.resize(lr, (w, h), interpolation=cv2.INTER_LINEAR),
                    "bicubic": cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)}
            if sgsr:
                outs["SGSR"] = sgsr.upscale(lr, w, h)
            for nname, pipe in nets:
                outs[nname] = pipe.upscale(lr, w, h, network=True, strength=strengths.get(nname, 1.0))
            for k, v in outs.items():
                scores.setdefault(k, []).append(rk.psnr(v, hr))
            if a.save:
                os.makedirs(a.save, exist_ok=True)
                cv2.imwrite(os.path.join(a.save, "pairs_" + name),
                            cv2.cvtColor(np.hstack([hr] + list(outs.values())), cv2.COLOR_RGB2BGR))
        print("%-8s %s  (dB, %d frames)" % ("game", " | ".join("%s %.2f" % (k, np.mean(v)) for k, v in scores.items()), len(pairs)), flush=True)
        if not images:
            return

    for mode in a.modes.split(","):
        scores = {}
        for name, hr in images:
            h, w = hr.shape[:2]
            lr = downscale(hr, mode)
            outs = {
                "bilinear": cv2.resize(lr, (w, h), interpolation=cv2.INTER_LINEAR),
                "bicubic": cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC),
            }
            if sgsr:
                outs["SGSR"] = sgsr.upscale(lr, w, h)
            for nname, pipe in nets:
                outs[nname] = pipe.upscale(lr, w, h, network=True, strength=strengths.get(nname, 1.0))
            for k, v in outs.items():
                scores.setdefault(k, []).append(rk.psnr(v, hr))
            if a.save:
                os.makedirs(a.save, exist_ok=True)
                strip = np.hstack([hr] + [outs[k] for k in outs])
                cv2.imwrite(os.path.join(a.save, "%s_%s.png" % (mode, os.path.splitext(name)[0])),
                            cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        line = " | ".join("%s %.2f" % (k, np.mean(v)) for k, v in scores.items())
        print("%-8s %s  (dB, %d images)" % (mode, line, len(images)), flush=True)


if __name__ == "__main__":
    main()
