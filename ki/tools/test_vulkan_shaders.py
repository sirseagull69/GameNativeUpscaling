#!/usr/bin/env python3
"""Execute the generated Vulkan KI shaders (ki_pass_NN.comp + ki_window.frag) on Mesa by
translating them mechanically to OpenGL ES 3.1 (compute + fragment), keeping Vulkan's
top-down image orientation. Reports Y-PSNR on lr_/hr_ test frame pairs.

Usage: test_vulkan_shaders.py <generated dir> <pairs dir> [strength=1.0]
"""
import ctypes
import glob
import importlib.util
import os
import re
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("run_ki", os.path.join(HERE, "run_ki.py"))
rk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rk)
gl = rk.gl

GL_RGBA16F = 0x881A
GL_COMPUTE_SHADER = 0x91B9
GL_WRITE_ONLY = 0x88B9
GL_ALL_BARRIER_BITS = 0xFFFFFFFF

VS = """#version 310 es
layout(location = 0) in vec2 position;
layout(location = 0) out vec2 fragTexCoord;
void main() { fragTexCoord = position; gl_Position = vec4(2.0 * position - 1.0, 0.0, 1.0); }"""


def to_es_compute(src):
    src = src.replace("#version 450", "#version 310 es\nprecision highp float;\nprecision highp int;\n"
                      "precision highp image2D;\nprecision highp sampler2D;")
    src = src.replace("layout(push_constant) uniform PC { ivec2 inputSize; } pc;", "uniform ivec2 kiInputSize;")
    return src.replace("pc.inputSize", "kiInputSize")


def to_es_frag(src):
    src = src.replace("#version 450", "#version 310 es\nprecision highp float;\nprecision highp int;")
    src = src.replace("layout(set = 0, binding = 0)", "layout(binding = 0)")
    src = src.replace("layout(set = 1, binding = 0)", "layout(binding = 1)")
    src = re.sub(r"layout\(push_constant\) uniform PC \{.*?\} pc;",
                 "uniform float pc_resW, pc_resH, pc_sharpness, pc_brightness, pc_contrast, pc_gamma;", src, flags=re.S)
    return src.replace("pc.", "pc_")


def compile_compute(src):
    s = gl.glCreateShader(GL_COMPUTE_SHADER)
    gl.glShaderSource(s, src)
    gl.glCompileShader(s)
    if not gl.glGetShaderiv(s, gl.GL_COMPILE_STATUS):
        raise RuntimeError(gl.glGetShaderInfoLog(s).decode())
    p = gl.glCreateProgram()
    gl.glAttachShader(p, s)
    gl.glLinkProgram(p)
    if not gl.glGetProgramiv(p, gl.GL_LINK_STATUS):
        raise RuntimeError(gl.glGetProgramInfoLog(p).decode())
    return p


def parse_header(path):
    """Pass table from ki_spirv.h: binds and save indices."""
    rows = re.findall(r"\{ ki_pass_(\d+)_spv, sizeof\(ki_pass_\d+_spv\), (\d+), \{ ([-\d, ]+) \}, (\d+) \}",
                      open(path).read())
    out_img = int(re.search(r"KI_OUTPUT_IMAGE = (\d+)", open(path).read()).group(1))
    n_img = int(re.search(r"KI_IMAGE_COUNT = (\d+)", open(path).read()).group(1))
    passes = [(int(i), int(c), [int(b) for b in bs.split(",")], int(s)) for i, c, bs, s in rows]
    return passes, out_img, n_img


def main():
    gen_dir, pairs_dir = sys.argv[1], sys.argv[2]
    strength = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    rk.make_context()
    passes, out_img, n_img = parse_header(os.path.join(gen_dir, "ki_spirv.h"))
    progs = [compile_compute(to_es_compute(open(os.path.join(gen_dir, "ki_pass_%02d.comp" % i)).read()))
             for i, _, _, _ in passes]
    out_prog = rk.compile_program(VS, to_es_frag(open(os.path.join(gen_dir, "ki_window.frag")).read()))
    quad = rk.Pipeline.__new__(rk.Pipeline)
    q = np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype=np.float32)
    quad.vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, quad.vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, q.nbytes, q, gl.GL_STATIC_DRAW)

    scores = []
    for lp in sorted(glob.glob(os.path.join(pairs_dir, "lr_*.png"))):
        rd = lambda p: np.ascontiguousarray(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
        lr, hr = rd(lp), rd(lp.replace("lr_", "hr_"))
        h, w = lr.shape[:2]
        rgba = np.ascontiguousarray(np.dstack([lr, np.full((h, w), 255, np.uint8)]))  # top row first = Vulkan
        src = rk.make_texture(w, h, gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, rgba, gl.GL_LINEAR)
        imgs = []
        for _ in range(n_img):
            t = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, t)
            gl.glTexStorage2D(gl.GL_TEXTURE_2D, 1, GL_RGBA16F, w, h)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            imgs.append(t)
        for (i, cnt, binds, save), prog in zip(passes, progs):
            gl.glUseProgram(prog)
            gl.glUniform2i(gl.glGetUniformLocation(prog, "kiInputSize"), w, h)
            for unit, b in enumerate(binds[:cnt]):
                gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
                gl.glBindTexture(gl.GL_TEXTURE_2D, src if b == -1 else imgs[b])
            gl.glBindImageTexture(3, imgs[save], 0, gl.GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F)
            gl.glDispatchCompute((w + 7) // 8, (h + 7) // 8, 1)
            gl.glMemoryBarrier(GL_ALL_BARRIER_BITS)
        H, W = hr.shape[:2]
        out_t = rk.make_texture(W, H, gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None, gl.GL_NEAREST)
        fbo = rk.make_fbo(out_t)
        gl.glViewport(0, 0, W, H)
        gl.glUseProgram(out_prog)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, src)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, imgs[out_img])
        u = lambda n: gl.glGetUniformLocation(out_prog, n)
        gl.glUniform1f(u("pc_resW"), w)
        gl.glUniform1f(u("pc_resH"), h)
        gl.glUniform1f(u("pc_sharpness"), strength / 1.5)
        gl.glUniform1f(u("pc_brightness"), 0.0)
        gl.glUniform1f(u("pc_contrast"), 0.0)
        gl.glUniform1f(u("pc_gamma"), 1.0)
        quad.draw()
        buf = np.zeros((H, W, 4), np.uint8)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glReadPixels(0, 0, W, H, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, buf)
        out = np.ascontiguousarray(buf[:, :, :3])  # fragTexCoord.y=0 (top) was drawn at row 0
        scores.append(rk.psnr(out, hr))
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glDeleteFramebuffers(1, [fbo])
        texs = [src, out_t] + imgs
        gl.glDeleteTextures(len(texs), texs)
    print("Vulkan shaders (via ES 3.1 on Mesa): Y-PSNR %.2f dB over %d frames" % (np.mean(scores), len(scores)))


if __name__ == "__main__":
    main()
