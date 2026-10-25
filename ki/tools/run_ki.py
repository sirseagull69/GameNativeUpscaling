#!/usr/bin/env python3
"""Run the KI upscaler shaders off-device on Mesa (EGL, OpenGL ES 3) and measure them.

Mirrors KiUpscaleEffect's GL calls: same shaders (dumped by ShaderDump.java), half-float
intermediate targets, bottom-up GL image orientation, and the same output pass.

Usage:
  run_ki.py <shader_dir> <out_dir> <image> [<image> ...]

For each high-resolution test image it downscales 2x (bicubic, as "the game rendering
at lower resolution"), upscales back with (a) plain bilinear, (b) the KI pipeline, and
reports Y-channel PSNR against the original. Also writes side-by-side comparison crops.
"""
import ctypes
import os
import sys

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import cv2  # noqa: E402
from OpenGL import EGL  # noqa: E402
from OpenGL import GLES3 as gl  # noqa: E402

GL_RGBA16F = 0x881A
GL_HALF_FLOAT = 0x140B


def make_context():
    dpy = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
    major, minor = EGL.EGLint(), EGL.EGLint()
    if not EGL.eglInitialize(dpy, ctypes.pointer(major), ctypes.pointer(minor)):
        raise SystemExit("eglInitialize failed")
    EGL.eglBindAPI(EGL.EGL_OPENGL_ES_API)
    cfg_attrs = (EGL.EGLint * 5)(EGL.EGL_RENDERABLE_TYPE, 0x40, EGL.EGL_SURFACE_TYPE, 0, EGL.EGL_NONE)  # ES3 bit
    cfg = EGL.EGLConfig()
    n = EGL.EGLint()
    EGL.eglChooseConfig(dpy, cfg_attrs, ctypes.pointer(cfg), 1, ctypes.pointer(n))
    if n.value < 1:
        raise SystemExit("no ES3 EGL config")
    ctx_attrs = (EGL.EGLint * 3)(EGL.EGL_CONTEXT_CLIENT_VERSION, 3, EGL.EGL_NONE)
    ctx = EGL.eglCreateContext(dpy, cfg, EGL.EGL_NO_CONTEXT, ctx_attrs)
    if not EGL.eglMakeCurrent(dpy, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, ctx):
        raise SystemExit("eglMakeCurrent failed")
    return dpy, ctx


def compile_program(vs_src, fs_src):
    def sh(kind, src):
        s = gl.glCreateShader(kind)
        gl.glShaderSource(s, src)
        gl.glCompileShader(s)
        if not gl.glGetShaderiv(s, gl.GL_COMPILE_STATUS):
            raise RuntimeError(gl.glGetShaderInfoLog(s).decode())
        return s
    p = gl.glCreateProgram()
    gl.glAttachShader(p, sh(gl.GL_VERTEX_SHADER, vs_src))
    gl.glAttachShader(p, sh(gl.GL_FRAGMENT_SHADER, fs_src))
    gl.glLinkProgram(p)
    if not gl.glGetProgramiv(p, gl.GL_LINK_STATUS):
        raise RuntimeError(gl.glGetProgramInfoLog(p).decode())
    return p


def make_texture(w, h, internal, fmt, typ, data, filt):
    t = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, t)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, typ, data)
    for k, v in ((gl.GL_TEXTURE_MIN_FILTER, filt), (gl.GL_TEXTURE_MAG_FILTER, filt),
                 (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE), (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
        gl.glTexParameteri(gl.GL_TEXTURE_2D, k, v)
    return t


def make_fbo(tex):
    f = gl.glGenFramebuffers(1)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, f)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, tex, 0)
    st = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
    if st != gl.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("fbo incomplete 0x%x" % st)
    return f


class Pipeline:
    def __init__(self, shader_dir):
        rd = lambda n: open(os.path.join(shader_dir, n), encoding="utf-8").read()
        vs = rd("vertex.vert")
        info = dict(l.split(":", 1) for l in rd("network.txt").split())
        self.output_name = info["output"]
        self.passes = []
        for i in range(int(info["passes"])):
            meta = rd("pass_%02d.meta" % i).split()
            save = [m[5:] for m in meta if m.startswith("save:")][0]
            binds = [m[5:] for m in meta if m.startswith("bind:")]
            prog = compile_program(vs, rd("pass_%02d.frag" % i))
            self.passes.append((prog, save, binds))
        self.out_prog = compile_program(vs, rd("output.frag"))
        # Full-screen quad as a triangle strip in [0,1]^2, attribute location 0.
        quad = np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype=np.float32)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad.nbytes, quad, gl.GL_STATIC_DRAW)
        gl.glDisable(gl.GL_BLEND)

    def draw(self):
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        gl.glDisableVertexAttribArray(0)

    def upscale(self, lr_rgb, out_w, out_h, network=True, preserve_aspect=False, strength=1.0):
        h, w = lr_rgb.shape[:2]
        rgba = np.dstack([lr_rgb, np.full((h, w), 255, np.uint8)])
        # GL stores bottom-up: the scene texture's first row is the image's bottom row.
        src = make_texture(w, h, gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                           np.ascontiguousarray(np.flipud(rgba)), gl.GL_NEAREST)
        targets = {}
        if network:
            for prog, save, binds in self.passes:
                if save not in targets:
                    t = make_texture(w, h, GL_RGBA16F, gl.GL_RGBA, GL_HALF_FLOAT, None, gl.GL_NEAREST)
                    targets[save] = (t, make_fbo(t))
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, targets[save][1])
                gl.glViewport(0, 0, w, h)
                gl.glUseProgram(prog)
                gl.glUniform2i(gl.glGetUniformLocation(prog, "inputSize"), w, h)
                for unit, b in enumerate(binds):
                    gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, src if b == "LUMA" else targets[b][0])
                    gl.glUniform1i(gl.glGetUniformLocation(prog, b), unit)
                self.draw()
        # Output pass (the composer samples the scene with LINEAR for this effect).
        gl.glBindTexture(gl.GL_TEXTURE_2D, src)
        out_t = make_texture(out_w, out_h, gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None, gl.GL_NEAREST)
        out_f = make_fbo(out_t)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, out_f)
        gl.glViewport(0, 0, out_w, out_h)
        p = self.out_prog
        gl.glUseProgram(p)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, src)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, targets[self.output_name][0] if network else 0)
        loc = lambda n: gl.glGetUniformLocation(p, n)
        gl.glUniform1i(loc("screenTexture"), 0)
        gl.glUniform1i(loc("subPixels"), 1)
        gl.glUniform2f(loc("inputResolution"), w, h)
        gl.glUniform2f(loc("outputResolution"), out_w, out_h)
        gl.glUniform1f(loc("networkActive"), 1.0 if network else 0.0)
        gl.glUniform1f(loc("preserveAspect"), 1.0 if preserve_aspect else 0.0)
        gl.glUniform1f(loc("detailStrength"), strength)
        self.draw()
        gl.glActiveTexture(gl.GL_TEXTURE0)
        buf = np.zeros((out_h, out_w, 4), np.uint8)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glReadPixels(0, 0, out_w, out_h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, buf)
        img = np.flipud(buf)[:, :, :3]
        # cleanup
        gl.glDeleteFramebuffers(1, [out_f])
        texs = [out_t, src] + [t for t, _ in targets.values()]
        gl.glDeleteTextures(len(texs), texs)
        gl.glDeleteFramebuffers(len(targets), [f for _, f in targets.values()])
        return np.ascontiguousarray(img)


def luma(img):
    f = img.astype(np.float64) / 255.0
    return f[..., 0] * 0.2126 + f[..., 1] * 0.7152 + f[..., 2] * 0.0722


def psnr(a, b, border=4):
    a, b = luma(a), luma(b)
    a, b = a[border:-border, border:-border], b[border:-border, border:-border]
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def downscale(hr, mode):
    """'lanczos' = antialiased (how FSRCNNX was trained, like a supersampled render);
    'aliased' = plain point-ish sampling, closer to a game rendering at low resolution."""
    h, w = hr.shape[:2]
    if mode == "aliased":
        return np.ascontiguousarray(hr[0::2, 0::2])
    from PIL import Image
    return np.asarray(Image.fromarray(hr).resize((w // 2, h // 2), Image.LANCZOS))


def main():
    shader_dir, out_dir = sys.argv[1], sys.argv[2]
    mode = os.environ.get("KI_DOWNSCALE", "lanczos")
    images = sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)
    make_context()
    pipe = Pipeline(shader_dir)
    print("renderer:", ctypes.cast(gl.glGetString(gl.GL_RENDERER), ctypes.c_char_p).value.decode())
    rows = []
    for path in images:
        hr = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = (hr.shape[0] // 2) * 2, (hr.shape[1] // 2) * 2
        hr = np.ascontiguousarray(hr[:h, :w])
        lr = downscale(hr, mode)
        bil = pipe.upscale(lr, w, h, network=False)
        ki = pipe.upscale(lr, w, h, network=True)
        cvbil = cv2.resize(lr, (w, h), interpolation=cv2.INTER_LINEAR)
        cub = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
        r = (os.path.basename(path), psnr(bil, hr), psnr(cub, hr), psnr(ki, hr), psnr(bil, cvbil))
        rows.append(r)
        print("%-16s bilinear %.2f dB | bicubic %.2f dB | KI %.2f dB | (GL bilinear vs OpenCV: %.1f dB)" % r)
        # comparison strip: original | bilinear | KI
        strip = np.hstack([hr, bil, ki])
        cv2.imwrite(os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_cmp.png"),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    if rows:
        print("MEAN             bilinear %.2f dB | bicubic %.2f dB | KI %.2f dB" % (
            np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows]), np.mean([r[3] for r in rows])))


if __name__ == "__main__":
    main()
