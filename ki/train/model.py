"""FSRCNNX-style network (d=8, no shrink, m=4 mapping layers, 2x) in JAX, with exact
round-tripping to/from the mpv user-shader format that GameNative's KI effect runs.

Semantics mirror the shader (and KiShaders.passFragmentShader):
  * input is luma Y in [0,1], image coordinates x right / y down, edges clamped
  * feature: 5x5 conv 1->8 (no activation)
  * mapping x4: 3x3 conv 8->8 + PReLU
  * residual: 1x1 conv 8->8 + feature, then PReLU
  * subpixel: 3x3 conv 8->4, output channel c = sx*2 + sy is HR pixel (2x+sx, 2y+sy)
GLSL mat4(a0..a15) * v  ==  W @ v  with  W[i][k] = a[4k + i]  (column-major).
"""
import re

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------- parsing

_TERM = re.compile(
    r"res \+= (vec4|mat4|mat4x4)\(([^)]*)\) \* (?:float\()?(\w+)_texOff\((vec2\([^)]*\)|0)\)")
_BIAS = re.compile(r"vec4 res = vec4\(([^)]*)\);")
_PRELU = re.compile(r"res = max\(res, vec4\(0\.0\)\) \+ vec4\(([^)]*)\) \* min\(res, vec4\(0\.0\)\);")
_ADD = re.compile(r"res \+= (\w+)_texOff\((?:vec2\([^)]*\)|0)\);")


def _floats(s):
    return [float(v) for v in s.split(",")]


def _offset(s):
    if s == "0":
        return (0, 0)
    a, b = _floats(s[5:-1])
    return (int(a), int(b))


def parse_passes(src):
    passes = []
    for block in re.split(r"(?m)^(?=//!HOOK)", src):
        if not block.startswith("//!HOOK"):
            continue
        meta = dict(re.findall(r"//!(DESC|SAVE)\s*(.*)", block))
        p = {"desc": meta.get("DESC", ""), "save": meta.get("SAVE"), "terms": [], "adds": [],
             "bias": None, "prelu": None}
        m = _BIAS.search(block)
        if m:
            p["bias"] = _floats(m.group(1))
        for kind, coefs, tex, off in _TERM.findall(block):
            p["terms"].append((kind, _floats(coefs), tex, _offset(off)))
        p["adds"] = _ADD.findall(block)
        m = _PRELU.search(block)
        if m:
            p["prelu"] = _floats(m.group(1))
        passes.append(p)
    return passes


def params_from_mpv(src):
    """Return a dict of numpy arrays: conv weights as [kh, kw, in, out] (HWIO, image coords)."""
    passes = {p["desc"]: p for p in parse_passes(src)}
    P = {}

    # feature 5x5 1->8
    w = np.zeros((5, 5, 1, 8), np.float32)
    b = np.zeros(8, np.float32)
    for half, name in enumerate(["feature map 1", "feature map 2"]):
        p = passes[name]
        b[half * 4:half * 4 + 4] = p["bias"]
        for kind, c, tex, (dx, dy) in p["terms"]:
            w[dy + 2, dx + 2, 0, half * 4:half * 4 + 4] = c
    P["feat_w"], P["feat_b"] = w, b

    def conv8(names, k, in_names):
        w = np.zeros((k, k, 8, 8), np.float32)
        b = np.zeros(8, np.float32)
        a = np.zeros(8, np.float32)
        r = k // 2
        for half, name in enumerate(names):
            p = passes[name]
            b[half * 4:half * 4 + 4] = p["bias"]
            if p["prelu"] is not None:
                a[half * 4:half * 4 + 4] = p["prelu"]
            for kind, c, tex, (dx, dy) in p["terms"]:
                in_half = in_names.index(tex)
                W = np.array(c, np.float32).reshape(4, 4).T  # W[i][k] = a[4k+i]
                w[dy + r, dx + r, in_half * 4:in_half * 4 + 4, half * 4:half * 4 + 4] += W.T
        return w, b, a

    ins = ["FEATURE1", "FEATURE2"]
    for layer in range(1, 5):
        w, b, a = conv8(["mapping %d_1" % layer, "mapping %d_2" % layer], 3, ins)
        P["map%d_w" % layer], P["map%d_b" % layer], P["map%d_a" % layer] = w, b, a
        ins = ["MODEL21", "MODEL22"] if layer % 2 == 1 else ["MODEL1", "MODEL2"]
    w, b, a = conv8(["sub-band residuals 1", "sub-band residuals 2"], 1, ["MODEL1", "MODEL2"])
    P["res_w"], P["res_b"], P["res_a"] = w, b, a

    p = passes["sub-pixel convolution 1"]
    w = np.zeros((3, 3, 8, 4), np.float32)
    for kind, c, tex, (dx, dy) in p["terms"]:
        in_half = ["RES1", "RES2"].index(tex)
        W = np.array(c, np.float32).reshape(4, 4).T
        w[dy + 1, dx + 1, in_half * 4:in_half * 4 + 4, :] += W.T
    P["sub_w"], P["sub_b"] = w, np.array(p["bias"], np.float32)
    return P


# ---------------------------------------------------------------- forward

def _conv(x, w, b):
    """x: [N,H,W,C]; w: HWIO; edge-clamped 'same' conv like the shader's clamped fetches."""
    k = w.shape[0]
    r = k // 2
    if r:
        x = jnp.pad(x, ((0, 0), (r, r), (r, r), (0, 0)), mode="edge")
    y = jax.lax.conv_general_dilated(x, w, (1, 1), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + b


def _prelu(x, a):
    return jnp.maximum(x, 0) + a * jnp.minimum(x, 0)


def forward(P, y):
    """y: [N,H,W,1] luma -> [N,2H,2W,1]."""
    feat = _conv(y, P["feat_w"], P["feat_b"])
    h = feat
    for layer in range(1, 5):
        h = _prelu(_conv(h, P["map%d_w" % layer], P["map%d_b" % layer]), P["map%d_a" % layer])
    h = _prelu(_conv(h, P["res_w"], P["res_b"]) + feat, P["res_a"])
    sub = _conv(h, P["sub_w"], P["sub_b"])  # [N,H,W,4], c = sx*2+sy
    n, H, W, _ = sub.shape
    s = sub.reshape(n, H, W, 2, 2)           # [..., sx, sy]
    s = jnp.transpose(s, (0, 1, 4, 2, 3))    # [N, H, sy, W, sx]
    return s.reshape(n, 2 * H, 2 * W, 1)


# ---------------------------------------------------------------- export

def _fmt(v):
    return "%.10f" % float(v)


def _vec4(vals):
    return "vec4(" + ",".join(_fmt(v) for v in vals) + ")"


def _mat4(W):
    # W[i][k] (out i, in k)  ->  GLSL column-major a[4k+i]
    return "mat4(" + ",".join(_fmt(W[i][k]) for k in range(4) for i in range(4)) + ")"


def to_mpv(P, header="// Trained for GameNative KI (FSRCNNX 8-0-4-1 structure)"):
    P = {k: np.asarray(v, np.float64) for k, v in P.items()}
    when = "//!WHEN OUTPUT.w LUMA.w / 1.300 > OUTPUT.h LUMA.h / 1.300 > *"
    out = [header, ""]

    def block(desc, binds, save, body):
        out.append("//!HOOK LUMA")
        out.append(when)
        out.append("//!DESC " + desc)
        for b in binds:
            out.append("//!BIND " + b)
        if save:
            out.append("//!SAVE " + save)
            out.append("//!COMPONENTS 4")
        out.append("vec4 hook()")
        out.append("{")
        out.extend(body)
        out.append("}")
        out.append("")

    for half in range(2):
        sl = slice(half * 4, half * 4 + 4)
        body = ["vec4 res = %s;" % _vec4(P["feat_b"][sl])]
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                body.append("res += %s * float(LUMA_texOff(vec2(%d,%d)));" % (
                    _vec4(P["feat_w"][dy + 2, dx + 2, 0, sl]), dx, dy))
        body.append("return res;")
        block("feature map %d" % (half + 1), ["LUMA"], "FEATURE%d" % (half + 1), body)

    def conv8_blocks(prefix, w, b, a, ins, saves, k):
        r = k // 2
        for half in range(2):
            sl = slice(half * 4, half * 4 + 4)
            body = ["vec4 res = %s;" % _vec4(b[sl])]
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for in_half, tex in enumerate(ins):
                        W = w[dy + r, dx + r, in_half * 4:in_half * 4 + 4, sl].T  # [out, in]
                        off = "vec2(%d,%d)" % (dx, dy) if k > 1 else "0"
                        body.append("res += %s * %s_texOff(%s);" % (_mat4(W), tex, off))
            yield half, body, sl

    ins = ["FEATURE1", "FEATURE2"]
    names = {1: ("MODEL21", "MODEL22"), 2: ("MODEL1", "MODEL2"), 3: ("MODEL21", "MODEL22"), 4: ("MODEL1", "MODEL2")}
    for layer in range(1, 5):
        w, b, a = P["map%d_w" % layer], P["map%d_b" % layer], P["map%d_a" % layer]
        for half, body, sl in conv8_blocks("", w, b, a, ins, names[layer], 3):
            body.append("res = max(res, vec4(0.0)) + %s * min(res, vec4(0.0));" % _vec4(a[sl]))
            body.append("return res;")
            block("mapping %d_%d" % (layer, half + 1), ins, names[layer][half], body)
        ins = list(names[layer])

    for half, body, sl in conv8_blocks("", P["res_w"], P["res_b"], P["res_a"], ins, None, 1):
        body.append("res += FEATURE%d_texOff(0);" % (half + 1))
        body.append("res = max(res, vec4(0.0)) + %s * min(res, vec4(0.0));" % _vec4(P["res_a"][sl]))
        body.append("return res;")
        block("sub-band residuals %d" % (half + 1), ins + ["FEATURE%d" % (half + 1)], "RES%d" % (half + 1), body)

    body = ["vec4 res = %s;" % _vec4(P["sub_b"])]
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for in_half, tex in enumerate(["RES1", "RES2"]):
                W = P["sub_w"][dy + 1, dx + 1, in_half * 4:in_half * 4 + 4, :].T
                body.append("res += %s * %s_texOff(vec2(%d,%d));" % (_mat4(W), tex, dx, dy))
    body.append("return vec4(res);")
    block("sub-pixel convolution 1", ["RES1", "RES2"], "SUBCONV1", body)

    out.extend([
        "//!HOOK LUMA", when, "//!WIDTH LUMA.w 2 *", "//!HEIGHT LUMA.h 2 *", "//!DESC aggregation", "//!BIND SUBCONV1",
        "vec4 hook()", "{",
        "vec2 fcoord = fract(SUBCONV1_pos * SUBCONV1_size);",
        "vec2 base = SUBCONV1_pos + (vec2(0.5) - fcoord) * SUBCONV1_pt;",
        "ivec2 index = ivec2(fcoord * vec2(2));",
        "vec4 res = SUBCONV1_tex(base);",
        "return vec4(res[index.x * 2 + index.y], 0, 0, 1);",
        "}", ""])
    return "\n".join(out)
