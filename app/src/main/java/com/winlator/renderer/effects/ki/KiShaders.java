package com.winlator.renderer.effects.ki;

/**
 * GLSL ES 3.00 sources for {@link KiUpscaleEffect}. Kept free of Android classes so
 * ki/tools can compile and validate exactly these shaders off-device.
 */
public final class KiShaders {
    private KiShaders() {}

    public static final String VERTEX_SHADER =
        "#version 300 es\n" +
        "layout(location = 0) in vec2 position;\n" +
        "out vec2 vUV;\n" +
        "void main() {\n" +
        "vUV = position;\n" +
        "gl_Position = vec4(2.0 * position.x - 1.0, 2.0 * position.y - 1.0, 0.0, 1.0);\n" +
        "}";
    /**
     * One network pass. All reads are exact texel fetches at the input resolution.
     * Offsets are in image space (y down, like mpv), so the network sees the image
     * the right way up even though GL framebuffers are stored bottom-up.
     */
    public static String passFragmentShader(String[] binds, String body) {
        StringBuilder sb = new StringBuilder();
        sb.append("#version 300 es\n");
        sb.append("precision highp float;\n");
        sb.append("precision highp int;\n");
        sb.append("uniform ivec2 inputSize;\n");
        sb.append("out vec4 fragColor;\n");
        sb.append("ivec2 kiCoord(vec2 off) {\n");
        sb.append("    ivec2 p = ivec2(gl_FragCoord.xy) + ivec2(int(off.x), -int(off.y));\n");
        sb.append("    return clamp(p, ivec2(0), inputSize - 1);\n");
        sb.append("}\n");
        for (String bind : binds) {
            sb.append("uniform sampler2D ").append(bind).append(";\n");
            if ("LUMA".equals(bind)) {
                sb.append("vec4 LUMA_texOff(vec2 off) {\n");
                sb.append("    vec3 c = texelFetch(LUMA, kiCoord(off), 0).rgb;\n");
                sb.append("    return vec4(dot(c, vec3(0.2126, 0.7152, 0.0722)));\n");
                sb.append("}\n");
            }
            else {
                sb.append("vec4 ").append(bind).append("_texOff(vec2 off) { return texelFetch(")
                    .append(bind).append(", kiCoord(off), 0); }\n");
            }
        }
        sb.append(body).append("\n");
        sb.append("void main() { fragColor = hookPass(); }\n");
        return sb.toString();
    }

    /**
     * Output pass at screen resolution: resample the network's 2x luma to the screen
     * and add its detail on top of the bilinear colour upscale.
     */
    public static final String OUTPUT_FRAGMENT_SHADER =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "uniform sampler2D screenTexture;\n" +
            "uniform sampler2D subPixels;\n" +
            "uniform vec2 inputResolution;\n" +
            "uniform vec2 outputResolution;\n" +
            "uniform float networkActive;\n" +
            "uniform float preserveAspect;\n" +
            "uniform float detailStrength;\n" +
            "in vec2 vUV;\n" +
            "out vec4 fragColor;\n" +
            "const vec3 LUMA_W = vec3(0.2126, 0.7152, 0.0722);\n" +
            // Luma of the 2x image at integer 2x-grid coordinate q (GL orientation).
            "float hiLuma(ivec2 q) {\n" +
            "    ivec2 hiSize = ivec2(inputResolution) * 2;\n" +
            "    q = clamp(q, ivec2(0), hiSize - 1);\n" +
            "    ivec2 texel = q / 2;\n" +
            "    ivec2 sub = q - texel * 2;\n" +
            // mpv picks res[index.x * 2 + index.y] with index.y counted downwards.
            "    int idx = sub.x * 2 + (1 - sub.y);\n" +
            "    vec4 s = texelFetch(subPixels, texel, 0);\n" +
            "    return idx == 0 ? s.x : (idx == 1 ? s.y : (idx == 2 ? s.z : s.w));\n" +
            "}\n" +
            "void main() {\n" +
            "    vec2 frag = gl_FragCoord.xy;\n" +
            "    vec2 area = outputResolution;\n" +
            "    vec2 offset = vec2(0.0);\n" +
            "    if (preserveAspect > 0.5) {\n" +
            "        float inAspect = inputResolution.x / inputResolution.y;\n" +
            "        float outAspect = outputResolution.x / outputResolution.y;\n" +
            "        if (outAspect > inAspect) area.x = outputResolution.y * inAspect;\n" +
            "        else area.y = outputResolution.x / inAspect;\n" +
            "        offset = 0.5 * (outputResolution - area);\n" +
            "    }\n" +
            "    vec2 local = frag - offset;\n" +
            "    if (local.x < 0.0 || local.y < 0.0 || local.x > area.x || local.y > area.y) {\n" +
            "        fragColor = vec4(0.0, 0.0, 0.0, 1.0);\n" +
            "        return;\n" +
            "    }\n" +
            "    vec2 uv = local / area;\n" +
            "    vec4 base = texture(screenTexture, uv);\n" +
            "    if (networkActive < 0.5) {\n" +
            "        fragColor = vec4(base.rgb, 1.0);\n" +
            "        return;\n" +
            "    }\n" +
            "    vec2 p = uv * inputResolution * 2.0 - 0.5;\n" +
            "    vec2 i0 = floor(p);\n" +
            "    vec2 f = p - i0;\n" +
            "    ivec2 q = ivec2(i0);\n" +
            "    float y00 = hiLuma(q);\n" +
            "    float y10 = hiLuma(q + ivec2(1, 0));\n" +
            "    float y01 = hiLuma(q + ivec2(0, 1));\n" +
            "    float y11 = hiLuma(q + ivec2(1, 1));\n" +
            "    float yHi = mix(mix(y00, y10, f.x), mix(y01, y11, f.x), f.y);\n" +
            "    float yBase = dot(base.rgb, LUMA_W);\n" +
            "    vec3 color = base.rgb + (yHi - yBase) * detailStrength;\n" +
            "    fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);\n" +
            "}";
}
