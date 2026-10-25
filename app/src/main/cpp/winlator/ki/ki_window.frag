#version 450
// KI upscaler output for the game window (Vulkan renderer). Same push constants as window.frag.
layout(set = 0, binding = 0) uniform sampler2D texSampler;   // game window, linear sampler
layout(set = 1, binding = 0) uniform sampler2D subPixels;    // network output, 4 sub-pixels / texel
layout(push_constant) uniform PC {
    float ndcX0; float ndcY0; float ndcX1; float ndcY1;
    int useTexAlpha; int effectId; float sharpness; float resW; float resH;
    int effectMask; float brightness; float contrast; float gamma; float outW; float outH;
} pc;
layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
const vec3 LUMA_W = vec3(0.2126, 0.7152, 0.0722);

float hiLuma(ivec2 q, ivec2 inSize) {
    q = clamp(q, ivec2(0), inSize * 2 - 1);
    ivec2 texel = q / 2;
    ivec2 sub = q - texel * 2;
    int idx = sub.x * 2 + sub.y;          // mpv: res[index.x * 2 + index.y], y down
    vec4 s = texelFetch(subPixels, texel, 0);
    return idx == 0 ? s.x : (idx == 1 ? s.y : (idx == 2 ? s.z : s.w));
}

vec3 colorAdjust(vec3 c) {
    c = c + pc.brightness;
    c = (c - 0.5) * (1.0 + pc.contrast) + 0.5;
    c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / max(pc.gamma, 0.01)));
    return c;
}

void main() {
    vec2 uv = fragTexCoord;
    vec3 base = texture(texSampler, uv).rgb;
    ivec2 inSize = ivec2(max(vec2(pc.resW, pc.resH), vec2(1.0)));
    vec2 p = uv * vec2(inSize) * 2.0 - 0.5;
    vec2 i0 = floor(p);
    vec2 f = p - i0;
    ivec2 q = ivec2(i0);
    float y00 = hiLuma(q, inSize);
    float y10 = hiLuma(q + ivec2(1, 0), inSize);
    float y01 = hiLuma(q + ivec2(0, 1), inSize);
    float y11 = hiLuma(q + ivec2(1, 1), inSize);
    float yHi = mix(mix(y00, y10, f.x), mix(y01, y11, f.x), f.y);
    // sharpness (0..1) carries the detail strength / 1.5 (so 0.667 = network as trained)
    vec3 rgb = clamp(base + (yHi - dot(base, LUMA_W)) * (pc.sharpness * 1.5), 0.0, 1.0);
    outColor = vec4(colorAdjust(rgb), 1.0);
}
