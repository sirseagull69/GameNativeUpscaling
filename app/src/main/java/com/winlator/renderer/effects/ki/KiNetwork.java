package com.winlator.renderer.effects.ki;

/**
 * A small convolutional upscaling network expressed as a list of GPU passes
 * (converted from an mpv user shader by ki/tools/convert_mpv_shader.py).
 *
 * Every pass runs at the input resolution and writes 4 channels into the texture
 * named by {@link #passSave(int)}. "LUMA" is the input image's luma. The last
 * pass ({@link #outputTexture()}) holds the 4 sub-pixels of a 2x upscale per texel.
 */
public interface KiNetwork {
    int passCount();
    String passName(int index);
    String[] passBinds(int index);
    String passSave(int index);
    /** GLSL body defining {@code vec4 hookPass()} using NAME_texOff(vec2 offset). */
    String passBody(int index);
    String outputTexture();
}
