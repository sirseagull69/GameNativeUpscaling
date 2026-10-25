package com.winlator.renderer.effects;

import com.winlator.renderer.GLRenderer;

/**
 * An effect that needs to run its own GPU passes (into its own framebuffers)
 * before its material draws into the composer's target. The composer calls
 * {@link #runPrePasses} first and re-binds its target framebuffer afterwards.
 */
public interface PrePassEffect {
    void runPrePasses(GLRenderer renderer, int sourceTextureId, int sourceWidth, int sourceHeight, int outputWidth, int outputHeight);
}
