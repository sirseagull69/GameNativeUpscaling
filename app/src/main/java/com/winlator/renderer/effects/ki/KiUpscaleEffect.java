package com.winlator.renderer.effects.ki;

import android.opengl.GLES20;
import android.opengl.GLES30;
import android.util.Log;

import com.winlator.renderer.GLRenderer;
import com.winlator.renderer.VertexAttribute;
import com.winlator.renderer.effects.Effect;
import com.winlator.renderer.effects.PrePassEffect;
import com.winlator.renderer.effects.RenderScaleEffect;
import com.winlator.renderer.effects.SourceTextureFilterEffect;
import com.winlator.renderer.material.ShaderMaterial;

import java.util.HashMap;
import java.util.Map;

/**
 * "KI" upscaling: runs a small convolutional neural network (FSRCNNX) on the GPU.
 *
 * Pipeline per frame:
 * 1. The game image (X server resolution) comes in as the composer's scene texture.
 * 2. The network's passes run at that input resolution on its luma, writing
 *    half-float RGBA textures. The last one holds the 4 sub-pixels of a 2x luma image.
 * 3. The output pass (this effect's material, drawn by the composer at screen
 *    resolution) resamples that 2x luma to the screen size and adds the luma detail
 *    on top of a bilinear upscale of the colour image.
 *
 * If half-float framebuffers are unavailable, or the screen is not meaningfully larger
 * than the game image, the output pass falls back to a plain bilinear upscale.
 */
public class KiUpscaleEffect extends Effect implements RenderScaleEffect, PrePassEffect, SourceTextureFilterEffect {
    private static final String TAG = "KiUpscaleEffect";
    /** Same rule as the original mpv shader: only run when upscaling by more than 1.3x. */
    private static final float MIN_SCALE = 1.3f;

    private final KiNetwork network;
    private final Map<String, FloatTarget> targets = new HashMap<>();
    private PassMaterial[] passMaterials;
    private boolean unsupported = false;
    private boolean networkRanThisFrame = false;
    private boolean preserveAspect = true;
    private float detailStrength = 1.0f;

    public KiUpscaleEffect() {
        this(new FsrcnnxX2Small());
    }

    public KiUpscaleEffect(KiNetwork network) {
        this.network = network;
    }

    public void setPreserveAspect(boolean preserveAspect) {
        this.preserveAspect = preserveAspect;
    }

    /** 0 = plain bilinear, 1 = full network detail, >1 = exaggerated. */
    public void setDetailStrength(float detailStrength) {
        this.detailStrength = detailStrength;
    }

    @Override
    public int getRenderWidth(GLRenderer renderer, int outputWidth) {
        return Math.max(1, Math.min(outputWidth, renderer.getXServerWidth()));
    }

    @Override
    public int getRenderHeight(GLRenderer renderer, int outputHeight) {
        return Math.max(1, Math.min(outputHeight, renderer.getXServerHeight()));
    }

    @Override
    public int getSourceMinFilter() {
        return GLES20.GL_LINEAR;
    }

    @Override
    public int getSourceMagFilter() {
        return GLES20.GL_LINEAR;
    }

    @Override
    protected ShaderMaterial createMaterial() {
        return new OutputMaterial();
    }

    @Override
    public void runPrePasses(GLRenderer renderer, int sourceTextureId, int sourceWidth, int sourceHeight, int outputWidth, int outputHeight) {
        networkRanThisFrame = false;
        if (unsupported || sourceWidth <= 0 || sourceHeight <= 0) return;

        float scaleX = (float) outputWidth / sourceWidth;
        float scaleY = (float) outputHeight / sourceHeight;
        if (scaleX <= MIN_SCALE && scaleY <= MIN_SCALE) return;

        try {
            if (passMaterials == null) {
                passMaterials = new PassMaterial[network.passCount()];
                for (int i = 0; i < passMaterials.length; i++) {
                    passMaterials[i] = new PassMaterial(network.passBinds(i), network.passBody(i));
                }
            }

            VertexAttribute quad = renderer.getQuadVertices();
            GLES20.glDisable(GLES20.GL_BLEND);

            for (int i = 0; i < passMaterials.length; i++) {
                FloatTarget out = target(network.passSave(i));
                out.allocate(sourceWidth, sourceHeight);

                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, out.framebuffer);
                GLES20.glViewport(0, 0, sourceWidth, sourceHeight);

                PassMaterial material = passMaterials[i];
                material.use();
                material.setUniformIVec2("inputSize", sourceWidth, sourceHeight);

                String[] binds = network.passBinds(i);
                for (int b = 0; b < binds.length; b++) {
                    int textureId = "LUMA".equals(binds[b]) ? sourceTextureId : targets.get(binds[b]).textureId;
                    GLES20.glActiveTexture(GLES20.GL_TEXTURE0 + b);
                    GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
                    material.setUniformInt(binds[b], b);
                }

                quad.bind(material.programId);
                GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, quad.count());
                quad.disable();

                for (int b = binds.length - 1; b >= 0; b--) {
                    GLES20.glActiveTexture(GLES20.GL_TEXTURE0 + b);
                    GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
                }
            }
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            networkRanThisFrame = true;
        }
        catch (RuntimeException e) {
            Log.e(TAG, "KI upscaler disabled, falling back to bilinear", e);
            unsupported = true;
            releaseNetwork();
        }
        finally {
            // The renderer keeps alpha blending on globally; the network passes need it off
            // because their alpha channel carries a feature, not opacity.
            GLES20.glEnable(GLES20.GL_BLEND);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        }
    }

    @Override
    protected void onUse(ShaderMaterial material, GLRenderer renderer) {
        material.setUniformFloat("networkActive", networkRanThisFrame ? 1.0f : 0.0f);
        material.setUniformFloat("preserveAspect", preserveAspect ? 1.0f : 0.0f);
        material.setUniformFloat("detailStrength", detailStrength);
        material.setUniformInt("subPixels", 1);
        FloatTarget out = targets.get(network.outputTexture());
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, networkRanThisFrame && out != null ? out.textureId : 0);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
    }

    private FloatTarget target(String name) {
        FloatTarget t = targets.get(name);
        if (t == null) {
            t = new FloatTarget();
            targets.put(name, t);
        }
        return t;
    }

    private void releaseNetwork() {
        if (passMaterials != null) {
            for (PassMaterial m : passMaterials) {
                if (m != null) m.destroy();
            }
            passMaterials = null;
        }
        for (FloatTarget t : targets.values()) t.destroy();
        targets.clear();
    }

    @Override
    public void destroy() {
        super.destroy();
        releaseNetwork();
        networkRanThisFrame = false;
    }

    /** Half-float RGBA framebuffer; the network's intermediate values are signed. */
    private static final class FloatTarget {
        int framebuffer = 0;
        int textureId = 0;
        int width = 0;
        int height = 0;

        void allocate(int w, int h) {
            if (framebuffer != 0 && width == w && height == h) return;
            destroy();
            width = w;
            height = h;

            int[] ids = new int[1];
            GLES20.glGenTextures(1, ids, 0);
            textureId = ids[0];
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
            GLES30.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES30.GL_RGBA16F, w, h, 0, GLES20.GL_RGBA, GLES30.GL_HALF_FLOAT, null);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

            GLES20.glGenFramebuffers(1, ids, 0);
            framebuffer = ids[0];
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, framebuffer);
            GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, textureId, 0);
            int status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
            if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
                destroy();
                throw new IllegalStateException("Half-float framebuffer incomplete: 0x" + Integer.toHexString(status));
            }
        }

        void destroy() {
            if (textureId != 0) GLES20.glDeleteTextures(1, new int[]{textureId}, 0);
            if (framebuffer != 0) GLES20.glDeleteFramebuffers(1, new int[]{framebuffer}, 0);
            textureId = 0;
            framebuffer = 0;
            width = 0;
            height = 0;
        }
    }


    /**
     * One network pass. All reads are exact texel fetches at the input resolution.
     * Offsets are in image space (y down, like mpv), so the network sees the image
     * the right way up even though GL framebuffers are stored bottom-up.
     */
    private static final class PassMaterial extends ShaderMaterial {
        private final String fragmentShader;
        private final HashMap<String, Integer> extraLocations = new HashMap<>();

        PassMaterial(String[] binds, String body) {
            fragmentShader = KiShaders.passFragmentShader(binds, body);

            String[] names = new String[binds.length];
            System.arraycopy(binds, 0, names, 0, binds.length);
            setUniformNames(names);
        }

        @Override
        protected String getVertexShader() {
            return KiShaders.VERTEX_SHADER;
        }

        @Override
        protected String getFragmentShader() {
            return fragmentShader;
        }

        private int checkedProgram = 0;

        @Override
        public void use() {
            super.use();
            if (checkedProgram != programId) {
                checkLinked(programId);
                checkedProgram = programId;
            }
        }

        void setUniformIVec2(String name, int x, int y) {
            Integer location = extraLocations.get(name);
            if (location == null) {
                location = GLES20.glGetUniformLocation(programId, name);
                extraLocations.put(name, location);
            }
            if (location != -1) GLES20.glUniform2i(location, x, y);
        }

        @Override
        public void destroy() {
            super.destroy();
            extraLocations.clear();
        }
    }

    /**
     * Output pass at screen resolution: resample the network's 2x luma to the screen
     * and add its detail on top of the bilinear colour upscale.
     */
    private static final class OutputMaterial extends ShaderMaterial {
        OutputMaterial() {
            setUniformNames("screenTexture", "subPixels", "inputResolution", "outputResolution",
                "networkActive", "preserveAspect", "detailStrength");
        }

        @Override
        protected String getVertexShader() {
            return KiShaders.VERTEX_SHADER;
        }

        @Override
        protected String getFragmentShader() {
            return KiShaders.OUTPUT_FRAGMENT_SHADER;
        }

        private int checkedProgram = 0;

        @Override
        public void use() {
            super.use();
            if (checkedProgram != programId) {
                checkLinked(programId);
                checkedProgram = programId;
            }
        }
    }

    private static void checkLinked(int programId) {
        int[] linked = new int[1];
        GLES20.glGetProgramiv(programId, GLES20.GL_LINK_STATUS, linked, 0);
        if (linked[0] == 0) {
            throw new RuntimeException("KI shader link failed: " + GLES20.glGetProgramInfoLog(programId));
        }
    }
}
