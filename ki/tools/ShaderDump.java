import com.winlator.renderer.effects.ki.KiNetwork;
import com.winlator.renderer.effects.ki.KiShaders;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Writes the exact GLSL that KiUpscaleEffect compiles on the device, so it can be
 * validated with glslang and executed with a desktop GLES implementation (ki/tools/run_ki.py).
 * Usage: ShaderDump <out dir> [NetworkClass]
 *
 * Output layout (in the given directory):
 *   vertex.vert            vertex shader incl. the applyXForm injection ShaderMaterial adds
 *   pass_NN.frag           network pass NN
 *   pass_NN.meta           "save:NAME" and "bind:NAME" lines for pass NN
 *   output.frag            the screen-resolution output pass
 */
public class ShaderDump {
    public static void main(String[] args) throws IOException {
        Path dir = Paths.get(args[0]);
        Files.createDirectories(dir);
        // Network class name in com.winlator.renderer.effects.ki (default: the trained game net).
        String cls = args.length > 1 ? args[1] : "KiGameX2";
        KiNetwork net;
        try {
            net = (KiNetwork) Class.forName("com.winlator.renderer.effects.ki." + cls).getDeclaredConstructor().newInstance();
        } catch (ReflectiveOperationException e) {
            throw new IOException("unknown network " + cls, e);
        }

        // Mirror ShaderMaterial.compileShaders' injection.
        String vs = KiShaders.VERTEX_SHADER;
        int begin = vs.indexOf("void main() {");
        vs = vs.substring(0, begin) +
            "vec2 applyXForm(vec2 p, float xform[6]) {\n" +
            "return vec2(xform[0] * p.x + xform[2] * p.y + xform[4], xform[1] * p.x + xform[3] * p.y + xform[5]);\n" +
            "}\n" +
            vs.substring(begin);
        write(dir.resolve("vertex.vert"), vs);

        for (int i = 0; i < net.passCount(); i++) {
            String name = String.format("pass_%02d", i);
            write(dir.resolve(name + ".frag"), KiShaders.passFragmentShader(net.passBinds(i), net.passBody(i)));
            StringBuilder meta = new StringBuilder("save:" + net.passSave(i) + "\n");
            for (String b : net.passBinds(i)) meta.append("bind:").append(b).append("\n");
            write(dir.resolve(name + ".meta"), meta.toString());
        }
        write(dir.resolve("output.frag"), KiShaders.OUTPUT_FRAGMENT_SHADER);
        write(dir.resolve("network.txt"), "output:" + net.outputTexture() + "\npasses:" + net.passCount() + "\n");
        System.out.println("dumped " + net.passCount() + " passes to " + dir);
    }

    private static void write(Path p, String s) throws IOException {
        Files.write(p, s.getBytes(StandardCharsets.UTF_8));
    }
}
