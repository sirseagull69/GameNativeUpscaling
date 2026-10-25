# KI-Upscaler (Testbuild)

Experimenteller Upscaler per neuronalem Netz für GameNative, gebaut für den AYN Thor.

- Netz: FSRCNNX x2 8-0-4-1 von igv (LGPL-3.0), `ki/shaders/`
- Konverter mpv-Shader → Java: `ki/tools/convert_mpv_shader.py`
- Effekt: `app/src/main/java/com/winlator/renderer/effects/ki/`
- Off-Device-Test (echte Shader auf Mesa/llvmpipe, PSNR gegen Original): `ki/tools/ShaderDump.java` + `ki/tools/run_ki.py`

Messung (Set5+Set14, 2x, Y-PSNR):

| Eingabe | Bilinear | Bicubic | KI |
|---|---|---|---|
| geglättet verkleinert (Lanczos) | 29.05 dB | 30.39 dB | **31.93 dB** |
| hart verkleinert (wie Spiel ohne AA) | 26.79 dB | 26.57 dB | 25.04 dB |

Folgerung: Bei Spielen mit Kantenglättung hilft das Netz deutlich. Bei Spielen ohne
Kantenglättung KI-Detailstärke runterdrehen. Nächster Schritt: eigenes Netz auf
spieltypische (treppige) Eingaben trainieren.

Nutzung: Container → Display renderer **GL**, Screen size z. B. 960×540,
im Spiel Bildschirmeffekte → Hochskalierung → **KI (FSRCNNX)**.
