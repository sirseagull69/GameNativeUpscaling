# KI-Upscaler (Testbuild)

Upscaler per neuronalem Netz für GameNative, gebaut für den AYN Thor (Snapdragon 8 Gen 2).
Läuft im **Vulkan-Renderer** (Compute-Shader) und im GL-Renderer.

## Netz

- Struktur FSRCNNX 8-0-4-1 (igv, LGPL-3.0), 2x, nur Helligkeit; Farbe kommt bilinear dazu.
- **KI_game**: von FSRCNNX aus weitertrainiert auf synthetische Spielbilder
  (`ki/train/`): gefilterte Texturen + harte Geometrie-/Schriftkanten wie beim Rendern
  ohne Kantenglättung, Ziel = sauber geglättetes Bild in 2x.
- Gewichte: `ki/shaders/KI_game_x2_8-0-4-1.glsl` (mpv-Format).
- Aufwand: ca. 5,7 kFLOP pro Eingabepixel, bei 960x540 rund 3 GFLOP pro Bild.

## Messung (Y-PSNR in dB, Mesa, gleiche Shader wie in der App)

| | Bilinear | Bicubic | SGSR v1 | FSRCNNX | **KI_game** |
|---|---|---|---|---|---|
| synthetische Spielbilder (12) | 35.60 | 36.41 | 34.23 | 34.61 | **37.26** |
| Set5+14 hart verkleinert | 26.79 | 26.57 | 25.17 | 25.04 | 26.44 |
| Set5+14 geglättet (Lanczos) | 29.06 | 30.39 | 28.82 | 31.93 | 31.48 |

## Werkzeuge

- `ki/train/model.py` – Netz in JAX, verlustfreier Import/Export im mpv-Format
- `ki/train/data.py` – synthetische Spielbild-Paare
- `ki/train/train.py` – Training (CPU)
- `ki/tools/convert_mpv_shader.py` – mpv → Java (GL-Renderer)
- `ki/tools/gen_vulkan.py` – mpv → Vulkan-Compute-Shader + SPIR-V (`app/src/main/cpp/winlator/ki/`)
- `ki/tools/compare.py` – Vergleich Bilinear/Bicubic/SGSR/Netze auf Mesa
- `ki/tools/test_vulkan_shaders.py` – Vulkan-Shader über GLES 3.1 auf Mesa ausführen

## Nutzung

Container: Anzeigeserver **X11**, Display-Renderer **Vulkan**, Bildschirmgröße z. B. 960×540.
Im Spiel: Bildschirmeffekte → Hochskalierung → **KI-Upscaler**, KI-Detailstärke nach Geschmack.
