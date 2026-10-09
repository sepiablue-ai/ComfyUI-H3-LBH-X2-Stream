# ComfyUI H3 LBH X2 Stream

[日本語](README_ja.md) | English

Generate three MiniMax H3 steps at **384×672**, upscale the clean video latent once with LBH, finish the fourth H3 step at **544×960**, then use X2 Stream to save **1088×1920** video with audio. Audio continues through all four original steps; it is not frozen at the resolution change.

This packages the selected LBH **3+1** configuration built on the author's 022 workflow. The tested 4+0 configuration was rejected for visual and audio quality and is not an included preset.

## Included

| Component | Configuration |
|---|---|
| Native F12 | W4A4 only for FC1 blocks 19, 22–29, 47–49; all four steps |
| Bounded QKV reuse | Up to 336 MiB, local to one attention call, released afterward |
| Native SLA | 5%, min_tokens 4096, extra_tokens 256, exact audio/conditioning rows |
| FFN | One chunk up to 8192 rows; eight above; threshold 4096 |
| Shared image conditioning | One high-image TE encode and two upfront VAE keyframes |
| LBH | Pinned 3D FP16 implementation, one spatial upscale, full temporal context |
| X2 / save | [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream), INT8 X2 VAE and asynchronous NVENC |

FC2/QKV/OutProj weights remain INT8. No VSA Gate, benchmark environment variables, forward-count limits, model downloads or ComfyUI core edits. The remaining original sigma schedule is retained; there is no RALU sigma correction.

## Requirements and setup

Validated development environment: Windows, RTX 4070 12GB, ComfyUI 0.38.0, Python 3.13.13, PyTorch 2.14.0+cu130, comfy-kitchen 0.2.36, approximately 48 GiB system RAM. Other hardware, operating systems and models require their own validation.

Native H3, SLA, Dynamic VRAM, ConvRot W4A4 and Sol-Attn are required. CUDA/PyTorch come from ComfyUI. Place this repo in custom_nodes or run:

```bat
setup.bat "C:\ComfyUI\ComfyUI" "C:\ComfyUI\.venv\Scripts\python.exe"
```

Use the Python belonging to that ComfyUI. Portable Python and venv candidates can be detected when the second argument is omitted, but ambiguous candidates require an explicit path. Global Python is refused. Existing different files are preserved.

`--install-dependencies` installs a missing X2 Stream at commit `eb148a3d0f91a273668bbe027bfcbafb85f2c5e8` and requirements into the chosen Python with pip `--no-deps`. It does not update an existing X2 Stream or replace PyTorch. `--check` performs read-only checks; add `--gpu-check` for a small native W4A4 offload/reload test. `--prepare-f12` prepares twelve local shards once. Stop ComfyUI before preparation. Setup does not start a server.

The default F12 cache is models/h3_lbh_x2/f12, approximately 0.93 GB. The F12 Loader can prepare it on first use. Existing native 50-block FC1 caches can be selected with cache_directory or setup `--cache`; only twelve shards are used, and Gate is ignored. An incomplete or incompatible cache is never overwritten.

## Models and generation

Models are not bundled or automatically downloaded. Download the exact MATLOW fused INT8 model, Qwen3-VL NVFP4 text encoder, ordinary H3 INT8 video VAE, H3 FP32 audio VAE, original X2 Detail v1 VAE, and LBH 3D FP16 upscaler. [The Japanese model table](README_ja.md#モデル) lists filenames, locations and primary download sources. Shared paths use native extra_model_paths.yaml; include latent_upscale_models for LBH. [Example](extra_model_paths.yaml.example).

Prepare the X2 INT8 decoder once with [00_prepare_x2.json](examples/00_prepare_x2.json), then load [10s](examples/i2va_lbh_3p1_10s.json), [14s](examples/i2va_lbh_3p1_14s.json), or [5s](examples/i2va_lbh_3p1_05s.json). [API graphs](examples/api) are also included. Setup copies the three example images into input. Change the image, prompt and model selections in ComfyUI, then queue normally.

These are **first-image I2VA** examples using an FL2VA-derived model; no final-frame conditioning is included. Seed 43, res_multistep/simple4, video/audio shift 12/3, 24 fps, CQ18, FP16 accumulation. Frame counts are 124 / 243 / 345. Output is saved under output/H3_LBH_X2_Stream.

Keep the low sampler output1 connected to LBH and output0 connected through native Separate/Concat AV to continue audio. LBH is followed by the final high-resolution H3 step, which updates both video and audio.

## Measurements and validation

| Duration preset | Existing 022 | Measured LBH 3+1 |
|---|---:|---:|
| 10s | 106.879 s | 85.393 s |
| 14s | 154.834 s | 116.594 s |

These are historical development-workflow measurements, one run each, with initial loading, conditioning, sampling, LBH, audio, X2 and saving included. Server startup, uploads, media checks and one-time conversion are excluded. Run-day variation was not isolated. The historical experiment did not generate 5s 3+1; the packaged 5s preset subsequently passed fresh-install generation.

The distributed loader removes benchmark audits, accepts portable cache paths and includes preparation. Image resizing uses the same PIL fit-crop method. All three packaged video presets subsequently completed on fresh ComfyUI 0.39.0. [benchmark.json](docs/benchmark.json) and [validation.json](docs/validation.json) separate historical generation results from package checks.

The three samples are lossless PNGs decoded from the original WebP files with PIL. This avoids small differences from ComfyUI's WebP decoder; all six resized condition images match the measured PNG inputs exactly. The current LBH download filename with `3d_conv_v1` is also accepted, but its identity with the measured legacy weights has not been verified. The measured weight hash is recorded in benchmark.json.

Package checks passed: six registered nodes, four GUI/API workflow pairs with native input validation and identical widgets/connections, setup.bat environment and GPU checks, all twelve newly converted F12 shards matching the measured tensors, and an actual 50-block model load. Four FP32/BF16 handoff cases match the preserved calculation exactly. QKV and FFN computation bodies are unchanged. These initial component checks were followed by the full fresh-install examples below.

Seven standard-library tests cover QKV budget/input/release guards and repeatable installation with conflict preservation. Run from the repository:

```bat
"C:\ComfyUI\.venv\Scripts\python.exe" -B -m unittest discover -s tests -v
```

The existing master ComfyUI 0.38.0 also passed setup and a 5s example generation in 57.506 s with full AV decode. No dependencies or the 022 baseline were changed. Three GUI presets were registered locally. [Master validation](docs/master_validation.json).

## Fresh-install generation

A fresh ComfyUI 0.39.0 clone, separate venv and setup.bat completed the X2 preparation example and all three unchanged video presets. Each video passed full video/audio decode at 1088×1920, 24 fps, AAC 32 kHz stereo.

| Preset | Execution | Model state |
|---|---:|---|
| 5s | 58.802 s | First video in a new process |
| 10s | 86.289 s | First video in a new process |
| 14s | 112.983 s | Same process after a different 10s image/prompt |

Timing excludes one-time F12/X2 preparation, server startup and output checks. The new venv references existing standard-runtime packages; only the five newly required package versions were installed locally. This was not an empty-dependency installation, and the standard venv was not changed. GUI data matched the API graphs and was submitted through the API; browser drag/drop and Queue button were not tested. Subjective quality review of these outputs is pending. [Evidence](docs/fresh_install_validation.json).

Code: GPL-3.0, with the vendored LBH MIT notice retained. Model and derived-weight terms remain separate. See [NOTICE.md](NOTICE.md).
