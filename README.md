# BlazeFace (MediaPipe face detector) on the Radxa Cubie A7A NPU

MediaPipe's `blaze_face_short_range` face detector converted to an A733 `.nb`. It was my first model on this NPU, mostly to prove that the Acuity -> `.nb` -> board path works for a MediaPipe network at all. It does, and it is fast: about 1 ms.

This repo is the converted network plus the conversion scripts. **I did not write the post-processing** (anchor decoding, NMS), so you get raw outputs, not boxes. The palm detector in my [MediaPipe hands repo](https://github.com/Aseel1206/radxa-cubie-a7a-npu-mediapipe-hands) uses the same kind of SSD anchors and has working decode code you can adapt.

## Results

| Build | File | Time on NPU | Result on my test crop |
|---|---|---|---|
| int16 | `models/blazeface_short_int16.nb` (242 KB) | **1.11 ms** | finds the face: best score 0.703, 4 anchors above 0.5 (float: 0.72, 5 anchors) |
| int8 per-channel | `models/blazeface_short_int8pcq.nb` (168 KB) | 0.52 ms | **misses the face** (best score 0.500, 0 anchors above 0.5) |

Use the int16 one. The simulator predicted this beforehand (int16 score 0.722 with logit correlation 1.000, int8 logit correlation 0.955), and the board agreed.

Honest limits: this is **one** face crop, nothing more. I didn't test several images or run a CPU baseline for this model, and the board's int16 result is close to but not identical to the simulator (0.703 vs 0.722).

Model: input `1x128x128x3` float, normalised as `(x - 127.5) / 127.5`; outputs `regressors (1,896,16)` and `classificators (1,896,1)` (896 = 16x16x2 + 8x8x6 anchors).

## What I used

- **Board:** Radxa Cubie A7A (Allwinner A733, 8x Cortex-A55, VeriSilicon Vivante NPU), Debian 11, kernel 5.15.147-a733. The NPU is driven through VIPLite with the `awnn_inference` wrapper that was already installed on my board (`~/awnn_inference`).
- **Converter:** the NPU toolchain docker image `ubuntu-npu:v2.0.10.1` (Acuity 6.30.22 + Vivante IDE 5.11), for the Allwinner A733 NPU. I didn't build it and I'm not redistributing it, so check Radxa's NPU documentation for where to get it. Everything below that says "in the container" runs there, with this repo mounted at `/workspace/mp` (for example `docker run -v $PWD:/workspace/mp ...`). The scripts assume that path.
- **Tools on the board:** `scripts/nb_bench.cpp` is a small timing / smoke-test program I wrote for `.nb` files. Build it on the board:

```
g++ -O2 -std=c++14 -pthread scripts/nb_bench.cpp -I ~/awnn_inference/include -L ~/awnn_inference/install/lib \
    -lawnn_inference -lNBGlinker -lVIPhal -Wl,-rpath,$HOME/awnn_inference/install/lib -o ~/nb_bench/nb_bench
export LD_LIBRARY_PATH=$HOME/awnn_inference/install/lib
~/nb_bench/nb_bench -m model.nb -n 100                      # latency with random input
~/nb_bench/nb_bench -m model.nb -n 1 -i0 input.bin -o out.bin   # real input (raw bytes), outputs saved as float32 (out.bin, out.bin.1, ...)
```

`scripts/board_eval.sh` wraps that: copy a model and a folder of `*.in.bin` files to the board, run them all, copy the outputs back.

## How I converted it

1. `scripts/get_model.sh` downloads `blaze_face_short_range.tflite` from MediaPipe's model bucket. `scripts/inspect_tflite.py` shows what is in it: only CONV_2D, DEPTHWISE_CONV_2D, RELU, ADD, PAD, MAX_POOL_2D, RESHAPE, CONCATENATION and DEQUANTIZE (the weights are stored as fp16). All imported with 0 errors.
2. Calibration: 47 images that look like the real input (face crops resized to 128x128), listed in a `dataset.txt`. I'm not including mine because they contain people. Use your own.
3. In the container: `bash scripts/mp_convert.sh face_det blaze_face_short_range.tflite /workspace/mp/calib128/dataset.txt 127.5 0.007843 i16 pcq`. The mean/scale go into the NPU as a preprocessing node, so the board feeds raw uint8 RGB.
4. `scripts/sim_compare.sh` runs the Acuity simulator on a test crop (put your own 128x128 face at `test128/face.png`) and compares int8 and int16 with the float TFLite model. This is how I knew in advance that int8 would lose the face.
5. On the board: `scripts/board_eval.sh` / `nb_bench` as above. Input is raw uint8 RGB, 128x128, HWC.

## Why int8 failed here

I haven't dug into why. My guess is that the output logits have a wide range, which 8-bit per-channel quantisation handles badly, while int16 fixed point keeps enough resolution. On the palm detector it is the other way round, so I'd just try both for each network and compare.

## Things that cost me time (Acuity / A733 in general)

These are not specific to this model, so if you convert something else you will probably hit them too.

1. **Export for the right chip.** `--optimize VIP9000NANODI_PLUS_PID0X1000003B` is the A733. An older target I tried first (`VIP9A00NANOQI_PID0X10001029`) gave empty `.nb` files and "Fatal model generation error".
2. **Use `pegasus_patched.py`.** The plain `pegasus.py` in my container failed because there was no torch installed. `env.sh` sets `$PEG` to the patched one.
3. **Put the normalisation on the NPU.** In the input meta I set `add_preproc_node: true`, `reverse_channel: false` and per-channel mean/scale, so the board feeds raw uint8 RGB. Same for the output: `add_postproc_node: true` gives you float32 outputs, which is all the wrapper's output reader understands.
4. **`reverse_channel: false` matters.** The generated default is `true`, which silently flips the channel order of the calibration data. For images that just hurts accuracy a bit; for non-image tensors it scrambled everything (see the SmolLM2 repo).
5. **Always compare the quantised simulator with the float model before trusting a `.nb`**, and then compare the board with the simulator. They do not always agree (see the palm detector below).
6. **Quantisation choice is per model.** int8 per-channel (`pcq`) is fast and fine for some networks, wrecks others; `int16` (`dynamic_fixed_point`) is about 1.5-3x slower but usually close to lossless.

## Credits

Model from [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache-2.0).

## Use it from Python

There is now a pip-installable Python wrapper for the NPU, [cubie-npu](https://github.com/Aseel1206/cubie-npu) (`pip install cubie-npu`, [on PyPI](https://pypi.org/project/cubie-npu/)). With it you can skip the C++ tools above:

```python
from cubie_npu import Model
m = Model("models/blazeface_short_int16.nb")
regressors, scores = m(rgb_128x128_uint8)                 # raw outputs, shapes (1,896,16) and (1,896,1); decoding is not included
```

## More models on the Cubie A7A NPU

Same board, same toolchain, same kind of notes:

- [cubie-npu](https://github.com/Aseel1206/cubie-npu): Python bindings for the NPU
- [radxa-cubie-a7a-npu-mediapipe-hands](https://github.com/Aseel1206/radxa-cubie-a7a-npu-mediapipe-hands): MediaPipe palm detector + hand landmarks (int8, ~5 ms total)
- [radxa-cubie-a7a-npu-deit-tiny](https://github.com/Aseel1206/radxa-cubie-a7a-npu-deit-tiny): DeiT-tiny vision transformer (14.5 ms int8)
- [radxa-cubie-a7a-npu-depth-anything-v2](https://github.com/Aseel1206/radxa-cubie-a7a-npu-depth-anything-v2): Depth Anything V2 small depth estimation (255 ms int8)
- [radxa-cubie-a7a-npu-llm-smollm2-135m](https://github.com/Aseel1206/radxa-cubie-a7a-npu-llm-smollm2-135m): SmolLM2-135M chat (~23 tokens/s)
- [fast-acvnet-radxa-npu](https://github.com/Aseel1206/fast-acvnet-radxa-npu): stereo depth

## Licence

The scripts and docs here are MIT (see `LICENSE`). The neural networks are not mine and keep their own licences; check the model card of the original before reusing a converted `.nb`.
