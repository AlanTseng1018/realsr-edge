# cpp_inference

Two C++ entry points, both load the same ONNX export and run inference on a PNG. Pick by use case:

| Entry point | Build system | EPs supported | Best for |
|---|---|---|---|
| **`edsr_runner.cpp`** (current) | `build.bat` (cl.exe direct) | CPU, CUDA, TensorRT | latency benchmark vs Python, cross-language correctness check |
| `src/main.cpp` (V1 minimal reference) | `CMakeLists.txt` (CMake) | CPU only | bit-for-bit numerical parity vs PyTorch |

Both share the same *purpose*: prove the model runs in a real C++ runtime and the export pipeline is numerically faithful, the way every TV-SoC / edge JD assumes you can demonstrate. **They are not a TV product latency benchmark** -- the actual deploy target is a TV SoC NPU running a vendor SDK, not a consumer NVIDIA GPU. The numbers below are deployment-prep references, not product targets.

## Validated results -- `edsr_runner.cpp` on RTX 3060 Laptop (consumer Ampere, sm86)

### Apples-to-apples vs Python benchmark (96x96 LR -> 192x192 HR, matches training patch size)

| EP | C++ latency | Python latency | Cross-language delta |
|---|---:|---:|---:|
| CUDA | 4.18 +/- 0.04 ms | 5.28 ms | C++ 21% faster (less wrapper overhead) |
| TensorRT | 1.41 +/- 0.06 ms | 1.28 ms | within noise |

PSNR matches Python within float-rounding noise -- confirms the C++ binary produces the same SR output as the Python pipeline. 96x96 LR is the **per-tile** number that would matter on a tile-based 1080p->4K pipeline (whether on this GPU or on an NPU), not a full-frame product latency.

### Full-frame illustrative run (`0879.png`, 1020x936 LR -> 2040x1872 SR)

20 timed iters per EP. *Not* a product target -- a 1080p frame on a TV product would be processed via tiling on the actual NPU, where absolute numbers will differ.

| EP | Latency | PSNR vs HR (cross-EP correctness) |
|---|---:|---:|
| CPU | 18,853 +/- 162 ms | 29.450 dB |
| CUDA | 795 +/- 3 ms | 29.450 dB |
| TensorRT | 99 +/- 1 ms | 29.446 dB |

PSNR matches across all three EPs to within float-rounding noise -- the **important result here** is the cross-EP correctness check, not the latency number. The full-frame latency is included only to show that the inference path scales correctly to large inputs.

---

## V2: `edsr_runner.cpp` -- full GPU-capable runner (recommended)

A single ~290-line TU that loads HR PNG, bicubic-downsamples to LR, runs ORT inference on chosen EP, writes SR PNG, reports latency mean/std and PSNR vs HR.

### Build

Prerequisites:
- VS 2022 BuildTools `cl.exe` (already installed in the dev environment)
- ONNX Runtime 1.25.0 GPU prebuilt extracted to `../third_party/onnxruntime/` (~281 MB zip from microsoft/onnxruntime GitHub releases)
- stb single-header libs in `../third_party/stb/`
- CUDA 12.x runtime DLLs (we reuse the ones bundled by `torch` in the project venv -- no separate CUDA Toolkit install needed)
- TensorRT 10.x runtime DLLs (reuse from `tensorrt_libs` Python wheel)

```cmd
build.bat
```

This calls `vcvars64.bat`, compiles against ORT headers, links `onnxruntime.lib`, and **copies ORT runtime DLLs next to the exe** so Windows DLL search picks them up before `C:\Windows\System32\onnxruntime.dll` (an older Windows-ML built-in 1.17.1 that would otherwise win).

Output: `build/edsr_runner.exe` plus the bundled `onnxruntime*.dll`.

### Run

```cmd
run.bat ^
    --onnx ..\results\onnx_exports\edsr_200ep\edsr_fp32.onnx ^
    --input ..\data\DIV2K\DIV2K_valid_HR\0879.png ^
    --output sr.png ^
    --provider tensorrt ^
    --iters 20 --warmup 5
```

`run.bat` prepends `third_party/onnxruntime/lib`, `.venv/.../torch/lib`, and `.venv/.../tensorrt_libs` to PATH so the EP DLLs find their CUDA / TRT dependencies.

CLI flags: `--onnx`, `--input`, `--output`, `--provider {cpu|cuda|tensorrt}`, `--scale`, `--iters`, `--warmup`. See `edsr_runner.cpp` for defaults.

### Why bicubic LR (not realistic-degradation LR)

The Python val pipeline uses `RealisticDegradation` (blur + banding + noise + JPEG) for accuracy benchmarks. Reproducing that pipeline in C++ would require porting the augmentations -- a separate engineering exercise that doesn't add deployment-pipeline value. Bicubic LR is deterministic, OpenCV-equivalent, and matches the `bicubic` track of `SRDataset` -- enough to prove the inference path works and produce a sensible PSNR cross-check.

### Known limitations / next steps

- **No INT8 path here.** ORT TRT EP with the QDQ INT8 ONNX hits the INT32-bias-DQ issue documented in `results/onnx_benchmark/edsr_200ep_full/deploy_summary.md` Section 8. Real INT8 deployment uses `benchmark_trt.py`'s native TRT API + calibrator path; the resulting `.engine` files in `results/trt_benchmark/edsr_200ep/engines/` would be loaded via `nvinfer1::IRuntime::deserializeCudaEngine` from a C++ binary linking against `NvInfer.h` + `nvinfer.lib`. That requires the TensorRT C++ SDK (~1 GB download, NVIDIA developer login). Out of scope for this V2; the artifact is ready when the toolchain is.
- **OOM warnings on TRT EP build.** TensorRT explores tactics that may need more GPU memory than the 6 GB laptop GPU has and falls back automatically. The warnings are noisy but harmless.
- **Static input shape.** The exported ONNX has fixed batch=1; running with a different input size triggers a one-time TRT re-build (cached under `trt_engine_cache/`).

---

## V1: `src/main.cpp` -- minimal CPU-only reference (bit-for-bit parity check)

Kept as the original purpose: prove the PyTorch -> ONNX export is numerically faithful. CPU-only on purpose: zero CUDA toolchain headache, runs anywhere, and "PyTorch CUDA vs ORT CUDA" speed comparisons live in the Python benchmark scripts where they belong.

---

## Prerequisites

- **CMake 3.18+**
- **A C++17 compiler**: MSVC 2019+ on Windows; gcc 9+ / clang 10+ on Linux
- **ONNX Runtime release build** (CPU is enough). Download from the
  [GitHub releases](https://github.com/microsoft/onnxruntime/releases) page,
  pick a recent stable like `onnxruntime-win-x64-1.20.0.zip`
  (Windows) or `onnxruntime-linux-x64-1.20.0.tgz` (Linux), and extract it
  somewhere — the path is what you'll pass as `-DONNXRUNTIME_DIR=...` below.
- **stb_image / stb_image_write** (header-only, public domain). Download
  from <https://github.com/nothings/stb>:
  ```
  curl -L -o cpp_inference/third_party/stb/stb_image.h \
       https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
  curl -L -o cpp_inference/third_party/stb/stb_image_write.h \
       https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
  ```

## Build (Windows, MSVC)

From the repo root, in a Developer Command Prompt:

```bat
cd cpp_inference
cmake -B build -DONNXRUNTIME_DIR="C:/path/to/onnxruntime-win-x64-1.20.0"
cmake --build build --config Release
```

The build copies `onnxruntime.dll` next to `sr_cli.exe`, so the binary is
self-contained in `cpp_inference/build/Release/`.

## Build (Linux)

```bash
cd cpp_inference
cmake -B build -DCMAKE_BUILD_TYPE=Release \
              -DONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.20.0
cmake --build build -j
# Add the lib dir to LD_LIBRARY_PATH at run time, OR set rpath at link time
```

## Run

```bash
# 1) Make sure you have an ONNX file (export from PyTorch first):
python -m src.deployment.export_onnx \
    --checkpoint results/checkpoints/edsr_baseline/final.pt \
    --output results/onnx_models/edsr_fp32.onnx

# 2) Run the C++ tool on a small input PNG:
cpp_inference/build/Release/sr_cli.exe \
    results/onnx_models/edsr_fp32.onnx \
    some_lr_image.png \
    sr_out.png
```

Expected output:

```
Input image: 96x96 (RGB)
Output tensor: 192x192 (CHW float32)
Inference time: <some-ms>
Wrote sr_out.png (192x192)
```

## Verifying parity with PyTorch

After `sr_cli` produces an SR PNG, you can compare it numerically with
PyTorch on the same input. From repo root:

```bash
python -c "
import cv2, numpy as np, torch
from src.models.edsr import EDSR
m = EDSR(scale_factor=2).eval()
m.load_state_dict(torch.load('results/checkpoints/edsr_baseline/final.pt',
                             map_location='cpu', weights_only=False)['model'])
lr = cv2.cvtColor(cv2.imread('some_lr_image.png'), cv2.COLOR_BGR2RGB)
x = torch.from_numpy(lr).permute(2,0,1).float().unsqueeze(0) / 255.0
with torch.no_grad():
    sr_pt = m(x).clamp(0,1).squeeze(0).permute(1,2,0).numpy()
sr_pt = (sr_pt * 255 + 0.5).astype(np.uint8)
sr_cpp = cv2.cvtColor(cv2.imread('sr_out.png'), cv2.COLOR_BGR2RGB)
diff = np.abs(sr_pt.astype(int) - sr_cpp.astype(int))
print(f'max  pixel diff: {diff.max()}')
print(f'mean pixel diff: {diff.mean():.4f}')
"
```

A max pixel diff of 0 or 1 is expected (rounding at the float-to-uint8 step
can flip a single LSB). Anything bigger means the export-then-deploy chain
has a real numeric bug — fix that before drawing any conclusions from
deploy-side benchmarks.

## File map

```
cpp_inference/
├── README.md             - you are here
├── edsr_runner.cpp       - V2: CPU/CUDA/TRT runner (~290 lines, today)
├── build.bat             - V2 build (cl.exe + DLL copy)
├── run.bat               - V2 launcher (DLL PATH wiring)
├── CMakeLists.txt        - V1 CMake-based build for src/main.cpp
├── src/
│   └── main.cpp          - V1 CPU-only reference (~200 lines)
├── build/                - exe + bundled DLLs (gitignored)
└── sr_*.png              - per-EP outputs from validation runs (gitignored)
../third_party/            (gitignored)
├── onnxruntime/          - ORT 1.25.0 GPU prebuilt (~370 MB extracted)
└── stb/                  - stb_image.h, stb_image_write.h, stb_image_resize2.h
```

## Swapping in a vendor SDK

The same skeleton works for any vendor inference runtime. The C++ shape is:

```
1. Initialize runtime / context     (Ort::Env  ↔  SNPE container ↔  TensorRT runtime)
2. Load compiled model              (Ort::Session ↔  zdl::SNPE ↔  IExecutionContext)
3. Prepare input tensor             (Ort::Value ↔  ITensor ↔  buffer + binding)
4. Run                              (session.Run ↔  snpe->execute ↔  context->enqueue)
5. Read output                      (.GetTensorData ↔  output_map ↔  output buffer)
```

Each step has a vendor-specific name but the same role. Swapping is mostly
mechanical once the inputs/outputs are well-defined ONNX tensors.
