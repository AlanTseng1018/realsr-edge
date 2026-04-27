# cpp_inference

Minimal C++ ONNX Runtime reference for running EDSR-baseline as a single-image
super-resolution command-line tool. Mirrors the Python `src/deployment/`
inference path. The point of this directory:

1. **Reproduce the PyTorch model's output bit-for-bit (within FP32 tolerance) in C++.**
   If C++ and Python disagree, the export pipeline is broken — fail there, not
   silently downstream.
2. **Show the deploy pattern.** Every TV-SoC NPU SDK (TensorRT, SNPE,
   NeuroPilot, …) follows the same skeleton: load model file → prepare
   input tensor → run → read output. Once this works, swapping ONNX RT for
   a vendor SDK is mostly an API rename.

The build is CPU-only on purpose: zero CUDA toolchain headache, runs anywhere,
and "PyTorch CUDA vs ORT CUDA" speed comparisons live in the Python benchmark
scripts where they belong.

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
├── CMakeLists.txt        - find ONNX RT, compile sr_cli, copy DLL
├── README.md             - you are here
├── src/
│   └── main.cpp          - end-to-end CLI in one file (~200 lines)
├── third_party/
│   └── stb/              - drop stb_image.h, stb_image_write.h here
└── build/                - cmake output (gitignored)
```

We deliberately keep everything in one `main.cpp` for V1. Once we have a
second use case (e.g. batched inference, video frames), the right move is
to split into `inference.{cpp,h}` and `preprocess.{cpp,h}` per the project
plan in `Project_Structure.txt`.

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
