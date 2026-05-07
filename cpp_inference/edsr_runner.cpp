// edsr_runner.cpp -- C++ deployment of EDSR-baseline via ONNX Runtime.
//
// Pipeline: HR PNG -> bicubic downsample to LR -> ONNX SR model -> SR PNG.
// Reports per-iteration latency (mean / std) over a configurable iter count.
// Mirrors the Python deploy benchmark so cross-language numbers can be sanity-checked.
//
// Build via build.bat (uses VS BuildTools cl.exe + ORT GPU prebuilt). Runtime DLL search
// path is wired up by run.bat to point at the ORT and the bundled CUDA/TRT runtimes.

#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>


// ---------------------------------------------------------------------------
// Tiny helpers
// ---------------------------------------------------------------------------

static void die(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr); std::exit(1);
}

struct Args {
    std::string onnx;
    std::string input;
    std::string output = "sr_out.png";
    std::string provider = "cuda";       // cpu | cuda | tensorrt
    int scale = 2;
    int iters = 100;
    int warmup = 10;
    int crop_hr = 0;                     // 0 = full image, >0 = center-crop HR to NxN
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) die("--%s missing value", name);
            return argv[++i];
        };
        if      (k == "--onnx")     a.onnx = next("onnx");
        else if (k == "--input")    a.input = next("input");
        else if (k == "--output")   a.output = next("output");
        else if (k == "--provider") a.provider = next("provider");
        else if (k == "--scale")    a.scale = std::atoi(next("scale"));
        else if (k == "--iters")    a.iters = std::atoi(next("iters"));
        else if (k == "--warmup")   a.warmup = std::atoi(next("warmup"));
        else if (k == "--crop-hr")  a.crop_hr = std::atoi(next("crop-hr"));
        else if (k == "--help" || k == "-h") {
            std::printf("usage: edsr_runner --onnx <path> --input <hr.png> [--output sr.png] "
                        "[--provider cpu|cuda|tensorrt] [--scale 2] [--iters 100]\n");
            std::exit(0);
        }
        else die("unknown arg: %s", argv[i]);
    }
    if (a.onnx.empty() || a.input.empty()) die("--onnx and --input are required");
    return a;
}

static std::wstring widen(const std::string& s) {
    return std::wstring(s.begin(), s.end());  // ASCII-only paths assumed
}


// ---------------------------------------------------------------------------
// Image IO + format conversion
// ---------------------------------------------------------------------------

// HWC uint8 -> CHW float32 [0,1].
static std::vector<float> hwc_uint8_to_chw_float(const unsigned char* hwc, int H, int W) {
    std::vector<float> out(static_cast<size_t>(3) * H * W);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                out[(c * H + y) * W + x] = hwc[(y * W + x) * 3 + c] / 255.0f;
    return out;
}

// CHW float32 [0,1] -> HWC uint8 (clamps + rounds).
static std::vector<unsigned char> chw_float_to_hwc_uint8(const float* chw, int H, int W) {
    std::vector<unsigned char> out(static_cast<size_t>(3) * H * W);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < 3; ++c) {
                float v = chw[(c * H + y) * W + x];
                v = std::max(0.0f, std::min(1.0f, v));
                out[(y * W + x) * 3 + c] = static_cast<unsigned char>(std::round(v * 255.0f));
            }
    return out;
}


// ---------------------------------------------------------------------------
// ONNX Runtime session setup
// ---------------------------------------------------------------------------

static Ort::Session make_session(Ort::Env& env, const Args& a) {
    Ort::SessionOptions opt;
    opt.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    opt.SetIntraOpNumThreads(1);

    if (a.provider == "cpu") {
        // default CPU EP, nothing to add
    } else if (a.provider == "cuda") {
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = 0;
        opt.AppendExecutionProvider_CUDA(cuda);
    } else if (a.provider == "tensorrt") {
        OrtTensorRTProviderOptions trt{};
        trt.device_id = 0;
        trt.trt_fp16_enable = 1;        // let TRT EP auto-fuse FP16 kernels for an FP32/FP16 model
        trt.trt_int8_enable = 0;        // INT8 here would require a calibrator dataset; we use ORT EP defaults only
        trt.trt_engine_cache_enable = 1;
        trt.trt_engine_cache_path = "trt_engine_cache";
        opt.AppendExecutionProvider_TensorRT(trt);
    } else {
        die("unknown provider: %s", a.provider.c_str());
    }

    return Ort::Session(env, widen(a.onnx).c_str(), opt);
}


// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) try {
    Args args = parse_args(argc, argv);

    // 1. Load HR PNG.
    int H_hr = 0, W_hr = 0, C = 0;
    unsigned char* hr_raw = stbi_load(args.input.c_str(), &W_hr, &H_hr, &C, 3);
    if (!hr_raw) die("failed to load %s: %s", args.input.c_str(), stbi_failure_reason());
    std::printf("loaded HR: %dx%d (%d channels read as 3)\n", W_hr, H_hr, C);

    // Crop strategy: --crop-hr N forces a centered NxN HR crop (snapped to
    // scale grid). Default (0) keeps the full image, just snapped to scale grid.
    int H_hr_s, W_hr_s;
    if (args.crop_hr > 0) {
        int n = (args.crop_hr / args.scale) * args.scale;
        if (n > H_hr || n > W_hr) die("crop-hr=%d larger than image %dx%d", n, W_hr, H_hr);
        H_hr_s = W_hr_s = n;
    } else {
        H_hr_s = (H_hr / args.scale) * args.scale;
        W_hr_s = (W_hr / args.scale) * args.scale;
    }
    int oy = (H_hr - H_hr_s) / 2;
    int ox = (W_hr - W_hr_s) / 2;
    if (H_hr_s != H_hr || W_hr_s != W_hr) {
        std::printf("center-cropping HR to %dx%d (offset %d,%d)\n", W_hr_s, H_hr_s, ox, oy);
    }
    std::vector<unsigned char> hr_hwc(static_cast<size_t>(3) * H_hr_s * W_hr_s);
    for (int y = 0; y < H_hr_s; ++y)
        std::memcpy(&hr_hwc[y * W_hr_s * 3],
                    &hr_raw[((y + oy) * W_hr + ox) * 3],
                    static_cast<size_t>(W_hr_s) * 3);
    stbi_image_free(hr_raw);

    // 2. Bicubic downsample HR -> LR (matches Python val pipeline's bicubic mode).
    int H_lr = H_hr_s / args.scale;
    int W_lr = W_hr_s / args.scale;
    std::vector<unsigned char> lr_hwc(static_cast<size_t>(3) * H_lr * W_lr);
    stbir_resize_uint8_srgb(
        hr_hwc.data(), W_hr_s, H_hr_s, 0,
        lr_hwc.data(), W_lr, H_lr, 0,
        STBIR_RGB
    );
    std::printf("downsampled to LR: %dx%d\n", W_lr, H_lr);

    // 3. Convert LR HWC uint8 -> NCHW float [0,1].
    std::vector<float> lr_chw = hwc_uint8_to_chw_float(lr_hwc.data(), H_lr, W_lr);

    // 4. Build ORT session on requested provider.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edsr_runner");
    Ort::Session session = make_session(env, args);

    // 5. Resolve input / output names + shapes from the model.
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name_alloc  = session.GetInputNameAllocated(0, alloc);
    auto out_name_alloc = session.GetOutputNameAllocated(0, alloc);
    const char* in_name  = in_name_alloc.get();
    const char* out_name = out_name_alloc.get();

    std::array<int64_t, 4> in_shape  = {1, 3, H_lr, W_lr};
    Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 6. Wrap input tensor (NCHW float32). We re-wrap each iteration so memory is owned by us.
    auto run_once = [&]() -> Ort::Value {
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            cpu_mem, lr_chw.data(), lr_chw.size(),
            in_shape.data(), in_shape.size()
        );
        const char* in_names[]  = { in_name };
        const char* out_names[] = { out_name };
        Ort::Value out_tensor{nullptr};
        session.Run(Ort::RunOptions{nullptr},
                    in_names,  &in_tensor,  1,
                    out_names, &out_tensor, 1);
        return out_tensor;
    };

    // 7. Warmup.
    std::printf("warmup x %d ...\n", args.warmup);
    for (int i = 0; i < args.warmup; ++i) (void)run_once();

    // 8. Timed iters. We record the LAST output for PNG saving so we don't pay copy cost in timing.
    std::printf("timed x %d on provider=%s ...\n", args.iters, args.provider.c_str());
    std::vector<double> times_ms; times_ms.reserve(args.iters);
    Ort::Value last{nullptr};
    for (int i = 0; i < args.iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        Ort::Value out = run_once();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
        last = std::move(out);
    }

    double mean = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
    double sq = 0.0; for (double t : times_ms) sq += (t - mean) * (t - mean);
    double sd = std::sqrt(sq / times_ms.size());
    std::printf("latency: %.3f +/- %.3f ms  (n=%d)\n", mean, sd, args.iters);

    // 9. Read back output, write SR PNG.
    auto info = last.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();  // expect [1, 3, H_lr*scale, W_lr*scale]
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3) die("unexpected output shape");
    int H_sr = static_cast<int>(shape[2]);
    int W_sr = static_cast<int>(shape[3]);
    std::printf("SR shape: %dx%d (scale %.1fx)\n",
                W_sr, H_sr, static_cast<double>(W_sr) / W_lr);

    const float* sr_chw = last.GetTensorData<float>();
    std::vector<unsigned char> sr_hwc = chw_float_to_hwc_uint8(sr_chw, H_sr, W_sr);
    int wr = stbi_write_png(args.output.c_str(), W_sr, H_sr, 3, sr_hwc.data(), W_sr * 3);
    if (!wr) die("failed to write %s", args.output.c_str());
    std::printf("wrote %s\n", args.output.c_str());

    // 10. Optional: PSNR vs the cropped HR (sanity-check that the C++ pipeline output
    //     matches the model's expected behaviour).
    double mse = 0.0;
    if (H_sr == H_hr_s && W_sr == W_hr_s) {
        for (size_t i = 0; i < sr_hwc.size(); ++i) {
            double d = (sr_hwc[i] / 255.0) - (hr_hwc[i] / 255.0);
            mse += d * d;
        }
        mse /= static_cast<double>(sr_hwc.size());
        double psnr = 10.0 * std::log10(1.0 / std::max(mse, 1e-10));
        std::printf("PSNR vs HR-crop: %.3f dB (bicubic-LR -> SR vs original HR)\n", psnr);
    }

    return 0;
}
catch (const Ort::Exception& e) {
    std::fprintf(stderr, "ORT error: %s\n", e.what());
    return 2;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
}
