// sr_cli: minimal C++ EDSR ONNX inference reference.
//
// Usage:
//   sr_cli <model.onnx> <input.png> <output.png>
//
// The program loads an ONNX model exported from PyTorch (see
// src/deployment/export_onnx.py), runs inference on one PNG image at the
// model's native scale (2x by default), and writes the super-resolved
// result as a PNG. CPU execution provider only -- the goal is to mirror
// the Python reference numerically, not to chase peak GPU throughput.
//
// Image I/O uses stb_image / stb_image_write (header-only, public domain).
// Drop them into cpp_inference/third_party/stb/ before building. See README.

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// --------------------------------------------------------------------------
// Image helpers (RGB <-> CHW float32 in [0,1])
// --------------------------------------------------------------------------

struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> data;  // HWC, RGB, row-major
};

Image load_rgb_png(const std::string& path) {
    int w = 0, h = 0, ch = 0;
    // 3rd arg is desired_channels; 3 forces RGB (stbi expands grayscale,
    // drops alpha) so downstream code can rely on a fixed layout.
    uint8_t* px = stbi_load(path.c_str(), &w, &h, &ch, 3);
    if (!px) {
        throw std::runtime_error("stbi_load failed for: " + path);
    }
    Image img;
    img.width = w;
    img.height = h;
    img.data.assign(px, px + static_cast<std::size_t>(w) * h * 3);
    stbi_image_free(px);
    return img;
}

void save_rgb_png(const std::string& path, const Image& img) {
    // 4th arg is stride_in_bytes; 0 means "pack tightly" (we do).
    int rc = stbi_write_png(path.c_str(), img.width, img.height, 3,
                            img.data.data(), img.width * 3);
    if (rc == 0) {
        throw std::runtime_error("stbi_write_png failed for: " + path);
    }
}

// HWC uint8 -> CHW float32 normalized to [0, 1]. Output buffer is laid out
// as [R-plane | G-plane | B-plane], each plane = width * height floats.
std::vector<float> hwc_uint8_to_chw_float01(const Image& img) {
    const std::size_t n = static_cast<std::size_t>(img.width) * img.height;
    std::vector<float> out(n * 3);
    for (std::size_t i = 0; i < n; ++i) {
        out[0 * n + i] = img.data[i * 3 + 0] / 255.0f;
        out[1 * n + i] = img.data[i * 3 + 1] / 255.0f;
        out[2 * n + i] = img.data[i * 3 + 2] / 255.0f;
    }
    return out;
}

// CHW float -> HWC uint8 RGB, with [0,1] clamp + round.
Image chw_float01_to_hwc_uint8(const float* chw, int width, int height) {
    Image img;
    img.width = width;
    img.height = height;
    img.data.resize(static_cast<std::size_t>(width) * height * 3);
    const std::size_t n = static_cast<std::size_t>(width) * height;
    for (std::size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = chw[c * n + i];
            v = std::clamp(v, 0.0f, 1.0f);
            img.data[i * 3 + c] = static_cast<uint8_t>(v * 255.0f + 0.5f);
        }
    }
    return img;
}

// --------------------------------------------------------------------------
// ONNX Runtime helpers
// --------------------------------------------------------------------------

// Convert a const char* path into a wide string on Windows (Ort::Session
// takes a wchar_t* there) and a plain string on POSIX.
#ifdef _WIN32
std::wstring to_session_path(const std::string& s) {
    return std::wstring(s.begin(), s.end());
}
#else
std::string to_session_path(const std::string& s) {
    return s;
}
#endif

}  // anonymous namespace

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> <input.png> <output.png>\n";
        return 1;
    }
    const std::string model_path  = argv[1];
    const std::string input_path  = argv[2];
    const std::string output_path = argv[3];

    try {
        // 1. Load image and convert to CHW float32 NCHW tensor.
        const Image lr = load_rgb_png(input_path);
        std::cout << "Input image: " << lr.width << "x" << lr.height
                  << " (RGB)\n";

        std::vector<float> input_chw = hwc_uint8_to_chw_float01(lr);
        const std::array<int64_t, 4> input_shape = {
            1, 3,
            static_cast<int64_t>(lr.height),
            static_cast<int64_t>(lr.width)
        };

        // 2. Build ONNX Runtime session (CPU EP).
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sr_cli");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);  // deterministic, easy to compare
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        const auto session_path = to_session_path(model_path);
        Ort::Session session(env, session_path.c_str(), opts);

        // Input/output names (allocate with default allocator; both names are
        // owned by the AllocatedStringPtr objects -- keep them alive for the
        // duration of session.Run by storing the AllocatedStringPtr).
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_alloc  = session.GetInputNameAllocated(0, allocator);
        auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
        const char* input_name  = input_name_alloc.get();
        const char* output_name = output_name_alloc.get();

        // 3. Wrap the input vector as an Ort::Value.
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_chw.data(), input_chw.size(),
            input_shape.data(), input_shape.size());

        // 4. Run.
        auto t0 = std::chrono::steady_clock::now();
        std::vector<Ort::Value> outputs = session.Run(
            Ort::RunOptions{nullptr},
            &input_name, &input_tensor, 1,
            &output_name, 1);
        auto t1 = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (outputs.size() != 1 || !outputs[0].IsTensor()) {
            throw std::runtime_error("Unexpected ONNX output");
        }

        // 5. Read output shape and pull the float buffer.
        Ort::TensorTypeAndShapeInfo info =
            outputs[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> out_shape = info.GetShape();
        if (out_shape.size() != 4 || out_shape[0] != 1 || out_shape[1] != 3) {
            std::cerr << "Unexpected output shape (expected [1,3,H,W]):";
            for (auto d : out_shape) std::cerr << ' ' << d;
            std::cerr << '\n';
            return 2;
        }
        const int out_h = static_cast<int>(out_shape[2]);
        const int out_w = static_cast<int>(out_shape[3]);
        const float* out_ptr = outputs[0].GetTensorData<float>();
        std::cout << "Output tensor: " << out_w << "x" << out_h
                  << " (CHW float32)\n";
        std::cout << "Inference time: " << elapsed_ms << " ms\n";

        // 6. Convert to uint8 RGB and save.
        Image sr = chw_float01_to_hwc_uint8(out_ptr, out_w, out_h);
        save_rgb_png(output_path, sr);
        std::cout << "Wrote " << output_path << " ("
                  << sr.width << "x" << sr.height << ")\n";

        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << '\n';
        return 3;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 4;
    }
}
