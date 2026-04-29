# ONNX Verification

Each ONNX file is compared against the PyTorch reference on the same random input across multiple shapes. Tolerance is precision-appropriate: FP32 expects near-bit-level match; FP16 and INT8 expect larger gaps.

## `edsr_fp32.onnx`

- **ORT provider**: `CUDAExecutionProvider`
- **Tolerance (atol)**: 1.0e-04
- **Overall**: **PASS**

| Shape | max abs diff | max rel diff | Passed |
|---|---:|---:|:---:|
| (1, 3, 96, 96) | 0.00e+00 | 0.00e+00 | PASS |
| (1, 3, 64, 64) | 0.00e+00 | 0.00e+00 | PASS |
| (1, 3, 128, 64) | 0.00e+00 | 0.00e+00 | PASS |

## `edsr_fp16.onnx`

- **ORT provider**: `CUDAExecutionProvider`
- **Tolerance (atol)**: 5.0e-02
- **Overall**: **PASS**

| Shape | max abs diff | max rel diff | Passed |
|---|---:|---:|:---:|
| (1, 3, 96, 96) | 9.22e-04 | 1.42e-01 | PASS |
| (1, 3, 64, 64) | 8.81e-04 | 4.57e-03 | PASS |
| (1, 3, 128, 64) | 9.29e-04 | 7.47e-03 | PASS |

## `edsr_int8_static.onnx`

- **ORT provider**: `CUDAExecutionProvider`
- **Tolerance (atol)**: 1.0e-01
- **Overall**: **PASS**

| Shape | max abs diff | max rel diff | Passed |
|---|---:|---:|:---:|
| (1, 3, 96, 96) | 3.15e-02 | 2.12e+01 | PASS |
| (1, 3, 64, 64) | 2.74e-02 | 1.31e-01 | PASS |
| (1, 3, 128, 64) | 3.11e-02 | 2.17e-01 | PASS |

