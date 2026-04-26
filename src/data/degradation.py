"""Realistic degradation pipeline for LR/HR pair synthesis.

Why this module exists
----------------------
Academic SR datasets typically generate the LR image with a single bicubic
downsample. Real TV content has gone through a long chain of lossy operations
before it reaches a TV-side SR engine: lens optics, sensor noise, encoder
chroma subsampling, scaler interpolation differences, JPEG/H.264 compression
artifacts, etc. A model trained only on bicubic LR generalizes poorly to that
distribution. This module composes degradations that approximate those
real-world phenomena, used for the project's "Track B (Realistic)" training
recipe (see SPECIFICATION.md §1.3).

Convention
----------
* Images are ``np.uint8`` of shape ``(H, W, 3)``.
* Channel order is BGR (OpenCV native). The pipeline is channel-symmetric
  except for ``apply_jpeg_compression`` — JPEG quantizes in YCbCr, so chroma
  artifacts are slightly different between RGB-as-BGR and true BGR. For SR
  training this difference is negligible; we keep BGR throughout to avoid
  spurious ``cvtColor`` round-trips.
* Every method takes an optional ``random_state`` (seed int or
  ``np.random.Generator``) so a fixed seed reproduces a fixed result. Passing
  the same ``Generator`` to multiple calls advances its state — that is the
  intended behavior inside :meth:`RealisticDegradation.random_degradation_pipeline`.
"""

from __future__ import annotations

from typing import Union

import cv2
import numpy as np

RandomState = Union[int, np.random.Generator, None]


def _resolve_rng(random_state: RandomState) -> np.random.Generator:
    """Coerce a seed / Generator / None into a ``np.random.Generator``.

    A ``Generator`` passed in is returned as-is (state is shared with the
    caller — this is what we want when the pipeline threads one rng through
    several steps).
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


class RealisticDegradation:
    """Composable degradations that approximate real TV-content artifacts.

    The class is stateless aside from its parameter ranges (kept as class
    attributes so they're easy to override per subclass / experiment). All
    methods are pure: they consume an image and an optional rng, and return a
    new image.
    """

    BLUR_KERNEL_CHOICES: tuple[int, ...] = (3, 5, 7)
    BLUR_SIGMA_RANGE: tuple[float, float] = (0.1, 2.0)
    NOISE_SIGMA_RANGE: tuple[float, float] = (0.0, 25.0)
    JPEG_QUALITY_RANGE: tuple[int, int] = (60, 95)

    INTERPOLATION_MAP: dict[str, int] = {
        "bicubic": cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
    }

    def apply_blur(
        self,
        image: np.ndarray,
        kernel_size: int | None = None,
        sigma: float | None = None,
        random_state: RandomState = None,
    ) -> np.ndarray:
        """Gaussian blur — models lens defocus and motion blur.

        Args:
            image: ``(H, W, 3)`` ``uint8`` image.
            kernel_size: Odd int. ``None`` -> random pick from
                :attr:`BLUR_KERNEL_CHOICES`.
            sigma: Gaussian std. ``None`` -> uniform in
                :attr:`BLUR_SIGMA_RANGE`.
            random_state: Seed or Generator for reproducibility.
        """
        rng = _resolve_rng(random_state)
        if kernel_size is None:
            kernel_size = int(rng.choice(self.BLUR_KERNEL_CHOICES))
        if sigma is None:
            lo, hi = self.BLUR_SIGMA_RANGE
            sigma = float(rng.uniform(lo, hi))
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)

    def apply_noise(
        self,
        image: np.ndarray,
        noise_type: str = "gaussian",
        sigma: float | None = None,
        random_state: RandomState = None,
    ) -> np.ndarray:
        """Additive sensor-style noise — models camera sensor read noise.

        Args:
            image: ``(H, W, 3)`` ``uint8`` image.
            noise_type: Currently only ``"gaussian"`` is supported.
            sigma: Noise std in pixel units (0..255 scale). ``None`` ->
                uniform in :attr:`NOISE_SIGMA_RANGE`.
            random_state: Seed or Generator for reproducibility.
        """
        if noise_type != "gaussian":
            raise ValueError(
                f"Unsupported noise_type={noise_type!r}; only 'gaussian' is implemented."
            )
        rng = _resolve_rng(random_state)
        if sigma is None:
            lo, hi = self.NOISE_SIGMA_RANGE
            sigma = float(rng.uniform(lo, hi))
        noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0.0, 255.0).astype(np.uint8)

    def apply_jpeg_compression(
        self,
        image: np.ndarray,
        quality: int | None = None,
        random_state: RandomState = None,
    ) -> np.ndarray:
        """JPEG re-encode — models broadcast / streaming compression artifacts.

        Args:
            image: ``(H, W, 3)`` ``uint8`` image.
            quality: JPEG quality 1..100 (higher = less artifact). ``None`` ->
                uniform integer in :attr:`JPEG_QUALITY_RANGE` (inclusive).
            random_state: Seed or Generator for reproducibility.
        """
        rng = _resolve_rng(random_state)
        if quality is None:
            lo, hi = self.JPEG_QUALITY_RANGE
            quality = int(rng.integers(lo, hi + 1))
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, encoded = cv2.imencode(".jpg", image, params)
        if not ok:
            raise RuntimeError("cv2.imencode('.jpg', ...) failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded

    def apply_downsample(
        self,
        image: np.ndarray,
        scale: int = 2,
        method: str | None = None,
        random_state: RandomState = None,
    ) -> np.ndarray:
        """Spatial downsample — models scaler differences across encoders / TVs.

        Different vendors implement scaling with different kernels (bicubic,
        bilinear, area, nearest), so we randomize the choice during training.

        Args:
            image: ``(H, W, 3)`` ``uint8`` image.
            scale: Integer downscale factor. Output size = ``(H//scale, W//scale)``.
            method: One of :attr:`INTERPOLATION_MAP` keys. ``None`` -> random.
            random_state: Seed or Generator for reproducibility.
        """
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")
        rng = _resolve_rng(random_state)
        if method is None:
            method = str(rng.choice(list(self.INTERPOLATION_MAP.keys())))
        if method not in self.INTERPOLATION_MAP:
            raise ValueError(
                f"Unknown method {method!r}; valid: {list(self.INTERPOLATION_MAP.keys())}"
            )
        h, w = image.shape[:2]
        new_w, new_h = w // scale, h // scale
        return cv2.resize(image, (new_w, new_h), interpolation=self.INTERPOLATION_MAP[method])

    def random_degradation_pipeline(
        self,
        image: np.ndarray,
        scale: int = 2,
        random_state: RandomState = None,
    ) -> np.ndarray:
        """Compose a random LR image from an HR image.

        Step order: ``blur -> downsample -> noise -> compression``. Each step
        is independently applied with probability 0.5, and applied steps draw
        their parameters at random from the class's ranges. The ordering
        matters: blur before downsample mirrors optical-then-sensor reality;
        noise after downsample mirrors sensor-then-encoder reality;
        compression last mirrors the broadcast / file step.

        Note that with 50% probability per step, the output may have the same
        spatial size as the input (when downsample is skipped). This is
        intentional augmentation: the SR model should be robust to whether
        downsampling actually happened. If you need a guaranteed LR output,
        call :meth:`apply_downsample` directly outside the pipeline.

        Args:
            image: HR ``(H, W, 3)`` ``uint8`` image.
            scale: Downsample factor used when the downsample step fires.
            random_state: Seed or Generator. A single ``Generator`` is threaded
                through every nested call so the whole pipeline is
                reproducible from one seed.
        """
        rng = _resolve_rng(random_state)
        out = image
        if rng.random() < 0.5:
            out = self.apply_blur(out, random_state=rng)
        if rng.random() < 0.5:
            out = self.apply_downsample(out, scale=scale, random_state=rng)
        if rng.random() < 0.5:
            out = self.apply_noise(out, random_state=rng)
        if rng.random() < 0.5:
            out = self.apply_jpeg_compression(out, random_state=rng)
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

    deg = RealisticDegradation()

    out_default = deg.random_degradation_pipeline(img, scale=2, random_state=42)
    out_repeat = deg.random_degradation_pipeline(img, scale=2, random_state=42)

    assert np.array_equal(out_default, out_repeat), (
        "Pipeline is not deterministic for a fixed seed."
    )

    print("RealisticDegradation quick test: OK")
    print(f"  input shape    : {img.shape}, dtype={img.dtype}")
    print(f"  output shape   : {out_default.shape}, dtype={out_default.dtype}")
    print(f"  output range   : [{out_default.min()}, {out_default.max()}]")
    print(f"  reproducible   : True (same seed -> identical output)")

    print()
    print("Per-step shape sanity:")
    print(f"  apply_blur            : {deg.apply_blur(img, random_state=1).shape}")
    print(f"  apply_noise           : {deg.apply_noise(img, random_state=1).shape}")
    print(f"  apply_jpeg_compression: {deg.apply_jpeg_compression(img, random_state=1).shape}")
    print(f"  apply_downsample(s=2) : {deg.apply_downsample(img, scale=2, random_state=1).shape}")
