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
    BANDING_BITS_CHOICES: tuple[int, ...] = (4, 5, 6)

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

    def apply_chroma_subsampling(
        self,
        image: np.ndarray,
        random_state: RandomState = None,
    ) -> np.ndarray:
        """4:2:0 chroma subsampling — models H.264 / H.265 color downsampling.

        Almost every TV-side input has been through chroma subsampling at some
        point in its pipeline: terrestrial broadcast (DVB-T/T2), streaming
        (HLS / DASH / MPEG-TS), Blu-ray, and the vast majority of MP4 files
        all carry 4:2:0 video. H.264 Main/High and H.265 Main both use 4:2:0
        by default. Modeling this is essential if the SR model is meant to
        consume real TV content rather than academic bicubic LR.

        4:2:0 in J:a:b notation (per ITU-R BT.601):
          J = 4   reference horizontal sample count
          a = 2   chroma samples in the FIRST row of every 4x2 luma block
          b = 0   chroma samples in the SECOND row (none — they reuse a's)
        Net: U and V are each downsampled 2x horizontally AND 2x vertically,
        while Y stays at full resolution. Visible artifact is "color bleeding"
        across sharp chrominance edges (e.g., red text on white).

        Implementation: BGR -> YUV (OpenCV's full-range BT.601 mapping),
        downsample U/V with INTER_AREA (proper area averaging — what an
        encoder would do), upsample back with INTER_NEAREST (no
        reconstruction filter — what a cheap decoder would do). Y is
        untouched. Round-trip back to BGR.

        Color-space caveat: cv2.COLOR_BGR2YUV is full-range Y'CbCr [0, 255].
        Real video uses studio-range BT.709 for HDTV (luma 16..235, chroma
        16..240). The difference matters for pixel-perfect reproduction but
        not for training a robustness-focused SR model, so we use OpenCV's
        default and document the gap here.

        ``random_state`` is accepted for API consistency with the other
        methods, but unused — this transform is fully deterministic.

        Status: implemented but **deliberately not used** in
        :meth:`random_degradation_pipeline` because on natural-image training
        data (DIV2K) the artifact produces ~38 dB PSNR -- essentially
        invisible, no useful invariance to learn. Kept here for analysis
        / future use when synthetic content (text, UI, anime) gets added to
        the training mix, where chroma artifacts actually bite.

        Args:
            image: ``(H, W, 3)`` ``uint8`` BGR image.
            random_state: Unused. Present for API symmetry.
        """
        del random_state  # explicit: parameter accepted, not consumed
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        h, w = yuv.shape[:2]
        y, u, v = cv2.split(yuv)

        half_w, half_h = max(1, w // 2), max(1, h // 2)
        u_low = cv2.resize(u, (half_w, half_h), interpolation=cv2.INTER_AREA)
        v_low = cv2.resize(v, (half_w, half_h), interpolation=cv2.INTER_AREA)
        u = cv2.resize(u_low, (w, h), interpolation=cv2.INTER_NEAREST)
        v = cv2.resize(v_low, (w, h), interpolation=cv2.INTER_NEAREST)

        yuv_out = cv2.merge([y, u, v])
        return cv2.cvtColor(yuv_out, cv2.COLOR_YUV2BGR)

    def apply_banding(
        self,
        image: np.ndarray,
        target_bits: int | None = None,
        random_state: RandomState = None,
    ) -> np.ndarray:
        """Bit-depth banding — approximates low-bitrate codec posterization.

        Banding is the artifact TV viewers complain about most: in smooth
        regions (sky, gradients, skin tones, walls) the codec's coarse
        quantization turns the underlying continuous gradient into visible
        "staircase" steps. It is the dominant failure mode of low-bitrate
        H.264 / H.265 streams and a common complaint in streaming services.

        Mechanism (principled approximation): real codec banding is produced
        by quantizing DCT coefficients in a frequency-domain transform. We
        approximate that with a much simpler **pixel-domain bit-depth
        quantization** -- collapsing each channel's 256 levels onto a coarse
        grid of `2 ** target_bits` levels. This produces visually similar
        "staircase" contours in flat regions while being trivially fast and
        differentiable-friendly.

        What real codecs do that we do NOT model (left as future work):
          * Smooth-region detection (codecs allocate more bits to smooth
            areas to suppress banding).
          * Dithering (some encoders inject low-amplitude noise to break up
            the steps).
          * Block-level adaptive quantization (different bits per block).
        These extensions could be added later if model robustness on
        real-decoded content turns out to need them.

        target_bits range:
          6 -- mild  (64 levels per channel; barely visible in mid-tones)
          5 -- moderate (32 levels; clear staircase in sky / gradients)
          4 -- heavy (16 levels; obvious posterization across the frame)

        Quantization formula uses a step of ``255 / (levels - 1)`` so that
        both endpoints (0 and 255) are preserved -- no DC shift / brightness
        bias, unlike naive ``image // step * step`` which clips the maximum
        to 248 for 5-bit.

        Args:
            image: ``(H, W, 3)`` ``uint8`` BGR image.
            target_bits: Target bit depth per channel, in 1..8. ``None`` ->
                random pick from :attr:`BANDING_BITS_CHOICES`.
            random_state: Seed or Generator for reproducibility.
        """
        rng = _resolve_rng(random_state)
        if target_bits is None:
            target_bits = int(rng.choice(self.BANDING_BITS_CHOICES))
        if not 1 <= target_bits <= 8:
            raise ValueError(f"target_bits must be in 1..8, got {target_bits}")
        levels = 2 ** target_bits
        step = 255.0 / (levels - 1)
        quantized = np.round(image.astype(np.float32) / step) * step
        return np.clip(quantized, 0, 255).astype(np.uint8)

    def random_degradation_pipeline(
        self,
        image: np.ndarray,
        scale: int = 2,
        random_state: RandomState = None,
    ) -> np.ndarray:
        """Compose a random LR image from an HR image.

        Step order: ``blur -> downsample -> banding -> noise -> compression``.
        Each step is independently applied with probability 0.5, and applied
        steps draw their parameters at random from the class's ranges. The
        ordering reflects a plausible imaging-and-broadcast chain: blur
        (optics) -> downsample (resolution mismatch between source and
        display) -> banding (low-bitrate codec quantization) -> noise
        (sensor / transmission) -> JPEG (re-encode artifacts). Putting
        noise after banding is intentional -- noise partially dithers the
        bands, mimicking the masking effect seen in real noisy compressed
        video.

        Note that :meth:`apply_chroma_subsampling` is implemented in this
        class but **deliberately excluded from the random pipeline**.
        Reasoning: on natural-image training data (DIV2K), 4:2:0 chroma
        subsampling produces ~38 dB PSNR -- the model would not learn a
        useful invariance from it because the artifact is essentially
        invisible on photographic content. The method is retained for
        analysis use (Cell 5 of the demo notebook) and could be re-enabled
        once synthetic / graphical training content (text, UI, anime) is
        added, where chroma artifacts actually bite. See
        ``docs/methodology.md`` for the full rationale.

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
            out = self.apply_banding(out, random_state=rng)
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

    chroma_a = deg.apply_chroma_subsampling(img)
    chroma_b = deg.apply_chroma_subsampling(img)
    assert np.array_equal(chroma_a, chroma_b), (
        "apply_chroma_subsampling must be deterministic."
    )

    banding_a = deg.apply_banding(img, random_state=2)
    banding_b = deg.apply_banding(img, random_state=2)
    assert np.array_equal(banding_a, banding_b), (
        "apply_banding must be reproducible for the same seed."
    )
    for bits in (4, 5, 6):
        out = deg.apply_banding(img, target_bits=bits)
        assert out.shape == img.shape and out.dtype == np.uint8, (
            f"banding target_bits={bits} broke shape/dtype"
        )

    print()
    print("Per-step shape sanity:")
    print(f"  apply_blur              : {deg.apply_blur(img, random_state=1).shape}")
    print(f"  apply_noise             : {deg.apply_noise(img, random_state=1).shape}")
    print(f"  apply_jpeg_compression  : {deg.apply_jpeg_compression(img, random_state=1).shape}")
    print(f"  apply_downsample(s=2)   : {deg.apply_downsample(img, scale=2, random_state=1).shape}")
    print(f"  apply_chroma_subsampling: {chroma_a.shape}  (deterministic: True)")
    print(f"  apply_banding(b=4)      : {deg.apply_banding(img, target_bits=4, random_state=1).shape}")
    print(f"  apply_banding(b=5)      : {deg.apply_banding(img, target_bits=5, random_state=1).shape}")
    print(f"  apply_banding(b=6)      : {deg.apply_banding(img, target_bits=6, random_state=1).shape}")
