"""
centroid.py — Standalone guider centroid finding.

Faithfully reproduces the algorithm from
``lsst.summit.utils.guiders.detection`` with no LSST stack dependency.
The only external requirements are numpy, scipy, astropy, pandas, and
(optionally) galsim.  If galsim is not importable a pure-numpy adaptive-
moments fallback is used automatically.

Two star-finding strategies are provided via ``Config.single_frame_mode``:

  False (default)
      Coadd the first ``n_seed_frames`` frames to build a high-SNR reference
      image, then detect sources on that coadd.  Matches the original
      ``buildReferenceCatalog`` / ``runSourceDetection`` pipeline.

  True  (real-time path)
      Use only frame 0 for source detection.  Noisier, but requires no
      accumulated history — suitable for the first stamp of a new guide cycle.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats

if TYPE_CHECKING:
    from reader import GuiderStamps

# Try to import galsim once; fall back gracefully.
try:
    import galsim

    _GALSIM_AVAILABLE = True
except ImportError:
    _GALSIM_AVAILABLE = False
    warnings.warn(
        "galsim not found — using pure-numpy adaptive moments fallback. "
        "Centroid precision may be slightly lower.",
        ImportWarning,
        stacklevel=2,
    )

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Config:
    """Algorithm parameters — mirrors ``GuiderStarTrackerConfig`` plus extras.

    Parameters
    ----------
    min_snr :
        Minimum signal-to-noise ratio to accept a measurement.
    edge_margin :
        Pixels to exclude near the stamp edge.
    max_ellipticity :
        Maximum allowed |e| = sqrt(e1²+e2²) for a valid measurement.
    cutout_size :
        Side length (pixels) of the cutout passed to the HSM fitter.
    aper_size_px :
        Aperture radius (pixels) used for flux / SNR measurement.
    gain :
        Detector gain in e⁻/ADU.
    n_seed_frames :
        Number of frames to coadd when building the reference image
        (ignored in single_frame_mode).
    detection_threshold :
        Source-detection threshold in units of the image noise sigma.
    n_pix_min :
        Minimum connected-pixel count for a detection to be kept.
    single_frame_mode :
        If True, use only frame 0 for reference-position finding.
        Set this for real-time operation.
    """

    min_snr: float = 10.0
    edge_margin: int = 5
    max_ellipticity: float = 0.7
    cutout_size: int = 50
    aper_size_px: float = 10.0
    gain: float = 1.0
    n_seed_frames: int = 5
    detection_threshold: float = 10.0
    n_pix_min: int = 10
    single_frame_mode: bool = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class StarMeasurement:
    """Per-frame centroid and shape measurement for one star."""

    x: float = field(default=np.nan)
    y: float = field(default=np.nan)
    xerr: float = field(default=0.0)
    yerr: float = field(default=0.0)
    e1: float = field(default=np.nan)
    e2: float = field(default=np.nan)
    ixx: float = field(default=np.nan)
    iyy: float = field(default=np.nan)
    ixy: float = field(default=np.nan)
    fwhm: float = field(default=np.nan)
    flux: float = field(default=np.nan)
    flux_err: float = field(default=0.0)
    snr: float = field(default=0.0)
    frame: int = field(default=-1)
    guider: str = field(default="")

    @property
    def is_valid(self) -> bool:
        return np.isfinite(self.x) and np.isfinite(self.y)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Background subtraction  (exact copy of detection.py:annulusBackgroundSubtraction)
# ---------------------------------------------------------------------------


def annulus_background_subtraction(
    data: np.ndarray, annulus: tuple[float, float]
) -> tuple[np.ndarray, float]:
    """Subtract background estimated from a circular annulus.

    Parameters
    ----------
    data :
        2-D image cutout.
    annulus :
        ``(inner_radius_px, outer_radius_px)`` defining the background annulus.

    Returns
    -------
    data_bkg_sub :
        Background-subtracted image.
    bkg_std :
        Robust standard deviation of the background.
    """
    rin, rout = annulus
    x0, y0 = data.shape[1] // 2, data.shape[0] // 2
    x, y = np.indices(data.shape)
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    ann_mask = (r2 >= rin**2) & (r2 <= rout**2) & np.isfinite(data)
    if not np.any(ann_mask):
        return data, 1.0
    _, bkg_med, bkg_std = sigma_clipped_stats(data[ann_mask], sigma=3.0)
    return data - bkg_med, float(bkg_std)


# ---------------------------------------------------------------------------
# Pure-numpy adaptive moments fallback
# ---------------------------------------------------------------------------


def _adaptive_moments_numpy(image: np.ndarray, max_iter: int = 50, tol: float = 1e-6) -> dict | None:
    """Iterative weighted second-moment centroid (Hirata & Seljak 2003 style).

    Returns a dict with keys matching the GalSim HSM output used by
    ``run_galsim``, or None on failure.
    """
    ny, nx = image.shape
    yi, xi = np.indices((ny, nx), dtype=float)

    total = np.nansum(image)
    if total <= 0:
        return None

    x0 = np.nansum(image * xi) / total
    y0 = np.nansum(image * yi) / total
    sigma = max(1.5, min(ny, nx) / 6.0)

    for _ in range(max_iter):
        dx = xi - x0
        dy = yi - y0
        W = np.exp(-0.5 * (dx**2 + dy**2) / sigma**2)
        WI = W * image
        norm = np.nansum(WI)
        if norm <= 0:
            return None

        x0_new = np.nansum(WI * xi) / norm
        y0_new = np.nansum(WI * yi) / norm

        dx = xi - x0_new
        dy = yi - y0_new
        Mxx = np.nansum(WI * dx**2) / norm
        Myy = np.nansum(WI * dy**2) / norm
        Mxy = np.nansum(WI * dx * dy) / norm

        det = Mxx * Myy - Mxy**2
        sigma_new = max(0.5, det**0.25) if det > 0 else sigma

        converged = (
            abs(x0_new - x0) < tol
            and abs(y0_new - y0) < tol
            and abs(sigma_new - sigma) < tol
        )
        x0, y0, sigma = x0_new, y0_new, sigma_new
        if converged:
            break

    denom = Mxx + Myy + 1e-30
    e1 = (Mxx - Myy) / denom
    e2 = 2.0 * Mxy / denom
    fwhm = 2.355 * sigma

    return dict(x=x0, y=y0, sigma=sigma, fwhm=fwhm, e1=e1, e2=e2,
                ixx=Mxx, iyy=Myy, ixy=Mxy, flux=norm)


# ---------------------------------------------------------------------------
# GalSim HSM wrapper  (mirrors detection.py:runGalSim)
# ---------------------------------------------------------------------------


def _make_elliptical_gaussian(
    shape: tuple[int, int],
    flux: float,
    sigma: float,
    e1: float,
    e2: float,
    center: tuple[float, float],
) -> np.ndarray:
    """Render an elliptical Gaussian model image (mirrors detection.py helper)."""
    y, x = np.indices(shape)
    x0, y0 = center
    u, v = x - x0, y - y0
    ixx = sigma**2 * (1 + e1)
    iyy = sigma**2 * (1 - e1)
    ixy = sigma**2 * e2
    det = ixx * iyy - ixy**2
    if det <= 0:
        det = 1e-30
    inv_ixx = iyy / det
    inv_iyy = ixx / det
    inv_ixy = -ixy / det
    r2 = inv_ixx * u**2 + inv_iyy * v**2 + 2 * inv_ixy * u * v
    e = np.sqrt(e1**2 + e2**2)
    norm = flux / (2 * np.pi * sigma**2 * np.sqrt(max(1 - e**2, 1e-10)))
    return norm * np.exp(-0.5 * r2)


def _calc_galsim_error(
    image_array: np.ndarray,
    x0: float,
    y0: float,
    sigma: float,
    e1: float,
    e2: float,
    flux: float,
    gain: float,
    bkg_std: float,
) -> tuple[float, float]:
    """Propagate noise to centroid errors (mirrors detection.py:calcGalsimError)."""
    kernel = _make_elliptical_gaussian(image_array.shape, 1.0, sigma, e1, e2, (x0, y0))
    weight = 1.0 / (bkg_std**2 + np.abs(flux * kernel / gain) + 1e-30)
    mask = weight == 0.0
    u = np.arange(image_array.shape[1], dtype=float) - x0
    v = np.arange(image_array.shape[0], dtype=float) - y0
    uu, vv = np.meshgrid(u, v)
    WI = kernel * image_array
    M00 = np.nansum(WI)
    if M00 == 0:
        return 0.0, 0.0
    WV = kernel**2 / (weight + 1e-30) / M00**2
    WV[mask] = 0.0
    var_x = 4.0 * np.nansum(WV * uu**2)
    var_y = 4.0 * np.nansum(WV * vv**2)
    return float(np.sqrt(max(var_x, 0))), float(np.sqrt(max(var_y, 0)))


def run_galsim(
    image_array: np.ndarray,
    gain: float = 1.0,
    bkg_std: float = 0.0,
) -> StarMeasurement:
    """Measure centroid and shape using GalSim HSM (or numpy fallback).

    Mirrors ``detection.py:runGalSim`` exactly.
    """
    if _GALSIM_AVAILABLE:
        gs_img = galsim.Image(image_array.astype(np.float64))
        hsm = galsim.hsm.FindAdaptiveMom(gs_img, strict=False)
        if hsm.error_message != "":
            return StarMeasurement()
        x0 = hsm.moments_centroid.x
        y0 = hsm.moments_centroid.y
        sigma = hsm.moments_sigma
        e1 = hsm.observed_shape.e1
        e2 = hsm.observed_shape.e2
        flux = hsm.moments_amp
    else:
        res = _adaptive_moments_numpy(image_array)
        if res is None:
            return StarMeasurement()
        x0, y0, sigma = res["x"], res["y"], res["sigma"]
        e1, e2, flux = res["e1"], res["e2"], res["flux"]

    fwhm = 2.355 * sigma
    x_err, y_err = _calc_galsim_error(image_array, x0, y0, sigma, e1, e2, flux, gain, bkg_std)

    ellipticity = np.sqrt(e1**2 + e2**2)
    n_eff = 2 * np.pi * sigma**2 * np.sqrt(max(1 - ellipticity**2, 1e-10))
    shot_noise = np.sqrt(n_eff * bkg_std**2)
    flux_err = np.sqrt(flux / gain + shot_noise**2)
    snr = flux / (shot_noise + 1e-9) if shot_noise > 0 else 0.0

    return StarMeasurement(
        x=x0, y=y0,
        xerr=x_err, yerr=y_err,
        e1=e1, e2=e2,
        ixx=sigma**2 * (1 + e1),
        iyy=sigma**2 * (1 - e1),
        ixy=sigma**2 * e2,
        fwhm=fwhm,
        flux=flux,
        flux_err=flux_err,
        snr=snr,
    )


# ---------------------------------------------------------------------------
# Per-stamp measurement  (mirrors detection.py:measureStarOnStamp)
# ---------------------------------------------------------------------------


def measure_star_on_stamp(
    stamp: np.ndarray,
    ref_center: tuple[float, float],
    cutout_size: int,
    aper_radius: float,
    gain: float = 1.0,
) -> StarMeasurement:
    """Measure one star on one stamp frame.

    Extracts a cutout centred on *ref_center*, subtracts background using a
    surrounding annulus, runs the HSM fitter, and performs aperture photometry.
    The returned centroid is in full-stamp pixel coordinates.

    Parameters
    ----------
    stamp :
        2-D image array (one guider frame).
    ref_center :
        ``(x, y)`` pixel position to centre the cutout on.
    cutout_size :
        Side length of the square cutout in pixels.
    aper_radius :
        Aperture radius in pixels for photometry.
    gain :
        Detector gain (e⁻/ADU).
    """
    rx, ry = ref_center
    cutout = Cutout2D(stamp, (rx, ry), size=cutout_size, mode="partial", fill_value=np.nan)
    data = cutout.data

    if np.all(data == 0) or not np.isfinite(data).all():
        return StarMeasurement()

    annulus = (aper_radius, aper_radius * 2.0)
    data_bkg, bkg_std = annulus_background_subtraction(data, annulus)

    star = run_galsim(data_bkg, gain=gain, bkg_std=bkg_std)
    if not star.is_valid:
        return StarMeasurement()

    # Aperture photometry (mirrors StarMeasurement.runAperturePhotometry)
    ny, nx = data_bkg.shape
    yy, xx = np.indices((ny, nx))
    aper_mask = (xx - star.x) ** 2 + (yy - star.y) ** 2 <= aper_radius**2
    flux_net = float(np.nansum(data_bkg[aper_mask]))
    flux_net = max(flux_net, 0.0)
    n_pix = int(aper_mask.sum())
    flux_err = np.sqrt(flux_net / gain + n_pix * bkg_std**2)
    snr = flux_net / (flux_err + 1e-9) if flux_err > 0 else 0.0
    star.flux = flux_net
    star.flux_err = float(flux_err)
    star.snr = float(snr)

    # Convert centroid back to full-stamp coordinates
    star.x += cutout.xmin_original
    star.y += cutout.ymin_original
    return star


# ---------------------------------------------------------------------------
# Single-frame or coadd-based reference-position finding
# (replaces buildReferenceCatalog / runSourceDetection)
# ---------------------------------------------------------------------------


def _is_blank(image: np.ndarray, flux_min: float = 300.0) -> bool:
    """Return True if no pixel deviates from the median by more than *flux_min*."""
    med = np.nanmedian(image)
    return not np.any(np.abs(image - med) > flux_min)


def find_reference_position(
    stamps_3d: np.ndarray,
    config: Config = Config(),
) -> tuple[float, float] | None:
    """Find the guide-star position to use as the fixed tracking reference.

    Replaces the LSST ``buildReferenceCatalog`` / ``runSourceDetection``
    pipeline using only scipy + astropy.

    Parameters
    ----------
    stamps_3d :
        All frames for one guider, shape ``[n_frames, ny, nx]``.
    config :
        Algorithm configuration.

    Returns
    -------
    ``(x, y)`` pixel position of the brightest detected source, or ``None``
    if no source passes the detection criteria.
    """
    if config.single_frame_mode:
        seed_image = stamps_3d[0].astype(np.float32)
    else:
        n = min(config.n_seed_frames, len(stamps_3d))
        stack = stamps_3d[:n].astype(np.float32).copy()
        rng = np.random.default_rng(seed=0)
        stack += rng.uniform(0.0, 1.0, size=stack.shape).astype(np.float32)
        seed_image = np.nanmedian(stack, axis=0)

    if _is_blank(seed_image):
        log.warning("Seed image appears blank — no source detected.")
        return None

    # Background subtraction
    _, bkg_med, _ = sigma_clipped_stats(seed_image, sigma=3.0)
    bkg_sub = seed_image - bkg_med

    # Noise estimate: MAD-based sigma, with percentile fallback
    try:
        from scipy.stats import median_abs_deviation
        noise = float(median_abs_deviation(bkg_sub.ravel(), scale="normal", nan_policy="omit"))
    except Exception:
        p16, p84 = np.nanpercentile(bkg_sub, [16.0, 84.0])
        noise = (p84 - p16) / 2.0

    if noise <= 0:
        log.warning("Noise estimate is zero — cannot threshold image.")
        return None

    # Threshold and label connected components
    thresh_mask = bkg_sub > config.detection_threshold * noise
    labeled, n_components = ndi.label(thresh_mask)
    if n_components == 0:
        log.warning("No sources above threshold in seed image.")
        return None

    ny, nx = seed_image.shape
    best_peak = -np.inf
    best_xy: tuple[float, float] | None = None

    for label_id in range(1, n_components + 1):
        obj_mask = labeled == label_id
        n_pix = obj_mask.sum()
        if n_pix < config.n_pix_min:
            continue

        # Reject elongated artifacts (column/row stripes): require the
        # component's bounding box to be reasonably compact.
        ys, xs = np.where(obj_mask)
        bbox_h = ys.max() - ys.min() + 1
        bbox_w = xs.max() - xs.min() + 1
        aspect = max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1)
        if aspect > 5.0:
            continue

        # Centroid of component
        cy, cx = ndi.center_of_mass(bkg_sub * obj_mask)
        # Edge margin check
        if (cx < config.edge_margin or cx > nx - config.edge_margin
                or cy < config.edge_margin or cy > ny - config.edge_margin):
            continue

        # Use peak pixel value to select the star — a point source has a
        # much higher peak than diffuse artifacts with more total flux.
        peak = float(np.nanmax(bkg_sub[obj_mask]))
        if peak > best_peak:
            best_peak = peak
            best_xy = (float(cx), float(cy))

    if best_xy is None:
        log.warning("All detected components failed edge or size cuts.")
    return best_xy


# ---------------------------------------------------------------------------
# Multi-frame tracking loop
# ---------------------------------------------------------------------------


def track_star(
    stamps_3d: np.ndarray,
    ref_center: tuple[float, float],
    config: Config = Config(),
    guider_name: str = "",
) -> pd.DataFrame:
    """Measure the star at *ref_center* across every frame.

    Parameters
    ----------
    stamps_3d :
        Shape ``[n_frames, ny, nx]``.
    ref_center :
        ``(x, y)`` reference position in stamp pixel coordinates.
    config :
        Algorithm configuration.
    guider_name :
        Label added to every row of the output DataFrame.

    Returns
    -------
    DataFrame with one row per frame that passes quality cuts.
    """
    rows = []
    n_frames, ny, nx = stamps_3d.shape

    # Bounds check
    rx, ry = ref_center
    if not (0 <= rx < nx and 0 <= ry < ny):
        log.warning("ref_center %s is outside stamp bounds (%d, %d)", ref_center, nx, ny)
        return pd.DataFrame()

    for i in range(n_frames):
        m = measure_star_on_stamp(
            stamps_3d[i],
            ref_center,
            config.cutout_size,
            config.aper_size_px,
            config.gain,
        )
        if not m.is_valid:
            continue
        m.frame = i
        m.guider = guider_name
        rows.append(m.to_dict())

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Quality cuts (mirrors tracking.py:applyQualityCuts)
    e_abs = np.hypot(df["e1"], df["e2"])
    mask = (
        (df["snr"] >= config.min_snr)
        & (df["flux"] > 0)
        & (e_abs <= config.max_ellipticity)
        & (df["x"] >= config.edge_margin)
        & (df["x"] <= nx - config.edge_margin)
        & (df["y"] >= config.edge_margin)
        & (df["y"] <= ny - config.edge_margin)
    )
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


def process_guider(
    guider_stamps,  # GuiderStamps from reader.py
    config: Config = Config(),
) -> tuple[tuple[float, float] | None, pd.DataFrame]:
    """Run the full pipeline for one guider.

    Returns
    -------
    ref_pos :
        ``(x, y)`` reference position used, or ``None`` if detection failed.
    catalog :
        DataFrame of per-frame measurements (empty if detection failed).
    """
    ref_pos = find_reference_position(guider_stamps.stamps, config)
    if ref_pos is None:
        log.warning("No reference position found for guider '%s'.", guider_stamps.guider_name)
        return None, pd.DataFrame()

    catalog = track_star(
        guider_stamps.stamps,
        ref_pos,
        config,
        guider_name=guider_stamps.guider_name,
    )
    return ref_pos, catalog
