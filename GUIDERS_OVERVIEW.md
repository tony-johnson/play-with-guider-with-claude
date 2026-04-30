# Rubin Observatory Guider Processing Module — Technical Overview

> Generated from code review of `lsst.summit.utils.guiders`  
> Date: 2026-04-30

---

## Table of Contents

1. [Purpose](#purpose)
2. [Module Structure](#module-structure)
3. [Data Flow](#data-flow)
4. [Module Descriptions](#module-descriptions)
   - [reading.py — Data Ingestion and Storage](#readingpy--data-ingestion-and-storage)
   - [transformation.py — Coordinate System Transformations](#transformationpy--coordinate-system-transformations)
   - [detection.py — Source Detection and Measurement](#detectionpy--source-detection-and-measurement)
   - [tracking.py — Multi-Frame Star Tracking](#trackingpy--multi-frame-star-tracking)
   - [metrics.py — Exposure-Level Metrics](#metricspy--exposure-level-metrics)
   - [seeing.py — Atmospheric Seeing Estimation](#seeingpy--atmospheric-seeing-estimation)
   - [plotting.py — Visualization](#plottingpy--visualization)
5. [Guide Star Identification — Detailed Walkthrough](#guide-star-identification--detailed-walkthrough)
6. [GalSim HSM Algorithm](#galsim-hsm-algorithm)
7. [Key Dependencies](#key-dependencies)

---

## Purpose

This module processes data from the Rubin Observatory's **guider cameras** — fast-readout sensors that monitor star positions during a science exposure. It reads raw guider stamp data, detects and tracks stars frame-by-frame, computes performance metrics, estimates atmospheric seeing, and produces visualizations. The outputs are used to assess tracking quality, telescope drift, and atmospheric conditions in near-real time at the summit.

---

## Module Structure

| File | Responsibility |
|---|---|
| `reading.py` | Butler data access, core data container (`GuiderData`) |
| `transformation.py` | Coordinate transforms (ROI → CCD → focal plane → Alt/Az) |
| `detection.py` | Source detection, GalSim centroid/shape measurement |
| `tracking.py` | Multi-frame star tracking, offset computation |
| `metrics.py` | Exposure-level summary metrics and drift trends |
| `seeing.py` | Atmospheric seeing estimation via centroid correlations |
| `plotting.py` | Mosaics, strip plots, and animations |

---

## Data Flow

```
Butler (raw guider data)
        │
        ▼
    reading.py
  GuiderReader → GuiderData
  (stamps, WCS, metadata, timestamps)
        │
        ▼
  transformation.py
  Coordinate transforms (ROI → CCD → focal plane → Alt/Az)
        │
        ▼
    detection.py
  buildReferenceCatalog → per-detector reference stars (from coadd)
  trackStarAcrossStamp  → per-(star, frame) centroid measurements
        │
        ▼
    tracking.py
  GuiderStarTracker → unified tracking catalog
  (adds CCD/focal/AltAz coords, offsets, quality cuts)
        │
        ├──────────────────────────┐
        ▼                          ▼
    metrics.py                  seeing.py
  GuiderMetricsBuilder       CorrelationAnalysis
  (drift rates, FWHM stats)   (r₀, tomographic seeing)
        │
        ▼
    plotting.py
  GuiderPlotter → mosaics, strip plots, animations
```

---

## Module Descriptions

### `reading.py` — Data Ingestion and Storage

**Core data model (`GuiderData`):** A Pydantic model that is the central container for a guider observation. It holds:
- Raw 3D stamp arrays per detector (shape: `[n_frames, height, width]`)
- Per-frame timestamps (two arrays: header-reported and seqNum-derived, later standardized)
- WCS objects per detector
- Visit/instrument metadata (instrument, day_obs, seq_num, exposure time, filter, rotator angle)
- Optional weather data (`WeatherInfo`: temperature, pressure, humidity, wind)

**`GuiderReader`:** Fetches data from an LSST Butler repository. Given a `day_obs` + `seq_num` (or `visit`), it:
1. Queries the butler for all guider detector datasets
2. Reads raw stamp arrays and associated metadata
3. Assembles a `GuiderData` object

**Stamp coordinate views:** Raw stamps arrive in ROI (Region of Interest) coordinates. `convertRawStampsToView()` transforms them into either DVCS (Device Coordinate System) or CCD coordinates using affine transforms, correcting for readout direction, amplifier tiling, and rotation.

**Timestamp standardization:** `standardizeGuiderTimestamps()` aligns per-frame timestamps across detectors so they share a common grid, accounting for jitter and small per-detector offsets.

**Bad column detection:** `getColumnMask()` finds columns with anomalously high or low median flux using robust statistics (median + MAD-based thresholding).

---

### `transformation.py` — Coordinate System Transformations

This is the geometric backbone of the module, implementing a chain of transforms between coordinate systems.

**Coordinate systems:**

| System | Description |
|---|---|
| ROI / stamp pixels | Raw readout pixels |
| CCD pixels | Full detector pixel coordinates |
| DVCS | Device Coordinate System — standardized detector orientation |
| Focal plane | Physical position in mm, telescope-centric |
| Alt/Az | Sky coordinates (degrees) |

**Key functions:**

- `makeRotationTransform(n)` — Returns an affine matrix for 90°×n rotation + translation to keep coordinates positive.
- `makeCcdToDvcsTransform(detector)` — Builds the CCD↔DVCS affine transform from the camera geometry, including flip/rotation.
- `makeRoiBbox(stamp_info)` — Reconstructs the ROI bounding box in CCD coordinates from stamp metadata (offset + size).
- `focalToPixel()` / `pixelToFocal()` — Linear scaling between focal-plane mm and pixel coordinates using the camera's pixel scale.
- `convertToFocalPlane(pixels, detector, bbox)` — Full pipeline: ROI pixels → CCD pixels → focal plane (mm).
- `convertToAltaz(pixels, detector, bbox, wcs)` — ROI pixels → Alt/Az sky coordinates via the WCS.
- `convertPixelsToAltaz(...)` — Vectorized batch version of the above.
- `makeInitGuiderWcs(detector, visit_info)` — Constructs an initial FITS WCS for a guider detector using the camera geometry, focal length, and visit info (rotator angle, boresight).
- `getCamRotAngle(visit_info)` — Computes the camera rotation angle from parallactic angle and rotator angle.
- `ampToCcdView(image, detector)` — Remaps amplifier-readout pixel data to the physical CCD layout.
- Atmospheric refraction correction functions to apply/remove differential atmospheric refraction to Alt/Az positions.
- Drift calculation (`DriftResult`) — fits a linear drift to star positions over time, returning slope, intercept, and residuals.

---

### `detection.py` — Source Detection and Measurement

**`StarMeasurement` dataclass:** Stores per-frame measurements for one star: centroid (x, y), flux, shape parameters (Ixx, Iyy, Ixy, e1, e2), FWHM, SNR, and error estimates.

**`GuiderStarTrackerConfig`:** Frozen dataclass holding algorithm parameters:

| Parameter | Default | Description |
|---|---|---|
| `minSnr` | 10.0 | Minimum S/N for detection |
| `minValidStampFraction` | 0.5 | Minimum fraction of frames with valid detection |
| `edgeMargin` | 5 px | Pixels to exclude near stamp edges |
| `maxEllipticity` | 0.7 | Maximum allowed ellipticity |
| `cutOutSize` | 50 px | Size of per-star cutout for tracking |
| `aperSizeArcsec` | 5.0 arcsec | Aperture radius for photometry |
| `gain` | 1.0 | Detector gain (e⁻/ADU) |

**`runSourceDetection(image)`:** Runs on the coadded stamp:
1. Wraps the numpy array in an LSST `ExposureF`
2. Subtracts the median; computes image noise via `STDEVCLIP` (with percentile fallback)
3. Calls `afwDetect.FootprintSet` to find all connected-pixel groups above the sigma threshold
4. For each footprint, calls `fp.getCentroid()` and then `measureStarOnStamp()`

**`measureStarOnStamp(stamp, refCenter, ...)`:** Core per-frame measurement:
1. Extracts a 50×50 pixel `Cutout2D` centered on `refCenter`
2. Subtracts background using a surrounding annulus (`1×` to `2×` aperture radius)
3. Calls `runGalSim()` for centroid and shape
4. Calls `runAperturePhotometry()` for flux and SNR
5. Converts centroid back to full-stamp coordinates

**`buildReferenceCatalog(guider_data)`:** For each guider detector:
1. Mean-stacks all frames into a coadded image (`getStampArrayCoadd`)
2. Runs `runSourceDetection` on the coadd
3. Sorts detected stars by SNR (brightest first)
4. Returns a combined reference catalog across all detectors

---

### `tracking.py` — Multi-Frame Star Tracking

**`GuiderStarTracker`:** Orchestrates the full tracking pipeline across all guider detectors for a single exposure.

**`trackGuiderStars(refCatalog=None)`:** Top-level entry point:
1. Optionally builds a reference catalog internally via `buildReferenceCatalog`
2. Calls `_trackStarForOneGuider` per detector
3. Assigns globally unique star IDs via `setUniqueId`

**`_trackStarForOneGuider(refCatalog, guiderName)`:** Iterative fallback logic:
1. Sorts reference stars by SNR (brightest first)
2. Calls `trackStarAcrossStamp(refCenter, ...)` to measure the star across all frames
3. Applies quality cuts (`applyQualityCuts`)
4. If ≥50% of frames pass, accepts this star and stops; otherwise falls back to the next candidate
5. Only **one** star per guider detector is ultimately tracked

**`computeOffsets(catalog)`:** Computes frame-to-frame differential offsets relative to the per-detector median position:
- `dx`, `dy` — pixel offsets in CCD coordinates
- `dxfp`, `dyfp` — offsets in focal plane coordinates (mm)
- `dalt`, `daz` — offsets in Alt/Az (arcsec), with cos(alt) correction applied to `daz`
- `magoffset` — magnitude offset from median flux (mmag)

**`applyQualityCuts(catalog, shape, config)`:** Removes measurements that are:
- Below minimum SNR or non-positive flux
- Too elliptical (`|e1|` or `|e2|` > `maxEllipticity`)
- Too close to the stamp edge (within `edgeMargin` pixels)

---

### `metrics.py` — Exposure-Level Metrics

**`GuiderMetricsBuilder`:** Computes summary metrics for an entire exposure from the tracking catalog.

**`computeTrendMetrics(catalog)`:** For each measurement column (dx, dy, dalt, daz, flux, FWHM, e1, e2, …), fits a robust linear trend vs. time using iteratively reweighted least squares. Returns slopes, intercepts, and RMS residuals.

**`GuiderDriftResult`:** Wraps trend-fit output with guider-specific field names (drift rate in Alt, drift rate in Az, FWHM trend, etc.).

**`detrendStars(catalog, trend_results)`:** Subtracts the fitted linear trend from each measurement column, yielding a residual catalog with slowly-varying systematics removed.

**`detrendFocalPlaneVariables(catalog, alt_slope, az_slope)`:** Removes Alt/Az-correlated spatial trends from focal-plane position measurements (handles field rotation and differential atmospheric effects).

**`GuiderMetricsBuilder.build()`:** Returns a dict of:
- Median FWHM, ellipticity, flux per detector
- Overall drift rates in Alt and Az
- Trend-corrected RMS scatter in position
- Star counts and frame counts

---

### `seeing.py` — Atmospheric Seeing Estimation

Estimates the **Fried parameter r₀** and **tomographic seeing** decomposition using spatial correlations of star centroid motions across detectors.

**`GuiderSeeing` dataclass:** Stores seeing estimates decomposed by atmospheric layer:
- `total` — combined seeing
- `low` — ground layer
- `mid` — boundary layer
- `high` — free atmosphere

Each layer is described by r₀ (cm) and FWHM (arcsec).

**Key physics functions:**

- `fwhmVonkarman(r0, L0)` — Computes expected FWHM from r₀ using the von Kármán turbulence spectrum (generalization of Kolmogorov, with outer scale L₀).
- `r0FromVariance(variance, baseline, L0)` — Inverts the von Kármán model to infer r₀ from measured centroid variance at a given angular baseline between detectors.

**`CorrelationAnalysis.measure()`:** Full analysis pipeline:
1. Detrend global telescope motion (common-mode subtraction)
2. Measure pairwise centroid correlations as a function of detector separation
3. Fit the Kolmogorov/von Kármán spatial correlation function to extract r₀
4. Optionally decompose into altitude layers using the angular dependence of correlations

**`measureTomographicSeeing(catalog, detector_positions)`:** Uses angular separation between detectors combined with wind direction/speed to separate contributions from atmospheric layers (ground vs. free atmosphere), returning a `GuiderSeeing` with per-layer r₀ estimates.

---

### `plotting.py` — Visualization

**`GuiderPlotter`:** Main visualization class. Takes a `GuiderData` and tracking catalog.

**`plotMosaic(frame_idx)`:** Renders all guider stamps for a single frame as a spatial mosaic positioned according to their approximate focal-plane layout, with detected stars marked.

**`stripPlot(column, time_axis)`:** Time-series strip plot of a metric (e.g., dx, FWHM) for each guider detector — one row per detector, all sharing the time axis. Overlays the robust trend line and marks outlier frames.

**`makeAnimation(output_path, fps)`:** Creates a GIF or MP4 animation cycling through all guider frames, optionally overlaying detected star positions and offset arrows.

**Drawing helpers:** `drawCrosshair()`, `drawCircle()`, `drawArrow()`, `annotateAxes()` — thin wrappers around matplotlib for consistent styling.

---

## Guide Star Identification — Detailed Walkthrough

Star identification uses a **two-stage approach**: a one-time detection on a high-SNR coadd, followed by fixed-window centroid measurement on each individual frame.

### Stage 1 — Reference Position from Coadd (`buildReferenceCatalog`)

1. `guiderData.getStampArrayCoadd(guiderName)` mean-stacks all frames for that guider into a single high-SNR image.
2. `runSourceDetection()` finds sources:
   - Subtracts the median; estimates image noise via `STDEVCLIP` (falls back to sigma68 = (p84−p16)/2 if `STDEVCLIP` returns zero due to quantization)
   - Calls `afwDetect.FootprintSet` to detect connected-pixel groups above `threshold × σ`
   - Gets the centroid of each footprint via `fp.getCentroid()`
3. Each detected source is measured with `measureStarOnStamp()` to get flux and SNR.
4. The catalog is sorted by SNR descending; the `(xroi, yroi)` of each source becomes a candidate `refCenter`.

### Stage 2 — Fixed-Window Measurement per Frame (`trackStarAcrossStamp`)

For each of the N guider frames, the code does **not** re-detect the star. Instead it:

1. Calls `getCutouts(stamp, refCenter, cutoutSize=50)` — extracts a 50×50 pixel `Cutout2D` centered on the **same `refCenter` for every frame**. There is no search radius, cross-correlation, or predicted-position update between frames.
2. Subtracts background using a surrounding annulus (inner radius = aperture radius, outer = 2× aperture radius).
3. Calls `runGalSim(dataBkgSub)` — runs GalSim's adaptive moments algorithm (`FindAdaptiveMom`) to locate the centroid within that cutout.
4. Adds the cutout origin back to recover full-stamp coordinates.

### Fallback Logic

`_trackStarForOneGuider` tries the brightest reference star first. If the resulting time-series fails quality cuts (SNR, ellipticity, edge margin, or fewer than 50% of frames surviving), it falls back to the next-brightest candidate. Only **one** star per guider detector is ultimately tracked.

### Key Assumptions and Limitations

- The guide star must remain within the 50-pixel cutout for the entire exposure. If it drifts to the edge, the `Cutout2D` fills out-of-bounds pixels with `NaN` and GalSim will fail — that frame is dropped.
- The algorithm converges to the **brightest source** in the cutout. A blend or cosmic ray entering the window produces high ellipticity and is rejected by quality cuts.

---

## GalSim HSM Algorithm

HSM stands for **Hirata, Seljak & Mandelbaum** (Hirata & Seljak 2003; Mandelbaum et al. 2005). Originally developed for weak gravitational lensing shape measurement, it is used here as a high-precision centroid and shape estimator.

### Core Idea: Adaptive Weighted Second Moments

Unweighted second moments are dominated by noise at large radii. HSM instead uses an elliptical **Gaussian weight function** `W(x, y; x₀, y₀, σ, e1, e2)` and solves for the weight parameters that are **self-consistent** with the moments they produce.

The weighted moments are:

```
M_xx = Σ W(x,y) · I(x,y) · (x−x₀)²  /  Σ W(x,y) · I(x,y)
M_yy = Σ W(x,y) · I(x,y) · (y−y₀)²  /  Σ W(x,y) · I(x,y)
M_xy = Σ W(x,y) · I(x,y) · (x−x₀)(y−y₀)  /  Σ W(x,y) · I(x,y)
```

### Iterative Procedure (`FindAdaptiveMom`)

Starting from an initial guess (circular Gaussian centered at the image center):

1. **Compute weighted centroid** using current `W`:
   ```
   x₀ = Σ W·I·x / Σ W·I
   y₀ = Σ W·I·y / Σ W·I
   ```

2. **Re-center** the weight function at `(x₀, y₀)`.

3. **Compute weighted second moments** `M_xx, M_yy, M_xy` relative to the new centroid.

4. **Update** the weight function's shape to match the measured moments:
   ```
   σ²_new = sqrt(M_xx · M_yy − M_xy²)   (geometric mean size)
   e1_new = (M_xx − M_yy) / (M_xx + M_yy)
   e2_new = 2·M_xy / (M_xx + M_yy)
   ```

5. **Repeat** from step 1 until `(x₀, y₀, σ, e1, e2)` converge (typically ~5 iterations).

### Outputs Used by This Module

| Field | Usage |
|---|---|
| `moments_centroid.x/y` | Sub-pixel centroid in the cutout |
| `moments_sigma` | Gaussian σ in pixels → FWHM = 2.355 × σ |
| `observed_shape.e1/e2` | Ellipticity components |
| `moments_amp` | Flux normalization of the fitted Gaussian |
| `error_message` | Empty on success; non-empty on convergence failure |

Derived quantities (`detection.py:432–446`):
```
fwhm = 2.355 * sigma
ixx  = σ²(1 + e1)
iyy  = σ²(1 − e1)
ixy  = σ²·e2
```

### Centroid Error Estimation (`calcGalsimError`)

The code propagates noise to centroid errors using a Fisher-information-style approach:

1. Constructs a model elliptical Gaussian `K(x,y)` using the fitted parameters.
2. Builds a per-pixel noise variance map: `σ²_pix = σ²_bkg + flux × K / gain` (background + photon noise).
3. Computes error on the first moment `M₁₀ = Σ W·I·x`:
   ```
   Var(M₁₀) ≈ 4 Σ [ (K² / σ²_pix) · (x−x₀)² ]  /  M₀₀²
   xerr = sqrt(Var(M₁₀))
   ```
4. Analogously for `yerr`.

### Why It Works Well for Guider Stars

- **Sub-pixel precision** — the iterative centroid typically converges to < 0.1 px for well-sampled stars.
- **Noise tolerance** — the matched weight suppresses background noise at large radii, unlike a simple center-of-mass.
- **Profile-agnostic** — makes no assumption about the star's PSF shape; the weight adapts to whatever profile is present.
- **`strict=False`** — convergence failure returns a non-empty `error_message` rather than raising an exception, allowing the per-frame loop to skip bad frames gracefully.

---

## Key Dependencies

| Library | Role |
|---|---|
| **LSST Science Pipelines** (`lsst.afw`, `lsst.meas`, `lsst.geom`) | Image representation, source detection footprints, WCS |
| **GalSim** | Adaptive moment measurement (HSM) |
| **Astropy** | Table/catalog management, `Cutout2D`, coordinate handling |
| **NumPy / SciPy** | Array math, robust fitting, correlation analysis |
| **Pandas** | Tabular data (tracking catalogs, reference catalogs) |
| **Matplotlib** | All plotting and animation |
| **Pydantic** | Data validation for `GuiderData` model |
