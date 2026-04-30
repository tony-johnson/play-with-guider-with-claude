#!/usr/bin/env python3
"""
run_test.py — CLI test runner for the standalone guider centroid framework.

Usage
-----
    python run_test.py guider1.fits guider2.fits ... [options]
    python run_test.py /path/to/data/*.fits --single-frame --no-plot

Options
-------
  --cutout-size INT         Cutout side length in pixels          [50]
  --min-snr FLOAT           Minimum S/N to accept a measurement   [10.0]
  --max-ellipticity FLOAT   Maximum |e| = sqrt(e1²+e2²)          [0.7]
  --edge-margin INT         Pixels to exclude near stamp edge     [5]
  --aper-size FLOAT         Aperture radius for photometry (px)   [10.0]
  --n-seed-frames INT       Frames to coadd for reference image   [5]
  --single-frame            Use only frame 0 for detection
                            (real-time mode, implies n_seed_frames=1)
  --gain FLOAT              Detector gain e⁻/ADU                  [1.0]
  --output-csv PATH         Write full catalog to a CSV file
  --no-plot                 Suppress matplotlib output
  --quiet                   Suppress per-frame log output
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from centroid import Config, find_reference_position, process_guider
from reader import read_guider_fits


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_centroid_strips(results: list[dict], config: Config) -> None:
    """Figure 1: centroid dx/dy vs frame for each guider (8-row strip plot)."""
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(12, max(2 * n, 6)), sharex=False)
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Centroid residuals (dx, dy) per guider frame", fontsize=13)

    for row_idx, res in enumerate(results):
        guider = res["guider_name"]
        cat = res["catalog"]
        ax_x, ax_y = axes[row_idx, 0], axes[row_idx, 1]

        if cat.empty:
            for ax in (ax_x, ax_y):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_ylabel(guider, fontsize=8)
            continue

        frames = cat["frame"].to_numpy()
        dx = cat["x"].to_numpy() - np.nanmedian(cat["x"])
        dy = cat["y"].to_numpy() - np.nanmedian(cat["y"])
        snr_med = float(np.nanmedian(cat["snr"]))

        label = f"{guider}  SNR={snr_med:.1f}"

        ax_x.scatter(frames, dx, s=6, color="steelblue", alpha=0.7)
        ax_x.axhline(0, color="k", lw=0.5, ls="--")
        ax_x.set_ylabel(label, fontsize=7)
        ax_x.tick_params(labelsize=7)

        ax_y.scatter(frames, dy, s=6, color="tomato", alpha=0.7)
        ax_y.axhline(0, color="k", lw=0.5, ls="--")
        ax_y.tick_params(labelsize=7)

        if row_idx == 0:
            ax_x.set_title("dx (px)", fontsize=9)
            ax_y.set_title("dy (px)", fontsize=9)
        if row_idx == n - 1:
            ax_x.set_xlabel("Frame index", fontsize=8)
            ax_y.set_xlabel("Frame index", fontsize=8)

    fig.tight_layout()


def _zscale(image: np.ndarray) -> tuple[float, float]:
    """Compute ZScale display limits (robust to columns and cosmic rays)."""
    try:
        from astropy.visualization import ZScaleInterval
        vmin, vmax = ZScaleInterval().get_limits(image)
    except Exception:
        # Fallback: sigma-clipped percentiles ignoring outlier columns
        flat = image.ravel()
        flat = flat[np.isfinite(flat)]
        med = np.median(flat)
        mad = np.median(np.abs(flat - med))
        vmin = float(med - 3 * mad * 1.4826)
        vmax = float(med + 5 * mad * 1.4826)
    return vmin, vmax


def plot_reference_mosaic(results: list[dict], config: Config) -> None:
    """Figure 2: reference coadd image per guider with detection overlay."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n = len(results)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.array(axes).ravel()

    fig.suptitle("Reference coadd images with detected star position", fontsize=13)

    for idx, res in enumerate(results):
        ax = axes_flat[idx]
        guider = res["guider_name"]
        ref_pos = res["ref_pos"]
        ref_image = res["ref_image"]

        vmin, vmax = _zscale(ref_image)
        ax.imshow(ref_image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
                  aspect="equal", interpolation="nearest")

        if ref_pos is not None:
            rx, ry = ref_pos
            # Green crosshair at reference position
            ax.axhline(ry, color="lime", lw=0.8, alpha=0.7)
            ax.axvline(rx, color="lime", lw=0.8, alpha=0.7)
            # Red circle at aperture radius
            circle = mpatches.Circle(
                (rx, ry), radius=config.aper_size_px,
                edgecolor="red", facecolor="none", lw=1.2,
            )
            ax.add_patch(circle)
            ax.set_title(f"{guider}\nref=({rx:.1f}, {ry:.1f})", fontsize=8)
        else:
            ax.set_title(f"{guider}\nNO DETECTION", fontsize=8, color="red")

        ax.tick_params(labelsize=6)

    # Hide unused panels
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(results: list[dict], total_wall_sec: float) -> None:
    header = (
        f"{'Guider':<20} {'Ref (x,y)':<22} {'N good/total':<15}"
        f"{'Med SNR':>9} {'Med FWHM':>10} {'Time (ms)':>11}"
    )
    print()
    print(header)
    print("-" * len(header))

    total_frames = 0
    for res in results:
        g = res["guider_name"]
        cat = res["catalog"]
        ref = res["ref_pos"]
        ms = res["elapsed_ms"]
        n_total = res["n_frames_total"]
        n_good = len(cat)
        total_frames += n_good

        ref_str = f"({ref[0]:.1f}, {ref[1]:.1f})" if ref is not None else "NONE"
        snr_str = f"{np.nanmedian(cat['snr']):.1f}" if not cat.empty else "—"
        fwhm_str = f"{np.nanmedian(cat['fwhm']):.2f}" if not cat.empty else "—"

        print(
            f"{g:<20} {ref_str:<22} {n_good}/{n_total:<12}"
            f"{snr_str:>9} {fwhm_str:>10} {ms:>10.1f}"
        )

    print()
    n_guiders = len(results)
    if n_guiders > 0:
        mean_ms = sum(r["elapsed_ms"] for r in results) / n_guiders
        fps = total_frames / total_wall_sec if total_wall_sec > 0 else 0
        print(f"Mean processing time per guider: {mean_ms:.1f} ms")
        print(f"Total frames processed:          {total_frames}")
        print(f"Throughput:                      {fps:.0f} frames/s")
        if mean_ms > 100:
            print("WARNING: mean time exceeds 100 ms real-time budget.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone guider centroid test framework.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("fits_files", nargs="+", metavar="FITS_FILE",
                   help="One FITS file per guider.")
    p.add_argument("--cutout-size", type=int, default=50)
    p.add_argument("--min-snr", type=float, default=10.0)
    p.add_argument("--max-ellipticity", type=float, default=0.7)
    p.add_argument("--edge-margin", type=int, default=5)
    p.add_argument("--aper-size", type=float, default=10.0,
                   help="Aperture radius in pixels.")
    p.add_argument("--n-seed-frames", type=int, default=5)
    p.add_argument("--single-frame", action="store_true",
                   help="Use only frame 0 for reference detection (real-time mode).")
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = Config(
        cutout_size=args.cutout_size,
        min_snr=args.min_snr,
        max_ellipticity=args.max_ellipticity,
        edge_margin=args.edge_margin,
        aper_size_px=args.aper_size,
        gain=args.gain,
        n_seed_frames=1 if args.single_frame else args.n_seed_frames,
        single_frame_mode=args.single_frame,
    )

    results = []
    all_catalogs = []
    wall_t0 = time.perf_counter()

    for fits_path in args.fits_files:
        fits_path = Path(fits_path)
        if not fits_path.exists():
            logging.error("File not found: %s", fits_path)
            continue

        # Read
        try:
            guider = read_guider_fits(fits_path)
        except Exception as exc:
            logging.error("Failed to read %s: %s", fits_path, exc)
            continue

        # Build reference image for plotting (before timing the algorithm)
        n_seed = 1 if args.single_frame else config.n_seed_frames
        ref_image = guider.coadd(n_seed)

        # Time the detection + tracking
        t0 = time.perf_counter()
        ref_pos, catalog = process_guider(guider, config)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not args.quiet:
            status = "OK" if ref_pos is not None else "FAILED"
            n_good = len(catalog)
            logging.info(
                "%-20s  %s  ref=%s  %d/%d frames  %.1f ms",
                guider.guider_name, status,
                f"({ref_pos[0]:.1f},{ref_pos[1]:.1f})" if ref_pos else "None",
                n_good, guider.n_frames, elapsed_ms,
            )

        results.append(dict(
            guider_name=guider.guider_name,
            ref_pos=ref_pos,
            ref_image=ref_image,
            catalog=catalog,
            n_frames_total=guider.n_frames,
            elapsed_ms=elapsed_ms,
        ))

        if not catalog.empty:
            all_catalogs.append(catalog)

    wall_elapsed = time.perf_counter() - wall_t0

    if not results:
        print("No guiders processed successfully.", file=sys.stderr)
        return 1

    print_summary(results, wall_elapsed)

    # CSV output
    if args.output_csv and all_catalogs:
        combined = pd.concat(all_catalogs, ignore_index=True)
        combined.to_csv(args.output_csv, index=False)
        print(f"\nCatalog written to {args.output_csv}  ({len(combined)} rows)")

    # Plots
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("TkAgg" if sys.stdout.isatty() else "Agg")
            import matplotlib.pyplot as plt

            plot_centroid_strips(results, config)
            plot_reference_mosaic(results, config)
            plt.show()
        except ImportError:
            logging.warning("matplotlib not available — skipping plots.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
