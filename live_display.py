"""
live_display.py — Bokeh server app for real-time guider centroid display.

Usage
-----
    bokeh serve live_display.py --args guider1.fits guider2.fits
    bokeh serve live_display.py --args --butler MC_O_20260513_000005

Then open http://localhost:5006/live_display in your browser.

The app displays all 8 guide sensors simultaneously in a 2×4 grid,
processing one frame at a time via a periodic callback.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    Div,
    Span,
    LinearColorMapper,
)
from bokeh.plotting import curdoc, figure

from centroid import Config, find_reference_position, measure_star_on_stamp
from reader import read_guider_fits, read_guider_butler, GuiderStamps


# ---------------------------------------------------------------------------
# Parse CLI arguments (passed after --args)
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Live guider centroid display")
    p.add_argument("fits_files", nargs="*", metavar="FITS_FILE")
    p.add_argument("--butler", metavar="EXPOSURE_ID")
    p.add_argument("--sensors", nargs="+", default=[
        "R00_SG0", "R00_SG1", "R04_SG0", "R04_SG1",
        "R40_SG0", "R40_SG1", "R44_SG0", "R44_SG1",
    ])
    p.add_argument("--butler-repo", default="embargo")
    p.add_argument("--butler-collections", nargs="+",
                   default=["LSSTCam/raw/all", "LSSTCam/raw/guider"])
    p.add_argument("--cutout-size", type=int, default=50)
    p.add_argument("--min-snr", type=float, default=10.0)
    p.add_argument("--max-ellipticity", type=float, default=0.7)
    p.add_argument("--edge-margin", type=int, default=5)
    p.add_argument("--aper-size", type=float, default=10.0)
    p.add_argument("--n-seed-frames", type=int, default=5)
    p.add_argument("--single-frame", action="store_true")
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--zoom-size", type=int, default=50,
                   help="Size of zoomed region around star (pixels).")
    p.add_argument("--interval", type=int, default=200,
                   help="Update interval in milliseconds.")
    return p.parse_args(sys.argv[1:])


args = parse_args()

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


# ---------------------------------------------------------------------------
# Load guider data
# ---------------------------------------------------------------------------


def load_guiders() -> list[GuiderStamps]:
    if args.butler:
        return read_guider_butler(
            exposure=args.butler,
            sensors=args.sensors,
            repo=args.butler_repo,
            collections=args.butler_collections,
        )
    guiders = []
    for path in args.fits_files:
        p = Path(path)
        if p.exists():
            guiders.append(read_guider_fits(p))
    return guiders


guiders = load_guiders()
if not guiders:
    raise SystemExit("No guider data loaded. Provide FITS files or --butler.")


# ---------------------------------------------------------------------------
# State: per-guider tracking
# ---------------------------------------------------------------------------


class GuiderState:
    def __init__(self, guider: GuiderStamps, cfg: Config):
        self.guider = guider
        self.config = cfg
        self.current_frame = 0
        self.n_good = 0
        self.ref_pos: tuple[float, float] | None = None
        self.ref_image: np.ndarray | None = None
        self._detect_reference()

    def _detect_reference(self):
        self.ref_pos = find_reference_position(self.guider.stamps, self.config)
        n_seed = 1 if self.config.single_frame_mode else self.config.n_seed_frames
        self.ref_image = self.guider.coadd(n_seed)

    @property
    def done(self) -> bool:
        return self.current_frame >= self.guider.n_frames

    def next_measurement(self) -> dict | None:
        if self.done or self.ref_pos is None:
            return None
        stamp = self.guider.stamps[self.current_frame]
        m = measure_star_on_stamp(
            stamp, self.ref_pos,
            self.config.cutout_size, self.config.aper_size_px, self.config.gain,
        )
        frame_idx = self.current_frame
        self.current_frame += 1
        if not m.is_valid:
            return None
        return dict(
            frame=frame_idx,
            x=m.x, y=m.y,
            dx=m.x - self.ref_pos[0],
            dy=m.y - self.ref_pos[1],
            snr=m.snr,
            fwhm=m.fwhm,
            flux=m.flux,
        )


states = [GuiderState(g, config) for g in guiders]


# ---------------------------------------------------------------------------
# Per-sensor Bokeh data sources and figures
# ---------------------------------------------------------------------------

STAMP_SIZE = 200
PLOT_WIDTH = 250
PLOT_HEIGHT = 100

sensor_sources = []
p_stamps = []
p_dxs = []
p_dys = []
color_mappers = []

for i, state in enumerate(states):
    ny, nx = state.guider.stamp_shape

    src = dict(
        dx=ColumnDataSource(data=dict(frame=[], dx=[])),
        dy=ColumnDataSource(data=dict(frame=[], dy=[])),
        trail=ColumnDataSource(data=dict(x=[], y=[])),
        image=ColumnDataSource(data=dict(image=[])),
        crosshair=ColumnDataSource(data=dict(x=[], y=[])),
    )
    sensor_sources.append(src)

    ref_img = state.ref_image if state.ref_image is not None else np.zeros((ny, nx))
    mapper = LinearColorMapper(
        palette="Greys256",
        low=float(np.nanpercentile(ref_img, 5)),
        high=float(np.nanpercentile(ref_img, 99)),
    )
    color_mappers.append(mapper)
    src["image"].data = dict(image=[ref_img])

    # Stamp figure — zoom to region around reference star
    half = args.zoom_size / 2
    if state.ref_pos:
        cx, cy = state.ref_pos
        x_lo, x_hi = cx - half, cx + half
        y_lo, y_hi = cy - half, cy + half
    else:
        x_lo, x_hi = nx / 2 - half, nx / 2 + half
        y_lo, y_hi = ny / 2 - half, ny / 2 + half

    p_s = figure(
        title=state.guider.guider_name,
        width=STAMP_SIZE, height=STAMP_SIZE,
        x_range=(x_lo, x_hi), y_range=(y_lo, y_hi),
        match_aspect=True,
    )
    p_s.axis.visible = False
    p_s.grid.visible = False
    p_s.title.text_font_size = "9pt"
    p_s.min_border = 2
    p_s.image(
        image="image", source=src["image"],
        x=0, y=0, dw=nx, dh=ny,
        color_mapper=mapper,
    )
    p_s.scatter("x", "y", source=src["trail"], size=2, color="lime", alpha=0.5)
    p_s.scatter("x", "y", source=src["crosshair"], size=10, color="red",
                marker="cross", line_width=2)
    if state.ref_pos:
        src["crosshair"].data = dict(x=[state.ref_pos[0]], y=[state.ref_pos[1]])
    p_stamps.append(p_s)

    # dx plot
    p_d = figure(
        width=PLOT_WIDTH, height=PLOT_HEIGHT,
        x_axis_label=None, y_axis_label="dx",
    )
    p_d.scatter("frame", "dx", source=src["dx"], size=2, color="steelblue", alpha=0.7)
    p_d.add_layout(Span(location=0, dimension="width", line_dash="dashed", line_width=0.5))
    p_d.min_border = 2
    p_d.xaxis.major_label_text_font_size = "7pt"
    p_d.yaxis.major_label_text_font_size = "7pt"
    p_d.yaxis.axis_label_text_font_size = "8pt"
    p_dxs.append(p_d)

    # dy plot
    p_y = figure(
        width=PLOT_WIDTH, height=PLOT_HEIGHT,
        x_axis_label="frame", y_axis_label="dy",
    )
    p_y.scatter("frame", "dy", source=src["dy"], size=2, color="tomato", alpha=0.7)
    p_y.add_layout(Span(location=0, dimension="width", line_dash="dashed", line_width=0.5))
    p_y.min_border = 2
    p_y.xaxis.major_label_text_font_size = "7pt"
    p_y.yaxis.major_label_text_font_size = "7pt"
    p_y.xaxis.axis_label_text_font_size = "8pt"
    p_y.yaxis.axis_label_text_font_size = "8pt"
    p_dys.append(p_y)


# ---------------------------------------------------------------------------
# Periodic callback — process one frame per sensor per tick
# ---------------------------------------------------------------------------


def update():
    for i, state in enumerate(states):
        if state.done:
            continue
        frame_idx = state.current_frame
        stamp = state.guider.stamps[frame_idx] if frame_idx < state.guider.n_frames else None
        result = state.next_measurement()
        if result is None:
            continue
        state.n_good += 1
        src = sensor_sources[i]
        src["dx"].stream(dict(frame=[result["frame"]], dx=[result["dx"]]))
        src["dy"].stream(dict(frame=[result["frame"]], dy=[result["dy"]]))
        src["trail"].stream(dict(x=[result["x"]], y=[result["y"]]))
        if stamp is not None:
            mapper = color_mappers[i]
            mapper.low = float(np.nanpercentile(stamp, 5))
            mapper.high = float(np.nanpercentile(stamp, 99))
            src["image"].data = dict(image=[stamp])


# ---------------------------------------------------------------------------
# Layout: 2 columns × 4 rows
# ---------------------------------------------------------------------------

n_sensors = len(states)
n_cols = 2
grid_rows = []
for row_start in range(0, n_sensors, n_cols):
    cells = []
    for i in range(row_start, min(row_start + n_cols, n_sensors)):
        cell = row(p_stamps[i], column(p_dxs[i], p_dys[i]))
        cells.append(cell)
    grid_rows.append(row(*cells))

header = Div(text="<b>Guider Centroid Live Display</b>", styles={"font-size": "12pt"})
layout = column(header, *grid_rows)

doc = curdoc()
doc.title = "Guider Centroid Live Display"
doc.add_root(layout)
doc.add_periodic_callback(update, args.interval)
