"""
reader.py — Read guider FITS files into a simple numpy-backed container.

Supported FITS layouts (auto-detected in order):
  1. Primary HDU with 3D data [n_frames, ny, nx]
  2. Primary HDU with 2D data  → treated as a single frame
  3. Multiple ImageHDU extensions each carrying a 2D frame
  4. Any single extension carrying a 3D array

The guider name defaults to the filename stem and can be overridden by a
DETNAME keyword in the primary header.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits


@dataclass
class GuiderStamps:
    """All stamps from one guider for one observation.

    Attributes
    ----------
    stamps :
        Image data, shape ``[n_frames, ny, nx]``, dtype float32.
    filename :
        Path to the source FITS file (as a string).
    guider_name :
        Human-readable guider identifier (filename stem by default).
    header :
        Primary HDU header keywords as a plain dict.
    """

    stamps: np.ndarray
    filename: str
    guider_name: str
    header: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return self.stamps.shape[0]

    @property
    def stamp_shape(self) -> tuple[int, int]:
        """(ny, nx) of a single stamp."""
        return self.stamps.shape[1], self.stamps.shape[2]

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        return self.stamps[frame_idx]

    def __len__(self) -> int:
        return self.n_frames

    def coadd(self, n_frames: int | None = None) -> np.ndarray:
        """Return a nanmedian coadd of the first *n_frames* stamps.

        A uniform [0, 1] dither is added before stacking to prevent integer
        quantisation from collapsing the pixel distribution (matches the
        behaviour of ``GuiderData.getStampArrayCoadd`` in the LSST pipeline).
        """
        n = self.n_frames if n_frames is None else min(n_frames, self.n_frames)
        stack = self.stamps[:n].astype(np.float32).copy()
        rng = np.random.default_rng(seed=0)
        stack += rng.uniform(0.0, 1.0, size=stack.shape).astype(np.float32)
        return np.nanmedian(stack, axis=0)


def read_guider_fits(filepath: str | Path) -> GuiderStamps:
    """Read a guider FITS file and return a `GuiderStamps` object.

    Parameters
    ----------
    filepath :
        Path to the FITS file.

    Returns
    -------
    GuiderStamps

    Raises
    ------
    ValueError
        If no recognisable image data is found.
    """
    filepath = Path(filepath)

    with fits.open(filepath) as hdul:
        primary_header = dict(hdul[0].header)
        guider_name = primary_header.get("DETNAME", filepath.stem)

        # --- Strategy 1 & 2: primary HDU has image data ---
        primary_data = hdul[0].data
        if primary_data is not None:
            if primary_data.ndim == 3:
                return GuiderStamps(
                    stamps=primary_data.astype(np.float32),
                    filename=str(filepath),
                    guider_name=guider_name,
                    header=primary_header,
                )
            if primary_data.ndim == 2:
                return GuiderStamps(
                    stamps=primary_data[np.newaxis].astype(np.float32),
                    filename=str(filepath),
                    guider_name=guider_name,
                    header=primary_header,
                )

        # --- Strategy 3: collect 2D frames from image extensions ---
        frames_2d = []
        for hdu in hdul[1:]:
            if hdu.data is None:
                continue
            if isinstance(hdu, (fits.ImageHDU, fits.CompImageHDU)):
                if hdu.data.ndim == 2:
                    frames_2d.append(hdu.data.astype(np.float32))
                elif hdu.data.ndim == 3:
                    # Strategy 4: first 3D extension wins
                    return GuiderStamps(
                        stamps=hdu.data.astype(np.float32),
                        filename=str(filepath),
                        guider_name=guider_name,
                        header=primary_header,
                    )

        if frames_2d:
            return GuiderStamps(
                stamps=np.stack(frames_2d, axis=0),
                filename=str(filepath),
                guider_name=guider_name,
                header=primary_header,
            )

    raise ValueError(
        f"Could not find image data in '{filepath}'. "
        "Expected a 3-D primary array [n_frames, ny, nx], a 2-D primary array, "
        "or multiple 2-D ImageHDU extensions."
    )


def read_all_guiders(file_list: list[str | Path]) -> list[GuiderStamps]:
    """Read a list of guider FITS files and return one `GuiderStamps` per file."""
    return [read_guider_fits(f) for f in file_list]


def read_guider_butler(
    exposure: str,
    sensors: list[str],
    repo: str = "embargo",
    collections: list[str] | None = None,
) -> list[GuiderStamps]:
    """Read guider FITS files via the LSST butler.

    Parameters
    ----------
    exposure :
        Exposure ID, e.g. ``'MC_O_20260513_000005'``.
    sensors :
        List of sensor names, e.g. ``['R00_SG0', 'R00_SG1']``.
    repo :
        Butler repository name.
    collections :
        Butler collections to search. Defaults to
        ``['LSSTCam/raw/all', 'LSSTCam/raw/guider']``.

    Returns
    -------
    list[GuiderStamps]
    """
    from lsst.daf.butler import Butler
    from lsst.resources import ResourcePath

    if collections is None:
        collections = ["LSSTCam/raw/all", "LSSTCam/raw/guider"]

    butler = Butler(repo, collections=collections)
    results = []

    for sensor in sensors:
        records = list(butler.registry.queryDimensionRecords(
            "detector",
            instrument="LSSTCam",
            where=f"detector.full_name='{sensor}'",
        ))
        if not records:
            raise ValueError(f"No detector found for sensor '{sensor}'")
        detector_id = records[0].id

        uri = butler.getURI(
            "guider_raw",
            instrument="LSSTCam",
            detector=detector_id,
            exposure=exposure,
        )
        path = uri.geturl()
        rb = ResourcePath(path)

        with rb.open(mode="rb") as f:
            with fits.open(f) as hdul:
                primary_header = dict(hdul[0].header)
                guider_name = primary_header.get("DETNAME", sensor)

                primary_data = hdul[0].data
                if primary_data is not None:
                    if primary_data.ndim == 3:
                        stamps = primary_data.astype(np.float32)
                    elif primary_data.ndim == 2:
                        stamps = primary_data[np.newaxis].astype(np.float32)
                    else:
                        stamps = None
                else:
                    stamps = None

                if stamps is None:
                    frames_2d = []
                    for hdu in hdul[1:]:
                        if hdu.data is None:
                            continue
                        if isinstance(hdu, (fits.ImageHDU, fits.CompImageHDU)):
                            if hdu.data.ndim == 2:
                                frames_2d.append(hdu.data.astype(np.float32))
                            elif hdu.data.ndim == 3:
                                stamps = hdu.data.astype(np.float32)
                                break
                    if stamps is None and frames_2d:
                        stamps = np.stack(frames_2d, axis=0)

                if stamps is None:
                    raise ValueError(
                        f"No image data found for sensor '{sensor}' "
                        f"in exposure '{exposure}'"
                    )

                results.append(GuiderStamps(
                    stamps=stamps,
                    filename=path,
                    guider_name=guider_name,
                    header=primary_header,
                ))

    return results
