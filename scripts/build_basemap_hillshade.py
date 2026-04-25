"""Build a hillshade raster basemap layer from SRTM elevation data.

Usage:
    uv run python scripts/build_basemap_hillshade.py <region_id> \
        [<lat_min> <lon_min> <lat_max> <lon_max>]

When only ``<region_id>`` is supplied, the bbox is looked up in
``data/basemap/regions.json``. The optional positional lat/lon form stays
supported for one-off builds that don't sit in the region registry.

Example (Spangdahlem, from registry):
    uv run python scripts/build_basemap_hillshade.py spangdahlem

Example (one-off, custom bbox):
    uv run python scripts/build_basemap_hillshade.py spangdahlem \
        49.5 6.0 50.3 7.0

The script:
 1. Downloads SRTM ``.hgt.gz`` tiles covering the bbox (AWS public-domain
    mirror), caches them under ``data/cache/dem/``, and stitches them into
    a single raster so the hillshade is continuous across tile seams.
 2. Gaussian-blurs the elevation grid to soften 30 m sample noise.
 3. Applies a vertical-exaggeration factor (default 1.5x) to the DEM so
    the hillshade reads stronger at chart scale.
 4. Computes a standard cartographic hillshade (azimuth 315 deg / NW sun,
    altitude 45 deg) from slope + aspect derived via numpy gradient.
 5. Writes a grayscale PNG (Pillow mode="L", compression 9) to
    ``data/basemap/<region>-hillshade.png``, plus a sidecar
    ``<region>-hillshade.json`` manifest with the bbox so the renderer
    knows where to position it.

The output is deterministic: same DEM + blur + exaggeration = same PNG.

Runtime deps: ``numpy`` + Pillow (already transitive via cairosvg).
Network is only used on the first run; subsequent builds hit the cache.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "Pillow is required for hillshade builds; install with "
        "`uv pip install pillow`"
    ) from e

REPO_ROOT = Path(__file__).resolve().parents[1]
DEM_CACHE_DIR = REPO_ROOT / "data" / "cache" / "dem"
BASEMAP_DIR = REPO_ROOT / "data" / "basemap"
REGIONS_PATH = BASEMAP_DIR / "regions.json"


def load_region_bbox(region_id: str) -> tuple[float, float, float, float]:
    """Look up ``region_id`` in ``data/basemap/regions.json`` and return
    ``(lat_min, lon_min, lat_max, lon_max)``.

    The registry stores bboxes in GeoJSON convention
    ``[lon_min, lat_min, lon_max, lat_max]``; this helper reorders them
    into the lat-first tuple that every build script consumes internally.
    """
    if not REGIONS_PATH.is_file():
        raise RuntimeError(f"region registry missing: {REGIONS_PATH}")
    with REGIONS_PATH.open("r", encoding="utf-8") as fh:
        registry = json.load(fh)
    if region_id not in registry:
        known = ", ".join(sorted(registry)) or "<empty>"
        raise RuntimeError(
            f"region {region_id!r} not in registry ({REGIONS_PATH}); "
            f"known regions: {known}"
        )
    entry = registry[region_id]
    bbox = entry.get("bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        raise RuntimeError(
            f"region {region_id!r} has invalid bbox: {bbox!r} "
            f"(expected [lon_min, lat_min, lon_max, lat_max])"
        )
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
    return (lat_min, lon_min, lat_max, lon_max)

# ---------------------------------------------------------------------------
# SRTM tile fetch + parse (ported from the retired build_basemap_topo.py)
# ---------------------------------------------------------------------------

DEM_URL_TEMPLATES = (
    "https://elevation-tiles-prod.s3.amazonaws.com/skadi/{latdir}/{name}",
    "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{latdir}/{name}",
)

SRTM1_PX = 3601
SRTM3_PX = 1201
NODATA = -32768
M_TO_FT = 3.28084


def tile_name(lat_floor: int, lon_floor: int) -> str:
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"{ns}{abs(lat_floor):02d}{ew}{abs(lon_floor):03d}.hgt.gz"


def tile_dir(lat_floor: int, lon_floor: int) -> str:
    ns = "N" if lat_floor >= 0 else "S"
    return f"{ns}{abs(lat_floor):02d}"


def fetch_tile(lat_floor: int, lon_floor: int) -> Path:
    name = tile_name(lat_floor, lon_floor)
    cached = DEM_CACHE_DIR / name
    if cached.is_file() and cached.stat().st_size > 1_000_000:
        return cached
    DEM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for template in DEM_URL_TEMPLATES:
        url = template.format(latdir=tile_dir(lat_floor, lon_floor), name=name)
        for attempt in (1, 2):
            try:
                print(f"  fetching {url} (attempt {attempt})", flush=True)
                req = urllib.request.Request(
                    url, headers={"User-Agent": "open-plates/0.1"},
                )
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = r.read()
                if len(data) < 1_000_000:
                    raise RuntimeError(
                        f"suspiciously small response ({len(data)} bytes)"
                    )
                cached.write_bytes(data)
                return cached
            except (
                urllib.error.URLError, urllib.error.HTTPError, RuntimeError,
            ) as e:
                last_err = e
                print(f"    failed: {e}", flush=True)
                if attempt == 1:
                    time.sleep(5)
    raise RuntimeError(f"could not fetch DEM tile {name}: {last_err}")


def parse_tile(path: Path) -> tuple[np.ndarray, int]:
    """Return (elev_m, px_edge). Rows are top-down (north-to-south)."""
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    total = len(raw) // 2
    edge = int(round(math.sqrt(total)))
    if edge * edge != total or edge not in (SRTM1_PX, SRTM3_PX):
        raise RuntimeError(
            f"unexpected DEM tile size: {len(raw)} bytes -> edge {edge}"
        )
    arr = np.frombuffer(raw, dtype=">i2").astype(np.float32).reshape(edge, edge)
    arr[arr == NODATA] = np.nan
    return arr, edge


def _tile_covers(
    lat_floor: int, lon_floor: int, bbox: tuple[float, float, float, float],
) -> bool:
    lat_min, lon_min, lat_max, lon_max = bbox
    return not (
        lat_floor + 1 < lat_min
        or lat_floor > lat_max
        or lon_floor + 1 < lon_min
        or lon_floor > lon_max
    )


def iterate_tiles(
    bbox: tuple[float, float, float, float],
) -> list[tuple[int, int]]:
    lat_min, lon_min, lat_max, lon_max = bbox
    out: list[tuple[int, int]] = []
    for lat in range(math.floor(lat_min), math.floor(lat_max) + 1):
        for lon in range(math.floor(lon_min), math.floor(lon_max) + 1):
            if _tile_covers(lat, lon, bbox):
                out.append((lat, lon))
    return out


def _mosaic_tiles(
    tiles: list[tuple[int, int]], bbox: tuple[float, float, float, float],
    pad_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stitch every covering tile into ONE unified grid, then crop to bbox+pad.

    Shading a single merged raster eliminates seam artefacts that appear
    when each 1 deg x 1 deg tile is processed independently. Adjacent tiles
    share their boundary cell exactly, so we drop the overlapping row/col
    when concatenating.
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    clip = (
        lon_min - pad_deg, lat_min - pad_deg,
        lon_max + pad_deg, lat_max + pad_deg,
    )

    lat_floors = sorted({t[0] for t in tiles}, reverse=True)  # north first
    lon_floors = sorted({t[1] for t in tiles})
    edge_cache: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
    for tf in tiles:
        edge_cache[tf] = parse_tile(fetch_tile(*tf))
    edge = next(iter(edge_cache.values()))[1]
    if not all(e == edge for _, e in edge_cache.values()):
        raise RuntimeError(
            "mixed SRTM1/SRTM3 resolutions across tiles; aborting"
        )

    rows: list[np.ndarray] = []
    for lat_floor in lat_floors:
        row_chunks: list[np.ndarray] = []
        for i, lon_floor in enumerate(lon_floors):
            if (lat_floor, lon_floor) not in edge_cache:
                row_chunks.append(
                    np.full((edge, edge), np.nan, dtype=np.float32),
                )
                continue
            arr_m, _ = edge_cache[(lat_floor, lon_floor)]
            chunk = arr_m.copy()
            if i > 0:
                chunk = chunk[:, 1:]
            row_chunks.append(chunk)
        rows.append(np.concatenate(row_chunks, axis=1))
    for i in range(1, len(rows)):
        rows[i] = rows[i][1:, :]
    mosaic_m = np.concatenate(rows, axis=0)

    rows_per_tile = edge - 1
    total_rows = rows_per_tile * len(lat_floors) + 1
    total_cols = rows_per_tile * len(lon_floors) + 1
    assert mosaic_m.shape == (total_rows, total_cols), (
        f"mosaic shape mismatch: got {mosaic_m.shape}, expected "
        f"({total_rows}, {total_cols})"
    )
    top_lat = max(lat_floors) + 1.0
    left_lon = min(lon_floors)
    d = 1.0 / rows_per_tile
    lats = top_lat - np.arange(total_rows, dtype=np.float32) * d
    lons = left_lon + np.arange(total_cols, dtype=np.float32) * d

    col_mask = (lons >= clip[0]) & (lons <= clip[2])
    row_mask = (lats >= clip[1]) & (lats <= clip[3])
    c_lo = int(np.argmax(col_mask))
    c_hi = total_cols - int(np.argmax(col_mask[::-1]))
    r_lo = int(np.argmax(row_mask))
    r_hi = total_rows - int(np.argmax(row_mask[::-1]))
    sub_lons = lons[c_lo:c_hi]
    sub_lats = lats[r_lo:r_hi]
    sub_mosaic = mosaic_m[r_lo:r_hi, c_lo:c_hi]
    return sub_lons, sub_lats, sub_mosaic


def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 1-D Gaussian blur. NaNs are treated as zero-weight.

    SRTM at 30 m has natural sample noise; blurring before shading is
    what makes the hillshade look like natural terrain rather than
    pixellated stair-steps. Without scipy so we convolve by hand.
    """
    if sigma <= 0:
        return arr
    radius = max(1, int(round(sigma * 3)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    mask = np.isfinite(arr).astype(np.float32)
    z = np.where(mask > 0, arr, 0.0).astype(np.float32)

    def _conv_rows(a: np.ndarray) -> np.ndarray:
        pad = ((0, 0), (radius, radius))
        padded = np.pad(a, pad, mode="reflect")
        out = np.zeros_like(a, dtype=np.float32)
        for k, w in enumerate(kernel):
            out += w * padded[:, k : k + a.shape[1]]
        return out

    def _conv_cols(a: np.ndarray) -> np.ndarray:
        pad = ((radius, radius), (0, 0))
        padded = np.pad(a, pad, mode="reflect")
        out = np.zeros_like(a, dtype=np.float32)
        for k, w in enumerate(kernel):
            out += w * padded[k : k + a.shape[0], :]
        return out

    num = _conv_cols(_conv_rows(z))
    den = _conv_cols(_conv_rows(mask))
    with np.errstate(invalid="ignore", divide="ignore"):
        result = num / den
    result[den == 0] = np.nan
    return result

# Default sun parameters (standard cartographic NW sun).
DEFAULT_AZIMUTH_DEG = 315.0
DEFAULT_ALTITUDE_DEG = 45.0
DEFAULT_VERTICAL_EXAGGERATION = 1.5
DEFAULT_BLUR_SIGMA_PX = 2.0
# Target longest-edge pixel length for the emitted PNG. Chart plan views
# are ~570 px wide at 72 dpi, so ~1200 px gives a comfortable 2x ceiling
# without ballooning file size. SRTM3 at 1 arc-second is ~3600 px per
# degree — way more detail than a procedure chart can carry.
DEFAULT_MAX_EDGE_PX = 1200


# ---------------------------------------------------------------------------
# Hillshade computation
# ---------------------------------------------------------------------------


def _cell_size_m(lats: np.ndarray, lons: np.ndarray) -> tuple[float, float]:
    """Return (cell_size_y_m, cell_size_x_m) for the cropped grid.

    Latitude spacing is the same everywhere; longitude spacing shrinks with
    latitude by ``cos(lat)``. We use the mid-latitude of the grid for a
    single scalar in x — good enough for the small regions we render and
    keeps the math simple.
    """
    # Degrees per cell along each axis.
    dlat_deg = float(abs(lats[1] - lats[0])) if len(lats) > 1 else 0.0
    dlon_deg = float(abs(lons[1] - lons[0])) if len(lons) > 1 else 0.0
    # Reference latitude: middle of the grid.
    mid_lat = float(np.mean(lats))
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.radians(mid_lat))
    return (
        dlat_deg * meters_per_deg_lat,
        dlon_deg * meters_per_deg_lon,
    )


def compute_hillshade(
    elev_m: np.ndarray,
    cell_size_y_m: float,
    cell_size_x_m: float,
    azimuth_deg: float = DEFAULT_AZIMUTH_DEG,
    altitude_deg: float = DEFAULT_ALTITUDE_DEG,
    vertical_exaggeration: float = DEFAULT_VERTICAL_EXAGGERATION,
) -> np.ndarray:
    """Return a uint8 hillshade grid (0..255) from a 2-D elevation array.

    Standard formula (ESRI / GDAL):

        slope = atan(sqrt(dzdx^2 + dzdy^2))
        aspect = atan2(dzdy, -dzdx)  (then wrap to [0, 2pi))
        hs = cos(pi/2 - altitude) * cos(slope)
           + sin(pi/2 - altitude) * sin(slope) * cos(azimuth - aspect)

    NaNs in the DEM are replaced with the grid-wide mean elevation so edges
    of the raster don't flash bright/dark artefacts.
    """
    # Fill NaNs with the grid mean so gradients are finite. (SRTM usually
    # only has NaN in sea cells; for our inland Eifel grid this is rare.)
    finite = np.isfinite(elev_m)
    if not finite.all():
        fill_val = float(np.nanmean(elev_m)) if finite.any() else 0.0
        elev_m = np.where(finite, elev_m, fill_val).astype(np.float32)

    # Vertical exaggeration: makes the shading read stronger at chart scale.
    z = elev_m.astype(np.float32) * float(vertical_exaggeration)

    # numpy.gradient returns (d/d_row, d/d_col). Rows go N->S so
    # dz/d_row = -dz/dy in a "y = north" sense; we flip the sign so dzdy
    # points "up is positive y" (which matches the standard formula).
    grad_y, grad_x = np.gradient(z)
    # Spec: dzdx = (right - left) / (2 * cell), dzdy = (down - up) / (2 * cell).
    # numpy.gradient with default spacing=1 already gives (right-left)/2.
    # Divide by the cell-size-in-metres to convert to rise/run.
    dzdx = grad_x / float(cell_size_x_m)
    dzdy = grad_y / float(cell_size_y_m)

    slope = np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    # Wrap aspect into [0, 2*pi).
    aspect = np.where(aspect < 0.0, aspect + 2.0 * np.pi, aspect)

    altitude = np.radians(altitude_deg)
    azimuth = np.radians(azimuth_deg)

    hs = (
        np.cos(np.pi / 2.0 - altitude) * np.cos(slope)
        + np.sin(np.pi / 2.0 - altitude) * np.sin(slope)
        * np.cos(azimuth - aspect)
    )
    hs = np.clip(hs, 0.0, 1.0)
    return (hs * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build(
    region_id: str,
    bbox: tuple[float, float, float, float],
    azimuth_deg: float = DEFAULT_AZIMUTH_DEG,
    altitude_deg: float = DEFAULT_ALTITUDE_DEG,
    vertical_exaggeration: float = DEFAULT_VERTICAL_EXAGGERATION,
    blur_sigma_px: float = DEFAULT_BLUR_SIGMA_PX,
    pad_deg: float = 0.0,
    max_edge_px: int = DEFAULT_MAX_EDGE_PX,
) -> tuple[Path, Path]:
    """Build the hillshade PNG + manifest. Returns (png_path, json_path)."""
    print(
        f"building hillshade for region={region_id} bbox={bbox}", flush=True,
    )
    tiles = iterate_tiles(bbox)
    print(f"  tiles required: {tiles}")

    # Mosaic + crop (reuses topo helper).
    sub_lons, sub_lats, sub_elev_m = _mosaic_tiles(tiles, bbox, pad_deg)
    print(
        f"  mosaic sub-grid {sub_elev_m.shape} "
        f"min={np.nanmin(sub_elev_m):.0f} max={np.nanmax(sub_elev_m):.0f} m",
        flush=True,
    )

    if blur_sigma_px > 0:
        sub_elev_m = _gaussian_blur(sub_elev_m, blur_sigma_px)
        print(f"  applied gaussian blur sigma={blur_sigma_px}px", flush=True)

    # Cell size in metres for proper slope units.
    cy_m, cx_m = _cell_size_m(sub_lats, sub_lons)
    print(
        f"  cell size (m): x={cx_m:.2f}  y={cy_m:.2f}  "
        f"(elevation in ft x{M_TO_FT:.3f} conversion not needed for shading)",
        flush=True,
    )

    hs = compute_hillshade(
        sub_elev_m,
        cell_size_y_m=cy_m, cell_size_x_m=cx_m,
        azimuth_deg=azimuth_deg, altitude_deg=altitude_deg,
        vertical_exaggeration=vertical_exaggeration,
    )
    print(
        f"  hillshade stats: min={int(hs.min())} max={int(hs.max())} "
        f"mean={float(hs.mean()):.1f}",
        flush=True,
    )

    # Establish the PNG bbox. The mosaic crop may extend slightly beyond
    # the requested bbox by pad_deg; we report the *actual* covered region
    # so the renderer places the image correctly.
    lon_min = float(sub_lons.min())
    lon_max = float(sub_lons.max())
    lat_min = float(sub_lats.min())
    lat_max = float(sub_lats.max())

    BASEMAP_DIR.mkdir(parents=True, exist_ok=True)
    png_path = BASEMAP_DIR / f"{region_id}-hillshade.png"
    img = Image.fromarray(hs, mode="L")
    # Downsample to a chart-friendly resolution. SRTM3 at 1" per pixel
    # gives ~3600 px/degree of longitude which is far more than a paper
    # chart can carry; downscaling keeps the PNG small (~200-500 KB) and
    # also smooths any residual pixel-scale noise in the hillshade.
    max_edge = max(img.size)
    if max_edge > max_edge_px:
        scale = max_edge_px / float(max_edge)
        new_size = (
            max(1, int(round(img.size[0] * scale))),
            max(1, int(round(img.size[1] * scale))),
        )
        img = img.resize(new_size, resample=Image.LANCZOS)
        print(
            f"  downscaled to {new_size[0]}x{new_size[1]} px "
            f"(max_edge={max_edge_px})",
            flush=True,
        )
    img.save(png_path, format="PNG", optimize=True, compress_level=9)
    png_size = png_path.stat().st_size
    print(
        f"  wrote {png_path} ({png_size:,} bytes, "
        f"{hs.shape[1]}x{hs.shape[0]} px)",
        flush=True,
    )

    # img.size is (width, height) = (cols, rows). Manifest reports (rows, cols).
    final_cols, final_rows = img.size
    manifest = {
        "type": "Hillshade",
        "region": region_id,
        "bbox": [lon_min, lat_min, lon_max, lat_max],
        "image": png_path.name,
        "pixel_shape": [int(final_rows), int(final_cols)],
        "generator": "scripts/build_basemap_hillshade.py",
        "azimuth_deg": float(azimuth_deg),
        "altitude_deg": float(altitude_deg),
        "vertical_exaggeration": float(vertical_exaggeration),
    }
    json_path = BASEMAP_DIR / f"{region_id}-hillshade.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"  wrote {json_path}", flush=True)
    return png_path, json_path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("region_id")
    # Positional bbox is optional: when omitted we resolve the bbox from
    # ``data/basemap/regions.json``. When present, all four must appear.
    ap.add_argument("lat_min", type=float, nargs="?")
    ap.add_argument("lon_min", type=float, nargs="?")
    ap.add_argument("lat_max", type=float, nargs="?")
    ap.add_argument("lon_max", type=float, nargs="?")
    ap.add_argument(
        "--azimuth", type=float, default=DEFAULT_AZIMUTH_DEG,
        help="Sun azimuth in degrees (0=N, 90=E). Default 315 (NW).",
    )
    ap.add_argument(
        "--altitude", type=float, default=DEFAULT_ALTITUDE_DEG,
        help="Sun altitude above horizon in degrees. Default 45.",
    )
    ap.add_argument(
        "--z-factor", type=float, default=DEFAULT_VERTICAL_EXAGGERATION,
        help="Vertical exaggeration applied to the DEM. Default 1.5.",
    )
    ap.add_argument(
        "--blur", type=float, default=DEFAULT_BLUR_SIGMA_PX,
        help="Gaussian blur sigma (px) applied to DEM pre-shading. Default 2.",
    )
    ap.add_argument(
        "--max-edge", type=int, default=DEFAULT_MAX_EDGE_PX,
        help=(
            "Longest-edge pixel length for the emitted PNG. "
            f"Default {DEFAULT_MAX_EDGE_PX}."
        ),
    )
    args = ap.parse_args(argv)

    positionals = (args.lat_min, args.lon_min, args.lat_max, args.lon_max)
    given = [v for v in positionals if v is not None]
    if len(given) not in (0, 4):
        print(
            "must pass either 0 bbox args (resolve via registry) or all 4 "
            "(lat_min lon_min lat_max lon_max)",
            file=sys.stderr,
        )
        return 2
    if len(given) == 0:
        try:
            bbox = load_region_bbox(args.region_id)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    else:
        bbox = (args.lat_min, args.lon_min, args.lat_max, args.lon_max)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        print(
            "bbox must have lat_max > lat_min and lon_max > lon_min",
            file=sys.stderr,
        )
        return 2

    build(
        args.region_id, bbox,
        azimuth_deg=args.azimuth,
        altitude_deg=args.altitude,
        vertical_exaggeration=args.z_factor,
        blur_sigma_px=args.blur,
        max_edge_px=args.max_edge,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
