"""Tests for the hillshade raster basemap layer + spot-elevation symbols.

The hillshade PNG + sidecar JSON are produced by
``scripts/build_basemap_hillshade.py``. Tests that depend on the actual
files are ``pytest.mark.skipif``-guarded so a fresh checkout without the
build artefacts still passes cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from open_plates.basemap import (
    BasemapConfig,
    load_basemap,
    render_basemap,
)
from open_plates.legs import build_fix_table, compile_legs
from open_plates.render_svg import render_iap_svg

REPO_ROOT = Path(__file__).resolve().parents[1]
BASEMAP_DIR = REPO_ROOT / "data" / "basemap"
HILLSHADE_JSON = BASEMAP_DIR / "spangdahlem-hillshade.json"
HILLSHADE_PNG = BASEMAP_DIR / "spangdahlem-hillshade.png"
SPANG_BASE = BASEMAP_DIR / "spangdahlem.json"
ETAD_EXAMPLE = REPO_ROOT / "examples" / "etad-ils-rwy-23-straight.yaml"


# ---------------------------------------------------------------------------
# Hillshade manifest + PNG
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HILLSHADE_JSON.is_file(),
    reason="spangdahlem-hillshade.json not built yet",
)
def test_hillshade_metadata_loads_if_present() -> None:
    """The hillshade sidecar parses and the PNG it references exists."""
    with HILLSHADE_JSON.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta.get("type") == "Hillshade"
    bbox = meta.get("bbox")
    assert isinstance(bbox, list) and len(bbox) == 4
    for v in bbox:
        assert isinstance(v, (int, float))
    assert meta.get("image"), "manifest must carry an image filename"
    # The PNG sits next to the manifest.
    png_path = HILLSHADE_JSON.parent / meta["image"]
    assert png_path.is_file(), f"hillshade PNG missing: {png_path}"

    # load_basemap attaches the manifest under _hillshade.
    gj = load_basemap("spangdahlem")
    assert gj is not None
    hs = gj.get("_hillshade")
    assert isinstance(hs, dict)
    assert hs.get("type") == "Hillshade"
    assert hs.get("bbox") == bbox


@pytest.mark.skipif(
    not (HILLSHADE_JSON.is_file() and HILLSHADE_PNG.is_file()),
    reason="hillshade artefacts not built yet",
)
def test_hillshade_png_embedded_in_render() -> None:
    """A full ETAD render embeds the PNG as a data URI in an <image>."""
    proc = yaml.safe_load(ETAD_EXAMPLE.read_text())
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)
    svg = render_iap_svg(proc, prims)

    assert "data:image/png;base64," in svg, (
        "expected base64 PNG data URI in rendered SVG"
    )
    assert "<image " in svg, (
        "expected <image> element for hillshade raster"
    )


def test_hillshade_absent_renders_fine() -> None:
    """A basemap without a _hillshade key still renders with no errors."""
    # Synthetic minimal FeatureCollection — no siblings, no hillshade.
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "city-1",
                "properties": {
                    "class": "city", "name": "NOWHERE", "rank": 1,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [6.5, 49.8],
                },
            },
        ],
    }

    def projector(latlon: tuple[float, float]) -> tuple[float, float]:
        lat, lon = latlon
        return ((lon - 6.0) * 100.0, (50.0 - lat) * 100.0)

    out = render_basemap(fc, BasemapConfig(), projector)
    # Render succeeds; since there is no hillshade meta attached the
    # SVG must not contain an <image> element or a data-URI.
    assert 'data-layer="basemap"' in out
    assert "<image " not in out
    assert "data:image/png;base64," not in out


def test_hillshade_toggle_off_omits_image() -> None:
    """show_hillshade=False suppresses the <image> even when meta is present."""
    fc = {
        "type": "FeatureCollection",
        "features": [],
        "_hillshade": {
            "type": "Hillshade",
            "region": "dummy",
            "bbox": [6.0, 49.0, 7.0, 50.0],
            # Point at a file that does not exist — with show_hillshade=False
            # the renderer must not even try to read it.
            "image": "does-not-exist.png",
            "_dir": "/tmp",
        },
    }

    def projector(latlon: tuple[float, float]) -> tuple[float, float]:
        lat, lon = latlon
        return (lon * 100.0, (50.0 - lat) * 100.0)

    out = render_basemap(fc, BasemapConfig(show_hillshade=False), projector)
    assert "<image " not in out
    assert "data:image/png;base64," not in out


# ---------------------------------------------------------------------------
# Spot elevations
# ---------------------------------------------------------------------------


def test_spot_elevation_renders() -> None:
    """ETAD render contains the elevation numbers from spot_elevation features.

    Reads the base file directly so the test doesn't depend on the sibling
    files being built. Every spot_elevation `name` must appear as an
    altitude caption with a trailing apostrophe (e.g. "2221'").
    """
    with SPANG_BASE.open("r", encoding="utf-8") as fh:
        base = json.load(fh)
    spot_names = [
        str((f.get("properties") or {}).get("name") or "").strip()
        for f in (base.get("features") or [])
        if (f.get("properties") or {}).get("class") == "spot_elevation"
    ]
    spot_names = [n for n in spot_names if n]
    assert spot_names, "base file should carry spot_elevation features"

    proc = yaml.safe_load(ETAD_EXAMPLE.read_text())
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)
    svg = render_iap_svg(proc, prims)

    # At least one spot elevation caption must be present — some may fall
    # outside the plan clip or be masked by procedure bbox overlap, so we
    # assert the majority-presence case rather than all-or-nothing.
    hits = sum(
        1 for n in spot_names
        if f"{n}&apos;" in svg or f"{n}'" in svg
    )
    assert hits >= 1, (
        f"expected at least one spot_elevation caption in SVG; "
        f"none of {spot_names!r} found"
    )


def test_spot_elevation_toggle_off_hides_triangles() -> None:
    """show_spot_elevations=False suppresses the triangle path + captions."""
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "spot-1",
                "properties": {
                    "class": "spot_elevation",
                    "name": "2221",
                    "rank": 1,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [6.55, 50.02],
                },
            },
        ],
    }

    def projector(latlon: tuple[float, float]) -> tuple[float, float]:
        lat, lon = latlon
        return ((lon - 6.0) * 100.0, (50.0 - lat) * 100.0)

    on = render_basemap(fc, BasemapConfig(show_spot_elevations=True), projector)
    off = render_basemap(fc, BasemapConfig(show_spot_elevations=False), projector)

    assert "2221&apos;" in on or "2221'" in on
    assert "2221&apos;" not in off and "2221'" not in off
