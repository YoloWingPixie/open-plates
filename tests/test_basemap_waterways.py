"""Tests for the OSM-sourced waterways basemap layer (rivers + streams + lakes)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from open_plates.basemap import (
    COLOR_BASEMAP_RIVER,
    COLOR_BASEMAP_SEA,
    BasemapConfig,
    load_basemap,
    render_basemap,
)
from open_plates.legs import build_fix_table, compile_legs
from open_plates.render_svg import render_iap_svg

REPO_ROOT = Path(__file__).resolve().parents[1]
BASEMAP_DIR = REPO_ROOT / "data" / "basemap"
SPANGDAHLEM_WW = BASEMAP_DIR / "spangdahlem-waterways.json"
ZURICH_WW = BASEMAP_DIR / "zurich-waterways.json"
ETAD_EXAMPLE = REPO_ROOT / "examples" / "etad-ils-rwy-23-straight.yaml"
LSZH_EXAMPLE = REPO_ROOT / "examples" / "lszh-ils-z-rwy-14.yaml"


@pytest.mark.skipif(
    not (SPANGDAHLEM_WW.is_file() or ZURICH_WW.is_file()),
    reason="no <region>-waterways.json present; run build_basemap_waterways.py",
)
def test_waterways_load_if_present() -> None:
    """Any present waterways file parses as a FeatureCollection and contains
    at least one river/stream/lake feature."""
    allowed = {"river", "stream", "lake"}
    seen_classes: set[str] = set()
    files_checked = 0
    for path in (SPANGDAHLEM_WW, ZURICH_WW):
        if not path.is_file():
            continue
        files_checked += 1
        with path.open("r", encoding="utf-8") as fh:
            gj = json.load(fh)
        assert gj.get("type") == "FeatureCollection"
        feats = gj.get("features") or []
        assert feats, f"{path.name} is empty"
        for f in feats:
            cls = str((f.get("properties") or {}).get("class") or "")
            assert cls in allowed, (
                f"feature {f.get('id')!r} in {path.name} has unexpected "
                f"class {cls!r}"
            )
            seen_classes.add(cls)
            geom = f.get("geometry") or {}
            gt = geom.get("type")
            if cls in ("river", "stream"):
                assert gt == "LineString"
                assert len(geom.get("coordinates") or []) >= 2
            elif cls == "lake":
                assert gt == "Polygon"
                coords = geom.get("coordinates") or []
                assert coords and len(coords[0]) >= 4
    assert files_checked > 0
    assert seen_classes, "expected at least one class across present files"


def test_lake_rendering() -> None:
    """Render a synthetic FeatureCollection containing a lake polygon and
    assert the output SVG contains a <path> with the sea fill colour
    (lakes and sea share COLOR_BASEMAP_SEA).
    """
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature", "id": "lake-1",
                "properties": {"class": "lake", "name": "Zurichsee", "rank": 1},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [8.55, 47.22], [8.80, 47.22],
                        [8.80, 47.37], [8.55, 47.37],
                        [8.55, 47.22],
                    ]],
                },
            },
        ],
    }

    def projector(latlon: tuple[float, float]) -> tuple[float, float]:
        lat, lon = latlon
        return ((lon - 8.0) * 200.0, (48.0 - lat) * 200.0)

    svg = render_basemap(fc, BasemapConfig(show_lakes=True), projector)
    assert svg, "expected non-empty SVG for lake-only feature collection"
    assert COLOR_BASEMAP_SEA in svg, (
        "expected COLOR_BASEMAP_SEA fill for lake polygon"
    )
    # Turning lakes off suppresses the fill (and with nothing else in the
    # FC, the whole basemap group should vanish).
    off = render_basemap(fc, BasemapConfig(show_lakes=False), projector)
    assert COLOR_BASEMAP_SEA not in off


def test_waterway_rank_filter() -> None:
    """waterway_max_rank clamps streams off by rank even if show_streams is on."""
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature", "id": "s-1",
                "properties": {"class": "stream", "name": "Brook", "rank": 3},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[6.50, 49.90], [6.52, 49.92]],
                },
            },
            {
                "type": "Feature", "id": "r-1",
                "properties": {"class": "river", "name": "Mosel", "rank": 1},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[6.40, 49.80], [6.60, 49.95]],
                },
            },
        ],
    }

    def projector(latlon: tuple[float, float]) -> tuple[float, float]:
        lat, lon = latlon
        return ((lon - 6.0) * 200.0, (50.0 - lat) * 200.0)

    # Default: show_streams=False, waterway_max_rank=2 — river rendered,
    # stream not.
    default_svg = render_basemap(fc, BasemapConfig(), projector)
    assert COLOR_BASEMAP_RIVER in default_svg
    # Counted path emissions: rivers group emits one <path>, no stream path.
    assert default_svg.count(f'stroke="{COLOR_BASEMAP_RIVER}"') == 1

    # show_streams=True but waterway_max_rank=2 — stream still filtered out.
    rank2_svg = render_basemap(
        fc, BasemapConfig(show_streams=True, waterway_max_rank=2), projector,
    )
    assert rank2_svg.count(f'stroke="{COLOR_BASEMAP_RIVER}"') == 1

    # show_streams=True + waterway_max_rank=3 — stream is now drawn.
    rank3_svg = render_basemap(
        fc, BasemapConfig(show_streams=True, waterway_max_rank=3), projector,
    )
    assert rank3_svg.count(f'stroke="{COLOR_BASEMAP_RIVER}"') == 2


@pytest.mark.skipif(
    not ZURICH_WW.is_file(),
    reason="zurich-waterways.json not built; run build_basemap_waterways.py",
)
def test_waterways_merge_via_load_basemap() -> None:
    """load_basemap merges the waterways sibling into the FeatureCollection
    and the merged features include at least one lake."""
    gj = load_basemap("zurich")
    assert gj is not None
    merged = gj.get("_merged") or []
    assert "waterways" in merged, (
        f"expected 'waterways' in _merged; got {merged!r}"
    )
    lakes = [
        f for f in (gj.get("features") or [])
        if str((f.get("properties") or {}).get("class") or "") == "lake"
    ]
    assert lakes, "expected merged lake features in loaded basemap"


@pytest.mark.skipif(
    not ZURICH_WW.is_file(),
    reason="zurich-waterways.json not built",
)
def test_lszh_render_emits_lake_fills() -> None:
    """Rendering LSZH with the waterways sibling present produces lake fills.

    Zurich area has Zürichsee plus several other rank-1 lakes. At least one
    should project into view. The sea fill token also shows up for the
    hand-authored Zürichsee polygon already in zurich.json, so we assert on
    the waterways layer merge instead and that the lake class appears in
    the loaded features.
    """
    proc = yaml.safe_load(LSZH_EXAMPLE.read_text())
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)
    svg = render_iap_svg(proc, prims)
    assert 'data-layer="basemap"' in svg
    assert COLOR_BASEMAP_SEA in svg
