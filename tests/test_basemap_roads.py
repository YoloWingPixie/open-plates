"""Tests for the OSM-sourced roads basemap layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from open_plates.basemap import (
    COLOR_BASEMAP_ROAD_FILL,
    BasemapConfig,
    load_basemap,
    render_basemap,
)
from open_plates.legs import build_fix_table, compile_legs
from open_plates.render_svg import render_iap_svg

REPO_ROOT = Path(__file__).resolve().parents[1]
BASEMAP_DIR = REPO_ROOT / "data" / "basemap"
ROADS_PATH = BASEMAP_DIR / "spangdahlem-roads.json"
ETAD_EXAMPLE = REPO_ROOT / "examples" / "etad-ils-rwy-23-straight.yaml"


@pytest.mark.skipif(
    not ROADS_PATH.is_file(),
    reason=f"{ROADS_PATH.name} not built yet; run build_basemap_roads.py",
)
def test_roads_file_loads_if_present() -> None:
    """The roads GeoJSON parses and contains well-formed road_* LineStrings."""
    with ROADS_PATH.open("r", encoding="utf-8") as fh:
        gj = json.load(fh)
    assert gj.get("type") == "FeatureCollection"

    features = gj.get("features") or []
    assert len(features) >= 10, (
        f"expected >=10 road features, got {len(features)}"
    )

    for f in features:
        props = f.get("properties") or {}
        cls = str(props.get("class") or "")
        assert cls.startswith("road_"), (
            f"feature {f.get('id')!r} has unexpected class {cls!r}"
        )
        assert cls in {
            "road_motorway", "road_trunk", "road_primary",
            "road_secondary", "road_tertiary",
        }
        geom = f.get("geometry") or {}
        assert geom.get("type") == "LineString"
        coords = geom.get("coordinates") or []
        assert len(coords) >= 2


def test_road_tiers_rendered() -> None:
    """Render ETAD; assert motorway fill AND tertiary hairline both show up.

    Uses a synthetic FeatureCollection (independent of whether the real
    Overpass build has been run) so this test is deterministic and runs
    on a fresh checkout.
    """
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature", "id": "m-1",
                "properties": {"class": "road_motorway", "name": "A1", "rank": 1},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[6.40, 49.90], [6.50, 49.92], [6.60, 49.94]],
                },
            },
            {
                "type": "Feature", "id": "t-1",
                "properties": {
                    "class": "road_tertiary", "name": "L12", "rank": 4,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[6.42, 49.91], [6.45, 49.93]],
                },
            },
        ],
    }

    def projector(latlon: tuple[float, float]) -> tuple[float, float]:
        lat, lon = latlon
        return ((lon - 6.0) * 200.0, (50.0 - lat) * 200.0)

    # Need road_max_rank=4 to see the tertiary.
    svg = render_basemap(
        fc, BasemapConfig(show_roads=True, road_max_rank=4), projector,
    )
    # Motorway fill colour (the atlas-style lighter inner stroke).
    assert COLOR_BASEMAP_ROAD_FILL in svg, (
        "expected motorway fill colour in rendered SVG"
    )
    # Tertiary tier uses COLOR_HAIRLINE (#B8B3A6) as its single stroke.
    assert "#B8B3A6" in svg, (
        "expected hairline tertiary stroke colour in rendered SVG"
    )


def test_road_config_toggle() -> None:
    """show_roads=False suppresses every road stroke."""
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature", "id": "m-1",
                "properties": {"class": "road_motorway", "name": "A1", "rank": 1},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[6.40, 49.90], [6.60, 49.94]],
                },
            },
        ],
    }

    def projector(latlon: tuple[float, float]) -> tuple[float, float]:
        lat, lon = latlon
        return ((lon - 6.0) * 200.0, (50.0 - lat) * 200.0)

    on = render_basemap(fc, BasemapConfig(show_roads=True), projector)
    off = render_basemap(fc, BasemapConfig(show_roads=False), projector)

    # On: expect motorway casing + fill + label.
    assert COLOR_BASEMAP_ROAD_FILL in on
    # Off: no roads at all — and with only road features the whole basemap
    # group should be omitted.
    assert COLOR_BASEMAP_ROAD_FILL not in off
    assert 'data-layer="basemap"' not in off


@pytest.mark.skipif(
    not ROADS_PATH.is_file(),
    reason="sibling roads file not built yet",
)
def test_load_basemap_merges_roads_sibling() -> None:
    """load_basemap merges the roads sibling into the FeatureCollection."""
    gj = load_basemap("spangdahlem")
    assert gj is not None
    merged = gj.get("_merged") or []
    assert "roads" in merged, f"expected 'roads' in _merged; got {merged!r}"

    road_feats = [
        f for f in (gj.get("features") or [])
        if str((f.get("properties") or {}).get("class") or "").startswith("road_")
    ]
    assert road_feats, "expected merged road features in loaded basemap"


@pytest.mark.skipif(
    not ROADS_PATH.is_file(),
    reason="sibling roads file not built yet",
)
def test_etad_render_emits_road_casings() -> None:
    """Full ETAD render emits road tier geometry when the sibling is present."""
    proc = yaml.safe_load(ETAD_EXAMPLE.read_text())
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)
    svg = render_iap_svg(proc, prims)

    assert 'data-layer="basemap"' in svg
    # Motorway/trunk fill is the main tell — it's the only place
    # COLOR_BASEMAP_ROAD_FILL appears in the entire chart palette.
    assert COLOR_BASEMAP_ROAD_FILL in svg
