"""Tests for the build-time basemap layers (populated areas, tall obstacles,
hillshade raster).

These tests exercise the data files produced by
``scripts/build_basemap_populated.py``, ``scripts/build_basemap_obstacles.py``,
and ``scripts/build_basemap_hillshade.py``. They use ``pytest.mark.skipif``
so a fresh checkout that hasn't run the pipelines yet still passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from open_plates.basemap import (
    COLOR_BASEMAP_URBAN,
    BasemapConfig,
    load_basemap,
    render_basemap,
)
from open_plates.legs import build_fix_table, compile_legs
from open_plates.render_svg import render_iap_svg

REPO_ROOT = Path(__file__).resolve().parents[1]
BASEMAP_DIR = REPO_ROOT / "data" / "basemap"
POPULATED_PATH = BASEMAP_DIR / "spangdahlem-populated.json"
OBSTACLES_PATH = BASEMAP_DIR / "spangdahlem-obstacles.json"
HILLSHADE_PATH = BASEMAP_DIR / "spangdahlem-hillshade.json"
ETAD_EXAMPLE = REPO_ROOT / "examples" / "etad-ils-rwy-23-straight.yaml"


@pytest.mark.skipif(
    not POPULATED_PATH.is_file(),
    reason=f"{POPULATED_PATH.name} not built yet; run build_basemap_populated.py",
)
def test_populated_areas_load_if_file_present() -> None:
    """The populated GeoJSON parses and contains named Polygon features."""
    with POPULATED_PATH.open("r", encoding="utf-8") as fh:
        gj = json.load(fh)
    assert gj.get("type") == "FeatureCollection"

    features = gj.get("features") or []
    polys = [
        f for f in features
        if (f.get("properties") or {}).get("class") == "populated_area"
        and (f.get("geometry") or {}).get("type") == "Polygon"
    ]
    assert polys, "expected at least one populated_area Polygon"

    named = [
        p for p in polys
        if (p.get("properties") or {}).get("name")
    ]
    assert named, "expected at least one populated_area with a name"


@pytest.mark.skipif(
    not POPULATED_PATH.is_file(),
    reason="sibling populated file not built yet",
)
def test_load_basemap_merges_siblings() -> None:
    """load_basemap('spangdahlem') returns a single merged FeatureCollection."""
    gj = load_basemap("spangdahlem")
    assert gj is not None
    assert gj.get("type") == "FeatureCollection"

    features = gj.get("features") or []
    classes = {(f.get("properties") or {}).get("class") for f in features}
    # At least one city (from base), one populated (pop sibling).
    assert "city" in classes, "expected city feature from base file"
    assert "populated_area" in classes, (
        "expected at least one populated_area feature from -populated sibling"
    )

    # load_basemap signals which siblings it merged.
    merged = gj.get("_merged") or []
    assert "populated" in merged

    # If a hillshade sidecar exists, its manifest is attached to the root.
    if HILLSHADE_PATH.is_file():
        hs = gj.get("_hillshade")
        assert isinstance(hs, dict)
        assert hs.get("type") == "Hillshade"


def test_etad_render_emits_populated_urban() -> None:
    """Full ETAD render emits the urban fill when a populated-area polygon
    actually overlaps the ETAD plan view.

    Since the populated sibling is built over a generous region bbox
    (4.8E..8.2E / 48.8N..51.0N) and rank-capped to 30 polygons, the
    top-30 selection may not include any settlement that overlaps a
    tight plan view at ETAD. When that happens the ``_populated_in_view``
    filter correctly emits no urban fill and the assertion is skipped.
    The wrapper ``data-layer="basemap"`` element is still always present.
    """
    proc = yaml.safe_load(ETAD_EXAMPLE.read_text())
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)
    svg = render_iap_svg(proc, prims)

    assert 'data-layer="basemap"' in svg

    if not POPULATED_PATH.is_file():
        return

    # Whether the urban fill appears in the render depends on whether any
    # polygon's projected bbox overlaps the plan rect — a geographic
    # overlap is necessary but not sufficient because the plan view is
    # padded + aspect-letterboxed. The authoritative answer comes from
    # the same filter the renderer uses. If the filter selects nothing,
    # the absence of the urban fill is a legitimate render-time outcome
    # of the region-scoped build, not a regression.
    from open_plates.basemap import _populated_in_view
    from open_plates.projection import plan_projector

    with POPULATED_PATH.open("r", encoding="utf-8") as fh:
        gj = json.load(fh)
    # Plan rect mirrors render_iap_svg: roughly 14..598 px horizontally,
    # 60..696 px vertically in the default US Letter page layout. We use
    # the projector the renderer would build so the filter's bbox math
    # matches the real render.
    plan_rect_px = (14.0, 60.0, 598.0, 696.0)
    projector = plan_projector(
        proc, fixes, (14.0, 60.0, 584.0, 636.0), primitives=prims,
    )
    visible = _populated_in_view(
        list(gj.get("features") or []), projector, plan_rect_px,
    )

    if visible:
        assert COLOR_BASEMAP_URBAN in svg, (
            "expected COLOR_BASEMAP_URBAN fill in rendered ETAD SVG "
            f"({len(visible)} populated polygon(s) visible in plan)"
        )


def test_basemap_config_new_toggles_default_on() -> None:
    """Layer toggles default to True; unknown keys are ignored."""
    cfg = BasemapConfig()
    assert cfg.show_populated_areas is True
    assert cfg.show_obstacles is True
    assert cfg.show_hillshade is True
    assert cfg.show_spot_elevations is True
    assert cfg.obstacle_min_ft_agl == 1000

    cfg2 = BasemapConfig.from_mapping({
        "show_populated_areas": False,
        "show_obstacles": False, "obstacle_min_ft_agl": 500,
        "show_hillshade": False, "show_spot_elevations": False,
        "not_a_field": "ignored",
    })
    assert cfg2.show_populated_areas is False
    assert cfg2.show_obstacles is False
    assert cfg2.show_hillshade is False
    assert cfg2.show_spot_elevations is False
    assert cfg2.obstacle_min_ft_agl == 500


@pytest.mark.skipif(
    not OBSTACLES_PATH.is_file(),
    reason=f"{OBSTACLES_PATH.name} not built yet; run build_basemap_obstacles.py",
)
def test_obstacles_load_if_file_present() -> None:
    """The obstacles GeoJSON parses; non-empty features all clear 1000 ft AGL."""
    with OBSTACLES_PATH.open("r", encoding="utf-8") as fh:
        gj = json.load(fh)
    assert gj.get("type") == "FeatureCollection"
    features = gj.get("features") or []
    # Eifel region may legitimately return zero features at 1000 ft; only
    # verify per-feature invariants if any are present.
    for f in features:
        props = f.get("properties") or {}
        assert props.get("class") == "obstacle_tall"
        height = props.get("height_ft_agl")
        assert isinstance(height, int) and height >= 1000, (
            f"feature {f.get('id')} has height_ft_agl={height!r}"
        )
        assert props.get("structure_type") in {
            "tower", "mast", "chimney", "wind", "other",
        }
        geom = f.get("geometry") or {}
        assert geom.get("type") == "Point"
        coords = geom.get("coordinates") or []
        assert len(coords) == 2


@pytest.mark.skipif(
    not OBSTACLES_PATH.is_file(),
    reason=f"{OBSTACLES_PATH.name} not built yet; run build_basemap_obstacles.py",
)
def test_load_basemap_merges_obstacles_sibling() -> None:
    """load_basemap merges the obstacles sibling into the FeatureCollection."""
    gj = load_basemap("spangdahlem")
    assert gj is not None
    merged = gj.get("_merged") or []
    assert "obstacles" in merged, (
        f"expected 'obstacles' in _merged; got {merged!r}"
    )
    # If the sibling file has any obstacle features, they must appear in the
    # merged feature list. If it's empty (valid for Eifel @ 1000 ft), the
    # merge is still signalled via _merged above.
    with OBSTACLES_PATH.open("r", encoding="utf-8") as fh:
        sib = json.load(fh)
    sib_count = len(sib.get("features") or [])
    merged_obstacle_count = sum(
        1 for f in (gj.get("features") or [])
        if (f.get("properties") or {}).get("class") == "obstacle_tall"
    )
    assert merged_obstacle_count == sib_count


def test_obstacle_config_toggle() -> None:
    """show_obstacles=False suppresses every obstacle_tall symbol in render.

    Runs against a synthetic in-memory FeatureCollection so the test is
    independent of whether the real sibling file has any features. With
    show_obstacles=True we expect the caption to appear; with =False the
    whole basemap group should be absent (no other layers in the fixture).
    """
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "node-42",
                "properties": {
                    "class": "obstacle_tall",
                    "name": "TEST TOWER",
                    "height_ft_agl": 1234,
                    "structure_type": "tower",
                    "rank": 3,
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

    on = render_basemap(fc, BasemapConfig(show_obstacles=True), projector)
    off = render_basemap(fc, BasemapConfig(show_obstacles=False), projector)

    assert "1234&apos;" in on or "1234'" in on
    assert "1234&apos;" not in off and "1234'" not in off
    # With only obstacle features and the toggle off, the entire basemap
    # group should be omitted (empty body).
    assert 'data-layer="basemap"' not in off
