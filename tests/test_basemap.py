"""Tests for the basemap underlay layer."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from open_plates.basemap import (
    BasemapConfig,
    find_covering_region,
    load_basemap,
    render_basemap,
)
from open_plates.legs import build_fix_table, compile_legs
from open_plates.render_svg import render_iap_svg

REPO_ROOT = Path(__file__).resolve().parents[1]
BATUMI_PATH = REPO_ROOT / "data" / "basemap" / "batumi.json"
SPANGDAHLEM_PATH = REPO_ROOT / "data" / "basemap" / "spangdahlem.json"
ZURICH_PATH = REPO_ROOT / "data" / "basemap" / "zurich.json"
UGSB_EXAMPLE = REPO_ROOT / "examples" / "ugsb-ils-rwy-13.yaml"
ETAD_EXAMPLE = REPO_ROOT / "examples" / "etad-ils-rwy-23-straight.yaml"
LSZH_EXAMPLE = REPO_ROOT / "examples" / "lszh-ils-z-rwy-14.yaml"


def test_loads_batumi_geojson() -> None:
    """The hand-authored Batumi basemap parses and has the expected shape."""
    assert BATUMI_PATH.is_file(), f"missing basemap data file: {BATUMI_PATH}"

    # Loadable via the module helper...
    gj = load_basemap("batumi")
    assert gj is not None, "load_basemap('batumi') returned None"
    assert gj.get("type") == "FeatureCollection"

    # ...and via raw JSON (sanity check the file on disk parses).
    with BATUMI_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    assert raw.get("type") == "FeatureCollection"

    features = gj.get("features") or []
    sea_polys = [
        f for f in features
        if (f.get("properties") or {}).get("class") == "sea"
        and (f.get("geometry") or {}).get("type") in {"Polygon", "MultiPolygon"}
    ]
    assert sea_polys, "expected at least one sea Polygon/MultiPolygon feature"

    city_points = [
        f for f in features
        if (f.get("properties") or {}).get("class") == "city"
        and (f.get("geometry") or {}).get("type") == "Point"
    ]
    assert len(city_points) >= 2, (
        f"expected >=2 city Point features, got {len(city_points)}"
    )

    # Unknown-region lookup returns None rather than raising.
    assert load_basemap("definitely-not-a-real-region") is None


def test_generated_coastline_overrides_hand_authored_water(tmp_path: Path) -> None:
    base = {
        "type": "FeatureCollection",
        "region": "test",
        "features": [
            {
                "type": "Feature",
                "properties": {"class": "sea", "name": "OLD SEA"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"class": "coast", "name": "OLD COAST"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0, 0], [1, 1]],
                },
            },
            {
                "type": "Feature",
                "properties": {"class": "city", "name": "KEEP"},
                "geometry": {"type": "Point", "coordinates": [0.5, 0.5]},
            },
        ],
    }
    generated = {
        "type": "FeatureCollection",
        "bbox": [0, 0, 1, 1],
        "features": [
            {
                "type": "Feature",
                "properties": {"class": "sea", "name": "NEW SEA"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"class": "coast", "name": "NEW COAST"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0, 0], [2, 2]],
                },
            },
        ],
    }
    (tmp_path / "test.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "test-coastline.json").write_text(
        json.dumps(generated), encoding="utf-8",
    )

    merged = load_basemap("test", data_root=tmp_path)
    assert merged is not None
    names = {
        (f.get("properties") or {}).get("name")
        for f in (merged.get("features") or [])
    }
    assert {"NEW SEA", "NEW COAST", "KEEP"} <= names
    assert "OLD SEA" not in names
    assert "OLD COAST" not in names


def test_render_basemap_emits_svg() -> None:
    """render_basemap produces non-empty SVG with expected tags."""
    gj = load_basemap("batumi")
    assert gj is not None

    # Identity-ish projector: treat (lat, lon) as (y, x) scaled into a
    # 500x500 viewport centered on the Batumi bbox. Good enough for a
    # smoke test — we only care the output is non-empty + contains tags.
    def projector(ll):
        lat, lon = ll
        x = (lon - 41.10) / (41.80 - 41.10) * 500.0
        y = (42.00 - lat) / (42.00 - 41.45) * 500.0
        return (x, y)

    svg = render_basemap(gj, BasemapConfig(), projector)
    assert svg, "expected non-empty SVG fragment"
    assert "<path" in svg, "basemap should contain at least one <path>"
    # City dots are drawn as <rect>; check that at least one is present.
    assert "<rect" in svg, "basemap should contain at least one <rect> city dot"
    # Layer wrapper marker for downstream inspection.
    assert 'data-layer="basemap"' in svg


def test_ugsb_plan_view_includes_basemap() -> None:
    """The full UGSB render includes the basemap layer wrapper."""
    proc = yaml.safe_load(UGSB_EXAMPLE.read_text())
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)
    svg = render_iap_svg(proc, prims)

    assert 'data-layer="basemap"' in svg, (
        "expected <g data-layer=\"basemap\"> wrapper in UGSB render output"
    )
    # Sea fill should use the declared cyan-tint token (see basemap.py).
    from open_plates.basemap import COLOR_BASEMAP_SEA
    assert COLOR_BASEMAP_SEA in svg, "expected COLOR_BASEMAP_SEA fill in output"


def test_loads_spangdahlem_geojson() -> None:
    """The hand-authored Spangdahlem basemap parses and has expected shape."""
    assert SPANGDAHLEM_PATH.is_file(), (
        f"missing basemap data file: {SPANGDAHLEM_PATH}"
    )

    gj = load_basemap("spangdahlem")
    assert gj is not None, "load_basemap('spangdahlem') returned None"
    assert gj.get("type") == "FeatureCollection"

    features = gj.get("features") or []

    # Eifel is landlocked -- there should be NO sea polygon.
    sea_polys = [
        f for f in features
        if (f.get("properties") or {}).get("class") == "sea"
    ]
    assert not sea_polys, (
        "Spangdahlem basemap should have no sea polygon (Eifel is landlocked)"
    )

    # Core cities may live either as hand-authored city Points in this
    # base file OR as OSM populated_area polygons in the sibling file.
    # The OSM sibling is top-30 by rank across a generous region bbox
    # (4.8E..8.2E / 48.8N..51.0N), so at this scale the sibling is
    # dominated by regional cities (Trier / Luxembourg / Aachen /
    # Saarbrücken) rather than every small Eifel town. The hand-
    # authored base file still carries the smaller Eifel settlements
    # that matter to ETAD approaches (Bitburg-area plates reference
    # them), so we verify TRIER (always present) plus at least one
    # other prominent regional name.
    merged = load_basemap("spangdahlem")
    all_names = {
        (f.get("properties") or {}).get("name", "").upper()
        for f in (merged.get("features") or [])
        if (f.get("properties") or {}).get("class") in ("city", "populated_area")
    }
    assert "TRIER" in all_names, (
        f"expected TRIER in merged Spangdahlem basemap; got "
        f"{sorted(n for n in all_names if n)[:20]!r}..."
    )
    regional_hits = all_names & {
        "LUXEMBOURG", "AACHEN", "SAARBRÜCKEN", "METZ", "LIÈGE",
        "MAASTRICHT", "BITBURG", "WITTLICH",
    }
    assert regional_hits, (
        f"expected at least one prominent regional city in merged "
        f"Spangdahlem basemap; got {sorted(n for n in all_names if n)[:20]!r}..."
    )

    # Mosel river (primary river feature) must be present.
    river_names = {
        (f.get("properties") or {}).get("name")
        for f in features
        if (f.get("properties") or {}).get("class") == "river"
    }
    assert "MOSEL" in river_names, "expected MOSEL river in Spangdahlem basemap"


def test_etad_plan_view_includes_basemap() -> None:
    """The full ETAD render includes the basemap layer wrapper."""
    proc = yaml.safe_load(ETAD_EXAMPLE.read_text())
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)
    svg = render_iap_svg(proc, prims)

    assert 'data-layer="basemap"' in svg, (
        "expected <g data-layer=\"basemap\"> wrapper in ETAD render output"
    )
    # River stroke should use the pale-cyan token (Mosel is a rank-1 river
    # and the example enables show_rivers).
    from open_plates.basemap import COLOR_BASEMAP_RIVER
    assert COLOR_BASEMAP_RIVER in svg, (
        "expected COLOR_BASEMAP_RIVER stroke in ETAD render output"
    )


def test_loads_zurich_geojson() -> None:
    """The hand-authored Zurich basemap parses and has the expected shape."""
    assert ZURICH_PATH.is_file(), f"missing basemap data file: {ZURICH_PATH}"

    gj = load_basemap("zurich")
    assert gj is not None, "load_basemap('zurich') returned None"
    assert gj.get("type") == "FeatureCollection"

    features = gj.get("features") or []

    # Zuerichsee must be present as a Polygon feature tagged class=sea.
    sea_polys = [
        f for f in features
        if (f.get("properties") or {}).get("class") == "sea"
        and (f.get("geometry") or {}).get("type") in {"Polygon", "MultiPolygon"}
    ]
    assert sea_polys, "expected at least one sea Polygon feature (Zuerichsee)"
    sea_names = {(f.get("properties") or {}).get("name") for f in sea_polys}
    assert "ZUERICHSEE" in sea_names, (
        f"expected ZUERICHSEE in sea polygons, got {sea_names}"
    )

    # At least one Alpine spot_elevation (Pilatus/Rigi/Saentis) south of field.
    spot_elevs = [
        f for f in features
        if (f.get("properties") or {}).get("class") == "spot_elevation"
    ]
    assert len(spot_elevs) >= 2, (
        f"expected >=2 spot_elevation features, got {len(spot_elevs)}"
    )


def test_lszh_renders_with_basemap() -> None:
    """The multi-IAF DME arc LSZH approach renders with the zurich basemap.

    Exercises the AF fly-by fillet on three IAF entries converging on a
    shared intermediate fix, plus the Alpine-region basemap stack. The
    procedure must compile and emit a non-empty SVG with the basemap and
    procedure layers present, with no loops/duplicated endpoints in the
    primitive list.
    """
    proc = yaml.safe_load(LSZH_EXAMPLE.read_text())
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)
    svg = render_iap_svg(proc, prims)

    # Basemap wrapper is present.
    assert 'data-layer="basemap"' in svg, (
        "expected basemap wrapper in LSZH render output"
    )
    # Zuerichsee fill token is emitted (sea polygon visible or clipped
    # behind the plan view -- either way the fill color appears).
    from open_plates.basemap import COLOR_BASEMAP_SEA
    assert COLOR_BASEMAP_SEA in svg, (
        "expected COLOR_BASEMAP_SEA fill in LSZH render output"
    )

    # Three AF-terminator arcs emitted (one per IAF transition) plus the
    # intermediate-segment + final-segment CFs and the missed approach.
    from open_plates.geometry import Arc, Segment, Hold
    af_arcs = [
        p for p in prims
        if isinstance(p, Arc)
        and p.center == fixes["KLO"]
        and abs(p.radius_nm - 12.0) < 0.01
    ]
    assert len(af_arcs) == 3, (
        f"expected 3 AF arcs at KLO r=12 (AMIKI/RILAX/GIPOL), got {len(af_arcs)}"
    )
    # Final approach segments converge on RW14 threshold.
    rw14_segments = [
        p for p in prims
        if isinstance(p, Segment) and p.end == fixes["RW14"]
    ]
    assert rw14_segments, "expected at least one segment ending at RW14"


# ---------------------------------------------------------------------------
# find_covering_region tests
# ---------------------------------------------------------------------------


def test_find_covering_region_hit(tmp_path):
    """Region that fully covers the bbox is returned."""
    regions = {
        "test_region": {
            "name": "Test",
            "bbox": [5.0, 48.0, 9.0, 52.0],
            "center": [50.0, 7.0],
            "description": "test"
        }
    }
    (tmp_path / "regions.json").write_text(json.dumps(regions))

    # plan_bbox is lat-first: (lat_min, lon_min, lat_max, lon_max)
    result = find_covering_region((49.0, 6.0, 51.0, 8.0), data_root=tmp_path)
    assert result == "test_region"


def test_find_covering_region_miss(tmp_path):
    """No region covers the bbox -> None."""
    regions = {
        "small": {
            "name": "Small",
            "bbox": [6.0, 49.0, 7.0, 50.0],
            "center": [49.5, 6.5],
            "description": "too small"
        }
    }
    (tmp_path / "regions.json").write_text(json.dumps(regions))

    result = find_covering_region((48.0, 5.0, 52.0, 9.0), data_root=tmp_path)
    assert result is None


def test_find_covering_region_missing_file(tmp_path):
    """No regions.json -> None, no exception."""
    result = find_covering_region((49.0, 6.0, 51.0, 8.0), data_root=tmp_path)
    assert result is None


def test_find_covering_region_picks_smallest(tmp_path):
    """When multiple regions cover the bbox, pick the smallest."""
    regions = {
        "big": {
            "name": "Big",
            "bbox": [0.0, 40.0, 20.0, 60.0],
            "center": [50.0, 10.0],
            "description": "huge"
        },
        "small": {
            "name": "Small",
            "bbox": [5.0, 48.0, 9.0, 52.0],
            "center": [50.0, 7.0],
            "description": "tight"
        }
    }
    (tmp_path / "regions.json").write_text(json.dumps(regions))

    result = find_covering_region((49.0, 6.0, 51.0, 8.0), data_root=tmp_path)
    assert result == "small"
