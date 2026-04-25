"""Tests for the region registry (``data/basemap/regions.json``).

The registry is the single source of truth for the generous per-region
bboxes the basemap builders target. These tests pin its shape, sanity-
check the bboxes, verify every hillshade sibling matches the registry,
and exercise the ``plan_bbox``-out-of-region warning path through
:func:`load_basemap`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from open_plates.basemap import load_basemap

REPO_ROOT = Path(__file__).resolve().parents[1]
BASEMAP_DIR = REPO_ROOT / "data" / "basemap"
REGIONS_PATH = BASEMAP_DIR / "regions.json"


def _load_registry() -> dict:
    assert REGIONS_PATH.is_file(), f"region registry missing: {REGIONS_PATH}"
    with REGIONS_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_regions_registry_loads() -> None:
    """The registry parses and every entry carries name/bbox/center."""
    registry = _load_registry()
    assert isinstance(registry, dict) and registry, (
        "registry must be a non-empty object"
    )
    for region_id, entry in registry.items():
        assert isinstance(region_id, str) and region_id
        assert isinstance(entry, dict), f"{region_id} entry must be an object"
        assert isinstance(entry.get("name"), str) and entry["name"], (
            f"{region_id}.name must be a non-empty string"
        )
        bbox = entry.get("bbox")
        assert isinstance(bbox, list) and len(bbox) == 4, (
            f"{region_id}.bbox must be [lon_min, lat_min, lon_max, lat_max]"
        )
        for v in bbox:
            assert isinstance(v, (int, float)), (
                f"{region_id}.bbox entries must be numeric"
            )
        center = entry.get("center")
        assert isinstance(center, list) and len(center) == 2, (
            f"{region_id}.center must be [lat, lon]"
        )
        for v in center:
            assert isinstance(v, (int, float))


def test_regions_bboxes_are_sane() -> None:
    """Each bbox lon span < 5 deg, lat span < 4 deg, and coordinates are
    inside valid WGS-84 ranges."""
    registry = _load_registry()
    for region_id, entry in registry.items():
        lon_min, lat_min, lon_max, lat_max = (float(v) for v in entry["bbox"])
        assert -180.0 <= lon_min < lon_max <= 180.0, (
            f"{region_id}: lon span invalid ({lon_min}..{lon_max})"
        )
        assert -90.0 <= lat_min < lat_max <= 90.0, (
            f"{region_id}: lat span invalid ({lat_min}..{lat_max})"
        )
        # Generosity ceiling — too-wide bboxes push Overpass timeouts
        # and blow up hillshade PNG sizes.
        assert lon_max - lon_min < 5.0, (
            f"{region_id}: lon span {lon_max - lon_min:.3f} >= 5 deg"
        )
        assert lat_max - lat_min < 4.0, (
            f"{region_id}: lat span {lat_max - lat_min:.3f} >= 4 deg"
        )
        # Registry center should be near-inside the bbox.
        clat, clon = (float(v) for v in entry["center"])
        assert lat_min - 0.5 <= clat <= lat_max + 0.5, (
            f"{region_id}: center lat {clat} outside bbox"
        )
        assert lon_min - 0.5 <= clon <= lon_max + 0.5, (
            f"{region_id}: center lon {clon} outside bbox"
        )


def test_all_hillshade_files_match_registry_bbox() -> None:
    """Every built hillshade sibling's bbox matches the registry to 0.01 deg.

    A mismatch means the region was rebuilt at a different extent than
    ``regions.json`` advertises — either the registry or the build is
    stale. The tolerance absorbs the SRTM tile-edge snap (manifest bbox
    is the actual cropped grid, which can land a hair off the requested
    bbox).
    """
    registry = _load_registry()
    tol = 0.01
    for region_id, entry in registry.items():
        hs_path = BASEMAP_DIR / f"{region_id}-hillshade.json"
        if not hs_path.is_file():
            pytest.skip(f"hillshade not built for {region_id}")
        with hs_path.open("r", encoding="utf-8") as fh:
            hs = json.load(fh)
        hs_bbox = hs.get("bbox")
        reg_bbox = entry["bbox"]
        assert isinstance(hs_bbox, list) and len(hs_bbox) == 4
        for i, (a, b) in enumerate(zip(hs_bbox, reg_bbox)):
            # Hillshade SRTM crops can pad a little OUTSIDE the requested
            # bbox (never shrink below it) — tolerate +-tol in either
            # direction so long as the axis-of-interest matches.
            assert abs(float(a) - float(b)) < tol, (
                f"{region_id}: hillshade bbox[{i}] {a} drifts from "
                f"registry {b} by more than {tol}"
            )


def test_plan_bbox_warning_fires_when_out_of_region(tmp_path: Path) -> None:
    """``load_basemap(plan_bbox=...)`` warns if plan extent escapes siblings."""
    # Build a minimal synthetic basemap rooted at a temp data_root so the
    # test is independent of the real data files. The sibling carries a
    # declared bbox that a plan_bbox can clearly escape.
    region = "regressiontest"
    base = tmp_path
    base.mkdir(exist_ok=True)
    (base / f"{region}.json").write_text(
        json.dumps({
            "type": "FeatureCollection",
            "region": region,
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "features": [],
        }),
        encoding="utf-8",
    )
    # ``-populated`` sibling uses lat-first ordering per the build scripts.
    (base / f"{region}-populated.json").write_text(
        json.dumps({
            "type": "FeatureCollection",
            "region": region,
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "features": [],
        }),
        encoding="utf-8",
    )

    # Plan bbox that clearly escapes to the north and east of the sibling.
    plan_bbox = (0.5, 0.5, 2.0, 2.0)  # (lat_min, lon_min, lat_max, lon_max)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        doc = load_basemap(region, data_root=base, plan_bbox=plan_bbox)
    assert doc is not None
    # At least one UserWarning mentioning the region should have fired.
    matching = [
        w for w in caught
        if issubclass(w.category, UserWarning) and region in str(w.message)
    ]
    assert matching, (
        "expected a UserWarning from load_basemap; got none. "
        f"All captured: {[str(w.message) for w in caught]}"
    )

    # And the inverse: a plan bbox wholly inside the sibling bbox must NOT
    # produce a warning.
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        load_basemap(
            region, data_root=base,
            plan_bbox=(0.1, 0.1, 0.9, 0.9),
        )
    inner_warns = [
        w for w in caught2
        if issubclass(w.category, UserWarning) and region in str(w.message)
    ]
    assert not inner_warns, (
        "plan_bbox fully inside sibling should not warn; "
        f"got: {[str(w.message) for w in inner_warns]}"
    )
