"""Smoke tests for the SVG renderer."""

from __future__ import annotations

from pathlib import Path

import yaml

from open_plates.legs import build_fix_table, compile_legs
from open_plates.projection import plan_projector
from open_plates.render_svg import render_iap_svg

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ugsb-ils-rwy-13.yaml"


def _load() -> dict:
    return yaml.safe_load(EXAMPLE.read_text())


def test_ugsb_renders() -> None:
    proc = _load()
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)
    svg = render_iap_svg(proc, prims)

    assert svg, "expected non-empty SVG"
    assert svg.startswith("<?xml"), "SVG must start with an XML declaration"
    assert "<svg" in svg, "must contain an <svg> element"
    assert "<path" in svg or "<polyline" in svg, "must contain vector path content"

    # Title strip content
    assert "UGSB" in svg
    assert "ILS RWY 13" in svg

    # Every referenced fix should appear as a label in the plan view.
    for fid in ("BLACK", "KOBUL", "BT13F", "RW13", "BTM"):
        assert fid in svg, f"fix label {fid} missing from SVG"

    # Minima should render DA value
    assert "232" in svg, "expected DA 232 in minima table"


def test_ugsb_render_is_deterministic() -> None:
    """Same input must produce byte-identical output."""
    proc = _load()
    a = render_iap_svg(proc)
    b = render_iap_svg(proc)
    assert a == b


def test_projector_preserves_order() -> None:
    """A fix east of another should project to a greater x coordinate;
    a fix north of another should project to a lesser y (SVG y grows down)."""
    proc = _load()
    fixes = build_fix_table(proc)
    projector = plan_projector(proc, fixes, (0.0, 0.0, 400.0, 300.0))

    # BLACK is NW, RW13 is SE.
    black = projector(fixes["BLACK"])
    rw13 = projector(fixes["RW13"])

    assert rw13[0] > black[0], "RW13 (east of BLACK) should have larger x"
    assert rw13[1] > black[1], "RW13 (south of BLACK) should have larger y"

    # BTM is south of KOBUL — check the lat axis.
    btm = projector(fixes["BTM"])
    kobul = projector(fixes["KOBUL"])
    assert btm[1] > kobul[1], "BTM (south of KOBUL) should have larger y"
