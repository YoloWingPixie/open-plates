"""Smoke tests for the SVG renderer."""

from __future__ import annotations

from pathlib import Path

import yaml

from open_plates.legs import build_fix_table, compile_legs
from open_plates.geodesy import destination, initial_bearing_deg
from open_plates.placement import BBox, LabelCandidate
from open_plates.projection import plan_projector
from open_plates.render_svg import (
    _BASEMAP_LABEL_TOKEN,
    _PlanLabelCtx,
    _layout_regions,
    _plan_view_placement_trace,
    render_iap_svg,
)

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


def test_render_does_not_write_debug_stdout(capsys) -> None:
    proc = _load()
    render_iap_svg(proc)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_deferred_basemap_labels_paint_before_procedure_labels() -> None:
    """Basemap labels participate in placement but stay in the basemap layer."""
    from open_plates.basemap import BasemapConfig, render_basemap

    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "city-testville",
                "properties": {
                    "class": "city",
                    "name": "Testville",
                    "rank": 1,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [0.0, 0.0],
                },
            },
        ],
    }

    def projector(_latlon):
        return (100.0, 100.0)

    ctx = _PlanLabelCtx(BBox(0.0, 0.0, 200.0, 200.0))
    with ctx:
        basemap_svg = render_basemap(
            fc,
            BasemapConfig(),
            projector,
            placement_ctx=ctx,
            deferred_label_token=_BASEMAP_LABEL_TOKEN,
        )
        ctx.add_label_request(
            feature_id="proc:label",
            anchor=(40.0, 40.0),
            content_size=(20.0, 10.0),
            candidates=(LabelCandidate(0.0, 0.0),),
            fallback="suppress",
            clearance=0.0,
            render=lambda _placed: "<text>PROC_LABEL</text>\n",
        )
        ctx.resolve()
        svg = basemap_svg.replace(
            _BASEMAP_LABEL_TOKEN, ctx.emit_layer("basemap"),
        )
        svg += ctx.emit_layer("procedure")

    assert "TESTVILLE" in svg
    assert svg.index("TESTVILLE") < svg.index("PROC_LABEL")


def test_final_localizer_course_label_sits_past_feather_tail() -> None:
    proc = _load()
    trace = _plan_view_placement_trace(proc) or []
    placed = next(
        p for p in trace
        if p.request.feature_id == "course:BT13F->RW13"
    )
    assert placed.bbox.w > 0
    assert placed.leader_line is None

    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)
    projector = plan_projector(
        proc, fixes, _layout_regions()["plan"], primitives=prims,
    )
    threshold_xy = projector(fixes["RW13"])
    final_course = initial_bearing_deg(fixes["BT13F"], fixes["RW13"])
    feather_far = destination(fixes["RW13"], (final_course + 180.0) % 360.0, 8.5)
    feather_far_xy = projector(feather_far)
    dx = feather_far_xy[0] - threshold_xy[0]
    dy = feather_far_xy[1] - threshold_xy[1]
    length = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / length, dy / length
    tail = (feather_far_xy[0] + ux * 12.0, feather_far_xy[1] + uy * 12.0)
    label_center = (
        placed.bbox.x + placed.bbox.w / 2.0,
        placed.bbox.y + placed.bbox.h / 2.0,
    )
    along_tail = (
        (label_center[0] - tail[0]) * ux
        + (label_center[1] - tail[1]) * uy
    )
    cross_tail = abs(
        (label_center[0] - tail[0]) * -uy
        + (label_center[1] - tail[1]) * ux
    )
    assert along_tail > 0.0
    assert cross_tail < 0.5


def test_profile_marks_minima_with_vertical_dashed_reference() -> None:
    proc = _load()
    svg = render_iap_svg(proc)

    marker = 'data-profile-minima="DA"'
    assert marker in svg
    marker_idx = svg.index(marker)
    line_idx = svg.index("<line", marker_idx)
    line_end = svg.index("/>", line_idx)
    line = svg[line_idx:line_end]
    assert 'stroke-dasharray="3 2"' in line

    attrs = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        attrs[key] = value.strip('"')
    assert attrs["x1"] == attrs["x2"]
    assert float(attrs["y1"]) > float(attrs["y2"])
    assert "DA 232' / DH 200'" in svg


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
