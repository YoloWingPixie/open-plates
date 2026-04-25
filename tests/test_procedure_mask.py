"""Tests for the procedure-bbox SVG mask.

The mask hides basemap linework / labels wherever a procedure element
would overlap them, while leaving big background fills (sea and
populated-area polygons) untouched.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from open_plates.legs import build_fix_table, compile_legs
from open_plates.projection import plan_projector
from open_plates.render_svg import (
    _PLAN_MASK_ID,
    _collect_procedure_bboxes,
    render_iap_svg,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ETAD_EXAMPLE = REPO_ROOT / "examples" / "etad-ils-rwy-23-straight.yaml"
UGSB_EXAMPLE = REPO_ROOT / "examples" / "ugsb-ils-rwy-13.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_mask_emitted_when_basemap_present() -> None:
    """ETAD declares a basemap; the render must emit the mask defs and the
    basemap-lines group must reference the mask by id."""
    proc = _load(ETAD_EXAMPLE)
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)
    svg = render_iap_svg(proc, prims)

    assert f'<mask id="{_PLAN_MASK_ID}"' in svg, (
        f"expected <mask id=\"{_PLAN_MASK_ID}\"> defs in ETAD render"
    )
    # The masked inner group must reference the mask by url(#...).
    mask_ref = f'mask="url(#{_PLAN_MASK_ID})"'
    assert mask_ref in svg, (
        f"expected {mask_ref!r} applied to the basemap-lines group"
    )
    # The split groups should both be present.
    assert 'data-layer="basemap-fills"' in svg
    assert 'data-layer="basemap-lines"' in svg


def test_mask_absent_when_no_basemap() -> None:
    """With the ``basemap:`` declaration removed, no mask defs are emitted
    (the masked split is only useful when a basemap is actually drawn)."""
    proc = copy.deepcopy(_load(UGSB_EXAMPLE))
    proc.pop("basemap", None)
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)
    svg = render_iap_svg(proc, prims)

    # Whole layer should be absent; the mask id should not appear either.
    assert 'data-layer="basemap"' not in svg
    assert _PLAN_MASK_ID not in svg


def test_bbox_collection_includes_fixes_and_alt_boxes() -> None:
    """``_collect_procedure_bboxes`` should produce at least one bbox per
    used fix (for the fix symbol) plus one per altitude-constrained fix.
    """
    proc = _load(ETAD_EXAMPLE)
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)

    # Match the region the plan view uses so projection math lines up.
    from open_plates.render_svg import MARGIN, PAGE_W, TITLE_H, BRIEFING_H, PLAN_H
    plan_region = (MARGIN, MARGIN + TITLE_H + BRIEFING_H,
                   PAGE_W - 2 * MARGIN, PLAN_H)
    projector = plan_projector(proc, fixes, plan_region, primitives=prims)

    bboxes = _collect_procedure_bboxes(
        proc, fixes, prims, projector, plan_region,
    )

    # Count the used fixes that actually project (same rule as the
    # renderer's iteration: id ∈ fixes AND _find_fix returns a dict).
    from open_plates.render_svg import (
        _collect_used_fix_ids,
        _find_altitude_constraint,
        _find_fix,
    )
    used = _collect_used_fix_ids(proc)
    n_fix_symbols = sum(
        1 for fid in used
        if fid in fixes and _find_fix(proc, fid) is not None
    )
    n_alt_boxes = sum(
        1 for fid in used
        if fid in fixes and _find_fix(proc, fid) is not None
        and _find_altitude_constraint(proc, fid) is not None
        and (_find_altitude_constraint(proc, fid) or {}).get("altitude_1_ft")
        is not None
    )

    assert n_fix_symbols > 0, "sanity check: expected some used/projected fixes"
    assert len(bboxes) >= n_fix_symbols + n_alt_boxes, (
        f"expected >= {n_fix_symbols + n_alt_boxes} bboxes "
        f"(fix symbols + alt boxes); got {len(bboxes)}"
    )

    # Every bbox is a (x, y, w, h) tuple with positive width/height.
    for bx in bboxes:
        assert len(bx) == 4
        assert bx[2] > 0 and bx[3] > 0


def test_bbox_collection_is_deterministic() -> None:
    """Two calls with the same inputs return identical output."""
    proc = _load(ETAD_EXAMPLE)
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)

    from open_plates.render_svg import MARGIN, PAGE_W, TITLE_H, BRIEFING_H, PLAN_H
    plan_region = (MARGIN, MARGIN + TITLE_H + BRIEFING_H,
                   PAGE_W - 2 * MARGIN, PLAN_H)
    projector = plan_projector(proc, fixes, plan_region, primitives=prims)

    a = _collect_procedure_bboxes(proc, fixes, prims, projector, plan_region)
    b = _collect_procedure_bboxes(proc, fixes, prims, projector, plan_region)
    assert a == b
