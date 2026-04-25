"""Tests for the ARINC 424 AF (Arc-to-Fix) leg terminator."""

from __future__ import annotations

import pytest

from open_plates.geodesy import destination, great_circle_distance_nm
from open_plates.geometry import Arc, Segment
from open_plates.legs import build_fix_table, compile_legs


def _proc_with_af(turn_direction: str) -> dict:
    """Build a minimal procedure: IF at arc-entry, AF to EXIT.

    Centers the arc at ``C`` (the origin). ``ENTRY`` sits 10 NM due east
    of C on the R-090 radial; ``EXIT`` sits 10 NM due north of C on the
    R-360 radial. A 10 DME arc from ENTRY to EXIT going
    counterclockwise traces a quarter-circle over the NE quadrant.
    """
    c = (0.0, 0.0)
    entry = destination(c, 90.0, 10.0)
    exit_fix = destination(c, 0.0, 10.0)
    return {
        "fixes": [
            {"id": "C", "type": "named_fix", "lat": c[0], "lon": c[1]},
            {"id": "ENTRY", "type": "named_fix",
             "lat": entry[0], "lon": entry[1]},
            {"id": "EXIT", "type": "named_fix",
             "lat": exit_fix[0], "lon": exit_fix[1]},
        ],
        "common_legs": [
            {"terminator": "IF", "transition_type": "common",
             "to_fix_id": "ENTRY"},
            {"terminator": "AF", "transition_type": "common",
             "from_fix_id": "ENTRY", "to_fix_id": "EXIT",
             "arc_center_fix_id": "C", "arc_radius_nm": 10.0,
             "turn_direction": turn_direction},
        ],
    }


def test_af_emits_arc_primitive() -> None:
    proc = _proc_with_af(turn_direction="left")
    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)

    arcs = [p for p in prims if isinstance(p, Arc)]
    assert len(arcs) == 1
    arc = arcs[0]
    assert arc.center == fixes["C"]
    assert arc.radius_nm == pytest.approx(10.0)
    assert arc.clockwise is False
    assert arc.start_bearing_deg == pytest.approx(90.0, abs=0.5)
    assert arc.end_bearing_deg == pytest.approx(0.0, abs=0.5)


def test_af_clockwise_vs_counterclockwise_sweep_direction() -> None:
    fixes_ccw = build_fix_table(_proc_with_af(turn_direction="left"))
    prims_ccw = compile_legs(_proc_with_af(turn_direction="left"), fixes_ccw)
    fixes_cw = build_fix_table(_proc_with_af(turn_direction="right"))
    prims_cw = compile_legs(_proc_with_af(turn_direction="right"), fixes_cw)

    arc_ccw = next(p for p in prims_ccw if isinstance(p, Arc))
    arc_cw = next(p for p in prims_cw if isinstance(p, Arc))

    assert arc_ccw.clockwise is False
    assert arc_cw.clockwise is True
    assert arc_ccw.center == arc_cw.center
    assert arc_ccw.radius_nm == pytest.approx(arc_cw.radius_nm)


def test_af_context_updates_to_end_fix() -> None:
    """After an AF leg, a subsequent TF leg should emit a Segment
    starting at the AF's to_fix."""
    proc = _proc_with_af(turn_direction="left")
    exit_ll = (proc["fixes"][2]["lat"], proc["fixes"][2]["lon"])
    far = destination(exit_ll, 270.0, 5.0)
    proc["fixes"].append(
        {"id": "FAR", "type": "named_fix", "lat": far[0], "lon": far[1]},
    )
    proc["common_legs"].append(
        {"terminator": "TF", "transition_type": "common",
         "from_fix_id": "EXIT", "to_fix_id": "FAR", "course_deg": 270.0},
    )

    fixes = build_fix_table(proc)
    prims = compile_legs(proc, fixes)

    segs = [p for p in prims if isinstance(p, Segment)]
    assert segs
    final_seg = segs[-1]
    assert final_seg.end == fixes["FAR"]
    # Dispatcher does NOT apply fillet setback any more — segment starts
    # exactly at EXIT.
    assert great_circle_distance_nm(final_seg.start, fixes["EXIT"]) < 1e-6
