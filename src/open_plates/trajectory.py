"""Symbolic chart trajectory builder.

Real chart production (Jeppesen / LIDO / FAA) separates flight physics
from chart symbology:

- Flight physics (actual turn radius, bank angle, performance) is used
  for procedure design and TERPS / PANS-OPS protection-area sizing, not
  for drawing the chart.
- The path drawn on an approach plate is a STYLIZED, deliberately
  schematic depiction of the procedure's intent. It communicates which
  fixes are flown and in what order, not the aircraft's minute-by-minute
  physical trajectory.

Concretely:
  * Fly-by waypoints render as a small tangent-arc rounded bend of a
    CONSTANT visual radius (~0.1-0.2 NM at plan scale), independent of
    aircraft speed. The chart reader sees a smooth curve through the
    waypoints with no visible "turn loop".
  * DME arcs (ARINC 424 AF) and radius-to-fix segments (RF) are the only
    true arcs — those are published chart features and render at their
    declared radius.
  * Fly-over waypoints (holds, climb-to altitude, etc.) are overflown:
    the rendered path actually touches the waypoint with no bend applied.

This module takes a flat list of :class:`TrajectoryWaypoint` produced by
the ARINC 424 leg dispatcher and returns a single continuous polyline
(``list[LatLon]``) ready for the renderer. The bend at every fly-by
waypoint uses ``visual_bend_radius_nm`` (chart-style constant) — it does
NOT depend on speed, bank angle, or turn rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians
from typing import Literal

from .geodesy import (
    LatLon,
    destination,
    great_circle_distance_nm,
    initial_bearing_deg,
)
from .tangent_arc import (
    arc_to_line_fillet,
    flyby_fillet_line_to_line,
    line_to_arc_fillet,
)

# Constant cosmetic bend radius applied at every fly-by waypoint.
# 0.5 NM at plan-view scale (~20 px/NM) is ~10 px — a clearly visible
# rounded bend that reads as a turn, not just a kink. Per research into
# real chart-production pipelines, the aesthetic radius sits in the
# 0.3-0.8 NM band regardless of actual aircraft turn physics — Jepp /
# LIDO / FAA all treat this as a chart-style constant, not a flight
# calculation.
DEFAULT_VISUAL_BEND_NM: float = 1.5

# Per research: clamp the fillet so the setback never eats more than
# this fraction of either adjacent segment. For tiny legs (short hops
# between dense waypoints) the fillet gracefully shrinks instead of
# overshooting the segment.
MAX_SETBACK_FRACTION_OF_SEG: float = 0.40

# Absolute setback cap. At sharp turn angles (θ > 120°) the aviation
# anticipation formula d = r·tan(θ/2) blows up: a 0.5 NM radius at θ=146°
# gives 1.64 NM setback, which drifts the polyline too far from the fix
# to read as "passed through here". Capping setback to 0.75 NM forces
# the radius to shrink at sharp turns (for 146°: r = 0.75/3.27 = 0.23 NM).
MAX_SETBACK_ABS_NM: float = 1.5

# Default sample count for a DME arc flattened to a polyline. 48 samples
# along a typical quarter-to-half-circle 10-12 NM arc keeps the chord
# error well under a chart pixel.
DEFAULT_ARC_SAMPLES: int = 48

# Minimum turn angle that gets a visible rounded bend. Below this the
# join is effectively straight and we skip the bend entirely.
_STRAIGHT_THROUGH_DEG: float = 1.0


WaypointKind = Literal["flyby", "flyover"]


@dataclass(frozen=True)
class TrajectoryWaypoint:
    """One waypoint in a symbolic chart trajectory.

    Attributes
    ----------
    position:
        (lat, lon) of the waypoint.
    kind:
        ``"flyby"`` — the rendered path rounds the corner just before
        this point with a constant visual bend radius.
        ``"flyover"`` — the rendered path passes through this point
        exactly (used for hold fixes, CA altitude-terminating points,
        and any other overflown anchor).
    arc_center, arc_radius_nm, arc_clockwise:
        If populated, the OUTBOUND leg from this waypoint is itself a
        DME / constant-radius arc around ``arc_center`` at
        ``arc_radius_nm``. The trajectory builder draws the polyline
        around the arc from this waypoint to the NEXT waypoint's
        position (which must lie on the same circle) along the declared
        sweep direction.
    is_arc_exit:
        True on the waypoint where an arc ENDS. Used by the builder to
        know the inbound track at this waypoint is a circular arc (not
        a straight), so the rounded-bend fillet should not try to clip
        the arc's end — it simply starts the outbound straight at the
        declared exit position.
    """

    position: LatLon
    kind: WaypointKind = "flyby"
    arc_center: LatLon | None = None
    arc_radius_nm: float | None = None
    arc_clockwise: bool | None = None
    is_arc_exit: bool = False


def _sample_arc(
    center: LatLon,
    radius_nm: float,
    start_pt: LatLon,
    end_pt: LatLon,
    clockwise: bool,
    samples: int,
) -> list[LatLon]:
    """Sample a circular arc around ``center`` from ``start_pt`` to ``end_pt``.

    Authored coordinates often have sub-NM radial mismatch from the
    declared arc radius (e.g. LSZH's AMIKI at 12.003 NM from KLO when
    the declared radius is 12.0 NM). We take the declared BEARING from
    center to each declared endpoint but pin the ACTUAL arc sampled
    points to the declared ``radius_nm``. To keep the polyline flush
    with the declared endpoints, we substitute the exact ``start_pt``
    and ``end_pt`` for the first/last samples rather than emitting the
    snapped-to-radius points at those bearings.

    Returns ``samples + 1`` points along the swept arc from start to end
    inclusive.
    """
    start_brg = initial_bearing_deg(center, start_pt)
    end_brg = initial_bearing_deg(center, end_pt)
    if clockwise:
        sweep = (end_brg - start_brg + 360.0) % 360.0
    else:
        sweep = -((start_brg - end_brg + 360.0) % 360.0)
    if abs(sweep) < 1e-9:
        return [start_pt]
    pts: list[LatLon] = [start_pt]
    for i in range(1, samples):
        t = i / samples
        brg = (start_brg + sweep * t) % 360.0
        pts.append(destination(center, brg, radius_nm))
    pts.append(end_pt)
    return pts


def _append_dedup(
    acc: list[LatLon], pts: list[LatLon], tol_nm: float = 1e-4
) -> None:
    """Append ``pts`` to ``acc``, deduping the first point against the
    last existing point within ``tol_nm``. 1e-4 NM ≈ 0.6 ft — below
    chart pixel resolution, above numerical round-trip error through
    ``destination(center, bearing, r)``.
    """
    if not pts:
        return
    if acc:
        last = acc[-1]
        first = pts[0]
        if great_circle_distance_nm(last, first) < tol_nm:
            pts = pts[1:]
    acc.extend(pts)


def _rounded_bend(
    inbound_course_deg: float,
    fix: LatLon,
    outbound_course_deg: float,
    radius_nm: float,
) -> tuple[LatLon, list[LatLon], LatLon] | None:
    """Construct the small rounded-bend at a fly-by waypoint.

    Uses the classical tangent-arc fillet but with a CONSTANT cosmetic
    ``radius_nm`` — no speed dependence. Returns
    ``(tangent_in, arc_samples, tangent_out)``, or ``None`` if the turn
    is below ``_STRAIGHT_THROUGH_DEG`` (effectively straight through).

    ``inbound_course_deg`` is the direction of flight arriving at the
    fix (0-360). Passing it directly avoids the numerical instability
    of deriving an inbound bearing from two very-close points, which
    matters when the inbound leg is an arc whose last sample sits <0.01
    NM from the declared exit waypoint.
    """
    # Synthesize a "prev point" 1 NM back along the inbound reciprocal
    # so the fillet helper can compute the inbound direction from it.
    inbound_reciprocal = (inbound_course_deg + 180.0) % 360.0
    prev_pt = destination(fix, inbound_reciprocal, 1.0)
    result = flyby_fillet_line_to_line(
        prev_start=prev_pt,
        prev_end=fix,
        fix=fix,
        outbound_course_deg=outbound_course_deg,
        turn_radius_nm=radius_nm,
    )
    if result is None:
        return None
    tangent_in, arc, tangent_out = result
    # Sample the arc with just a few points — the bend is tiny (~0.15 NM
    # radius = ~3 px at plan scale) so 5 samples reads smooth without
    # bloating the polyline.
    samples = 5
    start_brg = arc.start_bearing_deg
    end_brg = arc.end_bearing_deg
    if arc.clockwise:
        sweep = (end_brg - start_brg + 360.0) % 360.0
    else:
        sweep = -((start_brg - end_brg + 360.0) % 360.0)
    pts: list[LatLon] = []
    for i in range(samples + 1):
        t = i / samples
        brg = (start_brg + sweep * t) % 360.0
        pts.append(destination(arc.center, brg, arc.radius_nm))
    return (tangent_in, pts, tangent_out)


def build_trajectory(
    waypoints: list[TrajectoryWaypoint],
    visual_bend_radius_nm: float = DEFAULT_VISUAL_BEND_NM,
    arc_sample_count: int = DEFAULT_ARC_SAMPLES,
) -> list[LatLon]:
    """Build a continuous smooth polyline through ``waypoints``.

    Rules
    -----
    * For sequential straight-through fly-by waypoints (A, B, C): clip
      the inbound and outbound courses by ``visual_bend_radius_nm`` of
      setback around B and emit a small tangent arc through the rounded
      bend. Classical fly-by fillet with a CONSTANT cosmetic radius
      (not speed-derived).
    * For a waypoint whose OUTBOUND leg is a DME arc (``arc_center`` /
      ``arc_radius_nm`` / ``arc_clockwise`` populated): emit the arc
      itself sampled at ``arc_sample_count`` points along its swept arc
      from this waypoint's position to the next waypoint's position.
    * For a waypoint tagged ``is_arc_exit``: the inbound track into this
      waypoint is the tail of an arc. The declared position is used
      exactly (no bend applied against the arc tail — the arc itself is
      the smooth inbound). The outbound leg picks up as a straight (or
      another arc) from the declared position.
    * For fly-over waypoints (``kind="flyover"``): the path touches the
      waypoint position exactly, no rounding.

    The output is a single flat list of (lat, lon) points. The first
    point is ``waypoints[0].position`` and the last point is
    ``waypoints[-1].position`` (or the declared arc endpoint, if the
    last waypoint's inbound was an arc).

    Empty input → empty output. A single waypoint → a single-point list
    (the renderer will skip a <polyline> under 2 points).
    """
    if not waypoints:
        return []
    if len(waypoints) == 1:
        return [waypoints[0].position]

    out: list[LatLon] = [waypoints[0].position]
    # When a line→arc fillet fires at iteration i, the outbound arc at
    # iteration i+1 must start at T_arc (not cur.position). Stash the
    # override here and consume it in the arc-sampling branch next pass.
    pending_arc_start: LatLon | None = None

    def _outbound_course_at(i: int) -> float | None:
        """Outbound course from waypoint i's position. None if i is last."""
        if i + 1 >= len(waypoints):
            return None
        cur = waypoints[i]
        nxt = waypoints[i + 1]
        if cur.arc_center is not None and cur.arc_clockwise is not None:
            # Outbound leg is an arc — tangent at cur.position is
            # perpendicular to the radial.
            radial = initial_bearing_deg(cur.arc_center, cur.position)
            if cur.arc_clockwise:
                return (radial + 90.0) % 360.0
            return (radial - 90.0) % 360.0
        return initial_bearing_deg(cur.position, nxt.position)

    def _inbound_course_at(i: int) -> float | None:
        """Inbound course arriving at waypoint i. None if i is first."""
        if i == 0:
            return None
        prev = waypoints[i - 1]
        cur = waypoints[i]
        if prev.arc_center is not None and prev.arc_clockwise is not None:
            # Inbound leg is an arc — tangent at cur.position (arc exit)
            # is perpendicular to the exit radial in flight direction.
            radial = initial_bearing_deg(prev.arc_center, cur.position)
            if prev.arc_clockwise:
                return (radial + 90.0) % 360.0
            return (radial - 90.0) % 360.0
        return initial_bearing_deg(prev.position, cur.position)

    for i in range(1, len(waypoints)):
        prev = waypoints[i - 1]
        cur = waypoints[i]
        is_last = i == len(waypoints) - 1

        # Determine inbound direction into cur (for bend construction)
        # and outbound direction leaving cur.
        inbound_course = _inbound_course_at(i)
        outbound_course = _outbound_course_at(i)

        prev_was_arc = (
            prev.arc_center is not None and prev.arc_clockwise is not None
        )
        cur_starts_arc = (
            cur.arc_center is not None and cur.arc_clockwise is not None
        )

        # --- Line-to-arc fillet at cur ------------------------------------
        # Mirror of arc-to-line: the inbound to cur is a STRAIGHT line and
        # the outbound from cur is a DME arc. The declared junction rarely
        # sits naturally tangent to the arc (e.g. ETAD ARC1W, 91° off), so
        # we blend the two with an internal-tangency fillet of radius
        # `visual_bend_radius_nm`. T_line sits BEFORE cur on the inbound
        # straight; T_arc sits AFTER cur along the big arc. That means:
        #   - the inbound straight stops short at T_line,
        #   - the fillet arc is sampled from T_line to T_arc (smoothly
        #     crossing "cur"),
        #   - the outbound big arc is SAMPLED starting at T_arc (not
        #     cur.position) on the next iteration. We stash T_arc in
        #     `pending_arc_start` to hand it off to the arc-sampling
        #     branch next pass.
        line_to_arc: tuple[LatLon, list[LatLon], LatLon] | None = None
        if (
            cur_starts_arc
            and not prev_was_arc
            and cur.kind == "flyby"
            and not is_last
            and inbound_course is not None
        ):
            f = line_to_arc_fillet(
                line_entry_pt=prev.position,
                inbound_course_deg=inbound_course,
                arc_center=cur.arc_center,
                arc_radius_nm=cur.arc_radius_nm,
                arc_clockwise=cur.arc_clockwise,
                arc_entry_pt=cur.position,
                fillet_radius_nm=visual_bend_radius_nm,
            )
            if f is not None:
                t_line_pt, fillet, t_arc_pt = f
                fillet_samples = 12
                start_brg = fillet.start_bearing_deg
                end_brg = fillet.end_bearing_deg
                if fillet.clockwise:
                    sweep = (end_brg - start_brg + 360.0) % 360.0
                else:
                    sweep = -((start_brg - end_brg + 360.0) % 360.0)
                f_pts: list[LatLon] = [t_line_pt]
                for s in range(1, fillet_samples):
                    t = s / fillet_samples
                    brg = (start_brg + sweep * t) % 360.0
                    f_pts.append(destination(fillet.center, brg, fillet.radius_nm))
                f_pts.append(t_arc_pt)
                line_to_arc = (t_line_pt, f_pts, t_arc_pt)

        # --- Arc-to-line fillet at cur ------------------------------------
        # Special joint: the inbound to cur is a DME arc (prev carries arc
        # metadata) and the outbound from cur is a straight. The arc's
        # natural tangent at cur rarely matches the declared outbound
        # course, so we blend the two with an internal-tangency fillet of
        # radius `visual_bend_radius_nm`. The fillet's T_arc sits BEFORE
        # cur on the big arc; T_line sits AFTER cur on the outbound line.
        # That means:
        #   - the big arc is SAMPLED from prev.position to T_arc
        #     (stopping short of cur),
        #   - the fillet arc is sampled between T_arc and T_line
        #     (smoothly crossing "cur"),
        #   - the outbound straight picks up at T_line.
        arc_to_line: tuple[LatLon, list[LatLon], LatLon] | None = None
        if (
            prev_was_arc
            and cur.kind == "flyby"
            and not is_last
            and cur.arc_center is None
            and outbound_course is not None
        ):
            f = arc_to_line_fillet(
                arc_center=prev.arc_center,
                arc_radius_nm=prev.arc_radius_nm,
                arc_clockwise=prev.arc_clockwise,
                arc_exit_pt=cur.position,
                outbound_course_deg=outbound_course,
                fillet_radius_nm=visual_bend_radius_nm,
            )
            if f is not None:
                t_arc_pt, fillet, t_line_pt = f
                # Sample the fillet arc. A few samples are enough —
                # fillet radius is ~1.5 NM and typical joint turn angle
                # is modest, so ~12 samples reads smooth.
                fillet_samples = 12
                start_brg = fillet.start_bearing_deg
                end_brg = fillet.end_bearing_deg
                if fillet.clockwise:
                    sweep = (end_brg - start_brg + 360.0) % 360.0
                else:
                    sweep = -((start_brg - end_brg + 360.0) % 360.0)
                f_pts: list[LatLon] = [t_arc_pt]
                for s in range(1, fillet_samples):
                    t = s / fillet_samples
                    brg = (start_brg + sweep * t) % 360.0
                    f_pts.append(destination(fillet.center, brg, fillet.radius_nm))
                f_pts.append(t_line_pt)
                arc_to_line = (t_arc_pt, f_pts, t_line_pt)

        # Does cur get a rounded line-to-line bend?
        # - Not if it's fly-over (exact touch required).
        # - Not if it's the last waypoint (nothing to bend into).
        # - Not if outbound is an arc (the arc's own tangent handles
        #   the transition; the authored procedure is expected to be
        #   designed with the inbound leg already tangent, or to show a
        #   crisp visual corner if not).
        # - Not if inbound is an arc — the arc-to-line fillet above
        #   handles that case geometrically. The classical line-to-line
        #   fillet would set tangent_in BEHIND cur on a straight inbound
        #   line, which is wrong when the inbound is curved.
        needs_bend = (
            cur.kind == "flyby"
            and not is_last
            and inbound_course is not None
            and outbound_course is not None
            and cur.arc_center is None
            and not prev_was_arc
        )

        # --- Emit the INBOUND leg to cur ----------------------------------
        # Case A: inbound is an arc (prev started one). Sample it.
        if prev.arc_center is not None and prev.arc_clockwise is not None:
            # If an arc-to-line fillet is active, STOP the big arc at
            # T_arc instead of cur.position — that's where the fillet
            # picks up. Otherwise sample all the way to cur.position.
            arc_end_pt = arc_to_line[0] if arc_to_line is not None else cur.position
            # If the PREVIOUS iteration fired a line→arc fillet, start
            # sampling the big arc at T_arc (stashed in pending_arc_start),
            # not at prev.position — the fillet's T_arc lies on the circle
            # slightly past prev.position along the arc.
            arc_start_pt = pending_arc_start if pending_arc_start is not None else prev.position
            pending_arc_start = None
            arc_pts = _sample_arc(
                center=prev.arc_center,
                radius_nm=prev.arc_radius_nm,
                start_pt=arc_start_pt,
                end_pt=arc_end_pt,
                clockwise=prev.arc_clockwise,
                samples=arc_sample_count,
            )
            _append_dedup(out, arc_pts)
        # Case B: inbound is a straight. The segment is implicit: out[-1]
        # is the prev render endpoint, and we'll add cur (or the bend's
        # tangent_in) next. Nothing to append yet.

        # --- Emit the line-to-arc fillet + handoff, or arc-to-line, or bend, or touch

        if line_to_arc is not None:
            # Fillet samples span T_line → T_arc through (past) cur. The
            # inbound straight from out[-1] → T_line is implicit: appending
            # T_line as the first fillet sample closes it. The outbound
            # arc will be sampled next iteration starting at T_arc.
            _, f_pts, t_arc_pt = line_to_arc
            _append_dedup(out, f_pts)
            pending_arc_start = t_arc_pt
            continue

        if arc_to_line is not None:
            # fillet arc samples span T_arc → T_line through (past) cur.
            _, f_pts, t_line_pt = arc_to_line
            _append_dedup(out, f_pts)
            # The outbound straight now starts at T_line. The next
            # iteration's cur.position (= nxt) is the real next waypoint;
            # we already placed T_line, so the implicit straight from
            # T_line to nxt.position will render correctly.
            continue

        if not needs_bend:
            # Either flyover, terminus, or near-straight — place exactly.
            _append_dedup(out, [cur.position])
            continue

        # Apply rounded bend at cur — pass the exact inbound/outbound
        # courses (avoids numeric instability when the inbound is an
        # arc whose last sample sits microscopic distance from cur).
        # Radius gets clamped in three ways:
        #   (1) absolute setback cap — for sharp turns, d = r·tan(θ/2)
        #       blows up; cap the maximum setback so the polyline never
        #       drifts more than MAX_SETBACK_ABS_NM from the fix
        #   (2) fraction of adjacent straight segment (40%) — stops the
        #       bend from overshooting short hops
        #   (3) the base visual_bend_radius_nm ceiling
        from math import radians as _rad, tan as _tan
        bend_radius = visual_bend_radius_nm
        delta = _outbound_course_at(i) - _inbound_course_at(i)
        delta = (delta + 540.0) % 360.0 - 180.0
        abs_delta = abs(delta)
        if abs_delta >= _STRAIGHT_THROUGH_DEG:
            half_tan = max(_tan(_rad(abs_delta / 2.0)), 1e-6)
            # (1) absolute setback cap — keeps sharp-turn polylines near the fix
            max_radius_from_setback = MAX_SETBACK_ABS_NM / half_tan
            bend_radius = min(bend_radius, max_radius_from_setback)
            # (2) fraction of adjacent straight legs
            nxt = waypoints[i + 1]
            if prev.arc_center is None:
                inbound_len = great_circle_distance_nm(prev.position, cur.position)
                max_setback = inbound_len * MAX_SETBACK_FRACTION_OF_SEG
                bend_radius = min(bend_radius, max_setback / half_tan)
            if cur.arc_center is None:
                outbound_len = great_circle_distance_nm(cur.position, nxt.position)
                max_setback = outbound_len * MAX_SETBACK_FRACTION_OF_SEG
                bend_radius = min(bend_radius, max_setback / half_tan)

        bend = _rounded_bend(
            inbound_course_deg=inbound_course,
            fix=cur.position,
            outbound_course_deg=outbound_course,
            radius_nm=bend_radius,
        )
        if bend is None:
            # Turn below epsilon — treat as straight through.
            _append_dedup(out, [cur.position])
            continue

        tangent_in, arc_pts, tangent_out = bend
        _append_dedup(out, [tangent_in])
        _append_dedup(out, arc_pts)
        _append_dedup(out, [tangent_out])

    return out


__all__ = [
    "DEFAULT_ARC_SAMPLES",
    "DEFAULT_VISUAL_BEND_NM",
    "TrajectoryWaypoint",
    "WaypointKind",
    "build_trajectory",
]
