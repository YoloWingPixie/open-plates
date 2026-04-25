"""ARINC 424 leg-terminator dispatcher.

Walks a procedure's transitions + common_legs + missed_approach in
document order. Produces two outputs:

1. :func:`compile_legs` — a flat list of drawable primitives
   (Segment / Arc / Hold). Kept for bbox / projector / procedure-mask
   code that depends on knowing the set of line segments and arcs on
   the chart but does NOT care about rendering fidelity at joints.
   Waypoints render as vertex-joined segments; DME arcs (AF) render
   as true Arcs; holds as Hold primitives.

2. :func:`compile_flight_states` — the rendered path. Each state's
   waypoints are pushed through :func:`trajectory.build_trajectory`,
   which handles fly-by bends, fly-over waypoints, and DME arcs with
   a CONSTANT cosmetic bend radius. This is the polyline the renderer
   draws.

Design rationale: chart symbology is distinct from flight physics. A
real Jeppesen / LIDO chart does NOT render a speed-derived standard-rate
turn radius at every fly-by waypoint — that would put a ~3 NM lobe at
every joint on a missed approach. Instead, the bend at every fly-by
waypoint is a small constant (~0.15 NM at plan scale), and the chart
communicates INTENT, not minute-by-minute physical aircraft position.

v1 implements the terminators the hand-authored examples use: IF, TF,
CF, DF, CA, AF, HM. Remaining terminators (VA, FA, CD, VD, CR, VR, CI,
VI, RF, HA, HF, PI) are registered in the dispatch table but raise
NotImplementedError, keeping them discoverable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .geodesy import LatLon, destination, initial_bearing_deg
from .geometry import Arc, FlightState, Hold, Primitive, Segment
from .trajectory import (
    DEFAULT_ARC_SAMPLES,
    DEFAULT_VISUAL_BEND_NM,
    TrajectoryWaypoint,
    build_trajectory,
)

# CA legs have no named endpoint. Render as a fixed-length segment
# along the commanded course so the chart reader sees a stub arrow out
# of the missed-approach point.
CA_LEG_RENDER_LENGTH_NM: float = 2.0
# Hold leg-time fallback TAS when only a time is published.
HOLD_DEFAULT_TAS_KT: float = 210.0


@dataclass
class LegContext:
    """Running state threaded through the dispatcher.

    ``position`` is the current aircraft position (end of the previous
    leg, or the IF fix). ``course_deg`` is the track departing
    ``position``; None before any course-bearing leg has been issued.

    Kept for backwards compatibility with callers that pass a
    ``LegContext`` around, but no longer carries speed / turn-radius
    metadata — those are chart-style constants now, not per-phase
    state.
    """

    position: LatLon | None = None
    course_deg: float | None = None


Handler = Callable[
    [dict, LegContext, dict[str, LatLon]],
    tuple[list[TrajectoryWaypoint], list[Primitive], LegContext],
]


def _require_fix(fix_id: str, fixes: dict[str, LatLon]) -> LatLon:
    if fix_id not in fixes:
        raise KeyError(f"leg references unknown fix_id: {fix_id!r}")
    return fixes[fix_id]


# ---------------------------------------------------------------------------
# Terminator handlers
# ---------------------------------------------------------------------------
#
# Each handler returns a triple:
#   (waypoints, primitives, new_ctx)
#
# - ``waypoints``: zero or more TrajectoryWaypoints contributed by this
#   leg. The trajectory builder consumes these to emit the rendered
#   polyline.
# - ``primitives``: Segment / Arc / Hold primitives contributed by this
#   leg for the flat compile_legs output (used by bbox / mask /
#   projector code, NOT for rendering). No fillets here — waypoints are
#   vertex-joined.
# - ``new_ctx``: updated LegContext for the next leg.


def _handle_if(
    leg: dict, ctx: LegContext, fixes: dict[str, LatLon]
) -> tuple[list[TrajectoryWaypoint], list[Primitive], LegContext]:
    fix = _require_fix(leg["to_fix_id"], fixes)
    # IF anchors the start of a flight state. Emit the anchor waypoint
    # (flyby — it's an initial fix, not a hold). No primitive — the
    # first leg proper will draw from here.
    wp = TrajectoryWaypoint(position=fix, kind="flyby")
    return [wp], [], LegContext(position=fix, course_deg=None)


def _handle_tf(
    leg: dict, ctx: LegContext, fixes: dict[str, LatLon]
) -> tuple[list[TrajectoryWaypoint], list[Primitive], LegContext]:
    to_fix = _require_fix(leg["to_fix_id"], fixes)
    if ctx.position is None:
        # No prior position: anchor implicitly.
        wp = TrajectoryWaypoint(position=to_fix, kind="flyby")
        return [wp], [], LegContext(position=to_fix, course_deg=None)
    course = leg.get("course_deg")
    if course is None:
        course = float(initial_bearing_deg(ctx.position, to_fix))
    else:
        course = float(course)
    wp = TrajectoryWaypoint(position=to_fix, kind="flyby")
    prim = Segment(start=ctx.position, end=to_fix)
    return [wp], [prim], LegContext(position=to_fix, course_deg=course)


def _handle_cf(
    leg: dict, ctx: LegContext, fixes: dict[str, LatLon]
) -> tuple[list[TrajectoryWaypoint], list[Primitive], LegContext]:
    to_fix = _require_fix(leg["to_fix_id"], fixes)
    course = float(leg["course_deg"])  # required by schema for CF
    if ctx.position is None:
        wp = TrajectoryWaypoint(position=to_fix, kind="flyby")
        return [wp], [], LegContext(position=to_fix, course_deg=course)
    wp = TrajectoryWaypoint(position=to_fix, kind="flyby")
    prim = Segment(start=ctx.position, end=to_fix)
    return [wp], [prim], LegContext(position=to_fix, course_deg=course)


def _handle_df(
    leg: dict, ctx: LegContext, fixes: dict[str, LatLon]
) -> tuple[list[TrajectoryWaypoint], list[Primitive], LegContext]:
    """DF (direct-to-fix) — symbolic rendering: a single straight from
    the current position to the target fix, with a fly-by bend at the
    target.

    Chart-symbology rationale: a DF is published as "proceed direct to
    the fix". The chart reader interprets that as a gentle bend at the
    current position onto a track toward the fix. Real chart renderers
    don't draw a physics-accurate Dubins reversal loop — that would be
    visually noisy and would violate the convention that the rendered
    path communicates INTENT at a glance.
    """
    to_fix = _require_fix(leg["to_fix_id"], fixes)
    if ctx.position is None:
        wp = TrajectoryWaypoint(position=to_fix, kind="flyby")
        return [wp], [], LegContext(position=to_fix, course_deg=None)
    wp = TrajectoryWaypoint(position=to_fix, kind="flyby")
    prim = Segment(start=ctx.position, end=to_fix)
    outbound_course = initial_bearing_deg(ctx.position, to_fix)
    return [wp], [prim], LegContext(position=to_fix, course_deg=outbound_course)


def _handle_ca(
    leg: dict, ctx: LegContext, fixes: dict[str, LatLon]
) -> tuple[list[TrajectoryWaypoint], list[Primitive], LegContext]:
    course = float(leg["course_deg"])  # required by schema
    if ctx.position is None:
        return [], [], ctx
    end = destination(ctx.position, course, CA_LEG_RENDER_LENGTH_NM)
    # CA endpoint is a synthesized point along the commanded course.
    # Treat as a flyover — the next leg picks up from here and the
    # altitude-termination point is overflown (no bend setback).
    wp = TrajectoryWaypoint(position=end, kind="flyover")
    prim = Segment(start=ctx.position, end=end)
    return [wp], [prim], LegContext(position=end, course_deg=course)


def _handle_af(
    leg: dict, ctx: LegContext, fixes: dict[str, LatLon]
) -> tuple[list[TrajectoryWaypoint], list[Primitive], LegContext]:
    """AF (Arc-to-Fix) — emit the DME arc as a sampled trajectory leg.

    Emits two TrajectoryWaypoints:
      1. Arc ENTRY at the current position, carrying arc metadata
         (center, radius, clockwise). The trajectory builder will draw
         the arc FROM this point's outbound.
      2. Arc EXIT at the declared to_fix, flagged ``is_arc_exit`` so
         subsequent fly-by bend logic knows the inbound direction here
         is a tangent to the arc.

    Also emits one Arc primitive for compile_legs' flat output.
    """
    center = _require_fix(leg["arc_center_fix_id"], fixes)
    to_fix = _require_fix(leg["to_fix_id"], fixes)
    radius_nm = float(leg["arc_radius_nm"])
    turn_dir = leg["turn_direction"]  # required by schema for AF
    clockwise = turn_dir == "right"

    if ctx.position is None:
        wp = TrajectoryWaypoint(position=to_fix, kind="flyby", is_arc_exit=True)
        return [wp], [], LegContext(position=to_fix, course_deg=None)

    start_bearing = initial_bearing_deg(center, ctx.position)
    end_bearing = initial_bearing_deg(center, to_fix)
    # Outbound tangent course at the exit in the direction of flight.
    exit_course = (end_bearing + (90.0 if clockwise else -90.0)) % 360.0

    # Waypoints: arc entry (carries arc metadata) + arc exit.
    entry_wp = TrajectoryWaypoint(
        position=ctx.position,
        kind="flyby",
        arc_center=center,
        arc_radius_nm=radius_nm,
        arc_clockwise=clockwise,
    )
    exit_wp = TrajectoryWaypoint(
        position=to_fix,
        kind="flyby",
        is_arc_exit=True,
    )

    arc_prim = Arc(
        center=center,
        radius_nm=radius_nm,
        start_bearing_deg=start_bearing,
        end_bearing_deg=end_bearing,
        clockwise=clockwise,
    )
    return (
        [entry_wp, exit_wp],
        [arc_prim],
        LegContext(position=to_fix, course_deg=exit_course),
    )


def _handle_hm(
    leg: dict, ctx: LegContext, fixes: dict[str, LatLon]
) -> tuple[list[TrajectoryWaypoint], list[Primitive], LegContext]:
    to_fix = _require_fix(leg["to_fix_id"], fixes)
    hold = leg.get("hold") or {}
    inbound = float(hold.get("inbound_course_deg", 0.0))
    turn_dir = hold.get("turn_direction", "right")
    leg_length_nm = hold.get("leg_distance_nm")
    leg_time_sec = hold.get("leg_time_sec")
    if leg_length_nm is None and leg_time_sec is not None:
        leg_length_nm = HOLD_DEFAULT_TAS_KT * float(leg_time_sec) / 3600.0
    # Hold fix is overflown: the trajectory terminates at the hold fix
    # exactly, and the renderer draws a racetrack symbol separately.
    wp = TrajectoryWaypoint(position=to_fix, kind="flyover")
    hold_prim = Hold(
        fix=to_fix,
        inbound_course_deg=inbound,
        turn_direction=turn_dir,
        leg_length_nm=leg_length_nm,
        leg_time_sec=leg_time_sec,
    )
    return [wp], [hold_prim], LegContext(position=to_fix, course_deg=None)


def _stub(name: str) -> Handler:
    def _raise(
        leg: dict, ctx: LegContext, fixes: dict[str, LatLon]
    ) -> tuple[list[TrajectoryWaypoint], list[Primitive], LegContext]:
        raise NotImplementedError(f"{name} not yet implemented")

    return _raise


HANDLERS: dict[str, Handler] = {
    "IF": _handle_if,
    "TF": _handle_tf,
    "CF": _handle_cf,
    "DF": _handle_df,
    "CA": _handle_ca,
    "AF": _handle_af,
    "HM": _handle_hm,
    # Stubs — discoverable but explicitly not implemented yet.
    "VA": _stub("VA"),
    "FA": _stub("FA"),
    "CD": _stub("CD"),
    "VD": _stub("VD"),
    "CR": _stub("CR"),
    "VR": _stub("VR"),
    "CI": _stub("CI"),
    "VI": _stub("VI"),
    "RF": _stub("RF"),
    "HA": _stub("HA"),
    "HF": _stub("HF"),
    "PI": _stub("PI"),
}


def _dispatch(
    leg: dict, ctx: LegContext, fixes: dict[str, LatLon]
) -> tuple[list[TrajectoryWaypoint], list[Primitive], LegContext]:
    terminator = leg.get("terminator")
    handler = HANDLERS.get(terminator)
    if handler is None:
        raise ValueError(f"unknown leg terminator: {terminator!r}")
    return handler(leg, ctx, fixes)


def _dedupe_waypoints(
    waypoints: list[TrajectoryWaypoint],
    tol_nm: float = 1e-6,
) -> list[TrajectoryWaypoint]:
    """Drop consecutive waypoints at the same position (within ``tol_nm``).

    Common cases: an IF at the same lat/lon as the previous leg's
    to_fix_id (e.g. common_legs starting with IF to the transition's
    exit fix). A waypoint carrying arc metadata is KEPT — it sets up
    the outbound arc even when it coincides with the previous point.
    """
    from .geodesy import great_circle_distance_nm

    out: list[TrajectoryWaypoint] = []
    for wp in waypoints:
        if not out:
            out.append(wp)
            continue
        prev = out[-1]
        if great_circle_distance_nm(prev.position, wp.position) < tol_nm:
            # Prefer the wp that carries arc metadata or flags (more
            # informative). Otherwise drop the duplicate.
            if wp.arc_center is not None and prev.arc_center is None:
                out[-1] = wp
            elif wp.is_arc_exit and not prev.is_arc_exit:
                out[-1] = TrajectoryWaypoint(
                    position=prev.position,
                    kind=prev.kind,
                    arc_center=prev.arc_center,
                    arc_radius_nm=prev.arc_radius_nm,
                    arc_clockwise=prev.arc_clockwise,
                    is_arc_exit=True,
                )
            # else: drop wp.
            continue
        out.append(wp)
    return out


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def compile_legs(procedure: dict, fixes: dict[str, LatLon]) -> list[Primitive]:
    """Walk the procedure and emit a flat list of drawable primitives.

    Segments chain waypoint-to-waypoint with NO fillets at joints —
    vertex-joined. DME arcs (AF) are true Arc primitives. Holds are
    Hold primitives.

    This is a structural view used by bbox computation, procedure-mask
    rendering, and the projector. The renderer itself consumes
    :func:`compile_flight_states` for smooth path drawing.
    """
    out: list[Primitive] = []
    for transition in procedure.get("transitions") or []:
        ctx = LegContext()
        for leg in transition.get("legs") or []:
            _wps, prims, ctx = _dispatch(leg, ctx, fixes)
            out.extend(prims)

    ctx = LegContext()
    for leg in procedure.get("common_legs") or []:
        _wps, prims, ctx = _dispatch(leg, ctx, fixes)
        out.extend(prims)

    missed = procedure.get("missed_approach") or {}
    for leg in missed.get("legs") or []:
        _wps, prims, ctx = _dispatch(leg, ctx, fixes)
        out.extend(prims)

    return out


def _transition_label(transition: dict, common_terminal_fix_id: str | None) -> str:
    """Human-readable label for a flight state, e.g. ``"AMIKI\u2192RW14"``."""
    name = transition.get("name")
    if not name:
        for leg in transition.get("legs") or []:
            if leg.get("terminator") == "IF" and leg.get("to_fix_id"):
                name = leg["to_fix_id"]
                break
    name = name or "APPROACH"
    if common_terminal_fix_id:
        return f"{name}\u2192{common_terminal_fix_id}"
    return str(name)


def _common_terminal_fix_id(procedure: dict) -> str | None:
    for leg in reversed(procedure.get("common_legs") or []):
        to_id = leg.get("to_fix_id")
        if to_id:
            return to_id
    return None


def _missed_label(procedure: dict) -> str:
    missed = (procedure.get("missed_approach") or {}).get("legs") or []
    start_id = _common_terminal_fix_id(procedure) or "MAP"
    end_id: str | None = None
    for leg in reversed(missed):
        to_id = leg.get("to_fix_id")
        if to_id:
            end_id = to_id
            break
    if end_id:
        return f"Missed {start_id}\u2192{end_id}"
    return f"Missed {start_id}"


def _collect_waypoints_and_holds(
    legs: list[dict],
    ctx: LegContext,
    fixes: dict[str, LatLon],
) -> tuple[list[TrajectoryWaypoint], list[Hold], LegContext]:
    """Dispatch ``legs`` and collect the accumulated waypoints + any
    Hold primitives encountered, returning the end context."""
    waypoints: list[TrajectoryWaypoint] = []
    holds: list[Hold] = []
    for leg in legs:
        wps, prims, ctx = _dispatch(leg, ctx, fixes)
        waypoints.extend(wps)
        for p in prims:
            if isinstance(p, Hold):
                holds.append(p)
    return waypoints, holds, ctx


def compile_flight_states(
    procedure: dict, fixes: dict[str, LatLon]
) -> list[FlightState]:
    """Group the dispatcher's per-leg output into discrete flight states.

    A flight state is one continuous flyable trajectory from an entry
    fix to a termination fix. The trajectory builder smooths each state
    into a single polyline with constant cosmetic bends at fly-by
    waypoints.

    States:
      * Each approach transition → one state (transition legs +
        common_legs, walked with a fresh context at each transition).
      * Procedures without transitions → one approach state from
        common_legs alone.
      * Missed approach → one state.
    """
    states: list[FlightState] = []
    terminal_id = _common_terminal_fix_id(procedure)
    common_legs = procedure.get("common_legs") or []
    transitions_raw = procedure.get("transitions") or []

    # --- Approach states ---------------------------------------------------
    if transitions_raw:
        for transition in transitions_raw:
            ctx = LegContext()
            t_wps, _t_holds, ctx = _collect_waypoints_and_holds(
                transition.get("legs") or [], ctx, fixes
            )
            # Continue into common_legs with the transition's end ctx,
            # BUT common_legs typically starts with an IF that re-anchors
            # the position. We still need to thread ctx through so any
            # trailing CF's course flows correctly.
            c_wps, c_holds, ctx = _collect_waypoints_and_holds(
                common_legs, ctx, fixes
            )
            waypoints = _dedupe_waypoints(t_wps + c_wps)
            points = build_trajectory(waypoints)
            label = _transition_label(transition, terminal_id)
            states.append(
                FlightState(
                    phase="approach",
                    label=label,
                    primitives=[],
                    points=points,
                    holds=c_holds,
                )
            )
        # Thread ctx through for missed-approach continuation.
        ctx_after_common = ctx
    elif common_legs:
        ctx = LegContext()
        c_wps, c_holds, ctx = _collect_waypoints_and_holds(common_legs, ctx, fixes)
        waypoints = _dedupe_waypoints(c_wps)
        points = build_trajectory(waypoints)
        # Entry label from the first referenced fix.
        entry_id: str | None = None
        for leg in common_legs:
            for key in ("to_fix_id", "from_fix_id"):
                v = leg.get(key)
                if v:
                    entry_id = v
                    break
            if entry_id:
                break
        label = (
            f"{entry_id}\u2192{terminal_id}"
            if (entry_id and terminal_id)
            else "APPROACH"
        )
        states.append(
            FlightState(
                phase="approach",
                label=label,
                primitives=[],
                points=points,
                holds=c_holds,
            )
        )
        ctx_after_common = ctx
    else:
        ctx_after_common = LegContext()

    # --- Missed state ------------------------------------------------------
    missed_legs_raw = (procedure.get("missed_approach") or {}).get("legs") or []
    if missed_legs_raw:
        # The missed approach starts where common_legs ended (typically
        # the runway threshold). Build a fresh waypoint list — don't
        # include common_legs waypoints or we'd render the final approach
        # twice.
        m_wps, m_holds, _ = _collect_waypoints_and_holds(
            missed_legs_raw, ctx_after_common, fixes
        )
        # Prepend the starting position as an anchor so build_trajectory
        # sees an initial point to draw from. Otherwise the first CA/CF
        # waypoint becomes the first polyline point with no inbound.
        anchor_waypoints: list[TrajectoryWaypoint] = []
        if ctx_after_common.position is not None:
            anchor_waypoints.append(
                TrajectoryWaypoint(
                    position=ctx_after_common.position, kind="flyby"
                )
            )
        waypoints = _dedupe_waypoints(anchor_waypoints + m_wps)
        points = build_trajectory(waypoints)
        states.append(
            FlightState(
                phase="missed",
                label=_missed_label(procedure),
                primitives=[],
                points=points,
                holds=m_holds,
            )
        )

    return states


def build_fix_table(procedure: dict) -> dict[str, LatLon]:
    """Flatten the procedure's top-level `fixes` array into an id -> LatLon map."""
    return {
        f["id"]: (float(f["lat"]), float(f["lon"]))
        for f in (procedure.get("fixes") or [])
    }


__all__ = [
    "Arc",
    "CA_LEG_RENDER_LENGTH_NM",
    "DEFAULT_ARC_SAMPLES",
    "DEFAULT_VISUAL_BEND_NM",
    "FlightState",
    "Hold",
    "LegContext",
    "Primitive",
    "Segment",
    "TrajectoryWaypoint",
    "build_fix_table",
    "build_trajectory",
    "compile_flight_states",
    "compile_legs",
]
