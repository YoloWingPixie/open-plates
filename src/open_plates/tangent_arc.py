"""Tangent-arc constructor — the small rounded-bend primitive.

Given an inbound course, an outbound course, a waypoint, and a bend
radius, produce the tangent arc that meets both courses and the two
tangent points where the arc touches the straight segments.

Uses the planar approximation (local flat-Earth near the waypoint). At
the cosmetic bend radii used by the chart renderer (~0.15 NM) this is
accurate to well under a chart pixel.

Construction:

  d = r * tan(|Δ|/2)    # setback distance along each leg
  tangent_in  = fix back distance d along the inbound reciprocal
  tangent_out = fix forward distance d along the outbound
  center      = tangent_in offset perpendicular by r on the turn side

This is the single primitive used by :mod:`open_plates.trajectory`
to round fly-by waypoints with a constant cosmetic radius.

Plus one more primitive for DME-arc (ARINC 424 AF) → straight-leg
transitions: :func:`arc_to_line_fillet` computes the internal-tangency
fillet that blends a big arc into an outbound straight line at a
non-tangent waypoint.
"""

from __future__ import annotations

from math import cos, radians, sin, sqrt, tan

from .geodesy import LatLon, destination, great_circle_distance_nm, initial_bearing_deg
from .geometry import Arc

# Minimum turn angle that gets a visible rounded bend. Below this the
# two legs are effectively colinear; skip the fillet entirely.
EPSILON_TURN_DEG: float = 1.0


def _signed_turn_deg(inbound: float, outbound: float) -> float:
    """Signed turn angle in (-180, 180]. Positive = right, negative = left."""
    return (outbound - inbound + 540.0) % 360.0 - 180.0


def tangent_arc(
    inbound_course_deg: float,
    outbound_course_deg: float,
    turn_fix: LatLon,
    turn_radius_nm: float,
) -> tuple[LatLon, LatLon, Arc]:
    """Construct a tangent arc at ``turn_fix``.

    Returns ``(arc_start, arc_end, arc)``. ``arc_start`` lies on the
    inbound course before ``turn_fix``; ``arc_end`` lies on the outbound
    course after ``turn_fix``. The returned ``Arc`` is tangent to both
    courses at those points.

    Degenerate case: |Δ| < EPSILON_TURN_DEG → zero-radius arc at the fix.
    """
    delta = _signed_turn_deg(inbound_course_deg, outbound_course_deg)
    if abs(delta) < EPSILON_TURN_DEG:
        return (
            turn_fix,
            turn_fix,
            Arc(
                center=turn_fix,
                radius_nm=0.0,
                start_bearing_deg=inbound_course_deg % 360.0,
                end_bearing_deg=outbound_course_deg % 360.0,
                clockwise=delta >= 0,
            ),
        )

    clockwise = delta > 0
    half_abs = abs(delta) / 2.0
    anticipation_nm = turn_radius_nm * tan(radians(half_abs))

    inbound_reciprocal = (inbound_course_deg + 180.0) % 360.0
    arc_start = destination(turn_fix, inbound_reciprocal, anticipation_nm)
    arc_end = destination(turn_fix, outbound_course_deg, anticipation_nm)

    perp_from_track = (inbound_course_deg + (90.0 if clockwise else -90.0)) % 360.0
    center = destination(arc_start, perp_from_track, turn_radius_nm)

    start_bearing = (perp_from_track + 180.0) % 360.0
    end_perp = (outbound_course_deg + (90.0 if clockwise else -90.0)) % 360.0
    end_bearing = (end_perp + 180.0) % 360.0

    return (
        arc_start,
        arc_end,
        Arc(
            center=center,
            radius_nm=turn_radius_nm,
            start_bearing_deg=start_bearing,
            end_bearing_deg=end_bearing,
            clockwise=clockwise,
        ),
    )


def flyby_fillet_line_to_line(
    prev_start: LatLon,
    prev_end: LatLon,
    fix: LatLon,
    outbound_course_deg: float,
    turn_radius_nm: float,
) -> tuple[LatLon, Arc, LatLon] | None:
    """Fly-by fillet at a waypoint joining two straight tracks.

    Returns ``(tangent_in, Arc, tangent_out)`` for a fillet of radius
    ``turn_radius_nm`` tangent to both:
      - the inbound track (defined by ``prev_start`` -> ``fix``)
      - the outbound course through ``fix``

    Returns ``None`` when |turn angle| < :data:`EPSILON_TURN_DEG`.

    Used by :mod:`open_plates.trajectory` to round every fly-by
    waypoint with a constant cosmetic radius. Chart symbology, not
    flight physics — ``turn_radius_nm`` is a style constant (~0.15 NM),
    not a speed-derived value.

    Geometry (symmetric derivation):
      * setback d = r * tan(|θ|/2) along each track from the fix
        — standard aviation anticipation distance for a fly-by turn
      * center sits at the right-or-left perpendicular of the inbound
        track at tangent_in, offset by r on the turn side (concave).
        By symmetry it's also at distance r perpendicular to outbound
        at tangent_out.
      * start_bearing_deg = bearing from center to tangent_in
      * end_bearing_deg   = bearing from center to tangent_out
      * sweep direction = same as the turn (right=CW, left=CCW)
    """
    del prev_end  # derived purely from prev_start→fix bearing
    inbound_course_deg = initial_bearing_deg(prev_start, fix)
    delta = _signed_turn_deg(inbound_course_deg, outbound_course_deg)
    if abs(delta) < EPSILON_TURN_DEG:
        return None

    clockwise = delta > 0
    half_abs = abs(delta) / 2.0
    # Correct aviation anticipation distance: d = r * tan(θ/2). Grows
    # with turn angle; at θ = 90° it equals r, matching the simple
    # right-angle case everyone validates against. The alternative
    # formulation r/tan(θ/2) is incorrect and only happens to give the
    # same answer at 90° — it diverges badly for sharp turns.
    setback_nm = turn_radius_nm * tan(radians(half_abs))

    inbound_reciprocal = (inbound_course_deg + 180.0) % 360.0
    tangent_in = destination(fix, inbound_reciprocal, setback_nm)
    tangent_out = destination(fix, outbound_course_deg, setback_nm)

    # Center lies perpendicular to the inbound track at tangent_in,
    # offset by ``r`` on the turn side (right-perp for a right turn,
    # left-perp for a left turn).  By construction the center is
    # simultaneously perpendicular to the outbound track at
    # tangent_out — symmetry of the setback d = r/tan(|θ|/2).
    perp = (inbound_course_deg + (90.0 if clockwise else -90.0)) % 360.0
    center = destination(tangent_in, perp, turn_radius_nm)

    # Bearings from center to each tangent point (derived directly from
    # the geometry rather than from perpendicular math — both points
    # sit on the circle by construction).
    start_bearing = initial_bearing_deg(center, tangent_in)
    end_bearing = initial_bearing_deg(center, tangent_out)

    arc = Arc(
        center=center,
        radius_nm=turn_radius_nm,
        start_bearing_deg=start_bearing,
        end_bearing_deg=end_bearing,
        clockwise=clockwise,
    )
    return (tangent_in, arc, tangent_out)


def _project_to_plane(center: LatLon, point: LatLon) -> tuple[float, float]:
    """Project ``point`` into a local east-north NM frame centred on ``center``.

    Returns ``(east_nm, north_nm)``. This is the polar→Cartesian map used
    throughout the tangent-arc geometry: bearing 0 = +north, 90 = +east.
    Accurate to sub-pixel out to ~50 NM in mid-latitudes, which covers
    every realistic fillet radius vs DME-arc radius case.
    """
    d = great_circle_distance_nm(center, point)
    if d < 1e-12:
        return (0.0, 0.0)
    brg = initial_bearing_deg(center, point)
    return (d * sin(radians(brg)), d * cos(radians(brg)))


def _unproject_from_plane(center: LatLon, pt_en: tuple[float, float]) -> LatLon:
    """Inverse of :func:`_project_to_plane`. Uses ``destination`` so the
    round-trip error stays below the 1e-6 NM invariant tolerance."""
    e, n = pt_en
    d = sqrt(e * e + n * n)
    if d < 1e-12:
        return center
    # atan2(e, n) → bearing (0 = +north, 90 = +east), expressed in degrees.
    from math import atan2, degrees
    brg = (degrees(atan2(e, n)) + 360.0) % 360.0
    return destination(center, brg, d)


def arc_to_line_fillet(
    arc_center: LatLon,
    arc_radius_nm: float,
    arc_clockwise: bool,
    arc_exit_pt: LatLon,
    outbound_course_deg: float,
    fillet_radius_nm: float,
) -> tuple[LatLon, Arc, LatLon] | None:
    """Internal-tangency fillet between a big inbound arc and a straight
    outbound line at a waypoint where the two are NOT naturally tangent.

    Returns ``(arc_tangent_point, fillet_arc, line_tangent_point)``:
      - ``arc_tangent_point``: the point on the BIG arc where the fillet
        begins. The caller should trim the big arc to end here rather
        than at ``arc_exit_pt``.
      - ``fillet_arc``: a small :class:`Arc` of radius ``fillet_radius_nm``
        that is tangent to the big arc at ``arc_tangent_point`` and
        tangent to the outbound line at ``line_tangent_point``.
      - ``line_tangent_point``: the point on the outbound line where the
        straight leg picks up after the fillet.

    Returns ``None`` when:
      - ``fillet_radius_nm`` is ``>=`` ``arc_radius_nm`` (internal
        tangency impossible), or
      - the angular mismatch between the big arc's natural exit tangent
        and ``outbound_course_deg`` is below a small epsilon (the joint
        is effectively tangent already — no fillet needed), or
      - no real line-circle intersection exists (the outbound line never
        approaches within ``fillet_radius_nm`` of any internal-tangency
        locus around the big arc).

    Geometry (all done in a local east-north NM frame centred on
    ``arc_center``):

      Let ``O`` be the arc centre, ``R`` the big-arc radius, ``r`` the
      fillet radius. Internal tangency requires ``|C_f - O| = R - r``
      (the fillet's centre sits INSIDE the big circle by ``r``). The
      fillet is also tangent to the outbound line, so ``C_f`` sits at
      perpendicular distance exactly ``r`` from the line — i.e. on an
      offset line parallel to the outbound line by ``±r``.

      Intersect the offset line with the circle of radius ``R - r``
      around ``O``. That's a standard line–circle system; solve for
      the offset foot (``A*D, B*D`` with ``A^2+B^2 = 1``) and step
      along the line direction by ``sqrt((R-r)^2 - D^2)``. Up to four
      candidate centres emerge (two sign choices for the perpendicular
      offset × two roots of the quadratic). Of those:

        (1) ``C_f`` must be on the same side of the outbound line as
            ``arc_exit_pt`` lies relative to the "turn side" (the
            fillet has to curve into the turn, not away from it). In
            practice the correct candidate is the one whose projected
            distance to ``arc_exit_pt`` is minimal — the fillet sits
            right at the joint, not on the opposite side of the big
            circle.

        (2) The tangent direction of the fillet at the tangent points
            must match the flight direction (big arc → outbound), which
            is a direct consequence of choosing the nearest candidate;
            the diametrically-opposite candidate gives a tangent
            pointing the wrong way.

      We emit an assertion at the end verifying both tangent points lie
      on their respective geometry (big arc and outbound line) within
      1e-6 NM — catches sign / quadrant mistakes before they propagate.
    """
    if fillet_radius_nm >= arc_radius_nm:
        return None
    if fillet_radius_nm <= 0.0:
        return None

    # Big-arc's natural tangent direction at the declared exit point, in
    # the direction of flight.
    exit_radial = initial_bearing_deg(arc_center, arc_exit_pt)
    if arc_clockwise:
        arc_tangent_deg = (exit_radial + 90.0) % 360.0
    else:
        arc_tangent_deg = (exit_radial - 90.0) % 360.0

    # Angular mismatch between the arc's natural exit tangent and the
    # declared outbound course. If it's tiny, no fillet is needed — the
    # joint is already effectively tangent.
    delta = _signed_turn_deg(arc_tangent_deg, outbound_course_deg)
    if abs(delta) < EPSILON_TURN_DEG:
        return None

    R = float(arc_radius_nm)
    r = float(fillet_radius_nm)

    # Project arc_exit_pt into the local east-north frame centred on
    # arc_center. The frame is planar-NM; the arc's circle of radius R
    # has centre at the origin of this frame.
    pe, pn = _project_to_plane(arc_center, arc_exit_pt)

    # Outbound line direction (unit vector) in the (east, north) frame.
    # Bearing 0 = +north, 90 = +east, so u = (sin θ, cos θ).
    theta = radians(outbound_course_deg)
    u_e, u_n = (sin(theta), cos(theta))
    # Right-perpendicular to u (rotating u by -90°).
    n_e, n_n = (cos(theta), -sin(theta))

    # Signed perpendicular distance from the origin (= arc_center) to
    # the outbound line. The line passes through (pe, pn), so any point
    # P on the line satisfies (P - (pe,pn)) · n_hat = 0, i.e.
    # P · n_hat = k where k = pe·n_e + pn·n_n.
    k = pe * n_e + pn * n_n

    rho = R - r  # internal tangency radius for fillet centre

    # Try both perpendicular-offset signs (fillet centre on either side
    # of the outbound line). For each, solve the line-circle system
    # (offset-line ∩ circle of radius rho around origin).
    #
    # Offset line: P · n_hat = k + s·r   where s ∈ {+1, -1}.
    # Foot from origin: F = (k + s·r) · n_hat.
    # Points on offset line: F + t · u_hat.
    # Solve |F + t·u|² = rho² → t² = rho² - (k + s·r)²  (since |F| = |k+s·r|
    # and F ⊥ u).
    candidates: list[tuple[float, tuple[float, float]]] = []
    for s in (+1.0, -1.0):
        D = k + s * r
        discriminant = rho * rho - D * D
        if discriminant < 0.0:
            continue
        t_half = sqrt(discriminant)
        f_e = D * n_e
        f_n = D * n_n
        for t in (+t_half, -t_half):
            cf_e = f_e + t * u_e
            cf_n = f_n + t * u_n
            # Distance from fillet-centre candidate to the projected
            # arc_exit_pt — we'll pick the minimum.
            dx = cf_e - pe
            dy = cf_n - pn
            dist_to_exit_sq = dx * dx + dy * dy
            candidates.append((dist_to_exit_sq, (cf_e, cf_n)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])

    # Walk the candidates in closest-first order; pick the first one
    # whose tangent directions match the flight direction at both
    # tangent points. (Usually the very first is correct; we fall back
    # through the list if a weirdly-placed waypoint and an acute turn
    # conspire to put the geometrically-closest candidate on the wrong
    # tangent side.)
    for _, (cf_e, cf_n) in candidates:
        # T_arc: on the BIG arc, collinear with O → C_f. Because C_f is
        # inside the circle (distance R - r), T_arc = C_f * (R / (R-r)).
        d_cf = sqrt(cf_e * cf_e + cf_n * cf_n)
        if d_cf < 1e-9:
            continue  # fillet centre at origin — degenerate
        t_arc_e = cf_e * (R / d_cf)
        t_arc_n = cf_n * (R / d_cf)

        # T_line: foot of perpendicular from C_f to the outbound line.
        # The outbound line passes through (pe, pn) with direction u.
        # Project (C_f - P) onto u to get the foot.
        dcf_pe = cf_e - pe
        dcf_pn = cf_n - pn
        proj = dcf_pe * u_e + dcf_pn * u_n
        t_line_e = pe + proj * u_e
        t_line_n = pn + proj * u_n

        # Check: fillet's tangent direction at T_line must point ALONG
        # the outbound direction, not reciprocal. The fillet's tangent
        # at T_line is perpendicular to (T_line - C_f). Two candidates:
        # turn the perpendicular one way or the other; the "flight
        # direction" one is the one whose sign matches u.
        # tangent_at_line = rotate(T_line - C_f) by ±90°.
        tl_minus_cf_e = t_line_e - cf_e
        tl_minus_cf_n = t_line_n - cf_n
        # Rotate +90° (CCW in east-north frame): (e, n) → (-n, e).
        tan_ccw_e = -tl_minus_cf_n
        tan_ccw_n = tl_minus_cf_e
        # Dot with u. If positive, CCW rotation gives outbound direction
        # → fillet is traversed CCW; else CW.
        fillet_ccw_consistent = (tan_ccw_e * u_e + tan_ccw_n * u_n) > 0.0

        # Similar check at T_arc: fillet's tangent direction there must
        # match the big-arc's tangent direction (flight direction along
        # the big arc). big-arc tangent at T_arc is perpendicular to
        # (T_arc - O) in the clockwise-or-CCW direction of the arc.
        # Rotate (T_arc - O) by +90° → CCW tangent.
        big_tan_ccw_e = -t_arc_n
        big_tan_ccw_n = t_arc_e
        big_tangent_ccw = not arc_clockwise
        if big_tangent_ccw:
            big_tan_e = big_tan_ccw_e
            big_tan_n = big_tan_ccw_n
        else:
            big_tan_e = -big_tan_ccw_e
            big_tan_n = -big_tan_ccw_n

        # Fillet tangent at T_arc (in the flight direction):
        ta_minus_cf_e = t_arc_e - cf_e
        ta_minus_cf_n = t_arc_n - cf_n
        # Rotate +90° CCW:
        fillet_tan_at_arc_ccw_e = -ta_minus_cf_n
        fillet_tan_at_arc_ccw_n = ta_minus_cf_e
        if fillet_ccw_consistent:
            fillet_tan_at_arc_e = fillet_tan_at_arc_ccw_e
            fillet_tan_at_arc_n = fillet_tan_at_arc_ccw_n
        else:
            fillet_tan_at_arc_e = -fillet_tan_at_arc_ccw_e
            fillet_tan_at_arc_n = -fillet_tan_at_arc_ccw_n

        # Dot product to check matching direction.
        match = (
            fillet_tan_at_arc_e * big_tan_e
            + fillet_tan_at_arc_n * big_tan_n
        )
        if match <= 0.0:
            continue  # tangent direction inverted — wrong candidate.

        # Found it. Build the emitted primitives.
        t_arc_ll = _unproject_from_plane(arc_center, (t_arc_e, t_arc_n))
        t_line_ll = _unproject_from_plane(arc_center, (t_line_e, t_line_n))
        fillet_center_ll = _unproject_from_plane(arc_center, (cf_e, cf_n))

        start_bearing = initial_bearing_deg(fillet_center_ll, t_arc_ll)
        end_bearing = initial_bearing_deg(fillet_center_ll, t_line_ll)

        fillet_arc = Arc(
            center=fillet_center_ll,
            radius_nm=r,
            start_bearing_deg=start_bearing,
            end_bearing_deg=end_bearing,
            clockwise=(not fillet_ccw_consistent),
        )

        # Invariants: tangent points must lie exactly on the big arc and
        # the outbound line respectively. Tolerance 1e-6 NM ≈ 0.006 ft —
        # well below any meaningful chart resolution; any failure here
        # indicates a sign / quadrant bug upstream.
        d_arc = great_circle_distance_nm(arc_center, t_arc_ll)
        assert abs(d_arc - R) < 1e-6, (
            f"arc_to_line_fillet: T_arc off big arc by {d_arc - R:.2e} NM"
        )
        # For T_line: perpendicular distance to outbound line in the
        # plane. Because we derived T_line as the foot, this is
        # structurally zero; re-project and re-compute for defensive
        # check.
        tl_e, tl_n = _project_to_plane(arc_center, t_line_ll)
        off_line = (tl_e - pe) * n_e + (tl_n - pn) * n_n
        assert abs(off_line) < 1e-6, (
            f"arc_to_line_fillet: T_line off outbound line by "
            f"{off_line:.2e} NM"
        )

        return (t_arc_ll, fillet_arc, t_line_ll)

    return None


def line_to_arc_fillet(
    line_entry_pt: LatLon,
    inbound_course_deg: float,
    arc_center: LatLon,
    arc_radius_nm: float,
    arc_clockwise: bool,
    arc_entry_pt: LatLon,
    fillet_radius_nm: float,
) -> tuple[LatLon, Arc, LatLon] | None:
    """Internal-tangency fillet between a straight inbound line and a big
    outbound arc at a waypoint where the two are NOT naturally tangent.

    Mirror of :func:`arc_to_line_fillet` with inbound / outbound roles
    swapped. Returns ``(line_tangent_point, fillet_arc, arc_tangent_point)``:
      - ``line_tangent_point``: the point on the inbound line where the
        fillet BEGINS. The inbound straight should be trimmed to end here
        rather than at ``arc_entry_pt``.
      - ``fillet_arc``: a small :class:`Arc` of radius ``fillet_radius_nm``
        tangent to the inbound line at ``line_tangent_point`` and tangent
        to the big arc at ``arc_tangent_point``.
      - ``arc_tangent_point``: the point on the BIG arc where the fillet
        ENDS and the arc picks up. The outbound arc should be sampled
        starting here, NOT from ``arc_entry_pt``.

    Returns ``None`` under the same conditions as :func:`arc_to_line_fillet`:
      - ``fillet_radius_nm >= arc_radius_nm`` (internal tangency impossible),
      - angular mismatch between the inbound course and the arc's natural
        entry tangent is below :data:`EPSILON_TURN_DEG` (already tangent),
      - no real line-circle intersection exists.

    Geometry — identical quadratic to :func:`arc_to_line_fillet`. In a
    local east-north NM frame on ``arc_center``:

      ``|C_f - O| = R - r`` (internal tangency with the big arc); ``C_f``
      sits at perpendicular distance ``r`` from the inbound line. Intersect
      the offset line with the circle of radius ``R - r`` around the
      origin; up to four candidate centres (two offset signs × two roots).
      Rank by planar distance from the declared waypoint ``arc_entry_pt``
      and pick the first candidate whose fillet tangent matches the flight
      direction at BOTH tangent points (line and big-arc).
    """
    if fillet_radius_nm >= arc_radius_nm:
        return None
    if fillet_radius_nm <= 0.0:
        return None

    # Big-arc's natural tangent direction at the declared entry point, in
    # the direction of flight (i.e. the direction the aircraft WOULD be
    # flying if it were already established on the arc at arc_entry_pt).
    entry_radial = initial_bearing_deg(arc_center, arc_entry_pt)
    if arc_clockwise:
        arc_tangent_deg = (entry_radial + 90.0) % 360.0
    else:
        arc_tangent_deg = (entry_radial - 90.0) % 360.0

    # Angular mismatch between inbound course and the arc's natural entry
    # tangent. Below epsilon → already tangent, no fillet needed.
    delta = _signed_turn_deg(inbound_course_deg, arc_tangent_deg)
    if abs(delta) < EPSILON_TURN_DEG:
        return None

    R = float(arc_radius_nm)
    r = float(fillet_radius_nm)

    # Project arc_entry_pt into the local east-north frame centred on
    # arc_center. Arc's circle of radius R has its centre at the origin.
    pe, pn = _project_to_plane(arc_center, arc_entry_pt)

    # Inbound line direction (unit vector) in the (east, north) frame.
    # Bearing 0 = +north, 90 = +east, so u = (sin θ, cos θ).
    theta = radians(inbound_course_deg)
    u_e, u_n = (sin(theta), cos(theta))
    # Right-perpendicular to u (rotating u by -90°).
    n_e, n_n = (cos(theta), -sin(theta))

    # Signed perpendicular distance from origin (= arc_center) to the
    # inbound line. Line passes through (pe, pn), so P · n_hat = k with
    # k = pe·n_e + pn·n_n.
    k = pe * n_e + pn * n_n

    rho = R - r  # internal-tangency radius for the fillet centre

    # Offset-line ∩ circle-of-radius-rho system. Same quadratic as the
    # arc→line case — the line role swaps but the algebra is identical.
    candidates: list[tuple[float, tuple[float, float]]] = []
    for s in (+1.0, -1.0):
        D = k + s * r
        discriminant = rho * rho - D * D
        if discriminant < 0.0:
            continue
        t_half = sqrt(discriminant)
        f_e = D * n_e
        f_n = D * n_n
        for t in (+t_half, -t_half):
            cf_e = f_e + t * u_e
            cf_n = f_n + t * u_n
            dx = cf_e - pe
            dy = cf_n - pn
            dist_to_entry_sq = dx * dx + dy * dy
            candidates.append((dist_to_entry_sq, (cf_e, cf_n)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])

    # Walk candidates closest-first; accept the first whose tangent
    # directions match the flight direction at both tangent points.
    for _, (cf_e, cf_n) in candidates:
        # T_arc: on the big arc, collinear with O → C_f. Because |C_f| =
        # R - r (internal tangency), T_arc = C_f * (R / (R - r)).
        d_cf = sqrt(cf_e * cf_e + cf_n * cf_n)
        if d_cf < 1e-9:
            continue  # fillet centre at origin — degenerate
        t_arc_e = cf_e * (R / d_cf)
        t_arc_n = cf_n * (R / d_cf)

        # T_line: foot of perpendicular from C_f to the inbound line.
        dcf_pe = cf_e - pe
        dcf_pn = cf_n - pn
        proj = dcf_pe * u_e + dcf_pn * u_n
        t_line_e = pe + proj * u_e
        t_line_n = pn + proj * u_n

        # Fillet tangent direction at T_line must point ALONG the inbound
        # direction (fillet ENTERS from the straight leg; tangent at the
        # junction equals inbound course). Rotate (T_line - C_f) by ±90°.
        tl_minus_cf_e = t_line_e - cf_e
        tl_minus_cf_n = t_line_n - cf_n
        # CCW rotation (+90°) in east-north frame: (e, n) → (-n, e).
        tan_ccw_e = -tl_minus_cf_n
        tan_ccw_n = tl_minus_cf_e
        # Dot with u_in (inbound direction, which is the flight direction
        # into the fillet at T_line). Positive → CCW traversal of fillet
        # gives flight direction; negative → CW.
        fillet_ccw_consistent = (tan_ccw_e * u_e + tan_ccw_n * u_n) > 0.0

        # Big-arc tangent at T_arc in the flight direction:
        # (T_arc - O) rotated +90° gives the CCW tangent; rotate the other
        # way for CW arcs.
        big_tan_ccw_e = -t_arc_n
        big_tan_ccw_n = t_arc_e
        big_tangent_ccw = not arc_clockwise
        if big_tangent_ccw:
            big_tan_e = big_tan_ccw_e
            big_tan_n = big_tan_ccw_n
        else:
            big_tan_e = -big_tan_ccw_e
            big_tan_n = -big_tan_ccw_n

        # Fillet tangent at T_arc (in the flight direction):
        ta_minus_cf_e = t_arc_e - cf_e
        ta_minus_cf_n = t_arc_n - cf_n
        fillet_tan_at_arc_ccw_e = -ta_minus_cf_n
        fillet_tan_at_arc_ccw_n = ta_minus_cf_e
        if fillet_ccw_consistent:
            fillet_tan_at_arc_e = fillet_tan_at_arc_ccw_e
            fillet_tan_at_arc_n = fillet_tan_at_arc_ccw_n
        else:
            fillet_tan_at_arc_e = -fillet_tan_at_arc_ccw_e
            fillet_tan_at_arc_n = -fillet_tan_at_arc_ccw_n

        # Matching direction check at T_arc.
        match = (
            fillet_tan_at_arc_e * big_tan_e
            + fillet_tan_at_arc_n * big_tan_n
        )
        if match <= 0.0:
            continue

        # Found it. Build the emitted primitives.
        t_line_ll = _unproject_from_plane(arc_center, (t_line_e, t_line_n))
        t_arc_ll = _unproject_from_plane(arc_center, (t_arc_e, t_arc_n))
        fillet_center_ll = _unproject_from_plane(arc_center, (cf_e, cf_n))

        # Fillet goes FROM T_line TO T_arc (flight direction):
        start_bearing = initial_bearing_deg(fillet_center_ll, t_line_ll)
        end_bearing = initial_bearing_deg(fillet_center_ll, t_arc_ll)

        fillet_arc = Arc(
            center=fillet_center_ll,
            radius_nm=r,
            start_bearing_deg=start_bearing,
            end_bearing_deg=end_bearing,
            clockwise=(not fillet_ccw_consistent),
        )

        # Invariants: T_arc on the big arc within 1e-6 NM; T_line on the
        # inbound line within 1e-6 NM. Any failure means a sign/quadrant
        # bug — raise rather than emit wrong geometry.
        d_arc = great_circle_distance_nm(arc_center, t_arc_ll)
        assert abs(d_arc - R) < 1e-6, (
            f"line_to_arc_fillet: T_arc off big arc by {d_arc - R:.2e} NM"
        )
        tl_e, tl_n = _project_to_plane(arc_center, t_line_ll)
        off_line = (tl_e - pe) * n_e + (tl_n - pn) * n_n
        assert abs(off_line) < 1e-6, (
            f"line_to_arc_fillet: T_line off inbound line by "
            f"{off_line:.2e} NM"
        )

        return (t_line_ll, fillet_arc, t_arc_ll)

    return None


__all__ = [
    "EPSILON_TURN_DEG",
    "arc_to_line_fillet",
    "flyby_fillet_line_to_line",
    "line_to_arc_fillet",
    "tangent_arc",
]
