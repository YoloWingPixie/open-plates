"""SVG renderer for IAP charts — Swiss / Memphis / NATO hybrid style.

Hand-emits SVG strings — no DOM library, no external dependencies beyond
stdlib + already-present geometry. Produces an 8.5 x 11 in portrait page
at 72 dpi (612 x 792 px) laid out per the vertical regions:

    TITLE_H=30  BRIEFING_H=105  PLAN_H=310  PROFILE_H=125  MINIMA_H=125  COMMS_H=55

Swiss typographic grid, reversed-chrome section-bar headers, NATO corner
brackets, Memphis accent (magenta procedure path, cyan info overlays).

Output is deterministic — no timestamps, no random IDs.
"""

from __future__ import annotations

from math import atan2, cos, degrees, pi, radians, sin, tan
from typing import Iterable

from .basemap import (
    BasemapConfig,
    collect_basemap_label_requests,
    find_covering_region,
    load_basemap,
    render_basemap,
)
from .geodesy import LatLon, destination, great_circle_distance_nm
from .geometry import Arc, FlightState, Hold, Primitive, Segment
from .legs import (
    LegContext, _dispatch, build_fix_table, compile_flight_states, compile_legs,
)  # noqa: F401 — LegContext / _dispatch kept for back-compat helpers below
from .placement import (
    BBox as PBBox,
    LabelCandidate,
    Placed,
    PlacementEngine,
    PlacementRequest,
    PlacementTier,
    eight_position_candidates,
    four_position_candidates,
)
from .projection import (
    Projector, plan_projector, plan_world_bbox, projector_scale_px_per_nm,
)

# ---------------------------------------------------------------------------
# Page layout constants (US Letter portrait at 72 dpi)
# ---------------------------------------------------------------------------

PAGE_W = 612.0
PAGE_H = 792.0

MARGIN = 18.0  # 0.25 in

# Vertical partitioning (top-down). Do NOT change these heights.
TITLE_H = 30.0
BRIEFING_H = 105.0
PLAN_H = 310.0
PROFILE_H = 125.0
MINIMA_H = 125.0
COMMS_H = 55.0

# Section-bar header height (reversed-chrome). Same value used everywhere.
BAR_H = 14.0

# ---------------------------------------------------------------------------
# Style tokens
# ---------------------------------------------------------------------------

# Core ink / paper.
COLOR_INK = "#0A0A0A"          # off-black for all primary rules & text
COLOR_PAPER = "#FFFFFF"        # page background (pure white; simpler print)
COLOR_RULE = "#0A0A0A"         # heavy rules = ink
COLOR_HAIRLINE = "#B8B3A6"     # 0.5 px sub-cell rules inside grids
COLOR_MUTED = "#6E6A61"        # micro-caption tone (on paper)

# Memphis accent policy: ONE magenta (procedure), ONE cyan (reference).
COLOR_ACCENT_MAGENTA = "#E8175D"   # approach path, load-bearing accent
COLOR_ACCENT_CYAN = "#1FB5D6"      # MSA bezel, north, scale, DA reference

# Procedure strokes.
COLOR_PROCEDURE = COLOR_ACCENT_MAGENTA
COLOR_MISSED = COLOR_INK            # missed is schematic → black dashed
COLOR_RUNWAY = "#555555"            # dark grey pavement

# Back-compat aliases for any stray references (none in this file, but
# safe to keep — a caller could be inspecting them).
COLOR_BG = COLOR_PAPER
COLOR_LINE = COLOR_INK
COLOR_TEXT = COLOR_INK
COLOR_MSA = COLOR_ACCENT_CYAN

# Basemap underlay — strict tints/shades of the palette above. Re-exported
# from ``basemap`` so any downstream code inspecting render_svg can see
# them. The procedure magenta must remain visually dominant; these all
# sit well back of COLOR_ACCENT_MAGENTA in saturation/contrast.
COLOR_BASEMAP_SEA = "#F0F6F7"           # ~6% cyan tint on paper
COLOR_BASEMAP_LAND = COLOR_PAPER        # no fill — cartographic default
COLOR_BASEMAP_COAST = COLOR_HAIRLINE    # 0.5 px
COLOR_BASEMAP_RIVER = "#D4E3E7"         # 0.4 px, extra-pale cyan
COLOR_BASEMAP_CITY = COLOR_INK          # 2.5 px square
COLOR_BASEMAP_LABEL = COLOR_MUTED       # SIZE_MICRO tracked UPPERCASE

# Typography — single sans family, mono only for codes.
FONT_SANS = '"Helvetica Neue", Helvetica, Arial, sans-serif'
FONT_MONO = '"IBM Plex Mono", "Courier New", monospace'

# Strict 6-step size scale (px).
SIZE_MICRO = 6.5      # cell micro-captions (UPPERCASE tracked)
SIZE_BODY = 8.0       # table rows, body text, missed-approach prose
SIZE_STD = 9.5        # fix labels, standard data values
SIZE_SUBTITLE = 12.0  # procedure id in title bar
SIZE_TITLE = 16.0     # page title (airport + procedure)
SIZE_HERO = 22.0      # DA/MDA hero number

# Tracking applied to UPPERCASE captions / section bars.
TRACK = "0.08em"

Region = tuple[float, float, float, float]  # (x, y, w, h)


# ---------------------------------------------------------------------------
# Low-level SVG emission
# ---------------------------------------------------------------------------


def _esc(s: object) -> str:
    """Minimal XML text escape."""
    t = "" if s is None else str(s)
    return (
        t.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt(n: float, places: int = 2) -> str:
    """Trim trailing zeros from a float for compact SVG output."""
    s = f"{n:.{places}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _svg_open(width: float, height: float) -> str:
    # font-family attribute quoting: inner double-quoted family names must be
    # XML-escaped so the outer attribute quoting stays well-formed.
    sans_attr = FONT_SANS.replace('"', "&quot;")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{_fmt(width)}" height="{_fmt(height)}" '
        f'viewBox="0 0 {_fmt(width)} {_fmt(height)}" '
        f'font-family="{sans_attr}" fill="{COLOR_INK}">\n'
        f'<rect x="0" y="0" width="{_fmt(width)}" height="{_fmt(height)}" '
        f'fill="{COLOR_PAPER}"/>\n'
    )


def _svg_close() -> str:
    return "</svg>\n"


def _rect(
    region: Region,
    *,
    stroke: str = "none",
    fill: str = "none",
    sw: float = 1.0,
) -> str:
    x, y, w, h = region
    return (
        f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{_fmt(sw)}"/>\n'
    )


def _line(
    x1: float, y1: float, x2: float, y2: float,
    *, stroke: str = COLOR_INK, sw: float = 1.0, dash: str | None = None,
) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}" '
        f'stroke="{stroke}" stroke-width="{_fmt(sw)}"{d}/>\n'
    )


def _text(
    x: float,
    y: float,
    s: str,
    *,
    size: float = SIZE_STD,
    weight: str = "normal",
    anchor: str = "start",
    italic: bool = False,
    family: str | None = None,
    fill: str = COLOR_INK,
    tracking: str | None = None,
) -> str:
    style_parts = [f"font-size:{_fmt(size)}px"]
    if weight != "normal":
        style_parts.append(f"font-weight:{weight}")
    if italic:
        style_parts.append("font-style:italic")
    if family:
        # Inline style attribute is double-quoted; switch any literal double
        # quotes around font names to single quotes so the attribute stays
        # well-formed XML.
        style_parts.append(f"font-family:{family.replace(chr(34), chr(39))}")
    if tracking:
        style_parts.append(f"letter-spacing:{tracking}")
    style = ";".join(style_parts)
    return (
        f'<text x="{_fmt(x)}" y="{_fmt(y)}" text-anchor="{anchor}" '
        f'fill="{fill}" style="{style}">{_esc(s)}</text>\n'
    )


def _caption(x: float, y: float, s: str, *, fill: str = COLOR_MUTED) -> str:
    """Micro-caption: UPPERCASE, tracked, SIZE_MICRO."""
    return _text(
        x, y, s.upper(),
        size=SIZE_MICRO, weight="bold", fill=fill, tracking=TRACK,
    )


def _section_bar(region: Region, label: str) -> str:
    """Reversed-chrome section header — ink fill, paper text, tracked caps."""
    x, y, w, h = region
    bar = (x, y, w, BAR_H)
    out = _rect(bar, fill=COLOR_INK)
    out += _text(
        x + 6, y + BAR_H - 4, label.upper(),
        size=SIZE_BODY, weight="bold",
        fill=COLOR_PAPER, tracking=TRACK,
    )
    return out


def _corner_brackets(page_w: float, page_h: float) -> str:
    """Four NATO-style L-shaped corner brackets inset from the page edge,
    plus a hairline full-page border inside the brackets for containment.

    The border reads as a single Swiss-grid frame around the whole plate;
    the brackets remain as the NATO reference detail, one step heavier so
    they still punctuate each corner.
    """
    inset = 6.0
    leg = 18.0
    sw = 1.5
    out = ""
    # Hairline containment border (drawn first so brackets land on top).
    out += _rect(
        (inset, inset, page_w - 2 * inset, page_h - 2 * inset),
        stroke=COLOR_HAIRLINE, sw=0.5,
    )
    corners = [
        (inset, inset, +1, +1),                     # top-left
        (page_w - inset, inset, -1, +1),            # top-right
        (inset, page_h - inset, +1, -1),            # bottom-left
        (page_w - inset, page_h - inset, -1, -1),   # bottom-right
    ]
    for cx, cy, sx, sy in corners:
        # Horizontal leg
        out += _line(cx, cy, cx + sx * leg, cy, sw=sw)
        # Vertical leg
        out += _line(cx, cy, cx, cy + sy * leg, sw=sw)
    return out


def _layout_regions() -> dict[str, Region]:
    """Compute the stacked vertical regions for the page."""
    x = MARGIN
    w = PAGE_W - 2 * MARGIN
    y = MARGIN
    regions: dict[str, Region] = {}
    for name, h in (
        ("title", TITLE_H),
        ("briefing", BRIEFING_H),
        ("plan", PLAN_H),
        ("profile", PROFILE_H),
        ("minima", MINIMA_H),
        ("comms", COMMS_H),
    ):
        regions[name] = (x, y, w, h)
        y += h
    return regions


# ---------------------------------------------------------------------------
# Title strip + plate code
# ---------------------------------------------------------------------------


def _render_title_strip(procedure: dict, region: Region) -> str:
    x, y, w, h = region
    airport = str(procedure.get("airport_icao", "")).upper()
    proc_id = str(procedure.get("procedure_id", "")).upper()
    airac = str(procedure.get("airac_cycle", ""))
    effective = str(procedure.get("effective_date", ""))
    amendment = str(procedure.get("amendment", ""))

    # Heavy baseline rule under the title — Swiss horizontal bar.
    out = _line(x, y + h - 1, x + w, y + h - 1, stroke=COLOR_INK, sw=2.0)

    # Title: airport in large bold, procedure id in subtitle weight. Use
    # leading margin separation (8 px gap via approximate width).
    baseline_y = y + h - 10
    out += _text(x, baseline_y, airport,
                 size=SIZE_TITLE, weight="bold")
    # Approximate airport width (Helvetica bold ≈ 0.72 em per char) + gap.
    # Erring wider so the procedure id doesn't crowd the airport ICAO.
    airport_w = SIZE_TITLE * 0.72 * max(len(airport), 1) + 14
    out += _text(x + airport_w, baseline_y, proc_id,
                 size=SIZE_SUBTITLE, weight="normal")

    # Right-side meta strip — AIRAC / EFF / AMDT. Reserve 108 px for the
    # plate code that will be stamped below the title; render meta left of it.
    plate_w = 108.0
    meta_parts = []
    if airac:
        meta_parts.append(f"AIRAC {airac}")
    if effective:
        meta_parts.append(f"EFF {effective}")
    if amendment:
        meta_parts.append(f"AMDT {amendment}")
    if meta_parts:
        out += _text(
            x + w, baseline_y, "   ".join(meta_parts),
            size=SIZE_BODY, anchor="end", tracking=TRACK,
        )
    return out


def _plate_code(procedure: dict, briefing_region: Region) -> str:
    """NATO-style stamped plate code — overlays the right end of the
    briefing bar (which is already reversed-chrome), so the code reads as
    a stamp on a classified-document header strip.
    """
    bx0, by0, bw, _bh = briefing_region
    airport = str(procedure.get("airport_icao", "XXXX")).upper()
    ptype = (procedure.get("procedure_type") or "").lower()
    ptype_code = "IAP" if ptype == "iap" else (ptype.upper() or "PROC")
    runway = str(procedure.get("runway_ident", "")).strip()
    tail = f"-{runway}" if runway else ""
    code = f"{airport}-{ptype_code}{tail}  SHT 1/1"

    # Draw the code right-aligned inside the (already black) briefing bar.
    # The section bar is BAR_H tall; render text slightly inside its right edge.
    text_x = bx0 + bw - 6
    text_y = by0 + BAR_H - 4
    return _text(
        text_x, text_y, code,
        size=SIZE_MICRO, fill=COLOR_PAPER, anchor="end",
        family=FONT_MONO, tracking="0.04em",
    )


# ---------------------------------------------------------------------------
# Briefing grid
# ---------------------------------------------------------------------------


# Column widths for the briefing grid. Sum MUST equal PAGE_W - 2*MARGIN = 576.
# Order: APCH | LOC FRQ | IDENT | FAC | GS ALT | DA MIN
BRIEFING_COLS = (64.0, 80.0, 72.0, 72.0, 80.0, 208.0)


def _render_briefing_strip(procedure: dict, region: Region) -> str:
    x, y, w, h = region
    assert abs(sum(BRIEFING_COLS) - w) < 0.5, "briefing columns must sum to region width"

    # Section bar
    out = _section_bar(region, "BRIEFING")

    # Working area below the bar.
    body_y = y + BAR_H
    body_h = h - BAR_H
    # Two grid rows + a missed-approach band at the bottom.
    missed_h = 29.0
    grid_h = body_h - missed_h
    row_h = grid_h / 2.0

    # Horizontal hairline between the two grid rows, and between grid and missed band.
    rule_y_mid = body_y + row_h
    out += _line(x, rule_y_mid, x + w, rule_y_mid,
                 stroke=COLOR_HAIRLINE, sw=0.5)
    out += _line(x, body_y + grid_h, x + w, body_y + grid_h,
                 stroke=COLOR_INK, sw=1.0)

    # Vertical hairlines between columns (only through the 2 grid rows).
    cum = x
    for cw in BRIEFING_COLS[:-1]:
        cum += cw
        out += _line(cum, body_y, cum, body_y + grid_h,
                     stroke=COLOR_HAIRLINE, sw=0.5)

    # --- Gather data --------------------------------------------------------
    brief = procedure.get("briefing_strip") or {}
    # Row 1 data
    approach_type = (procedure.get("approach_subtype") or "").upper() or "IAP"
    loc_freq = ""
    loc_ident = ""
    for f in procedure.get("fixes") or []:
        if f.get("type") == "navaid" and f.get("navaid_type") == "loc":
            loc_ident = f.get("id", "")
            if f.get("frequency_mhz") is not None:
                loc_freq = f"{f['frequency_mhz']:.2f}"
            break
    final_course = brief.get("final_approach_course_deg")
    if isinstance(final_course, int):
        fc_cell = f"{final_course:03d}\u00B0"
    elif final_course is not None:
        fc_cell = str(final_course)
    else:
        fc_cell = "—"
    gs_int = brief.get("gs_intercept_altitude_ft")
    gs_cell = f"{gs_int}'" if gs_int is not None else "—"

    da_val: int | None = None
    hat_val: int | None = None
    for m in procedure.get("minima") or []:
        if m.get("variant") == "s_ils" and "da_ft_msl" in m and not m.get("not_authorized"):
            da_val = m["da_ft_msl"]
            hat_val = m.get("dh_ft_agl")
            break

    # Row 2 data
    ta = brief.get("transition_altitude_ft")
    ta_cell = f"{ta}'" if ta is not None else "—"
    msa = procedure.get("msa") or {}
    msa_center = msa.get("center_navaid_id")
    msa_r = msa.get("radius_nm")
    if msa_center and msa_r is not None:
        msa_cell = f"{msa_center} {int(msa_r)}NM"
    elif msa_center:
        msa_cell = str(msa_center)
    else:
        msa_cell = "—"
    ap_el = procedure.get("airport_elev_ft_msl")
    ap_el_cell = f"{ap_el}'" if ap_el is not None else "—"
    tdze = procedure.get("tdze_ft_msl")
    tdze_cell = f"{tdze}'" if tdze is not None else "—"
    runway = str(procedure.get("runway_ident", "") or "—")
    mv = procedure.get("mag_variation_deg")
    if mv is None:
        mv_cell = "—"
    else:
        sign = "E" if mv >= 0 else "W"
        mv_cell = f"{abs(mv):.0f}\u00B0{sign}"

    # --- Render cells -------------------------------------------------------

    def _col_x(i: int) -> float:
        return x + sum(BRIEFING_COLS[:i])

    def _cell(i: int, row: int, caption: str, value: str,
              *, value_size: float = SIZE_STD, value_font: str | None = None,
              value_weight: str = "bold") -> str:
        cx = _col_x(i) + 6
        ry = body_y + row * row_h
        frag = _caption(cx, ry + 9, caption)
        frag += _text(
            cx, ry + row_h - 6, value,
            size=value_size, weight=value_weight, family=value_font,
        )
        return frag

    # Row 1
    out += _cell(0, 0, "APCH", approach_type)
    out += _cell(1, 0, "LOC FRQ", loc_freq or "—", value_font=FONT_MONO)
    out += _cell(2, 0, "IDENT", loc_ident or "—", value_font=FONT_MONO)
    out += _cell(3, 0, "FAC", fc_cell)
    out += _cell(4, 0, "GS ALT", gs_cell)

    # DA hero cell — 22 px hero number + HAT/DH secondary.
    cx_da = _col_x(5) + 6
    ry_da = body_y
    out += _caption(cx_da, ry_da + 9, "DA (MDA)")
    if da_val is not None:
        hero_y = ry_da + row_h - 6
        out += _text(
            cx_da, hero_y, f"{da_val}'",
            size=SIZE_HERO, weight="bold",
        )
        if hat_val is not None:
            # Place HAT right of the hero number.
            # Rough x offset: hero roughly 22*0.62*len(text).
            hero_text = f"{da_val}'"
            hero_w = SIZE_HERO * 0.62 * len(hero_text)
            out += _text(
                cx_da + hero_w + 8, hero_y - 11,
                "HAT", size=SIZE_MICRO, weight="bold",
                fill=COLOR_MUTED, tracking=TRACK,
            )
            out += _text(
                cx_da + hero_w + 8, hero_y - 1,
                f"{hat_val}'", size=SIZE_STD, weight="bold",
            )
    else:
        out += _text(cx_da, ry_da + row_h - 6, "—",
                     size=SIZE_HERO, weight="bold")

    # Row 2
    out += _cell(0, 1, "TRANS", ta_cell)
    out += _cell(1, 1, "MSA", msa_cell)
    out += _cell(2, 1, "APT EL", ap_el_cell)
    out += _cell(3, 1, "TDZE", tdze_cell)
    out += _cell(4, 1, "RWY", runway, value_font=FONT_MONO)
    out += _cell(5, 1, "MAG VAR", mv_cell)

    # --- Missed approach band ----------------------------------------------
    mb_y = body_y + grid_h
    missed = procedure.get("missed_approach") or {}
    ma_text = missed.get("text_description") or ""
    out += _caption(x + 6, mb_y + 10, "MISSED APPROACH")
    # Body text; truncate if it overruns the band width at SIZE_BODY.
    out += _text(
        x + 6, mb_y + missed_h - 9,
        _fit(ma_text, w - 12, SIZE_BODY),
        size=SIZE_BODY,
    )

    return out


def _fit(s: str, max_px: float, size_px: float, char_w_ratio: float = 0.52) -> str:
    """Approximate single-line fit by ratio-based truncation with ellipsis."""
    s = s or ""
    if not s:
        return s
    max_chars = int(max(1, max_px / (size_px * char_w_ratio)))
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)] + "\u2026"


# ---------------------------------------------------------------------------
# Comms footer
# ---------------------------------------------------------------------------


def _render_footer(procedure: dict, region: Region) -> str:
    x, y, w, h = region
    out = _section_bar(region, "COMMS")
    comms = (procedure.get("communications") or {}).get("frequencies") or []
    body_y = y + BAR_H
    body_h = h - BAR_H
    if not comms:
        return out

    n = len(comms)
    cell_w = w / n
    # Draw vertical hairlines between cells.
    for i in range(1, n):
        cx = x + i * cell_w
        out += _line(cx, body_y, cx, body_y + body_h,
                     stroke=COLOR_HAIRLINE, sw=0.5)

    # Jeppesen-style role abbreviations so longest labels never overflow
    # a narrow cell. The YAML `role` enum values (atis, clnc, gnd, twr,
    # dep, app, ctr, emergency, ...) render as short caps here.
    role_abbrev = {
        "emergency": "EMRG",
        "departure": "DEP",
        "approach": "APP",
        "center": "CTR",
        "clearance": "CLNC",
        "ground": "GND",
        "tower": "TWR",
    }
    for i, c in enumerate(comms):
        cx = x + i * cell_w + 6
        role_raw = (c.get("role") or "").lower()
        role = role_abbrev.get(role_raw, role_raw.upper())
        freq = c.get("frequency_mhz")
        freq_s = f"{freq:.2f}" if isinstance(freq, (int, float)) else "—"
        out += _caption(cx, body_y + 12, role)
        out += _text(
            cx, body_y + body_h - 10, freq_s,
            size=SIZE_STD, weight="bold", family=FONT_MONO,
        )
    return out


# ---------------------------------------------------------------------------
# Plan view
# ---------------------------------------------------------------------------


# Bounding box tuple used by the procedure-mask machinery.
BBox = tuple[float, float, float, float]  # (x, y, w, h)

# Padding applied to every procedure bbox before it's drawn into the mask.
# Small value — enough to keep basemap linework from nicking fix-id
# characters without cutting a visible hole around the overlay.
_MASK_BBOX_PAD = 2.0

# The mask id used when the plan-view renderer emits procedure-mask defs.
# render_basemap accepts this via its ``line_mask_id`` parameter; both must
# agree for the browser/cairosvg to resolve the reference.
_PLAN_MASK_ID = "planProcedureMask"


def _collect_procedure_bboxes(
    procedure: dict,
    fixes: dict[str, LatLon],
    primitives: list[Primitive],
    projector: Projector,
    region: Region,
) -> list[BBox]:
    """Collect axis-aligned bounding boxes for every procedure element drawn
    on top of the basemap.

    The returned list is deterministic in both order and content. Each bbox
    is padded by :data:`_MASK_BBOX_PAD` so the mask cuts a small margin
    around the visible element.

    Elements included:
      * fix symbols (16x16 square around projected position)
      * fix-id labels (SIZE_STD mono; len*6.5 x 12)
      * role labels '(IAF)/(IF)/...' (micro mono; (len+2)*6.5 x 10)
      * altitude boxes (34x12 at _altitude_box_anchor)
      * course labels (approx 20x10 at projected leg midpoint)
      * MSA bezel (95x95 upper-right)
      * runway (line extent padded 20 px)
      * hold racetrack (50x50 box centred on the hold fix)
      * notes sidebar rect (picked by _pick_notes_corner)
      * scale bar (~80x20 lower-left) + north arrow (~30x50 lower-right)
      * PLAN VIEW tag (upper-left)
    """
    rx, ry, rw, rh = region
    bboxes: list[BBox] = []

    def _add(x: float, y: float, w: float, h: float) -> None:
        bboxes.append((
            x - _MASK_BBOX_PAD,
            y - _MASK_BBOX_PAD,
            w + 2 * _MASK_BBOX_PAD,
            h + 2 * _MASK_BBOX_PAD,
        ))

    # --- Fix-related rectangles (sorted fix iteration for determinism) ---
    used = _collect_used_fix_ids(procedure)
    roles = _fix_roles(procedure)
    role_plan = _plan_role_labels(used, roles, fixes, projector)
    leg_dirs = _fix_leg_directions(procedure, fixes, projector)

    for fid in sorted(used):
        if fid not in fixes:
            continue
        fix = _find_fix(procedure, fid)
        if fix is None:
            continue
        fx, fy = projector(fixes[fid])

        # Fix symbol — ~16x16 centred on the projected point.
        _add(fx - 8.0, fy - 8.0, 16.0, 16.0)

        # Fix-id label — positioned at (fx+8, fy+10) in render_svg; the
        # text baseline sits at y=10, so the visual bbox top is ~9 px
        # above that (the task spec: (x+8, y+10 - 9)).
        label_w = len(fid) * 6.5
        _add(fx + 8.0, fy + 10.0 - 9.0, label_w, 12.0)

        # Role label (e.g. "(FAF)"), positioned at (fx+8, fy+21).
        role_text = role_plan.get(fid)
        if role_text:
            inner = f"({role_text})"
            _add(fx + 8.0, fy + 21.0 - 8.0, len(inner) * 6.5, 10.0)

        # Altitude box — the rect emitted by `_altitude_box`. Its anchor
        # is at (bx, by) with the rect drawn at (bx, by - 12 + 2), w=34,
        # h=12. We use the anchor's (bx, by - 10) to get the rect's top.
        con = _find_altitude_constraint(procedure, fid)
        if con and con.get("altitude_1_ft") is not None:
            bx, by = _altitude_box_anchor(fx, fy, leg_dirs.get(fid))
            _add(bx, by - 12.0 + 2.0, 34.0, 12.0)

    # --- Course labels along each leg ------------------------------------
    seen_courses: set[tuple[str, str]] = set()

    def _add_course(from_id: str | None, to_id: str | None,
                    course: object) -> None:
        if not (from_id and to_id and course is not None):
            return
        key = (from_id, to_id)
        if key in seen_courses:
            return
        if from_id not in fixes or to_id not in fixes:
            return
        seen_courses.add(key)
        a = projector(fixes[from_id])
        b = projector(fixes[to_id])
        mx = (a[0] + b[0]) / 2.0
        my = (a[1] + b[1]) / 2.0
        # 20x10 centred on the midpoint.
        _add(mx - 10.0, my - 5.0, 20.0, 10.0)

    for t in procedure.get("transitions") or []:
        for leg in t.get("legs") or []:
            _add_course(leg.get("from_fix_id"), leg.get("to_fix_id"),
                        leg.get("course_deg"))
    for leg in procedure.get("common_legs") or []:
        _add_course(leg.get("from_fix_id"), leg.get("to_fix_id"),
                    leg.get("course_deg"))

    # --- Runway centerlines (plan) ---------------------------------------
    for rwy in procedure.get("runways") or []:
        low = rwy.get("low_threshold_lat_lon") or {}
        high = rwy.get("high_threshold_lat_lon") or {}
        if not (low and high):
            continue
        a = projector((float(low["lat"]), float(low["lon"])))
        b = projector((float(high["lat"]), float(high["lon"])))
        x0 = min(a[0], b[0]) - 20.0
        y0 = min(a[1], b[1]) - 20.0
        x1 = max(a[0], b[0]) + 20.0
        y1 = max(a[1], b[1]) + 20.0
        _add(x0, y0, x1 - x0, y1 - y0)

    # --- Hold racetrack ---------------------------------------------------
    for p in primitives:
        if isinstance(p, Hold):
            fx, fy = projector(p.fix)
            # Hold geometry drawn by _render_hold spans ~28 leg x 16 wide;
            # a 50x50 bbox centred on the fix covers both orientations.
            _add(fx - 25.0, fy - 25.0, 50.0, 50.0)

    # --- MSA bezel (upper-right) -----------------------------------------
    msa = procedure.get("msa") or {}
    if msa.get("sectors"):
        msa_size = 95.0
        _add(rx + rw - msa_size - 4.0, ry + 4.0, msa_size, msa_size)

    # --- Notes sidebar ---------------------------------------------------
    brief = procedure.get("briefing_strip") or {}
    notes = brief.get("notes") or []
    if notes:
        nx, ny = _pick_notes_corner(
            procedure, fixes, primitives, projector, region,
        )
        # _render_plan_notes anchors the bottom edge of the box at
        # (ny + 60) - 10. Approximate the box height at ~60 px (matches
        # the nominal value used by _pick_notes_corner's scorer).
        nbox_w = 220.0
        nbox_h = 60.0
        box_bottom = (ny + 60.0) - 10.0
        box_top = box_bottom - nbox_h
        _add(nx, box_top, nbox_w, nbox_h)

    # --- Scale bar + north arrow + PLAN VIEW tag --------------------------
    # Scale bar: anchored at (rx+12, ry+rh-14), extending a variable width.
    # The exact length depends on px-per-nm; 80x20 is a safe over-estimate.
    _add(rx + 6.0, ry + rh - 22.0, 80.0, 22.0)
    # North arrow: centred at (rx+rw-20, ry+rh-30), ~30 wide x 50 tall
    # (arrow runs from cy-14 label down through cy+22 VAR caption).
    _add(rx + rw - 36.0, ry + rh - 48.0, 32.0, 48.0)
    # PLAN VIEW reversed-chrome tag: 68 x BAR_H in the upper-left.
    _add(rx, ry, 68.0, BAR_H)

    return bboxes


def _render_procedure_mask_defs(
    mask_id: str, region: Region, bboxes: list[BBox],
) -> str:
    """Emit a ``<defs><mask>...</mask></defs>`` block whose interior starts
    white (fully visible) and then punches a black rectangle for each
    procedure bbox (black = hidden wherever the mask applies).
    """
    rx, ry, rw, rh = region
    parts = [
        f'<defs><mask id="{mask_id}" maskUnits="userSpaceOnUse" '
        f'x="{_fmt(rx)}" y="{_fmt(ry)}" '
        f'width="{_fmt(rw)}" height="{_fmt(rh)}">\n',
        f'<rect x="{_fmt(rx)}" y="{_fmt(ry)}" '
        f'width="{_fmt(rw)}" height="{_fmt(rh)}" fill="white"/>\n',
    ]
    for bx, by, bw, bh in bboxes:
        parts.append(
            f'<rect x="{_fmt(bx)}" y="{_fmt(by)}" '
            f'width="{_fmt(bw)}" height="{_fmt(bh)}" fill="black"/>\n'
        )
    parts.append("</mask></defs>\n")
    return "".join(parts)


def _render_plan_view(
    procedure: dict,
    primitives: list[Primitive],
    fixes: dict[str, LatLon],
    region: Region,
) -> str:
    x, y, w, h = region
    # The plan view is the hero — bordered frame, no full section bar.
    # We float a small reversed-chrome tag in the upper-left instead.
    out = _rect(region, stroke=COLOR_INK, sw=1.0)

    # Clip the plan content so nothing leaks outside the rectangle.
    clip_id = "planClip"
    out += (
        f'<defs><clipPath id="{clip_id}">'
        f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}"/>'
        f'</clipPath></defs>\n'
    )
    out += f'<g clip-path="url(#{clip_id})">\n'

    projector = plan_projector(procedure, fixes, (x, y, w, h), primitives=primitives)

    # All plan-view labels flow through a single placement engine. Fixed
    # geometry (procedure path, LOC feather, fix symbols, navaid symbols,
    # runways, MSA bezel, notes sidebar, scale bar, north arrow,
    # PLAN VIEW tag) reserves bboxes up-front so labels route around it.
    plan_box = PBBox(x, y, w, h)
    with _PlanLabelCtx(plan_box) as ctx:
        # --- Pre-register fixed-geometry reservations -----------------
        _register_plan_geometry(ctx, procedure, fixes, primitives, projector,
                                region)

        # Basemap underlay — silently omitted if the procedure doesn't
        # declare one or the data file is missing. Drawn FIRST so every
        # other layer (runways, procedure primitives, fix symbols) sits
        # on top. We keep the procedure-bbox SVG mask as a second line
        # of defence (basemap linework that sneaks past the placement
        # engine because it has no label associated with it — e.g. a
        # river segment crossing a fix symbol — still gets cut by the
        # mask).
        bm_decl = procedure.get("basemap") or None
        if bm_decl is True:
            bm_decl = {}
        basemap_label_requests: list = []
        if isinstance(bm_decl, dict):
            region_id = bm_decl.get("region")
            # Pass the geographic extent the projector fits to so
            # ``load_basemap`` can warn when the sibling files don't
            # cover the full plan view. Covers the regression case
            # where a procedure at an existing airport demands more
            # extent than the region was built for.
            world_bbox = plan_world_bbox(procedure, fixes, primitives)
            if not region_id and world_bbox is not None:
                padded = plan_world_bbox(procedure, fixes, primitives, pad_frac=0.50)
                if padded is not None:
                    region_id = find_covering_region(padded)
            geojson: dict | None = None
            if region_id:
                geojson = load_basemap(
                    str(region_id), plan_bbox=world_bbox,
                )
            if geojson is not None:
                cfg = BasemapConfig.from_mapping(bm_decl)
                bboxes = _collect_procedure_bboxes(
                    procedure, fixes, primitives, projector, (x, y, w, h),
                )
                out += _render_procedure_mask_defs(
                    _PLAN_MASK_ID, (x, y, w, h), bboxes,
                )
                out += render_basemap(
                    geojson, cfg, projector,
                    line_mask_id=_PLAN_MASK_ID,
                    plan_rect=(x, y, x + w, y + h),
                    placement_ctx=ctx,
                )

        # Runways (drawn first, under everything else).
        out += _render_runways(procedure, projector)

        # LOC feather drawn AFTER procedure primitives so its outlines sit
        # on top of the magenta casing (otherwise the paper-halo casing
        # masks the feather on straight-in ILS approaches where the
        # feather and inbound procedure course coincide).
        out += _render_primitives(primitives, projector, procedure)
        out += _render_localizer_ribbon(procedure, projector)
        out += _render_fix_symbols(
            procedure, fixes, primitives, projector,
            engine=ctx.engine, plan_rect=plan_box,
        )

        # Scale bar + north arrow (cyan reference overlays).
        out += _render_scale_bar(projector, fixes, region)
        out += _render_north_arrow(procedure, region)

        # Resolve every label request collected so far and emit SVG.
        # This is the single engine.place() call for the plan view.
        out += ctx.resolve_and_emit()

        out += "</g>\n"

    # MSA bezel (unclipped so it lives atop the plan rectangle border).
    msa_size = 95.0
    msa_region = (x + w - msa_size - 4, y + 4, msa_size, msa_size)
    out += _render_msa_bezel(procedure, fixes, msa_region)

    # Floating tag in the upper-left — reversed chrome matching every
    # other section bar (BAR_H tall, 6 px left pad, SIZE_BODY tracked
    # caps), so the four section-bar labels read as one unified run of
    # chrome across the page.
    tag_w = 68.0
    out += _rect((x, y, tag_w, BAR_H), fill=COLOR_INK)
    out += _text(
        x + 6, y + BAR_H - 4, "PLAN VIEW",
        size=SIZE_BODY, weight="bold",
        fill=COLOR_PAPER, tracking=TRACK,
    )

    # Notes sidebar — compact box tucked into whichever corner has the
    # fewest projected features. On ETAD, NE quadrant is busy (SPANG,
    # MSA bezel); on UGSB, SW is empty and matches the legacy position.
    notes_xy = _pick_notes_corner(
        procedure, fixes, primitives, projector, (x, y, w, h),
    )
    out += _render_plan_notes(procedure, *notes_xy)

    return out


def _register_plan_geometry(
    ctx: _PlanLabelCtx,
    procedure: dict,
    fixes: dict[str, LatLon],
    primitives: list[Primitive],
    projector: Projector,
    region: Region,
) -> None:
    """Reserve bboxes for every non-label plan-view element.

    Delegates to ``_collect_procedure_bboxes`` for the existing set
    (fix symbols, altitude box anchors, course-label slots, runways,
    holds, MSA bezel, notes sidebar, scale bar, north arrow, PLAN VIEW
    tag) and registers each as a ``PROCEDURE``/``FIX``/``AIRSPACE``/
    ``REFERENCE``-tier fixed reservation. We don't need to register
    altitude/fix/role/course slots here — the engine collects those
    through the label-request path — but the runways, hold racetrack,
    MSA bezel, and reference furniture MUST be reservations so label
    placement avoids them.
    """
    rx, ry, rw, rh = region

    # Procedure path primitives are NOT registered as fixed reservations.
    # They are thin (~2 px) magenta lines with paper-halo labels on top;
    # treating a 100 px-long Segment as an axis-aligned bbox over the
    # whole rectangle it spans would block every fix label. Instead, the
    # fix symbols (which sit ON the procedure track at its endpoints)
    # carry the small reservation that matters for label placement.
    #
    # Hold racetracks are tracked because they're a bigger visual mass
    # than a straight leg — labels should route around them.
    for p in primitives:
        if isinstance(p, Hold):
            fx, fy = projector(p.fix)
            ctx.add_fixed(
                f"hold:{_fmt(fx)},{_fmt(fy)}",
                PlacementTier.PROCEDURE,
                PBBox(fx - 25.0, fy - 25.0, 50.0, 50.0),
            )

    # Runways — threshold symbols only. Runways are long and narrow; a
    # full bbox would again block labels along the whole runway line
    # (including the threshold fix's label). Labels aren't supposed to
    # sit exactly on a runway threshold symbol but anywhere else along
    # the line is fine.
    for i, rwy in enumerate(procedure.get("runways") or []):
        for end_key in ("low_threshold_lat_lon", "high_threshold_lat_lon"):
            end = rwy.get(end_key) or {}
            if not end:
                continue
            tx, ty = projector((float(end["lat"]), float(end["lon"])))
            ctx.add_fixed(
                f"rwy:{i:02d}:{end_key[0]}",
                PlacementTier.NAVAID,
                PBBox(tx - 6.0, ty - 6.0, 12.0, 12.0),
            )

    # MSA bezel — upper-right 95x95.
    msa = procedure.get("msa") or {}
    if msa.get("sectors"):
        msa_size = 95.0
        ctx.add_fixed(
            "msa_bezel", PlacementTier.AIRSPACE,
            PBBox(rx + rw - msa_size - 4.0, ry + 4.0, msa_size, msa_size),
        )

    # Reference furniture (scale bar, north arrow, PLAN VIEW tag, notes).
    ctx.add_fixed(
        "plan_tag", PlacementTier.REFERENCE,
        PBBox(rx, ry, 68.0, BAR_H),
    )
    ctx.add_fixed(
        "scale_bar", PlacementTier.REFERENCE,
        PBBox(rx + 6.0, ry + rh - 22.0, 80.0, 22.0),
    )
    ctx.add_fixed(
        "north_arrow", PlacementTier.REFERENCE,
        PBBox(rx + rw - 36.0, ry + rh - 48.0, 32.0, 48.0),
    )
    brief = procedure.get("briefing_strip") or {}
    if (brief.get("notes") or []):
        nx, ny = _pick_notes_corner(
            procedure, fixes, primitives, projector, region,
        )
        nbox_w = 220.0
        nbox_h = 60.0
        box_bottom = (ny + 60.0) - 10.0
        box_top = box_bottom - nbox_h
        ctx.add_fixed(
            "notes", PlacementTier.REFERENCE,
            PBBox(nx, box_top, nbox_w, nbox_h),
        )


def _wrap_text(s: str, max_px: float, size_px: float,
               char_w_ratio: float = 0.52,
               max_lines: int = 3) -> list[str]:
    """Greedy word-wrap by approximate character width.

    Breaks on spaces only. If the text exceeds ``max_lines`` when wrapped,
    the final visible line is truncated with an ellipsis.
    """
    s = (s or "").strip()
    if not s:
        return []
    max_chars = max(1, int(max_px / (size_px * char_w_ratio)))
    words = s.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cand = w
        else:
            cand = cur + " " + w
        if len(cand) <= max_chars:
            cur = cand
            continue
        # Word doesn't fit on the current line.
        if cur:
            lines.append(cur)
        # If the single word is longer than max_chars, hard-break it.
        while len(w) > max_chars:
            lines.append(w[:max_chars])
            w = w[max_chars:]
            if len(lines) >= max_lines:
                break
        cur = w
        if len(lines) >= max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    # If the source text has more content than fit, mark the last line with an ellipsis.
    joined_len = sum(len(x) for x in lines) + max(0, len(lines) - 1)  # spaces between
    # Approximate: if we dropped words, add ellipsis to the last rendered line.
    total_src = sum(len(w) for w in words) + max(0, len(words) - 1)
    if joined_len < total_src and lines:
        last = lines[-1]
        if len(last) >= max_chars:
            last = last[: max_chars - 1].rstrip() + "\u2026"
        else:
            last = last.rstrip() + "\u2026"
        lines[-1] = last
    return lines


def _notes_box_dims(procedure: dict) -> tuple[float, float]:
    """Return (width, height) of the notes box for placement scoring."""
    brief = procedure.get("briefing_strip") or {}
    notes = brief.get("notes") or []
    if not notes:
        return (0.0, 0.0)
    box_w = 147.0
    pad = 3.5
    line_h = 7.0
    caption_h = 7.0
    sev_col_w = 30.0
    max_lines_per_note = 3
    note_font = SIZE_MICRO
    text_px = box_w - 2 * pad - sev_col_w
    total_lines = 0
    for n in notes:
        txt = n.get("text") or ""
        if not txt:
            continue
        lines = _wrap_text(txt, text_px, note_font, max_lines=max_lines_per_note)
        total_lines += len(lines)
    box_h = caption_h + total_lines * line_h + pad * 2
    return (box_w, box_h)


def _pick_notes_corner(
    procedure: dict,
    fixes: dict[str, LatLon],
    primitives: list[Primitive],
    projector: Projector,
    region: Region,
) -> tuple[float, float]:
    """Choose the plan-view corner with the fewest projected features
    inside the notes rectangle."""
    rx, ry, rw, rh = region

    box_w, box_h = _notes_box_dims(procedure)
    if box_w == 0:
        return (rx + 4.0, ry + rh - 60)
    inset = 4.0

    corners = [
        ("sw", rx + inset, ry + rh - box_h),
        ("nw", rx + inset, ry + BAR_H + 4),
        ("se", rx + rw - box_w - inset, ry + rh - box_h),
        ("ne", rx + rw - box_w - inset, ry + BAR_H + 4),
    ]

    # Build the set of projected feature points to test against.
    points: list[tuple[float, float]] = []
    # Fix positions.
    for fid, ll in fixes.items():
        points.append(projector(ll))
    # Runway threshold endpoints — these MUST read, so weight them by
    # adding a few samples along each runway centreline.
    for rw_def in procedure.get("runways") or []:
        low = rw_def.get("low_threshold_lat_lon") or {}
        high = rw_def.get("high_threshold_lat_lon") or {}
        if not (low and high):
            continue
        a = projector((float(low["lat"]), float(low["lon"])))
        b = projector((float(high["lat"]), float(high["lon"])))
        # 5 samples along the runway so a notes box covering ANY part of
        # it counts as a hit.
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            points.append((a[0] + (b[0] - a[0]) * t,
                           a[1] + (b[1] - a[1]) * t))
    # Procedure primitives — sample Segment endpoints and Arc centres.
    for p in primitives:
        if isinstance(p, Segment):
            points.append(projector(p.start))
            points.append(projector(p.end))
        elif isinstance(p, Arc):
            points.append(projector(p.center))
        elif isinstance(p, Hold):
            points.append(projector(p.fix))

    def _score(cx: float, cy: float) -> int:
        margin = 20.0
        x0, x1 = cx - margin, cx + box_w + margin
        y0, y1 = cy - margin, cy + box_h + margin
        n = 0
        for px, py in points:
            if x0 <= px <= x1 and y0 <= py <= y1:
                n += 1
        return n

    ranked = sorted(
        corners,
        key=lambda c: (_score(c[1], c[2]),
                       {"sw": 0, "nw": 1, "se": 2, "ne": 3}[c[0]]),
    )
    chosen = ranked[0]
    print(f"  notes-corner chosen: {chosen[0]}")
    # _render_plan_notes expects the anchor (x, y_top_candidate) where
    # the box's bottom sits 10 px above y+60. Re-derive its y parameter
    # so that the box BOTTOM lands at the corner-bottom we scored.
    # For bottom corners:   box bottom = ry + rh - inset  →  y param = box_bottom - 60 + 10
    # For top corners:      box top ≈ ry + 16           →  y param = box_top + ??? (legacy anchor math: y + 60 - 10 - box_h = box_top  →  y = box_top + box_h - 50)
    _, cx, cy = chosen
    if chosen[0] in ("sw", "se"):
        # Desired box bottom = ry + rh - inset; _render_plan_notes draws
        # the bottom at (y + 60) - 10. Solve for y:
        y_param = (ry + rh - inset) - 60 + 10
    else:
        # Top corners: desired box top = cy. Solve (y + 60) - 10 - h = cy
        # without knowing h ahead of time — approximate h = box_h and
        # nudge so the box sits just under the PLAN VIEW tag.
        y_param = cy + box_h - 50
    return (cx, y_param)


def _render_plan_notes(procedure: dict, x: float, y: float) -> str:
    brief = procedure.get("briefing_strip") or {}
    notes = brief.get("notes") or []
    if not notes:
        return ""
    # Widened to 220 px (still < 40 % of plan-view width 576 px) to give
    # wrapped text more room before truncation kicks in.
    box_w = 147.0
    pad = 3.5
    line_h = 7.0
    caption_h = 7.0
    sev_col_w = 30.0
    max_lines_per_note = 3
    note_font = SIZE_MICRO  # ~6.5 px instead of SIZE_BODY 8 px

    wrapped: list[tuple[str, list[str]]] = []
    text_px = box_w - 2 * pad - sev_col_w
    for n in notes:
        sev = (n.get("severity") or "info").upper()
        txt = n.get("text") or ""
        if not txt:
            continue
        lines = _wrap_text(txt, text_px, note_font, max_lines=max_lines_per_note)
        if lines:
            wrapped.append((sev, lines))
    if not wrapped:
        return ""

    total_text_lines = sum(len(ls) for _, ls in wrapped)
    box_h = caption_h + total_text_lines * line_h + pad * 2
    bottom_inset = 10.0
    box_y = (y + 60) - bottom_inset - box_h
    out = _rect((x, box_y, box_w, box_h),
                fill=COLOR_PAPER, stroke=COLOR_INK, sw=0.5)
    out += _text(x + pad, box_y + pad + 5, "NOTES",
                 size=SIZE_MICRO, weight="bold", tracking=TRACK)

    ly_cursor = box_y + caption_h + pad + line_h - 1
    for sev, lines in wrapped:
        out += _text(x + pad, ly_cursor, sev,
                     size=SIZE_MICRO - 1.5, weight="bold",
                     fill=COLOR_INK if sev == "WARNING" else COLOR_MUTED,
                     tracking=TRACK)
        for i, txt in enumerate(lines):
            out += _text(x + pad + sev_col_w, ly_cursor + i * line_h,
                         txt, size=note_font)
        ly_cursor += len(lines) * line_h
    return out


def _render_localizer_ribbon(procedure: dict, projector: Projector) -> str:
    """Jeppesen-style LOC feather — three thin lines diverging slightly
    outward from the runway threshold back along the reciprocal of the
    final approach course, plus perpendicular DME tick marks on the
    centerline at 2 NM intervals.

    Draws the LOC centerline and two edge lines at +/-1.5 deg of
    divergence from the course, each 8 NM long. Solid thin strokes, no
    hatching, no fill. Reads clearly at plate scale because the feather
    is literally three thin straight lines plus four ticks.

    Only emitted for ILS / LOC approaches. VOR/NDB/RNAV circling
    approaches without a localizer return an empty string.
    """
    subtype = (procedure.get("approach_subtype") or "").lower()
    if subtype not in ("ils", "loc", "ils_dme", "loc_dme", "ils_cat1", "ils_cat2", "ils_cat3"):
        return ""
    # Derive the feather direction from the PUBLISHED PROCEDURE GEOMETRY,
    # not from a declared course. Real chart production pipelines do the
    # same: the LOC feather is a visual extension of the final-segment
    # fix-to-threshold geodesic, guaranteed to sit along the rendered
    # approach path regardless of magnetic vs. true convention mismatches
    # in the source YAML.
    from .geodesy import initial_bearing_deg
    fixes_tbl = build_fix_table(procedure)
    # Find the last CF/TF leg in common_legs ending at a runway_threshold.
    source_fix_id = None
    thresh_fix_id = None
    fixes_by_id = {f.get("id"): f for f in (procedure.get("fixes") or [])}
    for leg in reversed(procedure.get("common_legs") or []):
        if leg.get("terminator") in ("CF", "TF") and leg.get("to_fix_id"):
            to_id = leg["to_fix_id"]
            if fixes_by_id.get(to_id, {}).get("type") == "runway_threshold":
                source_fix_id = leg.get("from_fix_id")
                thresh_fix_id = to_id
                break
    if not source_fix_id or not thresh_fix_id:
        return ""
    if source_fix_id not in fixes_tbl or thresh_fix_id not in fixes_tbl:
        return ""
    thresh_ll = fixes_tbl[thresh_fix_id]
    course_deg = initial_bearing_deg(fixes_tbl[source_fix_id], thresh_ll)

    # Standard LOC feather is 8 NM -- matches published LOC service
    # volume. Edges diverge at +/-3.0 deg from the centerline (~0.84 NM
    # half-width at 8 NM). This is VISUALLY EXAGGERATED from the real
    # ILS localizer sensitivity cone (~+/-1.5 deg) for plate-scale
    # readability — same cosmetic principle as the 8 px minimum runway
    # width: actual geometry would read as a hairline sliver against
    # the procedure casing and lose its distinct three-line silhouette.
    LENGTH_NM = 8.0
    EDGE_DIVERGENCE_DEG = 3.0
    TICK_INTERVAL_NM = 2.0
    TICK_LEN_PX = 3.0
    STROKE_W = 0.5

    recip = (float(course_deg) + 180.0) % 360.0
    # Three endpoints: centerline straight back, upper/lower edges diverged.
    center_far = destination(thresh_ll, recip, LENGTH_NM)
    upper_far = destination(thresh_ll, (recip + EDGE_DIVERGENCE_DEG) % 360.0, LENGTH_NM)
    lower_far = destination(thresh_ll, (recip - EDGE_DIVERGENCE_DEG + 360.0) % 360.0, LENGTH_NM)

    a = projector(thresh_ll)
    c = projector(center_far)
    u = projector(upper_far)
    l = projector(lower_far)

    frag = ""
    # Three thin solid lines: upper edge, centerline, lower edge.
    frag += _line(a[0], a[1], u[0], u[1], stroke=COLOR_INK, sw=STROKE_W)
    frag += _line(a[0], a[1], c[0], c[1], stroke=COLOR_INK, sw=STROKE_W)
    frag += _line(a[0], a[1], l[0], l[1], stroke=COLOR_INK, sw=STROKE_W)

    # Perpendicular tick marks on the centerline at integer DME values
    # (2, 4, 6, 8 NM from the threshold). Classic LOC DME tick markers;
    # tick length is in SCREEN px so they stay readable regardless of
    # map scale.
    from math import hypot
    cvx = c[0] - a[0]
    cvy = c[1] - a[1]
    cvl = hypot(cvx, cvy)
    if cvl > 0:
        # Unit perpendicular vector in screen space.
        px = -cvy / cvl
        py = cvx / cvl
        n_ticks = int(LENGTH_NM // TICK_INTERVAL_NM)
        for i in range(1, n_ticks + 1):
            t = (i * TICK_INTERVAL_NM) / LENGTH_NM
            tx = a[0] + t * cvx
            ty = a[1] + t * cvy
            x0 = tx - px * TICK_LEN_PX
            y0 = ty - py * TICK_LEN_PX
            x1 = tx + px * TICK_LEN_PX
            y1 = ty + py * TICK_LEN_PX
            frag += _line(x0, y0, x1, y1, stroke=COLOR_INK, sw=STROKE_W)
    return frag


def _render_runways(procedure: dict, projector: Projector) -> str:
    """Draw runways as pavement polygons with dashed centerline extensions.

    Overlapping runways (e.g. LSZH's three crossing strips) are merged
    into a single outline via polygon union so there are no interior
    stroke lines at intersections. Non-overlapping runways keep their
    own outlines.
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    FT_PER_NM = 6076.12
    DEFAULT_WIDTH_FT = 150.0
    MIN_RUNWAY_WIDTH_PX = 3.5
    CENTERLINE_EXT_NM = 1.0
    PAVEMENT_FILL = "#E8E4DA"

    runways = list(procedure.get("runways") or [])
    runways.sort(key=lambda rw: (
        str(rw.get("ident_low_end") or ""),
        str(rw.get("ident_high_end") or ""),
    ))

    shapely_polys: list[ShapelyPolygon] = []
    centerline_frags: list[str] = []

    for rw in runways:
        low = rw.get("low_threshold_lat_lon") or {}
        high = rw.get("high_threshold_lat_lon") or {}
        if not (low and high):
            continue
        low_ll = (float(low["lat"]), float(low["lon"]))
        high_ll = (float(high["lat"]), float(high["lon"]))
        a = projector(low_ll)
        b = projector(high_ll)
        vx = b[0] - a[0]
        vy = b[1] - a[1]
        length = (vx * vx + vy * vy) ** 0.5
        if length <= 0.0:
            continue
        ux, uy = vx / length, vy / length
        perp_x, perp_y = -uy, ux

        width_ft = float(rw.get("width_ft") or DEFAULT_WIDTH_FT)
        px_per_nm = projector_scale_px_per_nm(projector, low_ll)
        scale_px_per_ft = px_per_nm / FT_PER_NM
        width_px = max(width_ft * scale_px_per_ft, MIN_RUNWAY_WIDTH_PX)
        half = width_px / 2.0

        corners = [
            (a[0] + perp_x * half, a[1] + perp_y * half),
            (b[0] + perp_x * half, b[1] + perp_y * half),
            (b[0] - perp_x * half, b[1] - perp_y * half),
            (a[0] - perp_x * half, a[1] - perp_y * half),
        ]
        shapely_polys.append(ShapelyPolygon(corners))

        ext_px = CENTERLINE_EXT_NM * px_per_nm
        low_ext = (a[0] - ux * ext_px, a[1] - uy * ext_px)
        high_ext = (b[0] + ux * ext_px, b[1] + uy * ext_px)
        centerline_frags.append(
            _line(low_ext[0], low_ext[1], a[0], a[1],
                  stroke=COLOR_INK, sw=0.4, dash="3 2")
        )
        centerline_frags.append(
            _line(b[0], b[1], high_ext[0], high_ext[1],
                  stroke=COLOR_INK, sw=0.4, dash="3 2")
        )

    if not shapely_polys:
        return ""

    merged = unary_union(shapely_polys)

    frag = ""

    def _emit_polygon(geom) -> str:
        xs, ys = geom.exterior.coords.xy
        pts = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in zip(xs, ys))
        return (
            f'<polygon points="{pts}" '
            f'fill="{PAVEMENT_FILL}" stroke="{COLOR_INK}" '
            f'stroke-width="0.75" stroke-linejoin="miter"/>\n'
        )

    if merged.geom_type == "Polygon":
        frag += _emit_polygon(merged)
    elif merged.geom_type == "MultiPolygon":
        for poly in merged.geoms:
            frag += _emit_polygon(poly)

    frag += "".join(centerline_frags)

    return frag


def _is_missed_primitive(prim: Primitive, missed_fix_ids: set[str], fixes: dict[str, LatLon]) -> bool:
    """Heuristic — is this primitive part of the missed approach?

    Retained for back-compat with older callers and tests. The primary
    caller (:func:`_render_primitives`) now uses a deterministic walk-
    based partition instead — see :func:`_missed_primitive_start` — so
    fly-by arcs inside the missed phase (e.g. the DF semicircle at SPA)
    get correctly tagged even though their centre doesn't colocate
    with a published missed fix.
    """
    if isinstance(prim, Hold):
        return True  # UGSB's only Hold is the MAHP
    missed_points = [fixes[i] for i in missed_fix_ids if i in fixes]

    def _near(a: LatLon, b: LatLon) -> bool:
        return abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6

    if isinstance(prim, Segment):
        return any(_near(prim.start, p) or _near(prim.end, p) for p in missed_points)
    if isinstance(prim, Arc):
        return any(_near(prim.center, p) for p in missed_points)
    return False


def _missed_primitive_start(procedure: dict, fixes: dict[str, LatLon]) -> int:
    """Re-run just the non-missed legs through the dispatcher to count
    how many primitives they emit. Every primitive at or after this
    offset in ``compile_legs(...)`` output belongs to the missed approach.

    This mirrors :func:`open_plates.legs.compile_legs` but stops
    before ``missed_approach.legs`` — so it handles fly-by arcs on the
    approach side without double-counting and without requiring any new
    geometric heuristic for missed-side arcs.
    """
    out: list = []
    for transition in procedure.get("transitions") or []:
        ctx = LegContext()
        for leg in transition.get("legs") or []:
            _wps, prims, ctx = _dispatch(leg, ctx, fixes)
            out.extend(prims)
    ctx = LegContext()
    for leg in procedure.get("common_legs") or []:
        _wps, prims, ctx = _dispatch(leg, ctx, fixes)
        out.extend(prims)
    return len(out)


def _collect_missed_fix_ids(procedure: dict) -> set[str]:
    """Fix IDs that appear only in missed_approach legs."""
    missed = (procedure.get("missed_approach") or {}).get("legs") or []
    ids: set[str] = set()
    for leg in missed:
        for key in ("from_fix_id", "to_fix_id"):
            v = leg.get(key)
            if v:
                ids.add(v)
    for t in procedure.get("transitions") or []:
        for leg in t.get("legs") or []:
            for key in ("from_fix_id", "to_fix_id"):
                v = leg.get(key)
                if v:
                    ids.discard(v)
    for leg in procedure.get("common_legs") or []:
        for key in ("from_fix_id", "to_fix_id"):
            v = leg.get(key)
            if v:
                ids.discard(v)
    return ids


def _render_primitives(
    primitives: list[Primitive],
    projector: Projector,
    procedure: dict,
) -> str:
    """Emit the procedure path as ONE polyline per state of flight.

    A "state of flight" is one flyable trajectory from an entry point
    to a termination point (IAF→runway for each approach transition,
    runway→MAHP for the missed). Each state renders as exactly two
    SVG ``<polyline>`` elements — a wider paper casing below a thinner
    coloured core — so joints between primitives disappear entirely,
    the casing can never clip an adjacent primitive, and stroke style
    tunes per-state in one place.

    Hold primitives render separately after the polyline of the state
    they belong to (they're a racetrack symbol, not part of the track).

    The ``primitives`` argument is kept for signature compatibility
    (upstream callers pass the flat list from :func:`compile_legs`
    for bbox/mask construction); the renderer itself recomputes the
    flight states directly from ``procedure`` so it can label and
    style them coherently.
    """
    fixes = build_fix_table(procedure)
    states = compile_flight_states(procedure, fixes)

    frag = ""
    for state in states:
        frag += _render_flight_state(state, projector)
    return frag


# Stroke widths for the state polylines. Core widths match the legacy
# per-primitive values (2.0 / 1.5); the casing is 1.5 px wider so the
# paper halo reads at plate scale without fighting adjacent linework.
_APPROACH_CORE_SW: float = 2.0
_MISSED_CORE_SW: float = 1.5
_CASING_EXTRA_SW: float = 1.5


def _render_flight_state(state: FlightState, projector: Projector) -> str:
    """Emit one state's polyline casing + core, plus any Holds it owns."""
    if state.phase == "missed":
        color = COLOR_MISSED
        core_sw = _MISSED_CORE_SW
        core_dash = 'stroke-dasharray="4 3" '
    else:
        color = COLOR_PROCEDURE
        core_sw = _APPROACH_CORE_SW
        core_dash = ""

    frag = ""
    if state.points and len(state.points) >= 2:
        projected = [projector(pt) for pt in state.points]
        pts_str = " ".join(f"{_fmt(px)},{_fmt(py)}" for px, py in projected)

        # Casing: paper halo, solid even under dashed missed stroke.
        casing_sw = core_sw + _CASING_EXTRA_SW
        frag += (
            f'<polyline points="{pts_str}" fill="none" '
            f'stroke="{COLOR_PAPER}" stroke-width="{_fmt(casing_sw)}" '
            f'stroke-linejoin="round" stroke-linecap="round"/>\n'
        )
        # Core stroke in procedure / missed colour.
        frag += (
            f'<polyline points="{pts_str}" fill="none" '
            f'stroke="{color}" stroke-width="{_fmt(core_sw)}" '
            f'stroke-linejoin="round" stroke-linecap="round" '
            f'{core_dash}/>\n'
        )

    # Holds belonging to this state (e.g. the MAHP racetrack at the
    # missed state's termination fix). The missed state's colour is
    # already ink-dashed; the hold renders using the same colour.
    for hold in state.holds:
        frag += _render_hold(hold, projector, color)
    return frag


def _render_arc_as_polyline(arc: Arc, projector: Projector, color: str, dash: str, sw: float) -> str:
    samples = 16
    start = arc.start_bearing_deg
    end = arc.end_bearing_deg
    if arc.clockwise:
        sweep = (end - start + 360.0) % 360.0
    else:
        sweep = -((start - end + 360.0) % 360.0)
    pts: list[tuple[float, float]] = []
    for i in range(samples + 1):
        t = i / samples
        brg = (start + sweep * t) % 360.0
        ll = destination(arc.center, brg, arc.radius_nm)
        pts.append(projector(ll))
    d = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in pts)
    return (
        f'<polyline points="{d}" fill="none" stroke="{color}" '
        f'stroke-width="{_fmt(sw)}" {dash}/>\n'
    )


def _render_hold(hold: Hold, projector: Projector, color: str) -> str:
    """Render a holding-pattern racetrack anchored at the fix.

    Per ICAO Annex 4, FAA AIM §5-3-8, and Jeppesen chart legend: the
    hold fix sits at ONE END of the racetrack (the inbound-leg
    terminus), NOT at the center. The racetrack body extends OUTBOUND
    from the fix, offset laterally on the turn-direction side.

    Geometry (screen space, y-down):
      * fwd = unit vector along the inbound course (flight direction
        arriving at the fix). θ = 0 is north, so fwd = (sin θ, -cos θ).
      * lateral = perpendicular to fwd on the turn-direction side
        (right for right turns, left for left). Racetrack body lies
        on that side.
      * r = turn radius (half the racetrack width).
      * leg = straight-leg length along the inbound/outbound direction.

    Four anchor points:
      P1 = fix                                  (inbound-leg end = fix-end arc start)
      P2 = fix + lateral * 2r                   (fix-end arc end   = outbound-leg start)
      P3 = P2 − fwd * leg                       (outbound-leg end  = far-end arc start)
      P4 = P3 − lateral * 2r                    (far-end arc end   = inbound-leg start)

    Path: start at P4, fly inbound (L to P1), fix-end 180° arc (A to
    P2), outbound (L to P3), far-end 180° arc (A to P4, closing).

    Inbound-direction arrowhead is drawn on the inbound leg (P4→P1),
    pointing at P1 (the fix), per chart convention.
    """
    fx = projector(hold.fix)
    leg_px = 28.0
    r_px = 8.0  # turn radius; racetrack width = 2r = 16 px
    theta = radians(hold.inbound_course_deg)
    fwd = (sin(theta), -cos(theta))
    # Right-perp of fwd in screen-space y-down: rotate fwd 90° clockwise.
    right_perp = (-fwd[1], fwd[0])
    lateral = right_perp if hold.turn_direction == "right" else (-right_perp[0], -right_perp[1])

    p1 = fx
    p2 = (p1[0] + lateral[0] * 2 * r_px, p1[1] + lateral[1] * 2 * r_px)
    p3 = (p2[0] - fwd[0] * leg_px, p2[1] - fwd[1] * leg_px)
    p4 = (p3[0] - lateral[0] * 2 * r_px, p3[1] - lateral[1] * 2 * r_px)

    sweep = "1" if hold.turn_direction == "right" else "0"
    path = (
        f"M{_fmt(p4[0])},{_fmt(p4[1])} "
        f"L{_fmt(p1[0])},{_fmt(p1[1])} "
        f"A {_fmt(r_px)} {_fmt(r_px)} 0 0 {sweep} "
        f"{_fmt(p2[0])},{_fmt(p2[1])} "
        f"L{_fmt(p3[0])},{_fmt(p3[1])} "
        f"A {_fmt(r_px)} {_fmt(r_px)} 0 0 {sweep} "
        f"{_fmt(p4[0])},{_fmt(p4[1])} Z"
    )
    frag = (
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-dasharray="4 3"/>\n'
    )
    # Inbound-leg arrowhead pointing at the fix (P1). Place it 1/3 back
    # along the inbound leg from P1 so it reads as "flight direction
    # toward the fix", and draw a small filled triangle.
    ah_base = (p1[0] - fwd[0] * 8.0, p1[1] - fwd[1] * 8.0)
    ah_left = (ah_base[0] + right_perp[0] * 3.0, ah_base[1] + right_perp[1] * 3.0)
    ah_right = (ah_base[0] - right_perp[0] * 3.0, ah_base[1] - right_perp[1] * 3.0)
    frag += (
        f'<path d="M{_fmt(p1[0])},{_fmt(p1[1])} '
        f'L{_fmt(ah_left[0])},{_fmt(ah_left[1])} '
        f'L{_fmt(ah_right[0])},{_fmt(ah_right[1])} Z" '
        f'fill="{color}" stroke="none"/>\n'
    )
    return frag


# --- fix symbols ------------------------------------------------------------


# Storage for the most recent plan-view placement trace so tests can
# introspect where labels landed without re-rendering.
_LAST_PLAN_TRACE: list[Placed] | None = None


def _plan_view_placement_trace(procedure: dict) -> list[Placed] | None:
    """Render ``procedure`` and return the most recent plan-view trace.

    Used by tests to verify the placement engine produced no overlapping
    labels. Re-runs the full ``render_iap_svg`` pipeline (cheap relative
    to the IO needed to write the SVG elsewhere) to guarantee the trace
    reflects exactly what would be drawn.
    """
    global _LAST_PLAN_TRACE
    _LAST_PLAN_TRACE = None
    # Lazy import to match render_iap_svg's own dependency graph.
    render_iap_svg(procedure)
    return _LAST_PLAN_TRACE


def _render_fix_symbols(
    procedure: dict,
    fixes: dict[str, LatLon],
    primitives: list[Primitive],
    projector: Projector,
    *,
    engine: PlacementEngine | None = None,
    plan_rect: PBBox | None = None,
) -> str:
    """Emit fix symbols and route fix-id / role / altitude-box labels
    through the placement engine when one is supplied.

    The pre-engine code emitted labels at fixed NE offsets and relied on
    ad-hoc mitigation (a procedure-bbox mask; role-label merging on
    near-coincident fixes). With an engine the symbols still render at
    the projected fix position; every adjacent label goes through
    candidate iteration so two fixes whose projections land close
    together no longer produce overlapping IDs.

    When ``engine`` is ``None`` the function falls back to the legacy
    fixed-offset emission so callers outside the plan view keep working
    (no current external caller relies on that path).
    """
    used = _collect_used_fix_ids(procedure)
    roles = _fix_roles(procedure)
    role_plan = _plan_role_labels(used, roles, fixes, projector)
    leg_dirs = _fix_leg_directions(procedure, fixes, projector)
    mono_inline = FONT_MONO.replace('"', "'")

    # --- Legacy path (no engine) ----------------------------------------
    if engine is None:
        frag = ""
        for fid in sorted(used):
            fix = _find_fix(procedure, fid)
            if fix is None or fid not in fixes:
                continue
            x, y = projector(fixes[fid])
            frag += _fix_symbol(fix, x, y)
            frag += _emit_fix_id_label(x, y, fid, 8.0, 10.0, mono_inline)
            role_text = role_plan.get(fid)
            if role_text:
                frag += _emit_role_label(x, y, role_text, 8.0, 21.0, mono_inline)
            con = _find_altitude_constraint(procedure, fid)
            if con:
                bx, by = _altitude_box_anchor(x, y, leg_dirs.get(fid))
                frag += _altitude_box(bx, by, con)
        frag += _render_course_labels(procedure, fixes, projector)
        return frag

    # --- Engine path ----------------------------------------------------
    # Emit fix symbols directly (their position is non-negotiable) and
    # register a fixed bbox for each symbol so labels route around them.
    # Label requests are collected here; placement + SVG emission happens
    # in ``_finalise_plan_labels`` after every label site has submitted
    # its requests.
    frag = ""
    ctx = _PlanLabelCtx.current()
    assert ctx is not None

    for fid in sorted(used):
        fix = _find_fix(procedure, fid)
        if fix is None or fid not in fixes:
            continue
        x, y = projector(fixes[fid])
        frag += _fix_symbol(fix, x, y)
        # Fixed-bbox reservation for the symbol itself. Tight (~10 px
        # radius) so the 8-position label candidates can sit just past
        # the symbol edge; larger navaid types still fit because the
        # fix-id font sits a few pixels beyond the symbol in every
        # direction.
        ctx.add_fixed(
            f"fixsym:{fid}",
            PlacementTier.FIX,
            PBBox(x - 5.0, y - 5.0, 10.0, 10.0),
        )

        # Fix-id label request. Offset just past the fix symbol so the
        # minimal ~1 px clearance doesn't force the label into the symbol
        # reservation. ``displace`` fallback tries doubled offsets
        # before giving up to a leader; on tightly-packed plates
        # (UGSB BT13F/BTM/RW13) this is the difference between "label
        # visible next to its fix at 20 px" and "label yanked to the
        # plan margin".
        ctx.add_label_request(
            feature_id=f"fixid:{fid}",
            anchor=(x, y),
            content_size=(len(fid) * SIZE_STD * 0.62, SIZE_STD * 1.25),
            candidates=eight_position_candidates(preferred="NE", offset_px=10.0),
            fallback="displace",
            clearance=1.0,
            render=_fix_id_renderer(fid, mono_inline),
        )

        role_text = role_plan.get(fid)
        if role_text:
            role_str = f"({role_text})"
            ctx.add_label_request(
                feature_id=f"role:{fid}",
                anchor=(x, y),
                content_size=(len(role_str) * SIZE_BODY * 0.65, SIZE_BODY * 1.25),
                # Role sits below / beside the fix id by preference — SE
                # first, then rotate clockwise.
                candidates=eight_position_candidates(preferred="SE", offset_px=14.0),
                fallback="displace",
                clearance=1.0,
                render=_role_renderer(role_str, mono_inline),
            )

        con = _find_altitude_constraint(procedure, fid)
        if con and con.get("altitude_1_ft") is not None:
            # Altitude box is a 34x12 rectangle. Its preferred position
            # is on the perpendicular-left side of the inbound leg (the
            # legacy rule); fallbacks are the other three compass points.
            # Candidates below place the BOX CENTER ``offset_px`` from
            # the anchor in the chosen direction — NOT the top-left, so
            # a label drawn at E offset sits clearly east of the fix,
            # leaving room for fixid (NE) and role (SE) to place inside
            # the ring of 8-position candidates closer to the anchor.
            preferred = _altitude_preferred_side(leg_dirs.get(fid))
            ctx.add_label_request(
                feature_id=f"alt:{fid}",
                anchor=(x, y),
                content_size=(34.0, 12.0),
                candidates=_altitude_candidates(preferred, offset_px=22.0,
                                                box_w=34.0, box_h=12.0),
                fallback="displace",
                clearance=1.0,
                render=_altitude_box_renderer(con),
            )

    # Course labels + DME ticks add their requests directly.
    frag += _collect_course_label_requests(procedure, fixes, projector, ctx)
    # DME ticks are collected here so all plan-view labels go through the
    # same engine; the tick geometry (short perpendicular lines) is still
    # emitted immediately.
    frag += _collect_dme_tick_requests(procedure, fixes, projector, ctx)
    return frag


def _emit_fix_id_label(
    x: float, y: float, fid: str,
    offset_x: float, offset_y: float, mono_inline: str,
) -> str:
    label_common = (
        f'x="{_fmt(x + offset_x)}" y="{_fmt(y + offset_y)}" text-anchor="start" '
        f'style="font-size:{_fmt(SIZE_STD)}px;font-weight:bold;'
        f'font-family:{mono_inline}"'
    )
    return (
        f'<text {label_common} fill="none" stroke="{COLOR_PAPER}" '
        f'stroke-width="2.5" stroke-linejoin="round">{_esc(fid)}</text>\n'
        f'<text {label_common} fill="{COLOR_INK}">{_esc(fid)}</text>\n'
    )


def _emit_role_label(
    x: float, y: float, role_text: str,
    offset_x: float, offset_y: float, mono_inline: str,
) -> str:
    role_common = (
        f'x="{_fmt(x + offset_x)}" y="{_fmt(y + offset_y)}" text-anchor="start" '
        f'style="font-size:{_fmt(SIZE_BODY)}px;font-weight:bold;'
        f'font-family:{mono_inline};letter-spacing:{TRACK}"'
    )
    return (
        f'<text {role_common} fill="none" stroke="{COLOR_PAPER}" '
        f'stroke-width="1.8" stroke-linejoin="round">'
        f'({_esc(role_text)})</text>\n'
        f'<text {role_common} fill="{COLOR_ACCENT_MAGENTA}">'
        f'({_esc(role_text)})</text>\n'
    )


_ALT_COMPASS: dict[str, tuple[float, float]] = {
    "N": (0.0, -1.0),
    "E": (1.0, 0.0),
    "S": (0.0, 1.0),
    "W": (-1.0, 0.0),
}


def _altitude_candidates(
    preferred: str, offset_px: float, box_w: float, box_h: float,
) -> tuple[LabelCandidate, ...]:
    """Four-position candidates that anchor the BOX CENTER at
    ``offset_px`` from the anchor along each compass direction.

    ``offset_px`` is measured from the anchor to the centre of the box,
    not the box's top-left — so an altitude box at E/22 sits with its
    left edge at ``anchor.x + 22 - 17 = anchor.x + 5`` and the fix-id
    label (on NE at 10 px) still has room at ``anchor.x + 7 ... + 37``
    without overlap (overlap only in the 5-7 px gap which is within
    the ~1 px clearance).
    """
    order = ["N", "E", "S", "W"]
    if preferred in order:
        i = order.index(preferred)
        order = order[i:] + order[:i]
    out: list[LabelCandidate] = []
    for rank, d in enumerate(order):
        ux, uy = _ALT_COMPASS[d]
        cx = ux * offset_px
        cy = uy * offset_px
        # offset is top-left - anchor = (cx - box_w/2, cy - box_h/2)
        out.append(LabelCandidate(
            offset_x=cx - box_w / 2.0,
            offset_y=cy - box_h / 2.0,
            weight=rank,
        ))
    return tuple(out)


def _altitude_preferred_side(
    leg_dir: tuple[float, float] | None,
) -> str:
    """Pick the four-position candidate preferred side for an altitude box.

    Legacy code placed the box on the leg's left-hand perpendicular via
    custom anchor math. In the engine model we collapse that to a
    simple N/E/S/W preference — whichever quadrant the left-hand
    perpendicular lands in.
    """
    if leg_dir is None:
        return "N"
    dx, dy = leg_dir
    # Left-hand perpendicular (CCW 90° in screen space, y-down).
    perp_x, perp_y = -dy, dx
    # Map to the dominant compass axis.
    if abs(perp_x) >= abs(perp_y):
        return "E" if perp_x > 0 else "W"
    return "S" if perp_y > 0 else "N"


# --- Per-label render functions -------------------------------------------


def _fix_id_renderer(fid: str, mono_inline: str):
    def _render(placed: Placed) -> str:
        # Anchor is the fix point; offset_used is the chosen candidate
        # offset; the bbox top-left = anchor + offset. Baseline y = bbox
        # top + ~9 px so the 9.5 px glyphs sit correctly.
        bx = placed.bbox.x
        by = placed.bbox.y + 9.0
        frag = ""
        if placed.leader_line is not None:
            (ax, ay), (lx, ly) = placed.leader_line
            frag += (
                f'<line x1="{_fmt(ax)}" y1="{_fmt(ay)}" '
                f'x2="{_fmt(lx)}" y2="{_fmt(ly)}" '
                f'stroke="{COLOR_INK}" stroke-width="0.5"/>\n'
            )
        label_common = (
            f'x="{_fmt(bx)}" y="{_fmt(by)}" text-anchor="start" '
            f'style="font-size:{_fmt(SIZE_STD)}px;font-weight:bold;'
            f'font-family:{mono_inline}"'
        )
        frag += (
            f'<text {label_common} fill="none" stroke="{COLOR_PAPER}" '
            f'stroke-width="2.5" stroke-linejoin="round">{_esc(fid)}</text>\n'
            f'<text {label_common} fill="{COLOR_INK}">{_esc(fid)}</text>\n'
        )
        return frag
    return _render


def _role_renderer(role_str: str, mono_inline: str):
    def _render(placed: Placed) -> str:
        bx = placed.bbox.x
        by = placed.bbox.y + 8.0
        frag = ""
        if placed.leader_line is not None:
            (ax, ay), (lx, ly) = placed.leader_line
            frag += (
                f'<line x1="{_fmt(ax)}" y1="{_fmt(ay)}" '
                f'x2="{_fmt(lx)}" y2="{_fmt(ly)}" '
                f'stroke="{COLOR_ACCENT_MAGENTA}" stroke-width="0.5"/>\n'
            )
        role_common = (
            f'x="{_fmt(bx)}" y="{_fmt(by)}" text-anchor="start" '
            f'style="font-size:{_fmt(SIZE_BODY)}px;font-weight:bold;'
            f'font-family:{mono_inline};letter-spacing:{TRACK}"'
        )
        frag += (
            f'<text {role_common} fill="none" stroke="{COLOR_PAPER}" '
            f'stroke-width="1.8" stroke-linejoin="round">'
            f'{_esc(role_str)}</text>\n'
            f'<text {role_common} fill="{COLOR_ACCENT_MAGENTA}">'
            f'{_esc(role_str)}</text>\n'
        )
        return frag
    return _render


def _altitude_box_renderer(con: dict):
    def _render(placed: Placed) -> str:
        # Recreate the legacy (bx, by) anchor the box used: by = rect_top
        # + rect_h - 2 in the old math. For the engine, bbox.y IS the
        # rect top; translate back.
        bx = placed.bbox.x
        by = placed.bbox.y + placed.bbox.h - 2.0
        frag = ""
        if placed.leader_line is not None:
            (ax, ay), (lx, ly) = placed.leader_line
            frag += (
                f'<line x1="{_fmt(ax)}" y1="{_fmt(ay)}" '
                f'x2="{_fmt(lx)}" y2="{_fmt(ly)}" '
                f'stroke="{COLOR_INK}" stroke-width="0.5"/>\n'
            )
        frag += _altitude_box(bx, by, con)
        return frag
    return _render


def _course_label_renderer(course: float, angle_deg: float, mono_inline: str):
    """Build a renderer for a course label (bold numeric + degree)."""
    txt = f"{int(round(course)):03d}\u00B0"

    def _render(placed: Placed) -> str:
        # Centre the text inside the bbox.
        cx = placed.bbox.x + placed.bbox.w / 2.0
        cy = placed.bbox.y + placed.bbox.h / 2.0 + 3.0  # baseline nudge
        return (
            f'<text x="{_fmt(cx)}" y="{_fmt(cy)}" text-anchor="middle" '
            f'fill="{COLOR_INK}" style="font-size:{_fmt(SIZE_BODY)}px;font-weight:bold;'
            f'font-family:{mono_inline}" '
            f'transform="rotate({_fmt(angle_deg)} {_fmt(cx)} {_fmt(cy)})">'
            f"{_esc(txt)}</text>\n"
        )
    return _render


def _collect_course_label_requests(
    procedure: dict,
    fixes: dict[str, LatLon],
    projector: Projector,
    ctx: _PlanLabelCtx,
) -> str:
    """Submit course-label placement requests for every leg.

    The label anchor is the leg midpoint offset perpendicular-left of
    the leg by ~9 px (legacy offset). Four-position candidate set — N/
    E/S/W is effectively rotated so 'above the leg' is always first.
    Returns empty string (labels emit in the finalise step).
    """
    seen: set[tuple[str, str]] = set()
    mono_inline = FONT_MONO.replace('"', "'")

    def _emit(from_id: str | None, to_id: str | None, course):
        if not (from_id and to_id and course is not None):
            return
        key = (from_id, to_id)
        if key in seen or from_id not in fixes or to_id not in fixes:
            return
        seen.add(key)
        a = projector(fixes[from_id])
        b = projector(fixes[to_id])
        mx = (a[0] + b[0]) / 2.0
        my = (a[1] + b[1]) / 2.0
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        # Left-hand perpendicular offset of the anchor (matches legacy
        # default position, just expressed as an anchor + zero-offset
        # candidate so the engine can nudge it perpendicular-right when
        # the preferred side is blocked).
        ox = -dy / length * 9.0
        oy = dx / length * 9.0
        anchor = (mx + ox, my + oy)
        angle = degrees(atan2(dy, dx))
        if angle > 90 or angle < -90:
            angle += 180
        txt = f"{int(round(course)):03d}\u00B0"
        w = len(txt) * SIZE_BODY * 0.62
        h = SIZE_BODY * 1.25
        # 4-position offset candidates — keep the label hugging the leg
        # so the engine can swap to the opposite side if the preferred
        # one collides.
        candidates = (
            LabelCandidate(-w / 2.0, -h / 2.0, weight=0),  # at anchor (preferred)
            LabelCandidate(-w / 2.0 - 2 * ox, -h / 2.0 - 2 * oy, weight=1),  # flip side
        )
        ctx.add_label_request(
            feature_id=f"course:{from_id}->{to_id}",
            anchor=anchor,
            content_size=(w, h),
            candidates=candidates,
            fallback="displace",
            clearance=1.0,
            render=_course_label_renderer(float(course), angle, mono_inline),
        )

    for t in procedure.get("transitions") or []:
        for leg in t.get("legs") or []:
            _emit(leg.get("from_fix_id"), leg.get("to_fix_id"),
                  leg.get("course_deg"))
    for leg in procedure.get("common_legs") or []:
        _emit(leg.get("from_fix_id"), leg.get("to_fix_id"),
              leg.get("course_deg"))
    return ""


def _dme_label_renderer(label: str, mono_inline: str):
    def _render(placed: Placed) -> str:
        # Centre in bbox; emit with paper halo (ticks sit on ink).
        cx = placed.bbox.x + placed.bbox.w / 2.0
        by = placed.bbox.y + placed.bbox.h - 1.0
        label_common = (
            f'x="{_fmt(cx)}" y="{_fmt(by)}" text-anchor="middle" '
            f'style="font-size:{_fmt(SIZE_MICRO)}px;font-weight:bold;'
            f'font-family:{mono_inline};letter-spacing:{TRACK}"'
        )
        return (
            f'<text {label_common} fill="none" stroke="{COLOR_PAPER}" '
            f'stroke-width="1.8" stroke-linejoin="round">{_esc(label)}</text>\n'
            f'<text {label_common} fill="{COLOR_INK}">{_esc(label)}</text>\n'
        )
    return _render


def _collect_dme_tick_requests(
    procedure: dict,
    fixes: dict[str, LatLon],
    projector: Projector,
    ctx: _PlanLabelCtx,
) -> str:
    """Draw DME tick geometry and submit label placement requests.

    Tick marks themselves render immediately; their integer labels go
    through the engine so they can nudge away from course labels, fix
    IDs, and altitude boxes that sit along the same segment.
    """
    faf_id, map_id = _resolve_faf_map(procedure)
    if not (faf_id and map_id):
        return ""
    if faf_id not in fixes or map_id not in fixes:
        return ""
    navaid_id = _pick_primary_dme_navaid(procedure, fixes)
    if not navaid_id or navaid_id not in fixes:
        return ""

    faf_ll = fixes[faf_id]
    map_ll = fixes[map_id]
    nav_ll = fixes[navaid_id]

    faf_dme = great_circle_distance_nm(nav_ll, faf_ll)
    map_dme = great_circle_distance_nm(nav_ll, map_ll)
    lo_dme = min(faf_dme, map_dme)
    hi_dme = max(faf_dme, map_dme)
    first_int = int(lo_dme) + 1
    last_int = int(hi_dme)
    if last_int < first_int:
        return ""

    brief = procedure.get("briefing_strip") or {}
    course_deg = brief.get("final_approach_course_deg")
    if course_deg is None:
        return ""
    recip = (float(course_deg) + 180.0) % 360.0
    faf_to_map_nm = great_circle_distance_nm(faf_ll, map_ll)

    step_nm = 0.1
    nsamples = max(2, int(faf_to_map_nm / step_nm) + 1)
    samples: list[tuple[float, float, float]] = []
    for i in range(nsamples + 1):
        along_from_map = i * step_nm
        pt = destination(map_ll, recip, along_from_map)
        dme = great_circle_distance_nm(nav_ll, pt)
        samples.append((along_from_map, dme, 0.0))

    tick_points: list[tuple[float, int]] = []
    for target in range(first_int, last_int + 1):
        for k in range(len(samples) - 1):
            d0 = samples[k][1]
            d1 = samples[k + 1][1]
            if (d0 - target) * (d1 - target) > 0:
                continue
            if d1 == d0:
                t = 0.0
            else:
                t = (target - d0) / (d1 - d0)
            along = samples[k][0] + t * (samples[k + 1][0] - samples[k][0])
            tick_points.append((along, target))
            break

    if not tick_points:
        return ""

    map_px = projector(map_ll)
    faf_px = projector(faf_ll)
    dx = faf_px[0] - map_px[0]
    dy = faf_px[1] - map_px[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return ""
    ux, uy = dx / length, dy / length
    perp_x, perp_y = -uy, ux

    TICK_HALF_PX = 3.0
    mono_inline = FONT_MONO.replace('"', "'")
    frag = '<g class="dme-ticks">\n'
    for along_from_map, dme_int in tick_points:
        pt_ll = destination(map_ll, recip, along_from_map)
        px, py = projector(pt_ll)
        x0 = px - perp_x * TICK_HALF_PX
        y0 = py - perp_y * TICK_HALF_PX
        x1 = px + perp_x * TICK_HALF_PX
        y1 = py + perp_y * TICK_HALF_PX
        frag += (
            f'<line x1="{_fmt(x0)}" y1="{_fmt(y0)}" '
            f'x2="{_fmt(x1)}" y2="{_fmt(y1)}" '
            f'stroke="{COLOR_INK}" stroke-width="0.75" '
            f'stroke-linecap="round"/>\n'
        )
        label = str(dme_int)
        # Submit the label as a placement request anchored at the outer
        # end of the tick. Four candidates: on either side of the tick,
        # perpendicular to the course.
        anchor = (
            px + perp_x * (TICK_HALF_PX + 4.0),
            py + perp_y * (TICK_HALF_PX + 4.0),
        )
        w = len(label) * SIZE_MICRO * 0.62
        h = SIZE_MICRO * 1.25
        candidates = (
            LabelCandidate(-w / 2.0, -h / 2.0, weight=0),
            LabelCandidate(
                -w / 2.0 - 2 * perp_x * (TICK_HALF_PX + 4.0),
                -h / 2.0 - 2 * perp_y * (TICK_HALF_PX + 4.0),
                weight=1,
            ),
        )
        ctx.add_label_request(
            feature_id=f"dme:{dme_int:03d}",
            anchor=anchor,
            content_size=(w, h),
            candidates=candidates,
            fallback="displace",
            clearance=1.0,
            render=_dme_label_renderer(label, mono_inline),
        )
    frag += '</g>\n'
    return frag


# ---------------------------------------------------------------------------
# Plan-view placement context — owns the engine + per-label render fns.
# ---------------------------------------------------------------------------


class _PlanLabelCtx:
    """Per-render placement collector for the plan view.

    Instance-local because multiple procedures might be rendered in the
    same process. The renderer stashes the active instance in a module
    slot so deeply-nested emitters (``_render_fix_symbols``,
    ``_collect_course_label_requests``, basemap label emitters) can find
    it without threading the engine through every signature.
    """

    _active: "_PlanLabelCtx | None" = None

    def __init__(self, plan_rect: PBBox):
        self.engine = PlacementEngine(plan_rect=plan_rect)
        self.requests: list[PlacementRequest] = []
        self.render_fns: list[object] = []

    @classmethod
    def current(cls) -> "_PlanLabelCtx | None":
        return cls._active

    def __enter__(self) -> "_PlanLabelCtx":
        type(self)._active = self
        return self

    def __exit__(self, *exc) -> None:
        type(self)._active = None

    def add_fixed(
        self, feature_id: str, tier: PlacementTier, bbox: PBBox,
    ) -> None:
        """Register a non-label bbox reservation."""
        self.requests.append(
            PlacementRequest(
                tier=tier,
                anchor=(bbox.x, bbox.y),
                content_size=(bbox.w, bbox.h),
                candidates=(),
                fallback="suppress",
                feature_id=feature_id,
                fixed_bbox=bbox,
            )
        )
        # Parallel None entry — fixed reservations emit nothing themselves.
        self.render_fns.append(None)

    def add_label_request(
        self,
        *,
        feature_id: str,
        anchor: tuple[float, float],
        content_size: tuple[float, float],
        candidates: tuple[LabelCandidate, ...],
        fallback: str,
        clearance: float,
        render,
        tier: PlacementTier = PlacementTier.LABEL,
    ) -> None:
        self.requests.append(
            PlacementRequest(
                tier=tier,
                anchor=anchor,
                content_size=content_size,
                candidates=candidates,
                fallback=fallback,  # type: ignore[arg-type]
                feature_id=feature_id,
                required_clearance_px=clearance,
            )
        )
        self.render_fns.append(render)

    def resolve_and_emit(self) -> str:
        """Run the engine and emit SVG for every label request.

        Fixed-bbox reservations emit nothing here (they were already
        drawn by their geometry layer). Stores the full Placed list on
        the module-level trace for tests to introspect.
        """
        global _LAST_PLAN_TRACE
        placed_list = self.engine.place(self.requests)
        _LAST_PLAN_TRACE = placed_list
        parts: list[str] = []
        for placed, render in zip(placed_list, self.render_fns):
            if render is None:
                continue  # fixed bbox reservation
            if placed.bbox.w == 0 and placed.bbox.h == 0:
                continue  # suppressed
            parts.append(render(placed))
        return "".join(parts)


def _plan_role_labels(
    used: set[str],
    roles: dict[str, str],
    fixes: dict[str, LatLon],
    projector: Projector,
) -> dict[str, str]:
    """Merge role labels for fixes that project within ~5 px of each
    other so they render as a single ``(MAP/MAHP)``-style combined tag
    on the primary fix instead of overlapping.

    The primary (kept) fix is the one sort-first by id — matching the
    iteration order in :func:`_render_fix_symbols`. Secondary fixes
    covered by a merge drop their own role label.
    """
    # Fix projections within this many pixels get their role labels
    # merged into one stacked tag on the primary (alphabetically-first)
    # fix. Bumped from 5 px -> 12 px because fixes like BT13F / BTM /
    # RW13 at UGSB project ~10 px apart; at 5 px they stay separate and
    # their role labels overflow each other. 12 px still keeps KOBUL
    # (the IF, ~16 px from the threshold cluster) out of the merge.
    THRESHOLD_PX = 12.0
    out: dict[str, str] = {}
    # Project only the fixes that have a role.
    pts: dict[str, tuple[float, float]] = {}
    for fid in used:
        if fid in roles and fid in fixes:
            pts[fid] = projector(fixes[fid])

    consumed: set[str] = set()
    for fid in sorted(pts.keys()):
        if fid in consumed:
            continue
        role_parts = [roles[fid]]
        fx, fy = pts[fid]
        for other in sorted(pts.keys()):
            if other == fid or other in consumed:
                continue
            ox, oy = pts[other]
            dx = ox - fx
            dy = oy - fy
            if (dx * dx + dy * dy) ** 0.5 <= THRESHOLD_PX:
                role_parts.append(roles[other])
                consumed.add(other)
        out[fid] = "/".join(role_parts)
    return out


def _fix_leg_directions(
    procedure: dict,
    fixes: dict[str, LatLon],
    projector: Projector,
) -> dict[str, tuple[float, float]]:
    """Unit vector in plan-view pixel space pointing **into** each fix
    along the leg that terminates there.

    Used to place the altitude box on the opposite-perpendicular side of
    the leg, so the box never straddles the magenta procedure track.
    The last wins when a fix is the to_fix of multiple legs (shouldn't
    normally happen on a single-branch IAP).
    """
    out: dict[str, tuple[float, float]] = {}

    def _record(from_id: str | None, to_id: str | None) -> None:
        if not (from_id and to_id):
            return
        if from_id not in fixes or to_id not in fixes:
            return
        a = projector(fixes[from_id])
        b = projector(fixes[to_id])
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            return
        out[to_id] = (dx / length, dy / length)

    for t in procedure.get("transitions") or []:
        for leg in t.get("legs") or []:
            _record(leg.get("from_fix_id"), leg.get("to_fix_id"))
    for leg in procedure.get("common_legs") or []:
        _record(leg.get("from_fix_id"), leg.get("to_fix_id"))
    return out


def _altitude_box_anchor(
    fx: float, fy: float, leg_dir: tuple[float, float] | None,
) -> tuple[float, float]:
    """Pick an (x, y) anchor for :func:`_altitude_box` that sits roughly
    ``OFFSET`` px perpendicular to the approach track at the fix, on the
    side of the track that doesn't also carry the fix-id label (which
    hangs off the SE of the fix symbol).

    The legacy anchor ``(fx + 8, fy - 10)`` put the box NE of the fix,
    which landed on top of the procedure track when the leg approached
    from the SE (ETAD: EIFEL / ED23F / SPANG all showed this). Here the
    box is offset along the leg's **left-hand** perpendicular (CCW-90°
    from the inbound direction) so it sits clear of the track. Falls
    back to the legacy NE placement when no leg direction is available
    (e.g. IF fix at procedure entry).
    """
    if leg_dir is None:
        return (fx + 8, fy - 10)
    dx, dy = leg_dir
    # Perpendicular rotated 90° CCW in screen space (y grows down).
    perp_x = -dy
    perp_y = dx
    OFFSET = 10.0
    # Altitude box is 34 × 12 drawn with (x, y) describing the bottom-
    # right-ish corner the legacy code used. Nudge the anchor so the box
    # doesn't drift off the opposite side of the fix when perp points
    # left/up.
    ax = fx + perp_x * OFFSET
    ay = fy + perp_y * OFFSET
    # Match the legacy box orientation: the box extends to the right of
    # ``ax`` and upward from ``ay``. If the perpendicular pushes the box
    # into the upper-left quadrant relative to the fix, shift x so the
    # box stays on the fix's left side (rather than floating off-track).
    if perp_x < 0:
        ax -= 34.0  # box width — flip to the other side of the anchor
    if perp_y > 0:
        ay += 12.0  # box height — flip below the anchor so it sits south
    return (ax, ay)


def _fix_roles(procedure: dict) -> dict[str, str]:
    """Map fix_id -> role label for IAP / IF / FAF / MAP / MAHP annotations."""
    roles: dict[str, str] = {}

    # IAP: to_fix_id of the first IF leg in an approach_transition.
    for t in procedure.get("transitions") or []:
        if t.get("type") != "approach_transition":
            continue
        for leg in t.get("legs") or []:
            if leg.get("terminator") == "IF" and leg.get("to_fix_id"):
                roles.setdefault(leg["to_fix_id"], "IAF")
                break
        break

    # IF (intermediate fix): the first IF leg inside common_legs.
    for leg in procedure.get("common_legs") or []:
        if leg.get("terminator") == "IF" and leg.get("to_fix_id"):
            roles.setdefault(leg["to_fix_id"], "IF")
            break

    # FAF / MAP from the existing resolver.
    faf_id, map_id = _resolve_faf_map(procedure)
    if faf_id:
        roles[faf_id] = "FAF"
    if map_id:
        roles[map_id] = "MAP"

    # MAHP: the to_fix_id of a holding-pattern missed-approach leg (HM/HA/HF).
    for leg in (procedure.get("missed_approach") or {}).get("legs") or []:
        if leg.get("terminator") in ("HM", "HA", "HF") and leg.get("to_fix_id"):
            roles.setdefault(leg["to_fix_id"], "MAHP")
            break

    return roles


def _collect_used_fix_ids(procedure: dict) -> set[str]:
    ids: set[str] = set()
    for t in procedure.get("transitions") or []:
        for leg in t.get("legs") or []:
            for key in ("from_fix_id", "to_fix_id"):
                v = leg.get(key)
                if v:
                    ids.add(v)
    for leg in procedure.get("common_legs") or []:
        for key in ("from_fix_id", "to_fix_id"):
            v = leg.get(key)
            if v:
                ids.add(v)
    missed = (procedure.get("missed_approach") or {}).get("legs") or []
    for leg in missed:
        for key in ("from_fix_id", "to_fix_id"):
            v = leg.get(key)
            if v:
                ids.add(v)
    msa = procedure.get("msa") or {}
    if msa.get("center_navaid_id"):
        ids.add(msa["center_navaid_id"])
    return ids


def _find_fix(procedure: dict, fix_id: str) -> dict | None:
    for f in procedure.get("fixes") or []:
        if f.get("id") == fix_id:
            return f
    return None


def _fix_symbol(fix: dict, x: float, y: float) -> str:
    ftype = fix.get("type")
    if ftype == "navaid":
        sub = fix.get("navaid_type")
        # Point-up hexagon for VOR family. Vertex positions (r=7 px):
        # 0=top, 1=upper-right, 2=lower-right, 3=bottom, 4=lower-left, 5=upper-left
        if sub in ("vor", "vor_dme", "vortac", "tacan"):
            r = 7.0
            hex_path = _regular_polygon(x, y, r, 6, rotation_deg=0)
            # VOR hexagons render OUTLINE-ONLY (transparent fill) per
            # Jepp/FAA convention — when the navaid is colocated with
            # the airport reference, the runway beneath must remain
            # visible through the symbol. TACAN alone (rare) stays ink.
            fill = COLOR_INK if sub == "tacan" else "none"
            frag = (
                f'<path d="{hex_path}" fill="{fill}" '
                f'stroke="{COLOR_INK}" stroke-width="1.2"/>\n'
            )
            # Center dot — VOR family only; TACAN stays solid ink.
            if sub in ("vor", "vor_dme", "vortac"):
                frag += (
                    f'<circle cx="{_fmt(x)}" cy="{_fmt(y)}" r="1.3" '
                    f'fill="{COLOR_INK}"/>\n'
                )
            # VOR-DME: surrounding square.
            if sub == "vor_dme":
                s = r + 2.5
                frag += (
                    f'<rect x="{_fmt(x - s)}" y="{_fmt(y - s)}" '
                    f'width="{_fmt(2 * s)}" height="{_fmt(2 * s)}" '
                    f'fill="none" stroke="{COLOR_INK}" stroke-width="1"/>\n'
                )
            # VORTAC / TACAN: three small filled "ears" (triangular lobes) at
            # alternating vertices (top, lower-right, lower-left) — the
            # visual cue that distinguishes military TACAN capability.
            if sub in ("vortac", "tacan"):
                ear_r = 3.0
                for i in (0, 2, 4):  # alternating vertices
                    a = radians(0 + 60.0 * i - 90.0)
                    vx = x + r * cos(a)
                    vy = y + r * sin(a)
                    # Triangle tangent to the hexagon side, pointing outward.
                    nx, ny = cos(a), sin(a)  # outward normal
                    # Two base points on the adjacent hexagon edges.
                    a1 = radians(0 + 60.0 * (i - 1) - 90.0)
                    a2 = radians(0 + 60.0 * (i + 1) - 90.0)
                    bx1, by1 = x + r * cos(a1), y + r * sin(a1)
                    bx2, by2 = x + r * cos(a2), y + r * sin(a2)
                    tx = vx + nx * ear_r
                    ty = vy + ny * ear_r
                    # Midpoints of the two adjacent edges → form small ears.
                    mx1 = (vx + bx1) / 2.0
                    my1 = (vy + by1) / 2.0
                    mx2 = (vx + bx2) / 2.0
                    my2 = (vy + by2) / 2.0
                    frag += (
                        f'<path d="M{_fmt(mx1)},{_fmt(my1)} '
                        f'L{_fmt(tx)},{_fmt(ty)} '
                        f'L{_fmt(mx2)},{_fmt(my2)} Z" '
                        f'fill="{COLOR_INK}" stroke="none"/>\n'
                    )
            return frag
        if sub == "loc":
            return (
                f'<path d="M{_fmt(x - 4)},{_fmt(y - 8)} L{_fmt(x + 4)},{_fmt(y)} '
                f'L{_fmt(x - 4)},{_fmt(y + 8)} Z" fill="{COLOR_PAPER}" '
                f'stroke="{COLOR_INK}" stroke-width="1"/>\n'
            )
        if sub == "ndb":
            return (
                f'<circle cx="{_fmt(x)}" cy="{_fmt(y)}" r="5" fill="none" '
                f'stroke="{COLOR_INK}" stroke-width="1" stroke-dasharray="1 1.5"/>\n'
            )
        if sub == "dme":
            # DME-only: plain square (no internal hexagon).
            return (
                f'<rect x="{_fmt(x - 5)}" y="{_fmt(y - 5)}" width="10" height="10" '
                f'fill="{COLOR_PAPER}" stroke="{COLOR_INK}" stroke-width="1"/>\n'
            )
        return (
            f'<rect x="{_fmt(x - 4)}" y="{_fmt(y - 4)}" width="8" height="8" '
            f'fill="{COLOR_PAPER}" stroke="{COLOR_INK}" stroke-width="1"/>\n'
        )
    if ftype == "runway_threshold":
        return (
            f'<rect x="{_fmt(x - 2)}" y="{_fmt(y - 2)}" width="4" height="4" '
            f'fill="{COLOR_INK}"/>\n'
        )
    return (
        f'<path d="M{_fmt(x)},{_fmt(y - 5)} L{_fmt(x + 5)},{_fmt(y + 4)} '
        f'L{_fmt(x - 5)},{_fmt(y + 4)} Z" fill="{COLOR_INK}"/>\n'
    )


def _regular_polygon(cx: float, cy: float, r: float, n: int, rotation_deg: float = 0.0) -> str:
    parts = []
    for i in range(n):
        a = radians(rotation_deg + 360.0 * i / n - 90.0)
        px = cx + r * cos(a)
        py = cy + r * sin(a)
        parts.append(f"{'M' if i == 0 else 'L'}{_fmt(px)},{_fmt(py)}")
    parts.append("Z")
    return " ".join(parts)


# --- altitude constraints ---------------------------------------------------


def _find_altitude_constraint(procedure: dict, to_fix_id: str) -> dict | None:
    for t in procedure.get("transitions") or []:
        for leg in t.get("legs") or []:
            if leg.get("to_fix_id") == to_fix_id:
                con = leg.get("altitude_constraint")
                if con:
                    return con
    for leg in procedure.get("common_legs") or []:
        if leg.get("to_fix_id") == to_fix_id:
            con = leg.get("altitude_constraint")
            if con:
                return con
    return None


def _altitude_box(x: float, y: float, con: dict) -> str:
    desc = con.get("altitude_description") or "at"
    alt = con.get("altitude_1_ft")
    if alt is None:
        return ""
    label = str(alt)
    over = desc in ("at", "at_or_below")
    under = desc in ("at", "at_or_above", "glide_slope_altitude")
    bw = 34.0
    bh = 12.0
    frag = (
        f'<rect x="{_fmt(x)}" y="{_fmt(y - bh + 2)}" width="{_fmt(bw)}" height="{_fmt(bh)}" '
        f'fill="{COLOR_PAPER}" stroke="{COLOR_INK}" stroke-width="0.75"/>\n'
    )
    if over:
        frag += _line(x, y - bh + 2, x + bw, y - bh + 2, sw=1.5)
    if under:
        frag += _line(x, y + 2, x + bw, y + 2, sw=1.5)
    # The rect already fills paper; a plain fill-only text is enough.
    frag += _text(x + bw / 2.0, y - 1, label,
                  size=SIZE_BODY, anchor="middle", weight="bold")
    return frag


# --- course labels ----------------------------------------------------------


def _render_course_labels(
    procedure: dict,
    fixes: dict[str, LatLon],
    projector: Projector,
) -> str:
    frag = ""
    seen: set[tuple[str, str]] = set()

    def _label_leg(from_id: str | None, to_id: str | None, course: float | None) -> str:
        if not (from_id and to_id and course is not None):
            return ""
        key = (from_id, to_id)
        if key in seen or from_id not in fixes or to_id not in fixes:
            return ""
        seen.add(key)
        a = projector(fixes[from_id])
        b = projector(fixes[to_id])
        mx = (a[0] + b[0]) / 2.0
        my = (a[1] + b[1]) / 2.0
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        ox, oy = -dy / length * 9.0, dx / length * 9.0
        angle = degrees(atan2(dy, dx))
        if angle > 90 or angle < -90:
            angle += 180
        mono_inline = FONT_MONO.replace('"', "'")
        return (
            f'<text x="{_fmt(mx + ox)}" y="{_fmt(my + oy)}" text-anchor="middle" '
            f'fill="{COLOR_INK}" style="font-size:{_fmt(SIZE_BODY)}px;font-weight:bold;'
            f'font-family:{mono_inline}" '
            f'transform="rotate({_fmt(angle)} {_fmt(mx + ox)} {_fmt(my + oy)})">'
            f"{int(round(course)):03d}\u00B0</text>\n"
        )

    for t in procedure.get("transitions") or []:
        for leg in t.get("legs") or []:
            frag += _label_leg(leg.get("from_fix_id"), leg.get("to_fix_id"), leg.get("course_deg"))
    for leg in procedure.get("common_legs") or []:
        frag += _label_leg(leg.get("from_fix_id"), leg.get("to_fix_id"), leg.get("course_deg"))
    return frag


# --- DME ticks along the final approach course ------------------------------


def _pick_primary_dme_navaid(
    procedure: dict, fixes: dict[str, LatLon],
) -> str | None:
    """Identify the DME navaid whose range the FAC is tuned to.

    Preference order:
      1. MSA center navaid (chartwise, this is the primary reference).
      2. First VOR/DME or TACAN fix in the fixes array.
      3. First fix with any navaid_type containing 'dme'.
    Returns the fix id or ``None`` if no DME-equipped navaid is defined.
    """
    msa = procedure.get("msa") or {}
    mid = msa.get("center_navaid_id")
    if mid:
        f = _find_fix(procedure, mid)
        if f and f.get("navaid_type") in (
            "vor_dme", "vortac", "tacan", "dme", "ils_dme",
        ):
            return mid
    # Scan procedure fixes.
    for f in procedure.get("fixes") or []:
        if f.get("type") != "navaid":
            continue
        sub = (f.get("navaid_type") or "").lower()
        if sub in ("vor_dme", "vortac", "tacan", "dme"):
            return f.get("id")
    return None


def _render_dme_ticks(
    procedure: dict,
    fixes: dict[str, LatLon],
    projector: Projector,
) -> str:
    """Draw perpendicular DME tick marks on the final approach course
    between the FAF and runway threshold, one per integer DME value.

    Geometry: walk the FAF→MAP great-circle line, sampling many points
    in NM along the reciprocal of the final approach course back from the
    threshold. At each sample compute great-circle distance from the
    primary DME navaid. Emit a tick when that distance crosses an integer
    value that falls strictly between the FAF DME and the MAP DME.

    Ticks: 6 px long, perpendicular to the inbound course, centered on
    the course line; labelled with the integer DME on the outer end.
    """
    faf_id, map_id = _resolve_faf_map(procedure)
    if not (faf_id and map_id):
        return ""
    if faf_id not in fixes or map_id not in fixes:
        return ""
    navaid_id = _pick_primary_dme_navaid(procedure, fixes)
    if not navaid_id or navaid_id not in fixes:
        return ""

    faf_ll = fixes[faf_id]
    map_ll = fixes[map_id]
    nav_ll = fixes[navaid_id]

    faf_dme = great_circle_distance_nm(nav_ll, faf_ll)
    map_dme = great_circle_distance_nm(nav_ll, map_ll)
    lo_dme = min(faf_dme, map_dme)
    hi_dme = max(faf_dme, map_dme)
    # Integer DME values strictly between FAF and MAP (exclude the
    # endpoints so we don't double-mark a fix).
    first_int = int(lo_dme) + 1
    last_int = int(hi_dme)
    if last_int < first_int:
        return ""

    # FAF→MAP reciprocal course — from MAP back toward FAF in pixel
    # space so we can walk by equal NM steps using the destination
    # primitive on the WGS-84 geodesic.
    brief = procedure.get("briefing_strip") or {}
    course_deg = brief.get("final_approach_course_deg")
    if course_deg is None:
        return ""
    recip = (float(course_deg) + 180.0) % 360.0
    faf_to_map_nm = great_circle_distance_nm(faf_ll, map_ll)

    # Sample every 0.1 NM along the final segment; interpolate to find
    # points where DME-from-navaid crosses each integer value.
    step_nm = 0.1
    nsamples = max(2, int(faf_to_map_nm / step_nm) + 1)
    samples: list[tuple[float, float, float]] = []
    # samples: (along_nm_from_MAP, dme_from_navaid, azimuth-unused)
    for i in range(nsamples + 1):
        along_from_map = i * step_nm
        pt = destination(map_ll, recip, along_from_map)
        dme = great_circle_distance_nm(nav_ll, pt)
        samples.append((along_from_map, dme, 0.0))

    # For each target integer DME, find the along-NM that matches.
    tick_points: list[tuple[float, int]] = []  # (along_from_map_nm, dme_int)
    for target in range(first_int, last_int + 1):
        for k in range(len(samples) - 1):
            d0 = samples[k][1]
            d1 = samples[k + 1][1]
            if (d0 - target) * (d1 - target) > 0:
                continue  # same side, no crossing
            if d1 == d0:
                t = 0.0
            else:
                t = (target - d0) / (d1 - d0)
            along = samples[k][0] + t * (samples[k + 1][0] - samples[k][0])
            tick_points.append((along, target))
            break

    if not tick_points:
        return ""

    # Convert the perpendicular direction in pixel space from the
    # projected FAF/MAP points. The projector is a planar affine over
    # the plan-view bbox so a single (dx, dy) gives the inbound
    # direction for all ticks (courses in the final segment are straight).
    map_px = projector(map_ll)
    faf_px = projector(faf_ll)
    dx = faf_px[0] - map_px[0]
    dy = faf_px[1] - map_px[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return ""
    ux, uy = dx / length, dy / length
    perp_x, perp_y = -uy, ux  # 90° CCW in screen space

    TICK_HALF_PX = 3.0
    mono_inline = FONT_MONO.replace('"', "'")
    # Group the ticks under a single <g> tagged for the basemap-mask
    # machinery — ticks read as "ink" reference marks.
    frag = '<g class="dme-ticks">\n'
    for along_from_map, dme_int in tick_points:
        pt_ll = destination(map_ll, recip, along_from_map)
        px, py = projector(pt_ll)
        x0 = px - perp_x * TICK_HALF_PX
        y0 = py - perp_y * TICK_HALF_PX
        x1 = px + perp_x * TICK_HALF_PX
        y1 = py + perp_y * TICK_HALF_PX
        frag += (
            f'<line x1="{_fmt(x0)}" y1="{_fmt(y0)}" '
            f'x2="{_fmt(x1)}" y2="{_fmt(y1)}" '
            f'stroke="{COLOR_INK}" stroke-width="0.75" '
            f'stroke-linecap="round"/>\n'
        )
        # Label on the outer end (right-hand perpendicular). Small halo
        # first so it reads over basemap linework.
        lx = px + perp_x * (TICK_HALF_PX + 4.0)
        ly = py + perp_y * (TICK_HALF_PX + 4.0) + 2.5
        label = str(dme_int)
        label_common = (
            f'x="{_fmt(lx)}" y="{_fmt(ly)}" text-anchor="middle" '
            f'style="font-size:{_fmt(SIZE_MICRO)}px;font-weight:bold;'
            f'font-family:{mono_inline};letter-spacing:{TRACK}"'
        )
        frag += (
            f'<text {label_common} fill="none" stroke="{COLOR_PAPER}" '
            f'stroke-width="1.8" stroke-linejoin="round">{label}</text>\n'
        )
        frag += f'<text {label_common} fill="{COLOR_INK}">{label}</text>\n'
    frag += '</g>\n'
    return frag


# --- scale bar + north arrow ------------------------------------------------


def _render_scale_bar(projector: Projector, fixes: dict[str, LatLon], region: Region) -> str:
    x, y, w, h = region
    if not fixes:
        return ""
    anchor = next(iter(fixes.values()))
    px_per_nm = projector_scale_px_per_nm(projector, anchor)
    bar_nm = 5.0
    bar_px = px_per_nm * bar_nm
    bx = x + 12
    by = y + h - 14
    frag = _line(bx, by, bx + bar_px, by, stroke=COLOR_ACCENT_CYAN, sw=2.0)
    frag += _line(bx, by - 4, bx, by + 4, stroke=COLOR_ACCENT_CYAN, sw=2.0)
    frag += _line(bx + bar_px, by - 4, bx + bar_px, by + 4,
                  stroke=COLOR_ACCENT_CYAN, sw=2.0)
    frag += _text(bx, by - 6, f"{int(bar_nm)} NM",
                  size=SIZE_MICRO, weight="bold",
                  fill=COLOR_ACCENT_CYAN, tracking=TRACK)
    return frag


def _render_north_arrow(procedure: dict, region: Region) -> str:
    x, y, w, h = region
    cx = x + w - 20
    cy = y + h - 30
    frag = _line(cx, cy + 12, cx, cy - 12,
                 stroke=COLOR_ACCENT_CYAN, sw=1.5)
    frag += (
        f'<path d="M{_fmt(cx)},{_fmt(cy - 14)} L{_fmt(cx - 4)},{_fmt(cy - 6)} '
        f'L{_fmt(cx + 4)},{_fmt(cy - 6)} Z" fill="{COLOR_ACCENT_CYAN}"/>\n'
    )
    frag += _text(cx, cy - 16, "N",
                  size=SIZE_BODY, anchor="middle", weight="bold",
                  fill=COLOR_ACCENT_CYAN)
    mv = procedure.get("mag_variation_deg")
    if mv is not None:
        sign = "E" if mv >= 0 else "W"
        frag += _text(
            cx, cy + 22, f"VAR {abs(mv):.0f}\u00B0{sign}",
            size=SIZE_MICRO, anchor="middle",
            fill=COLOR_ACCENT_CYAN, tracking=TRACK,
        )
    return frag


# --- MSA bezel --------------------------------------------------------------


def _render_msa_bezel(procedure: dict, fixes: dict[str, LatLon], region: Region) -> str:
    x, y, w, h = region
    msa = procedure.get("msa") or {}
    sectors = msa.get("sectors") or []
    if not sectors:
        return ""
    cx = x + w / 2.0
    cy = y + h / 2.0
    r = min(w, h) / 2.0 - 10
    # Backdrop circle (paper) so fix labels don't bleed through the ring.
    frag = (
        f'<circle cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(r)}" fill="{COLOR_PAPER}" '
        f'stroke="{COLOR_ACCENT_CYAN}" stroke-width="1.25"/>\n'
    )
    frag += _text(cx, y + 10, "MSA",
                  size=SIZE_MICRO, weight="bold", anchor="middle",
                  fill=COLOR_ACCENT_CYAN, tracking=TRACK)
    center_id = msa.get("center_navaid_id") or ""
    radius = msa.get("radius_nm")
    sub = f"{center_id} {int(radius)}NM" if center_id and radius else center_id
    if sub:
        # Seat the caption cleanly below the bezel ring (below cy + r with
        # a small gap) so it never collides with radial lines or altitude
        # numbers on the rim. Tracked UPPERCASE at SIZE_MICRO.
        frag += _text(cx, cy + r + 8, sub.upper(),
                      size=SIZE_MICRO, anchor="middle",
                      fill=COLOR_ACCENT_CYAN, tracking=TRACK, weight="bold")

    for sec in sectors:
        start = float(sec.get("start_bearing_deg", 0))
        end = float(sec.get("end_bearing_deg", 0))
        alt = sec.get("altitude_ft")
        a = radians(start - 90.0)
        x1, y1 = cx + r * cos(a), cy + r * sin(a)
        frag += _line(cx, cy, x1, y1,
                      stroke=COLOR_ACCENT_CYAN, sw=0.75)
        mid = (start + end) / 2.0 if end > start else (start + (end + 360) / 2.0) % 360
        ma = radians(mid - 90.0)
        lx = cx + (r * 0.55) * cos(ma)
        ly = cy + (r * 0.55) * sin(ma) + 3
        frag += _text(
            lx, ly, f"{alt}'" if alt is not None else "",
            size=SIZE_BODY, anchor="middle", weight="bold",
            fill=COLOR_INK,
        )
    frag += _text(cx, cy - r - 2, "N",
                  size=SIZE_MICRO, anchor="middle",
                  fill=COLOR_ACCENT_CYAN, weight="bold", tracking=TRACK)
    return frag


# ---------------------------------------------------------------------------
# Profile view
# ---------------------------------------------------------------------------


def _render_profile_view(procedure: dict, region: Region) -> str:
    x, y, w, h = region
    out = _section_bar(region, "PROFILE")

    body_y = y + BAR_H
    body_h = h - BAR_H

    baseline_y = y + h - 28
    runway_x = x + w - 40

    # Baseline between FAF and RWY — the "ground" line.
    out += _line(x + 40, baseline_y, runway_x, baseline_y, sw=1.0)

    # Runway wedge at threshold.
    out += (
        f'<path d="M{_fmt(runway_x)},{_fmt(baseline_y)} '
        f'L{_fmt(runway_x + 24)},{_fmt(baseline_y)} '
        f'L{_fmt(runway_x + 24)},{_fmt(baseline_y - 5)} Z" '
        f'fill="{COLOR_RUNWAY}"/>\n'
    )
    # Runway label sits ABOVE the wedge so it can't collide with any
    # baseline annotation. TDZE is intentionally NOT duplicated here —
    # the briefing grid already carries it.
    out += _text(runway_x + 2, baseline_y - 8,
                 f"RWY {procedure.get('runway_ident','')}",
                 size=SIZE_BODY, weight="bold", family=FONT_MONO)

    brief = procedure.get("briefing_strip") or {}
    gs_angle = brief.get("gs_angle_deg") or 3.0
    gs_int = brief.get("gs_intercept_altitude_ft")

    faf_id = ""
    faf_alt = gs_int
    for leg in procedure.get("common_legs") or []:
        con = leg.get("altitude_constraint") or {}
        if con.get("altitude_description") == "glide_slope_altitude":
            faf_id = leg.get("to_fix_id") or ""
            faf_alt = con.get("altitude_1_ft") or gs_int
            break

    gp_start_x = x + 60
    rise_px = min(body_h - 40, max(30.0, (gs_angle / 3.0) * 60.0))
    gp_start_y = baseline_y - rise_px

    # ILS glideslope feather -- three thin lines from the FAF back to the
    # threshold: the magenta centerline (drawn below) plus upper and
    # lower edges at +/-2.0 deg about the commanded GS angle. NO hatch,
    # NO fill. Plus small perpendicular tick marks at integer DME values
    # (2, 3, 4, 5 NM from the threshold) with a muted micro-caption.
    # Only emit for ILS approaches; non-precision approaches keep a bare
    # centerline.
    #
    # +/-2.0 deg is VISUALLY EXAGGERATED from the real ILS glideslope
    # sensitivity cone (~+/-0.7 deg full-scale deflection); the honest
    # geometry renders as a sub-pixel sliver overlapping the centerline
    # at this panel scale. Same cosmetic exaggeration principle as the
    # 8 px minimum runway width and the LOC feather half-width above.
    subtype = (procedure.get("approach_subtype") or "").lower()
    is_ils = subtype in ("ils", "ils_dme", "loc", "loc_dme",
                         "ils_cat1", "ils_cat2", "ils_cat3")
    if is_ils:
        # Vertical spread at the FAF for +/-2.0 deg around a gs_angle
        # path. Derive geometrically from the drawn centerline so the
        # cone scales with the profile.
        run_px = runway_x - gp_start_x
        spread_px = run_px * tan(radians(2.0))
        upper_y = gp_start_y - spread_px
        lower_y = gp_start_y + spread_px
        # Upper edge from FAF-offset down to threshold.
        out += _line(gp_start_x, upper_y, runway_x, baseline_y,
                     stroke=COLOR_INK, sw=0.5)
        # Lower edge from FAF-offset up to threshold.
        out += _line(gp_start_x, lower_y, runway_x, baseline_y,
                     stroke=COLOR_INK, sw=0.5)

        # DME tick marks on the centerline at integer NM from the
        # threshold. Resolve FAF->MAP distance so ticks land on real
        # DME values; fall back to a spacing-by-default if unavailable.
        faf_map_dist_nm: float | None = None
        faf_fix_id, map_fix_id = _resolve_faf_map(procedure)
        fix_pos = build_fix_table(procedure)
        if faf_fix_id and map_fix_id and faf_fix_id in fix_pos and map_fix_id in fix_pos:
            faf_map_dist_nm = great_circle_distance_nm(
                fix_pos[faf_fix_id], fix_pos[map_fix_id]
            )
        if faf_map_dist_nm and faf_map_dist_nm > 0:
            # Centerline vector in screen space (FAF -> threshold).
            cvx = runway_x - gp_start_x
            cvy = baseline_y - gp_start_y
            from math import hypot
            cvl = hypot(cvx, cvy)
            if cvl > 0:
                px_perp = -cvy / cvl
                py_perp = cvx / cvl
                tick_len = 3.0
                # Integer DME values strictly between threshold (0) and FAF.
                n_max = int(faf_map_dist_nm)
                for dme in range(1, n_max + 1):
                    # Fraction along centerline measured from the FAF side.
                    t = 1.0 - (dme / faf_map_dist_nm)
                    # t=0 is FAF, t=1 is threshold; we interpolate FAF->threshold.
                    tx = gp_start_x + t * cvx
                    ty = gp_start_y + t * cvy
                    x0 = tx - px_perp * tick_len
                    y0 = ty - py_perp * tick_len
                    x1 = tx + px_perp * tick_len
                    y1 = ty + py_perp * tick_len
                    out += _line(x0, y0, x1, y1, stroke=COLOR_INK, sw=0.5)
                    # Micro caption below the tick.
                    out += _text(tx, ty + 10, f"D{dme}",
                                 size=SIZE_MICRO, anchor="middle",
                                 family=FONT_MONO, fill=COLOR_MUTED,
                                 tracking=TRACK)

    # Glidepath — the procedure, so use magenta.
    out += _line(gp_start_x, gp_start_y, runway_x, baseline_y,
                 stroke=COLOR_ACCENT_MAGENTA, sw=2.0)

    # FAF marker (Maltese-cross approximation).
    out += (
        f'<circle cx="{_fmt(gp_start_x)}" cy="{_fmt(gp_start_y)}" r="3.5" '
        f'fill="{COLOR_INK}"/>\n'
    )
    if faf_id:
        out += _text(gp_start_x - 4, gp_start_y - 6, faf_id,
                     size=SIZE_BODY, weight="bold", anchor="end",
                     family=FONT_MONO)
    if faf_alt is not None:
        out += _text(gp_start_x + 6, gp_start_y - 4, f"{faf_alt}'",
                     size=SIZE_BODY, weight="bold")

    # GS angle label — small mono caption anchored to the FAF end of
    # the centerline, tucked below the FAF symbol so it doesn't collide
    # with the FAF altitude text (which sits above-right of the symbol).
    # The rate-of-descent sidecar carries the angle at the bottom of
    # the minima band too, but this is the chart-reader's natural eye
    # landing point when tracing the glidepath.
    out += _text(gp_start_x - 4, gp_start_y + 14,
                 f"{gs_angle:.1f}\u00B0 GS",
                 size=SIZE_MICRO, anchor="end", weight="bold",
                 family=FONT_MONO, fill=COLOR_MUTED, tracking=TRACK)

    # DA reference — cyan overlay.
    da = None
    for m in procedure.get("minima") or []:
        if m.get("variant") == "s_ils" and "da_ft_msl" in m and not m.get("not_authorized"):
            da = m["da_ft_msl"]
            break
    if da is not None:
        da_y = baseline_y - 12
        out += _line(gp_start_x + 20, da_y, runway_x - 8, da_y,
                     stroke=COLOR_ACCENT_CYAN, sw=0.75, dash="3 2")
        out += _text(gp_start_x + 22, da_y - 3, f"DA {da}'",
                     size=SIZE_MICRO, weight="bold",
                     fill=COLOR_ACCENT_CYAN, tracking=TRACK)

    # TDZE intentionally omitted — carried by the briefing grid above, and
    # repeating it here crowds the RW13 label at the profile's right edge.

    return out


# ---------------------------------------------------------------------------
# Minima table
# ---------------------------------------------------------------------------


def _render_minima_table(procedure: dict, region: Region) -> str:
    """Split MINIMUMS region — consolidated-run minima block (left) and a
    rate-of-descent / timing sidecar (right), sharing one section bar.

    The left block groups minima by variant (S-ILS, CIRCLING, LNAV, LPV,
    ...) in their YAML order of appearance. Within each variant, adjacent
    categories A..D that share identical minima values are consolidated
    onto a single row labelled with interpunct-joined cats (e.g.
    "A\u00b7B\u00b7C"). This reclaims the vertical space a Jepp-style
    4-col matrix wastes by repeating numbers.

    The sidecar is a GS-indexed descent-rate / FAF\u2192MAP timing lookup.
    """
    x, y, w, h = region

    # --- Split geometry: left block ~355, gutter 6, right sidecar ~215 -----
    gutter = 6.0
    side_w = 215.0
    left_w = w - side_w - gutter
    side_x = x + left_w + gutter
    side_region: Region = (side_x, y, side_w, h)

    # Shared reversed-chrome header bar split into two labels with a
    # paper-coloured 1 px divider between them.
    out = _rect((x, y, w, BAR_H), fill=COLOR_INK)
    out += _text(
        x + 6, y + BAR_H - 4, "MINIMUMS",
        size=SIZE_BODY, weight="bold",
        fill=COLOR_PAPER, tracking=TRACK,
    )
    out += _text(
        side_x + 6, y + BAR_H - 4, "RATE OF DESCENT \u2022 TIMING",
        size=SIZE_BODY, weight="bold",
        fill=COLOR_PAPER, tracking=TRACK,
    )
    # 1 px paper-coloured divider inside the bar between the two labels.
    out += _line(
        side_x, y, side_x, y + BAR_H,
        stroke=COLOR_PAPER, sw=1.0,
    )

    body_y = y + BAR_H
    body_h = h - BAR_H

    minima = procedure.get("minima") or []
    if not minima:
        out += _text(x + left_w / 2, body_y + body_h / 2,
                     "(no minima)", size=SIZE_STD, anchor="middle")
        # Outer frame still drawn so the empty block reads as a region.
        out += _rect((x, body_y, left_w, body_h), stroke=COLOR_INK, sw=1.0)
        out += _render_descent_sidecar(procedure, side_region)
        return out

    # --- Group minima by variant (preserving YAML order) -------------------
    categories = ("A", "B", "C", "D")
    group_order: list[str] = []
    groups: dict[str, dict[str, dict]] = {}
    for m in minima:
        v = m.get("variant") or ""
        if v not in groups:
            groups[v] = {}
            group_order.append(v)
        cat = str(m.get("category") or "").upper()
        if cat:
            groups[v][cat] = m

    # --- Consolidation algorithm -------------------------------------------
    # For each variant, walk A..D in order and coalesce adjacent cats
    # whose minima rows produce the same (altitude, visibility) signature.
    # Cats with no entry are skipped. The result is a list of
    # (cat_label, alt_text, vis_text, muted) per consolidated row.
    def _alt_text(m: dict) -> str:
        da = m.get("da_ft_msl")
        dh = m.get("dh_ft_agl")
        mda = m.get("mda_ft_msl")
        mdh = m.get("mdh_ft_agl")
        if da is not None and dh is not None:
            return f"{da}'/{dh}'"
        if da is not None:
            return f"{da}'"
        if mda is not None and mdh is not None:
            return f"{mda}'/{mdh}'"
        if mda is not None:
            return f"{mda}'"
        return ""

    def _vis_text(m: dict) -> str:
        vis = m.get("visibility_sm")
        rvr = m.get("rvr_ft")
        parts: list[str] = []
        if isinstance(vis, (int, float)):
            # Strip the leading zero for sub-mile visibilities — Jepp
            # shorthand: .50, .75. 1.00 and above keep the integer.
            if vis < 1.0:
                parts.append(f"{vis:.2f}".lstrip("0"))
            else:
                parts.append(f"{vis:.2f}")
        if rvr is not None:
            parts.append(str(rvr))
        return "/".join(parts)

    def _signature(m: dict) -> tuple:
        """Hashable identity for run-consolidation. NA rows get a unique
        marker so they never merge with numeric rows."""
        if m.get("not_authorized"):
            return ("NA",)
        return (_alt_text(m), _vis_text(m))

    def _consolidate(cat_map: dict[str, dict]) -> list[tuple[str, dict]]:
        """Walk A..D; return list of (cat_label, representative_minima)."""
        runs: list[tuple[list[str], dict]] = []
        for cat in categories:
            m = cat_map.get(cat)
            if m is None:
                continue
            sig = _signature(m)
            if runs and _signature(runs[-1][1]) == sig:
                runs[-1][0].append(cat)
            else:
                runs.append(([cat], m))
        return [("\u00b7".join(cats), m) for cats, m in runs]

    # Pre-compute consolidated rows per variant.
    variant_rows: dict[str, list[tuple[str, dict]]] = {
        v: _consolidate(groups[v]) for v in group_order
    }

    # --- Layout geometry ---------------------------------------------------
    # Row-header column 80 px; data column is the remainder. Cat label is
    # left-aligned inside a ~45 px sub-column; alt+vis follow.
    row_hdr_w = 80.0
    data_w = left_w - row_hdr_w
    cat_label_w = 45.0

    # Base row height for a consolidated run. Target ~12 px; compress if
    # the full set of rows (plus variant labels) doesn't fit.
    line_h = 12.0
    group_pad_top = 4.0  # space above the first row of each variant group
    group_pad_bottom = 2.0

    total_rows = sum(max(1, len(rows)) for rows in variant_rows.values())
    group_pad = group_pad_top + group_pad_bottom

    # Budget check. Leave a bottom margin so the Memphis accent doesn't
    # crowd the last row.
    bottom_margin = 6.0
    required = total_rows * line_h + len(group_order) * group_pad
    avail = body_h - bottom_margin
    if required > avail and required > 0:
        scale = avail / required
        line_h *= scale
        group_pad_top *= scale
        group_pad_bottom *= scale

    # --- Outer frame --------------------------------------------------------
    out += _rect((x, body_y, left_w, body_h), stroke=COLOR_INK, sw=1.0)
    # Vertical rule between row-header and data column.
    out += _line(
        x + row_hdr_w, body_y, x + row_hdr_w, body_y + body_h,
        stroke=COLOR_INK, sw=0.75,
    )

    # --- Render each variant group ----------------------------------------
    row_top = body_y
    for gi, v in enumerate(group_order):
        rows = variant_rows[v] or [("", {})]
        n_rows = len(rows)
        group_h = n_rows * line_h + group_pad_top + group_pad_bottom

        # Hairline separator between variant groups (above all but the first).
        if gi > 0:
            out += _line(
                x, row_top, x + left_w, row_top,
                stroke=COLOR_HAIRLINE, sw=0.5,
            )

        # Row header (variant + sub-caption). Vertically anchored near top.
        variant_label = _variant_label(v)
        sub_caption = _variant_sub_caption(groups[v])
        hdr_y = row_top + group_pad_top + line_h - 3
        out += _text(
            x + 6, hdr_y, variant_label,
            size=SIZE_STD, weight="bold",
        )
        if sub_caption:
            out += _caption(x + 6, hdr_y + 9, sub_caption)

        # Data rows — one per consolidated run.
        for ri, (cat_label, m) in enumerate(rows):
            baseline = row_top + group_pad_top + (ri + 1) * line_h - 3
            cell_x0 = x + row_hdr_w + 8

            if not cat_label:
                # No data at all for this variant — render a muted em-dash.
                out += _text(
                    cell_x0, baseline, "\u2014",
                    size=SIZE_BODY, fill=COLOR_MUTED,
                )
                continue

            # Cat label: bold sans, left-aligned in its ~45 px sub-column.
            out += _text(
                cell_x0, baseline, cat_label,
                size=SIZE_BODY, weight="bold",
            )

            val_x = cell_x0 + cat_label_w

            if m.get("not_authorized"):
                # NA: mono muted to the right of the cat label.
                out += _text(
                    val_x, baseline, "NA",
                    size=SIZE_BODY, family=FONT_MONO,
                    fill=COLOR_MUTED, tracking=TRACK,
                )
                continue

            alt_s = _alt_text(m)
            vis_s = _vis_text(m)
            # Compose "<alt>  \u00b7  <vis>" with narrow-single-space
            # padding on either side of the middle-dot separator.
            if alt_s and vis_s:
                combined = f"{alt_s} \u00b7 {vis_s}"
            else:
                combined = alt_s or vis_s or "\u2014"
            out += _text(
                val_x, baseline, combined,
                size=SIZE_BODY, family=FONT_MONO,
            )

        row_top += group_h

    # Single Memphis accent: small 4x4 dot grid in the bottom-right margin
    # of the (narrower) minima block. Nudge up if the last variant group
    # extends close to the block's bottom edge.
    accent_x = x + left_w - 30
    accent_y = body_y + body_h - 30
    # If the last row's baseline is within ~20 px of the accent top, shift
    # the accent so it doesn't crowd the row.
    if row_top > accent_y - 6:
        # Move accent to the right edge of the last row's vertical band
        # instead — still bottom-right but tucked above last row if needed.
        accent_y = max(row_top + 4, body_y + body_h - 30)
        # Clamp inside the frame.
        accent_y = min(accent_y, body_y + body_h - 20)
    out += _dot_grid_accent(accent_x, accent_y)

    # Sidecar (descent-rate / FAF\u2192MAP timing) to the right of the block.
    out += _render_descent_sidecar(procedure, side_region)
    return out


def _resolve_faf_map(procedure: dict) -> tuple[str | None, str | None]:
    """Pick FAF / MAP fix ids from the procedure's legs.

    FAF: the `to_fix_id` of the last leg whose altitude_constraint
        altitude_description == "glide_slope_altitude" — i.e. the fix at
        which the aircraft crosses the GS and begins the precision
        descent. Fallback: from_fix_id of the last CF/TF before a leg
        that terminates at a runway_threshold fix.
    MAP: the `to_fix_id` of the leg terminating at a fix of
        type == "runway_threshold". Fallback: last leg of common_legs.
    """
    fixes_by_id = {
        f.get("id"): f for f in (procedure.get("fixes") or []) if f.get("id")
    }

    def _is_rwy(fix_id: str | None) -> bool:
        if not fix_id:
            return False
        f = fixes_by_id.get(fix_id)
        return bool(f and f.get("type") == "runway_threshold")

    common = list(procedure.get("common_legs") or [])

    faf: str | None = None
    for leg in common:
        ac = leg.get("altitude_constraint") or {}
        if ac.get("altitude_description") == "glide_slope_altitude":
            # The fix at which GS descent begins is the leg's endpoint.
            faf = leg.get("to_fix_id") or faf
    if faf is None:
        # Fallback: from_fix_id of the last CF/TF that precedes a leg
        # terminating at a runway_threshold.
        for i, leg in enumerate(common):
            nxt = common[i + 1] if i + 1 < len(common) else None
            if nxt and _is_rwy(nxt.get("to_fix_id")) \
                    and leg.get("terminator") in ("CF", "TF"):
                faf = leg.get("from_fix_id") or faf

    map_id: str | None = None
    for leg in common:
        if _is_rwy(leg.get("to_fix_id")):
            map_id = leg.get("to_fix_id")
    if map_id is None and common:
        map_id = common[-1].get("to_fix_id")

    return faf, map_id


def _render_descent_sidecar(procedure: dict, region: Region) -> str:
    """GS-indexed rate-of-descent / FAF→MAP timing lookup table.

    Columns: GS (KT) | FPM | FAF→MAP
      - GS fixed at (70, 90, 110, 140, 180, 200) kt.
      - FPM = gs * tan(angle) * 101.27 rounded to nearest 5 fpm.
        Dropped entirely when gs_angle_deg is absent (non-precision).
      - FAF→MAP time = distance_nm / gs * 60, formatted M:SS.
        If either FAF or MAP can't be resolved, a muted message is shown
        in place of the timing column.
    """
    x, y, w, h = region
    body_y = y + BAR_H
    body_h = h - BAR_H

    brief = procedure.get("briefing_strip") or {}
    gs_angle = brief.get("gs_angle_deg")
    has_fpm = isinstance(gs_angle, (int, float)) and gs_angle > 0

    # Resolve FAF / MAP → distance_nm.
    faf_id, map_id = _resolve_faf_map(procedure)
    fix_pos = build_fix_table(procedure)
    dist_nm: float | None = None
    if faf_id and map_id and faf_id in fix_pos and map_id in fix_pos:
        dist_nm = great_circle_distance_nm(fix_pos[faf_id], fix_pos[map_id])

    # --- Columns -----------------------------------------------------------
    # Layout the inner columns so the sidecar stays readable in 215 px.
    # Order: GS | FPM | TIME. When FPM is suppressed: GS | TIME (wider).
    col_labels: list[str]
    col_widths: list[float]
    if has_fpm:
        col_labels = ["GS (KT)", "FPM", "FAF\u2192MAP"]
        col_widths = [60.0, 65.0, w - 60.0 - 65.0]
    else:
        col_labels = ["GS (KT)", "FAF\u2192MAP"]
        col_widths = [80.0, w - 80.0]

    # --- Outer frame + header row -----------------------------------------
    out = _rect((x, body_y, w, body_h), stroke=COLOR_INK, sw=1.0)

    hdr_h = 14.0
    # Column header captions (paper ground, muted micro-caps).
    for ci, label in enumerate(col_labels):
        cx0 = x + sum(col_widths[:ci])
        cx_mid = cx0 + col_widths[ci] / 2.0
        out += _text(
            cx_mid, body_y + hdr_h - 4, label,
            size=SIZE_MICRO, weight="bold", anchor="middle",
            fill=COLOR_MUTED, tracking=TRACK,
        )
    # Hairline beneath the header row.
    out += _line(
        x, body_y + hdr_h, x + w, body_y + hdr_h,
        stroke=COLOR_INK, sw=0.75,
    )

    # --- Column dividers ---------------------------------------------------
    cum = x
    for cw in col_widths[:-1]:
        cum += cw
        out += _line(cum, body_y, cum, body_y + body_h,
                     stroke=COLOR_HAIRLINE, sw=0.75)

    # --- Rows --------------------------------------------------------------
    gs_values = (70, 90, 110, 140, 180, 200)
    # Reserve a slim footer strip only when we have a fallback message
    # (e.g. "FAF/MAP timing unavailable"); otherwise rows fill the whole
    # body so the sidecar table breathes.
    footer_h = 12.0 if dist_nm is None else 0.0
    rows_top = body_y + hdr_h
    rows_avail = body_h - hdr_h - footer_h
    row_h = rows_avail / len(gs_values)

    def _fmt_mss(minutes: float) -> str:
        total_sec = int(round(minutes * 60.0))
        return f"{total_sec // 60}:{total_sec % 60:02d}"

    for ri, gs in enumerate(gs_values):
        row_y = rows_top + ri * row_h
        if ri > 0:
            out += _line(
                x, row_y, x + w, row_y,
                stroke=COLOR_HAIRLINE, sw=0.5,
            )
        baseline = row_y + row_h / 2.0 + 3.0

        # GS cell — bold monospace.
        cx_mid = x + col_widths[0] / 2.0
        out += _text(
            cx_mid, baseline, str(gs),
            size=SIZE_BODY, weight="bold", anchor="middle",
            family=FONT_MONO,
        )

        if has_fpm:
            fpm = gs * tan(radians(float(gs_angle))) * 101.27
            fpm_rounded = int(round(fpm / 5.0) * 5)
            cx_mid = x + col_widths[0] + col_widths[1] / 2.0
            out += _text(
                cx_mid, baseline, str(fpm_rounded),
                size=SIZE_BODY, anchor="middle", family=FONT_MONO,
            )
            time_cx = x + col_widths[0] + col_widths[1] + col_widths[2] / 2.0
        else:
            time_cx = x + col_widths[0] + col_widths[1] / 2.0

        # Timing cell — or em-dash if distance unavailable.
        if dist_nm is not None:
            out += _text(
                time_cx, baseline, _fmt_mss(dist_nm / gs * 60.0),
                size=SIZE_BODY, anchor="middle", family=FONT_MONO,
            )
        else:
            out += _text(
                time_cx, baseline, "\u2014",
                size=SIZE_BODY, anchor="middle", fill=COLOR_MUTED,
            )

    # --- Footer messages ---------------------------------------------------
    if dist_nm is None:
        # Muted fallback across the bottom of the sidecar body.
        out += _text(
            x + w / 2.0, body_y + body_h - 4,
            "FAF/MAP timing unavailable",
            size=SIZE_MICRO, anchor="middle",
            fill=COLOR_MUTED, italic=True, tracking=TRACK,
        )
    # GS-angle caption removed — the profile carries the angle as a
    # small mono label at the FAF, so duplicating it here was dead weight.

    return out


def _variant_sub_caption(cat_map: dict[str, dict]) -> str:
    """Micro-caption describing what the stacked sub-values mean for
    this variant (e.g. "DA/DH • VIS/RVR" for S-ILS, "MDA/MDH • VIS"
    for CIRCLING/LNAV)."""
    has_da = False
    has_dh = False
    has_mda = False
    has_mdh = False
    has_vis = False
    has_rvr = False
    for m in cat_map.values():
        if m.get("not_authorized"):
            continue
        if m.get("da_ft_msl") is not None:
            has_da = True
        if m.get("dh_ft_agl") is not None:
            has_dh = True
        if m.get("mda_ft_msl") is not None:
            has_mda = True
        if m.get("mdh_ft_agl") is not None:
            has_mdh = True
        if isinstance(m.get("visibility_sm"), (int, float)):
            has_vis = True
        if m.get("rvr_ft") is not None:
            has_rvr = True
    alt_bits = []
    if has_da:
        alt_bits.append("DA")
    if has_dh:
        alt_bits.append("DH")
    if has_mda:
        alt_bits.append("MDA")
    if has_mdh:
        alt_bits.append("MDH")
    vis_bits = []
    if has_vis:
        vis_bits.append("VIS")
    if has_rvr:
        vis_bits.append("RVR")
    parts = []
    if alt_bits:
        parts.append("/".join(alt_bits))
    if vis_bits:
        parts.append("/".join(vis_bits))
    return " \u2022 ".join(parts)


def _dot_grid_accent(x: float, y: float) -> str:
    """4x4 grid of 1.5 px filled squares — NATO/Memphis punctuation.

    Emitted exactly once per chart (from the minima table).
    """
    sz = 1.5
    step = 5.0
    frag = ""
    for i in range(4):
        for j in range(4):
            px = x + j * step
            py = y + i * step
            frag += (
                f'<rect x="{_fmt(px)}" y="{_fmt(py)}" '
                f'width="{_fmt(sz)}" height="{_fmt(sz)}" fill="{COLOR_INK}"/>\n'
            )
    return frag


def _variant_label(v: str | None) -> str:
    if not v:
        return "—"
    return v.replace("_", "-").upper()


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def render_iap_svg(procedure: dict, primitives: list[Primitive] | None = None) -> str:
    """Render an IAP procedure to an SVG document string."""
    fixes = build_fix_table(procedure)
    if primitives is None:
        primitives = compile_legs(procedure, fixes)

    regions = _layout_regions()

    parts: list[str] = [_svg_open(PAGE_W, PAGE_H)]
    # Title comes first; plate code sits on top of the briefing bar below it.
    parts.append(_render_title_strip(procedure, regions["title"]))
    parts.append(_render_briefing_strip(procedure, regions["briefing"]))
    # Plate code overlays the right end of the briefing bar (reversed chrome).
    parts.append(_plate_code(procedure, regions["briefing"]))
    parts.append(_render_plan_view(procedure, primitives, fixes, regions["plan"]))
    parts.append(_render_profile_view(procedure, regions["profile"]))
    parts.append(_render_minima_table(procedure, regions["minima"]))
    parts.append(_render_footer(procedure, regions["comms"]))
    # Corner brackets last so they're on top.
    parts.append(_corner_brackets(PAGE_W, PAGE_H))
    parts.append(_svg_close())
    return "".join(parts)


__all__ = ["render_iap_svg"]
