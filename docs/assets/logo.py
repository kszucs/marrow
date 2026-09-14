# /// script
# requires-python = ">=3.11"
# dependencies = ["fonttools", "uharfbuzz", "resvg-py"]
# ///
"""Generate marrow's logo assets into this directory.

    uv run docs/assets/logo.py

Everything is drawn here: the chevrons are computed polygons and the type is
Urbanist (SIL OFL) converted to outlines, so no SVG depends on an installed
font. A `-dark` file is for dark backgrounds (light ink), a `-light` file for
light ones.

    logo-{dark,light}.svg             name and tagline beside three chevrons
    logo-horizontal-{dark,light}.svg  overlapping chevrons, then the name
    icon.svg                          the app tile
    favicon.png                       the tile's heavier cut, for 16-32 px
    social.png                        the 1200x630 Open Graph card
"""

import hashlib
import io
import math
import urllib.request
from pathlib import Path

import resvg_py
import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT = Path(__file__).parent

FONT_URL = (
    "https://raw.githubusercontent.com/google/fonts/"
    "69409947a524cb5c9fdc3977270e8bcb3a95a498/ofl/urbanist/Urbanist%5Bwght%5D.ttf"
)
FONT_SHA256 = "748362d51eb276840e9f0023cd2d98299bf709a95fb3a5ae4ffcb624c638ebb9"

TAGLINE = "APACHE ARROW IN MOJO"
LABEL = "marrow — Apache Arrow in Mojo"

XH = 100.0  # the name's x-height; every other size is relative to it

INK = {"dark": "#ffffff", "light": "#1c1e26"}
SUB = {"dark": "#c9cfdb", "light": "#1c1e26"}
RAMP = {  # the chevrons: red, orange, amber (docs/theme/marrow-{dark,light}.scss)
    "dark": ("#e5341f", "#ff6a00", "#ffc400"),
    "light": ("#d62c17", "#ea5a00", "#f5a300"),
}
TILE = ("#e5341f", "#ff6a00", "#ffb000")

# One chevron, used everywhere: the arm slope of the original artwork, and
# thickness, step and corner radius as fractions of its own size.
SLOPE = 1.25  # dy/dx of a '>' arm, about 51 degrees
THICKNESS = 0.235  # horizontal arm thickness / chevron height
STEP = 1.46  # distance between chevrons / arm thickness
ROUNDING = 0.16  # corner radius / stroke weight


def num(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def gradient(gid, x1, y1, x2, y2, stops):
    n = len(stops) - 1
    s = "".join(
        f'<stop offset="{num(i / n)}" stop-color="{c}"/>' for i, c in enumerate(stops)
    )
    return (
        f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" '
        f'x1="{num(x1)}" y1="{num(y1)}" x2="{num(x2)}" y2="{num(y2)}">{s}</linearGradient>'
    )


def document(box, body, defs="", label=LABEL):
    x, y, w, h = box
    defs = f"<defs>{defs}</defs>" if defs else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{num(w)}" height="{num(h)}" '
        f'viewBox="{num(x)} {num(y)} {num(w)} {num(h)}" role="img" aria-label="{label}">'
        f"{defs}{body}</svg>\n"
    )


# ------------------------------------------------------------------ type


def load_font():
    data = urllib.request.urlopen(FONT_URL).read()
    if hashlib.sha256(data).hexdigest() != FONT_SHA256:
        raise RuntimeError(f"unexpected font contents from {FONT_URL}")
    return data


class Run:
    """A shaped line of Urbanist, outlined, measured in px once sized."""

    def __init__(self, font, text, weight, tracking_em):
        os2 = TTFont(io.BytesIO(font))["OS/2"]
        self.x_height, self.cap_height = os2.sxHeight, os2.sCapHeight
        face = hb.Face(hb.Blob(font))
        self.font = hb.Font(face)
        self.font.set_variations({"wght": weight})
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.font, buf, {"kern": True, "liga": False})
        tracking = tracking_em * face.upem
        self.glyphs, x = [], 0
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            self.glyphs.append((info.codepoint, x + pos.x_offset, pos.y_offset))
            x += pos.x_advance + tracking
        self.scale = 1.0

    def ink(self):
        """Ink box in font units: (left, bottom, right, top)."""
        boxes = [
            (
                gx + e.x_bearing,
                gy + e.y_bearing + e.height,
                gx + e.x_bearing + e.width,
                gy + e.y_bearing,
            )
            for gid, gx, gy in self.glyphs
            if (e := self.font.get_glyph_extents(gid)).width
        ]
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    def width(self):
        left, _, right, _ = self.ink()
        return (right - left) * self.scale

    def path(self, x, baseline, fill):
        """The outlines with ink-left at `x`, in absolute px (so gradients mean what they say)."""
        left = self.ink()[0]
        k = self.scale
        commands = []
        for gid, gx, gy in self.glyphs:
            pen = SVGPathPen(None, ntos=lambda v: f"{v:.1f}".rstrip("0").rstrip("."))
            transform = (k, 0, 0, -k, x + (gx - left) * k, baseline - gy * k)
            self.font.draw_glyph_with_pen(gid, TransformPen(pen, transform))
            commands.append(pen.getCommands())
        return f'<path fill="{fill}" d="{"".join(commands)}"/>'


def name(font):
    run = Run(font, "marrow", 600, -0.012)
    run.scale = XH / run.x_height
    return run


def tagline(font, width):
    """Tracked caps whose ink spans exactly `width`."""
    run = Run(font, TAGLINE, 500, 0.19)
    run.scale = width / run.width()
    return run


def tagline_baseline(tag):
    return XH + 0.45 * XH + tag.cap_height * tag.scale


# ------------------------------------------------------------------ chevrons


def rounded(points, radii):
    """A closed polygon with a circular fillet of radii[i] at each vertex."""
    corners = []
    n = len(points)
    for i in range(n):
        p, a, b = points[i], points[i - 1], points[(i + 1) % n]
        u = (a[0] - p[0], a[1] - p[1])
        v = (b[0] - p[0], b[1] - p[1])
        lu, lv = math.hypot(*u), math.hypot(*v)
        u, v = (u[0] / lu, u[1] / lu), (v[0] / lv, v[1] / lv)
        angle = math.acos(max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1])))
        cut = min(radii[i] / math.tan(angle / 2), 0.49 * lu, 0.49 * lv)
        radius = cut * math.tan(angle / 2)
        start = (p[0] + u[0] * cut, p[1] + u[1] * cut)
        end = (p[0] + v[0] * cut, p[1] + v[1] * cut)
        turn = (p[0] - a[0]) * (b[1] - p[1]) - (p[1] - a[1]) * (b[0] - p[0])
        corners.append((start, end, radius, 1 if turn > 0 else 0))
    d = f"M{num(corners[0][0][0])} {num(corners[0][0][1])}"
    for i, (start, end, radius, sweep) in enumerate(corners):
        if i:
            d += f"L{num(start[0])} {num(start[1])}"
        d += f"A{num(radius)} {num(radius)} 0 0 {sweep} {num(end[0])} {num(end[1])}"
    return d + "Z"


def chevron(x, top, height, thickness):
    """A '>' whose arms end square to their own direction; returns (path data, width)."""
    angle = math.atan(SLOPE)
    s, c = math.sin(angle), math.cos(angle)
    weight = thickness * s
    half = height / 2
    inner_end = (0.0, weight * c)
    outer_end = (weight * s, 0.0)
    tip = (outer_end[0] + half / s * c, half)
    notch = (inner_end[0] + (half - inner_end[1]) / s * c, half)
    points = [
        inner_end,
        outer_end,
        tip,
        (outer_end[0], height),
        (inner_end[0], height - inner_end[1]),
        notch,
    ]
    points = [(x + px, top + py) for px, py in points]
    r = ROUNDING * weight
    return rounded(points, [r, r, 0.8 * r, r, r, 0.6 * r]), tip[0]


def chevron_row(x, top, height, thickness=THICKNESS):
    """Three chevrons side by side; returns (path data, width)."""
    t = thickness * height
    paths, width = [], 0.0
    for i in range(3):
        d, w = chevron(x + i * STEP * t, top, height, t)
        paths.append(d)
        width = i * STEP * t + w
    return "".join(paths), width


def chevron_stack(x, top, height, colours, gid):
    """Three chevrons, each in front of the last with a gap cut out behind it."""
    t = 0.30 * height
    step, gap = 0.80 * t, 0.042 * height
    paths, width = [], 0.0
    for i in range(3):
        d, w = chevron(x + i * step, top, height, t)
        paths.append(d)
        width = i * step + w
    box = f'x="{num(x - height)}" y="{num(top - height)}" width="{num(width + 2 * height)}" height="{num(3 * height)}"'
    defs, body = "", ""
    for i, d in enumerate(paths):
        if i < 2:
            defs += (
                f'<mask id="{gid}{i}" maskUnits="userSpaceOnUse" {box}><rect {box} fill="#fff"/>'
                f'<path d="{paths[i + 1]}" fill="#000" stroke="#000" stroke-width="{num(2 * gap)}" '
                f'stroke-linejoin="round"/></mask>'
            )
            body += f'<path fill="{colours[i]}" mask="url(#{gid}{i})" d="{d}"/>'
        else:
            body += f'<path fill="{colours[i]}" d="{d}"/>'
    return defs, body, width


# ------------------------------------------------------------------ tile


def squircle(x, y, size, radius, smoothing=0.6):
    """A square with continuously curved corners (Figma's corner-smoothing construction)."""
    p = min((1 + smoothing) * radius, size / 2)
    arc = 90 * (1 - smoothing)
    arc_length = math.sin(math.radians(arc / 2)) * radius * math.sqrt(2)
    alpha = (90 - arc) / 2
    beta = 45 * smoothing
    c = radius * math.tan(math.radians(alpha / 2)) * math.cos(math.radians(beta))
    d = c * math.tan(math.radians(beta))
    b = (p - arc_length - c - d) / 3
    a = 2 * b
    r, al, n = radius, arc_length, num
    return (
        f"M{n(x + size - p)} {n(y)}"
        f"c{n(a)} 0 {n(a + b)} 0 {n(a + b + c)} {n(d)}"
        f"a{n(r)} {n(r)} 0 0 1 {n(al)} {n(al)}"
        f"c{n(d)} {n(c)} {n(d)} {n(b + c)} {n(d)} {n(a + b + c)}"
        f"L{n(x + size)} {n(y + size - p)}"
        f"c0 {n(a)} 0 {n(a + b)} {n(-d)} {n(a + b + c)}"
        f"a{n(r)} {n(r)} 0 0 1 {n(-al)} {n(al)}"
        f"c{n(-c)} {n(d)} {n(-(b + c))} {n(d)} {n(-(a + b + c))} {n(d)}"
        f"L{n(x + p)} {n(y + size)}"
        f"c{n(-a)} 0 {n(-(a + b))} 0 {n(-(a + b + c))} {n(-d)}"
        f"a{n(r)} {n(r)} 0 0 1 {n(-al)} {n(-al)}"
        f"c{n(-d)} {n(-c)} {n(-d)} {n(-(b + c))} {n(-d)} {n(-(a + b + c))}"
        f"L{n(x)} {n(y + p)}"
        f"c0 {n(-a)} 0 {n(-(a + b))} {n(d)} {n(-(a + b + c))}"
        f"a{n(r)} {n(r)} 0 0 1 {n(al)} {n(-al)}"
        f"c{n(c)} {n(-d)} {n(b + c)} {n(-d)} {n(a + b + c)} {n(-d)}Z"
    )


def icon(small=False):
    """The app tile; `small` is a heavier cut that stays three chevrons at 16 px."""
    size = 256.0
    height = (0.56 if small else 0.50) * size
    thickness = 0.30 if small else THICKNESS
    _, width = chevron_row(0, 0, height, thickness)
    x = (
        size - width
    ) / 2 - 0.018 * size  # arrows read heavier on the side they point to
    chevrons, _ = chevron_row(x, (size - height) / 2, height, thickness)
    body = (
        f'<path fill="url(#mr-tile)" d="{squircle(0, 0, size, 0.235 * size)}"/>'
        f'<path fill="#ffffff" d="{chevrons}"/>'
    )
    return document(
        (0, 0, size, size), body, gradient("mr-tile", 0, size, size, 0, TILE), "marrow"
    )


# ------------------------------------------------------------------ lockups


def logo(font, mode):
    """The name over the tagline, with three chevrons to the right."""
    word = name(font)
    tag = tagline(font, word.width())
    baseline = tagline_baseline(tag)
    height = 1.2 * baseline
    top = baseline / 2 - height / 2
    x = word.width() + 0.52 * XH
    chevrons, width = chevron_row(x, top, height)
    body = (
        word.path(0, XH, INK[mode])
        + tag.path(0, baseline, SUB[mode])
        + f'<path fill="url(#mr-chevrons)" d="{chevrons}"/>'
    )
    defs = gradient("mr-chevrons", x, 0, x + width, 0, RAMP[mode])
    return document((0, top, x + width, height), body, defs)


def logo_horizontal(font, mode):
    """Overlapping chevrons, then the name, on one line."""
    word = name(font)
    height = 1.62 * XH
    top = XH / 2 - height / 2
    defs, body, width = chevron_stack(0, top, height, RAMP[mode], "mr-stack")
    x = width + 0.44 * XH
    body += word.path(x, XH, INK[mode])
    return document((0, top, x + word.width(), height), body, defs, "marrow")


def social(font):
    """The dark logo centred on the site's page colour."""
    inner = logo(font, "dark")
    box = [float(v) for v in inner.split('viewBox="')[1].split('"')[0].split()]
    scale = 1200 * 0.7 / box[2]
    w, h = 1200 / scale, 630 / scale
    x, y = box[0] - (w - box[2]) / 2, box[1] - (h - box[3]) / 2
    content = inner.split(">", 1)[1].rsplit("</svg>", 1)[0]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" '
        f'viewBox="{num(x)} {num(y)} {num(w)} {num(h)}">'
        f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" fill="#0b0d12"/>'
        f"{content}</svg>\n"
    )


def main():
    font = load_font()
    for mode in ("dark", "light"):
        (OUT / f"logo-{mode}.svg").write_text(logo(font, mode))
        (OUT / f"logo-horizontal-{mode}.svg").write_text(logo_horizontal(font, mode))
    (OUT / "icon.svg").write_text(icon())
    (OUT / "favicon.png").write_bytes(
        resvg_py.svg_to_bytes(svg_string=icon(small=True), width=64)
    )
    (OUT / "social.png").write_bytes(
        resvg_py.svg_to_bytes(svg_string=social(font), width=1200)
    )


if __name__ == "__main__":
    main()
