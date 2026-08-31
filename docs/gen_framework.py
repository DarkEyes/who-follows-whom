"""Generate framework.svg for who-follows-whom, matching the approved
pictogram design: video frames -> tracking -> V-JEPA (frozen) ->
embeddings + corr -> z heatmap (dashed diagonal, red cell) ->
[red arrow "P2, P4, t"] -> Gemma (frozen, fed by frame strip) ->
description bubble; z -> aggregate stack -> who-follows-whom digraph.
"""

E = []
W, H = 2010, 780
FONT = "font-family='Helvetica, Arial, sans-serif'"


def rect(x, y, w, h, rx=10, fill="#f7f7f7", stroke="#bbb", sw=1.5,
         dash=None):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    E.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' "
             f"fill='{fill}' stroke='{stroke}' stroke-width='{sw}'{d}/>")


def text(x, y, s, size=15, fill="#333", weight="normal", anchor="middle",
         style=""):
    E.append(f"<text x='{x}' y='{y}' font-size='{size}' fill='{fill}' "
             f"font-weight='{weight}' text-anchor='{anchor}' {FONT} "
             f"{style}>{s}</text>")


def arrow(x1, y1, x2, y2, color="#777", sw=2.2, marker="arr"):
    E.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' "
             f"stroke='{color}' stroke-width='{sw}' "
             f"marker-end='url(#{marker})'/>")


def path_arrow(d, color="#777", sw=2.2, marker="arr"):
    E.append(f"<path d='{d}' fill='none' stroke='{color}' "
             f"stroke-width='{sw}' marker-end='url(#{marker})'/>")


def snowflake(cx, cy, r=9, color="white"):
    import math as _m
    for k in range(6):
        a = _m.pi / 3 * k
        x2, y2 = cx + r * _m.cos(a), cy + r * _m.sin(a)
        E.append(f"<line x1='{cx}' y1='{cy}' x2='{x2}' y2='{y2}' "
                 f"stroke='{color}' stroke-width='1.8'/>")
        for t in (0.55,):
            bx, by = cx + r * t * _m.cos(a), cy + r * t * _m.sin(a)
            for da in (-0.5, 0.5):
                x3 = bx + r * 0.3 * _m.cos(a + da)
                y3 = by + r * 0.3 * _m.sin(a + da)
                E.append(f"<line x1='{bx}' y1='{by}' x2='{x3}' y2='{y3}' "
                         f"stroke='{color}' stroke-width='1.4'/>")


def person(cx, cy, s=1.0, color="#2b2b2b"):
    """Simple seated-person pictogram: head + shoulders/body."""
    E.append(f"<circle cx='{cx}' cy='{cy}' r='{5.2*s}' fill='{color}'/>")
    E.append(f"<path d='M {cx-7*s} {cy+16*s} q 0 -{9*s} {7*s} -{9*s} "
             f"q {7*s} 0 {7*s} {9*s} z' fill='{color}'/>")


def table_scene(x, y, w, h, boxed=False):
    """One video-frame pictogram: table ellipse + 4 people."""
    rect(x, y, w, h, rx=8, fill="#eef4fd", stroke="#a9c4ea", sw=1.2)
    cx, cy = x + w / 2, y + h / 2
    E.append(f"<ellipse cx='{cx}' cy='{cy+4}' rx='{w*0.30}' ry='{h*0.16}' "
             f"fill='#c9dcf5'/>")
    pos = [(cx - w * 0.26, cy - h * 0.22), (cx + w * 0.26, cy - h * 0.22),
           (cx - w * 0.26, cy + h * 0.16), (cx + w * 0.26, cy + h * 0.16)]
    cols = ["#2e9e4f", "#e8871e", "#2f6fd0", "#d63b3b"]
    for k, (px, py) in enumerate(pos):
        person(px, py, 0.85)
        if boxed:
            rect(px - 13, py - 12, 26, 34, rx=5, fill="none",
                 stroke=cols[k], sw=2.2)


def grid(x, y, cell=7, n=4, fill="#cfe0f7", stroke="#7fa6dd"):
    for i in range(n):
        for j in range(n):
            E.append(f"<rect x='{x+j*cell}' y='{y+i*cell}' "
                     f"width='{cell-1.2}' height='{cell-1.2}' "
                     f"fill='{fill}' stroke='{stroke}' "
                     f"stroke-width='0.6'/>")


def heatmap(x, y, cell=30, vals=None, red=None, dashed_diag=True):
    import random
    rnd = random.Random(7)
    for i in range(4):
        for j in range(4):
            if i == j and dashed_diag:
                E.append(f"<rect x='{x+j*cell}' y='{y+i*cell}' "
                         f"width='{cell-3}' height='{cell-3}' fill='white' "
                         f"stroke='#bbb' stroke-width='1.2' "
                         f"stroke-dasharray='3,3'/>")
                continue
            v = vals[i][j] if vals else rnd.uniform(0.1, 0.95)
            c = int(235 - v * 150)
            E.append(f"<rect x='{x+j*cell}' y='{y+i*cell}' "
                     f"width='{cell-3}' height='{cell-3}' "
                     f"fill='rgb({c},{c+10},245)' stroke='none'/>")
    if red:
        i, j = red
        E.append(f"<rect x='{x+j*cell-1.5}' y='{y+i*cell-1.5}' "
                 f"width='{cell}' height='{cell}' fill='none' "
                 f"stroke='#d62222' stroke-width='2.6'/>")


def note(x, y, lines, size=12.5, fill="#666"):
    for k, s in enumerate(lines):
        text(x, y + k * 16, s, size, fill=fill)


# ---------- canvas ----------
E.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' "
         f"height='{H}' viewBox='0 0 {W} {H}'>")
E.append("<defs>"
         "<marker id='arr' viewBox='0 0 10 10' refX='9' refY='5' "
         "markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
         "<path d='M 0 0 L 10 5 L 0 10 z' fill='#777'/></marker>"
         "<marker id='arrRed' viewBox='0 0 10 10' refX='9' refY='5' "
         "markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
         "<path d='M 0 0 L 10 5 L 0 10 z' fill='#d62222'/></marker>"
         "</defs>")
E.append(f"<rect width='{W}' height='{H}' fill='white'/>")

MID = 390

# 1. video container
rect(30, MID - 130, 330, 240, rx=16)
text(195, MID - 100, "video", 17)
for k in range(3):
    table_scene(50 + k * 105, MID - 80, 95, 130)
    text(97 + k * 105, MID + 80, f"T{k}", 14,
         style="font-style='italic'")

note(195, MID + 150, ["input: mp4, fixed camera,", "2-6 people around a table"])
arrow(370, MID, 420, MID)

# 2. tracking
rect(430, MID - 130, 190, 260, rx=16, fill="#eef4fd", stroke="#a9c4ea")
text(525, MID - 100, "tracking", 17)
table_scene(455, MID - 75, 140, 175, boxed=True)
cols = ["#2e9e4f", "#e8871e", "#2f6fd0", "#d63b3b"]
lab = [(470, MID - 82, "P1"), (585, MID - 82, "P2"),
       (470, MID + 122, "P3"), (585, MID + 122, "P4")]
for (lx, ly, s), c in zip(lab, cols):
    text(lx, ly, s, 15, fill=c, weight="bold")

note(525, MID + 170, ["detect + track each person", "output: boxes P1..PN per frame"])
arrow(630, MID, 680, MID)

# 3. V-JEPA block
rect(690, MID - 75, 170, 150, rx=18, fill="#2f6fd0", stroke="#2f6fd0")
text(775, MID + 6, "V-JEPA", 26, fill="white", weight="bold")
snowflake(830, MID + 52)

note(775, MID + 115, ["encodes each person's crop:", "16 embeddings / person / 5 s window"])
arrow(870, MID, 918, MID)

# 4. embeddings container
rect(928, MID - 130, 300, 260, rx=16)
text(955, MID - 70, "P2", 16, fill="#e8871e", weight="bold")
for k in range(4):
    grid(985 + k * 60, MID - 95)
    text(999 + k * 60, MID - 105, f"z{k+1}", 12)
text(955, MID + 95, "P4", 16, fill="#d63b3b", weight="bold")
for k in range(4):
    grid(985 + k * 60, MID + 62)
    text(999 + k * 60, MID + 52, f"z{k+1}", 12)
path_arrow(f"M 1020 {MID-40} Q 1044 {MID-8} 1020 {MID+28}",
           color="#888", sw=1.8)
path_arrow(f"M 1145 {MID+28} Q 1121 {MID-8} 1145 {MID-40}",
           color="#888", sw=1.8)
text(1082, MID - 2, "corr(&#916;z, lag)", 15,
     style="font-style='italic'")

note(1078, MID + 170, ["compare embedding-change sequences", "of every pair, at lags 0-1.25 s"])
arrow(1238, MID, 1286, MID)

# 5. z heatmap
rect(1296, MID - 90, 150, 180, rx=14)
text(1371, MID - 62, "z", 17, weight="bold",
     style="font-style='italic'")
heatmap(1312, MID - 48, cell=30, red=(1, 3))
note(1371, MID - 128, ["z: coordination vs chance,", "per pair, per 5 s window"])

# 6. red selection arrow to Gemma
arrow(1456, MID - 20, 1596, MID - 20, color="#d62222", sw=3.2,
      marker="arrRed")
text(1526, MID - 34, "P2, P4, t", 16, fill="#d62222", weight="bold")
note(1526, MID + 4, ["only the winning pair +", "its interval cross over"], fill="#b04a4a")

# 7. Gemma + frame strip above
rect(1500, 60, 370, 110, rx=14)
for k in range(5):
    table_scene(1512 + k * 71, 72, 64, 86)
arrow(1685, 176, 1685, 240)
rect(1606, 250, 160, 145, rx=18, fill="#2e9e4f", stroke="#2e9e4f")
text(1686, 328, "Gemma", 25, fill="white", weight="bold")
snowflake(1738, 372)

note(1686, 424, ["sees 5 raw frames + the pair's", "names; no scores, no embeddings"])

# 8. description bubble
arrow(1772, 322, 1808, 322)
E.append("<path d='M 1818 262 h 150 a 12 12 0 0 1 12 12 v 80 a 12 12 0 0 1 "
         "-12 12 h -105 l -22 24 v -24 h -23 a 12 12 0 0 1 -12 -12 v -80 "
         "a 12 12 0 0 1 12 -12 z' fill='#fdfdfd' stroke='#999' "
         "stroke-width='1.8'/>")
for k in range(3):
    E.append(f"<rect x='1838' y='{285+k*22}' width='{110-k*22}' "
             f"height='9' rx='4' fill='#b9b9b9'/>")
text(1893, 402, "output: what visibly", 13, fill="#666")
text(1893, 418, "happened (text)", 13, fill="#666")

# 9. aggregate stack + digraph
path_arrow(f"M 1371 {MID+96} L 1371 {MID+150} L 1420 {MID+150}")
for k in range(2, -1, -1):
    off = k * 12
    rect(1430 + off, MID + 96 + off, 130, 130, rx=10, fill="white",
         stroke="#aaa", sw=1.4)
heatmap(1445, MID + 112, cell=27, red=None)
text(1508, MID + 268, "aggregate", 15)
note(1508, MID + 288, ["combine all windows"])

arrow(1596, MID + 160, 1636, MID + 160)
rect(1646, MID + 70, 220, 190, rx=16)
nodes = {"P1": (1700, MID + 115, "#2e9e4f"),
         "P2": (1815, MID + 115, "#e8871e"),
         "P3": (1700, MID + 215, "#2f6fd0"),
         "P4": (1815, MID + 215, "#d63b3b")}
edges = [("P4", "P2", 4.6), ("P2", "P4", 2.2), ("P3", "P1", 2.6),
         ("P1", "P3", 2.0), ("P4", "P1", 2.2), ("P3", "P2", 1.6)]
import math
for a, b, w in edges:
    ax, ay, _ = nodes[a]
    bx, by, _ = nodes[b]
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    sx, sy = ax + ux * 16, ay + uy * 16
    ex, ey = bx - ux * 18, by - uy * 18
    mx, my = (sx + ex) / 2 - uy * 14, (sy + ey) / 2 + ux * 14
    path_arrow(f"M {sx} {sy} Q {mx} {my} {ex} {ey}", color="#666", sw=w)
for s, (nx, ny, c) in nodes.items():
    E.append(f"<circle cx='{nx}' cy='{ny}' r='13' fill='{c}'/>")
    ly = ny - 20 if ny < MID + 160 else ny + 30
    text(nx, ly, s, 14, fill=c, weight="bold")
text(1756, MID + 292, "output: who follows whom", 14)
note(1756, MID + 312, ["directed strengths F, S"])

E.append("</svg>")
open("/home/claude/fig/framework.svg", "w").write("\n".join(E))
print("written", sum(len(e) for e in E), "chars")
