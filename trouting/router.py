"""Negotiated-congestion routing of a whole board.

Every net is routed each round as if it had the board to itself, except that
copper another net already uses costs extra, and so do places that were
fought over in earlier rounds. The extra rises every round; nets that have a
way around take it, and the net that really needs a spot keeps it. Rounds
only reroute the nets still in a fight. What's left fighting at the end is
taken out and reported, with where it fought (the crowded spots the
placement suggestions work from).

Plane nets (a pour over >= 25 % of a layer) aren't routed pad to pad: each
pad not already in the pour gets a short track and a via into it.
"""
import math
import time

import numpy as np
import shapely
from scipy import ndimage

from .board import MARGIN, shape_of
from .core import Grid, Path, thin_of


def grow(mask, r):
    """Cells within r cells (Euclidean) of mask. (ndimage's binary_dilation steps
    in a diamond, which falls 30 % short on the diagonals.)"""
    if r <= 0:
        return mask.copy()
    return ndimage.distance_transform_edt(~mask) <= r


def shrink(mask, r):
    """Cells of mask at least r cells (Euclidean) inside it (more than r from its edge)."""
    if r <= 0:
        return mask.copy()
    m = np.pad(mask, 1)                   # outside the array counts as outside
    return (ndimage.distance_transform_edt(m) > r)[1:-1, 1:-1]


class Options:
    res = 0.05
    via_cost = 15.0          # cells (x res mm) a via is worth in track length
    plane_layer_cost = 2.5   # tracks on a plane layer cut the plane
    pres0 = 0.5              # first round's penalty per other net in the way
    pres_mult = 1.6
    pres_max = 1000.0
    hist_add = 0.3
    max_rounds = 150
    max_secs = None          # time for the negotiation, then the calmest round so far is kept (None: no limit)
    patience = 60            # rounds without a clearly calmer one, then the calmest so far is kept (None: all max_rounds)
    patience_drop = 0.2      # "clearly calmer": this much less in a fight than the round the wait began at
    near = 3.0               # mm the first search looks beyond its ends
    neighbour = 1.5          # mm: nets this close to a fight are rerouted with it
    neck_cost = 2.0          # a step of narrowed track costs this many of full width (0: never narrow)
    neck_width = None        # mm the wide nets narrow to (None: the Default class width, at least min track)
    fanout_neck = True       # nets of the Default width may narrow to the board's minimum track,
    fanout_mm = 1.0          # this close to their own pads (to get out from between fine-pitch pins)
    neck_switch = 6.0        # cells of track a change of width costs (else it flickers wide-narrow-wide)
    turn_cost = 0.3          # cells a change of direction costs (against needless zig-zags)
    spoke_cost = 0.0         # extra cost per step across a thermal pad's spoke and where it lands
                             # (up front, for every pad; 0: off, starved pads are fed after routing)
    min_spokes = 2           # KiCad's zone minimum spoke count
    log = print


class Router:
    def __init__(self, board, opt=None):
        self.b = b = board
        self.o = opt or Options()
        res = b.res
        rules = b.net_rule
        # neck-down: nets wider than the Default class may narrow to it (not below the
        # board's minimum track) where the full width doesn't fit
        min_track = b.d["rules"].get("min_track") or 0.0
        dflt = [r["width"] for r in rules if r.get("class") == "Default"]
        nw = self.o.neck_width or max(min_track, min(dflt) if dflt else min_track)
        # nets of the Default width (or narrower) may narrow to the board's minimum track
        # where pins are too close for their full width, as people route fine-pitch
        # chips (a board without a minimum track width: never)
        def neck(w):
            for x, near in ((nw, False), (min_track if self.o.fanout_neck and not self.o.neck_width else 0.0, True)):
                if self.o.neck_cost > 0 and 0 < x < w - 1e-6:
                    return x, near
            return None, False
        necks = [neck(r["width"]) for r in rules]
        self.neck_w = [x for x, _ in necks]
        # a net that would narrow only to get out from between fine-pitch pins: only
        # if a pad of it is boxed in for the full width, and only near those pads
        self.neck_cells = {}
        self._pads = {}           # (net, layer): its pads' shapes and centres
        for n, (x, near) in enumerate(necks):
            if x and near and not (len(b.terminals.get(n, [])) >= 2 or n in b.planes):
                self.neck_w[n] = None            # nothing to route
            elif x and near:
                cells = self.boxed_in(n, rules[n]["width"])
                if cells is None:
                    self.neck_w[n] = None
                else:
                    self.neck_cells[n] = cells
        # shape classes: each distinct (track width, clearance) and (via size, drill,
        # clearance). A net stamps, for each class, how close that shape's centre may
        # come to its copper, so "may a shape of class k go here" is one lookup.
        hole_clr = b.hole_clr
        h2h = b.d["rules"].get("hole_to_hole") or 0.25
        tcls = set()
        for n, r in enumerate(rules):
            tcls.add((r["width"], r["clearance"]))
            if self.neck_w[n]:
                tcls.add((self.neck_w[n], r["clearance"]))
        self.tcls = sorted(tcls) or [(0.25, 0.2)]
        self.vcls = sorted({(r["via_d"], r["via_drill"], r["clearance"]) for r in rules}) or [(0.6, 0.3, 0.2)]
        T = len(self.tcls)
        self.K = T + len(self.vcls)
        # (kind, half size, drill radius, clearance) of each class
        shapes = [("t", w / 2, 0.0, c) for w, c in self.tcls] + [("v", d / 2, dr / 2, c) for d, dr, c in self.vcls]

        def reach(a, k):
            """Closest two shapes' centres may be: a (one of the net's) and class k."""
            ka, ha, ra, ca = a
            kb, hb, rb, cb = shapes[k]
            d = ha + hb + max(ca, cb)                       # copper to copper
            if ka == "v":
                d = max(d, ra + hole_clr + hb)              # its hole to the other's copper
            if kb == "v":
                d = max(d, rb + hole_clr + ha)
            if ka == "v" and kb == "v":
                d = max(d, ra + rb + h2h)
            return (d + MARGIN) / res
        self.per = []
        for n, r in enumerate(rules):
            c = r["clearance"]
            w2 = self.neck_w[n] or r["width"]
            tw, tn = ("t", r["width"] / 2, 0.0, c), ("t", w2 / 2, 0.0, c)
            vs = ("v", r["via_d"] / 2, r["via_drill"] / 2, c)
            self.per.append({
                "kt": self.tcls.index((r["width"], c)), "kt2": self.tcls.index((w2, c)),
                "kv": T + self.vcls.index((r["via_d"], r["via_drill"], c)),
                # (the static arrays have the clearance taken off already; a drill
                # keeps the hole clearance, of which the copper clearance is taken off)
                "thr": r["width"] / 2 + MARGIN, "thr2": w2 / 2 + MARGIN,
                "vthr": max(r["via_d"] / 2, r["via_drill"] / 2 + hole_clr - c) + MARGIN,
                "rt": [reach(tw, k) for k in range(self.K)], "rt2": [reach(tn, k) for k in range(self.K)],
                "rv": [reach(vs, k) for k in range(self.K)],
            })
        lc = [self.o.plane_layer_cost if l in b.plane_layers else 1.0 for l in range(b.L)]
        self.grid = Grid(b, self.K, lc)

        # per net: own copper (always usable), no-via cells, plane targets
        self.nets = []
        for n, terms in sorted(b.terminals.items()):
            if n in b.planes:
                # kept even when all its pads are on the pour already: the other nets'
                # tracks cut the pour, and the pieces they cut off need stitching back
                pass
            elif len(terms) < 2:
                continue
            self.nets.append(n)
        self.own = {n: np.unique(np.concatenate([t["own"] for t in b.terminals[n]])) for n in self.nets}
        self.novia = {n: np.unique(np.concatenate([t["novia"] for t in b.terminals[n]])) for n in self.nets}
        self.via_pads = self.exposed_pads()
        # where a narrowed track lies inside its own copper
        self.own2 = {}
        for n in self.nets:
            if self.neck_w[n]:
                cs = [b.cells_in(g, l, shrink=self.neck_w[n] / 2)
                      for t in b.terminals[n] for it in t["items"] for l, g in it["geoms"].items()]
                self.own2[n] = np.unique(np.concatenate(cs)).astype(np.int32) if cs else None
        # hole to hole: a net's vias keep away from its own holes (other nets' are kept
        # further away by the copper clearance already)
        self.own_holes = {}
        for n in self.nets:
            vd = b.net_rule[n]["via_drill"]
            hs = []
            for t in b.terminals[n]:
                for it in t["items"]:
                    if it["kind"] == "via":
                        hs.append((it["at"], 0.15))
            for pd in b.d["pads"]:
                if pd["net"] == b.net_names[n] and "drill_r" in pd:
                    hs.append((pd["at"], pd["drill_r"]))
            self.own_holes[n] = np.unique(np.concatenate(
                [self.disk_cells(at, r + vd / 2 + h2h + MARGIN) for at, r in hs])) if hs else np.zeros(0, np.int32)
        self.h2h = h2h
        self.plane_mask = {}
        for n, (l, region) in b.planes.items():
            if n in self.nets:
                idx, X, Y = b.patch(region, 0)
                m = np.zeros(b.WH, bool)
                m[idx[shapely.contains_xy(region, X, Y)]] = True
                # where the pour can be at all: the region, minus other nets' fixed copper
                sl = slice(l * b.WH, (l + 1) * b.WH)
                ok = b.static_d(n, sl) >= self.per[n]["thr"]
                self.plane_mask[n] = (l, (m & ok).reshape(b.H, b.W))
        self.pour_shrink = {}
        for n in self.plane_mask:
            for l in range(b.L):
                zc, zw = b.pour_rules(n, l)
                need = zc + zw / 2
                have = b.net_rule[n]["clearance"] + b.net_rule[n]["width"] / 2
                self.pour_shrink[(n, l)] = max(0, int(math.ceil((need - have) / b.res - 1e-6)))
        # pads a pour joins with thermal spokes: a gap ring around them, joined only by spokes
        self.therm = {}
        for n in self.plane_mask:
            for l in range(b.L):
                gap, spoke, conn = b.thermal(n, l)
                pads = []
                for t in b.terminals[n]:
                    for it in t["items"]:
                        if it["kind"] != "pad" or l not in it["geoms"]:
                            continue
                        c = it.get("zone_connection", -1)
                        c = conn if c < 0 else c
                        if c == 3:
                            c = 1 if it["pth"] else 2
                        if c in (0, 1):
                            g = it["geoms"][l]
                            ctr = np.array(it["at"], float)
                            pts = np.array(g.exterior.coords if g.geom_type == "Polygon" else
                                           [q for gg in g.geoms for q in gg.exterior.coords])
                            spokes = []
                            if c == 1:
                                for k4 in range(4):        # KiCad's four spokes, at the pad's spoke angle
                                    a = math.radians(it["spokes"] + 90 * k4)
                                    u = np.array([math.cos(a), math.sin(a)])
                                    spokes.append((u, float(((pts - ctr) @ u).max())))
                            pads.append((b.cells_in(g, l) % b.WH, ctr, spokes))
                # the fill keeps its minimum width from the gap's edge too
                zw = b.pour_rules(n, l)[1]
                self.therm[(n, l)] = (int(math.ceil((gap + zw / 2) / b.res)), max(1, int(spoke / b.res)), pads)
        self.side_mask = {}
        for (n, l), region in b.side_pours.items():
            if n in self.plane_mask:
                idx, X, Y = b.patch(region, 0)
                m = np.zeros(b.WH, bool)
                m[idx[shapely.contains_xy(region, X, Y)]] = True
                sl = slice(l * b.WH, (l + 1) * b.WH)
                ok = b.static_d(n, sl) >= self.per[n]["thr"]
                self.side_mask[(n, l)] = (m & ok).reshape(b.H, b.W)
        self.reserve_spokes()
        # plane nets: which terminal a cell belongs to (a stub joins its own pad to the plane)
        self.term_of_cell = {}
        for n in self.plane_mask:
            m = {}
            for ti, t in enumerate(b.terminals[n]):
                for v in t["cells"]:
                    m[int(v)] = ti
            self.term_of_cell[n] = m
        self.paths = {}          # net -> list of node arrays
        self.failed = {}         # net -> list of terminals it couldn't reach
        self.stamped = set()
        self.hard = False        # the repair pass: other nets' copper is a wall

    def boxed_in(self, n, w):
        """Cells near net n's pads from which a track of width w can't get out (fixed
        copper all round, as between fine-pitch pins): within fanout_mm of them, on
        their layers. None if every pad lets the full width out."""
        b, R = self.b, self.o.fanout_mm
        out = []
        for t in b.terminals.get(n, []):
            for it in t["items"]:
                if it["kind"] == "pad" and not self.pad_escapes(n, it, w):
                    out += [b.cells_in(g.buffer(R), l) for l, g in it["geoms"].items()]
        return np.unique(np.concatenate(out)).astype(np.int32) if out else None

    def pad_escapes(self, n, it, w):
        """Can a track of width w get from pad item it (net n) to fanout_mm away from
        it, past the fixed copper around it, on one of its layers?"""
        b = self.b
        thr, R = w / 2 + MARGIN, self.o.fanout_mm
        eight = np.ones((3, 3), bool)
        for l, g in it["geoms"].items():
            gx0, gy0, gx1, gy1 = g.bounds
            xa, ya = b.to_cell(gx0 - R, gy0 - R)
            xb, yb = b.to_cell(gx1 + R, gy1 + R)
            xa, ya, xb, yb = max(xa, 0), max(ya, 0), min(xb, b.W - 1), min(yb, b.H - 1)
            h, wd = yb - ya + 1, xb - xa + 1
            ys, xs = np.mgrid[ya:yb + 1, xa:xb + 1]
            ok = (b.static_d(n, (l * b.WH + ys * b.W + xs).ravel()) >= thr).reshape(h, wd)

            def local(cells):
                m = np.zeros((h, wd), bool)
                xy = cells - l * b.WH
                yy, xx = xy // b.W - ya, xy % b.W - xa
                sel = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < wd)
                m[yy[sel], xx[sel]] = True
                return m
            ok |= local(b.cells_in(g, l, shrink=w / 2))      # its own copper
            start = local(b.cells_in(g, l)) & ok
            if not start.any():
                continue
            lab, _ = ndimage.label(ok, structure=eight)
            border = set(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist()) - {0}
            if border & set(lab[start].tolist()):
                return True
        return False

    def exposed_pads(self):
        """An exposed pad (a big SMD pad in the middle of its part's other pads, as
        under a QFN) that no track can leave, even at the narrowest width, between the
        pins around it: its net's vias may go inside it, well clear of its edge, as
        people route those. {net: [pad names]}; their inside is taken off the net's
        no-via cells."""
        b = self.b
        ring = {}
        for p in b.d["pads"]:
            ring.setdefault(p["ref"], []).append((p["num"], p["at"]))
        out = {}
        min_track = b.d["rules"].get("min_track") or 0.0
        for n in self.nets:
            vd = b.net_rule[n]["via_d"]
            w = min(b.net_rule[n]["width"], min_track) if min_track > 0 else b.net_rule[n]["width"]
            for t in b.terminals[n]:
                for it in t["items"]:
                    if it["kind"] != "pad" or it["pth"] or len(it["geoms"]) != 1:
                        continue
                    ref, num = it["name"].rsplit(".", 1)
                    others = [at for nm, at in ring.get(ref, []) if nm != num]
                    if len(others) < 3 or not shapely.MultiPoint(others).convex_hull.contains(shapely.Point(it["at"])):
                        continue
                    (l, g), = it["geoms"].items()
                    inner = b.cells_in(g, l, shrink=vd)     # the via's centre a via diameter inside
                    if not len(inner) or self.pad_escapes(n, it, w):
                        continue
                    self.novia[n] = np.setdiff1d(self.novia[n], inner).astype(self.novia[n].dtype)
                    out.setdefault(n, []).append(it["name"])
        return out

    def reserve_spokes(self):
        """Per thermal pad, the spokes that can land (the pour is there to land on)
        and each one's corridor: where another net's track centre or via would cut
        it, across the gap and on the bit of pour it lands on. With spoke_cost the
        corridors cost others extra from the start (as history)."""
        b, c = self.b, self.o.spoke_cost
        room = max(r["clearance"] + r["width"] / 2 for r in b.net_rule)
        res = b.res
        self.spoke_info = {}        # (net, layer) -> per ring (pad with spokes): [corridor node ids]
        for (n, l), (gap, spoke, pads) in self.therm.items():
            if l == self.plane_mask[n][0]:
                mask = self.plane_mask[n][1]
            elif (n, l) in self.side_mask:
                mask = self.side_mask[(n, l)]
            else:
                continue
            zw = b.pour_rules(n, l)[1]
            half = spoke * res / 2 + room
            rings = []
            for xy, ctr, spokes in pads:
                if not len(xy) or not spokes:
                    continue            # (as pour_pieces orders its rings)
                cors = []
                for u, ext in spokes:
                    land = ctr + u * (ext + gap * res + 1.5 * res)
                    x, y = b.to_cell(*land)
                    if not (0 <= x < b.W and 0 <= y < b.H and mask[y, x]) or \
                            not self.spoke_inside(n, l, ctr, u, ext, gap):
                        cors.append(None)           # never any pour to land on
                        continue
                    a0, a1 = ext, ext + gap * res + zw + room
                    R = int(math.ceil((a1 + half) / res)) + 1
                    cx, cy = b.to_cell(*ctr)
                    ys, xs = np.mgrid[max(0, cy - R):min(b.H, cy + R + 1), max(0, cx - R):min(b.W, cx + R + 1)]
                    dx, dy = b.cx(xs) - ctr[0], b.cy(ys) - ctr[1]
                    along = dx * u[0] + dy * u[1]
                    across = np.abs(-dx * u[1] + dy * u[0])
                    m = (along >= a0) & (along <= a1) & (across <= half)
                    cors.append((l * b.WH + ys[m] * b.W + xs[m]).astype(np.int32))
                rings.append((ctr, cors, xy))
                if c > 0:
                    for idx in cors:
                        if idx is not None:
                            self.grid.hist[idx] = np.maximum(self.grid.hist[idx], c)
            self.spoke_info[(n, l)] = (mask, rings)

    def pour_mask(self, n, l):
        return self.plane_mask[n][1] if l == self.plane_mask[n][0] else self.side_mask[(n, l)]

    def solid_fill(self, lab, group):
        """Where KiCad's fill is sure to be: well inside a piece (two cells in) of
        a group of at least 1 mm² (specks and slivers may not be filled, and a
        piece that joins only the pad is ignored by KiCad's count)."""
        g = group[lab]
        sizes = np.bincount(g.ravel())
        sizes[0] = 0
        big = sizes * self.b.res ** 2 >= 1.0
        return shrink(lab > 0, 2) & big[g], big

    def manual_spokes(self, n, l, cells, lab):
        """Net n's tracks from a pad (its xy cells on layer l) to the pour: a
        straight run starting on the pad and ending in the fill (KiCad counts
        each as a spoke)."""
        b = self.b
        cells = set(int(c) for c in cells)
        got = 0
        for p in self.paths.get(n, []):
            for q in (p, p[::-1]):
                if len(q) < 2 or int(q[0]) // b.WH != l or int(q[0]) % b.WH not in cells:
                    continue
                # to the end of the first straight run
                l0, x0, y0 = b.node_xy(int(q[0]))
                l1, x1, y1 = b.node_xy(int(q[1]))
                d = (x1 - x0, y1 - y0)
                if l1 != l or d == (0, 0):
                    continue
                k = 1
                while k + 1 < len(q):
                    la, xa, ya = b.node_xy(int(q[k]))
                    lb, xb, yb = b.node_xy(int(q[k + 1]))
                    if lb != l or (xb - xa, yb - ya) != d:
                        break
                    k += 1
                _, xe, ye = b.node_xy(int(q[k]))
                if lab[ye, xe]:
                    got += 1
        return got

    def starved(self, only=None):
        """Thermal pads KiCad will call starved: the pour reaches them (a spoke
        lands) but with fewer than min_spokes spokes, the net's own tracks from
        the pad into the pour counted as spokes too.
        [(net, layer, ring, spokes, the blocked spokes)]"""
        out = []
        for (n, l), (mask, rings) in self.spoke_info.items():
            if not rings or (only is not None and (n, l) != only):
                continue
            self.stamp(n, -1)
            lab, group, _ = self.pour_pieces(n, l, mask)
            status = self._spoke_cache[3]
            self.stamp(n, +1)
            sure, big = self.solid_fill(lab, group)
            for i, ((ctr, cors, xy), st) in enumerate(zip(rings, status)):
                if not any(a != 0 for a in st):
                    continue            # no pour near it: KiCad doesn't check it
                # (a spoke the model calls blocked may still be made by KiCad, which
                # then wants a second: such a pad counts as reached)
                got = sum(1 for a in st if a > 0 and big[group[a]])
                if got >= self.o.min_spokes:
                    continue
                got += self.manual_spokes(n, l, xy, sure)
                if got < self.o.min_spokes:
                    out.append((n, l, i, got, [k for k, a in enumerate(st) if a < 0 and cors[k] is not None]))
        return out

    def add_manual_spoke(self, n, l, i, want=1):
        """A short straight track of the pad's own net from the pad's centre into
        the pour (what KiCad counts as a spoke, and what a designer adds by hand):
        the shortest of the eight directions that reaches solid fill within
        1.5 mm, clear of everything else. True if one was added."""
        b, N = self.b, self.grid.N
        mask, rings = self.spoke_info[(n, l)]
        ctr, cors, xy = rings[i]
        p = self.per[n]
        self.stamp(n, -1)
        lab, group, _ = self.pour_pieces(n, l, mask)
        sure, _ = self.solid_fill(lab, group)
        sl = slice(l * b.WH, (l + 1) * b.WH)
        stat = b.static_d(n, sl)
        cx, cy = b.to_cell(*ctr)
        found = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                nodes = []
                x, y = cx, cy
                for _ in range(int(1.5 / b.res / math.hypot(dx, dy))):
                    if not (0 <= x < b.W and 0 <= y < b.H):
                        break
                    v = y * b.W + x
                    if stat[v] < p["thr"] or self.grid.occ[p["kt"] * N + l * b.WH + v] > 0:
                        break
                    nodes.append(l * b.WH + v)
                    if sure[y, x]:
                        found.append((len(nodes) * math.hypot(dx, dy), (dx, dy), nodes))
                        break
                    x, y = x + dx, y + dy
        found.sort(key=lambda f: f[0])
        took = []
        for f in found:
            took.append(f)
            if len(took) == want:
                break
        for f in took:
            sp = Path(f[2])
            sp.spoke = True                     # ends in the side pour, not on the plane
            self.paths.setdefault(n, []).append(sp)
        self.stamp(n, +1)
        return len(took)

    def feed_thermals(self):
        """Starved thermal pads: first a short track of their own into the pour
        (as a designer adds by hand); failing that, the nets crossing a blocked
        spoke move out of its way (walls up, the corridor walled too), kept only
        if nothing comes unjoined and the pad then has its spokes."""
        b, N, K = self.b, self.grid.N, self.K
        todo = self.starved()
        if not todo:
            return
        tracked = fed = 0
        for n, l, i, had, bl in todo:
            # a track is taken as enough: the model undercounts KiCad's spokes, and
            # KiCad adds each such track to the spokes it made
            if self.add_manual_spoke(n, l, i, self.o.min_spokes - had):
                tracked += 1
                continue
            mask, rings = self.spoke_info[(n, l)]
            ctr, cors, xy = rings[i]
            need = self.o.min_spokes - had
            cors = [cors[k] for k in bl]
            if need <= 0 or not cors:
                continue

            def crossing(idx):
                xy_ = np.unique(idx % b.WH)
                hit = []
                for m in self.nets:
                    if m == n:
                        continue
                    nodes, vias = self.net_nodes(m)
                    if np.isin(nodes, idx).any() or np.isin(vias, xy_).any():
                        hit.append(m)
                return hit
            ranked = sorted(((crossing(idx), idx) for idx in cors), key=lambda t: len(t[0]))
            take = ranked[:need]
            hit = sorted({m for h, _ in take for m in h})
            if not hit:
                continue
            wall = np.concatenate([idx for _, idx in take])
            missing = lambda: sum(len(self.failed.get(m, [])) for m in self.nets)
            before = missing()
            planes_ok = {m for m in self.plane_mask if self.plane_ok(m)}
            snap = {m: (list(self.paths.get(m, [])), list(self.failed.get(m, []))) for m in hit}
            for k in range(K):
                self.grid.occ[k * N + wall] += 1
            self.hard = True
            for m in hit:
                self.stamp(m, -1)
                self.paths[m] = [p for p in self.paths[m]
                                 if not (np.isin(p, wall).any() or np.isin(self.vias_of(p), wall % b.WH).any())]
                self.repair_net(m, 0.0)
                self.stamp(m, +1)
            self.hard = False
            for k in range(K):
                self.grid.occ[k * N + wall] -= 1
            clean = not any(len(self.conflicts(m)) for m in hit)
            now = [x for x in self.starved((n, l)) if x[:3] == (n, l, i)]
            # (moving them may also cut a plane pour off from stubs that land on it)
            if clean and missing() <= before and not now and all(self.plane_ok(m) for m in planes_ok):
                fed += 1
                continue
            for m in hit:                       # undo
                self.stamp(m, -1)
                self.paths[m], self.failed[m] = snap[m]
                self.stamp(m, +1)
        self.o.log("thermal pads short of spokes: %d; given a track %d, spokes cleared %d"
                   % (len(todo), tracked, fed))

    # ---- helpers ----
    def bbox_of(self, nodes, grow_mm):
        b = self.b
        xy = nodes % b.WH
        xs, ys = xy % b.W, xy // b.W
        g = int(grow_mm / b.res)
        return [int(xs.min()) - g, int(ys.min()) - g, int(xs.max()) + g, int(ys.max()) + g]

    def in_box(self, nodes, box):
        b = self.b
        xy = nodes % b.WH
        xs, ys = xy % b.W, xy // b.W
        return nodes[(xs >= box[0]) & (xs <= box[2]) & (ys >= box[1]) & (ys <= box[3])]

    def disk_cells(self, at, r_mm):
        """xy cells (layer 0 node ids) within r_mm of a point."""
        b = self.b
        cx, cy = b.to_cell(*at)
        R = int(math.ceil(r_mm / b.res))
        ys, xs = np.mgrid[-R:R + 1, -R:R + 1]
        m = (xs * b.res) ** 2 + (ys * b.res) ** 2 < r_mm ** 2
        xs, ys = xs[m] + cx, ys[m] + cy
        k = (xs >= 0) & (xs < b.W) & (ys >= 0) & (ys < b.H)
        return (ys[k] * b.W + xs[k]).astype(np.int32)

    def novia_for(self, n, paths):
        """No-via cells for net n's next search: its pads, its holes, and near
        the vias its paths so far have (on every layer)."""
        b = self.b
        xy = [self.own_holes[n]]
        vd = b.net_rule[n]["via_drill"]
        for p in paths:
            for v in self.vias_of(p):
                l, x, y = b.node_xy(v)
                xy.append(self.disk_cells((b.cx(x), b.cy(y)), vd + self.h2h + MARGIN))
        xy = np.unique(np.concatenate(xy)) if len(xy) > 1 else xy[0]
        layers = [xy + l * b.WH for l in range(b.L)] if len(xy) else []
        return np.concatenate([self.novia[n]] + layers).astype(np.int32)

    def close_vias(self, n, path):
        """Vias of one path closer to each other than the hole-to-hole rule allows."""
        b = self.b
        vs = self.vias_of(path)
        if len(vs) < 2:
            return []
        lim = (b.net_rule[n]["via_drill"] + self.h2h + MARGIN) / b.res
        xs, ys = vs % b.W, vs // b.W
        bad = []
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                if math.hypot(xs[i] - xs[j], ys[i] - ys[j]) < lim:
                    bad += [int(vs[i]), int(vs[j])]
        return bad

    def full(self):
        return [0, 0, self.b.W - 1, self.b.H - 1]

    def vias_of(self, path):
        if len(path) < 2:
            return np.zeros(0, np.int32)
        WH = self.b.WH
        a, c = path[:-1], path[1:]
        m = (a % WH) == (c % WH)
        return (a[m] % WH).astype(np.int32)

    def net_nodes(self, n, thin=False):
        """A net's track nodes and via xys (and with thin, which nodes are narrow)."""
        ps = self.paths.get(n, [])
        if not ps:
            z = np.zeros(0, np.int32)
            return (z, np.zeros(0, bool), z) if thin else (z, z)
        nodes = np.concatenate(ps).astype(np.int32)
        vias = np.concatenate([self.vias_of(p) for p in ps]).astype(np.int32)
        if thin:
            return nodes, np.concatenate([thin_of(p) for p in ps]), vias
        return nodes, vias

    # ---- one search ----
    def search(self, n, src, tgt, pres, grows, paths=()):
        """A* from src to tgt inside a box around tgt and the source cell nearest
        to it, grown by each of grows (mm; None = the whole board) until it works."""
        b, p = self.b, self.per[n]
        if not len(src) or not len(tgt):
            return None              # a pad with no cell on the board (off its edge): can't be reached
        tbox = self.bbox_of(tgt, 0)
        tx, ty = (tbox[0] + tbox[2]) / 2, (tbox[1] + tbox[3]) / 2
        sxy = src % b.WH
        k = int(np.argmin((sxy % b.W - tx) ** 2 + (sxy // b.W - ty) ** 2))
        novia = self.novia_for(n, paths)
        best = None
        for grow in grows:
            box = self.full() if grow is None else self.bbox_of(np.concatenate([src[k:k + 1], tgt]), grow)
            s = self.in_box(src, box)
            if not len(s):
                continue
            path = self.route_once(n, s, tgt, novia, box, tbox, pres)
            if path is not None:
                best = path
                # through someone else's copper: a bigger box may hold a way around
                if self.hard or self.clean(n, path):
                    break
        return best

    def clean(self, n, path):
        """Does a path (net n unstamped) keep clear of every other net's copper?"""
        b, p, N = self.b, self.per[n], self.grid.N
        th = thin_of(path)
        k = np.where(th, p["kt2"], p["kt"])
        if self.grid.occ[k * N + path].any():
            return False
        v = self.vias_of(path)
        return not len(v) or not any(self.grid.occ[p["kv"] * N + l * b.WH + v].any() for l in range(b.L))

    def route_once(self, n, src, tgt, novia, box, tbox, pres):
        """One A*; if the path has two of its own vias too close, forbid vias
        around them and search again."""
        b, p = self.b, self.per[n]
        for _ in range(4):
            path = self.grid.route(n, b.cc[n], p["kt"], p["kv"], p["thr"], p["vthr"], src, tgt, self.own[n], novia,
                                   box, tbox, self.o.via_cost, pres, hard=self.hard,
                                   kt2=p["kt2"], thr2=p["thr2"], own2=self.own2.get(n),
                                   neck_cost=self.o.neck_cost if self.neck_w[n] else 0.0,
                                   neck_switch=self.o.neck_switch, turn_cost=self.o.turn_cost,
                                   neck=self.neck_cells.get(n))
            if path is None:
                return None
            bad = self.close_vias(n, path)
            if not bad:
                return path
            extra = []
            for v in bad:
                x, y = v % b.W, v // b.W
                d = self.disk_cells((b.cx(x), b.cy(y)), b.net_rule[n]["via_drill"] / 2 + MARGIN)
                extra += [d + l * b.WH for l in range(b.L)]
            novia = np.concatenate([novia] + extra).astype(np.int32)
        return None

    def route_net(self, n, pres, keep=False):
        b, o = self.b, self.o
        terms = b.terminals[n]
        paths, failed = [], []
        kept = list(self.paths.get(n, [])) if keep else []
        if n in self.plane_mask:
            pl, mask = self.plane_mask[n]
            mask = self.main_island(n)
            good = self.joined_to_plane(n, kept, mask) if keep else set()
            # stubs that lost their way to the plane go (they'd only dangle)
            kept = [p for p in kept if int(p[-1]) in good]
            joined = [t["cells"] for t in terms
                      if self.in_island(t, pl, mask) or any(int(v) in good for v in t["cells"])]
            todo = [t for t in terms if not (self.in_island(t, pl, mask) or any(int(v) in good for v in t["cells"]))]
            # nearest to something joined first: then a pad can join a neighbour already joined
            # (a short track) instead of taking a via of its own
            while todo:
                jat = np.array([t["at"] for t in terms if any(np.array_equal(t["cells"], j) for j in joined)] or [[0, 0]])
                t = min(todo, key=lambda t: float(np.min(np.hypot(*(jat - t["at"]).T))))
                todo.remove(t)
                done = np.concatenate(paths + kept + joined) if paths or kept or joined else np.zeros(0, np.int32)
                found = None
                for grow in (o.near, 4 * o.near, None):
                    box = self.full() if grow is None else self.bbox_of(t["cells"], grow)
                    x0, y0, x1, y1 = max(box[0], 0), max(box[1], 0), min(box[2], b.W - 1), min(box[3], b.H - 1)
                    sub = mask[y0:y1 + 1, x0:x1 + 1]
                    ys, xs = np.nonzero(sub)
                    tgt = (pl * b.WH + (ys + y0) * b.W + (xs + x0)).astype(np.int32)
                    tgt = np.concatenate([tgt, self.in_box(done, box)]) if len(done) else tgt
                    if not len(tgt):
                        continue
                    path = self.route_once(n, t["entry"], tgt, self.novia_for(n, kept + paths),
                                           [x0, y0, x1, y1], [x0, y0, x1, y1], pres)
                    if path is not None:
                        found = path
                        if self.hard or self.clean(n, path):
                            break
                if found is not None:
                    paths.append(found)
                    joined.append(t["cells"])
                else:
                    failed.append(t)
            return kept + paths, failed

        # Prim: grow a tree from the first terminal, always to the nearest one left
        left = list(range(1, len(terms)))
        tree_nodes = [terms[0]["entry"]]
        tree_at = [terms[0]["at"]]
        while left:
            ats = np.array(tree_at)
            best = min(left, key=lambda i: np.min(np.hypot(*(ats - terms[i]["at"]).T)))
            left.remove(best)
            t = terms[best]
            src = np.concatenate(tree_nodes)
            path = self.search(n, src, t["entry"], pres, (o.near, 4 * o.near, None), paths)
            if path is None:
                failed.append(t)
                continue
            paths.append(path)
            tree_nodes += [path, t["entry"]]
            tree_at.append(t["at"])
            # the path's points join the tree too (a later pad may be nearest to it)
            sel = path[:: max(1, len(path) // 8)]
            for v in sel:
                l, x, y = b.node_xy(v)
                tree_at.append(np.array([b.cx(x), b.cy(y)]))
        return paths, failed

    # ---- planes ----
    def pour_pieces(self, n, l, mask):
        """Net n's pour on layer l as KiCad will fill it, with every other net's
        copper as it is now (net n must not be stamped): returns (labels, group of
        each label, groups a pad's spokes reach). Pieces joined by a thermal pad's
        spokes are in one group."""
        b = self.b
        kt = self.per[n]["kt"]
        occ = self.grid.occ[kt * self.grid.N + l * b.WH: kt * self.grid.N + (l + 1) * b.WH].reshape(b.H, b.W)
        free = self.filled(n, l, mask & (occ == 0))
        clear = mask & (occ == 0)            # where a spoke may cross (no other copper)
        # (free is where the centre of the fill's minimum-width brush can be; the gap
        # ring below includes that half width, so the slivers KiCad can't fill go too)
        gap, spoke, pads = self.therm.get((n, l), (0, 1, []))
        rings = []
        for xy, ctr, spokes in pads:
            if not len(xy):
                continue
            ys, xs = xy // b.W, xy % b.W
            R = gap + 3
            y0, x0 = max(0, ys.min() - R), max(0, xs.min() - R)
            y1, x1 = min(b.H, ys.max() + R + 1), min(b.W, xs.max() + R + 1)
            loc = np.zeros((y1 - y0, x1 - x0), bool)
            loc[ys - y0, xs - x0] = True
            inner = grow(loc, gap)
            free[y0:y1, x0:x1] &= ~inner
            if spokes:
                rings.append((ctr, spokes))
        lab, k = ndimage.label(free)
        res = b.res
        sw = spoke * res
        reach = 4 * res + self.b.pour_rules(n, l)[1]     # how far past the gap a spoke may meet the fill

        def cell(pnt):
            x, y = b.to_cell(*pnt)
            return (x, y) if 0 <= x < b.W and 0 <= y < b.H else None

        def spoke_label(ctr, u, ext):
            """The piece a spoke from this pad edge in direction u lands in (0: none):
            the strip across the gap must be clear and land fully on one piece."""
            v = np.array([-u[1], u[0]])
            land = set()
            for t in np.linspace(-sw / 2, sw / 2, 5):
                for dist in np.arange(ext, ext + gap * res + 1e-9, res):     # across the gap
                    c = cell(ctr + u * dist + v * t)
                    if c is None or not clear[c[1], c[0]]:
                        return 0
                # then on until it meets the fill (KiCad's spoke runs into it; the
                # model's fill edge may sit a cell or so short of KiCad's)
                hit = 0
                for dist in np.arange(ext + gap * res + res, ext + gap * res + reach + 1e-9, res):
                    c = cell(ctr + u * dist + v * t)
                    if c is None or not clear[c[1], c[0]]:
                        break
                    if lab[c[1], c[0]]:
                        hit = int(lab[c[1], c[0]])
                        break
                if not hit:
                    return 0
                land.add(hit)
            return land.pop() if len(land) == 1 else 0
        parent = list(range(k + 1))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def blocked(ctr, u, ext):
            """No spoke, but pour beyond the gap it would have reached (KiCad counts
            such a spoke as wanted and missing; one with no pour to reach isn't)."""
            for dist in np.arange(ext + gap * res, ext + gap * res + 1.0, res):
                c = cell(ctr + u * dist)
                if c is not None and lab[c[1], c[0]]:
                    return True
            return False
        landed, status = [], []
        for ctr, spokes in rings:
            st = []
            for u, ext in spokes:
                a = spoke_label(ctr, u, ext)
                st.append(a if a else (-1 if blocked(ctr, u, ext) else 0))
            status.append(st)
            ls = [a for a in st if a > 0]
            landed.append(ls)
            for a in ls[1:]:
                parent[find(a)] = find(ls[0])
        group = np.array([find(i) for i in range(k + 1)])
        spoked = {int(group[a]) for ls in landed for a in ls}
        self._spoke_cache = (n, l, landed, status)
        return lab, group, spoked

    def main_island(self, n):
        """Where net n's plane pour joins up: the biggest group of pieces it will
        fill (net n must not be stamped)."""
        pl, mask = self.plane_mask[n]
        lab, group, _ = self.pour_pieces(n, pl, mask)
        if len(group) < 2:
            return np.zeros_like(mask)
        g = group[lab]
        sizes = np.bincount(g.ravel())
        sizes[0] = 0
        return g == int(np.argmax(sizes))

    def filled(self, n, l, free):
        """Where the pour will really be: the free cells, shrunk by what the zone's
        own clearance and minimum width take beyond the net's rules."""
        k = self.pour_shrink.get((n, l), 0)
        return shrink(free, k) if k else free

    def in_island(self, t, pl, island):
        """Is terminal t joined to the island on layer pl: its copper in it, or
        (a pad with a thermal gap) spokes' room of the island just outside the gap."""
        b = self.b
        c = t["cells"]
        c = c[(c // b.WH) == pl] % b.WH
        if not len(c):
            return False
        if island.ravel()[c].any():
            return True
        # a thermal pad: does one of its spokes land on the island?
        gap, spoke, pads = self.therm.get((t["net"], pl), (0, 1, []))
        res = b.res
        sw = spoke * res
        for it in t["items"]:
            if it["kind"] != "pad" or pl not in it["geoms"]:
                continue
            ctr = np.array(it["at"], float)
            for xy, pc, spokes in pads:
                if not np.allclose(pc, ctr):
                    continue
                for u, ext in spokes:
                    if not self.spoke_inside(t["net"], pl, ctr, u, ext, gap):
                        continue
                    v = np.array([-u[1], u[0]])
                    ok = True
                    for tt in np.linspace(-sw / 2, sw / 2, 5):
                        x, y = b.to_cell(*(ctr + u * (ext + gap * res + 1.5 * res) + v * tt))
                        if not (0 <= x < b.W and 0 <= y < b.H and island[y, x]):
                            ok = False
                            break
                    if ok:
                        return True
        return False

    def spoke_inside(self, n, l, ctr, u, ext, gap):
        """Does the spoke's way across the gap lie in the pour's area? KiCad clips
        the spokes to the zone outline, so a pad just outside it (closer than the
        thermal gap) gets spokes that start on the pour and never reach the pad."""
        b = self.b
        mask = self.pour_mask(n, l)
        for dist in np.arange(ext + b.res / 2, ext + gap * b.res, b.res):
            x, y = b.to_cell(*(ctr + u * dist))
            if not (0 <= x < b.W and 0 <= y < b.H and mask[y, x]):
                return False
        return True

    def joined_to_plane(self, n, paths, island):
        """Nodes known joined to the plane: the island, and the stubs that land on it
        (or on a stub that does)."""
        b = self.b
        pl = self.plane_mask[n][0]
        terms = b.terminals[n]
        good = set((np.nonzero(island.ravel())[0] + pl * b.WH).tolist())
        for t in terms:
            if self.in_island(t, pl, island):
                good.update(int(v) for v in t["cells"])
        pending = list(range(len(paths)))
        changed = True
        while changed and pending:
            changed = False
            for i in list(pending):
                if int(paths[i][-1]) in good:
                    good.update(int(v) for v in paths[i])
                    ti = self.term_of_cell[n].get(int(paths[i][0]))
                    if ti is not None:                  # the pad it starts from is joined now
                        good.update(int(v) for v in terms[ti]["cells"])
                    pending.remove(i)
                    changed = True
        return good

    def plane_ok(self, n):
        """Is every terminal of plane net n joined to the pour's main island?"""
        b = self.b
        pl = self.plane_mask[n][0]
        self.stamp(n, -1)
        isl = self.main_island(n).ravel()
        self.stamp(n, +1)
        paths = self.paths.get(n, [])
        good = self.joined_to_plane(n, paths, isl.reshape(b.H, b.W))
        if any(int(p[-1]) not in good for p in paths if not getattr(p, "spoke", False)):
            return False
        for t in b.terminals[n]:
            if not self.in_island(t, pl, isl.reshape(b.H, b.W)) and not any(int(v) in good for v in t["cells"]):
                return False
        return True

    # ---- the rounds ----
    def stamp(self, n, sign):
        nodes, thin, vias = self.net_nodes(n, thin=True)
        p = self.per[n]
        self.grid.stamp(nodes, thin if thin.any() else None, vias, p["rt"], p["rt2"], p["rv"], sign)

    def conflicts(self, n, hadd=0.0):
        nodes, thin, vias = self.net_nodes(n, thin=True)
        p = self.per[n]
        return self.grid.conflicts(nodes, thin if thin.any() else None, vias, p["kt"], p["kt2"], p["kv"], hadd)

    def run(self):
        o, b = self.o, self.b
        t0 = time.time()

        def span(n):
            ats = np.array([t["at"] for t in b.terminals[n]])
            return np.ptp(ats[:, 0]) + np.ptp(ats[:, 1])
        todo = sorted(self.nets, key=span)
        pres = o.pres0
        self.history = []
        self.fights = []         # per round: the nets in a fight
        hot_xy = None            # near last round's fights (xy cells)
        best = None              # (cells in a fight, round, paths, failed) of the calmest round so far
        wait = None              # (cells, round) the patience counts from: the last clearly calmer round
        self.out_of_time = False
        self.stalled = False
        for rnd in range(1, o.max_rounds + 1):
            for n in todo:
                old = self.paths.get(n, [])
                kept, keep = old, False
                if n in self.plane_mask and old and hot_xy is not None:
                    # a plane net's stubs are each on their own: redo only those in a
                    # fight or near one, keep the rest
                    c = set(self.conflicts(n).tolist())
                    kept = [p for p in old if not hot_xy[p % b.WH].any() and not any(int(v) in c for v in p)]
                    keep = True
                if n in self.stamped:
                    self.stamp(n, -1)
                    self.stamped.discard(n)
                self.paths[n] = kept
                self.paths[n], self.failed[n] = self.route_net(n, pres, keep=keep)
                self.stamp(n, +1)
                self.stamped.add(n)
            bad = {}
            for n in self.nets:
                c = self.conflicts(n)
                if len(c):
                    bad[n] = len(c)
            for n in self.plane_mask:
                if n not in bad and not self.plane_ok(n):
                    bad[n] = 1                       # cut off from its plane: route it again
                    o.log("  %s: a stub is cut off from the plane, again" % b.net_names[n])
            self.history.append((rnd, len(bad), sum(bad.values())))
            self.fights.append(sorted(bad))
            o.log("round %2d: %3d nets rerouted, %3d still in a fight (%d cells)  %.0fs"
                  % (rnd, len(todo), len(bad), sum(bad.values()), time.time() - t0))
            if not bad:
                break
            cells = sum(bad.values())
            if best is None or cells < best[0]:
                best = (cells, rnd, {m: list(self.paths.get(m, [])) for m in self.nets},
                        {m: list(self.failed.get(m, [])) for m in self.nets})
            if wait is None or cells < wait[0] * (1 - o.patience_drop):
                # the fights swing from round to round: only a clear drop restarts the wait
                wait = (cells, rnd)
            stop = None
            if o.max_secs is not None and time.time() - t0 > o.max_secs:
                self.out_of_time = True
                stop = "out of time after %d rounds" % rnd
            elif o.patience and rnd - wait[1] >= o.patience:
                # a fight can go on for many rounds and still be won (one took 49 rounds
                # without a calmer one), but hardly one that has stopped getting calmer
                self.stalled = True
                stop = "no clearly calmer round in %d rounds" % o.patience
            elif rnd == o.max_rounds:
                stop = "%d rounds" % rnd
            if stop:
                # the calmest round is what settles
                if cells > best[0]:
                    for m in list(self.stamped):
                        self.stamp(m, -1)
                    self.paths, self.failed = best[2], best[3]
                    for m in self.nets:
                        self.stamp(m, +1)
                    self.stamped = set(self.nets)
                o.log("%s: kept round %d's routes (%d cells in a fight)" % (stop, best[1], best[0]))
                break
            hot = []
            for n in bad:
                hot.append(self.conflicts(n, o.hist_add))
            # capped: beyond this, float32 path costs lose the step lengths (and at 1.6^200, overflow)
            pres = min(pres * o.pres_mult, o.pres_max)
            hot_xy = self.near_mask(np.concatenate(hot), o.neighbour)
            todo = sorted(set(bad) | self.near_nets(hot_xy), key=span)
        self.settle()
        return self

    def near_mask(self, nodes, mm):
        """xy cells within mm of any of these nodes (any layer)."""
        b = self.b
        m = np.zeros(b.WH, bool)
        if len(nodes):
            m[np.unique(nodes % b.WH)] = True
            m = grow(m.reshape(b.H, b.W), max(1, int(mm / b.res))).ravel()
        return m

    def near_nets(self, m):
        """Nets with copper on these xy cells."""
        b = self.b
        out = set()
        for n in self.nets:
            nodes_n, _ = self.net_nodes(n)
            if len(nodes_n) and m[nodes_n % b.WH].any():
                out.add(n)
        return out

    def settle(self):
        """Take out the connections still in a fight, the most fought-over first,
        until nothing overlaps; then reconnect what that split, with other nets'
        copper as a wall."""
        self.hot = []            # where the leftover fights were: node ids
        while True:
            worst = None
            for n in self.nets:
                c = self.conflicts(n)
                if not len(c):
                    continue
                cs = set(int(v) for v in c)
                for i, p in enumerate(self.paths.get(n, [])):
                    k = sum(1 for v in p if int(v) in cs) + sum(1 for v in self.vias_of(p) for l in range(self.b.L)
                                                                if int(v) + l * self.b.WH in cs)
                    if k and (worst is None or k > worst[0]):
                        worst = (k, n, i, c)
            if worst is None:
                break
            _, n, i, c = worst
            self.hot.append(c)
            self.stamp(n, -1)
            del self.paths[n][i]
            self.stamp(n, +1)
        self.repair()

    def components(self, n):
        """Net n's terminals and paths grouped by what touches what: a list of
        (terminal indexes, path indexes)."""
        terms, paths = self.b.terminals[n], self.paths.get(n, [])
        T = len(terms)
        parent = list(range(T + len(paths)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        owner = {}
        for ti, t in enumerate(terms):
            for v in t["cells"]:
                owner.setdefault(int(v), set()).add(ti)
        for pi, p in enumerate(paths):
            for v in p:
                owner.setdefault(int(v), set()).add(T + pi)
        for pi, p in enumerate(paths):
            for end in (int(p[0]), int(p[-1])):
                for o in owner.get(end, ()):
                    parent[find(o)] = find(T + pi)
        groups = {}
        for i in range(T + len(paths)):
            groups.setdefault(find(i), ([], []))
            (groups[find(i)][0] if i < T else groups[find(i)][1]).append(i if i < T else i - T)
        return [g for g in groups.values() if g[0]]

    def group_nodes(self, n, g):
        terms, paths = self.b.terminals[n], self.paths.get(n, [])
        return np.concatenate([terms[i]["entry"] for i in g[0]] + [paths[i] for i in g[1]]).astype(np.int32)

    def repair(self):
        """Join each net's pieces with walls up: a path found here can't overlap anything."""
        self.hard = True
        for n in self.nets:
            self.stamp(n, -1)
            self.repair_net(n, 0.0)
            self.stamp(n, +1)
        self.hard = False
        self.endgame()
        self.feed_thermals()
        self.stitch_islands()

    def repair_net(self, n, pres):
        """Join net n's pieces (n unstamped); what can't be joined goes to failed[n]."""
        o = self.o
        if n in self.plane_mask:
            self.paths[n], self.failed[n] = self.route_net(n, pres, keep=True)
            return
        self.failed[n] = []
        before = None
        while True:
            comps = self.components(n)
            if len(comps) < 2:
                return
            if before is not None and len(comps) >= before:
                # the last path joined nothing (it shouldn't happen): don't search forever
                self.paths[n].pop()
                for g in comps[1:]:
                    self.failed[n].append({"names": sum((self.b.terminals[n][i]["names"] for i in g[0]), [])})
                return
            before = len(comps)
            comps.sort(key=lambda g: -len(g[0]))
            main = self.group_nodes(n, comps[0])
            for g in comps[1:]:
                path = self.search(n, main, self.group_nodes(n, g), pres, (o.near, 4 * o.near, None),
                                   self.paths[n])
                if path is not None:
                    self.paths[n].append(path)
                    break
            else:
                for g in comps[1:]:
                    self.failed[n].append({"names": sum((self.b.terminals[n][i]["names"] for i in g[0]), [])})
                return

    def endgame(self, tries=3, pres=30.0):
        """Rip-up and retry for what the rounds left unjoined: route the missing
        connection through the others (at a price per net in the way), take out
        the connections it crosses, and join those again around it, walls up.
        Kept only if that leaves fewer connections missing."""
        def missing():
            return sum(len(self.failed.get(n, [])) for n in self.nets)
        start = missing()
        for _ in range(tries):
            todo = [n for n in self.nets if self.failed.get(n)]
            if not todo:
                break
            better = False
            for n in todo:
                if not self.failed.get(n):
                    continue
                before = missing()
                planes_ok = {m for m in self.plane_mask if self.plane_ok(m)}
                snap = {m: list(self.paths.get(m, [])) for m in self.nets}
                fsnap = {m: list(self.failed.get(m, [])) for m in self.nets}
                # n's missing piece, through the others
                self.stamp(n, -1)
                self.hard = False
                self.repair_net(n, pres)
                self.stamp(n, +1)
                # the others in its way give up the connections it crosses, then rejoin
                self.hard = True
                hit = [m for m in self.nets if m != n and len(self.conflicts(m))]
                for m in hit:
                    c = set(self.conflicts(m).tolist())
                    self.stamp(m, -1)
                    self.paths[m] = [p for p in self.paths[m]
                                     if not any(int(v) in c for v in p)
                                     and not any(int(v) + l * self.b.WH in c
                                                 for v in self.vias_of(p) for l in range(self.b.L))]
                    self.stamp(m, +1)
                # n itself, if it still crosses someone (a net that was in the way of the
                # way it took), drops that and rejoins too
                for m in [n] + hit:
                    self.stamp(m, -1)
                    c = set(self.conflicts_unstamped(m))
                    if c:
                        self.paths[m] = [p for p in self.paths[m] if not any(int(v) in c for v in p)]
                    self.repair_net(m, 0.0)
                    self.stamp(m, +1)
                self.hard = False
                clean = not any(len(self.conflicts(m)) for m in [n] + hit)
                if clean and missing() < before and all(self.plane_ok(m) for m in planes_ok):
                    better = True
                    self.o.log("  endgame: %s joined (%d nets rerouted around it)" % (self.b.net_names[n], len(hit)))
                    continue
                self.o.log("  endgame: %s not joined (crossed %s; clean %s, missing %d -> %d)"
                           % (self.b.net_names[n], [self.b.net_names[m] for m in hit], clean, before, missing()))
                # undo
                for m in self.nets:
                    if self.paths.get(m, []) is not snap[m]:
                        self.stamp(m, -1)
                        self.paths[m] = snap[m]
                        self.failed[m] = fsnap[m]
                        self.stamp(m, +1)
            if not better:
                break
        if missing() < start:
            self.o.log("endgame: %d -> %d connections missing" % (start, missing()))

    def conflicts_unstamped(self, n):
        """Nodes of net n's paths (n not stamped) that another net's copper covers."""
        b, p, N = self.b, self.per[n], self.grid.N
        out = []
        for path in self.paths.get(n, []):
            th = thin_of(path)
            k = np.where(th, p["kt2"], p["kt"])
            occ = self.grid.occ[k * N + path]
            out += path[occ > 0].tolist()
            v = self.vias_of(path)
            for l in range(b.L):
                vv = l * b.WH + v
                out += vv[self.grid.occ[p["kv"] * N + vv] > 0].tolist()
        return out

    def stitch_islands(self):
        """A plane net's pour on another layer breaks into pieces around the other
        nets' tracks. A piece with none of the net's pads or tracks in it is
        connected to nothing: give it a via down to the plane."""
        b = self.b
        self.stitches = 0
        for (n, l), mask in self.side_mask.items():
            pl = self.plane_mask[n][0]
            p = self.per[n]
            self.stamp(n, -1)
            N = self.grid.N
            occ = lambda k, layer: self.grid.occ[k * N + layer * b.WH: k * N + (layer + 1) * b.WH].reshape(b.H, b.W)
            lab, group, spoked = self.pour_pieces(n, l, mask)
            landed = self._spoke_cache[2]       # (main_island below recomputes the cache)
            k = len(group) - 1
            if k:
                island = self.main_island(n)
                # joined: one of the net's vias or through-hole pads is in it (a track
                # or a pad in it proves nothing: it may lead only to this piece)
                vxy = [self.vias_of(pt) for pt in self.paths.get(n, [])]
                vxy = np.concatenate(vxy).astype(np.int64) if vxy else np.zeros(0, np.int64)
                joined = set(group[np.unique(lab.ravel()[vxy])].tolist()) if len(vxy) else set()
                pth_at = []
                for t in b.terminals[n]:
                    for it in t["items"]:
                        if it["kind"] == "via" or it.get("pth"):
                            pth_at.append(np.array(it["at"], float))
                            c = t["cells"][(t["cells"] // b.WH) == l] % b.WH
                            joined.update(group[np.unique(lab.ravel()[c])].tolist())
                # a thermal through-hole pad joins the pieces its spokes reach
                rings = [ctr for xy, ctr, spokes in self.therm.get((n, l), (0, 1, []))[2] if len(xy) and spokes]
                for ctr, ls in zip(rings, landed):
                    if any(np.allclose(ctr, a) for a in pth_at):
                        joined.update(int(group[a]) for a in ls)
                joined.discard(0)
                # where a via of this net may go: fixed copper, other nets, own holes and vias
                ok = np.ones(b.WH, bool)
                for layer in range(b.L):
                    sl = slice(layer * b.WH, (layer + 1) * b.WH)
                    ok &= b.static_d(n, sl) >= p["vthr"]
                    ok &= b.via_block[sl] == 0
                    ok &= occ(p["kv"], layer).ravel() == 0
                nv = self.novia_for(n, self.paths.get(n, []))
                ok[nv % b.WH] = False
                ok &= island.ravel()
                ok = ok.reshape(b.H, b.W)
                for i in range(1, k + 1):
                    if group[i] != i or i in joined:
                        continue
                    piece = group[lab] == i
                    # a speck smaller than the via itself: KiCad's fill doesn't keep it
                    # (and a via there joins nothing), so it's left to be removed
                    if piece.sum() * b.res ** 2 < math.pi * (b.net_rule[n]["via_d"] / 2) ** 2:
                        continue
                    cand = piece & ok
                    if not cand.any():
                        continue
                    depth = ndimage.distance_transform_edt(piece)     # as far inside the piece as it goes
                    v = int(np.argmax(np.where(cand, depth, -1)))
                    self.paths.setdefault(n, []).append(np.array([l * b.WH + v, pl * b.WH + v], np.int32))
                    self.stitches += 1
                    # the next via keeps its distance from this one
                    ok.ravel()[self.disk_cells((b.cx(v % b.W), b.cy(v // b.W)),
                                               b.net_rule[n]["via_d"] + p["thr"] * 2) ] = False
            self.stamp(n, +1)
        if self.stitches:
            self.o.log("stitched %d pour islands to their plane" % self.stitches)

    # ---- output ----
    def segments(self, n, cands=None):
        """Tracks (merged straight runs) and vias of a routed net, in mm. With
        cands: adds the pieces that would carry its ends on to pad centres."""
        b = self.b
        r = b.net_rule[n]
        tracks, vias = [], []
        # where a path starts or ends on another path, that one must have a segment end
        # too (KiCad counts a track end on the middle of another as dangling)
        ends = set()
        for path in self.paths.get(n, []):
            ends.add(int(path[0]))
            ends.add(int(path[-1]))
        for path in self.paths.get(n, []):
            first = len(tracks)
            pts = [b.node_xy(v) for v in path]
            th = thin_of(path)
            # a step is narrow if either end is (a width change sits on a wide node)
            narrow = lambda i: bool(th[i] or th[i + 1])
            i = 0
            while i < len(pts) - 1:
                l, x, y = pts[i]
                l2, x2, y2 = pts[i + 1]
                if (x, y) == (x2, y2):
                    vias.append((x, y))
                    i += 1
                    continue
                d = (x2 - x, y2 - y)
                nar = narrow(i)
                j = i + 1
                while j < len(pts) - 1:
                    la, xa, ya = pts[j]
                    lb, xb, yb = pts[j + 1]
                    if lb != l or (xb - xa, yb - ya) != d or int(path[j]) in ends or narrow(j) != nar:
                        break
                    j += 1
                lj, xj, yj = pts[j]
                tracks.append({"net": b.net_names[n], "layer": b.layers[l],
                               "a": [round(b.cx(x), 4), round(b.cy(y), 4)],
                               "b": [round(b.cx(xj), 4), round(b.cy(yj), 4)],
                               "w": self.neck_w[n] if nar else r["width"]})
                i = j
            # the search stops at the pad's edge: each end may go on to the pad's centre
            # (result() checks those pieces' clearance)
            if cands is not None and len(tracks) > first:
                for t, end, other in ((tracks[first], "a", "b"), (tracks[-1], "b", "a")):
                    at = self.pad_centre(n, t["layer"], t[end])
                    if at is not None:
                        ways = [w for w in [self.into_pad(t[other], t[end], at)] if w] + \
                            self.octilinear(t[end], at) + [[t[end], at]]
                        uniq = []
                        for w in ways:
                            if w not in uniq:
                                uniq.append(w)
                        cands.append((n, [[dict(t, a=p, b=q) for p, q in zip(w, w[1:])] for w in uniq]))
        out_v = [{"net": b.net_names[n], "at": [round(b.cx(x), 4), round(b.cy(y), 4)],
                  "d": r["via_d"], "drill": r["via_drill"], "layers": [b.layers[0], b.layers[-1]]}
                 for x, y in sorted(set(vias))]
        return tracks, out_v

    def pad_centre(self, n, layer, e):
        """The centre of the pad of net n that a track end e lies in, if that is
        somewhere else in the pad; else None."""
        pads = self._pads.get((n, layer))
        if pads is None:
            l = self.b.li[layer]
            pads = self._pads[(n, layer)] = [
                (it["geoms"][l], it["at"]) for t in self.b.terminals.get(n, [])
                for it in t["items"] if it["kind"] == "pad" and l in it["geoms"]]
        pe = shapely.Point(e)
        for g, at in pads:
            if g.buffer(1e-6).contains(pe):
                c = [round(at[0], 4), round(at[1], 4)]
                if math.dist(e, c) >= 1e-3 and g.contains(shapely.Point(c)):
                    return c
                return None
        return None

    @staticmethod
    def into_pad(prev, e, c):
        """The way from track end e (reached from prev) on to the pad centre c, as a
        designer draws it: straight on, then 45 degrees to the centre (or, where the
        centre is only off the line by the grid's rounding, a jog too small to see)."""
        ux, uy = e[0] - prev[0], e[1] - prev[1]
        L = math.hypot(ux, uy)
        if L < 1e-9:
            return None
        ux, uy = ux / L, uy / L
        dx, dy = c[0] - e[0], c[1] - e[1]
        s, h = dx * ux + dy * uy, dy * ux - dx * uy        # along the track, and off it
        r4 = lambda p: [round(p[0], 4), round(p[1], 4)]
        if s <= 1e-6:
            return None                                    # the centre is behind: no straight on
        if abs(h) <= 0.06:
            q = r4((e[0] + s * ux, e[1] + s * uy))
        elif s > abs(h):
            q = r4((e[0] + (s - abs(h)) * ux, e[1] + (s - abs(h)) * uy))
        else:
            return None
        return [e, q, c] if math.dist(q, c) > 1e-4 and math.dist(q, e) > 1e-4 else [e, c]

    @staticmethod
    def octilinear(e, c):
        """The two ways from e to c of one straight and one 45-degree leg."""
        dx, dy = c[0] - e[0], c[1] - e[1]
        ax, ay = abs(dx), abs(dy)
        if min(ax, ay) < 1e-4 or abs(ax - ay) < 1e-4:
            return []                              # one leg does it: the straight way
        sx, sy = math.copysign(1, dx), math.copysign(1, dy)
        m = min(ax, ay)
        diag = (e[0] + sx * m, e[1] + sy * m)      # diagonal first
        strt = (c[0] - sx * m, c[1] - sy * m)      # straight first
        r4 = lambda p: [round(p[0], 4), round(p[1], 4)]
        return [[e, r4(strt), c], [e, r4(diag), c]]

    def to_centres(self, tracks, vias, cands):
        """The pieces from track ends on to their pads' centres that keep every
        clearance: clear of other nets' copper, holes, keepouts and the board edge."""
        b = self.b
        ok = []
        if not cands:
            return ok
        obst = {}                          # layer: [(shape, net, clearance, fixed)]
        for l, g, m, c, fixed in b.items:
            obst.setdefault(l, []).append((g, m, c, fixed))
        for t in tracks:
            l = b.li[t["layer"]]
            obst.setdefault(l, []).append((shapely.LineString([t["a"], t["b"]]).buffer(t["w"] / 2),
                                           b.ni[t["net"]], b.net_clr(b.ni[t["net"]]), False))
        for v in vias:
            g = shapely.Point(v["at"]).buffer(v["d"] / 2)
            # its hole too: with a narrow ring, the hole clearance reaches further than the copper's
            hole = shapely.Point(v["at"]).buffer(v["drill"] / 2) if v.get("drill") else None
            for l in range(b.L):
                obst.setdefault(l, []).append((g, b.ni[v["net"]], b.net_clr(b.ni[v["net"]]), False))
                if hole is not None:
                    obst[l].append((hole, b.ni[v["net"]], b.hole_clr, True))
        trees = {l: shapely.STRtree([o[0] for o in os_]) for l, os_ in obst.items()}
        # no track at all in a keepout, not even inside a pad that reaches into one
        keep = {}
        for k in b.d["keepouts"]:
            if k["tracks"]:
                for ln in k["layers"]:
                    if ln in b.li:
                        keep.setdefault(b.li[ln], []).append(shape_of(k))
        # mask openings no copper pad makes: a track in one may bridge it to another net
        mask = {}
        for m in b.d.get("mask_openings", []):
            ln = b.layers[0] if m["side"] == "F" else b.layers[-1]
            mask.setdefault(b.li[ln], []).append(shape_of(m))
        reach = max([o[2] for os_ in obst.values() for o in os_] + [b.net_clr(c[0]) for c in cands]) + 0.05
        edge = b.outline.boundary
        min_track = b.d["rules"].get("min_track") or 0.0
        def fits(n, l, piece):
            # all of the piece, also inside its own pad: KiCad checks a track's clearance
            # there too, and fine-pitch pads can be closer to each other than that
            if any(piece.intersects(g) for g in keep.get(l, []) + mask.get(l, [])):
                return False
            if not b.outline.contains(piece) or piece.distance(edge) < b.edge_clr + 0.005:
                return False
            cn = b.net_clr(n)
            near = [obst[l][i] for i in trees[l].query(piece.buffer(reach))] if l in trees else []
            near += [(g, m, b.net_clr(m), False) for m, ll, g in ok if ll == l]
            return not any(m != n and piece.distance(g) < (c if fixed else max(c, cn)) + 0.005
                           for g, m, c, fixed in near)

        for n, ways in cands:
            t = ways[0][0]
            l = b.li[t["layer"]]
            if t["w"] < min_track - 1e-6:          # under the board's own minimum: no more of it
                continue
            for ts in ways:                        # the designer's way, else straight there
                piece = shapely.LineString([t["a"]] + [x["b"] for x in ts]).buffer(t["w"] / 2)
                if fits(n, l, piece):
                    ok.append((n, l, piece))
                    tracks += ts
                    break
        return ok

    def result(self):
        tracks, vias, cands = [], [], []
        for n in self.nets:
            if n in self.stamped:
                t, v = self.segments(n, cands)
                tracks += t
                vias += v
        self.to_centres(tracks, vias, cands)
        unrouted = []
        for n in self.nets:
            for t in self.failed.get(n, []):
                unrouted.append({"net": self.b.net_names[n], "pads": t["names"]})
        return {"tracks": tracks, "vias": vias, "unrouted": unrouted}
