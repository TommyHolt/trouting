"""The board as the router sees it: a grid of cells per copper layer, built
from the JSON trouting_kicad.py dumps.

Static copper becomes three arrays per layer (see core.c): d1/n1 = distance to
the nearest copper and its net, d2 = distance to the nearest copper of any
other net. Distances are exact (to the real polygons, at each cell centre)
and already have the clearance taken off, so the router's test is just
"distance >= half my width". KiCad keeps the larger of two items'
clearances, so there is one set of arrays per net clearance in use (D1[ci]
for nets of clearance clrs[ci]).
"""
import math

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

BIG = 1e6
NO_NET = -1        # copper of no net, board edge, keepouts, holes: in everyone's way
MARGIN = 0.01      # mm on every clearance, for the grid's straight lines between cell centres


def poly_of(rings, holes=None):
    """Outlines (lists of points) as one shape; holes: per outline, its holes."""
    holes = holes or [[]] * len(rings)
    ps = [Polygon(r, [h for h in hs if len(h) >= 3]) for r, hs in zip(rings, holes) if len(r) >= 3]
    ps = [p if p.is_valid else p.buffer(0) for p in ps]
    return unary_union(ps) if ps else Polygon()


def shape_of(item):
    """A zone's, keepout's or copper shape's area, its holes cut out."""
    return poly_of(item["polys"], item.get("holes"))


class Board:
    def __init__(self, d, res=0.05):
        self.d = d
        self.res = res
        self.layers = d["layers"]
        self.L = len(self.layers)
        self.li = {n: i for i, n in enumerate(self.layers)}
        rules = d["rules"]

        # nets: index, rules
        self.net_names = sorted(d["nets"])
        self.ni = {n: i for i, n in enumerate(self.net_names)}
        # (never under the board's own minimums: a net class narrower than those is
        # all DRC errors, and KiCad's router won't go under them either)
        # (copper narrower than the minimum connection width is flagged too: it counts
        # as the minimum track from here on)
        rules["min_track"] = max(rules.get("min_track") or 0.0, rules.get("min_connection") or 0.0)
        lo = {"width": rules["min_track"], "via_d": rules.get("min_via_d") or 0.0,
              "via_drill": rules.get("min_via_drill") or 0.0, "clearance": rules.get("min_clearance") or 0.0}
        self.net_rule = [dict(d["nets"][n], **{k: max(d["nets"][n][k], v) for k, v in lo.items()
                                                if k in d["nets"][n]}) for n in self.net_names]
        self.fanout_via = self.small_vias(rules)
        clrs = [r["clearance"] for r in self.net_rule] or [0.2]
        self.c_ref = min(clrs)
        self.clrs = sorted(set(clrs))                 # the clearance classes
        self.cc = [self.clrs.index(r["clearance"]) for r in self.net_rule]
        self.edge_clr = rules.get("edge_clearance") or 0.3
        self.hole_clr = rules.get("hole_clearance") or 0.25

        # outline (with cut-outs)
        outer = poly_of(d["outline"])
        holes = poly_of(d.get("outline_holes", []))
        self.outline = outer.difference(holes) if not holes.is_empty else outer
        x0, y0, x1, y1 = self.outline.bounds
        # cell centres on round coordinates (multiples of res), where most pads are
        self.ox = (math.floor(x0 / res) - 1.5) * res
        self.oy = (math.floor(y0 / res) - 1.5) * res
        self.W = int(math.ceil((x1 - x0) / res)) + 3
        self.H = int(math.ceil((y1 - y0) / res)) + 3
        self.WH = self.W * self.H

        self.items = []          # (layer index, geometry, net index, item clearance, fixed)
        # (fixed: the clearance is the item's alone, not the larger of it and the net's:
        # holes keep the hole clearance, keepouts none)
        self.collect_items()
        self.build_static()
        self.find_planes()
        self.build_terminals()

    def small_vias(self, rules):
        """Nets with a pin on a fine-pitch part (pins closer than the net's via and its
        clearance: its vias can't stand one per pin) take the board's smallest
        predefined via size instead of their class's, as people route such parts; not
        under the board's minimums (its annular ring included). {net: (class via, via taken)}."""
        sizes = [(dv, dr) for dv, dr in rules.get("via_sizes") or []
                 if dv >= (rules.get("min_via_d") or 0.0) and dr >= (rules.get("min_via_drill") or 0.0)
                 and (dv - dr) / 2 >= (rules.get("min_via_annular") or 0.0) - 1e-6]
        if not sizes:
            return {}
        small = min(sizes)
        per_fp = {}
        for p in self.d["pads"]:
            per_fp.setdefault(p["ref"], []).append(p)
        out = {}
        for pads in per_fp.values():
            if len(pads) < 2:
                continue
            at = np.array([p["at"] for p in pads], float)
            dist = np.hypot(at[:, None, 0] - at[None, :, 0], at[:, None, 1] - at[None, :, 1])
            np.fill_diagonal(dist, np.inf)
            pitch = dist.min(axis=1)
            for p, pt in zip(pads, pitch):
                n = self.ni.get(p["net"]) if p["net"] else None
                if n is None or n in out:
                    continue
                r = self.net_rule[n]
                if pt < r["via_d"] + r["clearance"] and small[0] < r["via_d"]:
                    out[n] = ((r["via_d"], r["via_drill"]), small)
                    r["via_d"], r["via_drill"] = small
        return out

    # ---- coordinates ----
    def cx(self, x):            # cell x -> mm
        return self.ox + (x + 0.5) * self.res

    def cy(self, y):
        return self.oy + (y + 0.5) * self.res

    def to_cell(self, xm, ym):
        return int(round((xm - self.ox) / self.res - 0.5)), int(round((ym - self.oy) / self.res - 0.5))

    def node_xy(self, v):
        l, xy = divmod(int(v), self.WH)
        y, x = divmod(xy, self.W)
        return l, x, y

    def patch(self, geom, grow):
        """Cell index ranges and centre coordinates covering geom's bbox + grow."""
        bx0, by0, bx1, by1 = geom.bounds
        xa = max(0, int((bx0 - grow - self.ox) / self.res))
        xb = min(self.W - 1, int((bx1 + grow - self.ox) / self.res) + 1)
        ya = max(0, int((by0 - grow - self.oy) / self.res))
        yb = min(self.H - 1, int((by1 + grow - self.oy) / self.res) + 1)
        xs = self.ox + (np.arange(xa, xb + 1) + 0.5) * self.res
        ys = self.oy + (np.arange(ya, yb + 1) + 0.5) * self.res
        X, Y = np.meshgrid(xs, ys)
        idx = (np.arange(ya, yb + 1)[:, None] * self.W + np.arange(xa, xb + 1)[None, :])
        return idx.ravel(), X.ravel(), Y.ravel()

    def cells_in(self, geom, layer, shrink=0.0):
        """Node ids of the cells whose centre is inside geom (optionally shrunk)."""
        g = geom.buffer(-shrink) if shrink > 0 else geom
        if g.is_empty:
            return np.zeros(0, np.int32)
        idx, X, Y = self.patch(g, self.res)
        m = shapely.contains_xy(g, X, Y)
        return (layer * self.WH + idx[m]).astype(np.int32)

    # ---- static copper ----
    def net_clr(self, n):
        return self.net_rule[n]["clearance"] if n >= 0 else self.c_ref

    def collect_items(self):
        d = self.d
        for p in d["pads"]:
            n = self.ni.get(p["net"], NO_NET) if p["net"] else NO_NET
            # the pad's own clearance (its overrides included): KiCad keeps the larger of the
            # two items' clearances, and the router's threshold adds the track's side
            c = p.get("clearance") or self.net_clr(n)
            for ln, rings in p["shapes"].items():
                if ln in self.li:
                    self.items.append((self.li[ln], poly_of(rings), n, c, False))
        for s in d["copper_shapes"]:
            n = self.ni.get(s["net"], NO_NET) if s["net"] else NO_NET
            if s["layer"] in self.li:
                self.items.append((self.li[s["layer"]], shape_of(s), n, self.net_clr(n), False))
        for t in d["tracks"]:          # what dump kept: locked (or all, without --rip-up)
            n = self.ni.get(t["net"], NO_NET)
            if t["layer"] in self.li:
                g = LineString([t["a"], t["b"]]).buffer(t["w"] / 2)
                self.items.append((self.li[t["layer"]], g, n, self.net_clr(n), False))
        for v in d["vias"]:
            n = self.ni.get(v["net"], NO_NET)
            g = Point(v["at"]).buffer(v["d"] / 2)
            # its hole keeps the hole clearance from other nets' copper, which is more
            # than the copper clearance from its pad when the ring is narrow
            hole = Point(v["at"]).buffer(v["drill"] / 2) if v.get("drill") else None
            for l in range(self.L):
                self.items.append((l, g, n, self.net_clr(n), False))
                if hole is not None:
                    self.items.append((l, hole, n, self.hole_clr, True))
        for h in d["holes"]:
            g = Point(h["at"]).buffer(h["r"])
            c = max(self.hole_clr, h.get("clearance") or 0.0)
            for l in range(self.L):
                self.items.append((l, g, NO_NET, c, True))
        for k in d["keepouts"]:
            if k["tracks"]:
                g = shape_of(k)
                for ln in k["layers"]:
                    if ln in self.li:
                        self.items.append((self.li[ln], g, NO_NET, 0.0, True))

    def build_static(self):
        L, WH, C = self.L, self.WH, len(self.clrs)
        self.D1 = np.full((C, L * WH), BIG, np.float32)
        self.D2 = np.full((C, L * WH), BIG, np.float32)
        self.N1 = np.full((C, L * WH), -2, np.int32)
        wmax = max([r["width"] for r in self.net_rule] + [0.25])
        vmax = max([r["via_d"] for r in self.net_rule] + [0.6])
        cmax = max([r["clearance"] for r in self.net_rule] + [self.edge_clr, self.hole_clr] +
                   [it[3] for it in self.items])
        self.rmax = max(wmax, vmax) / 2 + cmax + 0.1
        for l, g, n, c, fixed in self.items:
            if g.is_empty:
                continue
            idx, X, Y = self.patch(g, self.rmax)
            dist = shapely.distance(g, shapely.points(X, Y))
            for ci, cn in enumerate(self.clrs):
                self.merge(ci, l * WH + idx, (dist - (c if fixed else max(c, cn))).astype(np.float32), n)

        # board edge: outside is blocked outright, near the edge keeps edge clearance
        allidx, X, Y = self.patch(self.outline, self.res)
        inside = shapely.contains_xy(self.outline, X, Y)
        band = self.outline.boundary.buffer(self.rmax)
        shapely.prepare(band)
        near = inside & shapely.contains_xy(band, X, Y)
        dist = np.full(len(X), -BIG, np.float32)
        dist[inside] = BIG
        dist[near] = shapely.distance(self.outline.boundary, shapely.points(X[near], Y[near])) - self.edge_clr
        sel = dist < BIG
        for ci in range(C):
            for l in range(L):
                self.merge(ci, l * WH + allidx[sel], dist[sel], NO_NET)
        # (the first class's arrays, for drawing)
        self.d1, self.d2, self.n1 = self.D1[0], self.D2[0], self.N1[0]
        self.inside = np.zeros(WH, bool)
        self.inside[allidx[inside]] = True

        # where vias may not go
        self.via_block = np.zeros(L * WH, np.uint8)
        for k in self.d["keepouts"]:
            if k["vias"]:
                g = shape_of(k).buffer(vmax / 2)
                for ln in k["layers"]:
                    if ln in self.li:
                        self.via_block[self.cells_in(g, self.li[ln])] = 1

    def static_d(self, n, sl=slice(None)):
        """Net n's room from fixed copper (mm beyond the clearance) at nodes sl."""
        ci = self.cc[n]
        return np.where(self.N1[ci][sl] == n, self.D2[ci][sl], self.D1[ci][sl])

    def merge(self, ci, idx, dn, n):
        d1, d2, n1 = self.D1[ci][idx], self.D2[ci][idx], self.N1[ci][idx]
        same = n1 == n
        closer = ~same & (dn < d1)
        other = ~same & ~closer
        # same net as the nearest: it may only get nearer
        d1 = np.where(same, np.minimum(d1, dn), d1)
        # a new nearest, of another net: the old nearest becomes the nearest other
        d2 = np.where(closer, d1, d2)
        n1 = np.where(closer, n, n1)
        d1 = np.where(closer, dn, d1)
        # farther, of another net than the nearest
        d2 = np.where(other, np.minimum(d2, dn), d2)
        self.D1[ci][idx], self.D2[ci][idx], self.N1[ci][idx] = d1, d2, n1

    # ---- pours that act as planes ----
    def pour_region(self, n, l):
        """Where net n's pours on layer l can fill: its outlines on the board, minus
        no-pour keepouts and minus other nets' pours of higher priority (with clearance)."""
        ln = self.layers[l]
        mine = [p for p in self.d["pours"] if self.ni.get(p["net"]) == n and ln in p["layers"]]
        if not mine:
            return Polygon()
        region = unary_union([shape_of(p) for p in mine]).intersection(self.outline)
        prio = min(p["priority"] for p in mine)
        for k in self.d["keepouts"]:
            if k.get("pours") and ln in k["layers"]:
                region = region.difference(shape_of(k))
        for p in self.d["pours"]:
            m = self.ni.get(p["net"])
            if m != n and ln in p["layers"] and p["priority"] > prio:
                region = region.difference(shape_of(p).buffer(self.net_clr(n)))
        return region

    def pour_rules(self, n, l):
        """(clearance, min width) of net n's pours on layer l: the widest of them."""
        ln = self.layers[l]
        ps = [p for p in self.d["pours"] if self.ni.get(p["net"]) == n and ln in p["layers"]]
        c = max([p.get("clearance", 0) for p in ps] + [self.net_clr(n)])
        w = max([p.get("min_width", 0) for p in ps] + [0.0])
        return c, w

    def thermal(self, n, l):
        """(gap, spoke width, pad connection) of net n's pours on layer l."""
        ln = self.layers[l]
        ps = [p for p in self.d["pours"] if self.ni.get(p["net"]) == n and ln in p["layers"]]
        if not ps:
            return 0.5, 0.5, 1
        return (max(p.get("thermal_gap", 0.5) for p in ps), min(p.get("spoke_width", 0.5) for p in ps),
                ps[0].get("pad_connection", 1))

    def find_planes(self):
        """A net whose pour covers >= 25 % of the board on a layer gets that layer
        as its plane: its pads there are connected by the pour, its other pads
        get a short track and a via down to it."""
        area = self.outline.area
        cand = {}
        for p in self.d["pours"]:
            n = self.ni.get(p["net"])
            if n is None:
                continue
            for ln in p["layers"]:
                l = self.li.get(ln)
                if l is not None and (n, l) not in cand:
                    region = self.pour_region(n, l)
                    if region.area >= 0.25 * area:
                        cand[(n, l)] = region
        self.planes = {}
        for n in {n for n, _ in cand}:
            mine = {l: r for (m, l), r in cand.items() if m == n}
            biggest = max(r.area for r in mine.values())
            # the lowest layer among those within 10 % of the biggest (planes are usually at the back)
            l = max(l for l, r in mine.items() if r.area >= 0.9 * biggest)
            self.planes[n] = (l, mine[l])
        # the plane nets' pours on their other layers: these get stitched to the plane
        self.side_pours = {}
        for n, (pl, _) in self.planes.items():
            for p in self.d["pours"]:
                if self.ni.get(p["net"]) == n:
                    for ln in p["layers"]:
                        l = self.li.get(ln)
                        if l is not None and l != pl and (n, l) not in self.side_pours:
                            r = self.pour_region(n, l)
                            if not r.is_empty:
                                self.side_pours[(n, l)] = r
        self.plane_layers = {l for l, _ in self.planes.values()}

    # ---- what has to be connected ----
    def build_terminals(self):
        """Per net, groups of copper already joined (pads, locked tracks and
        vias that touch). Each group is one terminal the router connects."""
        d = self.d
        per = {}
        for p in d["pads"]:
            if not p["net"] or p["net"] not in self.ni:
                continue
            n = self.ni[p["net"]]
            geoms = {self.li[ln]: poly_of(r) for ln, r in p["shapes"].items() if ln in self.li}
            per.setdefault(n, []).append({"kind": "pad", "name": "%s.%s" % (p["ref"], p["num"]),
                                          "at": p["at"], "geoms": geoms, "pth": "drill_r" in p,
                                          "zone_connection": p.get("zone_connection", -1),
                                          "spokes": (p.get("rot", 0.0) + p.get("spoke_angle", 90.0)) % 90.0})
        for t in d["tracks"]:
            if t["net"] in self.ni and t["layer"] in self.li:
                n = self.ni[t["net"]]
                g = LineString([t["a"], t["b"]]).buffer(t["w"] / 2)
                per.setdefault(n, []).append({"kind": "track", "name": "track", "at": t["a"],
                                              "ends": [t["a"], t["b"]], "w": t["w"],
                                              "geoms": {self.li[t["layer"]]: g}})
        for v in d["vias"]:
            if v["net"] in self.ni:
                n = self.ni[v["net"]]
                g = Point(v["at"]).buffer(v["d"] / 2)
                per.setdefault(n, []).append({"kind": "via", "name": "via", "at": v["at"],
                                              "geoms": {l: g for l in range(self.L)}})
        # copper drawn with a net (a shape, copper text): KiCad wants it joined like a pad
        for c in d["copper_shapes"]:
            if c["net"] in self.ni and c["layer"] in self.li:
                g = shape_of(c)
                if g.is_empty:
                    continue
                at = [g.centroid.x, g.centroid.y]
                per.setdefault(self.ni[c["net"]], []).append(
                    {"kind": "shape", "name": "shape at (%.2f, %.2f)" % tuple(at), "at": at,
                     "geoms": {self.li[c["layer"]]: g}})
        self.terminals = {}
        for n, items in per.items():
            parent = list(range(len(items)))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a, b = items[i]["geoms"], items[j]["geoms"]
                    if any(l in b and a[l].intersects(b[l]) for l in a):
                        parent[find(i)] = find(j)
            groups = {}
            for i, it in enumerate(items):
                groups.setdefault(find(i), []).append(it)
            self.terminals[n] = [self.make_terminal(n, g) for g in groups.values()]

    def make_terminal(self, n, items):
        r = self.net_rule[n]
        w = r["width"]
        cells, own, novia = [], [], []
        for it in items:
            for l, g in it["geoms"].items():
                cells.append(self.cells_in(g, l))
                own.append(self.cells_in(g, l, shrink=w / 2))
                if it["kind"] == "pad":
                    novia.append(self.cells_in(g, l))
        cat = lambda a: np.unique(np.concatenate(a)).astype(np.int32) if a else np.zeros(0, np.int32)
        # a kept track end that touches nothing else of the group was left for the
        # router to carry on from: the group is entered there (else it dangles)
        entry = []
        for it in items:
            if it["kind"] != "track":
                continue
            (l, g), = it["geoms"].items()
            for e in it["ends"]:
                tip = Point(e).buffer(it["w"] / 2 + 0.01)
                if not any(o is not it and l in o["geoms"] and o["geoms"][l].intersects(tip) for o in items):
                    entry.append(self.cells_in(tip, l))
        # (only cells of the group itself: a path ending outside it wouldn't join it, and
        # the repair would look for that join forever)
        entry = np.intersect1d(cat(entry), cat(cells)).astype(np.int32) if entry else np.zeros(0, np.int32)
        t = {"net": n, "items": items, "cells": cat(cells), "own": cat(own), "novia": cat(novia),
             "entry": entry if len(entry) else cat(cells),
             "names": [it["name"] for it in items if it["kind"] == "pad"] or
                      [it["name"] for it in items if it["kind"] == "shape"],
             "at": np.mean([it["at"] for it in items], axis=0)}
        if n in self.planes:
            pl, region = self.planes[n]
            t["on_plane"] = any(pl in it["geoms"] and it["geoms"][pl].intersects(region) for it in items)
        return t
