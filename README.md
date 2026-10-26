# ai-rout-finder
#it finds the best rout according to the conditions of the user 

"""
AI Route Finder
================
A desktop prototype (Python + Tkinter GUI) demonstrating a multi-factor,
"smart" route-planning engine that goes beyond simple shortest-path
routing used by most consumer map apps (like Google Maps).

Core idea: instead of optimizing only for time/distance, this engine
blends travel time with a chosen priority (Speed or Safety) into one
cost function, and colors every road on the map with a FIXED category
(Green / Yellow / Red) based on that road's real-world traffic level
and safety score -- these colors never change between runs.
"""

import math
import heapq
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 1. DATA MODEL
# ---------------------------------------------------------------------------

@dataclass
class Node:
    id: str
    name: str
    x: float   # canvas x position
    y: float   # canvas y position


@dataclass
class Edge:
    a: str
    b: str
    distance_km: float
    base_speed_kmph: float
    safety_score: float      # 0 (unsafe) - 10 (very safe)      -- FIXED
    base_traffic: float      # 0 (empty)  - 10 (jammed)          -- FIXED, used for road COLOR
    traffic_level: float = 0.0   # dynamic value used for cost/time (varies with time of day)

    def base_time_min(self) -> float:
        return (self.distance_km / self.base_speed_kmph) * 60


# ---------------------------------------------------------------------------
# 2. SAMPLE CITY GRAPH  (swap this for real map/GPS data later)
# ---------------------------------------------------------------------------

def build_sample_city() -> Tuple[Dict[str, Node], List[Edge]]:
    # Positions are laid out to fit comfortably inside an 880 x 420 canvas.
    nodes = {
        "A": Node("A", "Home", 70, 320),
        "B": Node("B", "Main Market", 240, 200),
        "C": Node("C", "Tech Park", 470, 70),
        "D": Node("D", "Riverside Ave", 240, 380),
        "E": Node("E", "Old Town", 470, 380),
        "F": Node("F", "Bus Depot", 680, 200),
        "G": Node("G", "College", 810, 380),
        "H": Node("H", "Hospital", 470, 220),
    }

    # a, b, distance_km, speed_kmph, safety(0-10), base_traffic(0-10)
    # base_traffic is a FIXED real-world congestion rating for that road
    # (e.g. from historical data) -- it decides the road's color and does
    # NOT change every time you click a button.
    #
    # NOTE: B-C-F-G is a genuine "risky shortcut" -- short, fast, low
    # traffic, but unsafe (imagine a dark back-alley route). D-H-E is the
    # "long way round" -- slower, but safe. This gives Speed-priority and
    # Safety-priority real, different answers instead of always agreeing.
    raw_edges = [
        ("A", "B", 2.1, 30, 7, 3),   # decent road              -> YELLOW
        ("A", "D", 2.6, 25, 8, 2),   # quiet, safe road         -> GREEN
        ("B", "C", 3.4, 45, 3, 2),   # fast shortcut, UNSAFE    -> RED
        ("B", "H", 2.0, 30, 7, 3),   # decent road              -> YELLOW
        ("D", "E", 3.0, 28, 8, 3),   # quiet, safe road         -> GREEN
        ("D", "H", 2.4, 25, 7, 3),   # decent road              -> YELLOW
        ("H", "E", 2.2, 25, 8, 2),   # quiet, safe road         -> GREEN
        ("H", "C", 2.8, 35, 5, 5),   # medium road              -> YELLOW
        ("C", "F", 2.5, 50, 3, 2),   # fast shortcut, UNSAFE    -> RED
        ("E", "G", 3.6, 28, 7, 3),   # decent road              -> YELLOW
        ("F", "G", 3.0, 45, 3, 2),   # fast shortcut, UNSAFE    -> RED
        ("E", "F", 3.8, 30, 4, 6),   # jammed + risky           -> RED
    ]

    edges = [Edge(a, b, dist, spd, safety, traffic)
             for a, b, dist, spd, safety, traffic in raw_edges]
    return nodes, edges


# ---------------------------------------------------------------------------
# 3. LIVE-CONDITION SIMULATION (stand-in for a real traffic API feed)
# ---------------------------------------------------------------------------
# NOTE: This is DETERMINISTIC on purpose -- no randomness. The same road,
# at the same time of day, always gets the same traffic_level, so the
# "Estimated time" for a given route + time-of-day never changes between
# clicks. It never touches base_traffic either, so the GREEN / YELLOW / RED
# color of a road on the map stays fixed no matter what you select.
#
# Swap this out for a real traffic/AQI API call later -- just make sure
# whatever you plug in returns the same value for the same inputs, or
# you'll get the same "random result each click" problem again.

RUSH_HOUR_BUMP = 2.0   # extra congestion added on top of base_traffic during rush hour

def simulate_conditions(edges: List[Edge], time_of_day: str) -> None:
    rush = time_of_day in ("Morning Rush", "Evening Rush")
    bump = RUSH_HOUR_BUMP if rush else 0.0
    for e in edges:
        e.traffic_level = max(0, min(10, e.base_traffic + bump))


# ---------------------------------------------------------------------------
# 4. COST FUNCTION  (the "smart routing" engine)
# ---------------------------------------------------------------------------
# Simplified to a single priority mode, chosen with one radio button:
#   "speed"  -> pure fastest-path routing (classic Google-Maps style)
#   "safety" -> routes penalize unsafe roads, even if slightly slower
#
# The safety penalty is ADDITIVE (extra "minutes" tacked on per unsafety
# point), not a small multiplier. A multiplier was too weak to ever flip
# Dijkstra's answer -- it barely nudged the cost, so Speed and Safety
# always picked the exact same road. This is strong enough to genuinely
# reroute around a fast-but-unsafe road when Safety is selected.
SAFETY_PENALTY_PER_POINT = 4.0   # "minutes" of penalty per point of unsafety (0-10 scale)


def edge_cost(e: Edge, priority: str) -> float:
    time_cost = e.base_time_min() * (1 + e.traffic_level / 10)

    if priority == "safety":
        unsafety = 10 - e.safety_score      # 0 (very safe) - 10 (very unsafe)
        cost = time_cost + unsafety * SAFETY_PENALTY_PER_POINT
    else:  # priority == "speed"
        cost = time_cost

    return max(cost, 0.01)


# ---------------------------------------------------------------------------
# 5. DIJKSTRA WITH THE CUSTOM COST FUNCTION
# ---------------------------------------------------------------------------

def build_adjacency(edges: List[Edge]) -> Dict[str, List[Edge]]:
    adj: Dict[str, List[Edge]] = {}
    for e in edges:
        adj.setdefault(e.a, []).append(e)
        adj.setdefault(e.b, []).append(
            Edge(e.b, e.a, e.distance_km, e.base_speed_kmph,
                 e.safety_score, e.base_traffic, e.traffic_level)
        )
    return adj


def find_route(nodes, edges, start, end, priority):
    adj = build_adjacency(edges)
    dist = {n: math.inf for n in nodes}
    prev = {n: None for n in nodes}
    used_edge = {}
    dist[start] = 0
    pq = [(0, start)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == end:
            break
        for e in adj.get(u, []):
            v = e.b
            c = edge_cost(e, priority)
            nd = d + c
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                used_edge[v] = e
                heapq.heappush(pq, (nd, v))

    if dist[end] == math.inf:
        return None

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    route_edges = [used_edge[n] for n in path[1:]]
    return path, route_edges, dist[end]


def route_stats(route_edges: List[Edge]) -> Dict[str, float]:
    if not route_edges:
        return {"distance": 0, "time": 0, "safety": 0}
    total_dist = sum(e.distance_km for e in route_edges)
    total_time = sum(e.base_time_min() * (1 + e.traffic_level / 10) for e in route_edges)
    avg_safety = sum(e.safety_score for e in route_edges) / len(route_edges)
    return {"distance": total_dist, "time": total_time, "safety": avg_safety}


# ---------------------------------------------------------------------------
# 6. VISUAL CONSTANTS
# ---------------------------------------------------------------------------

BG_MAIN = "#f4f6fa"
BG_CANVAS = "#ffffff"
COLOR_ROUTE = "#1d3557"
COLOR_NODE = "#3a5a8c"
COLOR_START = "#2a9d8f"
COLOR_END = "#e63946"
COLOR_TEXT = "#1d1d1d"
FONT_UI = ("Segoe UI", 10)
FONT_NODE = ("Segoe UI", 9, "bold")
FONT_EDGE = ("Segoe UI", 8)

GREEN = "#2a9d8f"    # no traffic + safe
YELLOW = "#e9c46a"   # light traffic + a bit unsafe
RED = "#e76f51"      # high traffic + risky


def road_color(safety_score: float, base_traffic: float) -> str:
    """
    FIXED categorization -- depends only on each road's permanent
    safety_score and base_traffic, so a given road is always the same
    color no matter what button you click or what time of day is set.

        GREEN  : low traffic AND high safety
        RED    : high traffic OR  low safety
        YELLOW : everything in between
    """
    if base_traffic <= 3 and safety_score >= 7:
        return GREEN
    if base_traffic >= 7 or safety_score <= 3:
        return RED
    return YELLOW


# ---------------------------------------------------------------------------
# 7. TKINTER GUI
# ---------------------------------------------------------------------------

class RouteFinderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Route Finder - Prototype")
        self.configure(bg=BG_MAIN)
        self.geometry("980x880")
        self.minsize(980, 880)
        self.nodes, self.edges = build_sample_city()

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG_MAIN)
        style.configure("TLabel", background=BG_MAIN, font=FONT_UI, foreground=COLOR_TEXT)
        style.configure("Header.TLabel", font=("Segoe UI", 15, "bold"), background=BG_MAIN)
        style.configure("Sub.TLabel", font=("Segoe UI", 9), foreground="#555555", background=BG_MAIN)
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("TRadiobutton", background=BG_MAIN, font=FONT_UI)

        self._build_header()
        self._build_controls()
        self._build_canvas()
        self._build_legend()
        self._build_output()
        self.draw_city()

    # -- UI construction -----------------------------------------------
    def _build_header(self):
        header = ttk.Frame(self, padding=(16, 14, 16, 4))
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header, text="AI Route Finder", style="Header.TLabel").pack(anchor="w")
        ttk.Label(header, text="Multi-factor routing: pick Speed or Safety, "
                               "not just the shortest path.",
                  style="Sub.TLabel").pack(anchor="w")

    def _build_controls(self):
        frame = ttk.Frame(self, padding=(16, 8))
        frame.pack(side=tk.TOP, fill=tk.X)

        names = [n.name for n in self.nodes.values()]
        self.start_var = tk.StringVar(value=names[0])
        self.end_var = tk.StringVar(value=names[-1])

        row0 = ttk.Frame(frame)
        row0.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(row0, text="From").pack(side=tk.LEFT)
        ttk.Combobox(row0, textvariable=self.start_var, values=names,
                     width=16, state="readonly").pack(side=tk.LEFT, padx=(6, 20))
        ttk.Label(row0, text="To").pack(side=tk.LEFT)
        ttk.Combobox(row0, textvariable=self.end_var, values=names,
                     width=16, state="readonly").pack(side=tk.LEFT, padx=(6, 20))
        ttk.Label(row0, text="Time of day").pack(side=tk.LEFT)
        self.time_var = tk.StringVar(value="Day")
        ttk.Combobox(row0, textvariable=self.time_var,
                     values=["Day", "Night", "Morning Rush", "Evening Rush"],
                     width=16, state="readonly").pack(side=tk.LEFT, padx=6)

        # Single priority choice -- only ONE of Speed / Safety can be active
        priority_row = ttk.Frame(frame)
        priority_row.pack(fill=tk.X, pady=(4, 4))
        ttk.Label(priority_row, text="Route priority:").pack(side=tk.LEFT, padx=(0, 12))
        self.priority_var = tk.StringVar(value="speed")
        ttk.Radiobutton(priority_row, text="Prioritize Speed",
                         variable=self.priority_var, value="speed").pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(priority_row, text="Prioritize Safety",
                         variable=self.priority_var, value="safety").pack(side=tk.LEFT)

        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=(12, 4))
        ttk.Button(btns, text="Find Best Route",
                   command=self.on_find_route).pack(side=tk.LEFT)
        ttk.Button(btns, text="Compare with Fastest-Only Route",
                   command=self.on_compare).pack(side=tk.LEFT, padx=10)

    def _build_canvas(self):
        wrap = ttk.Frame(self, padding=(16, 6))
        wrap.pack(side=tk.TOP, fill=tk.X)
        self.canvas = tk.Canvas(wrap, width=940, height=430, bg=BG_CANVAS,
                                 highlightthickness=1, highlightbackground="#dcdfe6")
        self.canvas.pack()

    def _build_legend(self):
        legend = ttk.Frame(self, padding=(16, 0))
        legend.pack(side=tk.TOP, fill=tk.X)
        items = [("Start", COLOR_START), ("Destination", COLOR_END),
                 ("Chosen route", COLOR_ROUTE),
                 ("Green: no traffic + safe", GREEN),
                 ("Yellow: light traffic + a bit unsafe", YELLOW),
                 ("Red: high traffic + risky", RED)]
        for text, color in items:
            item = ttk.Frame(legend)
            item.pack(side=tk.LEFT, padx=(0, 16), pady=4)
            swatch = tk.Canvas(item, width=14, height=14, bg=BG_MAIN, highlightthickness=0)
            swatch.pack(side=tk.LEFT)
            swatch.create_oval(1, 1, 13, 13, fill=color, outline="")
            ttk.Label(item, text=text, style="Sub.TLabel").pack(side=tk.LEFT, padx=4)

    def _build_output(self):
        wrap = ttk.Frame(self, padding=(16, 8, 16, 16))
        wrap.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ttk.Label(wrap, text="Route details", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.output = tk.Text(wrap, width=100, height=10, font=("Consolas", 10),
                               bg="#ffffff", relief=tk.FLAT, padx=10, pady=8)
        self.output.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    # -- drawing ----------------------------------------------------------
    def draw_city(self, highlight_path=None, start_id=None, end_id=None):
        self.canvas.delete("all")

        # roads, colored by the FIXED green/yellow/red category
        for e in self.edges:
            a, b = self.nodes[e.a], self.nodes[e.b]
            color = road_color(e.safety_score, e.base_traffic)
            self.canvas.create_line(a.x, a.y, b.x, b.y, fill=color, width=4)
            mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
            self.canvas.create_text(mx, my, text=f"{e.distance_km:.1f} km",
                                     font=FONT_EDGE, fill="#888888")

        # highlighted chosen route on top
        if highlight_path:
            for i in range(len(highlight_path) - 1):
                a = self.nodes[highlight_path[i]]
                b = self.nodes[highlight_path[i + 1]]
                self.canvas.create_line(a.x, a.y, b.x, b.y, fill=COLOR_ROUTE, width=3,
                                         arrow=tk.LAST, arrowshape=(10, 12, 4))

        # nodes
        for n in self.nodes.values():
            color = COLOR_NODE
            radius = 12
            if start_id and n.id == start_id:
                color, radius = COLOR_START, 15
            elif end_id and n.id == end_id:
                color, radius = COLOR_END, 15
            self.canvas.create_oval(n.x - radius, n.y - radius, n.x + radius, n.y + radius,
                                     fill=color, outline="white", width=2)
            self.canvas.create_text(n.x, n.y - radius - 10, text=n.name, font=FONT_NODE, fill=COLOR_TEXT)

    # -- lookups ------------------------------------------------------------
    def _name_to_id(self, name):
        for n in self.nodes.values():
            if n.name == name:
                return n.id
        return None

    # -- actions ----------------------------------------------------------
    def on_find_route(self):
        start = self._name_to_id(self.start_var.get())
        end = self._name_to_id(self.end_var.get())
        if start == end:
            messagebox.showwarning("Route Finder", "Start and destination are the same.")
            return
        simulate_conditions(self.edges, self.time_var.get())
        result = find_route(self.nodes, self.edges, start, end, self.priority_var.get())
        if not result:
            messagebox.showerror("Route Finder", "No route found.")
            return
        path, route_edges, _ = result
        stats = route_stats(route_edges)
        self.draw_city(highlight_path=path, start_id=start, end_id=end)
        self._show_result("AI-Optimized Route", path, stats)

    def on_compare(self):
        start = self._name_to_id(self.start_var.get())
        end = self._name_to_id(self.end_var.get())
        if start == end:
            messagebox.showwarning("Route Finder", "Start and destination are the same.")
            return
        simulate_conditions(self.edges, self.time_var.get())

        chosen_priority = self.priority_var.get()

        ai_result = find_route(self.nodes, self.edges, start, end, chosen_priority)
        fast_result = find_route(self.nodes, self.edges, start, end, "speed")

        if not ai_result or not fast_result:
            messagebox.showerror("Route Finder", "No route found.")
            return

        ai_path, ai_edges, _ = ai_result
        fast_path, fast_edges, _ = fast_result
        ai_stats = route_stats(ai_edges)
        fast_stats = route_stats(fast_edges)

        self.draw_city(highlight_path=ai_path, start_id=start, end_id=end)
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "=== Fastest-only route (Google-Maps style) ===\n")
        self._append_stats(fast_path, fast_stats)
        self.output.insert(tk.END, f"\n=== AI-Optimized route (priority: {chosen_priority}) ===\n")
        self._append_stats(ai_path, ai_stats)

    def _show_result(self, title, path, stats):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"=== {title} ===\n")
        self._append_stats(path, stats)

    def _append_stats(self, path, stats):
        names = [self.nodes[n].name for n in path]
        self.output.insert(tk.END, " -> ".join(names) + "\n")
        self.output.insert(tk.END, f"Distance: {stats['distance']:.2f} km\n")
        self.output.insert(tk.END, f"Estimated time: {stats['time']:.1f} min\n")
        self.output.insert(tk.END, f"Average safety score: {stats['safety']:.1f}/10\n")


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    app = RouteFinderApp()
    app.mainloop()


if __name__ == "__main__":
    main()

