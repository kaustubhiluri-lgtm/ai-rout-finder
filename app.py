import math
import heapq
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Route Finder", page_icon="🧭", layout="wide")

# Demonstration city graph from the original Tkinter project.
NODES = {
    "A": {"name": "Home", "pos": (0.5, 1.1)},
    "B": {"name": "Main Market", "pos": (2.5, 2.2)},
    "C": {"name": "Tech Park", "pos": (5.0, 3.2)},
    "D": {"name": "Riverside Ave", "pos": (2.5, 0.5)},
    "E": {"name": "Old Town", "pos": (5.0, 0.5)},
    "F": {"name": "Bus Depot", "pos": (7.3, 2.2)},
    "G": {"name": "College", "pos": (9.2, 0.5)},
    "H": {"name": "Hospital", "pos": (5.0, 1.8)},
}
# a, b, distance km, base speed km/h, safety /10, base traffic /10
RAW_EDGES = [
    ("A", "B", 2.1, 30, 7, 3), ("A", "D", 2.6, 25, 8, 2),
    ("B", "C", 3.4, 45, 3, 2), ("B", "H", 2.0, 30, 7, 3),
    ("D", "E", 3.0, 28, 8, 3), ("D", "H", 2.4, 25, 7, 3),
    ("H", "E", 2.2, 25, 8, 2), ("H", "C", 2.8, 35, 5, 5),
    ("C", "F", 2.5, 50, 3, 2), ("E", "G", 3.6, 28, 7, 3),
    ("F", "G", 3.0, 45, 3, 2), ("E", "F", 3.8, 30, 4, 6),
]

def traffic_for(base, time):
    bump = 2 if time in ("Morning Rush", "Evening Rush") else 0
    return min(10, max(0, base + bump))

def edge_cost(e, priority, time):
    a, b, dist, speed, safety, traffic = e
    t = (dist / speed) * 60 * (1 + traffic_for(traffic, time) / 10)
    return t + (10 - safety) * 4 if priority == "Safety" else t

def shortest_route(start, end, priority, time):
    adj = {n: [] for n in NODES}
    for e in RAW_EDGES:
        a, b, *_ = e
        adj[a].append((b, e)); adj[b].append((a, (b, a, *e[2:])))
    distances = {n: math.inf for n in NODES}; distances[start] = 0
    previous = {}; pq = [(0, start)]; visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u)
        if u == end: break
        for v, e in adj[u]:
            nd = d + edge_cost(e, priority, time)
            if nd < distances[v]:
                distances[v] = nd; previous[v] = (u, e); heapq.heappush(pq, (nd, v))
    if start != end and end not in previous: return None
    path = [end]; route_edges = []; cur = end
    while cur != start:
        prev, e = previous[cur]; route_edges.append(e); path.append(prev); cur = prev
    path.reverse(); route_edges.reverse()
    distance = sum(e[2] for e in route_edges)
    minutes = sum((e[2] / e[3]) * 60 * (1 + traffic_for(e[5], time) / 10) for e in route_edges)
    safety = sum(e[4] for e in route_edges) / len(route_edges) if route_edges else 0
    return path, route_edges, {"Distance (km)": distance, "Estimated time (min)": minutes, "Average safety (/10)": safety}

def road_colour(e):
    safety, traffic = e[4], e[5]
    if traffic <= 3 and safety >= 7: return "#2a9d8f"
    if traffic >= 7 or safety <= 3: return "#e76f51"
    return "#e9c46a"

st.title("🧭 AI Route Finder System")
st.caption("Smart route planning using Dijkstra's algorithm with speed or safety priority.")
st.info("This demonstration uses a sample city graph and simulated traffic conditions. It does not use live GPS or real-time traffic feeds.")

names = {v["name"]: k for k, v in NODES.items()}
c1, c2, c3 = st.columns(3)
with c1: start_name = st.selectbox("From", list(names.keys()), index=0)
with c2: end_name = st.selectbox("Destination", list(names.keys()), index=6)
with c3: time = st.selectbox("Time of day", ["Day", "Night", "Morning Rush", "Evening Rush"])
priority = st.radio("Route priority", ["Speed", "Safety"], horizontal=True)

if start_name == end_name:
    st.warning("Please select different start and destination locations.")
else:
    result = shortest_route(names[start_name], names[end_name], priority, time)
    if result:
        path, route_edges, stats = result
        m1, m2, m3 = st.columns(3)
        m1.metric("Distance", f"{stats['Distance (km)']:.2f} km")
        m2.metric("Estimated travel time", f"{stats['Estimated time (min)']:.1f} min")
        m3.metric("Average safety", f"{stats['Average safety (/10)']:.1f}/10")
        st.subheader("Recommended route")
        st.success(" → ".join(NODES[n]["name"] for n in path))

        fig, ax = plt.subplots(figsize=(11, 4.8))
        for e in RAW_EDGES:
            a, b = e[0], e[1]; x1, y1 = NODES[a]["pos"]; x2, y2 = NODES[b]["pos"]
            ax.plot([x1, x2], [y1, y2], color=road_colour(e), linewidth=3, zorder=1)
            ax.text((x1+x2)/2, (y1+y2)/2, f"{e[2]:.1f} km", fontsize=8, color="#555555", ha="center")
        for i in range(len(path)-1):
            x1, y1 = NODES[path[i]]["pos"]; x2, y2 = NODES[path[i+1]]["pos"]
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops={"arrowstyle":"->", "color":"#264653", "lw":3.2}, zorder=2)
        for node, info in NODES.items():
            x, y = info["pos"]
            colour = "#2a9d8f" if node == path[0] else "#e63946" if node == path[-1] else "#3a5a8c"
            ax.scatter(x, y, s=260, color=colour, edgecolors="white", linewidths=1.5, zorder=3)
            ax.text(x, y+0.22, info["name"], ha="center", va="bottom", fontsize=9, weight="bold")
        ax.set_xlim(-0.2, 10); ax.set_ylim(0, 3.8); ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        with st.expander("Road details in selected route"):
            rows = [{"From": NODES[e[0]]["name"], "To": NODES[e[1]]["name"], "Distance (km)": e[2], "Safety (/10)": e[4], "Simulated traffic (/10)": traffic_for(e[5], time)} for e in route_edges]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if st.button("Compare with fastest-only route"):
            fast = shortest_route(names[start_name], names[end_name], "Speed", time)
            _, _, fast_stats = fast
            st.write("**Fastest-only route:** " + " → ".join(NODES[n]["name"] for n in fast[0]))
            st.write(f"Distance: {fast_stats['Distance (km)']:.2f} km · Estimated time: {fast_stats['Estimated time (min)']:.1f} min · Average safety: {fast_stats['Average safety (/10)']:.1f}/10")
