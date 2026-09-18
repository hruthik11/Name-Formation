"""
Problem 1: Displacement-based formation control over a random Erdos-Renyi
communication graph. N = 20 agents move, letter by letter, through the
of " Hruthik").


"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.font_manager import FontProperties
from sklearn.cluster import KMeans
from skimage.morphology import skeletonize

rng = np.random.default_rng(7)


N = 20                      
P_ER = 0.35                 
NAME = "HRUTHIK"            
DT = 0.02                  
T_PER_LETTER = 6.0         
HOLD_TIME = 1.0             
GAIN = 3.0                  


def connected_er_graph(n, p, rng):
    while True:
        seed = int(rng.integers(0, 1_000_000))
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        if nx.is_connected(G):
            return G

G = connected_er_graph(N, P_ER, rng)
L = nx.laplacian_matrix(G).toarray().astype(float)   # graph Laplacian

print(f"Connected Erdos-Renyi graph generated: N={N}, p={P_ER}, "
      f"|E|={G.number_of_edges()}, algebraic connectivity="
      f"{sorted(np.linalg.eigvalsh(L))[1]:.4f}")


def letter_to_points(letter, n_points=N, res=300):
    """Rasterize a capital letter with matplotlib, take the 'ink' pixels,
    and summarize them with n_points KMeans cluster centers so that any
    glyph (regardless of how many ink pixels it has) is reduced to exactly
    n_points representative (x, y) coordinates that trace out its shape."""
    fig = plt.figure(figsize=(res / 100, res / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    ax.text(0.5, 0.42, letter, fontsize=230, fontproperties=fp,
            ha="center", va="center")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, 0]  
    plt.close(fig)

    ink_mask = buf < 128                                  
    skeleton = skeletonize(ink_mask)                      
    ink_rows, ink_cols = np.where(skeleton)
    pts = np.column_stack([ink_cols, res - ink_rows])    
    pts = pts.astype(float)

    km = KMeans(n_clusters=n_points, n_init=6, random_state=0).fit(pts)
    centers = km.cluster_centers_

    
    centers -= centers.min(axis=0)
    span = centers.max(axis=0)
    span[span == 0] = 1
    centers = centers / span
    centers[:, 0] *= 5.0
    centers[:, 1] *= 7.0
    return centers

letters = list(NAME)
unique_letters = sorted(set(letters))
templates = {ch: letter_to_points(ch) for ch in unique_letters}

X0 = rng.uniform(low=[-4, -2], high=[4, 9], size=(N, 2))  
steps_per_letter = int(T_PER_LETTER / DT)
hold_steps = int(HOLD_TIME / DT)

frames = [X0.copy()]
X = X0.copy()
letter_marks = []      
for letter in letters:
    target = templates[letter].copy()
    offset = X.mean(axis=0) - target.mean(axis=0)
    Xstar = target + offset

    for _ in range(steps_per_letter):
        Xdot = -GAIN * (L @ (X - Xstar))
        X = X + DT * Xdot
        frames.append(X.copy())

    letter_marks.append((len(frames) - 1, letter))
    for _ in range(hold_steps):
        frames.append(X.copy())

print(f"Total animation frames: {len(frames)}  "
      f"(~{len(frames)*DT:.1f} s of simulated time)")


fig, axes = plt.subplots(1, len(letters), figsize=(3 * len(letters), 3.4))
if len(letters) == 1:
    axes = [axes]
edges = list(G.edges())
for ax, (fidx, letter) in zip(axes, letter_marks):
    Xf = frames[fidx]
    for (i, j) in edges:
        ax.plot([Xf[i, 0], Xf[j, 0]], [Xf[i, 1], Xf[j, 1]],
                color="lightgray", linewidth=0.5, zorder=1)
    ax.scatter(Xf[:, 0], Xf[:, 1], c="crimson", s=40, zorder=2)
    ax.set_title(letter, fontsize=16, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
fig.suptitle(f'Formation-control snapshots spelling "{NAME}" (N={N} agents)')
fig.tight_layout()
fig.savefig("figures/p1_letter_snapshots.png", dpi=150)
plt.close(fig)


fig2, ax2 = plt.subplots(figsize=(5, 5))
pos = nx.spring_layout(G, seed=1)
nx.draw_networkx(G, pos, ax=ax2, node_color="crimson", edge_color="gray",
                  node_size=250, font_color="white", font_size=8)
ax2.set_title(f"Erdos-Renyi communication graph  (N={N}, p={P_ER})")
ax2.axis("off")
fig2.tight_layout()
fig2.savefig("figures/p1_graph.png", dpi=150)
plt.close(fig2)


all_pts = np.vstack(frames)
xmin, ymin = all_pts.min(axis=0) - 1
xmax, ymax = all_pts.max(axis=0) + 1

fig3, ax3 = plt.subplots(figsize=(6, 6))
ax3.set_xlim(xmin, xmax); ax3.set_ylim(ymin, ymax)
ax3.set_aspect("equal"); ax3.axis("off")
title = ax3.set_title("")

edge_lines = [ax3.plot([], [], color="lightgray", linewidth=0.6, zorder=1)[0]
              for _ in edges]
scat = ax3.scatter([], [], c="crimson", s=45, zorder=2)


letter_of_frame = []
idx = 0
frames_per_seg = steps_per_letter + hold_steps
for k, letter in enumerate(letters):
    seg_len = steps_per_letter + hold_steps if k > 0 else steps_per_letter + 1 + hold_steps
    for _ in range(seg_len if k > 0 else steps_per_letter + 1 + hold_steps):
        letter_of_frame.append(letter)
letter_of_frame = (letter_of_frame + [letters[-1]] * len(frames))[:len(frames)]

def init():
    scat.set_offsets(np.empty((0, 2)))
    for ln in edge_lines:
        ln.set_data([], [])
    return [scat, *edge_lines, title]

def update(k):
    Xf = frames[k]
    scat.set_offsets(Xf)
    for ln, (i, j) in zip(edge_lines, edges):
        ln.set_data([Xf[i, 0], Xf[j, 0]], [Xf[i, 1], Xf[j, 1]])
    title.set_text(f'Spelling "{NAME}"  |  target letter: {letter_of_frame[k]}   '
                    f'(t = {k*DT:4.1f} s)')
    return [scat, *edge_lines, title]


sub = range(0, len(frames), 3)
ani = animation.FuncAnimation(fig3, update, frames=sub, init_func=init,
                               blit=False, interval=1)
writer = animation.FFMpegWriter(fps=30, bitrate=1800)
ani.save("name_formation.mp4", writer=writer, dpi=120)
plt.close(fig3)
