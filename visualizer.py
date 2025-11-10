"""
Visualizer for agent-based simulation data stored in JSON format.
"""
import json
import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def _extract_time_pos_list(series):
    out = []
    for entry in series:
        # detect time
        t = None
        for k in ("tick", "time", "t"):
            if k in entry:
                t = entry[k]
                break
        # detect position
        p = None
        for k in ("position", "pos", "position_vec", "position_raw"):
            if k in entry:
                p = np.asarray(entry[k], dtype=float)
                break
        if p is None:
            # fallback: first list/tuple-like value
            for v in entry.values():
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                    p = np.asarray(v, dtype=float)
                    break
        if t is None or p is None:
            continue
        out.append((float(t), p))
    out.sort(key=lambda x: x[0])
    return out

def build_positions(repo, carry_forward=True):
    # repo: dict agent_id -> list of timeseries dicts
    series_norm = {}
    all_times = set()
    for aid, series in repo.items():
        s = _extract_time_pos_list(series)
        if s:
            series_norm[aid] = s
            all_times.update(t for t, _ in s)
    if not series_norm:
        raise ValueError("No valid timeseries found in repo.")
    frame_times = np.array(sorted(all_times))
    agent_list = list(series_norm.keys())
    # determine dim from first sample
    sample_pos = next(p for s in series_norm.values() for _, p in s)
    dim = sample_pos.shape[0]
    n_agents = len(agent_list)
    n_frames = len(frame_times)
    positions = np.zeros((n_agents, n_frames, dim), dtype=float)
    for i, aid in enumerate(agent_list):
        s = series_norm[aid]
        ptr = 0
        last_pos = s[0][1]
        for j, ft in enumerate(frame_times):
            while ptr < len(s) and s[ptr][0] <= ft:
                last_pos = s[ptr][1]
                ptr += 1
            if carry_forward:
                positions[i, j] = last_pos
            else:
                if ptr > 0 and s[ptr-1][0] == ft:
                    positions[i, j] = s[ptr-1][1]
                else:
                    positions[i, j] = np.nan
    return agent_list, frame_times, positions

def animate_repo_from_json(path, interval=100, carry_forward=True, figsize=(8,6)):
    with open(path, "r") as f:
        repo = json.load(f)
    agent_list, frame_times, positions = build_positions(repo, carry_forward=carry_forward)
    n_agents, n_frames, dim = positions.shape

    fig = plt.figure(figsize=figsize)
    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        scat = ax.scatter(positions[:,0,0], positions[:,0,1], positions[:,0,2])
    else:
        ax = fig.add_subplot(111)
        scat = ax.scatter(positions[:,0,0], positions[:,0,1])
    ax.set_xlabel("X"); ax.set_ylabel("Y")

    mins = np.nanmin(positions.reshape(-1, dim), axis=0)
    maxs = np.nanmax(positions.reshape(-1, dim), axis=0)
    pad = (maxs - mins) * 0.05
    if dim == 3:
        ax.set_xlim(mins[0]-pad[0], maxs[0]+pad[0])
        ax.set_ylim(mins[1]-pad[1], maxs[1]+pad[1])
        #ax.set_zlim(mins[2]-pad[2], maxs[2]+pad[2])
    else:
        ax.set_xlim(mins[0]-pad[0], maxs[0]+pad[0])
        ax.set_ylim(mins[1]-pad[1], maxs[1]+pad[1])

    def update(frame):
        pts = positions[:, frame]
        if dim == 3:
            xs = pts[:,0]; ys = pts[:,1]; zs = pts[:,2]
            scat._offsets3d = (xs, ys, zs)
        else:
            scat.set_offsets(pts[:,:2])
        ax.set_title(f"t = {frame_times[frame]:.2f}")
        return (scat,)

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    plt.show()
    return ani

if __name__ == "__main__":
    # adjust filename/path if needed
    animate_repo_from_json("zombie_simulation_data.json", interval=100)
