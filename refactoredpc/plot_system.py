from __future__ import annotations

import itertools
from .core import ConstraintModel, Geometry, SystemConfig
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from numpy.typing import NDArray
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import scipy as sci
import scipy.interpolate as spi


# =========================
# Utilities: sphere + layout
# =========================


def create_sphere_trace(radius: float = 1.0, resolution: int = 40) -> go.Surface:
    """Return a light wireframe-ish sphere to occlude points."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))

    return go.Surface(
        x=x,
        y=y,
        z=z,
        showscale=False,
        opacity=1,
        surfacecolor=x ** 2 + y ** 2 + z ** 2,
        colorscale=["white", "white"],
        hoverinfo="skip",
    )


def remove_fig_clutter(fig: go.Figure) -> None:
    """Declutter axes, lock aspect, lighten background."""
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        template="plotly_white",
    )


# =========================
# Robust rotation machinery
# =========================


def _skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix [v]_x such that [v]_x @ x == v × x."""
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def rotation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Numerically safe Rodrigues rotation from unit vector a to unit vector b.
    Handles parallel and antiparallel cases explicitly.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    c = float(a @ b)  # cos(theta)

    if c >= 1.0 - 1e-12:  # nearly parallel
        return np.eye(3)

    if c <= -1.0 + 1e-12:  # antiparallel: 180° around any axis ⟂ a
        k = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(a, k)
        u /= np.linalg.norm(u)
        K = _skew(u)
        return np.eye(3) + 2 * (K @ K)

    v = np.cross(a, b)
    s = np.linalg.norm(v)  # sin(theta)
    V = _skew(v / s)  # normalized for stability
    return np.eye(3) + s * V + (1.0 - c) * (V @ V)


def rotate_grid_simple(
        grid: NDArray[np.float64],
        spin_states: NDArray[np.bool_],
        antipodal: bool = True,
        expanded_order: str | None = "base_then_neg",
        north: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Rotate `grid` so the average of the selected spin points aligns with `north`.

    Selection rule:
      - antipodal=True: for each *base* index i, choose either base[i] (if spin False)
        or its antipode (if spin True). The chosen set’s mean defines the rotation.
      - antipodal=False: pick points where spin_states is True.

    Returns:
      rotated_grid : (same shape as grid)
      rotated_selected_points : (K x 3) points used to define the rotation, rotated
    """
    grid = np.asarray(grid, dtype=float)
    spins = np.asarray(spin_states).astype(bool).ravel()

    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError("grid must have shape (M, 3).")

    M = grid.shape[0]

    if antipodal:
        # Expanded: [base, -base]
        N = M // 2
        base = grid[:N]
        neg = grid[N:]
        chosen = np.where(spins[:, None], neg, base)
    else:
        # Nonantipodal
        if len(spins) != M:
            raise ValueError(
                "For antipodal=False, len(spins) must equal grid.shape[0]."
            )
        chosen = grid[spins]

    if chosen.size == 0:
        # Nothing selected -> no rotation
        return grid.copy(), chosen

    rot_vec = chosen.mean(axis=0)
    norm = np.linalg.norm(rot_vec)
    if norm < 1e-12:
        # Ambiguous axis -> no rotation
        return grid.copy(), chosen

    rot_vec /= norm
    R = rotation_matrix(rot_vec, north)

    rotated_grid = (R @ grid.T).T
    rotated_chosen = (R @ chosen.T).T
    return rotated_grid, rotated_chosen


# =========================
# Antipodal expansion utils
# =========================


def _expand_antipodal_single(
        spins: NDArray[np.bool_], antipodal: bool
) -> NDArray[np.bool_]:
    """Single boolean spin per point; flip spins on the neg copy."""
    spins = np.asarray(spins, dtype=bool).ravel()
    if not antipodal:
        return spins
    return np.concatenate([spins, ~spins])


def _expand_antipodal_multi(spins: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Multi-column boolean spins (e.g., [A,B]); flip all columns on the neg copy."""
    spins = np.asarray(spins, dtype=bool)
    return np.vstack([spins, ~spins])


# =========================
# Shared plotting helper
# =========================


def _scatter_on_sphere(
        df: pd.DataFrame,
        color_col: str,
        fig_title: str,
        color_discrete_map: dict | None = None,
        plot_figure: bool = True,
):
    """Consistent Plotly styling for 3D scatter on a sphere."""
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color=color_col,
        color_discrete_map=color_discrete_map or {},
        title=fig_title,
    )
    fig.update_traces(
        marker=dict(size=3, symbol="square", line=dict(width=0)),
        selector=dict(mode="markers"),
    )
    fig.add_trace(create_sphere_trace(0.98))
    remove_fig_clutter(fig)
    fig.update_coloraxes(showscale=False)

    if plot_figure:
        fig.show()
    else:
        return fig


# =========================
# System-specific plotters
# =========================


def graph_system(
        grid_points: NDArray[np.float64],
        spin_states: NDArray[np.bool_],
        fig_title: str = "",
        antipodal: bool = True,
        rotate: bool = False,
        plot_figure: bool = True,
):
    """
    Single boolean spin per point.
    If antipodal=True, points are expanded to [points, -points] and spins are flipped on the neg copy.
    """
    spins = np.asarray(spin_states, dtype=bool).copy().ravel()
    s = _expand_antipodal_single(spins, antipodal=antipodal)
    pts = grid_points.copy()
    if rotate:
        # If expanded -> tell rotate_grid_simple about the ordering; pass base spins
        pts, _ = rotate_grid_simple(pts, spins, antipodal=antipodal)
    data = pd.DataFrame(pts, columns=["x", "y", "z"])
    labels = np.empty(s.shape[0], dtype=object)
    labels[s] = "A"
    labels[~s] = "B"
    color_map = {"A": "yellow", "B": "blue"}
    data["state"] = labels
    return _scatter_on_sphere(
        data,
        color_col="state",
        fig_title=fig_title,
        color_discrete_map=color_map,
        plot_figure=plot_figure,
    )


def graph_independent_system(
        grid_points: NDArray[np.float64],
        spin_states: NDArray[np.bool_],
        fig_title: str = "",
        rotate: bool = False,
        plot_figure: bool = True,
):
    """
    Two boolean spins per point: columns [A, B].
    If antipodal=True, expand to [points, -points] and flip both A & B on the neg copy.
    Colors: A, B, Both, Neither.
    """
    spins = np.asarray(spin_states, dtype=bool).copy()
    if spins.ndim != 2 or spins.shape[1] != 2:
        raise ValueError(
            "graph_independent_system expects spin_states with shape (N, 2) for [A, B]."
        )

    pts = grid_points.copy()
    s = _expand_antipodal_multi(spins)

    if rotate:
        # Use the *base* A-channel to decide the chosen set for rotation (customize if you prefer)
        base_A = spins[:, 0]
        pts, _ = rotate_grid_simple(pts, base_A, antipodal=True)

    A, B = s[:, 0], s[:, 1]
    labels = np.empty(s.shape[0], dtype=object)
    labels[A & B] = "Both"
    labels[A & ~B] = "A"
    labels[~A & B] = "B"
    labels[~A & ~B] = "Neither"

    data = pd.DataFrame(pts, columns=["x", "y", "z"])
    data["state"] = labels

    color_map = {"A": "red", "B": "blue", "Both": "purple", "Neither": "grey"}
    return _scatter_on_sphere(
        data,
        color_col="state",
        fig_title=fig_title,
        color_discrete_map=color_map,
        plot_figure=plot_figure,
    )


def graph_euclidean_system(
        grid_points: NDArray[np.float64],
        spin_states: NDArray[np.bool_],
        fig_title: str = "",
        plot_figure: bool = True,
):
    """Plot a Euclidean grid in 1D, 2D, or 3D using the first three coordinates."""
    pts = np.asarray(grid_points, dtype=float)
    spins = np.asarray(spin_states, dtype=bool).ravel()
    if pts.ndim != 2:
        raise ValueError("grid_points must be a 2D array.")
    if len(spins) != pts.shape[0]:
        raise ValueError("len(spin_states) must equal grid_points.shape[0].")

    labels = np.empty(spins.shape[0], dtype=object)
    labels[spins] = "A"
    labels[~spins] = "B"
    color_map = {"A": "yellow", "B": "blue"}

    if pts.shape[1] == 1:
        data = pd.DataFrame(
            {"x": pts[:, 0], "state_value": spins.astype(int), "state": labels}
        )
        fig = px.scatter(
            data,
            x="x",
            y="state_value",
            color="state",
            color_discrete_map=color_map,
            title=fig_title,
        )
        fig.update_yaxes(visible=False)
    elif pts.shape[1] == 2:
        data = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "state": labels})
        fig = px.scatter(
            data,
            x="x",
            y="y",
            color="state",
            color_discrete_map=color_map,
            title=fig_title,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    else:
        data = pd.DataFrame(
            {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2], "state": labels}
        )
        fig = px.scatter_3d(
            data,
            x="x",
            y="y",
            z="z",
            color="state",
            color_discrete_map=color_map,
            title=fig_title,
        )
        fig.update_layout(scene=dict(aspectmode="data"))

    fig.update_traces(marker=dict(size=4, line=dict(width=0)))
    fig.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=60, b=0))
    if plot_figure:
        fig.show()
    else:
        return fig


# =========================
# Unified wrapper
# =========================


def plot_spin_system(
        *,
        system_config: SystemConfig,
        grid_points: NDArray[np.float64],
        spin_states: NDArray[np.bool_],
        fig_title: str = "",
        rotate: bool = False,
        plot_figure: bool = True,
):
    """
    Dispatcher for inverse, independent, non-antipodal, and Euclidean systems:
      - 'INVERSE'       → single spin, antipodal
      - 'INDEPENDENT'   → two spins [A,B], antipodal
      - 'NONANTIPODAL'  → single spin, no antipodal
    """
    if system_config.geometry == Geometry.EUCLIDEAN:
        return graph_euclidean_system(
            grid_points,
            spin_states,
            fig_title=fig_title,
            plot_figure=plot_figure,
        )

    if system_config.constraint == ConstraintModel.INDEPENDENT:
        return graph_independent_system(
            grid_points,
            spin_states,
            fig_title=fig_title,
            rotate=rotate,
            plot_figure=plot_figure,
        )

    if system_config.antipodal:
        return graph_system(
            grid_points,
            spin_states,
            fig_title=fig_title,
            antipodal=True,
            rotate=rotate,
            plot_figure=plot_figure,
        )

    return graph_system(
        grid_points,
        spin_states,
        fig_title=fig_title,
        antipodal=False,
        rotate=rotate,
        plot_figure=plot_figure,
    )


# ==================
# Boundary Plotting
# ==================


def convert_ss(xyz):
    ptsnew = np.zeros((len(xyz), 2))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    ptsnew[:, 0] = np.arctan2(xyz[:, 1], xyz[:, 0])
    ptsnew[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])
    return ptsnew


def process_data(surface_points, get_shift=False):
    """
    Reorders and recenters surface data for periodic analysis.

    Parameters:
        surface_points (ndarray): N × 2 array where:
            - surface_points[:, 0] contains azimuthal angles (phi),
            - surface_points[:, 1] contains surface profile values (theta(phi)).

    Returns:
        phi_aligned (ndarray): Recentered and rotated phi values, starting at 0.
        theta_centered (ndarray): Corresponding surface profile, mean-centered and reordered.
    """
    # Extract azimuthal angles and surface values
    phi = surface_points[:, 0]
    theta = surface_points[:, 1]

    # Sort data by increasing phi
    sort_indices = np.argsort(phi)
    phi = phi[sort_indices]
    theta = theta[sort_indices]

    # Rotate so that the maximum of theta is at the beginning
    max_idx = np.argmax(theta)
    phi[:max_idx] += 2 * np.pi  # wrap-around for periodic continuity
    phi_rotated = np.concatenate((phi[max_idx:], phi[:max_idx]))
    theta_rotated = np.concatenate((theta[max_idx:], theta[:max_idx]))

    # Center theta and shift phi to start at 0
    theta_centered = theta_rotated - np.mean(theta_rotated)
    phi_aligned = phi_rotated - phi_rotated[0]

    if get_shift:
        return phi_aligned, theta_centered, (phi_rotated[0], np.mean(theta_rotated))
    else:
        return phi_aligned, theta_centered


def extract_boundary_points(grid, spin_states, limit=0.75, antipodal=True):
    h = np.sqrt(2 * np.pi / len(spin_states))
    grid_tree = KDTree(grid)
    nearest_neighbors = grid_tree.query_ball_point(grid, 2 * h, p=2)
    ratios = []
    spins = spin_states.copy()
    if antipodal:
        spins = np.hstack((spins, np.invert(spins)))
    for neighbors in nearest_neighbors:
        num_neighbors = len(neighbors)
        num_up = np.sum(spins[neighbors])
        num_down = num_neighbors - num_up
        ratio_up = num_up / num_neighbors
        ratio_down = num_down / num_neighbors
        if ratio_up > 0.5:
            ratios.append(ratio_up)
        else:
            ratios.append(ratio_down)
    ratios = np.array(ratios)
    boundary_points = grid[(ratios < limit)]
    return boundary_points


def graph_boundary(
        grid,
        spin_states,
        limit=0.75,
        antipodal=True,
        independent=False,
        plot_figure=True,
        ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure  # so we still return a valid fig

    if independent and not antipodal:
        raise ValueError(
            "Independent boundary points are only available for antipodal systems"
        )
    elif independent:
        rotated_points, lawn_pointsA = rotate_grid_simple(
            grid, spin_states[:, 0], antipodal=antipodal
        )
        boundary_pointsA = extract_boundary_points(
            rotated_points, spin_states[:, 0], limit=limit, antipodal=antipodal
        )
        boundary_pointsB = extract_boundary_points(
            rotated_points, spin_states[:, 1], limit=limit, antipodal=antipodal
        )
        process_data(boundary_pointsA)
        surface_pointsA = convert_ss(boundary_pointsA)
        surface_pointsB = convert_ss(boundary_pointsB)
        phi_movedA, theta_movedA, shiftA = process_data(surface_pointsA, get_shift=True)
        phi_movedB, theta_movedB, shiftB = process_data(surface_pointsB, get_shift=True)
        phi_movedB += -shiftA[0] + shiftB[0]
        theta_movedB += -shiftA[1] + shiftB[1]
        phi_movedB[phi_movedB > 2 * np.pi] -= 2 * np.pi
        phi_movedB[phi_movedB < 0] += 2 * np.pi

        ax.scatter(phi_movedA, theta_movedA, marker=".", s=3.5, c="r")
        ax.scatter(phi_movedB, theta_movedB, marker=".", s=3.5, c="b")
    else:
        rotated_points, lawn_points = rotate_grid_simple(
            grid, spin_states, antipodal=antipodal
        )
        boundary_points = extract_boundary_points(
            rotated_points, spin_states, limit=limit, antipodal=antipodal
        )
        surface_points = convert_ss(boundary_points)
        phi_moved, theta_moved = process_data(surface_points)
        ax.scatter(phi_moved, theta_moved, marker=".", s=3.5)

    ax.tick_params(axis="both", labelsize=15)
    fig.tight_layout()

    if plot_figure:
        plt.show()
    else:
        return fig, ax


def plot_fft_spectrum(grid, spin_states, ax=None, label=None, limit_freq=140):
    """
    Plot the FFT spectrum of a spin configuration on a given axis.

    Parameters:
    - grid: underlying coordinate grid
    - spin_states: configuration array
    - ax: optional matplotlib axis to draw on (if None, creates one)
    - label: optional label for the plot
    - limit_freq: max frequency to display on x-axis
    """
    if ax is None:
        fig, ax = plt.subplots()

    fft_x, fft_y = fft_values(grid, spin_states)

    ax.plot(fft_x, fft_y, label=label)
    ax.set_xlim(0, limit_freq)
    ax.set_ylim(0, 1.05 * np.max(fft_y))
    ax.set_xlabel("Fourier mode")
    ax.set_ylabel("Amplitude")
    if label:
        ax.legend()
    return ax


def fft_values(grid, spin_states, include_interp=False, antipodal=True):
    rotated_grid, spins_xyz = rotate_grid_simple(grid, spin_states, antipodal=antipodal)
    boundary_points = extract_boundary_points(
        rotated_grid, spin_states, limit=0.65, antipodal=antipodal
    )
    ss_points = convert_ss(boundary_points)
    phi_moved, theta_moved = process_data(ss_points)
    ss_points = np.asarray([phi_moved, theta_moved])
    spline = ss_points.copy()
    N = len(spline)
    T = 1 / N
    argsort = np.argsort(spline[:, 0])
    spline = spline[argsort]
    interp1d = spi.interp1d(phi_moved, theta_moved)
    x_values = np.linspace(np.min(phi_moved), np.max(phi_moved), N)
    y_values = interp1d(x_values)
    y_values -= np.mean(y_values)

    fft_y = sci.fft.fft(y_values)
    fft_x = sci.fft.fftfreq(len(x_values), T)[: N // 2]
    fft_normalized = 2.0 / N * np.abs(fft_y[: N // 2])

    if include_interp:
        return fft_x, fft_normalized, interp1d

    else:
        return fft_x, fft_normalized
