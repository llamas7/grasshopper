# grasshopper

`grasshopper` is a Python library for sparse spin-system optimization on spherical and Euclidean grids.

It includes:

- spherical inverse systems
- spherical independent systems
- spherical fixed-density non-antipodal systems
- Euclidean fixed-density systems
- simulated annealing
- greedy local descent
- serial and multiprocessing parallel tempering
- NetCDF save/load through `xarray`
- optional Numba acceleration for annealing kernels

## Installation

For local development:

```bash
pip install -e .
```

If you prefer using the repository requirements file:

```bash
pip install -r requirements.txt
```

Optional GPU neighborhood helpers require a CUDA-matched CuPy build. The default library path is CPU-based.

## Codex Runtime

This repository includes a repo-local Python runtime at `.codex-python/` for Codex desktop sessions. Inside this project, the intended test command is:

```bash
python -m unittest discover -s tests -v
```

If a newly opened Codex conversation still does not resolve `python`, restart the Codex app so it picks up the updated user `PATH`.

## Package layout

- [grasshopper](/C:/Users/polka/OneDrive/Documents/Codex/spherical_grasshopper/grasshopper) - installable library package
- [tests](/C:/Users/polka/OneDrive/Documents/Codex/spherical_grasshopper/tests) - `unittest` suite
- [collect_data.py](/C:/Users/polka/OneDrive/Documents/Codex/spherical_grasshopper/collect_data.py) - example batch script
- [pyproject.toml](/C:/Users/polka/OneDrive/Documents/Codex/spherical_grasshopper/pyproject.toml) - build metadata

## Quick start

```python
import numpy as np
import grasshopper as gh

grid = np.loadtxt("path/to/grid.txt")

system = gh.System.non_antipodal(grid)
system.compute_neighborhood(
    jump_distance=0.23 * np.pi,
    delta="two",
)
system.random_init(seed=123)
system.thermalize(steps_per_temp=100_000, seed=456)
system.parallel_temper(
    num_replicas=8,
    iterations=50,
    sweep_steps=50_000,
    n_workers=1,
    seed=789,
)
system.greedy(reinit=False)
system.save("output/non_antipodal_run.nc")
```

## Constructors

The recommended entrypoints are:

```python
system = gh.System.inverse(grid)
system = gh.System.independent(grid)
system = gh.System.non_antipodal(grid)
system = gh.System.euclidean(grid, num_spins=1000)
system = gh.System.square_lattice(width=64, height=64, density=0.5)
system = gh.System.hexagonal_lattice(width=32, height=32, num_spins=512)
```

There is also a lower-level explicit constructor:

```python
system = gh.System(grid, config=gh.SystemConfig.non_antipodal())
```

## Supported system families

| Family | Constructor | Geometry | Constraint | Update rule |
| --- | --- | --- | --- | --- |
| Inverse | `System.inverse(grid)` | Spherical | Antipodal | Flip |
| Independent | `System.independent(grid)` | Spherical | Independent | Flip |
| Non-antipodal | `System.non_antipodal(grid)` | Spherical | Fixed-density | Swap |
| Euclidean | `System.euclidean(grid, num_spins=...)` | Euclidean | Fixed-density | Swap |

Spherical grids should have shape `(N, 3)` with Cartesian coordinates on the unit sphere. Euclidean grids should have shape `(N, d)`.

## Regular 2D lattices

For Euclidean fixed-density systems on common regular lattices, build the grid
and system in one call:

```python
square = gh.System.square_lattice(64, 64, density=0.5)
rectangular = gh.System.rectangular_lattice(64, 32, density=0.5)
triangular = gh.System.triangular_lattice(64, 64, num_spins=2048)
hexagonal = gh.System.hexagonal_lattice(32, 32, density=0.5)
```

Euclidean systems use a fixed total occupied spin-cell area of `1.0` by
default. If `num_spins=k`, each occupied spin has area `1 / k`, so the 2D
length scale is `1 / sqrt(k)` even when the candidate lattice has more than
`k` sites. Raw `System.euclidean(grid, ...)` inputs and regular-lattice system
constructors normalize the candidate grid to volume `site_count / k` by
default. For example, with `num_spins=1000`, `jump_distance=0.30` has the same
physical meaning whether the square candidate grid is coarse or refined.

The exported grid builders, such as `gh.square_lattice(...)`, only create raw
candidate-site coordinates. For example, `gh.square_lattice(100, 100)` still
returns spacing `0.01`. The occupied-site normalization happens when those
coordinates are wrapped in `System.euclidean(...)` or created through a
`System.*_lattice(...)` constructor. To control a custom candidate geometry,
build the raw coordinates directly and pass them to `System.euclidean(...)`;
the system will still set the final physical scale from `num_spins`.

`hexagonal_lattice()` builds the honeycomb/hexagonal tiling lattice with two
sites per unit cell. `triangular_lattice()` builds the triangular Bravais
lattice with six nearest neighbors in the interior.

The same grid constructors are also exported directly:

```python
grid = gh.square_lattice(64, 64, center=True)
system = gh.System.euclidean(grid, num_spins=grid.shape[0] // 2)
```

All regular-lattice system constructors accept either `num_spins=` or
`density=`. If both are omitted, they default to half filling.

## Neighborhood construction

Neighborhoods are configured at call time:

```python
system.compute_neighborhood(
    jump_distance=...,
    delta="two",
    distance_metric="spherical",
)
```

Built-in distance metrics:

- `"spherical"` / `"angular"` / `"geodesic"`
- `"euclidean"` / `"l2"`
- `"manhattan"` / `"cityblock"` / `"l1"`
- `"chebyshev"` / `"linf"`

You can also provide a custom callable metric.

Supported kernel names:

- `"two"`
- `"one_star"` or `"one*"`
- `"two_star"` or `"two*"`
- `"exact"` / `"delta"` / `"dirac"` / `"real_delta"`

Unknown kernel names raise `ValueError` at every API layer. You can also pass a
custom callable kernel as `delta=...`; it should accept `(distances,
cell_length)` and return `(contributions, mask)`.

## Initialization

Depending on the system family, you can initialize with:

- `random_init(seed=...)`
- `hemisphere_init(seed=...)` for spherical systems
- `tri_cog_init(num_cogs=..., cog_height=...)`
- `cos_cog_init(num_cogs=..., cog_height=...)`
- `array_init(spin_array)`

Passing `seed=` to stochastic methods makes that call reproducible without
advancing the caller's global NumPy random state. The stochastic methods with
seed support are `random_init`, `hemisphere_init`, `thermalize`, `anneal`, and
`parallel_temper`.

## Optimization

### Thermalization

```python
system.thermalize(
    target_acceptance=0.6,
    tolerance=0.05,
    steps_per_temp=100_000,
    seed=123,
)
```

This updates the live runtime temperature exposed as `system.temperature`.

### Simulated annealing

```python
system.anneal(
    starting_temp=50.0,
    final_temp=0.05,
    cooling_rate=0.954993,
    steps=100_000,
    max_iterations=100,
    save_path="output/anneal.nc",
    save_interval=5,
    seed=123,
)
```

You can also pass `schedule=gh.AnnealSchedule(...)`.

### Greedy descent

```python
result = system.greedy(
    max_steps=1000,
    min_delta=0.0,
    reinit=False,
)
```

### Parallel tempering

```python
result = system.parallel_temper(
    num_replicas=16,
    iterations=100,
    sweep_steps=100_000,
    cold_temp=0.05,
    hot_temp=system.temperature,
    n_workers=4,
    save_path="output/pt.nc",
    seed=123,
)
```

On Windows, multiprocessing parallel tempering should be run from a normal Python script guarded with:

```python
if __name__ == "__main__":
    ...
```

In notebooks, `n_workers=1` is the safest default.

A complete square-lattice demonstration using `jump_distance=0.60` is available
as a runnable script:

```bash
python examples/parallel_tempering_square_lattice_demo.py
```

## Histories

Each optimizer records its own history on the system:

- `system.anneal_data`
- `system.greedy_history`
- `system.parallel_tempering_history`

## Save and load

Save/load uses a NetCDF file written through `xarray`:

```python
path = system.save("output/run.nc")
system.load("output/run.nc")
```

If you pass a path with no suffix, the library treats it as a directory and writes `system_state.nc` inside it.

Important persistence note:

- spin state, energies, temperature, and optimizer histories are saved
- neighborhood construction metadata is saved, including jump distance, kernel,
  distance metric, probability prefactor, grid shape/dtype, and neighborhood
  storage summary
- Euclidean saved metadata includes both candidate-site count (`site_count`) and
  occupied-site count (`num_spins`)
- the grid is not saved
- the neighborhood is not saved

To load a saved state, rebuild a matching `System` with the same grid and configuration, then call `load(...)`. Rebuild the neighborhood afterward if you want to continue optimizing. For saved states that used built-in kernel and distance metric names, `compute_neighborhood()` reuses the loaded jump distance, kernel, and distance metric when you omit them. If the original kernel or metric was a custom callable, pass that callable again with `delta=...` or `distance_metric=...`.

## Plotting

The package includes:

- `system.plot(...)`
- `system.plot_neighbors(...)` for 2D Euclidean neighbor/correlation views
- `system.plot_boundary(...)` for spherical systems

For regular 2D lattices:

```python
system = gh.System.hexagonal_lattice(16, 16, density=0.5)
system.compute_neighborhood(jump_distance=0.30)
system.plot_neighbors(selected_site=42)

system.random_init(seed=123)
system.plot(plot_current=True, show_edges=True)
```

These rely on `plotly`, `matplotlib`, `pandas`, and `scipy`.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Acceleration notes

- Numba acceleration is used automatically when available.
- GPU neighborhood helpers live in [grasshopper/build_neighbors_gpu.py](/C:/Users/polka/OneDrive/Documents/Codex/spherical_grasshopper/grasshopper/build_neighbors_gpu.py).
- GPU support is optional and requires installing a CuPy build that matches your CUDA setup.
