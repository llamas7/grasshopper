# spherical_grasshopper

Sparse Monte Carlo optimization tools for spherical and Euclidean spin systems.

The actively maintained package in this repository is `refactoredpc/`. It supports:

- spherical antipodal inverse systems
- spherical independent systems
- spherical fixed-density non-antipodal systems
- Euclidean fixed-density systems
- simulated annealing
- greedy local descent
- serial and multiprocessing parallel tempering
- NetCDF save/load via `xarray`
- optional Numba acceleration for the annealing kernels

## Repository layout

- `refactoredpc/` - current package
- `tests/` - `unittest` test suite
- `collect_data.py` - example batch script
- `refactored/` - older code kept in the repo, but not the current API target

## Installation

This project has been exercised with Python 3.12.

```bash
pip install -r requirements.txt
```

Optional GPU neighborhood building requires a CUDA-matched CuPy build. The CPU path is the default and is the only path wired into `System.compute_neighborhood()` right now.

## Core API

The recommended constructors are:

```python
import refactoredpc as rf

system = rf.System.inverse(grid)
system = rf.System.independent(grid)
system = rf.System.non_antipodal(grid)
system = rf.System.euclidean(grid, num_spins=1000)
```

There is also a lower-level constructor if you want to work directly with `SystemConfig`:

```python
system = rf.System(grid, config=rf.SystemConfig.non_antipodal())
```

## Supported system families

| Family | Constructor | Geometry | Constraint | Update rule |
| --- | --- | --- | --- | --- |
| Inverse | `System.inverse(grid)` | Spherical | Antipodal | Flip |
| Independent | `System.independent(grid)` | Spherical | Independent | Flip |
| Non-antipodal | `System.non_antipodal(grid)` | Spherical | Fixed-density | Swap |
| Euclidean | `System.euclidean(grid, num_spins=...)` | Euclidean | Fixed-density | Swap |

For spherical systems the grid is expected to have shape `(N, 3)` with Cartesian coordinates on the sphere. For Euclidean systems the grid is expected to have shape `(N, d)`.

## Typical workflow

1. Construct a system.
2. Build the neighborhood with `compute_neighborhood(...)`.
3. Initialize a state.
4. Optionally thermalize.
5. Run one or more optimizers.
6. Save the result.

### Example: spherical fixed-density workflow

```python
import numpy as np
import refactoredpc as rf

grid = np.loadtxt("path/to/grid.txt")

system = rf.System.non_antipodal(grid)
system.compute_neighborhood(
    jump_distance=0.23 * np.pi,
    delta="two",
)
system.random_init()
system.thermalize(steps_per_temp=100_000)
system.parallel_temper(
    num_replicas=8,
    iterations=50,
    sweep_steps=50_000,
    n_workers=1,
)
system.greedy(reinit=False)
system.save("output/non_antipodal_run.nc")
```

### Example: Euclidean workflow

```python
import numpy as np
import refactoredpc as rf

grid = np.random.rand(5000, 2)

system = rf.System.euclidean(grid, num_spins=1500)
system.compute_neighborhood(
    jump_distance=0.08,
    delta="two",
    distance_metric="euclidean",
)
system.random_init()
system.anneal(
    starting_temp=20.0,
    final_temp=0.05,
    steps=100_000,
    max_iterations=80,
)
```

## Neighborhoods and metrics

Neighborhood construction is explicit and call-time configurable:

```python
system.compute_neighborhood(
    jump_distance=...,
    delta="two",
    distance_metric="spherical",
)
```

Built-in distance metrics include:

- `"spherical"` / `"angular"` / `"geodesic"`
- `"euclidean"` / `"l2"`
- `"manhattan"` / `"cityblock"` / `"l1"`
- `"chebyshev"` / `"linf"`

You can also pass a custom callable metric.

Supported kernel names include:

- `"two"`
- `"one_star"` or `"one*"`
- `"two_star"` or `"two*"`
- `"exact"` / `"delta"` / `"dirac"` / `"real_delta"`

The exact delta kernel only interacts when the distance error is within floating-point tolerance.

## Initialization methods

Depending on the system family, you can initialize with:

- `random_init()`
- `hemisphere_init()` for spherical systems
- `tri_cog_init(num_cogs=..., cog_height=...)`
- `cos_cog_init(num_cogs=..., cog_height=...)`
- `array_init(spin_array)`

## Optimization methods

### Thermalization

```python
system.thermalize(
    target_acceptance=0.6,
    tolerance=0.05,
    steps_per_temp=100_000,
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
)
```

You can also pass an `rf.AnnealSchedule(...)` object as `schedule=...`.

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
)
```

On Windows, multiprocessing parallel tempering should be run from a normal Python script guarded with:

```python
if __name__ == "__main__":
    ...
```

In notebooks, using `n_workers=1` is the least surprising option.

## Histories and results

Each optimizer keeps its own history on the `System`:

- `system.anneal_data`
- `system.greedy_history`
- `system.parallel_tempering_history`

That separation keeps annealing, greedy, and PT records from being mixed together.

## Save and load

Save/load uses a NetCDF file produced through `xarray`:

```python
path = system.save("output/run.nc")
system.load("output/run.nc")
```

If you pass a directory-like path with no suffix:

```python
system.save("output/run_01")
```

the file written is:

```text
output/run_01/system_state.nc
```

Important persistence note:

- the spin state, energies, temperature, and optimizer histories are saved
- the grid is not saved
- the neighborhood is not saved

To load a saved state, first reconstruct a matching `System` with the same grid and configuration, then call `load(...)`. Rebuild the neighborhood after loading if you want to continue optimizing.

## Plotting

The package includes plotting helpers through:

- `system.plot(...)`
- `system.plot_boundary(...)` for spherical systems

These rely on `plotly`, `matplotlib`, `pandas`, and `scipy`.

## Running tests

```bash
python -m unittest discover -s tests -v
```

## Notes on acceleration

- Numba acceleration is used automatically when available for the annealing kernels.
- GPU neighborhood helpers live in `refactoredpc/build_neighbors_gpu.py`.
- GPU support is optional and requires installing a CuPy build compatible with your CUDA version.
