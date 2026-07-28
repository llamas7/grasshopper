# grasshopper

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21462699.svg)](https://doi.org/10.5281/zenodo.21462699)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Software citation:** David Llamas (2026), *llamas7/grasshopper: grasshopper - initial release*, v1.0.0, Zenodo. [https://doi.org/10.5281/zenodo.21462699](https://doi.org/10.5281/zenodo.21462699)

**Related publications:**

- David Llamas, Jaron Kent-Dobias, Kun Chen, Adrian Kent, and Olga Goulko, "Origin of Symmetry Breaking in the Grasshopper Model," *Physical Review Research* **6**, 023235 (2024). [https://doi.org/10.1103/PhysRevResearch.6.023235](https://doi.org/10.1103/PhysRevResearch.6.023235)
- David Llamas, Dmitry Chistikov, Adrian Kent, Mike Paterson, and Olga Goulko, "Optimal Local Simulations of a Quantum Singlet" (2025). [https://doi.org/10.48550/arXiv.2504.20953](https://doi.org/10.48550/arXiv.2504.20953)
- David Llamas, Dmitry Chistikov, Adrian Kent, Mike Paterson, and Olga Goulko, "The Grasshopper Problem on the Sphere" (2026). [https://doi.org/10.48550/arXiv.2603.08579](https://doi.org/10.48550/arXiv.2603.08579)

**Funding:** This work was supported by the National Science Foundation under Grant Nos. PHY-2112738 ("CQIS: The Grasshopper Problem") and OSI-2328774 ("ExpandQISE: Track 2: EQUIP-UMB"). David Llamas also acknowledges support from the College of Science and Mathematics Dean's Doctoral Research Fellowship, made possible through fellowship support from Oracle under project ID R0000000025727.

**License:** This software is distributed under the [MIT License](LICENSE).

`grasshopper` is a Python library for sparse spin-system optimization on spherical and Euclidean grids.

It includes:

- spherical complementary systems
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

## Package layout

- [`grasshopper/`](grasshopper/) - installable library package
- [`tests/`](tests/) - `unittest` suite
- [`examples/collect_gaussian.py`](examples/collect_gaussian.py) - adaptive
  Gaussian phase-transition scan
- [`examples/parallel_tempering.py`](examples/parallel_tempering.py) -
  runnable replica-exchange demonstration
- [`pyproject.toml`](pyproject.toml) - build metadata

## Quick start

```python
import numpy as np
import grasshopper as gh

grid = np.loadtxt("path/to/grid.txt")

system = gh.System.non_antipodal(grid)
system.compute_neighborhood(
    jump_distance=0.23 * np.pi,
    kernel=gh.KernelSpec.cosine(),
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
print(system.best_probability)
system.save("output/non_antipodal_run.nc")
```

Setup methods can be chained:

```python
system = (
    gh.System.complementary(grid)
    .compute_neighborhood(jump_distance=np.pi / 2)
    .random_init(seed=123)
)
```

Use `system.state_summary()` to inspect the current configuration, loaded
metadata, and whether the system is ready for optimization.

## Constructors

The recommended entrypoints are:

```python
system = gh.System.complementary(grid)
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
| Complementary | `System.complementary(grid)` | Spherical | Antipodal | Flip |
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
    kernel=gh.KernelSpec.cosine(),
    distance_metric="spherical",
)
```

Built-in distance metrics:

- `"spherical"` / `"angular"` / `"geodesic"`
- `"euclidean"` / `"l2"`
- `"manhattan"` / `"cityblock"` / `"l1"`
- `"chebyshev"` / `"linf"`

You can also provide a custom callable metric.

Built-in kernels are represented by immutable `KernelSpec` objects:

```python
gh.KernelSpec.cosine()
gh.KernelSpec.cell_averaged_hat()
gh.KernelSpec.cell_averaged_cosine()
gh.KernelSpec.exact()
gh.KernelSpec.gaussian(sigma=0.05, cutoff=5.0)
```

The compact cosine kernel has support `2h`. The cell-averaged hat kernel is
the quadratic B-spline obtained by averaging the linear hat function across
one grid cell and has support `3h/2`. The cell-averaged cosine kernel applies
the same averaging operation to the compact cosine kernel and has support
`5h/2`.

The Gaussian width is a physical distance in the same units as
`jump_distance`. Its default sparse neighborhood truncates the Gaussian at
five standard deviations. Pass `cutoff=None` to retain the full tail with an
all-pairs scan.

String kernel names remain accepted through `kernel=`, including `"cosine"`,
`"cell_averaged_hat"`, `"cell_averaged_cosine"`, and `"exact"`. Unknown names
raise `ValueError`.
Custom callable kernels may also be passed through `kernel=...`; they should
accept `(distance_errors, cell_length)` and return `(contributions, mask)`.

Jump distances are validated physically: Euclidean distances must be
nonnegative, and spherical angular distances must lie in `[0, pi]`.

Spherical systems also accept multiple weighted jump distances in one
neighborhood build:

```python
system.compute_neighborhood(
    jump_distance=[0.485 * np.pi, 0.515 * np.pi],
    jump_weights=[1.0, 0.75],
    kernel=gh.KernelSpec.cosine(),
)
```

Each shell contributes `weight * kernel(distance_to_shell)`. If the same
neighbor is reached by more than one shell, the edge weights are summed into a
single row entry. When `jump_weights` is omitted, every shell uses weight
`1.0`. Euclidean systems currently support only a scalar `jump_distance`.

## Energy and probability

Optimization continues to maximize the unnormalized pair energy. Probability
normalization is applied only when requested:

```python
system.random_init(seed=123)

p_current = system.current_probability
p_best = system.best_probability
p_other = system.probability(other_boolean_state)
```

These use the continuum Grasshopper functional. For Euclidean fixed-density
systems with `K` occupied cells in dimension `D`, the compact-kernel
prefactor is

```text
2 / (S_D(d) K^(2 - 1/D)).
```

For spherical systems it uses the geodesic-circle measure
`2 pi sin(theta)`. Gaussian kernels instead use the corresponding radial
integral over their implemented, possibly truncated, profile.

Finite grids can also be normalized by their available outgoing interaction
weight:

```python
p_graph = system.current_graph_probability
p_graph_best = system.best_graph_probability
p_graph_other = system.probability(other_boolean_state, normalization="graph")
```

Graph normalization is a discretization diagnostic and need not equal the
continuum-normalized value on a coarse or irregular grid. Probabilities are
not clamped, so discretization errors remain visible.

Continuum normalization is built in for spherical geodesic distance and
Euclidean L2 distance. Other metrics require their radial shell measure:

```python
system.compute_neighborhood(
    jump_distance=1.0,
    distance_metric="manhattan",
    shell_measure=lambda radius: 8.0 * radius,
    radial_bounds=(0.0, np.inf),
)
```

Exact diagnostic and arbitrary callable kernels support graph normalization
only. Graph normalization also requires nonnegative interaction weights.

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

`steps` sets a fixed number of Metropolis updates at every temperature. The
current `AnnealSchedule` fields are `starting_temp`, `final_temp`,
`cooling_rate`, `min_acceptance`, `max_runtime_hours`, `max_iterations`, and
`starting_steps`; it does not automatically increase the step count while
cooling.

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

A complete square-lattice replica-exchange demonstration is available as a
runnable script:

```bash
python examples/parallel_tempering.py
```

### Adaptive Gaussian transition scans

The Gaussian collector starts with a small set of eta anchors, refines detected
phase boundaries, adds runs where morphology labels disagree, and periodically
checks broad same-label intervals:

```bash
python examples/collect_gaussian.py transitions \
    --jump-distance 0.60 \
    --eta-min 0.0 \
    --eta-max 0.6
```

It resumes compatible records in the output directory automatically. Use
`--uniform` for the original fixed-step eta grid, or `--overwrite` to restart
only the matching scan. By default, the first unambiguous bracket between a
nonrotational and rotational morphology suspends all higher initial anchors.
The scheduler then bisects that bracket to locate the restoration threshold.
Use `--continue-after-restoration` to continue toward `--eta-max`, for example
when searching for a later disk-to-annulus transition. Adaptive progress, the
current restoration bracket, any persistence check, and selection reasons are
written to `adaptive_scan.json`.

Completed transition scans can be combined into an updated lawn-morphology
sequence and a preliminary \((d,\sigma/d)\) diagnostic diagram with:

```bash
python examples/plot_gaussian_morphology_diagram.py \
    --input-dir Gauss_data/transitions \
    --output-dir Gauss_data/morphology_diagram
```

The morphology sequence uses the best stored states at selected widths. The
diagnostic diagram shows only sampled points, marks run-to-run label
disagreement, and distinguishes resolved rotational-restoration brackets from
incomplete ones rather than interpolating regions that have not been
calculated.

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
`save()` returns the path written; `load()` mutates the existing system and
returns `None`.

Important persistence note:

- spin state, energies, temperature, and optimizer histories are saved
- neighborhood construction metadata is saved, including scalar jump distance
  or spherical multi-shell jump distances/weights, kernel, Gaussian
  width/cutoff, distance metric, normalization metadata, grid shape/dtype, and
  neighborhood storage summary
- Euclidean saved metadata includes both candidate-site count (`site_count`) and
  occupied-site count (`num_spins`)
- the grid is not saved
- the neighborhood is not saved

To load a saved state, rebuild a matching `System` with the same grid and
configuration, then call `load(...)`. Rebuild the neighborhood afterward if
you want to continue optimizing. For saved states that used built-in kernels
and distance metrics, `compute_neighborhood()` reuses the loaded metadata when
you omit those arguments. If the original kernel, metric, or shell measure was
a custom callable, pass it again while rebuilding. Format `5.0` is written;
format `4.0` checkpoints are also read and their complementary-model metadata
is migrated during loading.

Built-in metadata can be rebuilt automatically:

```python
loaded = gh.System.non_antipodal(grid)
loaded.load("output/run.nc", rebuild_neighborhood=True)
```

The default remains `rebuild_neighborhood=False`. Automatic rebuilding raises
a clear error when the saved run used a custom kernel, distance metric, or
shell measure; load without rebuilding and call `compute_neighborhood(...)`
with the custom callable in that case.

## Cookbook

### Spherical optimization

```python
system = (
    gh.System.complementary(grid)
    .compute_neighborhood(jump_distance=np.pi / 2)
    .hemisphere_init(seed=123)
)
system.anneal(steps=100_000, seed=456)
print(system.best_probability)
```

### Euclidean lattice optimization

```python
system = (
    gh.System.square_lattice(64, 64, density=0.5)
    .compute_neighborhood(
        jump_distance=0.60,
        distance_metric="euclidean",
    )
    .random_init(seed=123)
)
result = system.greedy(max_steps=1000)
```

### Weighted spherical shells

```python
system = (
    gh.System.non_antipodal(grid)
    .compute_neighborhood(
        jump_distance=[0.485 * np.pi, 0.515 * np.pi],
        jump_weights=[1.0, 0.75],
    )
    .random_init(seed=123)
)
```

### Inspecting readiness

```python
summary = system.state_summary()
print(summary["ready"], summary["kernel"], summary["jump_distance"])
```

## Common Errors

- Build the neighborhood before calling an initializer. Initializers raise a
  `RuntimeError` with the required `compute_neighborhood()` call otherwise.
- Initialize the system before annealing, greedy optimization, parallel
  tempering, or plotting a spherical spin state.
- Build a neighborhood before initialization or `plot_neighbors()`.
- Multiple jump distances are supported only for spherical systems.
- `plot_neighbors()` and Voronoi plotting are available only for 2D Euclidean
  systems; `plot_boundary()` is for spherical systems.
- Loading restores spin state and neighborhood metadata, but the neighborhood
  itself is rebuilt only when `rebuild_neighborhood=True` or when
  `compute_neighborhood()` is called afterward.
- Saved custom kernels, distance metrics, and shell measures cannot be
  reconstructed from NetCDF metadata. Supply those callables again manually.

## Plotting

The package includes:

- `system.plot(...)`
- `system.plot_neighbors(selected_site=...)` for 2D Euclidean
  neighbor/contribution views
- `system.plot_boundary(...)` for spherical systems

For regular 2D lattices:

```python
system = gh.System.hexagonal_lattice(16, 16, density=0.5)
system.compute_neighborhood(jump_distance=0.30)
system.plot_neighbors(selected_site=42)

system.random_init(seed=123)
system.plot(plot_current=True, show_edges=True)
```

Euclidean system plots use Matplotlib and return a Matplotlib `Figure` when
`plot_figure=False`. Spherical plots remain interactive Plotly figures.
Voronoi construction and neighborhood diagnostics also rely on SciPy.

## Examples

The introductory examples are short Python scripts that run from the
repository root:

```bash
python examples/spherical_complementary.py
python examples/spherical_independent.py
python examples/spherical_non_antipodal.py
python examples/euclidean_2d.py
python examples/euclidean_3d.py
python examples/gaussian_kernel.py
python examples/parallel_tempering.py
```

The spherical examples generate small antipodally paired Fibonacci grids and
write interactive Plotly HTML files. The Euclidean examples write Matplotlib
PNG files. Outputs are placed in a script-specific directory below
`examples/output/`.

Each script accepts `--help`. Common optimization controls include `--steps`,
`--anneal-iterations`, `--greedy-steps`, and `--seed`. Increase the grid size
and these optimization limits for serious numerical searches; the defaults
are intentionally small demonstrations and should not be interpreted as
globally optimized lawns. Use `--show` to display the generated figure in
addition to saving it.

The Gaussian example compares the compact cosine delta regularization with
`KernelSpec.gaussian(...)` from the same initial lawn. The parallel-tempering
example additionally reports its temperature ladder and replica-exchange
rates. The larger `collect_gaussian.py` and
`collect_gaussian_figure_data.py` programs are research data-collection tools
rather than introductory examples.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Acceleration notes

- Numba acceleration is used automatically when available.
- GPU neighborhood helpers live in
  [`grasshopper/build_neighbors_gpu.py`](grasshopper/build_neighbors_gpu.py).
- GPU support is optional and requires installing a CuPy build that matches your CUDA setup.
