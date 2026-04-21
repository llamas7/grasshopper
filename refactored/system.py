from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from .annealing import simulated_annealing, thermalize_system
from .build_neighbors import build_neighborhood, calculate_prob_prefactor
from .core import DeltaType, SystemType, NeighborhoodLike, Parameters, AnnealData
from .initialization import (
    init_random,
    init_random_hemisphere,
    init_from_array,
    init_cosine_cogwheel,
    init_triangle_cogwheel,
)
from .plot_system import plot_spin_system, graph_boundary


@dataclass
class System:
    """
    Representation of a spin system defined on a spherical point lattice.

    This class encapsulates all state, metadata, and operations required to run
    Monte-Carlo-based optimization (thermalization + simulated annealing) for the
    spherical grasshopper system. A `System` instance owns:

    • the spherical grid (Nx3 Cartesian coordinates),
    • system type (INVERSE, INDEPENDENT, NON_ANTIPODAL),
    • jump distance and δ-type (determines neighborhood connectivity),
    • computed neighborhood structure,
    • current and best spin configurations,
    • per-step annealing data (temperature, energy history, acceptance statistics),
    • save/load routines for reproducibility.

    The lifecycle of a `System` object is:

    1. **Construction**
       Provide a grid and jump distance. The constructor normalizes enums,
       validates grid shape, infers antipodal/non-antipodal rules, computes
       the number of spins, and prepares an `AnnealData` container.

    2. **Neighborhood construction**
       Call `compute_neighborhood()`. This builds the adjacency list used by
       all Metropolis updates and also sets the probability prefactor needed
       for computing Boltzmann factors.

    3. **Initialization**
       Choose one of:
           • `random_init()`
           • `hemisphere_init()`
           • `tri_cog_init()`
           • `cos_cog_init()`
           • `array_init(spin_array)`
       These create `current_system`, compute its energy, and record the initial
       “best” configuration.

    4. **Thermalization (optional, but recommended)**
       `thermalize()` adaptively adjusts the temperature to achieve a target
       Metropolis acceptance rate before annealing begins. The adjustment uses
       a proportional controller in log-temperature space for stability and
       scale-invariance.

    5. **Simulated Annealing**
       Calling `anneal()` executes the annealing schedule defined in
       `Parameters`, updates spin states in place, and records best energies
       and acceptance statistics.

    6. **Saving and Loading**
       `save(path)` writes the spin states and anneal statistics to NetCDF/CSV.
       Note that **grid and neighborhood are not saved**; reconstructing a system
       after loading requires rebuilding the neighborhood manually.
       `load(path)` restores spin arrays, energies, parameters, and anneal data
       while validating consistency with the current grid.

    Attributes
    ----------
    grid : ndarray of shape (N, 3)
        Cartesian coordinates of lattice points on the sphere.
    jump_distance : float
        Maximum distance used to construct neighborhood lists.
    system_type : SystemType
        Determines antipodal constraints and spin dimensionality.
    delta : DeltaType
        Delta discretization function used for neighborhood construction.
    parameters : Parameters
        Contains SA schedule parameters (initial temp, cooling, step count, etc.).

    internal attributes (set automatically)
    ---------------------------------------
    antipodal : bool
        True unless system_type is NON_ANTIPODAL.
    initialized : bool
        Tracks whether spin configuration has been initialized.
    neighborhood : list[NeighborhoodLike]
        Precomputed adjacency list for every lattice site.
    prob_prefactor : float
        Multiplicative factor used in Boltzmann probability calculations.
    num_spins : int
        Number of logical spins (N/2 for antipodal systems).
    system_shape : tuple or int
        Expected shape of spin arrays based on system_type.
    current_system : ndarray or None
        Current spin configuration.
    best_system : ndarray or None
        Best configuration encountered so far.
    current_energy : float
        Energy of current_system.
    best_energy : float
        Energy of best_system.
    anneal_data : AnnealData
        Temperature, energy history, and acceptance statistics for annealing.
    parameters_set : bool
        Tracks whether user provided custom Parameters.
    up_spins, down_spins : list[int] or None
        Only used for NON_ANTIPODAL systems.

    Notes
    -----
    • You must call `compute_neighborhood()` before initializing spins.
    • You must initialize spins before calling `anneal()` or `thermalize()`.
    • Saving includes spin configurations and annealing metadata, but does NOT
      include the grid itself; the same grid must exist on the object before `load()`.
    """

    grid: NDArray[np.float64]
    jump_distance: float
    system_type: SystemType | str = SystemType.INVERSE
    delta: DeltaType = DeltaType.TWO
    parameters: Parameters = field(init=True, default_factory=Parameters)

    antipodal: bool = field(init=False, default=True)
    initialized: bool = field(init=False, default=False)
    neighborhood: list[NeighborhoodLike] | None = field(init=False, default=None)
    prob_prefactor: float = field(init=False, default=0.0)
    num_spins: int = field(init=False, default=0)
    system_shape: tuple[int] | None = field(init=False, default=None)
    current_system: NDArray[np.float64] | None = field(init=False, default=None)
    current_energy: float = field(init=False, default=0.0)
    best_system: NDArray[np.float64] | None = field(init=False, default=None)
    best_energy: float = field(init=False, default=0.0)
    anneal_data: AnnealData | None = field(init=False, default=None)
    parameters_set: bool | None = field(init=False, default=False)
    # Non-Antipodal Only
    up_spins: list[int] | None = field(init=False, default=None)
    down_spins: list[int] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Normalize enums and derive antipodal flag; validate lattice size."""
        # Coerce strings to Enums (case-insensitive); fall back to defaults
        if isinstance(self.system_type, str):
            try:
                self.system_type = SystemType(self.system_type.lower())
            except ValueError:
                self.system_type = SystemType.INVERSE

        if isinstance(self.delta, str):
            try:
                self.delta = DeltaType(self.delta.lower())
            except ValueError:
                self.delta = DeltaType.TWO

        # Derive antipodal flag from system type
        self.antipodal = self.system_type != SystemType.NON_ANTIPODAL

        # Basic grid validation and site count
        if not isinstance(self.grid, np.ndarray):
            raise TypeError(f"grid must be a numpy.ndarray, got {type(self.grid)!r}")

        if self.grid.ndim != 2 or self.grid.shape[1] != 3:
            raise ValueError(
                f"grid must be a 2D array of 3D Cartesian coordinates with shape (N, 3), "
                f"but got shape {self.grid.shape}"
            )
        if not np.issubdtype(self.grid.dtype, np.number):
            raise TypeError("grid must contain numeric values")

        n_sites = int(self.grid.shape[0])

        # Enforce even number of sites only for antipodal systems
        if self.antipodal:
            if n_sites % 2 != 0:
                raise ValueError(
                    "INVERSE and INDEPENDENT systems require an even number of lattice sites. "
                    "The second half of lattice sites are assumed to be antipodes of the first half."
                )
        self.num_spins = n_sites // 2
        self.prob_prefactor = calculate_prob_prefactor(
            len(self.grid), self.jump_distance
        )
        system_shapes = {
            "inverse": self.num_spins,
            "independent": (self.num_spins, 2),
            "non-antipodal": n_sites,
        }
        self.system_shape = system_shapes.get(self.system_type.value)
        self.anneal_data = AnnealData(
            steps=int(2.5 * len(self.grid)), temperature=self.parameters.starting_temp
        )

    def compute_neighborhood(self, *, silent: bool = True) -> None:
        """
        Build and store the neighborhood; also compute probability prefactor.
        """
        self.neighborhood = build_neighborhood(
            self.grid,
            self.jump_distance,
            antipodal=self.antipodal,
            delta=self.delta.value,  # if the builder expects a string
            silent=silent,
        )

    def _require_neighborhood(self) -> None:
        if self.neighborhood is None:
            raise RuntimeError(
                "Neighborhood has not been built. Call compute_neighborhood() first."
            )

    def _require_init(self) -> None:
        if not self.initialized:
            raise RuntimeError(
                "System needs to be initialized. Call random_init(), hemisphere_init(), tri_cog_init(), cos_cog_init(), or from_array_init() first."
            )

    def set_parameters(self, parameters: Parameters) -> None:
        self.parameters = parameters
        self.parameters_set = True

    # --- initializers ---
    def random_init(self) -> None:
        self._require_neighborhood()
        init_random(self)
        self.initialized = True

    def hemisphere_init(self) -> None:
        self._require_neighborhood()
        init_random_hemisphere(self)
        self.initialized = True

    def tri_cog_init(self, num_cogs: int, cog_height: float) -> None:
        self._require_neighborhood()
        init_triangle_cogwheel(self, num_cogs, cog_height)
        self.initialized = True

    def cos_cog_init(self, num_cogs: int, cog_height: float) -> None:
        self._require_neighborhood()
        init_cosine_cogwheel(self, num_cogs, cog_height)
        self.initialized = True

    def array_init(self, spin_array: NDArray[np.bool_]) -> None:
        self._require_neighborhood()
        init_from_array(self, spin_array)
        self.initialized = True

    # --- Annealing ---
    def anneal(self, save: bool = True, verbose: bool = False) -> None:
        self._require_neighborhood()
        self._require_init()
        simulated_annealing(self, save=save, verbose=verbose)

    def thermalize(
        self,
        target_acceptance: float = 0.6,
        tolerance: float = 0.05,
        steps_per_temp: int | None = None,
        max_adjustments: int = 20,
        k_p: float = 0.5,
        verbose: bool = False,
    ) -> None:
        """Adjust the annealing temperature to reach a target Metropolis acceptance.

        Parameters
        ----------
        target_acceptance : float, optional
            Desired acceptance probability for SA start, by default 0.6.
        tolerance : float, optional
            Allowed deviation |acc - target|, by default 0.05.
        steps_per_temp : int | None, optional
            MC steps per temperature probe. If None, defaults to 10 * num_spins.
        max_adjustments : int, optional
            Maximum number of temperature adjustments, by default 20.
        k_p : float, optional
            Proportional gain in log(T) update used by `thermalize_system`,
            by default 0.5.
        verbose : bool, optional
            If True, print thermalization diagnostics, by default False.
        """
        if steps_per_temp is None:
            steps_per_temp = self.num_spins * 10

        thermalize_system(
            system=self,
            target_acceptance=target_acceptance,
            tolerance=tolerance,
            steps_per_temp=steps_per_temp,
            max_adjustments=max_adjustments,
            verbose=verbose,
            k_p=k_p,
        )

    # --- visualization ---
    def plot(
        self,
        plot_current: bool = False,
        auto_rotate: bool = False,
        plot_figure: bool = True,
    ) -> None:
        system_to_plot = self.current_system if plot_current else self.best_system
        plot_spin_system(
            system_type=self.system_type,
            grid_points=self.grid,
            spin_states=system_to_plot,
            fig_title="Current System" if plot_current else "Best System",
            rotate=auto_rotate,
            plot_figure=plot_figure,
        )

    def plot_boundary(
        self, plot_current: bool = False, plot_figure: bool = True
    ) -> None:
        system_to_plot = self.current_system if plot_current else self.best_system
        independent = self.system_type == SystemType.INDEPENDENT
        graph_boundary(
            grid=self.grid,
            spin_states=system_to_plot,
            antipodal=self.antipodal,
            independent=independent,
            plot_figure=plot_figure,
        )

    # --- Save/Load helpers  ---
    def _spin_dims_and_coords(self) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
        """Canonical dims/coords for storing spin arrays based on system_type."""
        n_sites = int(self.grid.shape[0])
        if self.system_type == SystemType.INVERSE:
            dims = ("spin",)
            coords = {"spin": np.arange(self.num_spins)}
        elif self.system_type == SystemType.INDEPENDENT:
            dims = ("spin", "comp")
            coords = {"spin": np.arange(self.num_spins), "comp": np.array([0, 1])}
        else:  # NON_ANTIPODAL
            dims = ("site",)
            coords = {"site": np.arange(n_sites)}
        return dims, coords

    @staticmethod
    def _to_jsonable(obj) -> str:
        if obj is None:
            return json.dumps(None)
        try:
            import enum

            if isinstance(obj, enum.Enum):
                return json.dumps(
                    {"__enum__": obj.__class__.__name__, "value": obj.value}
                )
        except Exception:
            pass
        if is_dataclass(obj):
            return json.dumps(
                {"__dataclass__": obj.__class__.__name__, "data": asdict(obj)}
            )
        return json.dumps(obj)

    @staticmethod
    def _from_jsonable(s: str):
        val = json.loads(s)
        if isinstance(val, dict):
            if "__enum__" in val:
                return val["value"]
            if "__dataclass__" in val:
                return val["data"]
        return val

    def save(self, save_path: str | Path) -> None:
        """
        Save only state (spins/energies/attrs) to NetCDF via xarray.
        Does NOT save the grid or neighborhood.

        If `save_path` is a directory (or has no suffix), the NetCDF file is written
        as `system_state.nc` inside that directory. If `save_path` is a file path,
        the NetCDF is written exactly there, and legacy outputs go in its parent dir.
        """
        save_path = Path(save_path)

        # Decide if save_path is a directory or a full file path
        if save_path.suffix == "" or save_path.is_dir():
            # Treat as directory
            save_dir = save_path
            save_dir.mkdir(parents=True, exist_ok=True)
            netcdf_path = save_dir / "system_state.nc"
        else:
            # Treat as file path
            save_dir = save_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            netcdf_path = save_path

        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        coords: dict[str, np.ndarray] = {}

        spin_dims, spin_coords = self._spin_dims_and_coords()
        coords.update(spin_coords)

        if self.current_system is not None:
            data_vars["current_system"] = (spin_dims, self.current_system)
        if self.best_system is not None:
            data_vars["best_system"] = (spin_dims, self.best_system)
        if self.up_spins is not None:
            data_vars["up_spins"] = (
                ("index",),
                np.asarray(self.up_spins, dtype=np.int64),
            )
        if self.down_spins is not None:
            data_vars["down_spins"] = (
                ("index",),
                np.asarray(self.down_spins, dtype=np.int64),
            )

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Minimal but sufficient attrs to validate on load
        ds.attrs.update(
            {
                "format_version": "1.0",
                "system_type": self.system_type.value,
                "delta": self.delta.value,
                "antipodal": bool(self.antipodal),
                "jump_distance": float(self.jump_distance),
                "initialized": bool(self.initialized),
                "num_spins": int(self.num_spins),
                "site_count": int(self.grid.shape[0]),
                "current_energy": float(self.current_energy),
                "best_energy": float(self.best_energy),
                "parameters_json": self._to_jsonable(self.parameters),
                "anneal_data_json": self._to_jsonable(self.anneal_data),
                "system_shape_json": self._to_jsonable(self.system_shape),
            }
        )

        # Write NetCDF
        ds.to_netcdf(netcdf_path)

        # ------------------------------------------------------------------
        # CSV Outputs
        # ------------------------------------------------------------------
        if self.parameters is None:
            raise RuntimeError("self.parameters is None.")

        if self.current_system is None or self.best_system is None:
            raise RuntimeError("current_system/best_system are missing.")
        if self.anneal_data is None:
            raise RuntimeError("self.anneal_data is None.")

        # You may need to adjust these attribute names to match your AnnealData
        anneal_data = self.anneal_data
        iteration = getattr(self.anneal_data, "iterations", 0)

        energies = np.asarray(self.anneal_data.energies, dtype=float)

        np.savetxt(
            save_dir / f"system_cycle{iteration}.gz",
            self.current_system,
            fmt="%i",
        )

        np.savetxt(
            save_dir / "best_system.gz",
            self.best_system,
            fmt="%i",
        )

        pd.DataFrame(
            anneal_data.update_stats,
            columns=["Temperature", "Acceptance Rate"],
        ).to_csv(save_dir / "anneal_stats.csv", index=False)

        # probability_prefactor -> self.prob_prefactor
        pd.DataFrame(
            np.column_stack(
                (
                    energies,
                    self.prob_prefactor * energies,
                )
            ),
            columns=["energy", "probability"],
        ).to_csv(
            save_dir / "cycle_probabilities.csv", index=False
        )  # Add best energy as first entry to CSV

    def load(self, load_path: str | Path) -> None:
        """
        Load state from NetCDF saved by `save()` without touching `self.grid`.
        Validates that the saved state matches this instance's grid shape and system_type.
        Rebuild neighborhood yourself after loading if needed.
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"No such file: {load_path}")

        ds = xr.load_dataset(load_path)

        # 1) Validate against current instance (grid must already exist)
        if (
            not isinstance(self.grid, np.ndarray)
            or self.grid.ndim != 2
            or self.grid.shape[1] != 3
        ):
            raise RuntimeError(
                "`self.grid` must be present on the instance before calling load()."
            )

        site_count_saved = int(ds.attrs.get("site_count", -1))
        if site_count_saved != self.grid.shape[0]:
            raise ValueError(
                f"Saved state expects site_count={site_count_saved}, "
                f"but current grid has {self.grid.shape[0]} sites."
            )

        # Enums and core attrs
        saved_system_type = SystemType(
            ds.attrs.get("system_type", self.system_type.value)
        )
        saved_delta = DeltaType(ds.attrs.get("delta", self.delta.value))
        if saved_system_type != self.system_type or saved_delta != self.delta:
            raise ValueError(
                "Saved state system_type/delta do not match this instance. "
                f"saved=({saved_system_type.value}, {saved_delta.value}) "
                f"current=({self.system_type.value}, {self.delta.value})"
            )

        self.antipodal = bool(
            ds.attrs.get("antipodal", self.system_type != SystemType.NON_ANTIPODAL)
        )
        self.jump_distance = float(ds.attrs.get("jump_distance", self.jump_distance))
        self.initialized = bool(ds.attrs.get("initialized", False))
        self.num_spins = int(ds.attrs.get("num_spins", self.grid.shape[0] // 2))
        self.current_energy = float(ds.attrs.get("current_energy", 0.0))
        self.best_energy = float(ds.attrs.get("best_energy", 0.0))
        self.system_shape = (
            self._from_jsonable(ds.attrs.get("system_shape_json", "null")) or None
        )

        params_dict = self._from_jsonable(ds.attrs.get("parameters_json", "null"))
        if isinstance(params_dict, dict):
            self.parameters = Parameters(**params_dict)

        anneal_dict = self._from_jsonable(ds.attrs.get("anneal_data_json", "null"))
        if isinstance(anneal_dict, dict):
            self.anneal_data = AnnealData(**anneal_dict)
        else:
            # reasonable default if missing
            self.anneal_data = AnnealData(
                steps=int(2.5 * len(self.grid)),
                temperature=self.parameters.starting_temp,
            )

        # 2) Arrays
        self.current_system = None
        if "current_system" in ds:
            self.current_system = np.asarray(ds["current_system"].values)

        self.best_system = None
        if "best_system" in ds:
            self.best_system = np.asarray(ds["best_system"].values)

        self.up_spins = None
        if "up_spins" in ds:
            self.up_spins = [int(x) for x in np.asarray(ds["up_spins"].values)]
        self.down_spins = None
        if "down_spins" in ds:
            self.down_spins = [int(x) for x in np.asarray(ds["down_spins"].values)]

        # 3) Recompute prob_prefactor from live grid + jump_distance (grid wasn't saved)
        self.prob_prefactor = calculate_prob_prefactor(
            len(self.grid), self.jump_distance
        )

        # 4) Safety: neighborhood must be rebuilt
        self.neighborhood = None

        # 5) If no spin state present, mark uninitialized
        if self.current_system is None and self.best_system is None:
            self.initialized = False
