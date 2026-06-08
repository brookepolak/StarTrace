"""
Star Cluster Analysis - Evolution Over Time
===========================================

Computes the half-mass radius (HMR) and dynamical relaxation time for each
simulation and saves the time series to an HDF5 file for easy plotting later.

The module is organised around a single ``ClusterAnalysis`` class whose
methods are split into two groups:

    * Processing   - load snapshots, compute HMR / relaxation time, save to HDF5
    * Plotting     - read the time series back and make figures

A lightweight ``ClusterAnalysisReader`` is provided so the saved data can be
consumed elsewhere with a simple mapping interface::

    from sc_analysis import ClusterAnalysisReader

    reader = ClusterAnalysisReader("cluster_evolution.h5")
    t   = reader["NSC3SEED0"]["time"]
    hmr = reader["NSC3SEED0"]["hmr"]
    trx = reader["NSC3SEED0"]["t_relax"]

Usage:
    # Process all sims and write the HDF5 file (+ a combined plot)
    python sc_analysis.py --data_path /path/to/sims/ --compare_all

    # Process a single simulation
    python sc_analysis.py --data_path /path/to/sims/ --nsc 3 --seed 42

    # Re-plot from an existing HDF5 file without re-processing
    python sc_analysis.py --from_h5 cluster_evolution.h5

    # Append a new snapshot range to each sim already in the HDF5 file
    # (only the new snapshots are computed; the file is updated in place)
    python sc_analysis.py --data_path /path/to/sims/ --compare_all --update \\
        --min_snapshot 21 --max_snapshot 50
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py

# Beautiful colormap
try:
    import cmasher as cmr
    CMAP = cmr.ocean
except ImportError:
    CMAP = plt.cm.viridis

# Plotting style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
})


# ═════════════════════════════════════════════════════════════════════
# READER
# ═════════════════════════════════════════════════════════════════════

class _SimView:
    """
    Mapping view over a single simulation group in the HDF5 file.

    Indexing returns the underlying data as a NumPy array, so that
    ``reader['sim']['hmr']`` yields an array directly.
    """

    def __init__(self, group: h5py.Group):
        self._group = group

    def __getitem__(self, key: str) -> np.ndarray:
        return self._group[key][...]

    def __contains__(self, key: str) -> bool:
        return key in self._group

    def keys(self):
        return list(self._group.keys())

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    @property
    def attrs(self) -> dict:
        return dict(self._group.attrs)

    def __repr__(self):
        return f"<SimView {self._group.name} fields={self.keys()}>"


class ClusterAnalysisReader:
    """
    Read-only accessor for a ClusterAnalysis HDF5 file.

    Supports a dict-like interface keyed by simulation name::

        reader = ClusterAnalysisReader("cluster_evolution.h5")
        reader["NSC3SEED0"]["time"]      # -> np.ndarray
        reader["NSC3SEED0"]["hmr"]       # -> np.ndarray
        reader["NSC3SEED0"]["t_relax"]   # -> np.ndarray
        list(reader)                      # -> list of sim names

    Can be used as a context manager to auto-close the file.
    """

    def __init__(self, h5_path: str):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        self._file = h5py.File(self.h5_path, 'r')

    def __getitem__(self, sim: str) -> _SimView:
        return _SimView(self._file[sim])

    def __contains__(self, sim: str) -> bool:
        return sim in self._file

    def __iter__(self):
        return iter(self._file.keys())

    def __len__(self):
        return len(self._file.keys())

    def keys(self):
        return list(self._file.keys())

    def sims(self):
        """List of simulation names in the file."""
        return list(self._file.keys())

    @property
    def attrs(self) -> dict:
        """Global file-level attributes (e.g. units, settings)."""
        return dict(self._file.attrs)

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return f"<ClusterAnalysisReader {self.h5_path.name} n_sims={len(self)}>"


# ═════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═════════════════════════════════════════════════════════════════════

class ClusterAnalysis:
    """
    Analyze star cluster evolution and persist the results.

    Processing methods build a ``self.data`` dictionary keyed by simulation
    name, each entry holding the time series::

        self.data["NSC3SEED0"] = {
            "time":    np.ndarray,   # snapshot index
            "hmr":     np.ndarray,   # half-mass radius [pc]
            "t_relax": np.ndarray,   # relaxation time  [Myr]
            "nsc":     int,
            "seed":    int,
        }

    Plotting methods consume ``self.data`` (populated either by processing or
    by ``load_h5``).
    """

    # Fields that are stored as time-series datasets in the HDF5 file
    SERIES_FIELDS = ("time", "hmr", "t_relax")

    def __init__(self, data_path: str, n_seeds: int = 300,
                 n_scs: int = 8, max_snapshot: int = 20, min_snapshot: int = 0):
        """
        Args:
            data_path: Path to simulation directory (NSC*SEED* folders)
            n_seeds: Number of random seeds per NSC class
            n_scs: Maximum NSC value in dataset (1..n_scs)
            max_snapshot: Maximum snapshot index to analyze
            min_snapshot: Minimum snapshot index to analyze. Process only a
                sub-range (e.g. 21..50) to append to an existing HDF5 file
                without recomputing the snapshots already stored.
        """
        self.data_path = Path(data_path) if data_path is not None else None
        self.n_seeds = n_seeds
        self.n_scs = n_scs
        self.max_snapshot = max_snapshot
        self.min_snapshot = min_snapshot

        # sim_name -> dict of time series + metadata
        self.data = {}

    # ─────────────────────────────────────────────────────────────────
    # PROCESSING
    # ─────────────────────────────────────────────────────────────────

    def load_snapshot(self, sim_dir: Path, snapshot: int):
        """
        Load a single snapshot file.

        Returns:
            (masses, positions, velocities) or None if missing/corrupted.
        """
        filename = sim_dir / f"data.{snapshot}"

        if not filename.exists():
            return None

        try:
            # Columns: mass, x, y, z, vx, vy, vz
            data = np.loadtxt(filename, skiprows=1)

            masses = data[:, 0]
            positions = data[:, 1:4]
            velocities = data[:, 4:7]

            return masses, positions, velocities

        except ValueError as e:
            if "number of columns changed" in str(e):
                print(f"  Warning: Corrupted file {filename.name}, skipping...")
                return None
            raise

        except Exception as e:
            print(f"  Warning: Error loading {filename.name}: {e}")
            return None

    def compute_center_of_mass(self, masses, positions, velocities):
        """Compute COM position and velocity."""
        total_mass = masses.sum()
        com_pos = (masses[:, None] * positions).sum(axis=0) / total_mass
        com_vel = (masses[:, None] * velocities).sum(axis=0) / total_mass
        return com_pos, com_vel

    def compute_half_mass_radius(self, masses, positions, com_pos):
        """
        Compute the half-mass radius (HMR) from the center of mass.

        The HMR is the radius of the sphere (centered on the COM) that
        encloses half of the total cluster mass.
        """
        r = positions - com_pos
        r_mag = np.linalg.norm(r, axis=1)

        # Sort stars by distance from COM and accumulate mass
        order = np.argsort(r_mag)
        r_sorted = r_mag[order]
        m_cumulative = np.cumsum(masses[order])

        half_mass = 0.5 * m_cumulative[-1]
        idx = np.searchsorted(m_cumulative, half_mass)
        idx = min(idx, len(r_sorted) - 1)

        return r_sorted[idx]

    def compute_relaxation_time(self, masses, positions, velocities,
                                com_pos, com_vel, radius):
        """
        Compute the dynamical (half-mass) relaxation time.

        Formula: t_relax = 2 * N * R / (8 * ln(N) * <v>)

        where:
            N   = number of stars
            R   = characteristic radius (here the half-mass radius)
            <v> = mean speed in the COM frame
        """
        N = len(masses)

        v_com_frame = velocities - com_vel
        v_mag = np.linalg.norm(v_com_frame, axis=1)
        avg_velocity = v_mag.mean()

        if avg_velocity > 0 and N > 1:
            t_relax = 2 * N * radius / (8 * np.log(N) * avg_velocity)
        else:
            t_relax = np.inf

        return t_relax

    def process_sim(self, nsc: int, seed: int):
        """
        Process one simulation: compute the HMR and relaxation-time series.

        Returns:
            A data dict (see class docstring) or None if no valid snapshots.
        """
        sim_name = f"NSC{nsc}SEED{seed}"
        sim_dir = self.data_path / sim_name

        if not sim_dir.exists():
            return None

        times, hmrs, t_relaxes = [], [], []

        for snapshot in range(self.min_snapshot, self.max_snapshot + 1):
            result = self.load_snapshot(sim_dir, snapshot)
            if result is None:
                continue

            masses, positions, velocities = result

            com_pos, com_vel = self.compute_center_of_mass(
                masses, positions, velocities
            )
            hmr = self.compute_half_mass_radius(masses, positions, com_pos)
            t_relax = self.compute_relaxation_time(
                masses, positions, velocities, com_pos, com_vel, hmr
            )

            times.append(snapshot)
            hmrs.append(hmr)
            t_relaxes.append(t_relax)

        if not times:
            return None

        return {
            "time": np.array(times),
            "hmr": np.array(hmrs),
            "t_relax": np.array(t_relaxes),
            "nsc": nsc,
            "seed": seed,
        }

    def process_all(self):
        """
        Process every available simulation (all NSC classes and seeds).

        Populates ``self.data`` and returns it.
        """
        print(f"\nProcessing simulations from {self.data_path} ...")
        n_loaded = 0

        for nsc in range(1, self.n_scs + 1):
            for seed in range(self.n_seeds):
                sim = self.process_sim(nsc, seed)
                if sim is None:
                    continue
                sim_name = f"NSC{nsc}SEED{seed}"
                self.data[sim_name] = sim
                n_loaded += 1

        print(f"  Loaded {n_loaded} simulations.")
        return self.data

    def save_h5(self, h5_path: str):
        """
        Save ``self.data`` to an HDF5 file.

        Layout::

            /<sim_name>/time      (dataset)
            /<sim_name>/hmr       (dataset)
            /<sim_name>/t_relax   (dataset)
            /<sim_name>           (attrs: nsc, seed)
        """
        if not self.data:
            print("Nothing to save (no processed data).")
            return

        h5_path = Path(h5_path)
        h5_path.parent.mkdir(exist_ok=True, parents=True)

        with h5py.File(h5_path, 'w') as f:
            # File-level metadata documenting units / conventions
            f.attrs['description'] = "StarTrace cluster evolution time series"
            f.attrs['time_units'] = "snapshot index"
            f.attrs['hmr_units'] = "pc"
            f.attrs['t_relax_units'] = "Myr"
            f.attrs['max_snapshot'] = self.max_snapshot

            for sim_name, sim in self.data.items():
                grp = f.create_group(sim_name)
                for field in self.SERIES_FIELDS:
                    grp.create_dataset(field, data=sim[field])
                grp.attrs['nsc'] = sim['nsc']
                grp.attrs['seed'] = sim['seed']

        print(f"Saved {len(self.data)} simulations to {h5_path}")

    @staticmethod
    def _merge_series(old, new):
        """
        Merge two per-sim data dicts by snapshot (time).

        Snapshots present in both are taken from ``new`` (recomputed values
        win); the union is returned sorted by time. Used to splice freshly
        processed snapshots into an existing record.
        """
        merged = {}
        for t, h, r in zip(old["time"], old["hmr"], old["t_relax"]):
            merged[int(t)] = (float(h), float(r))
        for t, h, r in zip(new["time"], new["hmr"], new["t_relax"]):
            merged[int(t)] = (float(h), float(r))

        times = sorted(merged)
        return {
            "time": np.array(times),
            "hmr": np.array([merged[t][0] for t in times]),
            "t_relax": np.array([merged[t][1] for t in times]),
            "nsc": new.get("nsc", old.get("nsc")),
            "seed": new.get("seed", old.get("seed")),
        }

    def update_h5(self, h5_path: str):
        """
        Append/merge ``self.data`` into an existing HDF5 file in place.

        Only the simulations present in ``self.data`` are touched: for each,
        the newly processed snapshots are merged with whatever is already
        stored (see ``_merge_series``) and that sim's datasets are rewritten.
        Simulations already in the file but not re-processed are left
        untouched -- the whole file is never rewritten.

        If the file does not exist yet, it is created (equivalent to
        ``save_h5`` for the processed sims).

        Note: HDF5 does not reclaim space from deleted datasets, so a file
        updated many times may grow on disk; ``h5repack`` can compact it.
        """
        if not self.data:
            print("Nothing to update (no processed data).")
            return

        h5_path = Path(h5_path)
        h5_path.parent.mkdir(exist_ok=True, parents=True)

        n_new, n_extended = 0, 0
        with h5py.File(h5_path, 'a') as f:
            # Ensure file-level metadata exists (new file or pre-existing)
            f.attrs.setdefault('description',
                               "StarTrace cluster evolution time series")
            f.attrs.setdefault('time_units', "snapshot index")
            f.attrs.setdefault('hmr_units', "pc")
            f.attrs.setdefault('t_relax_units', "Myr")

            for sim_name, sim in self.data.items():
                if sim_name in f:
                    grp = f[sim_name]
                    existing = {
                        "time": grp["time"][...],
                        "hmr": grp["hmr"][...],
                        "t_relax": grp["t_relax"][...],
                        "nsc": int(grp.attrs.get("nsc", sim["nsc"])),
                        "seed": int(grp.attrs.get("seed", sim["seed"])),
                    }
                    record = self._merge_series(existing, sim)
                    # Rewrite just this sim's datasets (delete + recreate)
                    for field in self.SERIES_FIELDS:
                        if field in grp:
                            del grp[field]
                    n_extended += 1
                else:
                    grp = f.create_group(sim_name)
                    record = sim
                    n_new += 1

                for field in self.SERIES_FIELDS:
                    grp.create_dataset(field, data=record[field])
                grp.attrs['nsc'] = record['nsc']
                grp.attrs['seed'] = record['seed']

            # Track the overall snapshot extent seen across updates
            prev_max = int(f.attrs.get('max_snapshot', self.max_snapshot))
            f.attrs['max_snapshot'] = max(prev_max, self.max_snapshot)

        print(f"Updated {h5_path}: {n_new} new sim(s), "
              f"{n_extended} extended.")

    def load_h5(self, h5_path: str):
        """
        Load time series from an HDF5 file back into ``self.data``.

        Useful for re-plotting without re-processing the raw snapshots.
        """
        with ClusterAnalysisReader(h5_path) as reader:
            self.data = {}
            for sim_name in reader.sims():
                view = reader[sim_name]
                self.data[sim_name] = {
                    "time": view["time"],
                    "hmr": view["hmr"],
                    "t_relax": view["t_relax"],
                    "nsc": int(view.attrs.get("nsc", 0)),
                    "seed": int(view.attrs.get("seed", 0)),
                }
        print(f"Loaded {len(self.data)} simulations from {h5_path}")
        return self.data

    # ─────────────────────────────────────────────────────────────────
    # PLOTTING
    # ─────────────────────────────────────────────────────────────────

    def _color_for_nsc(self, nsc: int):
        """Map an NSC value (1..n_scs) to a colormap color."""
        return CMAP((nsc - 1) / max(self.n_scs - 1, 1))

    def _add_nsc_colorbar(self, ax):
        """Attach an N_sc colorbar to the given axis."""
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=1, vmax=self.n_scs)
        sm = ScalarMappable(cmap=CMAP, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r'$N_{\rm sc}$', fontsize=13, rotation=270, labelpad=20)
        cbar.set_ticks(list(range(1, self.n_scs + 1)))
        return cbar

    def plot_hmr_evolution(self, save_path=None):
        """Plot half-mass radius evolution for all simulations."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for sim in self.data.values():
            ax.plot(sim["time"], sim["hmr"],
                    linewidth=1.0, color=self._color_for_nsc(sim["nsc"]),
                    alpha=0.3)

        ax.set_xlabel('t [Myr]', fontsize=14)
        ax.set_ylabel(r'$R_{\rm hm}$ [pc]', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        self._add_nsc_colorbar(ax)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved HMR evolution to {save_path}")
        plt.show()

    def plot_relaxation_time_evolution(self, save_path=None):
        """Plot relaxation time evolution for all simulations."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for sim in self.data.values():
            mask = np.isfinite(sim["t_relax"])
            ax.plot(sim["time"][mask], sim["t_relax"][mask],
                    linewidth=1.0, color=self._color_for_nsc(sim["nsc"]),
                    alpha=0.3)

        ax.set_xlabel('t [Myr]', fontsize=14)
        ax.set_ylabel(r'$t_{\rm relax}$ [Myr]', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        self._add_nsc_colorbar(ax)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved relaxation time evolution to {save_path}")
        plt.show()

    def plot_combined(self, save_path=None):
        """Plot HMR and relaxation time in a 2-panel figure with colorbar."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for sim in self.data.values():
            color = self._color_for_nsc(sim["nsc"])
            ax1.plot(sim["time"], sim["hmr"],
                     linewidth=1.0, color=color, alpha=0.3)

            mask = np.isfinite(sim["t_relax"])
            ax2.plot(sim["time"][mask], sim["t_relax"][mask],
                     linewidth=1.0, color=color, alpha=0.3)

        ax1.set_xlabel('t [Myr]', fontsize=13)
        ax1.set_ylabel(r'$R_{\rm hm}$ [pc]', fontsize=13)
        ax1.grid(True, alpha=0.3, linestyle='--')

        ax2.set_xlabel('t [Myr]', fontsize=13)
        ax2.set_ylabel(r'$t_{\rm relax}$ [Myr]', fontsize=13)
        ax2.grid(True, alpha=0.3, linestyle='--')

        self._add_nsc_colorbar(ax2)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved combined plot to {save_path}")
        plt.show()


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def main(args):
    """Main analysis script."""
    analysis = ClusterAnalysis(
        data_path=args.data_path,
        n_seeds=args.n_seeds,
        n_scs=args.n_scs,
        max_snapshot=args.max_snapshot,
        min_snapshot=args.min_snapshot,
    )

    # ── Re-plot only, from an existing HDF5 file ──────────────────────
    if args.from_h5:
        print("\n" + "=" * 60)
        print(f"Re-plotting from {args.from_h5}")
        print("=" * 60)
        analysis.load_h5(args.from_h5)
        analysis.plot_combined(save_path=args.output_dir / "evolution_combined.png")
        print("\nDone!\n")
        return

    # ── Process simulations ───────────────────────────────────────────
    if args.compare_all:
        print("\n" + "=" * 60)
        print(f"Processing all NSC classes (up to {args.n_seeds} seeds each)")
        print("=" * 60)
        analysis.process_all()
    else:
        print("\n" + "=" * 60)
        print(f"Processing NSC={args.nsc}, SEED={args.seed}")
        print("=" * 60)
        sim = analysis.process_sim(args.nsc, args.seed)
        if sim is None:
            print("ERROR: No valid snapshots found for this simulation!")
            return
        analysis.data[f"NSC{args.nsc}SEED{args.seed}"] = sim

    if not analysis.data:
        print("ERROR: No valid simulations found!")
        return

    # ── Write to HDF5 ─────────────────────────────────────────────────
    # --update merges into an existing file (appending new snapshots to each
    # sim); otherwise the file is written fresh. Auto-switch to update mode
    # if the target file already exists, to avoid clobbering it.
    if args.update or Path(args.output_h5).exists():
        analysis.update_h5(args.output_h5)
    else:
        analysis.save_h5(args.output_h5)

    # ── Plot (from the full on-disk record, including any prior snapshots) ─
    analysis.load_h5(args.output_h5)
    analysis.plot_combined(save_path=args.output_dir / "evolution_combined.png")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze star cluster evolution (HMR + relaxation time)"
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to simulation directory (required unless --from_h5)"
    )
    parser.add_argument(
        "--nsc", type=int, default=3,
        help="Number of subclusters (1-8) for single-sim mode"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for single-sim mode"
    )
    parser.add_argument(
        "--compare_all", action="store_true",
        help="Process all NSC classes (all seeds)"
    )
    parser.add_argument(
        "--n_seeds", type=int, default=300,
        help="Number of random seeds per NSC class (for --compare_all)"
    )
    parser.add_argument(
        "--n_scs", type=int, default=8,
        help="Maximum NSC value in dataset"
    )
    parser.add_argument(
        "--max_snapshot", type=int, default=20,
        help="Maximum snapshot to analyze"
    )
    parser.add_argument(
        "--min_snapshot", type=int, default=0,
        help="Minimum snapshot to analyze. Use with --update to append a new "
             "snapshot sub-range (e.g. --min_snapshot 21 --max_snapshot 50) "
             "to each sim already in the HDF5 file (default: 0)"
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Merge the processed snapshots into an existing HDF5 file in "
             "place instead of rewriting it (auto-enabled if the file exists)"
    )
    parser.add_argument(
        "--output_h5", type=str, default="outputs/analysis/cluster_evolution.h5",
        help="Path to output HDF5 file"
    )
    parser.add_argument(
        "--from_h5", type=str, default=None,
        help="Skip processing and re-plot from an existing HDF5 file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/analysis",
        help="Directory to save plots"
    )

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    args.output_h5 = Path(args.output_h5)

    if args.data_path is None and args.from_h5 is None:
        parser.error("--data_path is required unless --from_h5 is given")

    main(args)
