"""
Star Cluster Analysis - Evolution Over Time
===========================================

Analyzes how RMS radius and dynamical relaxation time evolve during collapse.

Usage:
    python sc_analysis.py --data_path /path/to/sims/ --nsc 3 --seed 42
    python sc_analysis.py --data_path /path/to/sims/ --compare_all
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

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


class ClusterAnalyzer:
    """Analyze star cluster evolution over time."""
    
    def __init__(self, data_path: str, nsc: int, seed: int):
        """
        Args:
            data_path: Path to simulation directory
            nsc: Number of subclusters (1-8)
            seed: Random seed
        """
        self.data_path = Path(data_path)
        self.nsc = nsc
        self.seed = seed
        self.sim_name = f"NSC{nsc}SEED{seed}"
        self.sim_dir = self.data_path / self.sim_name
        
        if not self.sim_dir.exists():
            raise FileNotFoundError(f"Simulation not found: {self.sim_dir}")
        
        # Storage for time series
        self.times = []
        self.rms_radii = []
        self.relaxation_times = []
    
    def load_snapshot(self, snapshot: int):
        """Load a single snapshot."""
        filename = self.sim_dir / f"data.{snapshot}"
        
        if not filename.exists():
            return None
        
        try:
            # Load data: columns are mass, x, y, z, vx, vy, vz
            data = np.loadtxt(filename, skiprows=1)
            
            masses = data[:, 0]
            positions = data[:, 1:4]  # x, y, z
            velocities = data[:, 4:7]  # vx, vy, vz
            
            return masses, positions, velocities
        
        except ValueError as e:
            # Handle corrupted files (inconsistent number of columns)
            if "number of columns changed" in str(e):
                print(f"  Warning: Corrupted file {filename.name}, skipping...")
                return None
            else:
                raise  # Re-raise if it's a different ValueError
        
        except Exception as e:
            # Catch any other errors
            print(f"  Warning: Error loading {filename.name}: {e}")
            return None
    
    def compute_center_of_mass(self, masses, positions, velocities):
        """Compute COM position and velocity."""
        total_mass = masses.sum()
        com_pos = (masses[:, None] * positions).sum(axis=0) / total_mass
        com_vel = (masses[:, None] * velocities).sum(axis=0) / total_mass
        
        return com_pos, com_vel
    
    def compute_rms_radius(self, masses, positions, com_pos):
        """Compute RMS radius from center of mass."""
        r = positions - com_pos  # Relative positions
        r_mag = np.linalg.norm(r, axis=1)  # Distances from COM
        
        # Mass-weighted RMS radius
        total_mass = masses.sum()
        rms_radius = np.sqrt((masses * r_mag**2).sum() / total_mass)
        
        return rms_radius
    
    def compute_relaxation_time(self, masses, positions, velocities, 
                                com_pos, com_vel, rms_radius):
        """
        Compute dynamical relaxation time.
        
        Formula: t_relax = 2 * N * R_rms / (8 * ln(N) * <v>)
        
        where:
            N = number of stars
            R_rms = RMS radius from COM
            <v> = average velocity in COM frame
        """
        N = len(masses)
        
        # Velocities in COM frame
        v_com_frame = velocities - com_vel
        v_mag = np.linalg.norm(v_com_frame, axis=1)
        
        # Average velocity
        avg_velocity = v_mag.mean()
        
        # Relaxation time
        if avg_velocity > 0:
            t_relax = 2 * N * rms_radius / (8 * np.log(N) * avg_velocity)
        else:
            t_relax = np.inf
        
        return t_relax
    
    def analyze_evolution(self, max_snapshot: int = 20):
        """Analyze cluster evolution over time."""
        print(f"\nAnalyzing {self.sim_name}...")
        
        for snapshot in range(max_snapshot + 1):
            result = self.load_snapshot(snapshot)
            
            if result is None:
                continue
            
            masses, positions, velocities = result
            
            # Compute COM
            com_pos, com_vel = self.compute_center_of_mass(
                masses, positions, velocities
            )
            
            # Compute RMS radius
            rms_radius = self.compute_rms_radius(masses, positions, com_pos)
            
            # Compute relaxation time
            t_relax = self.compute_relaxation_time(
                masses, positions, velocities, 
                com_pos, com_vel, rms_radius
            )
            
            # Store
            self.times.append(snapshot)
            self.rms_radii.append(rms_radius)
            self.relaxation_times.append(t_relax)
        
        # Convert to arrays
        self.times = np.array(self.times)
        self.rms_radii = np.array(self.rms_radii)
        self.relaxation_times = np.array(self.relaxation_times)
        
        if len(self.times) == 0:
            print(f"  ERROR: No valid snapshots loaded!")
            return False
        
        print(f"  Loaded {len(self.times)} snapshots")
        print(f"  RMS radius: {self.rms_radii[0]:.2f} → {self.rms_radii[-1]:.2f}")
        print(f"  Relaxation time: {self.relaxation_times[0]:.2f} → {self.relaxation_times[-1]:.2f}")
        
        return True


def plot_rms_evolution(analyzers, save_path=None):
    """Plot RMS radius evolution for multiple NSC classes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [CMAP(i / 7) for i in range(8)]
    
    for analyzer in analyzers:
        nsc = analyzer.nsc
        color = colors[nsc - 1]
        
        ax.plot(analyzer.times, analyzer.rms_radii, 
               linewidth=1.0, color=color, alpha=0.3)
    
    ax.set_xlabel('t [Myr]', fontsize=14)
    ax.set_ylabel(r'$R_{\rm rms}$ [pc]', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    norm = Normalize(vmin=1, vmax=8)
    sm = ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'$N_{\rm sc}$', fontsize=13, rotation=270, labelpad=20)
    cbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved RMS evolution to {save_path}")
    
    plt.show()


def plot_relaxation_time_evolution(analyzers, save_path=None):
    """Plot relaxation time evolution for multiple NSC classes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [CMAP(i / 7) for i in range(8)]
    
    for analyzer in analyzers:
        nsc = analyzer.nsc
        color = colors[nsc - 1]
        
        # Filter out infinities
        mask = np.isfinite(analyzer.relaxation_times)
        times = analyzer.times[mask]
        t_relax = analyzer.relaxation_times[mask]
        
        ax.plot(times, t_relax, 
               linewidth=1.0, color=color, alpha=0.3)
    
    ax.set_xlabel('t [Myr]', fontsize=14)
    ax.set_ylabel(r'$t_{\rm relax}$ [Myr]', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    norm = Normalize(vmin=1, vmax=8)
    sm = ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'$N_{\rm sc}$', fontsize=13, rotation=270, labelpad=20)
    cbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])
    
    # Optional: log scale if values vary widely
    # ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved relaxation time evolution to {save_path}")
    
    plt.show()


def plot_combined(analyzers, save_path=None):
    """Plot both metrics in a 2-panel figure with colorbar."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = [CMAP(i / 7) for i in range(8)]
    
    # Panel 1: RMS radius
    for analyzer in analyzers:
        nsc = analyzer.nsc
        color = colors[nsc - 1]
        
        ax1.plot(analyzer.times, analyzer.rms_radii, 
                linewidth=1.0, color=color, alpha=0.3)
    
    ax1.set_xlabel('t [Myr]', fontsize=13)
    ax1.set_ylabel(r'$R_{\rm rms}$ [pc]', fontsize=13)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Panel 2: Relaxation time
    for analyzer in analyzers:
        nsc = analyzer.nsc
        color = colors[nsc - 1]
        
        mask = np.isfinite(analyzer.relaxation_times)
        times = analyzer.times[mask]
        t_relax = analyzer.relaxation_times[mask]
        
        ax2.plot(times, t_relax, 
                linewidth=1.0, color=color, alpha=0.3)
    
    ax2.set_xlabel('t [Myr]', fontsize=13)
    ax2.set_ylabel(r'$t_{\rm relax}$ [Myr]', fontsize=13)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar on the right side
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    norm = Normalize(vmin=1, vmax=8)
    sm = ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, ax=ax2, pad=0.02)
    cbar.set_label(r'$N_{\rm sc}$', fontsize=13, rotation=270, labelpad=20)
    cbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined plot to {save_path}")
    
    plt.show()


def main(args):
    """Main analysis script."""
    
    if args.compare_all:
        # Compare all NSC classes (all seeds)
        print("\n" + "="*60)
        print(f"Comparing all NSC classes (all {args.n_seeds} seeds)")
        print("="*60)
        
        analyzers = []
        for nsc in range(1, 9):
            for seed in range(args.n_seeds):
                try:
                    analyzer = ClusterAnalyzer(args.data_path, nsc, seed=seed)
                    success = analyzer.analyze_evolution(args.max_snapshot)
                    if success:  # Only add if we got valid data
                        analyzers.append(analyzer)
                except FileNotFoundError:
                    # Silently skip missing simulations
                    pass
        
        print(f"\nSuccessfully loaded {len(analyzers)} simulations")
        
        if len(analyzers) == 0:
            print("ERROR: No valid simulations found!")
            return
        
        # Plot combined
        plot_combined(analyzers, save_path=args.output_dir / "evolution_combined.png")
        
        # Or separate plots
        # plot_rms_evolution(analyzers, save_path=args.output_dir / "rms_evolution.png")
        # plot_relaxation_time_evolution(analyzers, save_path=args.output_dir / "relaxation_evolution.png")
        
    else:
        # Analyze single simulation
        print("\n" + "="*60)
        print(f"Analyzing NSC={args.nsc}, SEED={args.seed}")
        print("="*60)
        
        analyzer = ClusterAnalyzer(args.data_path, args.nsc, args.seed)
        success = analyzer.analyze_evolution(args.max_snapshot)
        
        if not success:
            print("ERROR: Failed to load any valid snapshots!")
            return
        
        # Plot
        plot_combined([analyzer], 
                     save_path=args.output_dir / f"evolution_NSC{args.nsc}_SEED{args.seed}.png")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze star cluster evolution"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to simulation directory"
    )
    parser.add_argument(
        "--nsc",
        type=int,
        default=3,
        help="Number of subclusters (1-8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--compare_all",
        action="store_true",
        help="Compare all NSC classes (uses all seeds)"
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=300,
        help="Number of random seeds per NSC class (for --compare_all)"
    )
    parser.add_argument(
        "--max_snapshot",
        type=int,
        default=20,
        help="Maximum snapshot to analyze"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/analysis",
        help="Directory to save plots"
    )
    
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    main(args)