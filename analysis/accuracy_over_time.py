"""
StarTrace Accuracy Over Time
============================

Evaluates the trained model in each snapshot output folder (outputs-snapX)
and plots the per-class validation accuracy as a function of snapshot (time).

Each folder is expected to contain a `StarTrace_best_model.pt` checkpoint
produced by train.py (e.g. via submit_snapshots.sh). The snapshot timestep
and class configuration are read from each checkpoint, so the models are
evaluated on the same snapshot they were trained on.

Usage:
    python accuracy_over_time.py --data_path /path/to/sims/ --models_dir /path/to/models/
    python accuracy_over_time.py --data_path /path/to/sims/ --snapshots 10-50
"""

import sys
import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

# Make the StarTrace library (in ../model) importable
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_DIR))

from StarTrace import Config, Validator, get_class_labels  # noqa: E402

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


class AccuracyOverTime:
    """Evaluate per-class accuracy of snapshot models over time."""

    def __init__(self, data_path: str, models_dir: str,
                 n_seeds: int = 300, n_scs: int = 8):
        """
        Args:
            data_path: Path to simulation directory (NSC*SEED* folders)
            models_dir: Directory containing outputs-snapX model folders
            n_seeds: Number of random seeds per NSC class
            n_scs: Maximum NSC value in dataset
        """
        self.data_path = data_path
        self.models_dir = Path(models_dir)
        self.n_seeds = n_seeds
        self.n_scs = n_scs

        # Time series storage
        self.snapshots = []          # snapshot timestep per model
        self.per_class_acc = []      # list of arrays, one per snapshot
        self.overall_acc = []        # overall accuracy per snapshot
        self.n_classes = None

    def find_model_dirs(self, snapshots=None):
        """
        Locate outputs-snapX folders containing a trained model.

        Args:
            snapshots: Optional iterable of snapshot numbers to restrict to.

        Returns:
            Sorted list of (snapshot_number, model_path) tuples.
        """
        found = []
        pattern = re.compile(r"outputs[-_]snap(\d+)$")

        for folder in self.models_dir.glob("outputs*snap*"):
            if not folder.is_dir():
                continue
            match = pattern.search(folder.name)
            if not match:
                continue

            snap = int(match.group(1))
            if snapshots is not None and snap not in snapshots:
                continue

            model_path = folder / "StarTrace_best_model.pt"
            if not model_path.exists():
                print(f"  Warning: no model in {folder.name}, skipping...")
                continue

            found.append((snap, model_path))

        found.sort(key=lambda x: x[0])
        return found

    def evaluate_model(self, model_path: Path):
        """
        Run validation for a single model checkpoint.

        Returns:
            (snapshot, per_class_accuracy, overall_accuracy) or None on failure.
        """
        # Read checkpoint config so we evaluate on the correct snapshot/classes
        checkpoint = torch.load(model_path, map_location="cpu")
        cfg = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}

        Config.update(
            N_CLASSES=cfg.get('N_CLASSES', Config.N_CLASSES),
            SNAPSHOT=cfg.get('SNAPSHOT', Config.SNAPSHOT),
            K_NEIGHBORS=cfg.get('K_NEIGHBORS', Config.K_NEIGHBORS),
            HIDDEN_DIM=cfg.get('HIDDEN_DIM', Config.HIDDEN_DIM),
            USE_GLOBAL_FEATURES=cfg.get('USE_GLOBAL_FEATURES',
                                        Config.USE_GLOBAL_FEATURES),
        )
        snapshot = Config.SNAPSHOT

        # Use the Validator to load data + run inference (no plots generated)
        validator = Validator(
            model_path=str(model_path),
            data_path=self.data_path,
            output_dir=str(model_path.parent / "_acc_tmp"),
            n_seeds=self.n_seeds,
            n_scs=self.n_scs,
        )
        validator.load_model_and_data()
        results = validator.collect_predictions()

        predictions = results['predictions']
        true_labels = results['true_labels']
        n_classes = validator.n_classes

        # Per-class accuracy (NaN where a class is absent in the val set)
        class_acc = np.full(n_classes, np.nan)
        for cls in range(n_classes):
            mask = true_labels == cls
            if mask.sum() > 0:
                class_acc[cls] = (predictions[mask] == cls).mean()

        overall = (predictions == true_labels).mean()
        return snapshot, class_acc, overall, n_classes

    def run(self, snapshots=None):
        """Evaluate all snapshot models and store the time series."""
        model_dirs = self.find_model_dirs(snapshots)

        if not model_dirs:
            print("ERROR: No snapshot model folders found!")
            return False

        print(f"\nFound {len(model_dirs)} snapshot model(s) to evaluate.")

        for snap, model_path in model_dirs:
            print(f"\n{'─'*60}")
            print(f"Evaluating snapshot {snap}: {model_path.parent.name}")
            print(f"{'─'*60}")

            try:
                snapshot, class_acc, overall, n_classes = \
                    self.evaluate_model(model_path)
            except Exception as e:
                print(f"  Warning: failed to evaluate {model_path.parent.name}: {e}")
                continue

            if self.n_classes is None:
                self.n_classes = n_classes
            elif n_classes != self.n_classes:
                print(f"  Warning: {model_path.parent.name} has {n_classes} "
                      f"classes (expected {self.n_classes}), skipping...")
                continue

            self.snapshots.append(snapshot)
            self.per_class_acc.append(class_acc)
            self.overall_acc.append(overall)

            labels = get_class_labels(n_classes)
            acc_str = ", ".join(
                f"{labels[c]}: {class_acc[c]*100:.1f}%"
                for c in range(n_classes)
            )
            print(f"  Overall: {overall*100:.1f}%  |  {acc_str}")

        if not self.snapshots:
            print("\nERROR: No models evaluated successfully!")
            return False

        # Sort everything by snapshot and convert to arrays
        order = np.argsort(self.snapshots)
        self.snapshots = np.array(self.snapshots)[order]
        self.per_class_acc = np.array(self.per_class_acc)[order]  # (n_snap, n_cls)
        self.overall_acc = np.array(self.overall_acc)[order]

        print(f"\nEvaluated {len(self.snapshots)} snapshots successfully.")
        return True


def plot_accuracy_over_time(analyzer, save_path=None):
    """Plot per-class accuracy as a function of snapshot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_classes = analyzer.n_classes
    labels = get_class_labels(n_classes)
    colors = [CMAP(i / max(n_classes - 1, 1)) for i in range(n_classes)]

    snaps = analyzer.snapshots

    # Per-class curves
    for cls in range(n_classes):
        acc = analyzer.per_class_acc[:, cls] * 100.0
        mask = np.isfinite(acc)
        ax.plot(snaps[mask], acc[mask],
                marker='o', markersize=4, linewidth=1.5,
                color=colors[cls], label=labels[cls])

    # Overall accuracy as a reference (dashed black)
    ax.plot(snaps, analyzer.overall_acc * 100.0,
            linestyle='--', linewidth=1.5, color='black',
            alpha=0.7, label='Overall')

    ax.set_xlabel('Snapshot', fontsize=14)
    ax.set_ylabel('Accuracy [%]', fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(title=r'$N_{\rm sc}$ class', fontsize=10,
              title_fontsize=11, loc='best')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved accuracy-over-time plot to {save_path}")

    plt.show()


def parse_snapshots(spec):
    """Parse a snapshot spec like '10-50' or '10,20,30' into a set of ints."""
    if spec is None:
        return None
    snaps = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            snaps.update(range(int(lo), int(hi) + 1))
        elif part:
            snaps.add(int(part))
    return snaps


def main(args):
    """Main analysis script."""
    print("\n" + "=" * 60)
    print("StarTrace: Per-Class Accuracy Over Time")
    print("=" * 60)

    snapshots = parse_snapshots(args.snapshots)

    analyzer = AccuracyOverTime(
        data_path=args.data_path,
        models_dir=args.models_dir,
        n_seeds=args.n_seeds,
        n_scs=args.n_scs,
    )

    success = analyzer.run(snapshots)
    if not success:
        return

    plot_accuracy_over_time(
        analyzer,
        save_path=args.output_dir / "accuracy_over_time.png",
    )

    # Also save the raw numbers for later use
    np.savez(
        args.output_dir / "accuracy_over_time.npz",
        snapshots=analyzer.snapshots,
        per_class_accuracy=analyzer.per_class_acc,
        overall_accuracy=analyzer.overall_acc,
        class_labels=np.array(get_class_labels(analyzer.n_classes)),
    )
    print(f"Saved raw results to {args.output_dir / 'accuracy_over_time.npz'}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot per-class accuracy of snapshot models over time"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to simulation directory containing NSC*SEED* folders"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=".",
        help="Directory containing the outputs-snapX model folders (default: .)"
    )
    parser.add_argument(
        "--snapshots",
        type=str,
        default=None,
        help="Restrict to snapshots, e.g. '10-50' or '10,20,30' "
             "(default: all found)"
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=300,
        help="Number of random seeds per NSC class (default: 300)"
    )
    parser.add_argument(
        "--n_scs",
        type=int,
        default=8,
        help="Maximum NSC value in dataset (default: 8)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/analysis",
        help="Directory to save plots (default: outputs/analysis)"
    )

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True, parents=True)

    main(args)
