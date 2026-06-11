"""
StarTrace Accuracy Over Time
============================

Reads the per-class accuracy text files written by the validation step
(`accuracy.txt` inside each outputs-snapX folder) and plots the per-class
validation accuracy as a function of snapshot (time).

Each `accuracy.txt` is produced by Validator.save_accuracy() and looks like:

    # StarTrace per-class validation accuracy
    # snapshot: 30
    # n_classes: 3
    # overall_accuracy: 0.861200
    # class_labels: 1 subcluster, 2 subclusters, 3+ subclusters
    # snapshot class_index accuracy n_samples
    30 0 0.853400 120
    30 1 0.791200 118
    30 2 0.900100 340

Usage:
    python accuracy_over_time.py --models_dir /path/to/models/
    python accuracy_over_time.py --models_dir . --snapshots 10-50
"""

import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Beautiful colormap
try:
    import cmasher as cmr
    CMAP = cmr.sapphire
except ImportError:
    CMAP = plt.cm.viridis

# Plotting style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
})


def parse_accuracy_file(path: Path) -> dict:
    """
    Parse an accuracy.txt file written by Validator.save_accuracy().

    The current format is a tidy confusion matrix with one row per cell:

        # snapshot true_class pred_class count
        30 0 0 102
        30 0 1 12
        ...

    Per-class accuracy and sample counts are derived from the raw counts:
        accuracy[i]  = cm[i, i] / sum_j cm[i, j]
        n_samples[i] = sum_j cm[i, j]

    (The older 4-column "snapshot class_index accuracy n_samples" layout is
    still recognised for backward compatibility.)

    Returns a dict with keys:
        snapshot         (int)
        n_classes        (int)
        overall_accuracy (float)
        class_labels     (list[str])
        class_index      (np.ndarray)
        accuracy         (np.ndarray)   per-class accuracy
        n_samples        (np.ndarray)   per-class sample counts
        confusion        (np.ndarray)   full count matrix, or None
    """
    meta = {
        'snapshot': None,
        'n_classes': None,
        'overall_accuracy': np.nan,
        'class_labels': [],
    }
    is_tidy = None          # True once we detect the confusion-matrix header
    rows = []               # raw data rows (list of token lists)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                body = line.lstrip('#').strip()
                if body.startswith('snapshot:'):
                    meta['snapshot'] = int(body.split(':', 1)[1])
                elif body.startswith('n_classes:'):
                    meta['n_classes'] = int(body.split(':', 1)[1])
                elif body.startswith('overall_accuracy:'):
                    meta['overall_accuracy'] = float(body.split(':', 1)[1])
                elif body.startswith('class_labels:'):
                    labels = body.split(':', 1)[1]
                    meta['class_labels'] = [s.strip() for s in labels.split(',')]
                elif 'pred_class' in body:
                    is_tidy = True
                elif 'class_index' in body:
                    is_tidy = False
                continue

            parts = line.split()
            if len(parts) < 4:
                continue
            if meta['snapshot'] is None:
                meta['snapshot'] = int(parts[0])
            rows.append(parts)

    # Fall back to detecting the layout from the data if no column header
    # comment was present: an old-format accuracy column contains a '.'.
    if is_tidy is None and rows:
        is_tidy = '.' not in rows[0][2]

    if is_tidy:
        n = meta['n_classes']
        if n is None:
            n = int(round(len(rows) ** 0.5))
            meta['n_classes'] = n
        cm = np.zeros((n, n), dtype=float)
        for parts in rows:
            i, j, count = int(parts[1]), int(parts[2]), float(parts[3])
            cm[i, j] = count
        row_sums = cm.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            accuracy = np.where(row_sums > 0, np.diag(cm) / row_sums, np.nan)
        meta['confusion'] = cm
        meta['class_index'] = np.arange(n)
        meta['accuracy'] = accuracy
        meta['n_samples'] = row_sums.astype(int)
    else:
        # Legacy per-class layout: snapshot class_index accuracy n_samples
        class_index = [int(p[1]) for p in rows]
        meta['confusion'] = None
        meta['class_index'] = np.array(class_index)
        meta['accuracy'] = np.array([float(p[2]) for p in rows])
        meta['n_samples'] = np.array([int(p[3]) for p in rows])
        if meta['n_classes'] is None:
            meta['n_classes'] = len(class_index)

    return meta


class AccuracyOverTime:
    """Collect per-class accuracy from snapshot output folders over time."""

    def __init__(self, models_dir: str, accuracy_filename: str = "accuracy.txt"):
        """
        Args:
            models_dir: Directory containing the outputs-snapX folders.
            accuracy_filename: Name of the per-folder accuracy file.
        """
        self.models_dir = Path(models_dir)
        self.accuracy_filename = accuracy_filename

        # Time series storage
        self.snapshots = []          # snapshot per folder
        self.per_class_acc = []      # list of per-class accuracy arrays
        self.overall_acc = []        # overall accuracy per folder
        self.n_classes = None
        self.class_labels = None

    def find_accuracy_files(self, snapshots=None):
        """
        Locate outputs-snapX/accuracy.txt files.

        Args:
            snapshots: Optional iterable of snapshot numbers to restrict to.

        Returns:
            Sorted list of (snapshot_number, accuracy_file_path) tuples. The
            snapshot number is taken from the folder name; the authoritative
            value is read from the file itself during parsing.

        The accuracy file is read from the ``plots/`` subdirectory of each
        snapshot folder, where the validation step launched by train.py
        writes it.
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

            acc_file = folder / "plots" / self.accuracy_filename
            if not acc_file.exists():
                print(f"  Warning: no plots/{self.accuracy_filename} in "
                      f"{folder.name}, skipping...")
                continue

            found.append((snap, acc_file))

        found.sort(key=lambda x: x[0])
        return found

    def run(self, snapshots=None):
        """Read all accuracy files and assemble the time series."""
        acc_files = self.find_accuracy_files(snapshots)

        if not acc_files:
            print("ERROR: No accuracy.txt files found in snapshot folders!")
            return False

        print(f"\nFound {len(acc_files)} accuracy file(s).")

        for snap, acc_file in acc_files:
            try:
                info = parse_accuracy_file(acc_file)
            except Exception as e:
                print(f"  Warning: failed to parse {acc_file}: {e}")
                continue

            n_classes = info['n_classes']

            if self.n_classes is None:
                self.n_classes = n_classes
                self.class_labels = info['class_labels']
            elif n_classes != self.n_classes:
                print(f"  Warning: {acc_file.parent.name} has {n_classes} "
                      f"classes (expected {self.n_classes}), skipping...")
                continue

            # Order per-class accuracy by class index
            class_acc = np.full(self.n_classes, np.nan)
            for idx, acc in zip(info['class_index'], info['accuracy']):
                if 0 <= idx < self.n_classes:
                    class_acc[idx] = acc

            self.snapshots.append(info['snapshot'])
            self.per_class_acc.append(class_acc)
            self.overall_acc.append(info['overall_accuracy'])

            acc_str = ", ".join(
                f"cls{c}: {class_acc[c]*100:.1f}%" for c in range(self.n_classes)
            )
            print(f"  snapshot {info['snapshot']:>3d}: "
                  f"overall {info['overall_accuracy']*100:.1f}%  |  {acc_str}")

        if not self.snapshots:
            print("\nERROR: No accuracy files parsed successfully!")
            return False

        # Sort by snapshot and convert to arrays
        order = np.argsort(self.snapshots)
        self.snapshots = np.array(self.snapshots)[order]
        self.per_class_acc = np.array(self.per_class_acc)[order]  # (n_snap, n_cls)
        self.overall_acc = np.array(self.overall_acc)[order]

        if self.class_labels is None or len(self.class_labels) != self.n_classes:
            self.class_labels = [f"class {c}" for c in range(self.n_classes)]

        print(f"\nCollected {len(self.snapshots)} snapshots.")
        return True


def plot_accuracy_over_time(analyzer, save_path=None):
    """Plot per-class accuracy as a function of snapshot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_classes = analyzer.n_classes
    labels = analyzer.class_labels
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
    overall = analyzer.overall_acc * 100.0
    mask = np.isfinite(overall)
    ax.plot(snaps[mask], overall[mask],
            linestyle='--', linewidth=1.5, color='black',
            alpha=0.7, label='Overall')

    ax.set_xlabel('t [Myr]', fontsize=14)
    ax.set_ylabel('accuracy [%]', fontsize=14)
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
        models_dir=args.models_dir,
        accuracy_filename=args.accuracy_filename,
    )

    success = analyzer.run(snapshots)
    if not success:
        return

    plot_accuracy_over_time(
        analyzer,
        save_path=args.output_dir / "accuracy_over_time.png",
    )

    # Also save the assembled time series for later use
    np.savez(
        args.output_dir / "accuracy_over_time.npz",
        snapshots=analyzer.snapshots,
        per_class_accuracy=analyzer.per_class_acc,
        overall_accuracy=analyzer.overall_acc,
        class_labels=np.array(analyzer.class_labels),
    )
    print(f"Saved combined results to {args.output_dir / 'accuracy_over_time.npz'}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot per-class accuracy from snapshot accuracy.txt files"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=".",
        help="Directory containing the outputs-snapX folders (default: .)"
    )
    parser.add_argument(
        "--snapshots",
        type=str,
        default=None,
        help="Restrict to snapshots, e.g. '10-50' or '10,20,30' "
             "(default: all found)"
    )
    parser.add_argument(
        "--accuracy_filename",
        type=str,
        default="accuracy.txt",
        help="Name of the per-folder accuracy file (default: accuracy.txt)"
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
