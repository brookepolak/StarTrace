#!/usr/bin/env python3
"""
StarTrace Training Script
=========================

Command-line interface for training StarTrace GNN models.

This script provides a user-friendly interface to the Trainer class
from the StarTrace library. It handles argument parsing, configuration,
and optionally runs validation after training completes.

Usage:
    # Basic training (3-class system)
    python train.py --data_path /path/to/sims/
    
    # Train with automatic validation
    python train.py --data_path /path/to/sims/ --validate
    
    # Customize hyperparameters
    python train.py --data_path /path/to/sims/ --n_classes 4 --hidden_dim 256
    
    # Use different snapshot and k-NN settings
    python train.py --data_path /path/to/sims/ --snapshot 10 --k_neighbors 64

For more information, see: python train.py --help
"""

import argparse
import sys
from pathlib import Path

# Import from StarTrace library
from StarTrace import Trainer, Validator, Config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train StarTrace GNN for star cluster subcluster classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train.py --data_path /path/to/sims/
  
  # Train 4-class model with validation
  python train.py --data_path /path/to/sims/ --n_classes 4 --validate
  
  # Customize training parameters
  python train.py --data_path /path/to/sims/ --epochs 150 --lr 0.0001 --batch_size 128
  
  # Use specific snapshot and k-NN settings
  python train.py --data_path /path/to/sims/ --snapshot 10 --k_neighbors 64

Class Systems:
  3-class: [1 subcluster, 2 subclusters, 3+ subclusters]
  4-class: [1 subcluster, 2 subclusters, 3 subclusters, 4+ subclusters]

The model automatically handles imbalanced datasets using class weighting.
        """
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to simulation data directory containing NSC*SEED* folders"
    )
    data_group.add_argument(
        "--n_seeds",
        type=int,
        default=300,
        help="Number of random seeds per NSC class (default: 300)"
    )
    data_group.add_argument(
        "--n_scs",
        type=int,
        default=8,
        help="Maximum NSC value in dataset, e.g., 8 for NSC1-NSC8 (default: 8)"
    )
    data_group.add_argument(
        "--snapshot",
        type=int,
        default=15,
        help="Simulation snapshot timestep to load (default: 15)"
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument(
        "--n_classes",
        type=int,
        default=3,
        choices=[3, 4, 5, 6],
        help="Number of classification classes (default: 3 for [1, 2, 3+])"
    )
    model_group.add_argument(
        "--k_neighbors",
        type=int,
        default=32,
        help="Number of nearest neighbors for k-NN graph construction (default: 32)"
    )
    model_group.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden layer dimensionality in GNN (default: 128)"
    )
    model_group.add_argument(
        "--use_global_features",
        action="store_true",
        help="Include global cluster-level features (may not improve accuracy)"
    )
    
    # Training arguments
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs (default: 100)"
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size (default: 64)"
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer (default: 1e-4)"
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="L2 regularization strength (default: 1e-4)"
    )
    train_group.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability in classifier (default: 0.5)"
    )
    train_group.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs (default: 15)"
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output Settings')
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for saving model and plots (default: outputs/)"
    )
    output_group.add_argument(
        "--validate",
        action="store_true",
        help="Run validation automatically after training completes"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Update configuration with command-line arguments
    Config.update(
        N_CLASSES=args.n_classes,
        SNAPSHOT=args.snapshot,
        K_NEIGHBORS=args.k_neighbors,
        HIDDEN_DIM=args.hidden_dim,
        USE_GLOBAL_FEATURES=args.use_global_features,
        BATCH_SIZE=args.batch_size,
        EPOCHS=args.epochs,
        LR=args.lr,
        WEIGHT_DECAY=args.weight_decay,
        DROPOUT=args.dropout,
        PATIENCE=args.patience,
        OUTPUT_DIR=Path(args.output_dir),
        MODEL_PATH=Path(args.output_dir) / "StarTrace_best_model.pt",
        PLOT_DIR=Path(args.output_dir) / "plots"
    )
    
    # Create trainer
    trainer = Trainer(
        data_path=args.data_path,
        n_seeds=args.n_seeds,
        n_scs=args.n_scs,
        config=Config
    )
    
    # Train model
    try:
        model = trainer.train()
        print("\n✓ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        raise
    
    # Optional validation
    if args.validate:
        print("\n" + "="*60)
        print("Running Validation")
        print("="*60)
        
        try:
            validator = Validator(
                model_path=str(Config.MODEL_PATH),
                data_path=args.data_path,
                output_dir=str(Config.PLOT_DIR),
                n_seeds=args.n_seeds,
                n_scs=args.n_scs
            )
            validator.run()
            print("\n✓ Validation completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Validation failed with error: {e}")
            raise
    
    print(f"\nModel saved to: {Config.MODEL_PATH}")
    print(f"Plots saved to: {Config.PLOT_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
