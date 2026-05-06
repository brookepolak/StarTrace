#!/usr/bin/env python3
"""
StarTrace Validation Script
===========================

Command-line interface for validating trained StarTrace GNN models.

This script loads a trained model and evaluates it on the validation set,
generating publication-quality diagnostic plots including confusion matrices,
accuracy breakdowns, and uncertainty calibration.

Usage:
    # Basic validation
    python validate.py --model_path outputs/StarTrace_best_model.pt
    
    # Specify custom data path
    python validate.py --model_path outputs/StarTrace_best_model.pt --data_path /path/to/sims/
    
    # Save plots to custom directory
    python validate.py --model_path outputs/StarTrace_best_model.pt --output_dir results/validation/

For more information, see: python validate.py --help
"""

import argparse
import sys
from pathlib import Path

# Import from StarTrace library
from StarTrace import Validator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate trained StarTrace GNN model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation with default data path
  python validate.py --model_path outputs/StarTrace_best_model.pt
  
  # Specify custom data and output paths
  python validate.py --model_path outputs/StarTrace_best_model.pt \\
                     --data_path /path/to/sims/ \\
                     --output_dir validation_results/
  
  # Use different dataset size
  python validate.py --model_path outputs/StarTrace_best_model.pt --n_seeds 100

Generated Plots:
  - validation_summary.png: Mean predicted vs true NSC with 90% CI
  - confusion_matrix.png: Normalized confusion matrix heatmap
  
All plots are saved as high-resolution PNG files (300 DPI).
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)"
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument(
        "--data_path",
        type=str,
        default="/gpfs/work3/0/ulc15220/bp/StarTrace/sims/",
        help="Path to simulation data directory (default: cluster path)"
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
        help="Maximum NSC value in dataset (default: 8)"
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output Settings')
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="outputs/validation",
        help="Directory to save validation plots (default: outputs/validation/)"
    )
    
    return parser.parse_args()


def main():
    """Main validation function."""
    args = parse_args()
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"✗ Error: Model not found at {model_path}")
        print(f"  Please check the path and try again.")
        sys.exit(1)
    
    # Create validator
    validator = Validator(
        model_path=str(model_path),
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_seeds=args.n_seeds,
        n_scs=args.n_scs
    )
    
    # Run validation
    try:
        results = validator.run()
        print("\n✓ Validation completed successfully!")
        print(f"\nResults saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Validation failed with error: {e}")
        raise
    
    print("\nDone!")


if __name__ == "__main__":
    main()
