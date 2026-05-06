#!/usr/bin/env python3
"""
StarTrace Prediction Script
===========================

Command-line interface for making predictions on new star cluster data.

This script loads a trained StarTrace model and predicts the number of
subclusters in new star clusters from phase-space coordinate files.
Predictions include uncertainty quantification via Monte Carlo dropout.

Usage:
    # Basic prediction
    python predict.py cluster_data.txt
    
    # Specify custom model
    python predict.py cluster_data.txt --model_path model/outputs/StarTrace_best_model.pt
    
    # Save results to JSON
    python predict.py cluster_data.txt --output_json results.json

Input File Format:
    Text file with 6 or 7 space/tab-separated columns:
        [mass] x y z vx vy vz
    
    - Mass column is optional and will be ignored
    - One star per row
    - Any number of stars (GNN handles variable sizes naturally)
    - Comment lines starting with # are ignored

For more information, see: python predict.py --help
"""

import argparse
import sys
import json
from pathlib import Path

# Import from StarTrace library
from StarTrace import Predictor


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict star cluster subclusters with uncertainty quantification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction
  python predict.py cluster_coords.txt
  
  # With custom model and JSON output
  python predict.py cluster_coords.txt \\
                   --model_path best_model.pt \\
                   --output_json results.json
  
  # Show full probability distribution
  python predict.py cluster_coords.txt --full_distribution

Input File Format:
  Text file with 6 or 7 columns (space/tab separated):
      [mass] x  y  z  vx  vy  vz
  
  - Mass column is optional and will be ignored
  - One star per row, any number of stars
  - Comment lines starting with # are skipped
  
  Example:
      # Star cluster phase-space data
      1.0  0.23  -0.15   0.08   0.012  -0.003   0.001
      1.0 -0.11   0.42  -0.19  -0.008   0.015  -0.002
      ...

Output:
  Prediction summary printed to console with:
    - Predicted number of subclusters
    - Confidence percentage
    - Uncertainty (entropy)
    - Top 3 predictions with probabilities
  
  Optional JSON output contains full probability distributions.
        """
    )
    
    # Required arguments
    parser.add_argument(
        "data_file",
        type=str,
        help="Path to cluster coordinate file (6 or 7 columns: [mass] x y z vx vy vz)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/outputs/StarTrace_best_model.pt",
        help="Path to trained model checkpoint (default: model/outputs/StarTrace_best_model.pt)"
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Save results to JSON file (optional)"
    )
    output_group.add_argument(
        "--full_distribution",
        action="store_true",
        help="Include full probability distribution in output"
    )
    
    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()
    
    # Check if data file exists
    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"✗ Error: Data file not found at {data_path}")
        print(f"  Please check the path and try again.")
        sys.exit(1)
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"✗ Error: Model not found at {model_path}")
        print(f"  Please train a model first or specify a different path.")
        sys.exit(1)
    
    # Initialize predictor
    try:
        predictor = Predictor(model_path=str(model_path))
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)
    
    # Make prediction
    try:
        result = predictor.predict_from_file(
            str(data_path),
            return_full_distribution=args.full_distribution
        )
        
        # Print summary
        predictor.print_summary(result)
        
        # Save to JSON if requested
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"✓ Results saved to {output_path}")
        
        print("\nDone!")
        
    except KeyboardInterrupt:
        print("\n\nPrediction interrupted by user.")
        sys.exit(1)
    except ValueError as e:
        print(f"\n✗ Error processing input file: {e}")
        print(f"  Please check the file format and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Prediction failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
