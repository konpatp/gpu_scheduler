"""Command-line interface for GPU scheduler."""

import argparse
import sys
from .gpu_info import find_free_gpus


def main():
    """Main CLI entry point for finding free GPUs."""
    parser = argparse.ArgumentParser(
        description="Find free GPUs and output comma-separated GPU IDs for CUDA_VISIBLE_DEVICES"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of free GPUs to find (default: 1)",
    )
    parser.add_argument(
        "--memory-threshold-gb",
        type=float,
        default=20,
        help="Minimum free (unused) memory in GB required for a GPU to be considered free (default: 20)",
    )
    parser.add_argument(
        "--power-threshold-w",
        type=float,
        default=80,
        help="Maximum power usage in Watts to consider a GPU as free (default: 80)",
    )
    parser.add_argument(
        "--utilization-threshold-percent",
        type=float,
        default=None,
        help="Maximum GPU utilization percentage to consider a GPU as free (default: None, not checked)",
    )
    
    args = parser.parse_args()
    
    # Find free GPUs based on the criteria
    free_gpus = find_free_gpus(
        memory_free_threshold_gb=args.memory_threshold_gb,
        power_threshold_w=args.power_threshold_w,
        utilization_threshold_percent=args.utilization_threshold_percent,
    )
    
    # Check if we have enough free GPUs
    if len(free_gpus) < args.num_gpus:
        print(
            f"Error: Found only {len(free_gpus)} free GPU(s), but {args.num_gpus} requested",
            file=sys.stderr,
        )
        sys.exit(1)
    
    # Extract GPU IDs and take the first N
    gpu_ids = [gpu["gpu_id"] for gpu in free_gpus[: args.num_gpus]]
    
    # Output comma-separated GPU IDs
    print(",".join(map(str, gpu_ids)))


