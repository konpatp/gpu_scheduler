"""Command-line interface for GPU scheduler."""

import argparse
import sys
import time
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
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Exit immediately if GPUs are not available (default: wait for GPUs)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Interval in seconds between checks when waiting for GPUs (default: 10 seconds)",
    )
    parser.add_argument(
        "--max-wait-time",
        type=int,
        default=None,
        help="Maximum time in seconds to wait for GPUs (default: None, wait indefinitely)",
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    last_report_time = 0
    
    while True:
        # Find free GPUs based on the criteria
        free_gpus = find_free_gpus(
            memory_free_threshold_gb=args.memory_threshold_gb,
            power_threshold_w=args.power_threshold_w,
            utilization_threshold_percent=args.utilization_threshold_percent,
        )
        
        # Check if we have enough free GPUs
        if len(free_gpus) >= args.num_gpus:
            # Extract GPU IDs and take the first N
            gpu_ids = [gpu["gpu_id"] for gpu in free_gpus[: args.num_gpus]]
            
            # Output comma-separated GPU IDs to stdout
            print(",".join(map(str, gpu_ids)))
            sys.exit(0)
        
        # Not enough GPUs found
        if args.no_wait:
            # If --no-wait is specified, exit with error immediately
            print(
                f"Error: Found only {len(free_gpus)} free GPU(s), but {args.num_gpus} requested",
                file=sys.stderr,
            )
            sys.exit(1)
        
        # Check if we've exceeded max wait time
        elapsed = time.time() - start_time
        if args.max_wait_time is not None and elapsed >= args.max_wait_time:
            print(
                f"Error: Timeout after {elapsed:.0f} seconds. Found only {len(free_gpus)} free GPU(s), but {args.num_gpus} requested",
                file=sys.stderr,
            )
            sys.exit(1)
        
        # Report progress every poll_interval seconds
        current_time = time.time()
        if current_time - last_report_time >= args.poll_interval:
            print(
                f"Waiting for {args.num_gpus} free GPU(s)... Found {len(free_gpus)} free GPU(s). "
                f"Waiting {args.poll_interval} seconds... (elapsed: {elapsed:.0f}s)",
                file=sys.stderr,
            )
            last_report_time = current_time
        
        # Wait before next check
        time.sleep(args.poll_interval)


