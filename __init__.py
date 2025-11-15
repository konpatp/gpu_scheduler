"""GPU Scheduler - A Python package for managing GPU allocation and scheduling."""

import atexit
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from .scheduler import GPUScheduler
from .gpu_info import (
    get_gpu_info,
    get_gpu_memory_usage,
    get_gpu_names,
    get_gpu_power_usage,
    get_gpu_utilization,
    get_total_num_gpus,
    find_free_gpus,
    cleanup_stale_locks,
)

__version__ = "0.1.0"

__all__ = [
    "GPUScheduler",
    "get_gpu_info",
    "get_gpu_memory_usage",
    "get_gpu_names",
    "get_gpu_power_usage",
    "get_gpu_utilization",
    "get_total_num_gpus",
    "find_free_gpus",
    "cleanup_stale_locks",
]

# Register cleanup function to run at process exit
atexit.register(cleanup_stale_locks)

