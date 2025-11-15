"""GPU information retrieval functions."""

import os
import logging
import torch
from filelock import FileLock, Timeout

from .utils import get_nvidia_smi_data, match_pytorch_to_nvidia

logger = logging.getLogger("gpu_scheduler")

# Create a directory for GPU lock files if it doesn't exist
LOCK_DIR = os.path.expanduser("~/.gpu_locks")
os.makedirs(LOCK_DIR, exist_ok=True)

# List of GPU IDs that should never be used (e.g., reserved for other tasks)
# Can be overridden via environment variable or configuration
BANNED_GPUS = []

# List of GPU names that should never be used (e.g., specific models to avoid)
# Can be overridden via environment variable or configuration
BANNED_GPU_NAMES = ["Quadro RTX 6000"]


def get_gpu_memory_usage():
    """Get GPU memory usage in PyTorch ordering"""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        return []
    
    try:
        nvidia_gpus = get_nvidia_smi_data()
        gpu_info = []
        
        for pytorch_id in range(torch.cuda.device_count()):
            nvidia_data = match_pytorch_to_nvidia(pytorch_id, nvidia_gpus)
            if nvidia_data:
                gpu_info.append({
                    "gpu_id": pytorch_id,  # Use PyTorch index
                    "memory_used": nvidia_data['memory_used'],
                    "memory_total": nvidia_data['memory_total'],
                    "memory_free": nvidia_data['memory_free'],
                })
            else:
                # Fallback: use PyTorch properties directly
                props = torch.cuda.get_device_properties(pytorch_id)
                gpu_info.append({
                    "gpu_id": pytorch_id,
                    "memory_used": 0,  # Can't get current usage without nvidia-smi
                    "memory_total": props.total_memory // (1024 * 1024),  # Convert to MB
                    "memory_free": props.total_memory // (1024 * 1024),
                })
        
        return gpu_info
    except Exception as e:
        logger.error(f"Error getting GPU memory usage: {e}")
        return []


def get_gpu_names():
    """Get GPU names in PyTorch ordering"""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        return []
    
    try:
        nvidia_gpus = get_nvidia_smi_data()
        gpu_names = []
        
        for pytorch_id in range(torch.cuda.device_count()):
            nvidia_data = match_pytorch_to_nvidia(pytorch_id, nvidia_gpus)
            if nvidia_data:
                gpu_names.append({
                    "gpu_id": pytorch_id,  # Use PyTorch index
                    "name": nvidia_data['name'],
                })
            else:
                # Fallback: use PyTorch properties directly
                props = torch.cuda.get_device_properties(pytorch_id)
                gpu_names.append({
                    "gpu_id": pytorch_id,
                    "name": props.name,
                })
        
        return gpu_names
    except Exception as e:
        logger.error(f"Error getting GPU names: {e}")
        return []


def get_gpu_power_usage():
    """Get GPU power usage in PyTorch ordering"""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        return []
    
    try:
        nvidia_gpus = get_nvidia_smi_data()
        gpu_power_info = []
        
        for pytorch_id in range(torch.cuda.device_count()):
            nvidia_data = match_pytorch_to_nvidia(pytorch_id, nvidia_gpus)
            if nvidia_data:
                gpu_power_info.append({
                    "gpu_id": pytorch_id,  # Use PyTorch index
                    "power_draw": nvidia_data['power_draw'] or 0.0,
                    "power_limit": nvidia_data['power_limit'] or 0.0,
                })
            else:
                # Fallback: can't get power info without nvidia-smi
                gpu_power_info.append({
                    "gpu_id": pytorch_id,
                    "power_draw": 0.0,
                    "power_limit": 0.0,
                })
        
        return gpu_power_info
    except Exception as e:
        logger.error(f"Error getting GPU power usage: {e}")
        return []


def get_gpu_utilization():
    """Get GPU utilization in PyTorch ordering"""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        return []
    
    try:
        nvidia_gpus = get_nvidia_smi_data()
        gpu_utilization_info = []
        
        for pytorch_id in range(torch.cuda.device_count()):
            nvidia_data = match_pytorch_to_nvidia(pytorch_id, nvidia_gpus)
            if nvidia_data:
                gpu_utilization_info.append({
                    "gpu_id": pytorch_id,  # Use PyTorch index
                    "utilization": nvidia_data['utilization'] or 0,
                })
            else:
                # Fallback: can't get utilization info without nvidia-smi
                gpu_utilization_info.append({
                    "gpu_id": pytorch_id,
                    "utilization": 0,
                })
        
        return gpu_utilization_info
    except Exception as e:
        logger.error(f"Error getting GPU utilization: {e}")
        return []


def get_gpu_info():
    """Get comprehensive GPU information including memory, power usage, utilization, and names."""
    memory_info = get_gpu_memory_usage()
    power_info = get_gpu_power_usage()
    utilization_info = get_gpu_utilization()
    name_info = get_gpu_names()
    
    # Combine memory, power, utilization, and name info
    gpu_info = {}
    for gpu in memory_info:
        gpu_info[gpu["gpu_id"]] = gpu
    
    for gpu in power_info:
        gpu_id = gpu["gpu_id"]
        if gpu_id in gpu_info:
            gpu_info[gpu_id].update(gpu)
        else:
            # In case power info has GPUs not in memory info
            gpu_info[gpu_id] = gpu
    
    for gpu in utilization_info:
        gpu_id = gpu["gpu_id"]
        if gpu_id in gpu_info:
            gpu_info[gpu_id].update(gpu)
        else:
            # In case utilization info has GPUs not in other info
            gpu_info[gpu_id] = gpu
    
    for gpu in name_info:
        gpu_id = gpu["gpu_id"]
        if gpu_id in gpu_info:
            gpu_info[gpu_id].update(gpu)
        else:
            # In case name info has GPUs not in other info
            gpu_info[gpu_id] = gpu
    
    return list(gpu_info.values())


def get_total_num_gpus():
    """Get total number of available GPUs."""
    return torch.cuda.device_count()


def get_lock_path(gpu_id):
    """Get the lock file path for a specific GPU."""
    return os.path.join(LOCK_DIR, f"gpu_{gpu_id}.lock")


def is_gpu_locked(gpu_id):
    """Check if a GPU is already locked."""
    lock_path = get_lock_path(gpu_id)
    if not os.path.exists(lock_path):
        return False

    # Try to acquire the lock without blocking
    try:
        with FileLock(lock_path, timeout=0):
            return False
    except Timeout:
        return True


def cleanup_stale_locks():
    """
    Cleanup stale lock files that might have been left behind by crashed processes.
    This should be called just before acquiring a lock to avoid side effects and ensure
    stale locks don't prevent GPU acquisition.
    """
    if not os.path.exists(LOCK_DIR):
        return

    for filename in os.listdir(LOCK_DIR):
        if filename.endswith(".lock"):
            lock_path = os.path.join(LOCK_DIR, filename)
            try:
                # Try to acquire the lock without blocking
                with FileLock(lock_path, timeout=0):
                    # If we get here, the lock was not held by another process
                    # We can remove it as it might be stale
                    os.remove(lock_path)
                    logger.info(f"Removed stale lock file: {lock_path}")
            except Timeout:
                # Lock is currently held by a process, leave it alone
                continue
            except Exception as e:
                logger.warning(
                    f"Error while trying to clean lock file {lock_path}: {e}"
                )


def find_free_gpus(memory_free_threshold_gb=1, power_threshold_w=None, utilization_threshold_percent=1):
    """
    Find free GPUs based on free memory, power consumption, and GPU utilization.
    
    Args:
        memory_free_threshold_gb (float): Minimum free memory (in GB) required to consider a GPU as free.
                                         If 0, only checks power threshold (if provided) or returns all GPUs.
        power_threshold_w (float, optional): Maximum power usage (in Watts) to consider a GPU as free.
                                           If None, power is not considered.
        utilization_threshold_percent (float): Maximum GPU utilization (in %) to consider a GPU as free.
                                              Default is 1% to account for idle fluctuations.
    
    Returns:
        list: List of GPU info dictionaries for free GPUs.
    """
    if memory_free_threshold_gb == 0 and power_threshold_w is None and utilization_threshold_percent >= 100:
        # Return all non-banned GPUs without checking usage (only if utilization check is disabled)
        num_gpus = get_total_num_gpus()
        
        # Get GPU names to check against banned names
        name_info = get_gpu_names()
        name_dict = {gpu["gpu_id"]: gpu.get("name", "") for gpu in name_info}
        
        dummy_gpus = []
        for i in range(num_gpus):
            # Skip banned GPU IDs
            if i in BANNED_GPUS:
                continue
            
            # Skip banned GPU names
            gpu_name = name_dict.get(i, "")
            if any(banned_name in gpu_name for banned_name in BANNED_GPU_NAMES):
                logger.info(f"Skipping GPU {i} ({gpu_name}) - matches banned name pattern")
                continue
            
            # Skip locked GPUs
            if is_gpu_locked(i):
                logger.debug(f"Skipping GPU {i} - already locked by another process")
                continue
                
            dummy_gpus.append({"gpu_id": i})
        
        return dummy_gpus
    
    # Get comprehensive GPU information
    gpu_info = get_gpu_info()

    if not gpu_info:
        return []

    # Find GPUs that meet both memory and power criteria
    free_gpus = []
    for gpu in gpu_info:
        # Skip banned GPU IDs
        if gpu["gpu_id"] in BANNED_GPUS:
            continue
        
        # Skip banned GPU names
        gpu_name = gpu.get("name", "")
        if any(banned_name in gpu_name for banned_name in BANNED_GPU_NAMES):
            logger.info(f"Skipping GPU {gpu['gpu_id']} ({gpu_name}) - matches banned name pattern")
            continue
        
        # Check memory threshold (if memory_free_threshold_gb > 0)
        memory_ok = True
        if memory_free_threshold_gb > 0:
            memory_ok = gpu.get("memory_free", 0) >= memory_free_threshold_gb * 1024  # Convert GB to MB
        
        # Check power threshold (if power_threshold_w is specified)
        power_ok = True
        if power_threshold_w is not None:
            power_ok = gpu.get("power_draw", float('inf')) <= power_threshold_w
        
        # Check utilization threshold (default 1%)
        utilization_ok = True
        if utilization_threshold_percent < 100:
            utilization_ok = gpu.get("utilization", float('inf')) <= utilization_threshold_percent
        
        # GPU is free if it meets all criteria
        if memory_ok and power_ok and utilization_ok:
            # Also check if GPU is locked by another process
            if not is_gpu_locked(gpu["gpu_id"]):
                free_gpus.append(gpu)
            else:
                logger.debug(f"Skipping GPU {gpu['gpu_id']} - already locked by another process")

    return free_gpus

