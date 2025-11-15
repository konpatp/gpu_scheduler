"""GPU Scheduler class for managing GPU allocation."""

import os
import time
import torch
import logging
import random
from filelock import FileLock, Timeout

from .gpu_info import (
    get_gpu_info,
    find_free_gpus,
    get_lock_path,
    cleanup_stale_locks,
)

logger = logging.getLogger("gpu_scheduler")


class GPUScheduler:
    """
    A context manager for GPU scheduling that ensures exclusive access to GPUs.
    It waits until free GPUs are available, acquires locks on them, and sets one as
    the default CUDA device.
    """

    def __init__(
        self,
        memory_free_threshold_gb=20,
        power_threshold_w=80,
        utilization_threshold_percent=1,
        max_wait_time=float("inf"),
        poll_interval=10,
        specific_gpus=None,
        timeout=10,
        num_gpus=1,
        debug_mode=False,
    ):
        """
        Initialize the GPU scheduler.

        Args:
            memory_free_threshold_gb (float): Minimum free memory (in GB) required to consider a GPU as free
            power_threshold_w (float, optional): Maximum power usage (in Watts) to consider a GPU as free.
                                               If None, power is not considered. Typically 50W is a good idle threshold.
            utilization_threshold_percent (float): Maximum GPU utilization (in %) to consider a GPU as free.
                                                  Default is 1% to account for idle fluctuations.
            max_wait_time (int): Maximum time to wait for a free GPU (in seconds)
            poll_interval (int): Time between checks for a free GPU (in seconds)
            specific_gpus (list): List of specific GPU IDs to consider, or None to consider all
            timeout (int): Timeout for acquiring the lock (in seconds)
            num_gpus (int): Number of GPUs to allocate
            debug_mode (bool): If True, bypasses all GPU checks and just creates lock files for available GPUs
        """
        self.memory_free_threshold_gb = memory_free_threshold_gb
        self.power_threshold_w = power_threshold_w
        self.utilization_threshold_percent = utilization_threshold_percent
        self.max_wait_time = max_wait_time
        self.poll_interval = poll_interval
        self.specific_gpus = specific_gpus
        self.timeout = timeout
        self.num_gpus = num_gpus
        self.debug_mode = debug_mode
        self.locks = []
        self.gpu_ids = []
        self.prev_device = None

    def __enter__(self):
        """
        Wait for and acquire the requested number of free GPUs.
        Returns a list of acquired GPU IDs.
        """
        # Clean up stale locks before attempting to acquire GPUs
        cleanup_stale_locks()

        # Debug mode: bypass all checks and just grab available GPUs
        if self.debug_mode:
            logger.info("🔧 DEBUG MODE: Bypassing all GPU checks and grabbing first available GPUs")
            return self._debug_mode_acquire_gpus()

        start_time = time.time()
        while time.time() - start_time < self.max_wait_time:
            # Get free GPUs based on free memory, power consumption, and utilization
            free_gpus = find_free_gpus(self.memory_free_threshold_gb, self.power_threshold_w, self.utilization_threshold_percent)

            if not free_gpus:
                logger.info(
                    f"No free GPUs available. Waiting {self.poll_interval} seconds..."
                )
                time.sleep(self.poll_interval)
                continue

            # Filter GPUs if specific ones are requested
            if self.specific_gpus is not None:
                free_gpus = [
                    gpu for gpu in free_gpus if gpu["gpu_id"] in self.specific_gpus
                ]
                if not free_gpus:
                    logger.info(
                        f"No free GPUs available from the specified set. Waiting {self.poll_interval} seconds..."
                    )
                    time.sleep(self.poll_interval)
                    continue

            # Check if we have enough free GPUs
            if len(free_gpus) < self.num_gpus:
                logger.info(
                    f"Only {len(free_gpus)} GPUs available, but {self.num_gpus} requested. Waiting {self.poll_interval} seconds..."
                )
                time.sleep(self.poll_interval)
                continue

            # Try to acquire locks on the required number of GPUs
            acquired_locks = []
            acquired_gpu_ids = []

            # Shuffle the free_gpus list to prevent systematic deadlocks
            random.shuffle(free_gpus)

            # Try to lock as many GPUs as possible
            for gpu in free_gpus:
                if len(acquired_gpu_ids) >= self.num_gpus:
                    break  # We have enough GPUs, stop trying more

                gpu_id = gpu["gpu_id"]
                lock_path = get_lock_path(gpu_id)

                try:
                    # Try to acquire the lock
                    lock = FileLock(
                        lock_path, timeout=0
                    )  # Non-blocking to try all available GPUs
                    lock.acquire()

                    # Lock acquired, add to our lists
                    acquired_locks.append(lock)
                    acquired_gpu_ids.append(gpu_id)
                    logger.debug(f"Acquired lock for GPU {gpu_id}")
                    
                    # Log GPU metrics at acquisition time
                    memory_free_mb = gpu.get("memory_free", 0)
                    memory_free_gb = memory_free_mb / 1024 if memory_free_mb else 0
                    power_draw = gpu.get("power_draw", 0)
                    utilization = gpu.get("utilization", 0)
                    gpu_name = gpu.get("name", "Unknown")
                    
                    logger.info(f"GPU {gpu_id} ({gpu_name}) acquired - Memory: {memory_free_gb:.2f}GB free, "
                               f"Power: {power_draw}W, Utilization: {utilization}%")

                except Timeout:
                    # This GPU is locked by another process, try the next one
                    logger.debug(f"GPU {gpu_id} is locked by another process")
                    continue

            # Check if we got enough GPUs
            if len(acquired_gpu_ids) >= self.num_gpus:
                # If we have more GPUs than needed, release the extra ones
                while len(acquired_gpu_ids) > self.num_gpus:
                    extra_lock = acquired_locks.pop()
                    extra_gpu_id = acquired_gpu_ids.pop()
                    extra_lock.release()
                    logger.debug(f"Released extra GPU {extra_gpu_id}")

                # Set the GPUs we've acquired
                self.locks = acquired_locks
                self.gpu_ids = acquired_gpu_ids

                # Save the current device and set the first GPU as the default
                if torch.cuda.is_available():
                    self.prev_device = torch.cuda.current_device()
                    torch.cuda.set_device(self.gpu_ids[0])
                    logger.info(f"Acquired {len(self.gpu_ids)} GPUs: {self.gpu_ids}")
                    logger.info(f"Set GPU {self.gpu_ids[0]} as the default CUDA device")
                else:
                    logger.warning("PyTorch CUDA is not available. Cannot set device.")

                return self.gpu_ids
            else:
                # We didn't get enough GPUs, release the ones we acquired
                for lock in acquired_locks:
                    lock.release()

                logger.debug(
                    f"Failed to acquire {self.num_gpus} GPUs. Got only {len(acquired_gpu_ids)}. Retrying..."
                )

                # Use randomized backoff to prevent deadlocks
                backoff_time = self.poll_interval * (0.5 + random.random())
                logger.debug(f"Backing off for {backoff_time:.2f} seconds")
                time.sleep(backoff_time)

        # If we get here, we've exceeded the max wait time
        raise TimeoutError(
            f"Couldn't acquire {self.num_gpus} GPUs within {self.max_wait_time} seconds"
        )

    def _debug_mode_acquire_gpus(self):
        """
        Debug mode: Bypass all checks and just grab the first available unlocked GPUs.
        This is for debugging when you don't care about memory/power/utilization.
        """
        # Get all GPUs from nvidia-ml-py or fall back to basic list
        try:
            all_gpus = get_gpu_info()
            available_gpu_ids = [gpu["gpu_id"] for gpu in all_gpus]
        except Exception as e:
            logger.warning(f"Could not get GPU info, falling back to basic list: {e}")
            # Assume we have GPUs 0-7 available
            if torch.cuda.is_available():
                available_gpu_ids = list(range(torch.cuda.device_count()))
            else:
                logger.error("No CUDA GPUs available!")
                raise RuntimeError("No CUDA GPUs available in debug mode!")
        
        # Filter by specific_gpus if requested
        if self.specific_gpus is not None:
            available_gpu_ids = [gpu_id for gpu_id in available_gpu_ids if gpu_id in self.specific_gpus]
        
        # Try to acquire locks on available GPUs without any health checks
        acquired_locks = []
        acquired_gpu_ids = []
        
        for gpu_id in available_gpu_ids:
            if len(acquired_gpu_ids) >= self.num_gpus:
                break
            
            lock_path = get_lock_path(gpu_id)
            try:
                # Try to acquire the lock
                lock = FileLock(lock_path, timeout=0)  # Non-blocking
                lock.acquire()
                
                # Lock acquired
                acquired_locks.append(lock)
                acquired_gpu_ids.append(gpu_id)
                
                # In debug mode, try to get current GPU metrics for logging
                try:
                    gpu_info_list = get_gpu_info()
                    gpu_data = next((g for g in gpu_info_list if g["gpu_id"] == gpu_id), None)
                    if gpu_data:
                        memory_free_mb = gpu_data.get("memory_free", 0)
                        memory_free_gb = memory_free_mb / 1024 if memory_free_mb else 0
                        power_draw = gpu_data.get("power_draw", 0)
                        utilization = gpu_data.get("utilization", 0)
                        gpu_name = gpu_data.get("name", "Unknown")
                        logger.info(f"🔧 DEBUG: GPU {gpu_id} ({gpu_name}) acquired - Memory: {memory_free_gb:.2f}GB free, "
                                   f"Power: {power_draw}W, Utilization: {utilization}%")
                    else:
                        logger.info(f"🔧 DEBUG: Acquired GPU {gpu_id} (bypassing health checks)")
                except Exception:
                    logger.info(f"🔧 DEBUG: Acquired GPU {gpu_id} (bypassing health checks)")
                
            except Timeout:
                # This GPU is locked, try the next one
                logger.debug(f"🔧 DEBUG: GPU {gpu_id} is locked, trying next one")
                continue
        
        # Check if we got enough GPUs
        if len(acquired_gpu_ids) < self.num_gpus:
            # Release any we did acquire
            for lock in acquired_locks:
                lock.release()
            raise RuntimeError(f"🔧 DEBUG MODE: Could only acquire {len(acquired_gpu_ids)}/{self.num_gpus} GPUs")
        
        # Set the acquired GPUs
        self.locks = acquired_locks
        self.gpu_ids = acquired_gpu_ids
        
        # Save the current device and set the first GPU as the default
        if torch.cuda.is_available():
            self.prev_device = torch.cuda.current_device()
            torch.cuda.set_device(self.gpu_ids[0])
            logger.info(f"🔧 DEBUG: Acquired {len(self.gpu_ids)} GPUs: {self.gpu_ids}")
            logger.info(f"🔧 DEBUG: Set GPU {self.gpu_ids[0]} as the default CUDA device")
        else:
            logger.warning("🔧 DEBUG: PyTorch CUDA is not available. Cannot set device.")
        
        return self.gpu_ids

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the locks on all acquired GPUs."""
        for i, lock in enumerate(self.locks):
            if lock:
                gpu_id = self.gpu_ids[i]
                lock_path = get_lock_path(gpu_id)
                lock.release()

                # Remove the lock file after release
                if os.path.exists(lock_path):
                    try:
                        os.remove(lock_path)
                        logger.info(f"Removed lock file for GPU {gpu_id}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove lock file for GPU {gpu_id}: {e}"
                        )

        if self.gpu_ids:
            logger.info(f"Released GPUs: {self.gpu_ids}")

        # Restore the previous device if we changed it
        if self.prev_device is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.prev_device)

        self.locks = []
        self.gpu_ids = []
        self.prev_device = None

