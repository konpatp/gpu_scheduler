# GPU Scheduler

A Python package for managing GPU allocation and scheduling in multi-GPU environments. Provides automatic GPU discovery, health checks, and file-based locking to ensure exclusive access to GPUs.

## Features

- **Automatic GPU Discovery**: Detects available GPUs using nvidia-smi and PyTorch
- **Health Checks**: Filters GPUs based on memory, power consumption, and utilization
- **File-based Locking**: Ensures exclusive access to GPUs across processes
- **Context Manager Interface**: Easy-to-use context manager for GPU allocation
- **Debug Mode**: Bypass health checks for development and debugging

## Installation

This package is designed to be used as a git submodule. To use it in your project:

1. Add as a git submodule:
```bash
git submodule add <private-repo-url> gpu_scheduler_package
git submodule update --init --recursive
```

2. Add the submodule directory to your Python path (see usage examples below)

## Dependencies

- `torch` (PyTorch)
- `filelock`

## Quick Start

```python
from gpu_scheduler import GPUScheduler

# Acquire a single GPU with default settings
with GPUScheduler() as gpu_ids:
    # Option 1: Explicitly use the scheduled GPU (recommended)
    device = f"cuda:{gpu_ids[0]}"
    model = model.to(device)
    
    # Option 2: Use default device (scheduler sets it automatically)
    device = torch.device("cuda")  # Uses the scheduled GPU
    model = model.to(device)
```

## Important: Device Handling

**⚠️ Important Note about `cuda:0` vs Default Device:**

When `GPUScheduler` sets the default device, it uses `torch.cuda.set_device(gpu_ids[0])`. However, **`cuda:0` always refers to GPU index 0** (the first GPU in PyTorch's ordering), not necessarily the scheduled GPU.

**Example:**
```python
with GPUScheduler() as gpu_ids:
    # If scheduler acquired GPU 3:
    # - torch.cuda.current_device() returns 3
    # - torch.cuda.set_device(3) was called
    
    device1 = torch.device("cuda:0")      # ❌ Still refers to GPU 0, not GPU 3!
    device2 = torch.device(f"cuda:{gpu_ids[0]}")  # ✅ Correct: uses GPU 3
    device3 = torch.device("cuda")        # ✅ Uses default (GPU 3, set by scheduler)
```

**Best Practices:**
1. **Explicit device string**: Use `f"cuda:{gpu_ids[0]}"` to explicitly target the scheduled GPU
2. **Default device**: Use `torch.device("cuda")` to use the default device (which the scheduler set)
3. **Avoid hardcoding**: Never hardcode `cuda:0` - it may not be the scheduled GPU

## Usage Examples

### Basic Usage

```python
from gpu_scheduler import GPUScheduler

# Single GPU with memory constraint
with GPUScheduler(memory_free_threshold_gb=20) as gpu_ids:
    # Explicitly use the scheduled GPU (recommended)
    device = f"cuda:{gpu_ids[0]}"
    print(f"Using GPU: {gpu_ids[0]} on device: {device}")
    
    # Or use default device (scheduler already set it)
    device = torch.device("cuda")
    
    # Your code here
    model = model.to(device)
```

### Multiple GPUs

```python
# Acquire multiple GPUs
with GPUScheduler(
    num_gpus=2,
    memory_free_threshold_gb=10,
    power_threshold_w=50,
    utilization_threshold_percent=1
) as gpu_ids:
    print(f"Using GPUs: {gpu_ids}")
    
    # Use the first scheduled GPU as default
    device = f"cuda:{gpu_ids[0]}"
    
    # Or access specific GPUs
    device1 = f"cuda:{gpu_ids[0]}"
    device2 = f"cuda:{gpu_ids[1]}"
    
    # Your code here
```

### Debug Mode

```python
# Bypass all health checks (useful for development)
with GPUScheduler(debug_mode=True) as gpu_ids:
    print(f"Using GPU: {gpu_ids[0]}")
    # Your code here
```

### GPU Information

```python
from gpu_scheduler import get_gpu_info, find_free_gpus

# Get comprehensive GPU information
gpu_info = get_gpu_info()
for gpu in gpu_info:
    print(f"GPU {gpu['gpu_id']}: {gpu['name']}")
    print(f"  Memory: {gpu['memory_free']}MB free / {gpu['memory_total']}MB total")
    print(f"  Power: {gpu['power_draw']}W / {gpu['power_limit']}W")
    print(f"  Utilization: {gpu['utilization']}%")

# Find free GPUs
free_gpus = find_free_gpus(
    memory_free_threshold_gb=20,
    power_threshold_w=50,
    utilization_threshold_percent=1
)
print(f"Found {len(free_gpus)} free GPUs")
```

## API Reference

### GPUScheduler

Main class for GPU scheduling. Use as a context manager.

**Parameters:**
- `memory_free_threshold_gb` (float): Minimum free memory in GB (default: 20)
- `power_threshold_w` (float, optional): Maximum power usage in Watts (default: 80)
- `utilization_threshold_percent` (float): Maximum GPU utilization % (default: 1)
- `max_wait_time` (float): Maximum time to wait for free GPU in seconds (default: inf)
- `poll_interval` (int): Time between checks in seconds (default: 10)
- `specific_gpus` (list, optional): List of specific GPU IDs to consider
- `timeout` (int): Timeout for acquiring lock in seconds (default: 10)
- `num_gpus` (int): Number of GPUs to allocate (default: 1)
- `debug_mode` (bool): Bypass health checks (default: False)

**Returns:**
- List of GPU IDs when used as context manager

### Utility Functions

- `get_gpu_info()`: Get comprehensive GPU information
- `get_gpu_memory_usage()`: Get GPU memory usage
- `get_gpu_names()`: Get GPU names
- `get_gpu_power_usage()`: Get GPU power usage
- `get_gpu_utilization()`: Get GPU utilization
- `get_total_num_gpus()`: Get total number of GPUs
- `find_free_gpus(...)`: Find free GPUs based on criteria

## Configuration

The package uses the following default configuration:

- **Lock Directory**: `~/.gpu_locks`
- **Banned GPUs**: Can be configured via `BANNED_GPUS` list in `gpu_info.py`
- **Banned GPU Names**: Can be configured via `BANNED_GPU_NAMES` list in `gpu_info.py`

## Git Submodule Integration

When using as a git submodule, imports work directly since the package is at the root level:

```python
from gpu_scheduler import GPUScheduler

with GPUScheduler() as gpu_ids:
    # Your code here
```

No path manipulation is needed - the submodule directory (`./gpu_scheduler/`) is automatically in the Python path when it's at the repository root.

## License

Private library - for internal use only.

