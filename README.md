# GPU Scheduler

A Python package for managing GPU allocation and scheduling in multi-GPU environments. Provides automatic GPU discovery, health checks, and file-based locking to ensure exclusive access to GPUs.

## Features

- **Automatic GPU Discovery**: Detects available GPUs using nvidia-smi and PyTorch
- **Health Checks**: Filters GPUs based on memory, power consumption, and utilization
- **File-based Locking**: Ensures exclusive access to GPUs across processes (uses `/tmp/gpu_locks` to avoid NFS sharing issues)
- **Context Manager Interface**: Easy-to-use context manager for GPU allocation
- **CLI Tool**: Command-line interface for finding free GPUs and setting `CUDA_VISIBLE_DEVICES`
- **Debug Mode**: Bypass health checks for development and debugging

## Installation

This package is designed to be used as a git submodule. To use it in your project:

1. Add as a git submodule:
```bash
git submodule add git@github.com:konpatp/gpu_scheduler.git lib/gpu_scheduler
git submodule update --init --recursive
```

2. Install as an editable package:
```bash
pip install -e lib/gpu_scheduler
```

## Dependencies

- `torch` (PyTorch)
- `filelock`

## Quick Start

### Python API

```python
from gpu_scheduler import GPUScheduler

# Acquire a single GPU with default settings
with GPUScheduler() as gpu_ids:
    device = f"cuda:{gpu_ids[0]}"  # Recommended: explicit device
    model = model.to(device)
```

### CLI Usage

```bash
# Find 1 free GPU and set CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(gpu-scheduler --num-gpus 1) accelerate launch ...

# Find 2 free GPUs
CUDA_VISIBLE_DEVICES=$(gpu-scheduler --num-gpus 2) python train.py

# Custom thresholds
CUDA_VISIBLE_DEVICES=$(gpu-scheduler --num-gpus 1 --memory-threshold-gb 40 --power-threshold-w 50) python train.py
```

## Usage Examples

### Python API

#### Basic Usage
```python
from gpu_scheduler import GPUScheduler

# Single GPU with memory constraint
with GPUScheduler(memory_free_threshold_gb=20) as gpu_ids:
    device = f"cuda:{gpu_ids[0]}"
    print(f"Using GPU: {gpu_ids[0]}")
    model = model.to(device)
```

#### Multiple GPUs
```python
# Acquire multiple GPUs
with GPUScheduler(
    num_gpus=2,
    memory_free_threshold_gb=10,
    power_threshold_w=50,
    utilization_threshold_percent=None  # None = not checked
) as gpu_ids:
    print(f"Using GPUs: {gpu_ids}")
    device1 = f"cuda:{gpu_ids[0]}"
    device2 = f"cuda:{gpu_ids[1]}"
```

#### Debug Mode
```python
# Bypass all health checks
with GPUScheduler(debug_mode=True) as gpu_ids:
    print(f"Using GPU: {gpu_ids[0]}")
```

### CLI Options

```bash
gpu-scheduler --help
```

**Options:**
- `--num-gpus`: Number of free GPUs to find (default: 1)
- `--memory-threshold-gb`: Minimum free memory in GB (default: 20)
- `--power-threshold-w`: Maximum power usage in Watts (default: 80)
- `--utilization-threshold-percent`: Maximum GPU utilization % (default: None, not checked)

## Important: Device Handling

**⚠️ Important Note about `cuda:0` vs Default Device:**

When `GPUScheduler` sets the default device, it uses `torch.cuda.set_device(gpu_ids[0])`. However, **`cuda:0` always refers to GPU index 0**, not necessarily the scheduled GPU.

**Example:**
```python
with GPUScheduler() as gpu_ids:
    # If scheduler acquired GPU 3:
    device1 = torch.device("cuda:0")      # ❌ Still refers to GPU 0!
    device2 = torch.device(f"cuda:{gpu_ids[0]}")  # ✅ Correct: uses GPU 3
    device3 = torch.device("cuda")        # ✅ Uses default (GPU 3)
```

**Best Practice:** Always use `f"cuda:{gpu_ids[0]}"` to explicitly target the scheduled GPU.

## API Reference

### GPUScheduler

Main class for GPU scheduling. Use as a context manager.

**Parameters:**
- `memory_free_threshold_gb` (float): Minimum free memory in GB (default: 20)
- `power_threshold_w` (float, optional): Maximum power usage in Watts (default: 80)
- `utilization_threshold_percent` (float, optional): Maximum GPU utilization % (default: None, not checked)
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
- `find_free_gpus(memory_free_threshold_gb, power_threshold_w, utilization_threshold_percent)`: Find free GPUs based on criteria

## Configuration

- **Lock Directory**: `/tmp/gpu_locks` (local to each machine, avoids NFS sharing issues)
- **Banned GPUs**: Configure via `BANNED_GPUS` list in `gpu_info.py`
- **Banned GPU Names**: Configure via `BANNED_GPU_NAMES` list in `gpu_info.py`

## License

Private library - for internal use only.
