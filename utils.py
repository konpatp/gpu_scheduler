"""Utility functions for GPU scheduler package."""

import subprocess
import logging

logger = logging.getLogger("gpu_scheduler")


def safe_int_convert(value, default=None):
    """Safely convert a value to int, handling N/A and error states."""
    if value is None:
        return default
    value_str = str(value).strip()
    if value_str in ('N/A', '', 'Unknown') or value_str.startswith('['):
        return default
    try:
        return int(float(value_str))  # Convert via float first to handle "123.0" strings
    except (ValueError, TypeError):
        return default


def safe_float_convert(value, default=None):
    """Safely convert a value to float, handling N/A and error states."""
    if value is None:
        return default
    value_str = str(value).strip()
    if value_str in ('N/A', '', 'Unknown') or value_str.startswith('['):
        return default
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return default


def get_nvidia_smi_data():
    """Get GPU data from nvidia-smi keyed by UUID"""
    try:
        cmd = ['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,memory.free,memory.used,power.draw,power.limit,temperature.gpu,utilization.gpu', '--format=csv,noheader,nounits']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        nvidia_gpus = {}
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 10:
                    # Check if GPU is in a bad state (has error messages in any field)
                    has_error_state = any(
                        part.startswith('[') or part in ('N/A', 'Unknown', '')
                        for part in parts[3:]  # Check numeric fields
                    )
                    
                    # Try to parse nvidia_index and name first
                    try:
                        nvidia_index = safe_int_convert(parts[0])
                        gpu_name = parts[1]
                        gpu_uuid = parts[2]
                        
                        # If GPU appears to be in error state, log and skip it
                        if has_error_state or nvidia_index is None:
                            logger.warning(
                                f"Skipping GPU {nvidia_index} ({gpu_name}) - "
                                f"GPU appears to be in error state or not ready. "
                                f"Fields: memory_total={parts[3]}, memory_free={parts[4]}, "
                                f"power_draw={parts[6]}, utilization={parts[9]}"
                            )
                            continue
                        
                        # Parse all fields safely
                        memory_total = safe_int_convert(parts[3], default=0)
                        memory_free = safe_int_convert(parts[4], default=0)
                        memory_used = safe_int_convert(parts[5], default=0)
                        power_draw = safe_float_convert(parts[6])
                        power_limit = safe_float_convert(parts[7])
                        temperature = safe_int_convert(parts[8])
                        utilization = safe_int_convert(parts[9], default=0)
                        
                        nvidia_gpus[gpu_uuid] = {
                            'nvidia_index': nvidia_index,
                            'name': gpu_name,
                            'uuid': gpu_uuid,
                            'memory_total': memory_total,
                            'memory_free': memory_free,
                            'memory_used': memory_used,
                            'power_draw': power_draw,
                            'power_limit': power_limit,
                            'temperature': temperature,
                            'utilization': utilization
                        }
                    except Exception as e:
                        # If we can't parse this GPU, log and skip it
                        logger.warning(
                            f"Failed to parse GPU data from line: {line[:100]}. "
                            f"Error: {e}. Skipping this GPU."
                        )
                        continue
        return nvidia_gpus
    except Exception as e:
        logger.error(f"Could not get nvidia-smi data: {e}")
        return {}


def match_pytorch_to_nvidia(device_id, nvidia_gpus):
    """Match PyTorch GPU to nvidia-smi GPU using UUID"""
    import torch
    try:
        props = torch.cuda.get_device_properties(device_id)
        pytorch_uuid = str(props.uuid)
        
        # Add GPU- prefix to match nvidia-smi format
        if not pytorch_uuid.startswith('GPU-'):
            pytorch_uuid = f'GPU-{pytorch_uuid}'
            
        if pytorch_uuid in nvidia_gpus:
            return nvidia_gpus[pytorch_uuid]
    except Exception:
        pass
    return None

