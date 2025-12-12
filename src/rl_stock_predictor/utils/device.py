"""Device detection and management for training."""
import torch
import os


def get_device(device: str = None) -> str:
    """
    Get the appropriate device for training/inference.

    Automatically detects available devices with priority:
    1. Explicit device argument
    2. DEVICE environment variable
    3. Auto-detection: MPS (Apple Silicon) > CUDA > CPU

    Args:
        device: Optional explicit device specification ('cpu', 'cuda', 'mps', or 'auto')

    Returns:
        Device string ('cpu', 'cuda', or 'mps')

    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device('mps')  # Force MPS
        >>> device = get_device('auto')  # Explicitly auto-detect
    """
    # If explicitly specified, use that (but validate)
    if device and device != 'auto':
        device = device.lower()
        if device == 'mps':
            if not torch.backends.mps.is_available():
                print(f"Warning: MPS requested but not available. Falling back to CPU.")
                return 'cpu'
            return 'mps'
        elif device == 'cuda':
            if not torch.cuda.is_available():
                print(f"Warning: CUDA requested but not available. Falling back to CPU.")
                return 'cpu'
            return 'cuda'
        elif device == 'cpu':
            return 'cpu'
        else:
            print(f"Warning: Unknown device '{device}'. Falling back to auto-detection.")

    # Check environment variable
    env_device = os.getenv('DEVICE', '').lower()
    if env_device and env_device != 'auto':
        return get_device(env_device)

    # Auto-detection with priority: MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        print("Auto-detected: Using MPS (Apple Silicon GPU) for acceleration")
        return 'mps'
    elif torch.cuda.is_available():
        print("Auto-detected: Using CUDA (NVIDIA GPU) for acceleration")
        return 'cuda'
    else:
        print("Auto-detected: Using CPU (no GPU acceleration available)")
        return 'cpu'


def print_device_info(device: str):
    """
    Print information about the selected device.

    Args:
        device: Device string ('cpu', 'cuda', or 'mps')
    """
    print(f"\n{'='*60}")
    print(f"Device Configuration")
    print(f"{'='*60}")
    print(f"Selected device: {device.upper()}")

    if device == 'mps':
        print(f"Backend: Metal Performance Shaders (Apple Silicon)")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
    elif device == 'cuda':
        print(f"Backend: CUDA (NVIDIA)")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"Backend: CPU (no GPU acceleration)")
        print(f"Note: Training will be slower without GPU acceleration")
        print(f"  - For Apple Silicon: Set DEVICE=mps in .env")
        print(f"  - For NVIDIA GPUs: Set DEVICE=cuda in .env")

    print(f"{'='*60}\n")


def optimize_for_device(device: str):
    """
    Apply device-specific optimizations.

    Args:
        device: Device string ('cpu', 'cuda', or 'mps')
    """
    if device == 'mps':
        # MPS-specific optimizations
        # Note: As of PyTorch 2.0+, MPS is relatively stable but some ops may fall back to CPU
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print("Enabled MPS fallback for unsupported operations")
    elif device == 'cuda':
        # CUDA-specific optimizations
        torch.backends.cudnn.benchmark = True
        print("Enabled cuDNN autotuner for optimal performance")
