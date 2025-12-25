#!/usr/bin/env python3
"""Diagnostic script for CUDA and JAX setup"""

import sys
print("="*60)
print("CUDA AND JAX DIAGNOSTIC SCRIPT")
print("="*60)

# 1. Check Python version
print(f"\n1. Python Version: {sys.version}")

# 2. Check JAX installation
print("\n2. JAX Installation:")
try:
    import jax
    print(f"   ✓ JAX version: {jax.__version__}")
    print(f"   ✓ JAX location: {jax.__file__}")
except Exception as e:
    print(f"   ✗ JAX import failed: {e}")
    sys.exit(1)

# 3. Check jaxlib installation
print("\n3. JAXlib Installation:")
try:
    import jaxlib
    print(f"   ✓ JAXlib version: {jaxlib.__version__}")
    print(f"   ✓ JAXlib location: {jaxlib.__file__}")
except Exception as e:
    print(f"   ✗ JAXlib import failed: {e}")

# 4. Check CUDA libraries
print("\n4. CUDA Libraries:")
try:
    from jaxlib import xla_extension
    print(f"   ✓ XLA extension loaded")
    try:
        cuda_version = xla_extension.GpuDevice.cuda_version()
        print(f"   ✓ CUDA version from JAX: {cuda_version}")
    except:
        print(f"   ✗ Could not get CUDA version from JAX")
except Exception as e:
    print(f"   ✗ XLA extension failed: {e}")

# 5. Check available backends
print("\n5. JAX Available Backends:")
try:
    from jax.lib import xla_bridge
    print(f"   Backend: {xla_bridge.get_backend().platform}")
    print(f"   Available platforms: {xla_bridge.get_backend().platform}")
except Exception as e:
    print(f"   ✗ Could not get backend info: {e}")

# 6. Check devices
print("\n6. JAX Devices:")
try:
    devices = jax.devices()
    print(f"   ✓ Number of devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"   Device {i}: {device}")
except Exception as e:
    print(f"   ✗ Could not get devices: {e}")

# 7. Check local devices
print("\n7. JAX Local Devices:")
try:
    local_devices = jax.local_devices()
    print(f"   ✓ Number of local devices: {len(local_devices)}")
    for i, device in enumerate(local_devices):
        print(f"   Local Device {i}: {device}")
except Exception as e:
    print(f"   ✗ Could not get local devices: {e}")

# 8. Try to initialize CUDA
print("\n8. CUDA Initialization Test:")
try:
    import jax.numpy as jnp
    # Try a simple operation on GPU
    x = jnp.array([1.0, 2.0, 3.0])
    y = x + 1
    print(f"   ✓ Simple JAX operation successful")
    print(f"   Device used: {x.device()}")
except Exception as e:
    print(f"   ✗ JAX operation failed: {e}")

# 9. Check environment variables
print("\n9. Environment Variables:")
import os
cuda_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'LD_LIBRARY_PATH', 
             'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_FLAGS']
for var in cuda_vars:
    value = os.environ.get(var, 'NOT SET')
    print(f"   {var}: {value}")

# 10. Check CUDA from system
print("\n10. System CUDA Check:")
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("   ✓ nvidia-smi works:")
        print("   " + "\n   ".join(result.stdout.split('\n')[:10]))
    else:
        print(f"   ✗ nvidia-smi failed: {result.stderr}")
except Exception as e:
    print(f"   ✗ Could not run nvidia-smi: {e}")

# 11. Check nvcc
print("\n11. CUDA Compiler (nvcc):")
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("   ✓ nvcc found:")
        print("   " + result.stdout.strip())
    else:
        print(f"   ✗ nvcc not found or failed")
except Exception as e:
    print(f"   ✗ Could not run nvcc: {e}")

# 12. Detailed JAX debug
print("\n12. Detailed JAX CUDA Debug:")
print("   Set TF_CPP_MIN_LOG_LEVEL=0 for more details")
try:
    # Force JAX to print more info
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['JAX_DEBUG_NANS'] = 'True'
    
    # Try to explicitly get CUDA backend
    from jax.lib import xla_bridge
    try:
        cuda_backend = xla_bridge.get_backend('cuda')
        print(f"   ✓ CUDA backend accessible: {cuda_backend}")
    except Exception as e:
        print(f"   ✗ Cannot get CUDA backend: {e}")
    
    try:
        gpu_backend = xla_bridge.get_backend('gpu')
        print(f"   ✓ GPU backend accessible: {gpu_backend}")
    except Exception as e:
        print(f"   ✗ Cannot get GPU backend: {e}")
        
except Exception as e:
    print(f"   ✗ Debug failed: {e}")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)

