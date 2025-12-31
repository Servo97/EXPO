#!/usr/bin/env python3
"""Script to inspect AWAC data format structure"""
import numpy as np

# Load one of the existing AWAC format files to understand structure
door_data = np.load('/home/mananaga/.datasets/awac-data/door2_sparse.npy', allow_pickle=True)
door_bc = np.load('/home/mananaga/.datasets/awac-data/door_bc_sparse4.npy', allow_pickle=True)

print("door2_sparse.npy structure:")
print(f"Type: {type(door_data)}")
print(f"Shape: {door_data.shape}")
print(f"dtype: {door_data.dtype}")
if door_data.shape[0] > 0:
    print(f"\nFirst element type: {type(door_data[0])}")
    print(f"First element keys: {door_data[0].keys() if hasattr(door_data[0], 'keys') else 'N/A'}")
    if hasattr(door_data[0], 'keys'):
        for key in list(door_data[0].keys())[:10]:
            val = door_data[0][key]
            if hasattr(val, 'shape'):
                print(f"  {key}: shape={val.shape if hasattr(val, '__len__') else 'scalar'}, dtype={val.dtype if hasattr(val, 'dtype') else 'N/A'}")
            else:
                print(f"  {key}: {type(val)}")

print("\n" + "="*60)
print("\ndoor_bc_sparse4.npy structure:")
print(f"Type: {type(door_bc)}")
if hasattr(door_bc, 'shape'):
    print(f"Shape: {door_bc.shape}")
    print(f"dtype: {door_bc.dtype}")
else:
    print(f"Length: {len(door_bc)}")
if len(door_bc) > 0:
    print(f"\nFirst element type: {type(door_bc[0])}")
    print(f"First element keys: {door_bc[0].keys() if hasattr(door_bc[0], 'keys') else 'N/A'}")
    if hasattr(door_bc[0], 'keys'):
        for key in list(door_bc[0].keys())[:15]:
            val = door_bc[0][key]
            if hasattr(val, 'shape'):
                print(f"  {key}: shape={val.shape if hasattr(val, '__len__') else 'scalar'}, dtype={val.dtype if hasattr(val, 'dtype') else 'N/A'}")
            else:
                print(f"  {key}: {type(val)}")
