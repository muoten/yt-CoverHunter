import os
import sys
import psutil

def log_detailed_memory():
    process = psutil.Process(os.getpid())
    
    # Get total VMS
    vms_total = process.memory_info().vms / (1024 * 1024)  # Convert to MB
    print(f"\n=== Total VMS: {vms_total:.2f}MB ===")
    
    print("\n=== Detailed Memory Mapping ===")
    
    # Track total mapped memory
    total_mapped = 0
    
    # Log memory maps
    for m in process.memory_maps():
        map_size = m.size / (1024 * 1024)  # Convert to MB
        total_mapped += map_size
        if map_size > 10:
            print(f"Path: {m.path}")
            print(f"RSS: {m.rss / 1024 / 1024:.2f}MB")
            print(f"Size: {map_size:.2f}MB")
            print("---")
    
    print(f"\nTotal mapped memory: {total_mapped:.2f}MB")
    print(f"Difference from VMS: {(vms_total - total_mapped):.2f}MB")
    
    import torch
    print(torch.__version__)
    