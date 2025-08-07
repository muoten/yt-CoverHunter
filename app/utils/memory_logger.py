import os
import sys
import psutil
from app.parse_config import config

def log_detailed_memory():
    process = psutil.Process(os.getpid())
    
    # Get total VMS
    vms_total = process.memory_info().vms / (1024 * 1024)  # Convert to MB
    print(f"\n=== Total VMS: {vms_total:.2f}MB ===")
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    print(f"=== System Memory ===")
    print(f"Total: {system_memory.total / (1024**3):.2f}GB")
    print(f"Available: {system_memory.available / (1024**3):.2f}GB")
    print(f"Used: {system_memory.used / (1024**3):.2f}GB")
    print(f"Free: {system_memory.free / (1024**3):.2f}GB")
    print(f"Percent used: {system_memory.percent:.1f}%")
    
    if config['LOG_MEMORY']:
        print("\n=== Detailed Memory Mapping ===")
        total_mapped = 0
        for m in process.memory_maps():
            map_size = m.size / (1024 * 1024)
            total_mapped += map_size
            
            if map_size > 10:
                print(f"Path: {m.path}")
                print(f"RSS: {m.rss / 1024 / 1024:.2f}MB")
                print(f"Size: {map_size:.2f}MB")
                print("---")
        
        print(f"\nTotal mapped memory: {total_mapped:.2f}MB")
        print(f"Difference from VMS: {(vms_total - total_mapped):.2f}MB")
    

    