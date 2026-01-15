import sys
with open('debug_log.txt', 'w') as f:
    f.write(f"Python executable: {sys.executable}\n")
    try:
        import torch
        f.write("Success: Imported torch\n")
    except ImportError as e:
        f.write(f"val: Failed to import torch: {e}\n")
        
    try:
        import pygame
        f.write("Success: Imported pygame\n")
    except ImportError as e:
        f.write(f"val: Failed to import pygame: {e}\n")

    try:
        import matplotlib
        f.write("Success: Imported matplotlib\n")
    except ImportError as e:
        f.write(f"val: Failed to import matplotlib: {e}\n")
