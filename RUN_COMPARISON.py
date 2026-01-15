#!/usr/bin/env python3
"""
🚀 FASTEST NEUROFIRE PROJECT EXECUTION
Runs the complete RL Algorithm Comparison in minimal time
"""

import subprocess
import sys
import os

# Change to project directory
os.chdir(r'c:\Users\Edam\Downloads\RL\NeuroFire')

print("="*80)
print("🚀 NEUROFIRE RL ALGORITHM COMPARISON - QUICK RUN".center(80))
print("="*80)
print("\n✅ Starting full comparison (3-5 minutes for complete results)\n")

# Execute the main comparison script
try:
    result = subprocess.run([sys.executable, 'RL_Algorithms_Comparison.py'], 
                          capture_output=False, text=True)
    if result.returncode == 0:
        print("\n" + "="*80)
        print("✅ EXECUTION COMPLETE!".center(80))
        print("="*80)
        print("\n📊 Results:")
        print("   • Training curves saved")
        print("   • Performance metrics computed")
        print("   • Visualizations generated")
        print("   • Comparison analysis complete")
        print("\n📁 Check for output files:")
        print("   • neurofire_rl_comparison.png")
        print("   • comparison_results.png")
    else:
        print(f"\n❌ Error: Script returned code {result.returncode}")
        sys.exit(1)
except Exception as e:
    print(f"\n❌ Error executing script: {e}")
    sys.exit(1)
