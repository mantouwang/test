#!/usr/bin/env python3
"""
Experiment 1: CPDB network (ppi=0)
Cancer gene prediction using CPDB protein-protein interaction network
"""

import subprocess
import sys

def run_experiment():
    print("=" * 60)
    print("Experiment 1/6: CPDB network (ppi=0)")
    print("=" * 60)
    
    cmd = [
        "python", "main_transductive.py",
        "--ppi=0",
        "--max_epoch=2000",      # 增加预训练轮数
        "--max_epoch_f=1000",    # 增加微调轮数
        "--lr_f=0.01", 
        "--weight_decay_f=0.001",
        "--scheduler",            # 启用学习率调度器
        "--residual",
        "--inductive_ppi=-1"
    ]
    
    print("Command:", " ".join(cmd))
    print("\nStarting experiment...")
    
    try:
        result = subprocess.run(cmd + ["--save_model"], check=True)
        print("\nExperiment 1 completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nExperiment 1 failed with error code: {e.returncode}")
        return False

if __name__ == "__main__":
    success = run_experiment()
    if success:
        print("\nPress Enter to continue to next experiment...")
        input()
    else:
        print("\nExperiment failed. Please check the error messages above.")
        sys.exit(1)