#!/usr/bin/env python3
"""
Project Initialization Script for CSE480 Machine Vision Project
Creates the required directory structure for the Action & Emotion Recognition project.
"""

import os
from pathlib import Path


def create_directory_structure(base_path="."):
    """
    Creates the required directory structure for the CSE480 project.
    
    Args:
        base_path (str): Base path where directories will be created (default: current directory)
    """
    directories = [
        "data/raw",           # For original datasets (FER-2013, UCF-101)
        "data/processed",     # For resized images and processed data
        "src",                # For source code
        "models",             # To save trained .keras files
        "notebooks",          # For experiments
        "reports",            # For milestone reports
    ]
    
    base = Path(base_path)
    created_dirs = []
    existing_dirs = []
    
    for directory in directories:
        dir_path = base / directory
        if dir_path.exists():
            existing_dirs.append(str(dir_path))
            print(f"✓ Directory already exists: {dir_path}")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(dir_path))
            print(f"✓ Created directory: {dir_path}")
    
    print("\n" + "="*60)
    print("Project structure initialization complete!")
    print("="*60)
    if created_dirs:
        print(f"\nCreated {len(created_dirs)} new directory(ies):")
        for d in created_dirs:
            print(f"  - {d}")
    if existing_dirs:
        print(f"\nFound {len(existing_dirs)} existing directory(ies):")
        for d in existing_dirs:
            print(f"  - {d}")
    print()


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    create_directory_structure(script_dir)

