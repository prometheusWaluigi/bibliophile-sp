#!/usr/bin/env python3
"""
Test script to verify the project setup.
This script checks if all required packages are installed and if the project structure is correct.
"""

import os
import sys
import importlib.util
import subprocess


def check_package(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return False
    return True


def check_file_exists(file_path):
    """Check if a file exists."""
    return os.path.isfile(file_path)


def check_directory_exists(dir_path):
    """Check if a directory exists."""
    return os.path.isdir(dir_path)


def print_status(message, status):
    """Print a status message."""
    status_str = "✅" if status else "❌"
    print(f"{status_str} {message}")
    return status


def main():
    """Run the setup test."""
    print("🔍 Testing BibliophileSP project setup...\n")
    
    # Check required packages
    packages = [
        "requests",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "dotenv"
    ]
    
    print("📦 Checking required packages:")
    all_packages_installed = True
    for package in packages:
        package_installed = check_package(package)
        if not package_installed:
            all_packages_installed = False
        print_status(f"{package}", package_installed)
    
    # Check project structure
    print("\n📁 Checking project structure:")
    
    structure_correct = True
    
    # Check files
    files = [
        "pyproject.toml",
        ".env.template",
        "Dockerfile",
        "src/main.py",
        "src/analysis.py",
        "src/spapi.py"
    ]
    
    for file in files:
        file_exists = check_file_exists(file)
        if not file_exists:
            structure_correct = False
        print_status(f"File: {file}", file_exists)
    
    # Check directories
    directories = [
        "src",
        "output"
    ]
    
    for directory in directories:
        dir_exists = check_directory_exists(directory)
        if not dir_exists and directory != "output":  # output will be created at runtime
            structure_correct = False
        print_status(f"Directory: {directory}", dir_exists)
    
    # Print summary
    print("\n📋 Summary:")
    print_status("All required packages installed", all_packages_installed)
    print_status("Project structure correct", structure_correct)
    
    if not all_packages_installed:
        print("\n⚠️ Some required packages are missing. Install them with:")
        print("poetry install")
    
    if not structure_correct:
        print("\n⚠️ Project structure is incomplete.")
    
    if all_packages_installed and structure_correct:
        print("\n🎉 Setup test passed! The project is ready to run.")
        print("Run the project with: poetry run python src/main.py")


if __name__ == "__main__":
    main()
