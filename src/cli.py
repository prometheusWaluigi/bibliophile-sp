#!/usr/bin/env python3
"""
Command-line interface for BibliophileSP.
"""

import argparse
import os
import sys
try:
    # Try relative import (when used as a package)
    from .main import run_inventory_analysis
    from .oneapi_accelerator import ONEAPI_AVAILABLE, APPLE_ACCELERATE_AVAILABLE, ACCELERATION_AVAILABLE, IS_MAC, IS_ARM_MAC
except ImportError:
    # Try absolute import (when run directly)
    from main import run_inventory_analysis
    from oneapi_accelerator import ONEAPI_AVAILABLE, APPLE_ACCELERATE_AVAILABLE, ACCELERATION_AVAILABLE, IS_MAC, IS_ARM_MAC
try:
    # Try relative import (when used as a package)
    from .visualization import visualize_results
except ImportError:
    # Try absolute import (when run directly)
    from visualization import visualize_results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BibliophileSP - SP-API powered inventory sync and analysis tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--use-spapi",
        action="store_true",
        help="Use SP-API to fetch real data (requires valid credentials)"
    )
    
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization generation"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output files"
    )
    
    # Acceleration options
    acceleration_group = parser.add_argument_group('Acceleration Options')
    
    # Common option for all platforms
    acceleration_group.add_argument(
        "--no-acceleration",
        action="store_true",
        help="Disable all acceleration features"
    )
    
    # Intel oneAPI specific option
    if ONEAPI_AVAILABLE:
        acceleration_group.add_argument(
            "--no-oneapi",
            action="store_true",
            help="Disable Intel oneAPI acceleration specifically (available on this system)"
        )
    else:
        acceleration_group.add_argument(
            "--force-oneapi",
            action="store_true",
            help="Try to use Intel oneAPI acceleration even though required modules were not detected"
        )
    
    # Apple Silicon/Accelerate specific option
    if IS_MAC:
        if APPLE_ACCELERATE_AVAILABLE:
            platform_note = "Apple Silicon M1/M2/M3" if IS_ARM_MAC else "macOS with Accelerate"
            acceleration_group.add_argument(
                "--no-apple-accelerate",
                action="store_true",
                help=f"Disable Apple Accelerate framework acceleration ({platform_note})"
            )
        else:
            acceleration_group.add_argument(
                "--install-numpy-accelerate",
                action="store_true",
                help="Show instructions for installing NumPy with Apple Accelerate framework support"
            )
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the analysis
    print(f"🚀 Starting BibliophileSP analysis...")
    
    if args.use_spapi:
        print("📡 Using SP-API to fetch real data...")
    else:
        print("🧪 Using dummy data (sandbox mode)...")
    
    # Show platform information
    if IS_MAC:
        platform_str = f"macOS with {'Apple Silicon (M1/M2/M3)' if IS_ARM_MAC else 'Intel processor'}"
        print(f"🍎 Running on {platform_str}")
        if APPLE_ACCELERATE_AVAILABLE:
            print("⚡ Apple Accelerate framework is available for NumPy operations.")
    elif ONEAPI_AVAILABLE:
        print("⚡ Intel oneAPI acceleration is available.")
        
    # Show Apple Accelerate installation instructions if requested
    if hasattr(args, 'install_numpy_accelerate') and args.install_numpy_accelerate:
        print("\n📋 Instructions for installing NumPy with Apple Accelerate support:")
        print("1. Install Homebrew if not already installed:")
        print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        print("2. Install Python dependencies:")
        print("   brew install openblas lapack")
        print("3. Install NumPy with Accelerate framework:")
        print("   pip install numpy --no-binary numpy")
        print("4. Verify installation:")
        print("   python -c \"import numpy; print(numpy.__config__.show())\"")
        return 0
    
    # Determine acceleration setting
    use_acceleration = True
    
    # Handle global acceleration flag
    if hasattr(args, 'no_acceleration') and args.no_acceleration:
        use_acceleration = False
        print("⚠️ All acceleration disabled by user.")
    
    # Handle oneAPI specific flags
    elif hasattr(args, 'no_oneapi') and args.no_oneapi and ONEAPI_AVAILABLE:
        print("⚠️ Intel oneAPI acceleration specifically disabled by user.")
        if IS_MAC and APPLE_ACCELERATE_AVAILABLE:
            print("ℹ️ Still using Apple Accelerate framework for NumPy operations.")
    elif hasattr(args, 'force_oneapi') and args.force_oneapi:
        print("⚠️ Forcing Intel oneAPI acceleration (may fail if modules not available).")
    
    # Handle Apple specific flags
    elif IS_MAC and hasattr(args, 'no_apple_accelerate') and args.no_apple_accelerate:
        print("⚠️ Apple Accelerate framework specifically disabled by user.")
        if ONEAPI_AVAILABLE:
            print("ℹ️ Still using Intel oneAPI acceleration for compatible operations.")
    
    # Default acceleration message
    elif ACCELERATION_AVAILABLE:
        accl_type = []
        if ONEAPI_AVAILABLE:
            accl_type.append("Intel oneAPI")
        if APPLE_ACCELERATE_AVAILABLE:
            accl_type.append("Apple Accelerate")
        print(f"⚡ Using acceleration: {', '.join(accl_type)}")
    else:
        print("ℹ️ No acceleration available. Using standard implementations.")
    
    try:
        # Run the analysis
        results_df = run_inventory_analysis(
            use_spapi=args.use_spapi,
            use_acceleration=use_acceleration
        )
        
        # Generate visualizations if requested
        if not args.no_visualize:
            print("\n📈 Generating visualizations...")
            try:
                visualize_results(f"{args.output_dir}/inventory_analysis.csv")
                print("✅ Visualizations complete. Check the output directory for the generated plots.")
            except Exception as e:
                print(f"⚠️ Error generating visualizations: {str(e)}")
                print("⚠️ Skipping visualization step.")
        
        print("\n✨ Analysis complete! ✨")
        print(f"📊 Results saved to {args.output_dir}/inventory_analysis.csv")
        
        if not args.no_visualize:
            print(f"📈 Visualizations saved to {args.output_dir}/")
        
        return 0
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
