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
    from .sp_data_analysis import run_sp_data_analysis
except ImportError:
    # Try absolute import (when run directly)
    from main import run_inventory_analysis
    from oneapi_accelerator import ONEAPI_AVAILABLE, APPLE_ACCELERATE_AVAILABLE, ACCELERATION_AVAILABLE, IS_MAC, IS_ARM_MAC
    from sp_data_analysis import run_sp_data_analysis
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
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Standard inventory analysis command
    inventory_parser = subparsers.add_parser(
        'inventory', 
        help='Run standard inventory analysis with optional SP-API integration'
    )
    inventory_parser.add_argument(
        "--use-spapi",
        action="store_true",
        help="Use SP-API to fetch real data (requires valid credentials)"
    )
    inventory_parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization generation"
    )
    inventory_parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output files"
    )
    
    # SP data analysis command
    sp_data_parser = subparsers.add_parser(
        'sp-data', 
        help='Analyze extracted SP-API report data files'
    )
    sp_data_parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing SP-API data files (default: data)"
    )
    sp_data_parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save analysis results (default: output)"
    )
    sp_data_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Common argument for output directory (used when no subcommand is specified)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output files"
    )
    
    # Common argument for visualization (used when no subcommand is specified)
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization generation"
    )
    
    # Common argument for SP-API (used when no subcommand is specified)
    parser.add_argument(
        "--use-spapi",
        action="store_true",
        help="Use SP-API to fetch real data (requires valid credentials)"
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
    
    # Determine which command to run
    if hasattr(args, 'command') and args.command == 'sp-data':
        return run_sp_data_command(args, use_acceleration)
    elif hasattr(args, 'command') and args.command == 'inventory':
        return run_inventory_command(args, use_acceleration)
    else:
        # Default to inventory analysis for backward compatibility
        print("No command specified, defaulting to standard inventory analysis.")
        return run_inventory_command(args, use_acceleration)


def run_inventory_command(args, use_acceleration):
    """Run the standard inventory analysis command."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the analysis
    print(f"🚀 Starting BibliophileSP inventory analysis...")
    
    if args.use_spapi:
        print("📡 Using SP-API to fetch real data...")
    else:
        print("🧪 Using dummy data (sandbox mode)...")
    
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


def run_sp_data_command(args, use_acceleration):
    """Run the SP data analysis command."""
    # Validate the data directory
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory '{args.data_dir}' does not exist")
        return 1
    
    data_files = [f for f in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, f))]
    if not data_files:
        print(f"❌ No files found in data directory '{args.data_dir}'")
        return 1
    
    print(f"🚀 Starting SP-API data analysis on {len(data_files)} files...")
    print(f"📂 Data directory: {args.data_dir}")
    print(f"📊 Output directory: {args.output_dir}")
    
    try:
        # Run the SP data analysis
        results = run_sp_data_analysis(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            use_acceleration=use_acceleration
        )
        
        print("\n✨ SP data analysis complete! ✨")
        
        # Summarize results
        if results.get("basic_analysis") is not None:
            n_records = len(results["basic_analysis"])
            print(f"📊 Analyzed {n_records} inventory items")
            
            # Count flagged items
            if "Flag" in results["basic_analysis"].columns:
                flagged = results["basic_analysis"]["Flag"].value_counts().get("⚠️", 0)
                if flagged > 0:
                    print(f"⚠️ Found {flagged} items requiring attention ({flagged/n_records:.1%})")
        
        if results.get("clustering") is not None:
            clusters = results["clustering"]["Cluster"].nunique()
            print(f"🔍 Identified {clusters} distinct inventory clusters")
        
        print(f"📊 Results saved to {args.output_dir}/")
        
        # List output files
        output_files = [f for f in os.listdir(args.output_dir) if os.path.isfile(os.path.join(args.output_dir, f)) and f.endswith(".csv")]
        for file in output_files:
            print(f"  - {file}")
        
        # Check for visualizations
        viz_dir = os.path.join(args.output_dir, "visualizations")
        if os.path.exists(viz_dir):
            viz_files = [f for f in os.listdir(viz_dir) if os.path.isfile(os.path.join(viz_dir, f)) and f.endswith(".png")]
            if viz_files:
                print(f"📈 {len(viz_files)} visualizations generated in {viz_dir}/")
        
        return 0
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
