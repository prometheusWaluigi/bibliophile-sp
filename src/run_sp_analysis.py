#!/usr/bin/env python
"""
Command-line script to analyze SP-API data files using oneAPI acceleration.

This script extracts, processes, and analyzes data from SP-API report files,
utilizing Intel oneAPI acceleration when available.
"""

import os
import argparse
import logging
import sys
from pathlib import Path

from .sp_data_analysis import run_sp_data_analysis
from .oneapi_accelerator import ONEAPI_AVAILABLE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def main():
    """
    Parse command-line arguments and run SP data analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze SP-API data with oneAPI acceleration"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing SP-API data files (default: data)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save analysis results (default: output)"
    )
    
    parser.add_argument(
        "--disable-acceleration",
        action="store_true",
        help="Disable oneAPI acceleration even if available"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory '{data_dir}' does not exist")
        return 1
    
    # Check for data files
    data_files = list(data_dir.glob("*"))
    if not data_files:
        logger.error(f"No files found in data directory '{data_dir}'")
        return 1
    
    logger.info(f"Found {len(data_files)} files in data directory")
    
    # Determine if oneAPI acceleration should be used
    use_acceleration = ONEAPI_AVAILABLE and not args.disable_acceleration
    if use_acceleration:
        logger.info("oneAPI acceleration is ENABLED")
    else:
        if args.disable_acceleration:
            logger.info("oneAPI acceleration is manually DISABLED")
        else:
            logger.info("oneAPI acceleration is NOT AVAILABLE")
    
    try:
        # Run the analysis
        logger.info(f"Starting SP data analysis...")
        results = run_sp_data_analysis(
            data_dir=str(data_dir),
            output_dir=args.output_dir,
            use_acceleration=use_acceleration
        )
        
        # Print summary of results
        logger.info("Analysis completed successfully")
        
        if results.get("basic_analysis") is not None:
            n_records = len(results["basic_analysis"])
            logger.info(f"Analyzed {n_records} inventory items")
            
            # Count flagged items
            flagged = results["basic_analysis"]["Flag"].value_counts().get("⚠️", 0)
            if flagged > 0:
                logger.info(f"Found {flagged} items requiring attention ({flagged/n_records:.1%})")
        
        if results.get("clustering") is not None:
            clusters = results["clustering"]["Cluster"].nunique()
            logger.info(f"Identified {clusters} distinct inventory clusters")
        
        # Provide path to results
        output_dir = Path(args.output_dir)
        logger.info(f"Results saved to {output_dir.absolute()}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
