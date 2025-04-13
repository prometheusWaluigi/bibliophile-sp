import os
import requests
import pandas as pd
from datetime import datetime
try:
    # Try relative import (when used as a package)
    from .analysis import run_analysis, InventoryAnalyzer
except ImportError:
    # Try absolute import (when run directly)
    from analysis import run_analysis, InventoryAnalyzer
try:
    # Try relative import (when used as a package)
    from .oneapi_accelerator import ONEAPI_AVAILABLE, APPLE_ACCELERATE_AVAILABLE, ACCELERATION_AVAILABLE, enable_acceleration, disable_acceleration
except ImportError:
    # Try absolute import (when run directly)
    from oneapi_accelerator import ONEAPI_AVAILABLE, APPLE_ACCELERATE_AVAILABLE, ACCELERATION_AVAILABLE, enable_acceleration, disable_acceleration
try:
    # Try relative import (when used as a package)
    from .spapi import get_client
    from .visualization import visualize_results
except ImportError:
    # Try absolute import (when run directly)
    from spapi import get_client
    from visualization import visualize_results

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file.")
except ImportError:
    print("⚠️ python-dotenv not installed. Using environment variables as is.")

# Amazon SP-API credentials
CLIENT_ID = os.getenv("LWA_CLIENT_ID")
CLIENT_SECRET = os.getenv("LWA_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("LWA_REFRESH_TOKEN")
LWA_ENDPOINT = "https://api.amazon.com/auth/o2/token"


def get_access_token():
    """
    Get an access token from Amazon's Login with Amazon service.
    
    Returns:
        str: The access token.
    """
    print("🔑 Fetching LWA Access Token...")
    
    # Check if credentials are available
    if not all([CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN]):
        print("⚠️ Missing SP-API credentials. Using dummy token for sandbox mode.")
        return "dummy_token"
    
    try:
        payload = {
            'grant_type': 'refresh_token',
            'refresh_token': REFRESH_TOKEN,
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET
        }
        response = requests.post(LWA_ENDPOINT, data=payload)
        response.raise_for_status()
        token = response.json()
        print("✅ Token received.")
        return token['access_token']
    except Exception as e:
        print(f"⚠️ Error fetching token: {str(e)}")
        print("⚠️ Using dummy token for sandbox mode.")
        return "dummy_token"


def run_inventory_analysis(use_spapi=False, use_acceleration=True):
    """
    Run the inventory analysis and save the results to a CSV file.
    
    Args:
        use_spapi (bool, optional): Whether to use the SP-API client to fetch data.
                                   Defaults to False (use dummy data).
        use_acceleration (bool, optional): Whether to use oneAPI acceleration if available.
                                         Defaults to True.
    """
    print("📊 Running inventory analysis...")
    
    # Check if oneAPI acceleration is available
    if ONEAPI_AVAILABLE and use_acceleration:
        print("⚡ oneAPI acceleration is enabled.")
        enable_acceleration()
    else:
        if use_acceleration and not ONEAPI_AVAILABLE:
            print("⚠️ oneAPI acceleration requested but not available. Using standard implementation.")
        elif not use_acceleration:
            print("ℹ️ oneAPI acceleration is disabled.")
    
    try:
        if use_spapi:
            # Get SP-API client
            token = get_access_token()
            client = get_client(token)
            
            # Fetch data from SP-API
            inventory_data = client.get_inventory_summary()
            sales_data = client.get_sales_history()
            
            # Run analysis with real data
            analyzer = InventoryAnalyzer(sales_data, inventory_data, use_acceleration=use_acceleration)
            results_df = analyzer.analyze()
        else:
            # Use dummy data
            results_df = run_analysis(use_acceleration=use_acceleration)
    finally:
        # Disable acceleration if it was enabled
        if ONEAPI_AVAILABLE and use_acceleration:
            disable_acceleration()
    
    # Save the results to a CSV file
    output_path = "output/inventory_analysis.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"✅ Analysis complete. Results written to {output_path}")
    
    # Display a summary of the results
    stale_count = results_df[results_df['Flag'] == '⚠️'].shape[0]
    good_count = results_df[results_df['Flag'] == '✅'].shape[0]
    
    print(f"\nSummary:")
    print(f"📚 Total items analyzed: {len(results_df)}")
    print(f"⚠️ Items flagged for attention: {stale_count}")
    print(f"✅ Good sellers: {good_count}")
    
    return results_df


if __name__ == "__main__":
    import argparse
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="BibliophileSP - SP-API powered inventory sync and analysis tool")
    parser.add_argument("--use-spapi", action="store_true", help="Use SP-API to fetch real data (requires valid credentials)")
    parser.add_argument("--no-acceleration", action="store_true", help="Disable oneAPI acceleration")
    parser.add_argument("--no-visualize", action="store_true", help="Skip visualization generation")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Print oneAPI availability
    if ONEAPI_AVAILABLE:
        print("⚡ oneAPI acceleration is available.")
    else:
        print("ℹ️ oneAPI acceleration is not available.")
    
    # Run the analysis
    results_df = run_inventory_analysis(
        use_spapi=args.use_spapi,
        use_acceleration=not args.no_acceleration
    )
    
    # Generate visualizations
    if not args.no_visualize:
        try:
            print("\n📈 Generating visualizations...")
            visualize_results()
            print("✅ Visualizations complete. Check the output directory for the generated plots.")
        except Exception as e:
            print(f"⚠️ Error generating visualizations: {str(e)}")
            print("⚠️ Skipping visualization step.")
