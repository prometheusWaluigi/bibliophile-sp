"""
SP Data Analysis module for BibliophileSP.

This module uses the SP-API data extracted from reports to perform accelerated analysis.
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

from analysis import InventoryAnalyzer
from oneapi_accelerator import AccelerationContext, ONEAPI_AVAILABLE, accelerated_kmeans

# Set up logging
logger = logging.getLogger(__name__)


class SPDataLoader:
    """
    Loads and processes SP-API data files.
    """
    
    def __init__(self, data_dir="data", output_dir="output"):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Directory containing the SP-API data files
            output_dir (str): Directory to save analysis results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.sales_data = None
        self.inventory_data = None
        self.performance_data = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
    
    def load_all_data(self):
        """
        Load all available data files.
        
        Returns:
            self: For method chaining
        """
        # Load inventory data first since sales data needs it for SKU mapping
        self.load_inventory_data()
        self.load_sales_data()
        self.load_performance_data()
        return self
    
    def load_sales_data(self):
        """
        Load sales data from relevant files.
        
        Returns:
            pd.DataFrame: Processed sales data
        """
        # Try to load from cleaned sales data first
        sales_file = self.data_dir / "cleaned/cleaned_SalesandTrafficByDate.csv"
        
        if sales_file.exists():
            logger.info(f"Loading sales data from {sales_file}")
            # Read with fixed column names
            column_names = [
                'Date', 'Ordered Product Sales', 'Ordered Product Sales - B2B',
                'Units Ordered', 'Units Ordered - B2B', 'Total Order Items',
                'Total Order Items - B2B', 'Average Sales per Order Item',
                'Average Sales per Order Item - B2B', 'Average Units per Order Item',
                'Average Units per Order Item - B2B', 'Average Selling Price',
                'Average Selling Price - B2B', 'Sessions - Total', 'Sessions - Total - B2B',
                'Order Item Session Percentage', 'Order Item Session Percentage - B2B',
                'Average Offer Count'
            ]
            
            # Read data with proper delimiter and handle quoted values
            sales_df = pd.read_csv(sales_file, names=column_names, skiprows=1, 
                                 encoding='utf-8', quoting=3)  # QUOTE_NONE
            
            # Process the sales data
            if "Date" in sales_df.columns:
                sales_df["Date"] = pd.to_datetime(sales_df["Date"])
            
            # Map sales data columns with proper column names and types
            sales_df['OrderedProductSales'] = pd.to_numeric(sales_df['Ordered Product Sales'].astype(str).str.replace('"', ''), errors='coerce')
            sales_df['Sales'] = sales_df['OrderedProductSales']  # Map directly to Sales
            sales_df['Quantity'] = pd.to_numeric(sales_df['Units Ordered'].astype(str).str.replace('"', ''), errors='coerce').fillna(0).clip(lower=0)
            sales_df['quantity'] = sales_df['Quantity']  # Add lowercase version for analysis
            
            # Handle Date column
            if 'Date' in sales_df.columns:
                sales_df['Date'] = pd.to_datetime(sales_df['Date'])
            else:
                sales_df['Date'] = pd.Timestamp.now()
            
            # Create unique SKU for each row with timestamp
            sales_df['SKU'] = sales_df.apply(lambda row: f"SALE_{pd.Timestamp(row['Date']).strftime('%Y%m%d')}_{row.name}", axis=1)
            
            # Ensure required columns exist with proper values
            required_cols = {'Date', 'SKU', 'quantity', 'Sales'}
            for col in required_cols:
                if col not in sales_df.columns:
                    sales_df[col] = 0 if col in ['quantity', 'Sales'] else None
            
            # Fill NaN values
            sales_df['Quantity'] = sales_df['Quantity'].fillna(0)
            sales_df['Sales'] = sales_df['Sales'].fillna(0)
            
            self.sales_data = sales_df
            logger.info(f"Loaded {len(sales_df)} sales records")
            
        else:
            # Try alternative sales files
            alt_files = [
                "DetailPageSalesandTrafficByDate.csv",
                "SalesandOrdersbyMonth.csv"
            ]
            
            for alt_file in alt_files:
                file_path = self.data_dir / alt_file
                if file_path.exists():
                    logger.info(f"Loading sales data from alternative file: {alt_file}")
                    self.sales_data = pd.read_csv(file_path)
                    
                    # Process date column if it exists
                    if "Date" in self.sales_data.columns:
                        self.sales_data["Date"] = pd.to_datetime(self.sales_data["Date"])
                    
                    break
        
        if self.sales_data is None:
            logger.warning("No sales data files found. Analysis will be limited.")
            # Create empty DataFrame with expected columns
            self.sales_data = pd.DataFrame(columns=["Date", "SKU", "quantity", "Sales"])
        
        return self.sales_data
    
    def load_inventory_data(self):
        """
        Load inventory data from relevant files.
        
        Returns:
            pd.DataFrame: Processed inventory data
        """
        # Try to load from cleaned inventory report files
        inventory_files = [
            "cleaned/cleaned_Inventory+Report+03-22-2025.txt",
            "cleaned/cleaned_Open+Listings+Report+03-22-2025.txt",
            "cleaned/cleaned_All+Listings+Report+03-22-2025.txt"
        ]
        
        for file_name in inventory_files:
            file_path = self.data_dir / file_name
            if file_path.exists():
                logger.info(f"Loading inventory data from {file_path}")
                
                try:
                    # Read data with fixed column names
                    chunks = []
                    for chunk in pd.read_csv(file_path, sep='\t', encoding='utf-8', chunksize=10000, 
                                           header=0,  # Use first row as header
                                           dtype={
                                               'SKU': str,
                                               'asin': str,
                                               'Price': float,
                                               'Quantity': float
                                           }):
                        # Convert numeric columns immediately and map to standard names
                        chunk['Price'] = pd.to_numeric(chunk['Price'], errors='coerce')
                        chunk['quantity'] = pd.to_numeric(chunk['Quantity'], errors='coerce').fillna(0).clip(lower=0)
                        chunk['SKU'] = chunk['SKU'].astype(str)
                        chunk['ASIN'] = chunk['asin']
                        
                        chunks.append(chunk)
                    inventory_df = pd.concat(chunks, ignore_index=True)
                    
                    # Convert numeric columns and ensure proper data types
                    inventory_df['Price'] = pd.to_numeric(inventory_df['Price'], errors='coerce')
                    inventory_df['SKU'] = inventory_df['SKU'].astype(str)
                    
                    # Map columns to standard format expected by analysis
                    column_mappings = {
                        'seller-sku': 'SKU',
                        'sku': 'SKU',
                        'asin1': 'ASIN',
                        'item-name': 'Title',
                        'item_name': 'Title',
                        'item-description': 'Title',
                        'product-name': 'Title',
                        'name': 'Title',
                        'title': 'Title',
                        'price': 'Price',
                        'price-amount': 'Price',
                        'quantity': 'quantity',
                        'quantity-available': 'quantity',
                        'condition': 'Condition',
                        'fulfillment-channel': 'FulfillmentChannel',
                        'open-date': 'OpenDate'
                    }
                    
                    # Ensure Title column exists
                    if 'Title' not in inventory_df.columns:
                        # Try to find any column containing product names
                        title_cols = [col for col in inventory_df.columns if any(x in col.lower() for x in ['name', 'title', 'description'])]
                        if title_cols:
                            inventory_df['Title'] = inventory_df[title_cols[0]]
                        else:
                            inventory_df['Title'] = 'Unknown Title'
                    
                    # Apply mappings for columns that exist
                    for old_col, new_col in column_mappings.items():
                        if old_col in inventory_df.columns:
                            inventory_df[new_col] = inventory_df[old_col]
                    
                    # Ensure required columns exist and are properly formatted
                    required_cols = {'SKU', 'ASIN', 'Price', 'quantity'}
                    for col in required_cols:
                        if col not in inventory_df.columns:
                            # Try to find matching column with different case
                            matching_cols = [c for c in inventory_df.columns if c.lower() == col.lower()]
                            if matching_cols:
                                inventory_df[col] = inventory_df[matching_cols[0]]
                            else:
                                inventory_df[col] = 0 if col in ['Price', 'quantity'] else ''
                    
                    # Convert numeric columns
                    inventory_df['Price'] = pd.to_numeric(inventory_df['Price'], errors='coerce').fillna(0)
                            
                    # Ensure SKU column exists
                    if 'SKU' not in inventory_df.columns and 'ASIN' in inventory_df.columns:
                        inventory_df['SKU'] = inventory_df['ASIN']
                    
                    # Convert price to numeric if it exists
                    if 'Price' in inventory_df.columns:
                        inventory_df['Price'] = pd.to_numeric(inventory_df['Price'], errors='coerce')
                    
                    # Check for image availability
                    if "has_image" in inventory_df.columns:
                        inventory_df["Has_Image"] = inventory_df["has_image"].astype(bool)
                    elif "main-image-url" in inventory_df.columns:
                        inventory_df["Has_Image"] = ~inventory_df["main-image-url"].isna()
                    else:
                        inventory_df["Has_Image"] = False
                    
                    # Check for ISBN
                    isbn_cols = ["isbn", "external_product_id", "product-id"]
                    found_isbn = False
                    for col in isbn_cols:
                        if col in inventory_df.columns:
                            inventory_df["ISBN"] = inventory_df[col]
                            found_isbn = True
                            break
                    if not found_isbn:
                        inventory_df["ISBN"] = ""
                    
                    self.inventory_data = inventory_df
                    logger.info(f"Loaded {len(inventory_df)} inventory records")
                    break
                
                except Exception as e:
                    logger.error(f"Error reading inventory file {file_path}: {str(e)}")
        
        if self.inventory_data is None:
            logger.warning("No inventory data files found. Analysis will be limited.")
            # Create empty DataFrame with expected columns
            self.inventory_data = pd.DataFrame(columns=["SKU", "Title", "Price", "Has_Image", "ISBN", "quantity"])
        
        return self.inventory_data
    
    def load_performance_data(self):
        """
        Load seller performance data.
        
        Returns:
            pd.DataFrame: Processed performance data
        """
        # Try different cleaned performance files
        performance_files = [
            "cleaned/cleaned_SellerPerformance.csv",
            "cleaned/cleaned_UnitOnTimeDelivery_1282029020169.csv",
            "cleaned/cleaned_OrderDefects_1282032020169.csv"
        ]
        
        for file_name in performance_files:
            file_path = self.data_dir / file_name
            if file_path.exists():
                logger.info(f"Loading performance data from {file_path}")
                
                try:
                    performance_df = pd.read_csv(file_path)
                    self.performance_data = performance_df
                    logger.info(f"Loaded {len(performance_df)} performance records")
                    break
                except Exception as e:
                    logger.error(f"Error reading performance file {file_path}: {str(e)}")
        
        if self.performance_data is None:
            logger.warning("No performance data files found")
            # Create empty DataFrame
            self.performance_data = pd.DataFrame()
        
        return self.performance_data


class SPDataAnalyzer:
    """
    Analyzes SP-API data using accelerated methods.
    """
    
    def __init__(self, data_loader=None, use_acceleration=True):
        """
        Initialize the SP data analyzer.
        
        Args:
            data_loader (SPDataLoader, optional): Data loader with SP-API data
            use_acceleration (bool, optional): Whether to use oneAPI acceleration
        """
        self.data_loader = data_loader or SPDataLoader()
        self.use_acceleration = use_acceleration and ONEAPI_AVAILABLE
        self.inventory_analyzer = None
        
        if self.use_acceleration:
            logger.info("Using oneAPI acceleration for SP data analysis")
        else:
            logger.info("oneAPI acceleration is disabled or not available")
    
    def prepare_analyzer(self):
        """
        Prepare the inventory analyzer with SP data.
        
        Returns:
            InventoryAnalyzer: Configured analyzer
        """
        if self.data_loader.sales_data is None or self.data_loader.inventory_data is None:
            logger.info("Loading data for analysis")
            self.data_loader.load_all_data()
        
        self.inventory_analyzer = InventoryAnalyzer(
            sales_data=self.data_loader.sales_data,
            inventory_data=self.data_loader.inventory_data,
            use_acceleration=self.use_acceleration
        )
        
        return self.inventory_analyzer
    
    def run_full_analysis(self):
        """
        Run comprehensive analysis on SP data.
        
        Returns:
            dict: Analysis results
        """
        results = {}
        
        # Ensure analyzer is prepared
        if self.inventory_analyzer is None:
            self.prepare_analyzer()
        
        # Run basic inventory analysis
        logger.info("Running basic inventory analysis")
        if self.inventory_analyzer is not None:
            try:
                basic_analysis = self.inventory_analyzer.analyze()
                results["basic_analysis"] = basic_analysis
                
                # Create timestamped output directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = self.data_loader.output_dir / f"analysis_{timestamp}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Save main analysis with proper column separation
                main_output = output_dir / "inventory_analysis.csv"
                # Update column references for filtering
                days_since_last_sale = 'Days Since Last Sale'
                sales_last_90d = 'Sales Last 90d'
                sales_last_30d = 'Sales Last 30d'
                
                # Rename columns to ensure proper spacing
                basic_analysis.columns = [col.replace(' ', '_') for col in basic_analysis.columns]
                # Add spaces between columns for CSV output
                basic_analysis.columns = [','.join(col.split('_')) for col in basic_analysis.columns]
                
                # Save main analysis
                basic_analysis.to_csv(main_output, index=False, sep=',')
                
                # Save detailed reports with proper filtering
                # Stale detection criteria from README
                stale_items = basic_analysis[
                    (basic_analysis[days_since_last_sale] > 120) |  # No sales in 4 months
                    ((basic_analysis[sales_last_90d] == 0) & (basic_analysis['quantity'] > 0)) |  # Stock but no recent sales
                    (basic_analysis[sales_last_90d] * 4 < 3)  # Less than 3 sales per year
                ].copy()
                stale_items.to_csv(output_dir / "stale_items.csv", index=False, sep=',')
                logger.info(f"Found {len(stale_items)} stale items")
                
                # Bad metadata criteria from README
                bad_metadata = basic_analysis[
                    (basic_analysis['Title'].str.len() < 5) |  # Title too short
                    (basic_analysis['ISBN'].isna() | (basic_analysis['ISBN'] == '')) |  # No ISBN
                    (~basic_analysis['Has_Image']) |  # No cover image
                    (basic_analysis['Price'] < 2.0)  # Price too low
                ].copy()
                bad_metadata.to_csv(output_dir / "bad_metadata.csv", index=False, sep=',')
                logger.info(f"Found {len(bad_metadata)} items with bad metadata")
                
                good_sellers = basic_analysis[
                    (basic_analysis[sales_last_30d] > 0) &  # Recent sales
                    (basic_analysis[days_since_last_sale] <= 30) &  # Active in last month
                    (basic_analysis['Title'].notna()) &  # Has proper metadata
                    (basic_analysis['Title'] != '') &
                    (basic_analysis['Title'] != 'Unknown Title') &
                    (basic_analysis['Has_Image']) &  # Has image
                    (basic_analysis['quantity'] > 0)  # In stock
                ].copy()
                good_sellers.to_csv(output_dir / "good_sellers.csv", index=False, sep=',')
                logger.info(f"Found {len(good_sellers)} good sellers")
                
                sales_velocity = basic_analysis[
                    ['SKU', 'Title', sales_last_30d, sales_last_90d, days_since_last_sale]
                ].sort_values(sales_last_90d, ascending=False)
                sales_velocity.to_csv(output_dir / "sales_velocity.csv", index=False, sep=',')
                
                # Calculate heuristic score (0-1) based on README criteria
                basic_analysis['Heuristic_Score'] = 1.0
                # Deduct for stale items
                basic_analysis.loc[basic_analysis[days_since_last_sale] > 120, 'Heuristic_Score'] -= 0.25
                basic_analysis.loc[basic_analysis[sales_last_90d] * 4 < 3, 'Heuristic_Score'] -= 0.25
                # Deduct for bad metadata
                basic_analysis.loc[basic_analysis['Title'].str.len() < 5, 'Heuristic_Score'] -= 0.125
                basic_analysis.loc[basic_analysis['ISBN'].isna() | (basic_analysis['ISBN'] == ''), 'Heuristic_Score'] -= 0.125
                basic_analysis.loc[~basic_analysis['Has_Image'], 'Heuristic_Score'] -= 0.125
                basic_analysis.loc[basic_analysis['Price'] < 2.0, 'Heuristic_Score'] -= 0.125
                # Ensure score stays within 0-1 range
                basic_analysis['Heuristic_Score'] = basic_analysis['Heuristic_Score'].clip(0, 1)
                
                # Save metadata quality analysis
                metadata_quality = basic_analysis[
                    ['SKU', 'Title', 'ISBN', 'Has_Image', 'Price', 'Heuristic_Score']
                ].sort_values('Heuristic_Score', ascending=False)
                metadata_quality.to_csv(output_dir / "metadata_quality.csv", index=False, sep=',')
                
                revenue_last_30d = 'Revenue_Last_30d'
                revenue_last_90d = 'Revenue_Last_90d'
                
                if revenue_last_30d in basic_analysis.columns:
                    revenue_analysis = basic_analysis[
                        ['SKU', 'Title', revenue_last_30d, revenue_last_90d, 
                         sales_last_30d, sales_last_90d]
                    ].sort_values(revenue_last_90d, ascending=False)
                    revenue_analysis.to_csv(output_dir / "revenue_analysis.csv", index=False, sep=',')
                
                logger.info(f"\nAnalysis complete. Results written to {output_dir}/")
                logger.info("\nDetailed reports generated:")
                logger.info(f"📊 Main Analysis: {main_output}")
                logger.info(f"⚠️ Stale Items: {output_dir}/stale_items.csv")
                logger.info(f"🔍 Bad Metadata: {output_dir}/bad_metadata.csv")
                logger.info(f"✨ Good Sellers: {output_dir}/good_sellers.csv")
                logger.info(f"📈 Sales Velocity: {output_dir}/sales_velocity.csv")
                if 'Metadata_Quality_Score' in basic_analysis.columns:
                    logger.info(f"📝 Metadata Quality: {output_dir}/metadata_quality.csv")
                if revenue_last_30d in basic_analysis.columns:
                    logger.info(f"💰 Revenue Analysis: {output_dir}/revenue_analysis.csv")
            except Exception as e:
                logger.error(f"Error in basic analysis: {str(e)}")
        else:
            logger.error("Inventory analyzer is None, cannot perform analysis")
        
        # Run clustering if enough data
        if (self.inventory_analyzer is not None and 
            self.data_loader.inventory_data is not None and 
            len(self.data_loader.inventory_data) >= 5):
            
            logger.info("Running inventory clustering")
            try:
                # Use acceleration context for clustering
                with AccelerationContext():
                    # Determine cluster count based on data size
                    n_clusters = min(3, max(2, len(self.data_loader.inventory_data) // 10))
                    clustered_data = self.inventory_analyzer.cluster_inventory(n_clusters=n_clusters)
                
                results["clustering"] = clustered_data
                
                # Save results
                output_path = self.data_loader.output_dir / "inventory_clusters.csv"
                clustered_data.to_csv(output_path, index=False, sep=',')
                logger.info(f"Saved clustering analysis to {output_path}")
                
                # Generate cluster visualization
                self._generate_cluster_visualization(clustered_data)
                
            except Exception as e:
                logger.error(f"Error in clustering: {str(e)}")
        
        # Analyze performance data if available
        if self.data_loader.performance_data is not None and not self.data_loader.performance_data.empty:
            logger.info("Analyzing performance data")
            try:
                performance_analysis = self._analyze_performance_data()
                results["performance"] = performance_analysis
                
                # Save results
                output_path = self.data_loader.output_dir / "performance_analysis.csv"
                if isinstance(performance_analysis, pd.DataFrame):
                    performance_analysis.to_csv(output_path, index=False, sep=',')
                    logger.info(f"Saved performance analysis to {output_path}")
            except Exception as e:
                logger.error(f"Error in performance analysis: {str(e)}")
        
        return results
    
    def _generate_cluster_visualization(self, clustered_data):
        """
        Generate visualization for cluster analysis.
        
        Args:
            clustered_data (pd.DataFrame): Data with cluster assignments
        """
        try:
            # Create folder for visualizations
            viz_dir = self.data_loader.output_dir / "visualizations"
            if not os.path.exists(viz_dir):
                os.makedirs(viz_dir)
            
            # Extract available numerical features
            plot_data = clustered_data[['Price', 'Cluster', 'Cluster_Description']].copy()
            
            # Replace missing values
            plot_data = plot_data.fillna(0)
            
            # Generate price distribution by cluster
            plt.figure(figsize=(10, 6))
            for cluster in plot_data['Cluster'].unique():
                cluster_data = plot_data[plot_data['Cluster'] == cluster]
                plt.hist(
                    cluster_data['Price'],
                    bins=30,
                    alpha=0.5,
                    label=f"Cluster {cluster}: {cluster_data['Cluster_Description'].iloc[0]}"
                )
            
            plt.title('Price Distribution by Cluster')
            plt.xlabel('Price ($)')
            plt.ylabel('Number of Items')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save figure
            output_path = viz_dir / "price_distribution_clusters.png"
            plt.savefig(output_path)
            logger.info(f"Saved cluster visualization to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating cluster visualization: {str(e)}")
    
    def _analyze_performance_data(self):
        """
        Analyze seller performance data.
        
        Returns:
            pd.DataFrame: Performance analysis results
        """
        perf_data = self.data_loader.performance_data
        
        # Basic statistics and aggregations
        if perf_data is None or perf_data.empty:
            return pd.DataFrame()
        
        # Look for common performance metrics
        metrics = []
        
        # Check for on-time delivery
        delivery_cols = ['on_time_delivery_rate', 'late_shipment_rate', 'LateShipmentRate']
        for col in delivery_cols:
            if col in perf_data.columns:
                metrics.append(col)
        
        # Check for defect rates
        defect_cols = ['order_defect_rate', 'OrderDefectRate', 'defect_rate']
        for col in defect_cols:
            if col in perf_data.columns:
                metrics.append(col)
        
        # If no standard metrics found, use all numeric columns
        if not metrics:
            metrics = perf_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Perform basic analysis on available metrics
        if metrics:
            # Compute basic statistics
            analysis = perf_data[metrics].describe()
            
            # Add trend analysis if time dimension is available
            time_cols = ['Date', 'date', 'period', 'week', 'month']
            time_col = None
            
            for col in time_cols:
                if col in perf_data.columns:
                    time_col = col
                    break
            
            if time_col:
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_dtype(perf_data[time_col]):
                    perf_data[time_col] = pd.to_datetime(perf_data[time_col], errors='coerce')
                
                # Sort by time and compute trends
                perf_data = perf_data.sort_values(by=time_col)
                
                # Generate time series visualization
                self._generate_performance_visualization(perf_data, time_col, metrics)
            
            return analysis
        
        return pd.DataFrame()
    
    def _generate_performance_visualization(self, perf_data, time_col, metrics):
        """
        Generate visualizations for performance data.
        
        Args:
            perf_data (pd.DataFrame): Performance data
            time_col (str): Name of the time column
            metrics (list): List of metric columns to plot
        """
        try:
            # Create folder for visualizations
            viz_dir = self.data_loader.output_dir / "visualizations"
            if not os.path.exists(viz_dir):
                os.makedirs(viz_dir)
            
            # Plot each metric over time
            for metric in metrics:
                plt.figure(figsize=(12, 6))
                plt.plot(perf_data[time_col], perf_data[metric], marker='o', linestyle='-')
                
                plt.title(f'{metric.replace("_", " ").title()} Over Time')
                plt.xlabel('Date')
                plt.ylabel(metric.replace("_", " ").title())
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save figure
                output_path = viz_dir / f"{metric}_trend.png"
                plt.savefig(output_path)
                logger.info(f"Saved performance visualization to {output_path}")
                
        except Exception as e:
            logger.error(f"Error generating performance visualization: {str(e)}")


def run_sp_data_analysis(data_dir="data", output_dir="output", use_acceleration=True):
    """
    Run analysis on SP-API data.
    
    Args:
        data_dir (str): Directory containing the SP-API data files
        output_dir (str): Directory to save analysis results
        use_acceleration (bool, optional): Whether to use oneAPI acceleration
        
    Returns:
        dict: Analysis results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "analysis.log"), mode='w'),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Starting SP data analysis with oneAPI acceleration: {use_acceleration}")
    
    # Initialize loader and analyzer
    data_loader = SPDataLoader(data_dir=data_dir, output_dir=output_dir)
    analyzer = SPDataAnalyzer(data_loader=data_loader, use_acceleration=use_acceleration)
    
    # Run analysis
    results = analyzer.run_full_analysis()
    
    logger.info("SP data analysis completed")
    
    return results


if __name__ == "__main__":
    # Run with default settings
    run_sp_data_analysis()
