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

from .analysis import InventoryAnalyzer
from .oneapi_accelerator import AccelerationContext, ONEAPI_AVAILABLE, accelerated_kmeans

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
        self.load_sales_data()
        self.load_inventory_data()
        self.load_performance_data()
        return self
    
    def load_sales_data(self):
        """
        Load sales data from relevant files.
        
        Returns:
            pd.DataFrame: Processed sales data
        """
        # Try to load from SalesandTrafficByDate.csv first
        sales_file = self.data_dir / "SalesandTrafficByDate.csv"
        
        if sales_file.exists():
            logger.info(f"Loading sales data from {sales_file}")
            sales_df = pd.read_csv(sales_file)
            
            # Process the sales data
            if "Date" in sales_df.columns:
                sales_df["Date"] = pd.to_datetime(sales_df["Date"])
            
            # Map to standard columns expected by InventoryAnalyzer
            column_mapping = {
                "Date": "Date",
                "OrderedProductSales": "Sales",
                "OrderedUnits": "Quantity",
                "ASIN": "SKU"  # Assuming ASIN can be used as SKU for analysis
            }
            
            # Rename columns if they exist
            for old_col, new_col in column_mapping.items():
                if old_col in sales_df.columns and old_col != new_col:
                    sales_df[new_col] = sales_df[old_col]
            
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
            self.sales_data = pd.DataFrame(columns=["Date", "SKU", "Quantity", "Sales"])
        
        return self.sales_data
    
    def load_inventory_data(self):
        """
        Load inventory data from relevant files.
        
        Returns:
            pd.DataFrame: Processed inventory data
        """
        # Try to load from inventory report files
        inventory_files = [
            "Inventory+Report+03-22-2025.txt",
            "Open+Listings+Report+03-22-2025.txt",
            "All+Listings+Report+03-22-2025.txt"
        ]
        
        for file_name in inventory_files:
            file_path = self.data_dir / file_name
            if file_path.exists():
                logger.info(f"Loading inventory data from {file_path}")
                
                # For tab-delimited text files
                try:
                    inventory_df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
                    
                    # Map columns to standard format expected by analysis
                    if "seller-sku" in inventory_df.columns:
                        inventory_df["SKU"] = inventory_df["seller-sku"]
                    elif "sku" in inventory_df.columns:
                        inventory_df["SKU"] = inventory_df["sku"]
                    
                    if "asin1" in inventory_df.columns:
                        inventory_df["ASIN"] = inventory_df["asin1"]
                    
                    if "item-name" in inventory_df.columns:
                        inventory_df["Title"] = inventory_df["item-name"]
                    elif "item_name" in inventory_df.columns:
                        inventory_df["Title"] = inventory_df["item_name"]
                    
                    if "price" in inventory_df.columns:
                        inventory_df["Price"] = pd.to_numeric(inventory_df["price"], errors="coerce")
                    
                    # Check for image availability
                    if "has_image" in inventory_df.columns:
                        inventory_df["Has_Image"] = inventory_df["has_image"].astype(bool)
                    elif "main-image-url" in inventory_df.columns:
                        inventory_df["Has_Image"] = ~inventory_df["main-image-url"].isna()
                    
                    # Check for ISBN
                    isbn_cols = ["isbn", "external_product_id", "product-id"]
                    for col in isbn_cols:
                        if col in inventory_df.columns:
                            inventory_df["ISBN"] = inventory_df[col]
                            break
                    
                    self.inventory_data = inventory_df
                    logger.info(f"Loaded {len(inventory_df)} inventory records")
                    break
                
                except Exception as e:
                    logger.error(f"Error reading inventory file {file_path}: {str(e)}")
        
        if self.inventory_data is None:
            logger.warning("No inventory data files found. Analysis will be limited.")
            # Create empty DataFrame with expected columns
            self.inventory_data = pd.DataFrame(columns=["SKU", "Title", "Price", "Has_Image", "ISBN"])
        
        return self.inventory_data
    
    def load_performance_data(self):
        """
        Load seller performance data.
        
        Returns:
            pd.DataFrame: Processed performance data
        """
        # Try different performance files
        performance_files = [
            "SellerPerformance.csv",
            "UnitOnTimeDelivery_1282029020169.csv",
            "OrderDefects_1282032020169.csv"
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
                
                # Save results
                output_path = self.data_loader.output_dir / "inventory_analysis.csv"
                basic_analysis.to_csv(output_path, index=False)
                logger.info(f"Saved basic analysis to {output_path}")
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
                clustered_data.to_csv(output_path, index=False)
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
                    performance_analysis.to_csv(output_path, index=False)
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
            
            # Extract numerical features
            numeric_cols = ['Price', 'Sales_Last_30d', 'Days_Since_Last_Sale']
            plot_data = clustered_data[numeric_cols + ['Cluster', 'Cluster_Description']].copy()
            
            # Replace missing values
            plot_data = plot_data.fillna(0)
            
            # Generate scatter plot of Price vs. Sales
            plt.figure(figsize=(10, 6))
            for cluster in plot_data['Cluster'].unique():
                cluster_data = plot_data[plot_data['Cluster'] == cluster]
                plt.scatter(
                    cluster_data['Price'], 
                    cluster_data['Sales_Last_30d'],
                    label=f"Cluster {cluster}: {cluster_data['Cluster_Description'].iloc[0]}"
                )
            
            plt.title('Price vs. Sales by Cluster')
            plt.xlabel('Price ($)')
            plt.ylabel('Sales (Last 30 Days)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save figure
            output_path = viz_dir / "price_vs_sales_clusters.png"
            plt.savefig(output_path)
            logger.info(f"Saved cluster visualization to {output_path}")
            
            # Generate another plot for Price vs. Days Since Last Sale
            plt.figure(figsize=(10, 6))
            for cluster in plot_data['Cluster'].unique():
                cluster_data = plot_data[plot_data['Cluster'] == cluster]
                plt.scatter(
                    cluster_data['Price'], 
                    cluster_data['Days_Since_Last_Sale'],
                    label=f"Cluster {cluster}: {cluster_data['Cluster_Description'].iloc[0]}"
                )
            
            plt.title('Price vs. Days Since Last Sale by Cluster')
            plt.xlabel('Price ($)')
            plt.ylabel('Days Since Last Sale')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save figure
            output_path = viz_dir / "price_vs_recency_clusters.png"
            plt.savefig(output_path)
            
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
