"""
Data profiling and cleaning script for BibliophileSP.

This script analyzes source data files for quality issues and cleans them.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProfiler:
    """
    Profiles and cleans source data files.
    """
    
    def __init__(self, data_dir="data", output_dir="data/cleaned"):
        """
        Initialize the data profiler.
        
        Args:
            data_dir (str): Directory containing source data files
            output_dir (str): Directory to save cleaned files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store profiling results
        self.profiles = {}
        
    def profile_all_files(self):
        """
        Profile all data files in the data directory.
        """
        logger.info("Starting data profiling...")
        
        # Profile each file type
        self.profile_sales_data()
        self.profile_inventory_data()
        self.profile_performance_data()
        
        # Save profiling report
        self.save_profiling_report()
        
    def profile_sales_data(self):
        """
        Profile sales data files.
        """
        sales_files = [
            "SalesandTrafficByDate.csv",
            "DetailPageSalesandTrafficByDate.csv",
            "SalesandOrdersbyMonth.csv"
        ]
        
        for file_name in sales_files:
            file_path = self.data_dir / file_name
            if file_path.exists():
                logger.info(f"Profiling sales data: {file_name}")
                
                try:
                    # Read the file
                    df = pd.read_csv(file_path)
                    
                    # Generate profile
                    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                        profile = {
                            'file_name': file_name,
                            'row_count': len(df),
                            'column_count': len(df.columns),
                            'columns': df.columns.tolist(),
                            'missing_values': df.isnull().sum().to_dict(),
                            'data_types': df.dtypes.astype(str).to_dict(),
                            'sample_values': {col: df[col].head().tolist() for col in df.columns},
                            'issues': []
                        }
                    else:
                        profile = {
                            'file_name': file_name,
                            'row_count': 0,
                            'column_count': 0,
                            'columns': [],
                            'missing_values': {},
                            'data_types': {},
                            'sample_values': {},
                            'issues': ['Empty or invalid DataFrame']
                        }
                    
                    # Check for common issues
                    
                    # Date format issues
                    if 'Date' in df.columns:
                        try:
                            pd.to_datetime(df['Date'])
                        except:
                            profile['issues'].append("Date format inconsistencies detected")
                    
                    # Currency format issues
                    money_cols = [col for col in df.columns if any(x in col.lower() for x in ['price', 'sales', 'revenue'])]
                    for col in money_cols:
                        if df[col].dtype == object:
                            if df[col].str.contains('$|,', na=False).any():
                                profile['issues'].append(f"Currency formatting in column: {col}")
                    
                    # Store profile
                    self.profiles[file_name] = profile
                    
                except Exception as e:
                    logger.error(f"Error profiling {file_name}: {str(e)}")
    
    def profile_inventory_data(self):
        """
        Profile inventory data files.
        """
        inventory_files = [
            "Inventory+Report+03-22-2025.txt",
            "Open+Listings+Report+03-22-2025.txt",
            "All+Listings+Report+03-22-2025.txt"
        ]
        
        for file_name in inventory_files:
            file_path = self.data_dir / file_name
            if file_path.exists():
                logger.info(f"Profiling inventory data: {file_name}")
                
                try:
                    # Read in chunks for large files
                    chunks = []
                    total_rows = 0
                    column_sample = None
                    missing_values = {}
                    data_types = {}
                    sample_values = {}
                    issues = []
                    
                    for chunk in pd.read_csv(file_path, sep='\t', encoding='utf-8', chunksize=10000):
                        total_rows += len(chunk)
                        
                        # Get column info from first chunk
                        if column_sample is None:
                            column_sample = chunk.columns.tolist()
                            missing_values = chunk.isnull().sum().to_dict()
                            data_types = chunk.dtypes.astype(str).to_dict()
                            sample_values = {col: chunk[col].head().tolist() for col in chunk.columns}
                        else:
                            # Update missing values counts
                            for col in chunk.columns:
                                missing_values[col] += chunk[col].isnull().sum()
                    
                    # Generate profile
                    profile = {
                        'file_name': file_name,
                        'row_count': total_rows,
                        'column_count': len(column_sample),
                        'columns': column_sample,
                        'missing_values': missing_values,
                        'data_types': data_types,
                        'sample_values': sample_values,
                        'issues': issues
                    }
                    
                    # Store profile
                    self.profiles[file_name] = profile
                    
                except Exception as e:
                    logger.error(f"Error profiling {file_name}: {str(e)}")
    
    def profile_performance_data(self):
        """
        Profile performance data files.
        """
        performance_files = [
            "SellerPerformance.csv",
            "UnitOnTimeDelivery_1282029020169.csv",
            "OrderDefects_1282032020169.csv"
        ]
        
        for file_name in performance_files:
            file_path = self.data_dir / file_name
            if file_path.exists():
                logger.info(f"Profiling performance data: {file_name}")
                
                try:
                    # Read the file
                    df = pd.read_csv(file_path)
                    
                    # Generate profile
                    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                        profile = {
                            'file_name': file_name,
                            'row_count': len(df),
                            'column_count': len(df.columns),
                            'columns': df.columns.tolist(),
                            'missing_values': df.isnull().sum().to_dict(),
                            'data_types': df.dtypes.astype(str).to_dict(),
                            'sample_values': {col: df[col].head().tolist() for col in df.columns},
                            'issues': []
                        }
                    else:
                        profile = {
                            'file_name': file_name,
                            'row_count': 0,
                            'column_count': 0,
                            'columns': [],
                            'missing_values': {},
                            'data_types': {},
                            'sample_values': {},
                            'issues': ['Empty or invalid DataFrame']
                        }
                    
                    # Check for common issues
                    
                    # Date format issues
                    date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'period'])]
                    for col in date_cols:
                        try:
                            pd.to_datetime(df[col])
                        except:
                            profile['issues'].append(f"Date format inconsistencies in column: {col}")
                    
                    # Percentage format issues
                    pct_cols = [col for col in df.columns if any(x in col.lower() for x in ['rate', 'percentage', 'pct'])]
                    for col in pct_cols:
                        if df[col].dtype == object:
                            if df[col].str.contains('%', na=False).any():
                                profile['issues'].append(f"Percentage formatting in column: {col}")
                    
                    # Store profile
                    self.profiles[file_name] = profile
                    
                except Exception as e:
                    logger.error(f"Error profiling {file_name}: {str(e)}")
    
    def save_profiling_report(self):
        """
        Save profiling results to a report file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"data_profile_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("BibliophileSP Data Profiling Report\n")
            f.write("=================================\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for file_name, profile in self.profiles.items():
                f.write(f"\nFile: {file_name}\n")
                f.write("-" * (len(file_name) + 6) + "\n")
                
                f.write(f"Row count: {profile['row_count']}\n")
                f.write(f"Column count: {profile['column_count']}\n\n")
                
                f.write("Columns:\n")
                for col in profile['columns']:
                    missing = profile['missing_values'].get(col, 0)
                    dtype = profile['data_types'].get(col, 'unknown')
                    missing_pct = (missing / profile['row_count']) * 100 if profile['row_count'] > 0 else 0
                    f.write(f"  - {col}\n")
                    f.write(f"    Type: {dtype}\n")
                    f.write(f"    Missing: {missing} ({missing_pct:.1f}%)\n")
                    f.write(f"    Sample values: {profile['sample_values'].get(col, [])[:3]}\n")
                
                if profile['issues']:
                    f.write("\nIssues:\n")
                    for issue in profile['issues']:
                        f.write(f"  - {issue}\n")
                
                f.write("\n")
        
        logger.info(f"Profiling report saved to {report_path}")
    
    def clean_data(self):
        """
        Clean data files based on profiling results.
        """
        logger.info("Starting data cleaning...")
        
        # Clean each file type
        self.clean_sales_data()
        self.clean_inventory_data()
        self.clean_performance_data()
    
    def clean_sales_data(self):
        """
        Clean sales data files.
        """
        sales_files = [
            "SalesandTrafficByDate.csv",
            "DetailPageSalesandTrafficByDate.csv",
            "SalesandOrdersbyMonth.csv"
        ]
        
        for file_name in sales_files:
            file_path = self.data_dir / file_name
            if file_path.exists():
                logger.info(f"Cleaning sales data: {file_name}")
                
                try:
                    # Read the file
                    df = pd.read_csv(file_path)
                    
                    # Clean date columns
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                    
                    # Clean currency columns
                    money_cols = [col for col in df.columns if any(x in col.lower() for x in ['price', 'sales', 'revenue'])]
                    for col in money_cols:
                        if df[col].dtype == object:
                            df[col] = df[col].str.replace('$', '').str.replace(',', '').astype(float)
                    
                    # Save cleaned file
                    output_path = self.output_dir / f"cleaned_{file_name}"
                    df.to_csv(output_path, index=False)
                    logger.info(f"Cleaned file saved to {output_path}")
                    
                except Exception as e:
                    logger.error(f"Error cleaning {file_name}: {str(e)}")
    
    def clean_inventory_data(self):
        """
        Clean inventory data files.
        """
        inventory_files = [
            "Inventory+Report+03-22-2025.txt",
            "Open+Listings+Report+03-22-2025.txt",
            "All+Listings+Report+03-22-2025.txt"
        ]
        
        for file_name in inventory_files:
            file_path = self.data_dir / file_name
            if file_path.exists():
                logger.info(f"Cleaning inventory data: {file_name}")
                
                try:
                    # Process in chunks
                    chunks = []
                    for chunk in pd.read_csv(file_path, sep='\t', encoding='utf-8', chunksize=10000):
                        # Clean chunk
                        chunk = self._clean_inventory_chunk(chunk)
                        chunks.append(chunk)
                    
                    # Combine chunks
                    df = pd.concat(chunks, ignore_index=True)
                    
                    # Save cleaned file
                    output_path = self.output_dir / f"cleaned_{file_name}"
                    df.to_csv(output_path, sep='\t', index=False)
                    logger.info(f"Cleaned file saved to {output_path}")
                    
                except Exception as e:
                    logger.error(f"Error cleaning {file_name}: {str(e)}")
    
    def _clean_inventory_chunk(self, chunk):
        """
        Clean a chunk of inventory data.
        
        Args:
            chunk (pd.DataFrame): Chunk of inventory data
            
        Returns:
            pd.DataFrame: Cleaned chunk
        """
        # Standardize column names
        col_map = {
            'seller-sku': 'SKU',
            'sku': 'SKU',
            'asin1': 'ASIN',
            'item-name': 'Title',
            'item_name': 'Title',
            'item-description': 'Title',
            'product-name': 'Title',
            'price': 'Price',
            'price-amount': 'Price',
            'quantity': 'Quantity',
            'quantity-available': 'Quantity'
        }
        chunk = chunk.rename(columns=col_map)
        
        # Clean price columns
        if 'Price' in chunk.columns:
            chunk['Price'] = pd.to_numeric(chunk['Price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
        
        # Clean quantity columns
        if 'Quantity' in chunk.columns:
            chunk['Quantity'] = pd.to_numeric(chunk['Quantity'], errors='coerce')
        
        # Fill missing values
        chunk = chunk.fillna({
            'SKU': 'UNKNOWN',
            'Title': 'UNKNOWN',
            'Price': 0.0,
            'Quantity': 0
        })
        
        return chunk
    
    def clean_performance_data(self):
        """
        Clean performance data files.
        """
        performance_files = [
            "SellerPerformance.csv",
            "UnitOnTimeDelivery_1282029020169.csv",
            "OrderDefects_1282032020169.csv"
        ]
        
        for file_name in performance_files:
            file_path = self.data_dir / file_name
            if file_path.exists():
                logger.info(f"Cleaning performance data: {file_name}")
                
                try:
                    # Read the file
                    df = pd.read_csv(file_path)
                    
                    # Clean date columns
                    date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'period'])]
                    for col in date_cols:
                        try:
                            # Handle complex GMT/PST timestamps
                            if df[col].str.contains('GMT').any():
                                # Extract the GMT part before the parentheses
                                df[col] = df[col].str.extract(r'(.*?)\s*(?:\(|$)').iloc[:, 0]
                                df[col] = pd.to_datetime(df[col], format='%m/%d/%y %H:%M:%S GMT')
                            else:
                                df[col] = pd.to_datetime(df[col])
                        except Exception as e:
                            logger.warning(f"Could not parse dates in column {col}: {str(e)}")
                            
                    # Clean SKU columns that have ="..." format
                    sku_cols = ['SKU', 'sku', 'seller-sku']
                    for col in sku_cols:
                        if col in df.columns and df[col].dtype == object:
                            # Remove =" and " from values
                            df[col] = df[col].str.replace(r'^="(.*)"$', r'\1', regex=True)
                    
                    # Clean percentage columns
                    pct_cols = [col for col in df.columns if any(x in col.lower() for x in ['rate', 'percentage', 'pct'])]
                    for col in pct_cols:
                        if df[col].dtype == object:
                            df[col] = df[col].str.rstrip('%').astype(float) / 100
                    
                    # Save cleaned file
                    output_path = self.output_dir / f"cleaned_{file_name}"
                    df.to_csv(output_path, index=False)
                    logger.info(f"Cleaned file saved to {output_path}")
                    
                except Exception as e:
                    logger.error(f"Error cleaning {file_name}: {str(e)}")


if __name__ == "__main__":
    # Create profiler
    profiler = DataProfiler()
    
    # Profile data
    profiler.profile_all_files()
    
    # Clean data
    profiler.clean_data()
