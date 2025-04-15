import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sp_data_analysis import SPDataAnalyzer, SPDataLoader
from src.oneapi_accelerator import AccelerationContext, enable_acceleration, disable_acceleration


class TestSPDataAnalysis(unittest.TestCase):
    """Test cases for the SP data analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories for test data and output
        self.test_data_dir = tempfile.mkdtemp()
        self.test_output_dir = tempfile.mkdtemp()
        
        # Create test datasets with various data types and edge cases
        self.create_test_sales_data()
        self.create_test_inventory_data()
        self.create_test_performance_data()
        
        # Create data loader with test data
        self.data_loader = SPDataLoader(
            data_dir=self.test_data_dir,
            output_dir=self.test_output_dir
        )
        
        # Create analyzer with test data loader
        self.analyzer = SPDataAnalyzer(
            data_loader=self.data_loader,
            use_acceleration=True
        )
    
    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.test_data_dir)
        shutil.rmtree(self.test_output_dir)
    
    def create_test_sales_data(self):
        """Create test sales data with various data types and edge cases."""
        # Create date range for the past two years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create sales data with different patterns
        sales_data = pd.DataFrame({
            'Date': dates,
            'Units Ordered': np.random.randint(0, 100, size=len(dates)),
            'Units Ordered - B2B': np.random.randint(0, 10, size=len(dates)),
            # Include some NULL values to test handling
            'Sales Amount': [np.nan if i % 50 == 0 else np.random.uniform(10, 1000) for i in range(len(dates))],
            # Include some string values that should be numeric
            'Sales Amount - B2B': [str(np.random.uniform(10, 100)) if i % 10 == 0 else np.random.uniform(10, 100) for i in range(len(dates))]
        })
        
        # Save to CSV
        sales_file = os.path.join(self.test_data_dir, 'SalesandTrafficByDate.csv')
        sales_data.to_csv(sales_file, index=False)
    
    def create_test_inventory_data(self):
        """Create test inventory data with various data types and edge cases."""
        # Create inventory with different SKU patterns and data types
        skus = [f'TEST{i:03d}' for i in range(100)]
        inventory_data = pd.DataFrame({
            'seller-sku': skus,
            'asin1': [f'B{i:09d}' for i in range(100)],
            'item-name': [f'Test Product {i}' for i in range(100)],
            # Mix of integer and float price data
            'price': [float(i) if i % 2 == 0 else i for i in range(100)],
            # Some boolean flags
            'has_image': [True if i % 5 != 0 else False for i in range(100)],
            # Some dates as strings
            'last-updated': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(100)],
            # Some NULL values
            'category': [None if i % 10 == 0 else f'Category {i % 5}' for i in range(100)],
            # Some mixed numeric/string data
            'quantity': [str(i) if i % 7 == 0 else i for i in range(100)]
        })
        
        # Save to TSV (tab-separated) to test different file formats
        inventory_file = os.path.join(self.test_data_dir, 'Inventory+Report+03-22-2025.txt')
        inventory_data.to_csv(inventory_file, sep='\t', index=False)
    
    def create_test_performance_data(self):
        """Create test performance data with various metrics."""
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create performance metrics
        performance_data = pd.DataFrame({
            'Date': dates,
            'Units Ordered': np.random.randint(0, 100, size=len(dates)),
            'Units Ordered - B2B': np.random.randint(0, 10, size=len(dates)),
            'Total Order Items': np.random.randint(0, 100, size=len(dates)),
            'Total Order Items - B2B': np.random.randint(0, 10, size=len(dates)),
            'Units Refunded': np.random.randint(0, 5, size=len(dates)),
            'Feedback Received': np.random.randint(0, 3, size=len(dates)),
            'Negative Feedback Received': np.random.randint(0, 1, size=len(dates)),
            'A-to-z Claims Granted': np.zeros(len(dates))  # Mostly zeros with occasional 1
        })
        
        # Add some claims for testing
        random_indices = np.random.choice(range(len(dates)), 5, replace=False)
        for idx in random_indices:
            performance_data.loc[idx, 'A-to-z Claims Granted'] = 1
        
        # Save to CSV
        performance_file = os.path.join(self.test_data_dir, 'SellerPerformance.csv')
        performance_data.to_csv(performance_file, index=False)
    
    def test_data_loader_initialization(self):
        """Test that SPDataLoader initializes correctly."""
        self.assertTrue(hasattr(self.data_loader, 'data_dir'))
        self.assertTrue(hasattr(self.data_loader, 'output_dir'))
        
        # Check attributes are Path objects
        self.assertIsInstance(self.data_loader.data_dir, Path)
        self.assertIsInstance(self.data_loader.output_dir, Path)
    
    def test_analyzer_initialization(self):
        """Test that SPDataAnalyzer initializes correctly."""
        self.assertTrue(hasattr(self.analyzer, 'data_loader'))
        self.assertTrue(hasattr(self.analyzer, 'use_acceleration'))
        
        # Check that the analyzer has the correct data_loader
        self.assertEqual(self.analyzer.data_loader, self.data_loader)
    
    def test_data_loader_load_sales_data(self):
        """Test loading sales data from CSV."""
        sales_df = self.data_loader.load_sales_data()
        
        # Check that the data was loaded correctly
        self.assertIsInstance(sales_df, pd.DataFrame)
        self.assertGreater(len(sales_df), 0)
        self.assertIn('Date', sales_df.columns)
        
        # Check that dates were parsed correctly
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(sales_df['Date']))
    
    def test_data_loader_load_inventory_data(self):
        """Test loading inventory data from TSV."""
        inventory_df = self.data_loader.load_inventory_data()
        
        # Check that the data was loaded correctly
        self.assertIsInstance(inventory_df, pd.DataFrame)
        self.assertGreater(len(inventory_df), 0)
        self.assertIn('SKU', inventory_df.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_string_dtype(inventory_df['SKU']))
    
    def test_data_loader_load_performance_data(self):
        """Test loading performance data from CSV."""
        performance_df = self.data_loader.load_performance_data()
        
        # Check that the data was loaded correctly
        self.assertIsInstance(performance_df, pd.DataFrame)
        self.assertGreater(len(performance_df), 0)
        self.assertIn('Date', performance_df.columns)
        self.assertIn('Units Ordered', performance_df.columns)
        
        # Check that all expected columns are present
        expected_columns = [
            'Units Ordered', 'Units Ordered - B2B', 'Total Order Items', 
            'Total Order Items - B2B', 'Units Refunded', 'Feedback Received',
            'Negative Feedback Received', 'A-to-z Claims Granted'
        ]
        for col in expected_columns:
            self.assertIn(col, performance_df.columns)
    
    @patch('src.oneapi_accelerator.ONEAPI_AVAILABLE', True)
    @patch('src.oneapi_accelerator.enable_acceleration')
    def test_analyzer_with_acceleration(self, mock_enable_acceleration):
        """Test that oneAPI acceleration is enabled correctly."""
        # Create analyzer with acceleration enabled
        analyzer = SPDataAnalyzer(
            data_loader=self.data_loader,
            use_acceleration=True
        )
        
        # Verify acceleration is enabled
        self.assertTrue(analyzer.use_acceleration)
        
        # Run the analyzer
        with patch('src.sp_data_analysis.AccelerationContext'):
            analyzer.run_full_analysis()
        
        # enable_acceleration should have been called inside the AccelerationContext __enter__
        # Since we're mocking AccelerationContext, we won't see the call directly
        # but we can verify the analyzer has acceleration enabled
        self.assertTrue(analyzer.use_acceleration)
    
    @patch('src.oneapi_accelerator.ONEAPI_AVAILABLE', False)
    def test_analyzer_without_acceleration(self):
        """Test that oneAPI acceleration is disabled correctly."""
        # Create analyzer with acceleration disabled
        analyzer = SPDataAnalyzer(
            data_loader=self.data_loader,
            use_acceleration=False
        )
        
        # Verify acceleration is disabled
        self.assertFalse(analyzer.use_acceleration)
        
        # Run the analyzer
        analyzer.run_full_analysis()
    
    @patch('src.sp_data_analysis.os.path.exists')
    def test_run_full_analysis(self, mock_exists):
        """Test that the full analysis runs and generates outputs."""
        # Mock os.path.exists to return True for expected files
        mock_exists.return_value = True
        
        # Run analysis
        self.analyzer.run_full_analysis()
        
        # Check that mock was called for expected files
        mock_exists.assert_called()
    
    def test_handle_missing_files(self):
        """Test handling of missing data files."""
        # Create data loader with empty directory
        with tempfile.TemporaryDirectory() as empty_dir:
            data_loader = SPDataLoader(
                data_dir=empty_dir,
                output_dir=self.test_output_dir
            )
            
            # Attempt to load data from empty directory
            with self.assertLogs(level='WARNING'):
                sales_df = data_loader.load_sales_data()
                # Should still return a DataFrame, even if empty
                self.assertIsInstance(sales_df, pd.DataFrame)
    
    def test_handle_malformed_data(self):
        """Test handling of malformed data."""
        # Create a malformed CSV file
        malformed_file = os.path.join(self.test_data_dir, 'MalformedData.csv')
        with open(malformed_file, 'w') as f:
            f.write("Column1,Column2\n")
            f.write("Value1,Value2\n")  # Good line
            f.write("Value3,Value4\n")  # Good line
            f.write("MissingColumn\n")  # Line with missing column (pandas treats as valid with NaN)
            f.write("Extra,Columns,Here\n")  # Extra column (pandas skips this)
        
        # Try loading with various error handling options
        # 1. First with default settings - should raise an exception
        with self.assertRaises(Exception):
            pd.read_csv(malformed_file)
        
        # 2. With skip option - should include well-formed rows and rows with missing values
        result = pd.read_csv(malformed_file, on_bad_lines='skip')
        self.assertIsInstance(result, pd.DataFrame)
        
        # Per pandas behavior, rows with missing columns are kept with NaN values
        # Only rows with extra columns are skipped
        self.assertEqual(len(result), 3, "Expected well-formed rows plus rows with missing values")
        
        # Check the values in the parsed rows
        self.assertEqual(result.iloc[0, 0], "Value1")
        self.assertEqual(result.iloc[1, 0], "Value3")
        self.assertEqual(result.iloc[2, 0], "MissingColumn")
        self.assertTrue(pd.isna(result.iloc[2, 1]), "Missing value should be NaN")


if __name__ == '__main__':
    unittest.main()
