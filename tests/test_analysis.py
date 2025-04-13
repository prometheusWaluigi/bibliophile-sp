import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analysis import InventoryAnalyzer


class TestInventoryAnalyzer(unittest.TestCase):
    """Test cases for the InventoryAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test dataset
        now = pd.Timestamp.now()
        
        # Create sales data
        self.sales_data = pd.DataFrame({
            'SKU': ['TEST001', 'TEST002', 'TEST003'],
            'Date': [
                now - pd.Timedelta(days=150),  # Stale
                now - pd.Timedelta(days=2),    # Recent
                now - pd.Timedelta(days=100)   # Borderline
            ],
            'Quantity': [1, 3, 1],
            'Price': [10.99, 15.99, 5.99]
        })
        
        # Create inventory data
        self.inventory_data = pd.DataFrame({
            'SKU': ['TEST001', 'TEST002', 'TEST003', 'TEST004'],
            'Title': ['Stale Book', 'Good Seller', 'Borderline Book', 'No Sales Data'],
            'ISBN': ['1234567890', '0987654321', '1122334455', None],
            'Price': [10.99, 15.99, 5.99, 1.99],
            'Has_Image': [True, True, True, False]
        })
        
        # Initialize the analyzer
        self.analyzer = InventoryAnalyzer(self.sales_data, self.inventory_data)
    
    def test_analyze_flags_stale_items(self):
        """Test that stale items are correctly flagged."""
        results = self.analyzer.analyze()
        
        # TEST001 should be flagged as stale (150 days since last sale)
        self.assertEqual(results.loc[results['SKU'] == 'TEST001', 'Flag'].iloc[0], '⚠️')
        
        # TEST002 should be flagged as good (2 days since last sale)
        self.assertEqual(results.loc[results['SKU'] == 'TEST002', 'Flag'].iloc[0], '✅')
    
    def test_analyze_handles_missing_sales_data(self):
        """Test that items with no sales data are handled correctly."""
        results = self.analyzer.analyze()
        
        # TEST004 has no sales data, should be in results but might be flagged
        self.assertTrue('TEST004' in results['SKU'].values)
    
    def test_analyze_detects_bad_metadata(self):
        """Test that items with bad metadata are correctly flagged."""
        # Modify TEST004 to have bad metadata (no ISBN, no image)
        results = self.analyzer.analyze()
        
        # TEST004 should be flagged due to bad metadata
        self.assertEqual(results.loc[results['SKU'] == 'TEST004', 'Flag'].iloc[0], '⚠️')
    
    def test_analyze_returns_correct_columns(self):
        """Test that the analyze method returns the correct columns."""
        results = self.analyzer.analyze()
        
        expected_columns = ['SKU', 'Title', 'Sales Last 30d', 'Days Since Last Sale', 'Flag', 'Notes']
        for col in expected_columns:
            self.assertIn(col, results.columns)


if __name__ == '__main__':
    unittest.main()
