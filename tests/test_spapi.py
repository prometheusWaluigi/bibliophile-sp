import unittest
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spapi import SPAPIClient, get_client


class TestSPAPIClient(unittest.TestCase):
    """Test cases for the SPAPIClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = SPAPIClient(access_token="test_token")
    
    def test_init(self):
        """Test that the client initializes correctly."""
        self.assertEqual(self.client.access_token, "test_token")
        self.assertEqual(self.client.endpoint_base, "https://sellingpartnerapi-na.amazon.com")
    
    def test_get_inventory_summary(self):
        """Test that get_inventory_summary returns a DataFrame."""
        inventory_data = self.client.get_inventory_summary()
        
        # Check that it returns a DataFrame
        self.assertIsInstance(inventory_data, pd.DataFrame)
        
        # Check that it has the expected columns
        expected_columns = ['SKU', 'ASIN', 'Title', 'Condition', 'Price', 'Quantity']
        for col in expected_columns:
            self.assertIn(col, inventory_data.columns)
    
    def test_get_sales_history(self):
        """Test that get_sales_history returns a DataFrame."""
        sales_data = self.client.get_sales_history()
        
        # Check that it returns a DataFrame
        self.assertIsInstance(sales_data, pd.DataFrame)
        
        # Check that it has the expected columns
        expected_columns = ['SKU', 'Date', 'Quantity', 'Price']
        for col in expected_columns:
            self.assertIn(col, sales_data.columns)
    
    def test_get_catalog_data(self):
        """Test that get_catalog_data returns a DataFrame."""
        catalog_data = self.client.get_catalog_data(skus=["B000FJS1B4"])
        
        # Check that it returns a DataFrame
        self.assertIsInstance(catalog_data, pd.DataFrame)
        
        # Check that it has the expected columns
        expected_columns = ['SKU', 'ASIN', 'Title', 'Author', 'ISBN', 'Has_Image', 'Publication_Date']
        for col in expected_columns:
            self.assertIn(col, catalog_data.columns)
    
    @patch('src.spapi.SPAPIClient')
    def test_get_client(self, mock_client):
        """Test that get_client returns a client instance."""
        # Set up the mock
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Call the function
        client = get_client(access_token="test_token")
        
        # Check that it returns the mock instance
        self.assertEqual(client, mock_instance)
        
        # Check that it was called with the correct arguments
        mock_client.assert_called_once_with(access_token="test_token")


if __name__ == '__main__':
    unittest.main()
