import unittest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the src directory to the path if not already there
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import oneapi_accelerator module to get platform info and availability flags
try:
    from src.oneapi_accelerator import (
        ONEAPI_AVAILABLE, APPLE_ACCELERATE_AVAILABLE, ACCELERATION_AVAILABLE,
        IS_MAC, IS_ARM_MAC, enable_acceleration, disable_acceleration
    )
except ImportError:
    # Default values if module not available
    ONEAPI_AVAILABLE = False
    APPLE_ACCELERATE_AVAILABLE = False
    ACCELERATION_AVAILABLE = False
    IS_MAC = False
    IS_ARM_MAC = False
    
    def enable_acceleration():
        """Dummy function that always returns False."""
        return False
        
    def disable_acceleration():
        """Dummy function that always returns False."""
        return False


class TestOneAPIIntegration(unittest.TestCase):
    """Test cases for oneAPI integration."""
    
    @unittest.skipIf(not ONEAPI_AVAILABLE, "oneAPI modules not available")
    def test_daal4py_available(self):
        """Test that daal4py is available."""
        try:
            import daal4py
            self.assertTrue(ONEAPI_AVAILABLE)
            self.assertIsNotNone(daal4py)
        except ImportError:
            self.skipTest("daal4py module not available")
    
    @unittest.skipIf(not ONEAPI_AVAILABLE, "oneAPI modules not available")
    def test_sklearnex_available(self):
        """Test that scikit-learn-intelex is available."""
        try:
            import sklearnex
            self.assertTrue(ONEAPI_AVAILABLE)
            self.assertIsNotNone(sklearnex)
        except ImportError:
            self.skipTest("sklearnex module not available")
    
    @unittest.skipIf(not ONEAPI_AVAILABLE, "oneAPI modules not available")
    def test_sklearnex_patching(self):
        """Test that scikit-learn-intelex patches scikit-learn."""
        try:
            import sklearnex
            from sklearn import cluster
            
            # Patch scikit-learn
            sklearnex.patch_sklearn()
            
            # Create a simple dataset
            X = np.random.rand(100, 2)
            
            # Run KMeans
            kmeans = cluster.KMeans(n_clusters=2, random_state=0)
            kmeans.fit(X)
            
            # Unpatch scikit-learn
            sklearnex.unpatch_sklearn()
            
            # Check that the model was created
            self.assertIsNotNone(kmeans.cluster_centers_)
        except ImportError:
            self.skipTest("sklearnex module not available")
    
    @unittest.skipIf(not ONEAPI_AVAILABLE, "oneAPI modules not available")
    def test_daal4py_kmeans(self):
        """Test daal4py KMeans implementation."""
        try:
            import daal4py
            
            # Create a simple dataset
            X = np.random.rand(100, 2)
            
            # Initialize the algorithm
            kmeans_init = daal4py.kmeans_init(nClusters=2, method="randomDense")
            kmeans_init_result = kmeans_init.compute(X)
            
            # Compute KMeans
            kmeans_algo = daal4py.kmeans(nClusters=2, maxIterations=10)
            kmeans_result = kmeans_algo.compute(X, kmeans_init_result.centroids)
            
            # Check that the model was created
            self.assertIsNotNone(kmeans_result.centroids)
        except ImportError:
            self.skipTest("daal4py module not available")
    
    def test_numpy_operations(self):
        """Test basic NumPy operations (should work with or without oneAPI)."""
        # Create a simple dataset
        X = np.random.rand(100, 2)
        
        # Perform some operations
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        normalized_X = (X - mean) / std
        
        # Check the results
        self.assertEqual(normalized_X.shape, X.shape)
        self.assertAlmostEqual(np.mean(normalized_X[:, 0]), 0.0, delta=1e-10)
        self.assertAlmostEqual(np.std(normalized_X[:, 0]), 1.0, delta=1e-10)


class TestOneAPIAnalysis(unittest.TestCase):
    """Test cases for oneAPI-accelerated analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Create sales data
        dates = pd.date_range(start='2024-01-01', periods=n_samples)
        skus = [f'SKU{i:04d}' for i in range(100)]
        
        self.sales_data = pd.DataFrame({
            'SKU': np.random.choice(skus, n_samples),
            'Date': np.random.choice(dates, n_samples),
            'Quantity': np.random.randint(1, 10, n_samples),
            'Price': np.random.uniform(5, 50, n_samples)
        })
        
        # Create inventory data
        self.inventory_data = pd.DataFrame({
            'SKU': skus,
            'Title': [f'Book {i}' for i in range(100)],
            'ISBN': [f'ISBN{i:010d}' for i in range(100)],
            'Price': np.random.uniform(5, 50, 100),
            'Has_Image': np.random.choice([True, False], 100, p=[0.8, 0.2])
        })
    
    def test_large_dataset_performance(self):
        """Test performance with a large dataset."""
        from src.analysis import InventoryAnalyzer
        
        # Create analyzer
        analyzer = InventoryAnalyzer(self.sales_data, self.inventory_data)
        
        # Measure time to analyze
        import time
        start_time = time.time()
        results = analyzer.analyze()
        end_time = time.time()
        
        # Print performance info
        print(f"\nAnalysis time: {end_time - start_time:.4f} seconds")
        print(f"Dataset size: {len(self.sales_data)} sales records, {len(self.inventory_data)} inventory records")
        
        # Check results
        self.assertIsNotNone(results)
        self.assertEqual(len(results), len(self.inventory_data))


if __name__ == '__main__':
    unittest.main()
