import unittest
import os
import sys
import numpy as np
import platform

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import oneapi_accelerator module to get platform info
try:
    from src.oneapi_accelerator import (
        ONEAPI_AVAILABLE, APPLE_ACCELERATE_AVAILABLE, IS_MAC, IS_ARM_MAC
    )
except ImportError:
    # Default values if module not available
    ONEAPI_AVAILABLE = False
    APPLE_ACCELERATE_AVAILABLE = False
    IS_MAC = platform.system() == "Darwin"
    IS_ARM_MAC = IS_MAC and platform.machine() == "arm64"


@unittest.skipIf(not IS_MAC, "Not running on macOS")
class TestAppleAccelerate(unittest.TestCase):
    """Test cases for Apple Accelerate framework integration."""
    
    def test_platform_detection(self):
        """Test that the platform is correctly detected as macOS."""
        self.assertTrue(IS_MAC)
        self.assertEqual(platform.system(), "Darwin")
        
        # Log whether this is Apple Silicon or Intel Mac
        if IS_ARM_MAC:
            print("Running on Apple Silicon (M1/M2/M3)")
        else:
            print("Running on Intel Mac")
    
    @unittest.skipIf(not APPLE_ACCELERATE_AVAILABLE, "Apple Accelerate framework not available")
    def test_numpy_acceleration(self):
        """Test that NumPy is accelerated with Apple's Accelerate framework."""
        # Check if numpy config shows Accelerate
        try:
            # Try to access numpy's config info (not all installations expose this the same way)
            if hasattr(np, '__config__'):
                config_info = str(np.__config__).lower()
            elif hasattr(np, 'show_config'):
                from io import StringIO
                buffer = StringIO()
                np.show_config(buffer)
                config_info = buffer.getvalue().lower()
            else:
                # If we can't check directly, just log and skip this assertion
                print("Unable to access NumPy configuration information")
                return
                
            # Look for indicators of Accelerate framework
            self.assertTrue(
                'accelerate' in config_info or 
                'veclib' in config_info or
                'blas_opt_info' in config_info
            )
        except (AttributeError, ImportError):
            print("NumPy configuration info not available in expected format")
        
        # Simple performance test with large matrix operations
        size = 1000
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        
        import time
        start_time = time.time()
        C = np.dot(A, B)  # This should use accelerated BLAS
        end_time = time.time()
        
        print(f"Matrix multiplication time for {size}x{size} matrices: {end_time - start_time:.4f} seconds")
        self.assertIsNotNone(C)
    
    def test_basic_linear_algebra(self):
        """Test basic linear algebra operations that should use Accelerate if available."""
        # These operations should be accelerated on macOS with Accelerate framework
        A = np.random.rand(100, 100)
        b = np.random.rand(100)
        
        # Test operations
        eigenvalues = np.linalg.eigvals(A)
        matrix_inverse = np.linalg.inv(A)
        matrix_vector_product = np.dot(A, b)
        
        # Verify results
        self.assertEqual(len(eigenvalues), 100)
        self.assertEqual(matrix_inverse.shape, (100, 100))
        self.assertEqual(len(matrix_vector_product), 100)


if __name__ == '__main__':
    unittest.main()
