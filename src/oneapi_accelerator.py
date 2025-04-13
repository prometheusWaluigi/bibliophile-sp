"""
oneAPI Accelerator module for BibliophileSP.

This module provides functions to accelerate analysis using Intel's oneAPI or Apple's Accelerate framework.
"""

import numpy as np
import pandas as pd
import logging
import platform
import os
import sys

# Set up logging
logger = logging.getLogger(__name__)

# Detect platform
PLATFORM = platform.system()
PROCESSOR = platform.processor()
IS_MAC = PLATFORM == "Darwin"
IS_INTEL_CPU = "intel" in PROCESSOR.lower() or "x86_64" in PROCESSOR.lower()
IS_ARM_MAC = IS_MAC and platform.machine() == "arm64"  # M1/M2 Mac

# Check for environment variables that might indicate oneAPI availability
ONEAPI_ENV_VARS = ["MKLROOT", "ONEAPI_ROOT", "DAAL_ROOT"]
HAS_ONEAPI_ENV = any(var in os.environ for var in ONEAPI_ENV_VARS)

# Log platform information
logger.info(f"Platform: {PLATFORM}, Processor: {PROCESSOR}")
if IS_MAC:
    logger.info(f"Running on macOS with {'Apple Silicon (M1/M2/M3)' if platform.machine() == 'arm64' else 'Intel'} chip")

# Try to import oneAPI-specific modules
ONEAPI_AVAILABLE = False
APPLE_ACCELERATE_AVAILABLE = False

# Check for oneAPI
try:
    import daal4py
    import sklearnex
    ONEAPI_AVAILABLE = True
    logger.info("Intel oneAPI modules found. oneAPI acceleration is available.")
except ImportError:
    if HAS_ONEAPI_ENV and IS_INTEL_CPU:
        logger.warning("oneAPI environment detected but modules not found. Try installing with: pip install daal4py scikit-learn-intelex")
    else:
        logger.info("Intel oneAPI modules not found.")

# Check for Apple Accelerate framework on Mac
if IS_MAC:
    try:
        # Check if numpy is using Accelerate
        np_config = np.__config__
        if hasattr(np_config, 'blas_opt_info') and 'accelerate' in str(np_config.blas_opt_info).lower():
            APPLE_ACCELERATE_AVAILABLE = True
            logger.info("NumPy is using Apple's Accelerate framework.")
        elif IS_ARM_MAC:
            # On M1/M2 Macs, numpy often uses Accelerate automatically
            APPLE_ACCELERATE_AVAILABLE = True
            logger.info("Running on Apple Silicon - Accelerate framework likely in use.")
    except (AttributeError, ImportError) as e:
        logger.info(f"Unable to detect Apple Accelerate framework: {str(e)}")
            
# Set the appropriate acceleration flag
ACCELERATION_AVAILABLE = ONEAPI_AVAILABLE or APPLE_ACCELERATE_AVAILABLE

if not ACCELERATION_AVAILABLE:
    logger.warning("No acceleration available. Falling back to standard implementations.")


def enable_acceleration():
    """
    Enable oneAPI acceleration if available.
    
    Returns:
        bool: True if acceleration was enabled, False otherwise.
    """
    if ONEAPI_AVAILABLE:
        try:
            # Patch scikit-learn with Intel extensions
            sklearnex.patch_sklearn()
            logger.info("scikit-learn patched with Intel extensions.")
            return True
        except Exception as e:
            logger.error(f"Failed to enable oneAPI acceleration: {str(e)}")
            return False
    else:
        logger.warning("oneAPI modules not available. Acceleration not enabled.")
        return False


def disable_acceleration():
    """
    Disable oneAPI acceleration if it was enabled.
    
    Returns:
        bool: True if acceleration was disabled, False otherwise.
    """
    if ONEAPI_AVAILABLE:
        try:
            # Unpatch scikit-learn
            sklearnex.unpatch_sklearn()
            logger.info("scikit-learn unpatched from Intel extensions.")
            return True
        except Exception as e:
            logger.error(f"Failed to disable oneAPI acceleration: {str(e)}")
            return False
    else:
        logger.warning("oneAPI modules not available. No acceleration to disable.")
        return False


def accelerated_kmeans(data, n_clusters=8, max_iterations=100, seed=None):
    """
    Run K-means clustering with oneAPI acceleration if available.
    
    Args:
        data (numpy.ndarray): Input data for clustering.
        n_clusters (int, optional): Number of clusters. Defaults to 8.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 100.
        seed (int, optional): Random seed. Defaults to None.
        
    Returns:
        tuple: (centroids, labels) - cluster centroids and labels for each data point.
    """
    if ONEAPI_AVAILABLE:
        try:
            # Use daal4py for K-means
            logger.info("Using daal4py for K-means clustering.")
            
            # Initialize the algorithm
            init_algo = daal4py.kmeans_init(
                nClusters=n_clusters,
                method="randomDense",
                seed=seed if seed is not None else 777
            )
            init_result = init_algo.compute(data)
            
            # Compute K-means
            algo = daal4py.kmeans(
                nClusters=n_clusters,
                maxIterations=max_iterations
            )
            result = algo.compute(data, init_result.centroids)
            
            # Get results
            centroids = result.centroids
            labels = result.assignments.flatten()
            
            return centroids, labels
        except Exception as e:
            logger.error(f"daal4py K-means failed: {str(e)}. Falling back to scikit-learn.")
    
    # Fall back to scikit-learn (which may still be accelerated if patched)
    from sklearn.cluster import KMeans
    logger.info("Using scikit-learn for K-means clustering.")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iterations,
        random_state=seed
    )
    labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_
    
    return centroids, labels


def accelerated_pca(data, n_components=2):
    """
    Run PCA dimensionality reduction with oneAPI acceleration if available.
    
    Args:
        data (numpy.ndarray): Input data for PCA.
        n_components (int, optional): Number of components. Defaults to 2.
        
    Returns:
        numpy.ndarray: Transformed data.
    """
    if ONEAPI_AVAILABLE:
        try:
            # Use daal4py for PCA
            logger.info("Using daal4py for PCA.")
            
            # Initialize the algorithm
            algo = daal4py.pca(
                method="correlationDense",
                resultsToCompute="mean|variance|eigenvalue",
                isDeterministic=True
            )
            
            # Compute PCA
            result = algo.compute(data)
            
            # Transform the data
            transform_algo = daal4py.pca_transform(
                nComponents=n_components
            )
            transform_result = transform_algo.compute(data, result.eigenvectors)
            
            return transform_result.transformedData
        except Exception as e:
            logger.error(f"daal4py PCA failed: {str(e)}. Falling back to scikit-learn.")
    
    # Fall back to scikit-learn (which may still be accelerated if patched)
    from sklearn.decomposition import PCA
    logger.info("Using scikit-learn for PCA.")
    
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    
    return transformed_data


def accelerated_tsne(data, n_components=2, perplexity=30.0, learning_rate=200.0):
    """
    Run t-SNE dimensionality reduction.
    Note: daal4py doesn't have a direct t-SNE implementation, so we use scikit-learn.
    
    Args:
        data (numpy.ndarray): Input data for t-SNE.
        n_components (int, optional): Number of components. Defaults to 2.
        perplexity (float, optional): Perplexity parameter. Defaults to 30.0.
        learning_rate (float, optional): Learning rate. Defaults to 200.0.
        
    Returns:
        numpy.ndarray: Transformed data.
    """
    # Enable acceleration if available
    if ONEAPI_AVAILABLE:
        enable_acceleration()
    
    # Use scikit-learn for t-SNE (which may be accelerated if patched)
    from sklearn.manifold import TSNE
    logger.info("Using scikit-learn for t-SNE.")
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=42
    )
    transformed_data = tsne.fit_transform(data)
    
    # Disable acceleration if it was enabled
    if ONEAPI_AVAILABLE:
        disable_acceleration()
    
    return transformed_data


def accelerated_dbscan(data, eps=0.5, min_samples=5):
    """
    Run DBSCAN clustering with oneAPI acceleration if available.
    
    Args:
        data (numpy.ndarray): Input data for clustering.
        eps (float, optional): The maximum distance between two samples for them to be considered as in the same neighborhood. Defaults to 0.5.
        min_samples (int, optional): The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 5.
        
    Returns:
        numpy.ndarray: Cluster labels for each data point.
    """
    # Enable acceleration if available
    if ONEAPI_AVAILABLE:
        enable_acceleration()
    
    # Use scikit-learn for DBSCAN (which may be accelerated if patched)
    from sklearn.cluster import DBSCAN
    logger.info("Using scikit-learn for DBSCAN.")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    
    # Disable acceleration if it was enabled
    if ONEAPI_AVAILABLE:
        disable_acceleration()
    
    return labels


class AccelerationContext:
    """
    Context manager for oneAPI acceleration.
    
    Example:
        ```python
        with AccelerationContext():
            # Code that uses scikit-learn will be accelerated
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(data)
        # Acceleration is automatically disabled after the context
        ```
    """
    
    def __enter__(self):
        """Enable acceleration when entering the context."""
        self.acceleration_enabled = enable_acceleration()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Disable acceleration when exiting the context."""
        if self.acceleration_enabled:
            disable_acceleration()
