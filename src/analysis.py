import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
try:
    # Try relative import (when used as a package)
    from .oneapi_accelerator import AccelerationContext, accelerated_kmeans, ONEAPI_AVAILABLE
except ImportError:
    # Try absolute import (when run directly)
    from oneapi_accelerator import AccelerationContext, accelerated_kmeans, ONEAPI_AVAILABLE

# Set up logging
logger = logging.getLogger(__name__)


class InventoryAnalyzer:
    """
    Analyzes inventory data to identify stale or low-quality book listings.
    """
    
    def __init__(self, sales_data=None, inventory_data=None, use_acceleration=True):
        """
        Initialize the analyzer with sales and inventory data.
        
        Args:
            sales_data (pd.DataFrame, optional): DataFrame containing sales history.
            inventory_data (pd.DataFrame, optional): DataFrame containing inventory metadata.
            use_acceleration (bool, optional): Whether to use oneAPI acceleration if available. Defaults to True.
        """
        self.sales_data = sales_data
        self.inventory_data = inventory_data
        self.use_acceleration = use_acceleration and ONEAPI_AVAILABLE
        
        if self.use_acceleration:
            logger.info("oneAPI acceleration is enabled for analysis.")
        else:
            logger.info("oneAPI acceleration is disabled or not available.")
        
    def load_dummy_data(self):
        """
        Load dummy data for testing purposes.
        """
        # Create dummy sales data with pandas Timestamp objects
        now = pd.Timestamp.now()
        self.sales_data = pd.DataFrame({
            'SKU': ['B000FJS1B4', '0385472579', 'B07D6CTWPZ', 'B07CHNQLTF', '0451524934'],
            'Date': [
                now - pd.Timedelta(days=200),
                now - pd.Timedelta(days=3),
                now - pd.Timedelta(days=15),
                now - pd.Timedelta(days=45),
                now - pd.Timedelta(days=90)
            ],
            'Quantity': [1, 2, 1, 1, 1],
            'Price': [12.99, 15.99, 9.99, 7.99, 5.99]
        })
        
        # Create dummy inventory data
        self.inventory_data = pd.DataFrame({
            'SKU': ['B000FJS1B4', '0385472579', 'B07D6CTWPZ', 'B07CHNQLTF', '0451524934'],
            'Title': ['The Hobbit', 'Zen Mind, Beginner\'s Mind', 'Educated: A Memoir', 'Sapiens', '1984'],
            'ISBN': ['9780547928227', '9781590308493', '9780399590504', '9780062316097', '9780451524935'],
            'Price': [12.99, 15.99, 9.99, 7.99, 5.99],
            'Has_Image': [True, True, True, False, True]
        })
        
        return self
    
    def analyze(self):
        """
        Perform analysis on the inventory data.
        
        Returns:
            pd.DataFrame: Analysis results.
        """
        if self.sales_data is None or self.inventory_data is None:
            raise ValueError("Sales and inventory data must be loaded before analysis")
        
        # Merge sales and inventory data
        analysis_df = self.inventory_data.copy()
        
        # Calculate days since last sale
        now = pd.Timestamp.now()
        last_sale_dates = self.sales_data.groupby('SKU')['Date'].max().reset_index()
        
        # Calculate days since last sale for each SKU
        days_since = []
        for date in last_sale_dates['Date']:
            days_since.append((now - date).days)
        
        last_sale_dates['Days_Since_Last_Sale'] = days_since
        analysis_df = analysis_df.merge(last_sale_dates[['SKU', 'Days_Since_Last_Sale']], on='SKU', how='left')
        
        # Calculate sales in last 30 days
        thirty_days_ago = now - pd.Timedelta(days=30)
        recent_sales = self.sales_data[self.sales_data['Date'] >= thirty_days_ago]
        sales_30d = recent_sales.groupby('SKU')['Quantity'].sum().reset_index()
        sales_30d.rename(columns={'Quantity': 'Sales_Last_30d'}, inplace=True)
        analysis_df = analysis_df.merge(sales_30d[['SKU', 'Sales_Last_30d']], on='SKU', how='left')
        
        # Fill NaN values
        analysis_df['Sales_Last_30d'] = analysis_df['Sales_Last_30d'].fillna(0)
        
        # Apply stale detection logic
        analysis_df['Is_Stale'] = (analysis_df['Days_Since_Last_Sale'] > 120) | \
                                 ((analysis_df['Sales_Last_30d'] == 0) & (analysis_df['Days_Since_Last_Sale'] > 90))
        
        # Apply bad metadata detection
        bad_metadata_conditions = [analysis_df['Title'].str.len() < 5]
        
        # Add ISBN check if column exists
        if 'ISBN' in analysis_df.columns:
            bad_metadata_conditions.append(analysis_df['ISBN'].isna())
        
        # Add image check if column exists
        if 'Has_Image' in analysis_df.columns:
            bad_metadata_conditions.append(~analysis_df['Has_Image'])
        
        # Add price check if column exists
        if 'Price' in analysis_df.columns:
            bad_metadata_conditions.append(analysis_df['Price'] < 2.0)
        
        # Combine conditions
        analysis_df['Bad_Metadata'] = False
        for condition in bad_metadata_conditions:
            analysis_df['Bad_Metadata'] = analysis_df['Bad_Metadata'] | condition
        
        # Create flag and notes
        analysis_df['Flag'] = '✅'
        analysis_df.loc[analysis_df['Is_Stale'] | analysis_df['Bad_Metadata'], 'Flag'] = '⚠️'
        
        analysis_df['Notes'] = ''
        analysis_df.loc[analysis_df['Is_Stale'], 'Notes'] = 'Reprice or remove'
        analysis_df.loc[analysis_df['Bad_Metadata'], 'Notes'] = analysis_df.loc[analysis_df['Bad_Metadata'], 'Notes'] + 'Fix metadata issues'
        analysis_df.loc[(~analysis_df['Is_Stale']) & (~analysis_df['Bad_Metadata']), 'Notes'] = 'Good seller'
        
        # Select and rename columns for final output
        result_df = analysis_df[['SKU', 'Title', 'Sales_Last_30d', 'Days_Since_Last_Sale', 'Flag', 'Notes']]
        result_df = result_df.rename(columns={'Sales_Last_30d': 'Sales Last 30d', 'Days_Since_Last_Sale': 'Days Since Last Sale'})
        
        return result_df


    def cluster_inventory(self, n_clusters=3):
        """
        Cluster inventory items based on sales patterns and metadata.
        
        Args:
            n_clusters (int, optional): Number of clusters. Defaults to 3.
            
        Returns:
            pd.DataFrame: DataFrame with cluster assignments.
        """
        if self.sales_data is None or self.inventory_data is None:
            raise ValueError("Sales and inventory data must be loaded before clustering")
        
        # Prepare features for clustering
        features = self._prepare_features_for_clustering()
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Run clustering with oneAPI acceleration if available
        if self.use_acceleration:
            logger.info("Using oneAPI acceleration for K-means clustering.")
            centroids, labels = accelerated_kmeans(
                scaled_features, 
                n_clusters=n_clusters,
                max_iterations=100,
                seed=42
            )
        else:
            # Use standard scikit-learn
            from sklearn.cluster import KMeans
            logger.info("Using standard scikit-learn for K-means clustering.")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(scaled_features)
        
        # Add cluster labels to inventory data
        result = self.inventory_data.copy()
        result['Cluster'] = labels
        
        # Add cluster descriptions
        cluster_descriptions = self._describe_clusters(result, features.columns)
        for cluster_id, description in cluster_descriptions.items():
            result.loc[result['Cluster'] == cluster_id, 'Cluster_Description'] = description
        
        return result
    
    def _prepare_features_for_clustering(self):
        """
        Prepare features for clustering.
        
        Returns:
            pd.DataFrame: Features for clustering.
        """
        if self.sales_data is None or self.inventory_data is None:
            raise ValueError("Sales and inventory data must be loaded before clustering")
            
        # Start with inventory data
        features = self.inventory_data.copy()
        
        # Get current timestamp
        now = pd.Timestamp.now()
        
        # Add sales aggregates
        sales_last_30d = self.sales_data[
            self.sales_data['Date'] >= now - pd.Timedelta(days=30)
        ].groupby('SKU')['Quantity'].sum().reset_index()
        sales_last_30d.rename(columns={'Quantity': 'Sales_Last_30d'}, inplace=True)
        
        sales_last_90d = self.sales_data[
            self.sales_data['Date'] >= now - pd.Timedelta(days=90)
        ].groupby('SKU')['Quantity'].sum().reset_index()
        sales_last_90d.rename(columns={'Quantity': 'Sales_Last_90d'}, inplace=True)
        
        # Calculate days since last sale
        last_sale_dates = self.sales_data.groupby('SKU')['Date'].max().reset_index()
        
        # Calculate days since last sale for each SKU
        days_since = []
        for date in last_sale_dates['Date']:
            days_since.append((now - date).days)
        
        last_sale_dates['Days_Since_Last_Sale'] = days_since
        
        # Merge sales data with inventory
        features = features.merge(sales_last_30d, on='SKU', how='left')
        features = features.merge(sales_last_90d, on='SKU', how='left')
        features = features.merge(
            last_sale_dates[['SKU', 'Days_Since_Last_Sale']], on='SKU', how='left'
        )
        
        # Fill NaN values
        features['Sales_Last_30d'] = features['Sales_Last_30d'].fillna(0)
        features['Sales_Last_90d'] = features['Sales_Last_90d'].fillna(0)
        features['Days_Since_Last_Sale'] = features['Days_Since_Last_Sale'].fillna(365)  # Assume 1 year if no sales
        
        # Convert categorical variables to numeric if they exist
        if 'Has_Image' in features.columns:
            features['Has_Image'] = features['Has_Image'].astype(int)
        else:
            features['Has_Image'] = 0  # Default value if column doesn't exist
            
        if 'ISBN' in features.columns:
            features['Has_ISBN'] = (~features['ISBN'].isna()).astype(int)
        else:
            features['Has_ISBN'] = 0  # Default value if column doesn't exist
        
        # Select numerical features for clustering
        numerical_features = features[[
            'Price', 'Sales_Last_30d', 'Sales_Last_90d', 
            'Days_Since_Last_Sale', 'Has_Image', 'Has_ISBN'
        ]]
        
        return numerical_features
    
    def _describe_clusters(self, clustered_data, feature_names):
        """
        Generate descriptions for each cluster.
        
        Args:
            clustered_data (pd.DataFrame): Data with cluster assignments.
            feature_names (list): Names of features used for clustering.
            
        Returns:
            dict: Dictionary mapping cluster IDs to descriptions.
        """
        descriptions = {}
        
        # Get cluster statistics
        cluster_stats = clustered_data.groupby('Cluster').agg({
            'Price': 'mean',
            'Sales_Last_30d': 'mean',
            'Days_Since_Last_Sale': 'mean',
            'Has_Image': 'mean',
            'Has_ISBN': 'mean',
            'SKU': 'count'
        }).reset_index()
        
        # Generate descriptions
        for _, row in cluster_stats.iterrows():
            cluster_id = row['Cluster']
            count = row['SKU']
            
            if row['Sales_Last_30d'] > 5 and row['Days_Since_Last_Sale'] < 30:
                category = "High Performers"
            elif row['Sales_Last_30d'] < 1 and row['Days_Since_Last_Sale'] > 90:
                category = "Stale Inventory"
            elif row['Has_Image'] < 0.5 or row['Has_ISBN'] < 0.5:
                category = "Poor Metadata"
            else:
                category = "Average Performers"
            
            descriptions[cluster_id] = f"{category} (n={count})"
        
        return descriptions


def run_analysis(use_acceleration=True):
    """
    Run the inventory analysis and return the results.
    
    Args:
        use_acceleration (bool, optional): Whether to use oneAPI acceleration if available. Defaults to True.
    
    Returns:
        pd.DataFrame: Analysis results.
    """
    # Create analyzer and load dummy data
    analyzer = InventoryAnalyzer(use_acceleration=use_acceleration).load_dummy_data()
    
    # Basic analysis
    results = analyzer.analyze()
    
    # Advanced analysis (if enough data)
    if analyzer.inventory_data is not None and len(analyzer.inventory_data) >= 5:
        try:
            # Determine number of clusters (at least 2, at most 3, but not more than data points - 1)
            n_clusters = min(3, max(2, len(analyzer.inventory_data) - 1))
            
            # Cluster inventory
            clustered_data = analyzer.cluster_inventory(n_clusters=n_clusters)
            
            # Add cluster information to results
            cluster_info = clustered_data[['SKU', 'Cluster', 'Cluster_Description']]
            results = results.merge(cluster_info, on='SKU', how='left')
            
            logger.info("Advanced analysis with clustering completed successfully.")
        except Exception as e:
            logger.error(f"Advanced analysis failed: {str(e)}")
            logger.debug(f"Error details: {str(e)}", exc_info=True)
    
    return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run analysis
    results = run_analysis(use_acceleration=True)
    print(results)
