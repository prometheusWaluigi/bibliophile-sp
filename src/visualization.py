import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


class InventoryVisualizer:
    """
    Visualizes inventory analysis results.
    """
    
    def __init__(self, data=None):
        """
        Initialize the visualizer with data.
        
        Args:
            data (pd.DataFrame, optional): DataFrame containing analysis results.
        """
        self.data = data
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self, file_path):
        """
        Load data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file.
            
        Returns:
            InventoryVisualizer: Self for method chaining.
        """
        self.data = pd.read_csv(file_path)
        return self
    
    def plot_sales_distribution(self, save=True):
        """
        Plot the distribution of sales in the last 30 days.
        
        Args:
            save (bool, optional): Whether to save the plot to a file. Defaults to True.
            
        Returns:
            plt.Figure: The matplotlib figure.
        """
        if self.data is None:
            raise ValueError("No data available. Load data first.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot sales distribution
        sales_counts = self.data['Sales Last 30d'].value_counts().sort_index()
        ax.bar(sales_counts.index, sales_counts.values, color='skyblue')
        
        # Add labels and title
        ax.set_xlabel('Sales in Last 30 Days')
        ax.set_ylabel('Number of Items')
        ax.set_title('Distribution of Sales in Last 30 Days')
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the plot if requested
        if save:
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'sales_distribution.png'))
        
        return fig
    
    def plot_stale_inventory(self, save=True):
        """
        Plot a scatter plot of days since last sale vs. sales in last 30 days.
        
        Args:
            save (bool, optional): Whether to save the plot to a file. Defaults to True.
            
        Returns:
            plt.Figure: The matplotlib figure.
        """
        if self.data is None:
            raise ValueError("No data available. Load data first.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot
        colors = ['red' if flag == '⚠️' else 'green' for flag in self.data['Flag']]
        ax.scatter(self.data['Days Since Last Sale'], self.data['Sales Last 30d'], 
                  c=colors, alpha=0.7, s=100)
        
        # Add labels and title
        ax.set_xlabel('Days Since Last Sale')
        ax.set_ylabel('Sales in Last 30 Days')
        ax.set_title('Stale Inventory Analysis')
        
        # Add grid
        ax.grid(linestyle='--', alpha=0.7)
        
        # Add a legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Good Seller'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Needs Attention')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        # Add threshold lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=120, color='gray', linestyle='--', alpha=0.5)
        
        # Add annotations for thresholds
        ax.text(120, max(self.data['Sales Last 30d']) * 0.9, 'Stale Threshold (120 days)', 
                rotation=90, verticalalignment='top')
        
        # Save the plot if requested
        if save:
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'stale_inventory.png'))
        
        return fig
    
    def plot_flag_distribution(self, save=True):
        """
        Plot a pie chart of the distribution of flags.
        
        Args:
            save (bool, optional): Whether to save the plot to a file. Defaults to True.
            
        Returns:
            plt.Figure: The matplotlib figure.
        """
        if self.data is None:
            raise ValueError("No data available. Load data first.")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Count flags
        flag_counts = self.data['Flag'].value_counts()
        
        # Create labels
        labels = ['Needs Attention' if flag == '⚠️' else 'Good Seller' for flag in flag_counts.index]
        
        # Create pie chart
        ax.pie(flag_counts.values, labels=labels, autopct='%1.1f%%', startangle=90, 
              colors=['red', 'green'], explode=[0.1, 0], shadow=True)
        
        # Add title
        ax.set_title('Inventory Quality Distribution')
        
        # Save the plot if requested
        if save:
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'flag_distribution.png'))
        
        return fig
    
    def generate_dashboard(self):
        """
        Generate a dashboard with all plots.
        
        Returns:
            list: List of generated figure objects.
        """
        figures = []
        
        figures.append(self.plot_sales_distribution())
        figures.append(self.plot_stale_inventory())
        figures.append(self.plot_flag_distribution())
        
        print(f"✅ Dashboard generated. Plots saved to {self.output_dir}/")
        
        return figures


def visualize_results(data_path="output/inventory_analysis.csv"):
    """
    Visualize the inventory analysis results.
    
    Args:
        data_path (str, optional): Path to the CSV file with analysis results.
                                  Defaults to "output/inventory_analysis.csv".
    
    Returns:
        list: List of generated figure objects.
    """
    visualizer = InventoryVisualizer().load_data(data_path)
    return visualizer.generate_dashboard()


if __name__ == "__main__":
    # Check if the analysis results exist
    if os.path.exists("output/inventory_analysis.csv"):
        visualize_results()
    else:
        print("❌ Analysis results not found. Run the analysis first.")
