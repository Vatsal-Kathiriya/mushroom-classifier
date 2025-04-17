import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import base64

# Import project modules
from src.visualization.core import (
    plot_class_distribution, plot_categorical_features, plot_numerical_features,
    plot_pca_visualization, create_output_dir
)
from src.models.training import plot_model_comparison
from src.feature_analysis import (
    plot_feature_importance, calculate_permutation_importance, mutual_info_feature_selection
)
from src.evaluation import (
    plot_confusion_matrix, plot_roc_curve, plot_learning_curve, plot_coefficients
)

def create_all_visualizations():
    """Generate all possible visualizations for the dashboard"""
    print("Generating all visualizations...")
    
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Load data
    try:
        df = pd.read_csv('data/secondary_data.csv', delimiter=';')
        print(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    try:
        # Class distribution
        print("Generating class distribution...")
        plot_class_distribution(df, save_path='visualizations/class_distribution.png')
        
        # Feature distributions
        categorical_cols = df.select_dtypes(include=['object']).columns.drop('class').tolist()
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if categorical_cols:
            print("Generating categorical features plot...")
            # Limit to top categorical features to avoid overcrowding
            top_cat_cols = categorical_cols[:min(6, len(categorical_cols))]
            plot_categorical_features(df, top_cat_cols, save_path='visualizations/categorical_features.png')
        
        if numerical_cols:
            print("Generating numerical features plot...")
            plot_numerical_features(df, numerical_cols, save_path='visualizations/numerical_features.png')
        
        # PCA visualization
        print("Generating PCA visualization...")
        features = df.columns.drop('class').tolist()
        plot_pca_visualization(df, features, save_path='visualizations/pca_visualization.png')
        
        # Load model and feature data if available
        model_path = 'models/refined_feature_model.pkl'
        feature_names_path = 'models/feature_names.pkl'
        
        if os.path.exists(model_path) and os.path.exists(feature_names_path):
            print("Found model and feature names, generating model visualizations...")
            model = joblib.load(model_path)
            feature_names = joblib.load(feature_names_path)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                print("Generating feature importance plot...")
                plot_feature_importance(model, feature_names, save_path='visualizations/feature_importance.png')
        
        # Model comparison if results exist
        results_path = 'models/model_results.pkl'
        if os.path.exists(results_path):
            print("Generating model comparison chart...")
            results_df = joblib.load(results_path)
            plot_model_comparison(results_df, save_path='visualizations/model_comparison.png')
        else:
            print("Warning: model_results.pkl not found. Cannot generate model comparison.")
        
        print("All visualizations generated successfully!")
        return True
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return False

if __name__ == "__main__":
    create_all_visualizations()