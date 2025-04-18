import os
import io
import base64
import joblib  # Add missing import
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Standard color scheme for consistent visualization
EDIBLE_COLOR = '#4CAF50'     # Green for edible mushrooms
POISONOUS_COLOR = '#F44336'  # Red for poisonous mushrooms

class VisualizationManager:
    """Unified manager for all visualization functions with consistent styling"""
    
    def __init__(self, df=None, output_dir="visualizations"):
        self.df = df
        self.output_dir = create_output_dir(output_dir)
    
    def set_dataframe(self, df):
        """Set the dataframe to use for visualizations"""
        self.df = df
        return self
    
    def plot_distribution(self, df=None, feature=None, save=False):
        """Plot class or feature distribution"""
        df = df if df is not None else self.df
        if df is None:
            raise ValueError("No dataframe provided")
            
        return plot_class_distribution(df, show=not save) if feature is None else plot_feature_distribution(df, feature)
    
    def plot_numerical_features(self, df=None, features=None):
        """Plot distribution of numerical features"""
        df = df if df is not None else self.df
        if df is None:
            raise ValueError("No dataframe provided")
            
        return plot_numerical_features(df, features)
    
    def plot_categorical_features(self, df=None, features=None):
        """Plot distribution of categorical features"""
        df = df if df is not None else self.df
        if df is None:
            raise ValueError("No dataframe provided")
            
        return plot_categorical_features(df, features)
    
    def plot_correlation_matrix(self, df=None, features=None):
        """Plot correlation matrix for numerical features"""
        df = df if df is not None else self.df
        if df is None:
            raise ValueError("No dataframe provided")
            
        return plot_correlation_matrix(df, features)
    
    def plot_model_comparison(self, results_df):
        """Plot model comparison visualization"""
        from src.models.training import plot_model_comparison
        return plot_model_comparison(results_df)
    
    def evaluate_and_visualize_model(self, model, X_test, y_test, feature_names):
        """Evaluate model and create visualizations"""
        from src.evaluation import (
            evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_coefficients
        )
        
        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Create visualizations
        plot_confusion_matrix(y_test, metrics['y_pred'])
        
        if metrics['y_pred_proba'] is not None:
            plot_roc_curve(y_test, metrics['y_pred_proba'])
        
        # Plot coefficients or feature importance
        if hasattr(model, 'coef_'):
            plot_coefficients(model, feature_names)
        elif hasattr(model, 'feature_importances_'):
            from src.feature_analysis import plot_feature_importance
            plot_feature_importance(model, feature_names)
        
        return metrics
    
    def plot_learning_curve(self, model, X, y):
        """Plot learning curve for model"""
        from src.evaluation import plot_learning_curve
        return plot_learning_curve(model, X, y)
    
    def get_image_base64(self, plt_obj_or_path):
        """Convert matplotlib plot or saved image path to base64 encoded string"""
        if isinstance(plt_obj_or_path, str) and os.path.exists(plt_obj_or_path):
            # If it's a path to an existing file
            with open(plt_obj_or_path, 'rb') as img_file:
                img_data = img_file.read()
            return base64.b64encode(img_data).decode()
        else:
            # If it's a matplotlib figure
            img = io.BytesIO()  # Fixed: using BytesIO
            plt_obj_or_path.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plt_obj_or_path.close()
            return base64.b64encode(img.getvalue()).decode()
    
    def create_dashboard_visualizations(self):
        """Generate all visualizations for the web dashboard"""
        viz_data = {}
        viz_errors = {}
        
        try:
            # Load the data
            df = pd.read_csv('data/secondary_data.csv', delimiter=';')
            
            # Load model comparison results if available
            try:
                # Try to load model comparison from file first
                comparison_path = 'visualizations/model_comparison.png'
                if os.path.exists(comparison_path):
                    with open(comparison_path, 'rb') as f:
                        viz_data['model_comparison'] = base64.b64encode(f.read()).decode('utf-8')
                        print(f"Successfully loaded model_comparison from {comparison_path}")
                else:
                    # Generate a new comparison if we can find results
                    results_df_path = 'models/model_results.pkl'
                    if os.path.exists(results_df_path):
                        results_df = joblib.load(results_df_path)
                        comparison_path = plot_model_comparison(results_df, save_path='visualizations/model_comparison.png')
                        with open(comparison_path, 'rb') as f:
                            viz_data['model_comparison'] = base64.b64encode(f.read()).decode('utf-8')
                            print(f"Successfully generated and loaded model_comparison")
            except Exception as e:
                print(f"Error generating model comparison: {e}")
                viz_errors['model_comparison'] = str(e)
            
            # Instead of generating visualizations in the request thread,
            # load pre-generated images from the filesystem
            visualization_files = {
                'model_feature_importance_image': 'visualizations/feature_importance.png',  # Renamed for clarity
                'class_distribution': 'visualizations/class_distribution.png',
                'categorical_features': 'visualizations/categorical_features.png',
                'numerical_features': 'visualizations/numerical_features.png',
                'pca_visualization': 'visualizations/pca_visualization.png',
                'permutation_importance': 'visualizations/permutation_importance.png',
                'confusion_matrix': 'visualizations/confusion_matrix.png',
                'roc_curve': 'visualizations/roc_curve.png',
                'learning_curve': 'visualizations/learning_curve.png',
                'mutual_info': 'visualizations/mutual_info.png',
                'feature_selection_curve': 'models/feature_selection_curve.png'
            }
            
            # Add inside the for-loop in create_dashboard_visualizations method where you load visualizations
            for key, filepath in visualization_files.items():
                try:
                    if os.path.exists(filepath):
                        with open(filepath, 'rb') as f:
                            viz_data[key] = base64.b64encode(f.read()).decode('utf-8')
                            print(f"Successfully loaded {key} from {filepath}")
                    else:
                        print(f"WARNING: Visualization file not found: {filepath}")
                        viz_errors[key] = f"File not found: {filepath}"
                except Exception as e:
                    print(f"ERROR loading {key} visualization from {filepath}: {e}")
                    viz_errors[key] = str(e)
            
            # Load feature correlations visualization if it exists
            corr_path = 'visualizations/feature_correlations.png'
            if os.path.exists(corr_path):
                with open(corr_path, 'rb') as f:
                    viz_data['feature_correlations'] = base64.b64encode(f.read()).decode('utf-8')
            else:
                viz_errors['feature_correlations'] = 'feature_correlations.png not found'
            
            # --- Load Key Feature Distributions ---
            top_features_path = os.path.join(self.output_dir, 'top_features.png')
            if os.path.exists(top_features_path):
                try:
                    with open(top_features_path, "rb") as image_file:
                        viz_data['top_features'] = base64.b64encode(image_file.read()).decode('utf-8')
                    print("Loaded top_features.png for dashboard.")
                except Exception as e:
                    if 'errors' not in viz_data:
                        viz_data['errors'] = {}
                    viz_data['errors']['top_features'] = f"Error reading top_features.png: {e}"
                    print(f"Error reading top_features.png: {e}")
            else:
                if 'errors' not in viz_data:
                    viz_data['errors'] = {}
                viz_data['errors']['top_features'] = "top_features.png not found."
                print("Warning: top_features.png not found.")
            
            # --- Load Feature Correlation Matrix ---
            corr_matrix_path = os.path.join(self.output_dir, 'extended_correlation_matrix.png')
            if os.path.exists(corr_matrix_path):
                try:
                    with open(corr_matrix_path, "rb") as image_file:
                        viz_data['correlation_matrix'] = base64.b64encode(image_file.read()).decode('utf-8')
                    print("Loaded extended_correlation_matrix.png for dashboard.")
                except Exception as e:
                    viz_data['errors']['correlation_matrix'] = f"Error reading extended_correlation_matrix.png: {e}"
                    print(f"Error reading extended_correlation_matrix.png: {e}")
            else:
                # Fall back to original correlation matrix or heatmap
                original_matrix_path = os.path.join(self.output_dir, 'correlation_matrix.png')
                if os.path.exists(original_matrix_path):
                    try:
                        with open(original_matrix_path, "rb") as image_file:
                            viz_data['correlation_matrix'] = base64.b64encode(image_file.read()).decode('utf-8')
                        print("Loaded correlation_matrix.png for dashboard.")
                    except Exception as e:
                        viz_data['errors']['correlation_matrix'] = f"Error reading correlation_matrix.png: {e}"
                        print(f"Error reading correlation_matrix.png: {e}")
                else:
                    # Final fallback to heatmap
                    corr_heatmap_path = os.path.join(self.output_dir, 'correlation_heatmap.png')
                    if os.path.exists(corr_heatmap_path):
                        try:
                            with open(corr_heatmap_path, "rb") as image_file:
                                viz_data['correlation_matrix'] = base64.b64encode(image_file.read()).decode('utf-8')
                            print("Loaded correlation_heatmap.png as fallback for dashboard.")
                        except Exception as e:
                            viz_data['errors']['correlation_matrix'] = f"Error reading correlation_heatmap.png: {e}"
                            print(f"Error reading correlation_heatmap.png: {e}")
                    else:
                        viz_data['errors']['correlation_matrix'] = "No correlation matrix images found."
            
            # Also check for the feature_importance.png in the static directory - use clear naming
            static_feature_importance = 'static/generated_viz/feature_importance.png'
            if os.path.exists(static_feature_importance):
                try:
                    with open(static_feature_importance, 'rb') as f:
                        viz_data['static_feature_importance_image'] = base64.b64encode(f.read()).decode('utf-8')
                        print(f"Successfully loaded static feature importance from {static_feature_importance}")
                except Exception as e:
                    print(f"ERROR loading static feature importance: {e}")
                    viz_errors['static_feature_importance_image'] = str(e)
            
            # Create a primary feature importance key that pulls from the best available source
            if 'model_feature_importance_image' in viz_data and viz_data['model_feature_importance_image']:
                viz_data['primary_feature_importance'] = viz_data['model_feature_importance_image']
                print("Using model feature importance as primary source")
            elif 'static_feature_importance_image' in viz_data and viz_data['static_feature_importance_image']:
                viz_data['primary_feature_importance'] = viz_data['static_feature_importance_image']
                print("Using static feature importance as primary source")
            else:
                print("WARNING: No feature importance image found in any location")
                viz_data['primary_feature_importance'] = ""
            
            # Add key features data for the field identification table
            try:
                # Load top features
                top_features_path = 'models/top_features.pkl'
                if os.path.exists(top_features_path):
                    top_features = joblib.load(top_features_path)
                    viz_data['key_features'] = top_features[:7]  # Limit to top 7 features
                    
                    # Create dictionaries for feature tips and importance levels - renamed to avoid collision
                    feature_tips = {
                        'spore-print-color': 'Place cap on paper overnight to observe spore color; white spores often indicate caution',
                        'odor': 'Smell the mushroom: sweet or almond scents tend toward edible, while foul odors often indicate toxicity',
                        'gill-color': 'Observe under the cap: certain gill colors correlate strongly with edibility or toxicity',
                        'ring-type': 'Examine the ring on stem (when present): different types correlate with certain species',
                        'gill-size': 'Note the relative size of the gills compared to cap thickness',
                        'stalk-surface-above-ring': 'Examine stem texture above the ring for distinctive patterns',
                        'habitat': 'Note where the mushroom is growing; many species have specific habitat preferences',
                        'does-bruise-or-bleed': 'Check if the mushroom changes color or releases liquid when cut or bruised',
                        'cap-color': 'Observe the color of the mushroom cap which can be distinctive for certain species',
                        'cap-shape': 'Note the shape of the mushroom cap which can help identify specific species',
                        'season': 'Consider the time of year when the mushroom was found as species appear in specific seasons',
                        'stem-height': 'Measure the height of the stem relative to cap diameter',
                        'stem-width': 'Measure the thickness of the stem which varies between species',
                        'cap-diameter': 'Measure the cap diameter which can help identify mature versus young specimens',
                        'stem-color': 'Note the color of the stem which often differs from the cap in certain species',
                        'cap-surface': 'Observe the texture of the cap surface which can be distinctive'
                    }
                    
                    # Dynamically assign importance levels based on feature rank - renamed to avoid collision
                    feature_importance_levels = {}
                    for i, feature in enumerate(viz_data['key_features']):
                        if i == 0:
                            feature_importance_levels[feature] = 'Very High'
                        elif i <= 2:
                            feature_importance_levels[feature] = 'High'
                        elif i <= 3:
                            feature_importance_levels[feature] = 'Medium-High'
                        elif i <= 5:
                            feature_importance_levels[feature] = 'Medium'
                        else:
                            feature_importance_levels[feature] = 'Medium-Low'
                    
                    viz_data['feature_tips'] = feature_tips
                    viz_data['feature_importance_levels'] = feature_importance_levels  # Renamed for clarity
                    
                else:
                    print("Top features file not found. Using default features for visualization.")
                    viz_errors['key_features'] = "Top features file not found"
            except Exception as e:
                print(f"Error loading key features data: {e}")
                viz_errors['key_features'] = str(e)
            
            # Ensure all expected keys exist with at least empty values to prevent "NoneType has no attribute 'get'" errors
            required_keys = [
                'primary_feature_importance',  # Updated to new naming
                'model_feature_importance_image',  # Updated to new naming
                'static_feature_importance_image',  # Updated to new naming
                'permutation_importance',
                'mutual_info', 
                'feature_correlations', 
                'correlation_matrix'
            ]
            
            for key in required_keys:
                if key not in viz_data:
                    viz_data[key] = ""
                    print(f"Added empty placeholder for {key} to avoid template errors")
            
            # Legacy support - provide backward compatibility with old template references
            viz_data['feature_importance'] = viz_data['primary_feature_importance']  # For old templates
            viz_data['feature_importances'] = viz_data['primary_feature_importance']  # For old templates
            print("Added legacy feature importance aliases for backward compatibility")
            
            # Make sure viz_data has an 'errors' key before setting viz_errors
            if 'errors' not in viz_data:
                viz_data['errors'] = viz_errors
            else:
                # Add any remaining errors from viz_errors if not already in viz_data['errors']
                for key, error in viz_errors.items():
                    if key not in viz_data['errors']:
                        viz_data['errors'][key] = error
            
            # Support both singular and plural variants of feature_importance for template compatibility
            if 'feature_importance' in viz_data and viz_data['feature_importance']:
                viz_data['feature_importances'] = viz_data['feature_importance']
                print("Added feature_importances alias for backward compatibility")
            elif 'feature_importances' in viz_data and viz_data['feature_importances']:
                viz_data['feature_importance'] = viz_data['feature_importances']
                print("Added feature_importance alias for backward compatibility")

            # Support both naming conventions for permutation importance as well
            if 'permutation_importance' in viz_data and viz_data['permutation_importance']:
                viz_data['permutation_importances'] = viz_data['permutation_importance']
                print("Added permutation_importances alias for backward compatibility")
            elif 'permutation_importances' in viz_data and viz_data['permutation_importances']:
                viz_data['permutation_importance'] = viz_data['permutation_importances']
                print("Added permutation_importance alias for backward compatibility")
            
            # Set up backward compatibility for different naming conventions
            # For tree-based feature importance
            if 'tree_feature_importance' in viz_data:
                viz_data['feature_importances'] = viz_data['tree_feature_importance']
                viz_data['feature_importance'] = viz_data['tree_feature_importance']
                print("Added feature importance aliases for backward compatibility")
            
            # For permutation importance
            if 'permutation_importance' in viz_data:
                viz_data['permutation_importances'] = viz_data['permutation_importance']
                print("Added permutation importance alias for backward compatibility")
            
        except Exception as e:
            print(f"Error in create_visualizations: {e}")
            viz_data['errors'] = {'general': str(e)}
            
        return viz_data

# Original functions that VisualizationManager wraps
def create_output_dir(dir_name="visualizations"):
    """Create and return directory for saving visualizations"""
    import os
    # Get the project root directory
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_dir, dir_name)
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    return output_dir

def save_plot(filename, dpi=300):
    """Save the current plot to the visualizations directory"""
    output_dir = create_output_dir()
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {filepath}")
    return filepath

# These functions kept for backward compatibility
def plot_class_distribution(df, show=False, save_path=None):
    """Plot the distribution of mushroom classes (edible vs poisonous)"""
    plt.figure(figsize=(8, 6))
    
    class_counts = df['class'].value_counts()
    ax = sns.countplot(x='class', data=df)
    
    plt.title('Mushroom Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    if df['class'].dtype == 'object':
        plt.xticks(range(len(class_counts)), ['Edible (e)', 'Poisonous (p)'])
    else:
        plt.xticks(range(len(class_counts)), ['Edible (0)', 'Poisonous (1)'])
    
    for i, patch in enumerate(ax.patches):
        patch.set_color(EDIBLE_COLOR if i == 0 else POISONOUS_COLOR)
    
    for i, count in enumerate(class_counts.values):
        ax.text(i, count + 30, f'{count} ({count/len(df):.1%})', 
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    elif not show:
        save_plot('class_distribution.png')
    
    if show:
        return plt  # Return the plt object for web display
    else:
        plt.close()
        return None

def plot_feature_distribution(df, feature_name):
    """Plot distribution of a single feature by class"""
    plt.figure(figsize=(10, 6))
    
    if df[feature_name].dtype.name in ['object', 'category']:
        # For categorical features
        crosstab = pd.crosstab(df[feature_name], df['class'])
        crosstab.plot(kind='bar', stacked=True, color=[EDIBLE_COLOR, POISONOUS_COLOR])
        plt.title(f'Distribution of {feature_name} by Class')
        plt.xlabel(feature_name)
        plt.ylabel('Count')
        plt.legend(['Edible', 'Poisonous'])
    else:
        # For numerical features
        sns.histplot(data=df, x=feature_name, hue='class', bins=20, 
                     palette=[EDIBLE_COLOR, POISONOUS_COLOR], alpha=0.7)
        plt.title(f'Distribution of {feature_name} by Class')
        plt.xlabel(feature_name)
        plt.ylabel('Count')
    
    save_plot(f'feature_{feature_name}.png')
    plt.tight_layout()

def plot_correlation_heatmap(df, features):
    """Plot correlation heatmap for numerical features"""
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr = df[features].corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=0.5)
    
    plt.title('Feature Correlation Matrix')
    save_plot('correlation_heatmap.png')
    plt.tight_layout()

def plot_pca_visualization(df, features, show=False, save_path=None):
    """Visualize dataset using PCA"""
    # Prepare data
    X = df[features].copy()
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'class': df['class']
    })
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with dynamic palette based on class type
    if df['class'].dtype == 'object':  # If classes are 'e'/'p'
        palette_dict = {'e': EDIBLE_COLOR, 'p': POISONOUS_COLOR}
    else:  # If classes are 0/1
        palette_dict = {0: EDIBLE_COLOR, 1: POISONOUS_COLOR}
        
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='class', data=pca_df,
        palette=palette_dict,
        alpha=0.7, s=80
    )
    
    # Add variance explanation
    explained_variance = pca.explained_variance_ratio_
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
    plt.title('PCA Visualization of Mushroom Dataset')
    plt.legend(title='Class', labels=['Edible', 'Poisonous'])
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    elif not show:
        save_plot('pca_visualization.png')
    
    if show:
        return plt  # Return the plt object for web display
    else:
        plt.close()
        return None

def plot_top_features(df, top_features, class_col='class'):
    """Plot the most informative features side by side"""
    n_features = len(top_features)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    plt.figure(figsize=(12, n_rows * 4))
    
    for i, feature in enumerate(top_features):
        plt.subplot(n_rows, n_cols, i+1)
        
        if df[feature].dtype.name in ['object', 'category']:
            # Categorical feature
            sns.countplot(x=feature, hue=class_col, data=df, 
                         palette=[EDIBLE_COLOR, POISONOUS_COLOR])
            plt.xticks(rotation=45)
        else:
            # Numerical feature
            sns.histplot(data=df, x=feature, hue=class_col, bins=20, 
                        palette=[EDIBLE_COLOR, POISONOUS_COLOR], alpha=0.7)
        
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
    
    save_plot('top_features.png')
    plt.tight_layout()

def plot_numerical_features(df, numerical_features=None, show=False, save_path=None):
    """Plot distribution of numerical features by class"""
    if numerical_features is None:
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'class' in numerical_features:
            numerical_features.remove('class')
    
    if not numerical_features:
        print("No numerical features found in the dataset.")
        return None
    
    n_features = len(numerical_features)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    plt.figure(figsize=(12, n_rows * 4))
    
    for i, feature in enumerate(numerical_features):
        plt.subplot(n_rows, n_cols, i+1)
        sns.histplot(data=df, x=feature, hue='class', bins=20, 
                    palette=[EDIBLE_COLOR, POISONOUS_COLOR], alpha=0.7, kde=True)
        plt.title(f'Distribution of {feature}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    elif not show:
        save_plot('numerical_features.png')
    
    if show:
        return plt  # Return the plt object for web display
    else:
        plt.close()
        return None

def plot_categorical_features(df, categorical_features=None, show=False, save_path=None):
    """Plot distribution of categorical features by class"""
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'class' in categorical_features:
            categorical_features.remove('class')
    
    if not categorical_features:
        print("No categorical features found in the dataset.")
        return None
    
    n_features = len(categorical_features)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    plt.figure(figsize=(14, n_rows * 4))
    
    for i, feature in enumerate(categorical_features):
        plt.subplot(n_rows, n_cols, i+1)
        sns.countplot(x=feature, hue='class', data=df, 
                      palette=[EDIBLE_COLOR, POISONOUS_COLOR])
        plt.title(f'Distribution of {feature}')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    elif not show:
        save_plot('categorical_features.png')
    
    if show:
        return plt  # Return the plt object for web display
    else:
        plt.close()
        return None

def plot_correlation_matrix(df, features=None):
    """Plot correlation matrix for dataset features - alias for plot_correlation_heatmap"""
    if features is None:
        # If no features specified, use all numerical features
        features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Exclude class column if it's numerical
        if 'class' in features:
            features.remove('class')
    
    return plot_correlation_heatmap(df, features)

def plot_feature_correlations(df: pd.DataFrame, save_path: str):
    """
    Compute and plot correlation of each feature with the binary target 'class'.
    Saves a horizontal bar chart to `save_path`.
    """
    try:
        # Map class to numeric properly
        df_num = df.copy()
        if 'class' not in df_num.columns:
            print("Error: 'class' column not found in DataFrame for correlation.")
            return
        df_num['class'] = df_num['class'].map({'e': 0, 'p': 1})
        
        # One-hot encode ALL categorical features (excluding target)
        categorical_cols = df_num.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'class' in categorical_cols:
            categorical_cols.remove('class')
            
        # Make sure df_enc includes both numerical AND encoded categorical features
        df_enc = pd.get_dummies(df_num, columns=categorical_cols)
        
        # Calculate correlations with target
        corrs = df_enc.corr()['class'].drop('class').sort_values()
        
        # Select top features by absolute correlation (both positive and negative)
        top_n = 20
        top_corrs_indices = corrs.abs().nlargest(top_n).index
        plot_corrs = corrs.loc[top_corrs_indices].sort_values()
        
        # Create plot with appropriate size for feature count
        plt.figure(figsize=(10, max(6, len(plot_corrs) * 0.3)))
        
        # Color bars by correlation direction: red for positive (poisonous), green for negative (edible)
        colors = ['#4CAF50' if c < 0 else '#F44336' for c in plot_corrs]
        
        # Plot bars
        ax = sns.barplot(x=plot_corrs.values, y=plot_corrs.index, palette=colors)
        plt.title("Feature Correlations with Edibility")
        plt.xlabel("Correlation coefficient")
        plt.ylabel("Features")
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Successfully saved feature correlations plot to {save_path}")
        
    except Exception as e:
        print(f"Error in plot_feature_correlations: {e}")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Error generating plot:\n{e}', ha='center', va='center', color='red')
        plt.savefig(save_path)
        plt.close()

def plot_extended_correlation_matrix(df, top_n=12, save_path=None):
    """
    Plot correlation matrix including both numerical and encoded categorical features.
    Shows the top N most correlated features with the target.
    
    Parameters:
        df: DataFrame with features and target
        top_n: Number of top features to show
        save_path: Path to save the visualization
    """
    print("Generating extended correlation matrix...")
    
    # Create a copy with class converted to numeric
    df_corr = df.copy()
    if 'class' in df_corr.columns and df_corr['class'].dtype == object:
        df_corr['class'] = df_corr['class'].map({'e': 0, 'p': 1})
    
    # One-hot encode categorical features
    categorical_cols = df_corr.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'class' in categorical_cols:
        categorical_cols.remove('class')
    
    if categorical_cols:
        df_corr = pd.get_dummies(df_corr, columns=categorical_cols)
    
    # Calculate correlations with target
    correlations = df_corr.corr()
    if 'class' in correlations.columns:
        # Get the top features by absolute correlation with target
        target_corrs = correlations['class'].drop('class')
        top_features = target_corrs.abs().nlargest(top_n).index.tolist()
        # Add 'class' back
        selected_columns = top_features + ['class']
        # Filter correlation matrix to only these columns
        corr_subset = df_corr[selected_columns].corr()
    else:
        # No target column, just take top_n columns
        corr_subset = correlations.iloc[:top_n, :top_n] 
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    mask = np.zeros_like(corr_subset, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                square=True, linewidths=0.5, vmin=-1, vmax=1)
    
    plt.title('Extended Feature Correlation Matrix')
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved extended correlation matrix to {save_path}")
    else:
        save_plot('extended_correlation_matrix.png')
    
    plt.close()
    return corr_subset