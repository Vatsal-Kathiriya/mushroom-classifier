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
                        viz_data['model_comparison'] = base64.b64encode(f.read()).decode()
                else:
                    # Generate a new comparison if we can find results
                    results_df_path = 'models/model_results.pkl'
                    if os.path.exists(results_df_path):
                        results_df = joblib.load(results_df_path)
                        comparison_path = plot_model_comparison(results_df, save_path='visualizations/model_comparison.png')
                        with open(comparison_path, 'rb') as f:
                            viz_data['model_comparison'] = base64.b64encode(f.read()).decode()
            except Exception as e:
                print(f"Error generating model comparison: {e}")
                viz_errors['model_comparison'] = str(e)
            
            # Instead of generating visualizations in the request thread,
            # load pre-generated images from the filesystem
            visualization_files = {
                'class_distribution': 'visualizations/class_distribution.png',
                'categorical_features': 'visualizations/categorical_features.png',
                'numerical_features': 'visualizations/numerical_features.png',
                'pca_visualization': 'visualizations/pca_visualization.png',
                'feature_importance': 'visualizations/feature_importance.png',
                'permutation_importance': 'visualizations/permutation_importance.png',
                'confusion_matrix': 'visualizations/confusion_matrix.png',
                'roc_curve': 'visualizations/roc_curve.png',
                'learning_curve': 'visualizations/learning_curve.png',
                'mutual_info': 'visualizations/mutual_info.png',
                'feature_selection_curve': 'models/feature_selection_curve.png'
            }
            
            # Load each visualization file if it exists
            for key, filepath in visualization_files.items():
                try:
                    if os.path.exists(filepath):
                        with open(filepath, 'rb') as f:
                            viz_data[key] = base64.b64encode(f.read()).decode()
                    else:
                        print(f"Visualization file not found: {filepath}")
                except Exception as e:
                    print(f"Error loading {key} visualization: {e}")
                    viz_errors[key] = str(e)
            
        except Exception as e:
            print(f"Error in create_visualizations: {e}")
            viz_errors['general'] = str(e)
        
        # Add errors to viz_data
        viz_data['errors'] = viz_errors
        
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
    plt.show()

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
    plt.show()

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
    plt.show()

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