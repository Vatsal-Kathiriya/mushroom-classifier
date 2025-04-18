import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Add this line BEFORE other matplotlib/seaborn imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import base64
from scipy import sparse # Import sparse

# Import project modules - Ensure all these functions exist and have the expected signatures
try:
    from src.visualization.core import (
        plot_class_distribution, plot_categorical_features, plot_numerical_features,
        plot_pca_visualization, create_output_dir, plot_correlation_heatmap,
        plot_correlation_matrix, plot_feature_correlations, plot_top_features,

        save_plot, plot_extended_correlation_matrix # Import save_plot if used internally by plotting functions
    )
    from src.models.training import plot_model_comparison
    from src.feature_analysis import (
        plot_feature_importance, calculate_permutation_importance, analyze_feature_correlations,
        mutual_info_feature_selection, recursive_feature_selection, plot_mutual_info,
        evaluate_feature_set, select_features
    )
    from src.evaluation import (
        plot_confusion_matrix, plot_roc_curve, plot_learning_curve, plot_coefficients,
        try_shap_analysis
    )
    # Assuming plot_feature_distribution exists and saves internally via save_plot
    from src.visualization.core import plot_feature_distribution

except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required functions are defined in src subdirectories and have correct signatures.")
    exit() # Exit if imports fail

def create_all_visualizations():
    """Generate all possible visualizations for the dashboard"""
    print("Generating all visualizations...")

    # --- Setup ---
    try:
        # Use create_output_dir which returns the absolute path
        output_dir = create_output_dir('visualizations')
        shap_dir = os.path.join(output_dir, 'shap')
        os.makedirs(shap_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    except Exception as e:
        print(f"Error setting up directories: {e}")
        return False

    # --- Load Data ---
    try:
        df = pd.read_csv('data/secondary_data.csv', delimiter=';')
        print(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

    # --- Basic EDA Visualizations (using raw df) ---
    print("\n--- Generating Basic EDA Visualizations ---")
    try:
        # Class distribution
        print("Generating class distribution...")
        plot_class_distribution(df, save_path=os.path.join(output_dir, 'class_distribution.png'))

        # Feature distributions (Categorical & Numerical)
        categorical_cols = df.select_dtypes(include=['object']).columns.drop('class', errors='ignore').tolist()
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if categorical_cols:
            print("Generating categorical features plot...")
            top_cat_cols = categorical_cols[:min(10, len(categorical_cols))] # Show top 10
            plot_categorical_features(df, top_cat_cols, save_path=os.path.join(output_dir, 'categorical_features.png'))

        if numerical_cols:
            print("Generating numerical features plot...")
            plot_numerical_features(df, numerical_cols, save_path=os.path.join(output_dir, 'numerical_features.png'))

        # PCA visualization
        print("Generating PCA visualization...")
        features_for_pca = df.columns.drop('class', errors='ignore').tolist()
        plot_pca_visualization(df, features_for_pca, save_path=os.path.join(output_dir, 'pca_visualization.png'))

        # Correlation Heatmap (Numerical only)
        if numerical_cols:
            print("Generating correlation heatmap...")
            # Assuming plot_correlation_heatmap saves internally via save_plot
            plot_correlation_heatmap(df, numerical_cols)

        # Correlation Matrix (Numerical only) - Check if different from heatmap
        if numerical_cols:
            print("Generating correlation matrix...")
             # Assuming plot_correlation_matrix saves internally via save_plot
            plot_correlation_matrix(df, numerical_cols)

        # Extended correlation matrix with encoded features
        print("Generating extended correlation matrix...")
        from src.visualization.core import plot_extended_correlation_matrix
        plot_extended_correlation_matrix(df, save_path=os.path.join(output_dir, 'extended_correlation_matrix.png'))

        # Feature Correlations with Target
        print("Generating feature correlations with target plot...")
        plot_feature_correlations(df, save_path=os.path.join(output_dir, 'feature_correlations.png'))

        # Individual Feature Distributions (Top 10 Raw)
        print("Generating individual feature distributions (top 10 raw)...")
        orig_feats = df.columns.drop('class', errors='ignore').tolist()
        for feat in orig_feats[:10]:
            try:
                print(f"  Plotting: {feat}")
                # Assuming plot_feature_distribution saves internally via save_plot
                plot_feature_distribution(df, feat)
            except Exception as e_feat:
                print(f"    Error plotting {feat}: {e_feat}")

    except Exception as e:
        print(f"Error during Basic EDA Visualizations: {e}")
        # Continue if possible, but some later steps might fail

    # --- Model Loading and Preprocessing ---
    print("\n--- Loading Model and Preprocessing Data ---")
    model = None
    full_feature_names = []
    selected_feature_names = []
    X_train, X_test, y_train, y_test = None, None, None, None
    X_train_sel, X_test_sel = None, None
    X_full_sel, y_full = None, None
    feat_inds = []

    try:
        # Load model (try refined first, then best)
        model_path = 'models/refined_feature_model.pkl'
        if not os.path.exists(model_path):
            model_path = 'models/best_model.pkl' # Fallback
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Loaded model from: {model_path}")
        else:
            print("Warning: No trained model found (refined_feature_model.pkl or best_model.pkl). Skipping model-dependent visualizations.")

        # Preprocess data
        from src.data.processing import DataProcessor
        dp = DataProcessor()
        prep = dp.preprocess(df, test_size=0.2)
        X_train = prep['X_train']
        X_test = prep['X_test']
        y_train = prep['y_train']
        y_test = prep['y_test']
        full_feature_names = prep['feature_names']
        print(f"Preprocessed data. Full feature count: {len(full_feature_names)}")

        # Apply feature selection if applicable
        perf_path = 'models/feature_selection_performance.pkl'
        selected_feature_names = list(full_feature_names) # Default to all
        X_train_sel = X_train
        X_test_sel = X_test

        if os.path.exists(perf_path):
            perf = joblib.load(perf_path)
            feat_inds = perf.get('feature_indices', [])
            if feat_inds:
                # Ensure indices are valid before slicing
                max_index = max(feat_inds) if feat_inds else -1
                if max_index < X_train.shape[1]:
                    print(f"Applying feature selection ({len(feat_inds)} features).")
                    X_train_sel = X_train[:, feat_inds]
                    X_test_sel = X_test[:, feat_inds]
                    selected_feature_names = [full_feature_names[i] for i in feat_inds]
                else:
                    print(f"Warning: Invalid feature indices found (max index {max_index} >= {X_train.shape[1]}). Using all features.")
                    feat_inds = [] # Reset indices if invalid
            else:
                 print("Feature selection file found, but no indices. Using all features.")
        else:
            print("No feature selection file found. Using all features.")

        # Combine selected features for learning curve etc.
        if X_train_sel is not None and X_test_sel is not None:
            X_full_sel = np.vstack([X_train_sel, X_test_sel])
            y_full = np.concatenate([y_train, y_test])
        else:
             print("Warning: Selected train/test sets not available. Skipping combined set creation.")

    except Exception as e:
        print(f"Error during Model Loading/Preprocessing: {e}")
        model = None # Ensure model is None if this section fails

    # --- Model-Dependent Visualizations ---
    if model is not None and X_test_sel is not None and y_test is not None and selected_feature_names:
        print("\n--- Generating Model-Dependent Visualizations ---")
        try:
            # Feature Importance / Coefficients
            print("Generating feature importance/coefficients plot...")
            if hasattr(model, 'feature_importances_'):
                try:
                    # Close any existing figures
                    plt.close('all')
                    
                    # Generate the plot
                    feature_importance_path = os.path.join(output_dir, 'feature_importance.png')
                    print(f"Saving feature importance to: {feature_importance_path}")
                    
                    # Use more robust plotting approach
                    plt.figure(figsize=(12, 8))
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    # Plot directly here rather than relying on external function
                    plt.title('Feature Importance from Tree-based Model', fontsize=16)
                    plt.bar(range(len(selected_feature_names)), 
                            [importances[i] for i in indices], 
                            color='yellowgreen')
                    plt.xticks(range(len(selected_feature_names)), 
                            [selected_feature_names[i] for i in indices], 
                            rotation=90)
                    plt.tight_layout()
                    
                    # Save with explicit dpi and bbox settings
                    plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Verify the file was created and is readable
                    if os.path.exists(feature_importance_path):
                        filesize = os.path.getsize(feature_importance_path)
                        print(f"✓ Feature importance saved successfully: {filesize} bytes")
                        
                        # Test if file can be read
                        with open(feature_importance_path, 'rb') as f:
                            _ = f.read(10)  # Try to read first 10 bytes
                            print(f"✓ Feature importance file is readable")
                    else:
                        print(f"✗ Failed to generate feature importance plot")
                
                except Exception as e:
                    print(f"Error generating feature importance: {e}")
                    plt.close('all')
            elif hasattr(model, 'coef_'):
                 # Pass the correct feature names corresponding to X_test_sel
                 plot_coefficients(model, selected_feature_names, save_path=os.path.join(output_dir, 'logistic_regression_coefficients.png'))
            else:
                 print("Model type does not support feature importance or coefficients.")

            # Evaluation Plots
            print("Generating confusion matrix...")
            y_pred = model.predict(X_test_sel)
            plot_confusion_matrix(y_test, y_pred, save_path=os.path.join(output_dir, 'confusion_matrix.png'))

            print("Generating ROC curve...")
            y_pred_proba = model.predict_proba(X_test_sel)[:, 1] if hasattr(model, 'predict_proba') else None
            plot_roc_curve(y_test, y_pred_proba, save_path=os.path.join(output_dir, 'roc_curve.png'))

            # Learning Curve (using combined selected data)
            if X_full_sel is not None and y_full is not None:
                print("Generating learning curve...")
                plot_learning_curve(model, X_full_sel, y_full, save_path=os.path.join(output_dir, 'learning_curve.png'))
            else:
                print("Skipping learning curve (combined selected data not available).")

            # Permutation Importance
            print("Generating permutation importance plot...")
            # Pass the correct feature names corresponding to X_test_sel
            calculate_permutation_importance(model, X_test_sel, y_test, selected_feature_names, save_path=os.path.join(output_dir, 'permutation_importance.png'))

            # SHAP Analysis
            # print("Running SHAP analysis...")
            # # Pass the absolute path for save_dir
            # shap_save_path = os.path.join(shap_dir, 'shap_summary.png') # Define full path
            # # Pass the correct feature names corresponding to X_test_sel (or X_train_sel if preferred for SHAP)
            # try_shap_analysis(model, X_train_sel, selected_feature_names, save_path=shap_save_path) # Pass save_path

        except Exception as e:
            print(f"Error during Model-Dependent Visualizations: {e}")
    else:
        print("\nSkipping Model-Dependent Visualizations (Model or Data missing/invalid).")

    # --- Feature Analysis Visualizations (some model-independent) ---
    print("\n--- Generating Feature Analysis Visualizations ---")
    try:
        # Mutual Information (using full processed data before selection)
        if X_train is not None and y_train is not None and full_feature_names:
            print("Generating mutual information plot...")
            # Calculate scores first (plot=False)
            mi_df = mutual_info_feature_selection(X_train, y_train, full_feature_names, plot=False)
            if mi_df is not None and not mi_df.empty:
                 # Then plot the scores
                 plot_mutual_info(mi_df, save_path=os.path.join(output_dir, 'mutual_info.png'))
            else:
                 print("Mutual information calculation failed or returned empty.")
        else:
            print("Skipping Mutual Information plot (missing processed data/names).")

        # Top Features Plot (using original feature names identified during selection)
        print("Generating top features distribution plot...")
        top_features_list = []
        top_features_path = 'models/top_features.pkl'
        if os.path.exists(top_features_path):
             top_features_list = joblib.load(top_features_path)
        if not top_features_list: # Fallback if file not found or empty
             # Use top 10 original features if list is empty
             top_features_list = df.columns.drop('class', errors='ignore').tolist()[:10]
        # Assuming plot_top_features saves internally via save_plot
        if top_features_list:
            plot_top_features(df, top_features_list)
        else:
            print("Skipping top features plot as no features were identified.")


        # Analyze Feature-Feature Correlations (Numerical only)
        if numerical_cols:
             print("Analyzing feature-feature correlations...")
             try:
                 # Assuming analyze_feature_correlations saves plot internally
                 corr_results_series = analyze_feature_correlations(df, numerical_cols)
                 if corr_results_series is not None:
                     # Save the series data
                     corr_results_series.to_csv(os.path.join(output_dir, 'feature_correlation_analysis.csv'))
             finally:
                 plt.close('all')  # Ensure all figures are closed

        # --- Advanced/Optional ---
        # These are often part of the main training/selection pipeline.
        # Generating them here might be redundant or require specific setups.
        # Marked as skipped for clarity and to avoid potential errors/long runtimes.

        # Recursive Feature Selection
        print("Skipping Recursive Feature Selection (computationally expensive, assumed done in Phase 2).")
        # Evaluate Feature Set
        print("Skipping explicit feature set evaluation (covered by selection process).")
        # Automatic Feature Selection (using select_features function)
        print("Skipping automatic feature selection run (assumed done in Phase 2).")

    except Exception as e:
        print(f"Error during Feature Analysis Visualizations: {e}")

    # --- Model Comparison Visualization ---
    print("\n--- Generating Model Comparison Visualization ---")
    try:
        # Close any open figures before starting this section
        plt.close('all')
        
        results_path = 'models/model_results.pkl'
        if os.path.exists(results_path):
            results_df = joblib.load(results_path)
            plot_model_comparison(results_df, save_path=os.path.join(output_dir, 'model_comparison.png'))
            # Ensure figure is closed after plotting
            plt.close('all')
        else:
            print("Warning: model_results.pkl not found. Cannot generate model comparison.")
    except Exception as e:
        plt.close('all')  # Close figures even on error
        print(f"Error generating model comparison chart: {e}")

    print(f"\nVisualization generation process finished. Check the '{output_dir}' directory.")
    
    verify_image_loading(output_dir)

    # Add this line to copy the image for web display
    copy_feature_importance_for_web()
    
    return True

# Add after visualization generation is complete
def verify_image_loading(output_dir):
    """Verify that key images can be loaded properly"""
    print("\nVerifying image loading for dashboard...")
    # Prioritize feature importance in checks
    key_images = ['feature_importance.png', 'model_comparison.png', 'confusion_matrix.png', 'roc_curve.png']
    
    for img_name in key_images:
        img_path = os.path.join(output_dir, img_name)
        if os.path.exists(img_path):
            try:
                file_size = os.path.getsize(img_path)
                with open(img_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                    print(f"✓ Successfully loaded and encoded: {img_name} ({file_size} bytes)")
                    
                    # For feature importance, add extra check
                    if img_name == 'feature_importance.png':
                        print(f"  > Feature importance absolute path: {os.path.abspath(img_path)}")
                        # Create a simple HTML file to test image display
                        test_html = os.path.join(output_dir, 'test_image.html')
                        with open(test_html, 'w') as html_file:
                            html_file.write(f'<html><body><h1>Test Image</h1><img src="data:image/png;base64,{img_data}" alt="Feature Importance"></body></html>')
                        print(f"  > Created test HTML file: {test_html}")
            except Exception as e:
                print(f"✗ Error loading {img_name}: {e}")
        else:
            print(f"✗ Image not found: {img_path}")
            # For feature importance, try to find any PNG files
            if img_name == 'feature_importance.png':
                print("  > Searching for any PNG files in output directory...")
                png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
                print(f"  > Found {len(png_files)} PNG files: {png_files}")

def copy_feature_importance_for_web():
    """Copy feature importance image to web-accessible location"""
    try:
        source = os.path.join('visualizations', 'feature_importance.png')
        
        # Make sure the target directory exists
        target_dir = os.path.join('static', 'generated_viz')
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy the file
        target = os.path.join(target_dir, 'feature_importance.png')
        import shutil
        shutil.copy2(source, target)
        print(f"Copied feature importance image to {target}")
        return True
    except Exception as e:
        print(f"Error copying feature importance image: {e}")
        return False

def cleanup_figures():
    """Close all open matplotlib figures to free memory"""
    plt.close('all')
    print("Closed all matplotlib figures to free memory")
    
if __name__ == "__main__":
    create_all_visualizations()