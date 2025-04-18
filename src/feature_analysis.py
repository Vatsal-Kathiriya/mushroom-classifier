import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFECV
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from scipy import sparse
import joblib
import os
from src.visualization import EDIBLE_COLOR, POISONOUS_COLOR, save_plot

def plot_feature_importance(model, feature_names, top_n=20, save_path=None, show=False):
    """Plot feature importance for tree-based models"""
    if not hasattr(model, 'feature_importances_'):
        print("Model doesn't have feature_importances_ attribute")
        return None
        
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top N features
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_feature_names = [feature_names[i] for i in top_indices]
    
    # Create figure with larger size for better readability
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a colormap identical to the reference image - yellowgreen gradient
    colors = plt.cm.YlGn(np.linspace(0.1, 0.9, len(top_importances)))
    
    # Plot horizontal bars with decreasing importance
    y_pos = np.arange(len(top_feature_names))
    ax.barh(y_pos, top_importances, align='center', color=colors)
    
    # Set y-tick labels - features ordered by importance
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feature_names)
    
    # Add labels and title that match reference
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title('Top Features for Mushroom Classification', fontsize=14)
    
    # Add colorbar with proper formatting
    sm = plt.cm.ScalarMappable(cmap="YlGn", norm=plt.Normalize(0, max(top_importances)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Importance', fontsize=10)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    elif not show:
        save_plot('feature_importance.png')
    
    if show:
        return plt
    else:
        plt.close()
        # Create and return dictionary mapping features to importance
        return dict(zip(top_feature_names, top_importances))

def calculate_permutation_importance(model, X_test, y_test, feature_names, n_repeats=10, save_path=None):
    """Calculate permutation importance"""
    # Convert sparse matrix to dense if needed
    X_test_dense = X_test.toarray() if sparse.issparse(X_test) else X_test
        
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test_dense, y_test, n_repeats=n_repeats)
    
    # Create a DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)
    
    # Plot with updated seaborn syntax
    plt.figure(figsize=(12, 8))
    top_20 = importance_df.head(20)
    
    # Use the new recommended approach (explicitly setting x and y parameters)
    ax = sns.barplot(x='Importance', y='Feature', data=top_20, color='yellowgreen')
    
    # Add error bars showing std deviation
    for i, (imp, std) in enumerate(zip(top_20['Importance'], top_20['Std'])):
        ax.errorbar(imp, i, xerr=std, fmt='none', color='black', capsize=3)
    
    plt.title('Permutation Feature Importance', fontsize=14)
    plt.xlabel('Importance (Decrease in Model Performance)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = 'permutation_importance.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return importance_df

def analyze_feature_correlations(df, numerical_features, target_col='class', save_path=None):
    """
    Analyze feature correlations with the target
    
    Parameters:
        df: DataFrame with features and target
        numerical_features: List of numerical feature names
        target_col: Name of target column
        save_path: Path to save the plot (if None, uses default path)
        
    Returns:
        Series with correlation coefficients
    """
    # Create a copy of the dataframe with the needed columns
    correlation_df = df[numerical_features + [target_col]].copy()
    
    # Convert class to numeric (e=0, p=1)
    if correlation_df[target_col].dtype == object:
        correlation_df[target_col] = correlation_df[target_col].map({'e': 0, 'p': 1})
    
    # Calculate correlations
    corr_matrix = correlation_df.corr()
    target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    
    # Plot correlations
    plt.figure(figsize=(10, 6))
    
    # Use different colors for positive and negative correlations
    colors = ['#4CAF50' if c < 0 else '#F44336' for c in target_corr]
    
    sns.barplot(x=target_corr.values, y=target_corr.index, hue=target_corr.index, palette=colors, legend=False)
    plt.title('Feature Correlation with Target (Class)', fontsize=14)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = 'feature_correlations.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return target_corr

def mutual_info_feature_selection(X, y, feature_names, top_n=20, plot=True, save_path=None):
    """
    Select features using mutual information
    
    Parameters:
        X: Features matrix
        y: Target vector
        feature_names: List of feature names
        top_n: Number of top features to select
        plot: Whether to plot the results
        save_path: Path to save the plot (if None, uses default path)
        
    Returns:
        DataFrame with mutual information scores
    """
    # Check if X is a sparse matrix and convert to dense if needed
    if sparse.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = X
        
    # Calculate mutual information
    mi = mutual_info_classif(X_dense, y, random_state=42)
    
    # Create a DataFrame
    mi_df = pd.DataFrame({
        'Feature': feature_names,
        'Mutual_Information': mi
    })
    
    # Sort by mutual information
    mi_df = mi_df.sort_values('Mutual_Information', ascending=False)
    
    if plot:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Mutual_Information', y='Feature', data=mi_df.head(top_n), palette='YlGn')
        plt.title('Mutual Information with Target', fontsize=14)
        plt.xlabel('Mutual Information Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = 'mutual_info.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    return mi_df

def recursive_feature_selection(model, X, y, feature_names, cv=5, step=1, min_features=5):
    """Perform recursive feature elimination with cross-validation"""
    # Convert sparse matrix to dense if needed
    X_dense = X.toarray() if sparse.issparse(X) else X
    
    # Perform recursive feature elimination with cross-validation
    rfecv = RFECV(
        estimator=model,
        step=step,
        cv=cv,
        scoring='accuracy',
        min_features_to_select=min_features,
        n_jobs=-1
    )
    
    rfecv.fit(X_dense, y)
    
    # Create a DataFrame with rankings
    rfe_df = pd.DataFrame({
        'Feature': feature_names,
        'Ranking': rfecv.ranking_,
        'Selected': rfecv.support_
    }).sort_values('Ranking')
    
    # Plot number of features vs CV score
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(rfecv.grid_scores_) + 1),
        rfecv.grid_scores_,
        marker='o'
    )
    plt.xlabel('Number of Features')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Recursive Feature Elimination')
    plt.tight_layout()
    plt.savefig('rfe_cv.png', dpi=300)
    plt.close()
    
    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"Best cross-validation score: {max(rfecv.grid_scores_):.4f}")
    
    # Return selected feature indices
    selected_indices = np.where(rfecv.support_)[0]
    selected_features = [feature_names[i] for i in selected_indices]
    
    return selected_features, rfe_df

def plot_mutual_info(mi_df, top_n=20, save_path=None):
    """Plot mutual information between features and target"""
    plt.figure(figsize=(12, 8))
    # Update to use new seaborn syntax
    sns.barplot(x='Mutual_Information', y='Feature', data=mi_df.head(top_n), color='yellowgreen')
    plt.title('Mutual Information with Target', fontsize=14)
    plt.xlabel('Mutual Information Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def evaluate_feature_set(model, X, y, feature_indices, cv=5):
    """
    Evaluate a feature set using cross-validation
    
    Parameters:
        model: Model to evaluate
        X: Feature matrix
        y: Target vector
        feature_indices: Indices of features to select
        cv: Number of cross-validation folds
        
    Returns:
        Mean cross-validation accuracy
    """
    X_subset = X[:, feature_indices]
    cv_scores = cross_val_score(model, X_subset, y, cv=cv, scoring='accuracy')
    return cv_scores.mean()

def select_features(df, model, X_train, X_test, y_train, y_test, feature_names, 
                   methods=['permutation', 'importance', 'mutual_info'], 
                   n_features=10, save_dir='models', cv=5):
    """
    Select top features using multiple methods and evaluate feature interactions
    
    Parameters:
        df: Original dataframe with features and target
        model: Trained model
        X_train, y_train: Training data
        X_test, y_test: Test data
        feature_names: Names of all features after preprocessing
        methods: Feature selection methods to use
        n_features: Maximum number of features to select
        save_dir: Directory to save results
        cv: Number of cross-validation folds
        
    Returns:
        List of selected feature names
    """
    all_scores = {}
    os.makedirs(save_dir, exist_ok=True)
    
    # Get initial feature rankings using different methods
    if 'permutation' in methods:
        perm_df = calculate_permutation_importance(
            model, X_test, y_test, feature_names, 
            save_path=os.path.join(save_dir, 'permutation_importance.png')
        )
        all_scores['permutation'] = dict(zip(perm_df['Feature'], perm_df['Importance']))
    
    if 'importance' in methods and hasattr(model, 'feature_importances_'):
        imp_dict = plot_feature_importance(
            model, feature_names, 
            save_path=os.path.join(save_dir, 'feature_importance.png')
        )
        if imp_dict:
            all_scores['importance'] = imp_dict
    
    if 'mutual_info' in methods:
        X_dense = X_train.toarray() if sparse.issparse(X_train) else X_train
        mi = mutual_info_classif(X_dense, y_train, random_state=42)
        mi_df = pd.DataFrame({
            'Feature': feature_names,
            'Mutual_Information': mi
        }).sort_values('Mutual_Information', ascending=False)
        
        plot_mutual_info(mi_df, save_path=os.path.join(save_dir, 'mutual_info.png'))
        
        all_scores['mutual_info'] = dict(zip(mi_df['Feature'], mi_df['Mutual_Information']))
    
    # Generate initial ranked_features list based on combined scores
    combined_ranks = {feature: 0 for feature in feature_names}
    
    for method, scores in all_scores.items():
        ranked_features = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        for rank, feature in enumerate(ranked_features):
            if feature in combined_ranks:
                combined_ranks[feature] += rank
    
    # Sort features by combined rank (ascending)
    initial_ranked_features = sorted(combined_ranks.keys(), key=lambda x: combined_ranks[x])
    
    # Evaluate feature interactions using forward selection with cross-validation
    selected_features = []
    best_score = 0
    best_feature_set = []
    max_features = min(n_features, len(initial_ranked_features))
    
    # Create a copy of the model to use for evaluation
    base_model = clone(model)
    
    # Data for feature selection plot
    feature_counts = []
    cv_scores = []
    
    print("\nPerforming sequential feature selection:")
    # Forward selection process
    for i in range(max_features):
        candidate_feature = None
        best_new_score = best_score
        
        # Try each unselected feature
        for feature in initial_ranked_features[:50]:  # Limit to top 50 for efficiency
            if feature in selected_features:
                continue
                
            # Create a temporary feature set with this candidate feature
            temp_features = selected_features + [feature]
            
            # Extract indices of these features
            feature_indices = [feature_names.index(f) for f in temp_features]
            
            # Evaluate this feature set
            cv_score = evaluate_feature_set(base_model, X_train, y_train, feature_indices, cv=cv)
            
            if cv_score > best_new_score:
                best_new_score = cv_score
                candidate_feature = feature
        
        # If we found a feature that improves the score, add it
        if candidate_feature and best_new_score > best_score:
            selected_features.append(candidate_feature)
            best_score = best_new_score
            best_feature_set = selected_features.copy()
            
            # Save progress for the plot
            feature_counts.append(len(selected_features))
            cv_scores.append(best_score)
            
            print(f"  Added feature: {candidate_feature}, CV score: {best_score:.4f}")
        else:
            # No improvement, stop adding features
            print(f"  No further improvement with additional features. Stopping at {len(selected_features)} features.")
            break
    
    # Plot feature selection process
    plt.figure(figsize=(10, 6))
    plt.plot(feature_counts, cv_scores, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Feature Selection Performance')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_selection_curve.png'), dpi=300)
    plt.close()
    
    # Extract original feature names (before encoding)
    original_features = [feat.split('_')[0] if '_' in feat else feat for feat in best_feature_set]
    valid_features = list(dict.fromkeys([f for f in original_features if f in df.columns and f != 'class']))
    
    # Save final selected features
    joblib.dump(valid_features, os.path.join(save_dir, 'top_features.pkl'))
    print(f"Selected {len(valid_features)} top features and saved to {os.path.join(save_dir, 'top_features.pkl')}")
    print(f"Final feature set: {valid_features}")
    print(f"Cross-validation accuracy with selected features: {best_score:.4f}")
    
    # Evaluate feature set on test data for final validation
    if valid_features:
        # Create a feature set with the selected features
        columns_to_select = []
        for feature in valid_features:
            # Find all columns that start with this feature name (including one-hot encoded)
            matching_cols = [col for col in feature_names if col == feature or col.startswith(f"{feature}_")]
            columns_to_select.extend(matching_cols)
        
        # Get indices of these features
        feature_indices = [feature_names.index(f) for f in columns_to_select if f in feature_names]
        
        if feature_indices:
            # Select these features from training and test data
            X_train_final = X_train[:, feature_indices]
            X_test_final = X_test[:, feature_indices]
            
            # Train a new model with only selected features
            final_model = clone(model)
            final_model.fit(X_train_final, y_train)
            
            # Evaluate on test data
            test_accuracy = final_model.score(X_test_final, y_test)
            print(f"Test accuracy with selected features: {test_accuracy:.4f}")
            
            # Save this performance metric
            performance_metrics = {
                'cv_accuracy': best_score,
                'test_accuracy': test_accuracy,
                'n_features': len(valid_features),
                'feature_indices': feature_indices,
                'selected_features': valid_features  # Save actual feature names too
            }
            joblib.dump(performance_metrics, os.path.join(save_dir, 'feature_selection_performance.pkl'))
            
            # Also save the refined model specifically for the webapp
            refined_model_path = os.path.join(save_dir, 'refined_feature_model.pkl')
            joblib.dump(final_model, refined_model_path)
            print(f"Saved refined feature model to {refined_model_path}")
    
    return valid_features