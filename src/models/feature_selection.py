from sklearn.base import clone
import numpy as np
import pandas as pd
import joblib
import os

class FeatureSelector:
    """Class for selecting important features and training refined models"""
    
    def select_features(self, df, model, X_train, X_test, y_train, y_test, feature_names, 
                        methods=['permutation', 'importance', 'mutual_info']):
        """
        Select important features using multiple methods
        """
        # Import here to avoid circular imports
        from src.feature_analysis import select_features as feature_analysis_select_features
        
        selected_features = feature_analysis_select_features(
            df, model, X_train, X_test, y_train, y_test, feature_names, 
            methods=methods, save_dir='models', cv=5
        )
        
        # Create a result dictionary similar to what's saved by the original function
        results = joblib.load('models/feature_selection_performance.pkl')
        feature_indices = results.get('feature_indices', [])
        
        # Select data based on feature indices
        if feature_indices:
            X_train_selected = X_train[:, feature_indices]
            X_test_selected = X_test[:, feature_indices]
            
            # Extract selected feature names
            selected_feature_names = [feature_names[i] for i in feature_indices 
                                     if i < len(feature_names)]
        else:
            # Fallback - use all features
            X_train_selected = X_train
            X_test_selected = X_test
            selected_feature_names = feature_names
        
        return {
            'selected_features': selected_features,
            'feature_indices': feature_indices,
            'X_train_selected': X_train_selected,
            'X_test_selected': X_test_selected,
            'selected_feature_names': selected_feature_names
        }
    
    def train_with_selected_features(self, model, X_train, X_test, y_train, y_test, feature_indices):
        """Train a model with selected features only"""
        # Select features
        if feature_indices:
            X_train_selected = X_train[:, feature_indices]
            X_test_selected = X_test[:, feature_indices]
            print(f"Applied feature selection: {X_train.shape[1]} → {X_train_selected.shape[1]} features")
        else:
            X_train_selected = X_train
            X_test_selected = X_test
            print(f"No feature selection applied. Using all {X_train.shape[1]} features")
        
        # Create a new model instance with same parameters
        refined_model = clone(model)
        
        # Train on selected features
        refined_model.fit(X_train_selected, y_train)
        
        # Evaluate and print performance
        accuracy = refined_model.score(X_test_selected, y_test)
        print(f"Refined model accuracy on test set: {accuracy:.4f}")
        
        return refined_model