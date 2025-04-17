# Add this at the top of the file, before other imports
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import os
import logging
from flask import Flask, render_template, request, flash, redirect, url_for
import joblib
import pandas as pd
import numpy as np

# Import from refactored modules
from src.utils.io import load_file
from src.visualization.core import VisualizationManager
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.secret_key = 'mushroom-classifier-secret-key'

# Initialize visualization manager
viz_manager = VisualizationManager()

# Load the model, preprocessor, and features
try:
    # First try to load refined feature model
    if os.path.exists('models/refined_feature_model.pkl'):
        model = load_file('models/refined_feature_model.pkl', required=True)
        logger.info("Using optimized model with selected features")
        model_type = "Refined Model with Selected Features"
    else:
        model = load_file('models/best_model.pkl', required=True)
        model_type = "Full Feature Model"
    
    preprocessor = load_file('models/preprocessor.pkl', required=True)
    feature_map = load_file('models/feature_map.pkl', required=True)
    
    # Try to load top features
    try:
        performance_metrics = load_file('models/feature_selection_performance.pkl')
        key_features = load_file('models/top_features.pkl')
        logger.info(f"Loaded {len(key_features)} top features")
    except FileNotFoundError:
        logger.warning("Top features not found. Using default features.")
        key_features = ['cap-shape', 'cap-color', 'does-bruise-or-bleed', 
                         'gill-color', 'habitat', 'season']
    
    # Update model type based on model properties
    if hasattr(model, 'feature_importances_'):
        model_type = f"Tree-based {model_type}"
    elif hasattr(model, 'coef_'):
        model_type = f"Logistic Regression {model_type}"
    
    logger.info(f"Model loaded successfully: {model_type}")
except FileNotFoundError as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None
    preprocessor = None
    feature_map = None
    key_features = ['cap-shape', 'cap-color', 'does-bruise-or-bleed', 
                   'gill-color', 'habitat', 'season']
    model_type = "Not Available"

# Feature options (comprehensive set of all possible values)
feature_options = {
    'cap-shape': {
        'b': 'Bell', 'c': 'Conical', 'x': 'Convex', 'f': 'Flat', 's': 'Sunken', 'p': 'Spherical'
    },
    'cap-color': {
        'n': 'Brown', 'b': 'Buff', 'g': 'Gray', 'r': 'Green', 'p': 'Pink',
        'u': 'Purple', 'e': 'Red', 'w': 'White', 'y': 'Yellow', 'o': 'Orange', 'l': 'Blue', 'k': 'Black'
    },
    'does-bruise-or-bleed': {
        't': 'Yes', 'f': 'No'
    },
    'gill-color': {
        'n': 'Brown', 'b': 'Buff', 'g': 'Gray', 'p': 'Pink', 'u': 'Purple', 
        'e': 'Red', 'w': 'White', 'y': 'Yellow', 'o': 'Orange', 'r': 'Green',
        'f': 'None', 'k': 'Black'
    },
    'habitat': {
        'g': 'Grasses', 'l': 'Leaves', 'm': 'Meadows', 'p': 'Paths',
        'h': 'Heaths', 'u': 'Urban', 'w': 'Waste', 'd': 'Woods'
    },
    'season': {
        's': 'Spring', 'u': 'Summer', 'a': 'Autumn', 'w': 'Winter'
    },
    'gill-attachment': {
        'a': 'Adnate', 'd': 'Adnexed', 'f': 'Free', 'n': 'None', 'p': 'Pores', 'e': 'Emarginate'
    },
    'gill-spacing': {
        'c': 'Close', 'd': 'Distant', 'f': 'None'
    },
    'stem-root': {
        'b': 'Bulbous', 'c': 'Club', 'e': 'Equal', 'r': 'Rooted', '?': 'Missing', 's': 'Swollen'
    },
    'stem-surface': {
        'f': 'Fibrous', 'g': 'Grooved', 'y': 'Scaly', 's': 'Smooth', 'k': 'Silky', 'i': 'Shiny'
    },
    'stem-color': {
        'n': 'Brown', 'b': 'Buff', 'g': 'Gray', 'p': 'Pink', 'u': 'Purple',
        'e': 'Red', 'w': 'White', 'y': 'Yellow', 'o': 'Orange', 'r': 'Green', 'k': 'Black'
    },
    'veil-type': {
        'p': 'Partial', 'u': 'Universal'
    },
    'veil-color': {
        'n': 'Brown', 'b': 'Buff', 'g': 'Gray', 'p': 'Pink', 'u': 'Purple',
        'e': 'Red', 'w': 'White', 'y': 'Yellow', 'o': 'Orange', 'r': 'Green', 'k': 'Black'
    },
    'has-ring': {
        't': 'Yes', 'f': 'No'
    },
    'ring-type': {
        'c': 'Cobwebby', 'e': 'Evanescent', 'f': 'None', 'l': 'Large', 'p': 'Pendant',
        's': 'Sheathing', 'z': 'Zone', 'g': 'Grooved', 'm': 'Movable', 'r': 'Flaring'
    },
    'spore-print-color': {
        'n': 'Brown', 'b': 'Buff', 'g': 'Gray', 'p': 'Pink', 'u': 'Purple', 'h': 'Chocolate',
        'e': 'Red', 'w': 'White', 'y': 'Yellow', 'o': 'Orange', 'r': 'Green', 'k': 'Black'
    },
    'cap-surface': {
        'f': 'Fibrous', 'g': 'Grooves', 'y': 'Scaly', 's': 'Smooth', 'h': 'Shiny', 'k': 'Silky', 't': 'Tacky'
    },
    'cap-diameter': {},  # Numerical feature
    'stem-height': {},   # Numerical feature
    'stem-width': {}     # Numerical feature
}

@app.route('/')
def index():
    """Home page with mushroom classifier"""
    logger.debug("Rendering index page")
    return render_template('index.html', 
                          feature_options=feature_options, 
                          key_features=key_features,
                          feature_map=feature_map,
                          model_type=model_type)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if model is None or preprocessor is None:
        flash('Model not loaded. Please run Phase 2 first.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Get form data for key features
        data = {}
        
        # Handle all features properly
        for feature in key_features:
            if feature in request.form:
                if feature in feature_map and feature_map[feature]['type'] == 'numerical':
                    # Convert numerical features properly
                    try:
                        data[feature] = float(request.form[feature])
                    except ValueError:
                        # Use midpoint if conversion fails
                        values = feature_map[feature]['values']
                        data[feature] = sum(values) / 2
                        logger.warning(f"Invalid numerical value for {feature}. Using midpoint.")
                else:
                    # Categorical feature
                    data[feature] = request.form[feature]
        
        # Fill missing features with default values
        for feature in feature_map.keys():
            if feature not in data and feature != 'class':
                if feature_map[feature]['type'] == 'numerical':
                    # Use midpoint of the range
                    values = feature_map[feature]['values']
                    data[feature] = sum(values) / 2
                else:
                    # Use first value for categorical features
                    if feature_map[feature]['values']:
                        data[feature] = feature_map[feature]['values'][0]
                    else:
                        data[feature] = ''
        
        # Create DataFrame and transform input data
        input_df = pd.DataFrame([data])
        input_processed = preprocessor.transform(input_df)
        
        # Check model's expected feature count
        expected_features = getattr(model, 'n_features_in_', None)
        logger.info(f"Model expects {expected_features} features, input has {input_processed.shape[1]} features")
        
        # Always apply feature selection for refined models
        if expected_features and expected_features != input_processed.shape[1]:
            try:
                # Load feature indices from performance file
                results = load_file('models/feature_selection_performance.pkl')
                feature_indices = results.get('feature_indices', [])
                
                if feature_indices:
                    # Check if feature indices are valid
                    if max(feature_indices) < input_processed.shape[1]:
                        # Select only features the model was trained on
                        input_processed = input_processed[:, feature_indices]
                        logger.info(f"Applied feature selection. Features reduced from 119 to {len(feature_indices)}")
                    else:
                        raise ValueError(f"Invalid feature index {max(feature_indices)} for input with {input_processed.shape[1]} features")
            except Exception as e:
                logger.error(f"Error applying feature selection: {str(e)}", exc_info=True)
                flash(f"Error applying feature selection: {str(e)}", "error")
                return redirect(url_for('index'))
        
        # Final dimension check before prediction
        logger.debug(f"Final input shape for prediction: {input_processed.shape}")
        
        # Make prediction
        prediction = model.predict(input_processed)[0]
        prediction_proba = model.predict_proba(input_processed)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
        
        # Get class label and probability
        is_poisonous = bool(prediction)
        confidence = prediction_proba[1] if is_poisonous else prediction_proba[0]
        
        # Prepare result data
        result = {
            'prediction': 'Poisonous' if is_poisonous else 'Edible',
            'confidence': f"{confidence * 100:.1f}%",
            'is_poisonous': is_poisonous
        }
        
        # Convert input data to display format
        display_data = {}
        for feature, value in data.items():
            if feature in key_features:
                if feature_map[feature]['type'] == 'numerical':
                    display_data[feature] = value
                elif feature in feature_options and value in feature_options[feature]:
                    display_data[feature] = feature_options[feature][value]
                else:
                    display_data[feature] = value
        
        return render_template('result.html', result=result, input_data=display_data)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        flash(f'Error during prediction: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/visualizations')
def visualizations():
    """Data visualization dashboard page"""
    viz_data = create_visualizations()
    return render_template('visualizations.html', viz_data=viz_data)

def create_visualizations():
    """Generate visualizations for the dashboard using core visualization module"""
    try:
        # Create visualizations using the visualization manager
        viz_data = viz_manager.create_dashboard_visualizations()
        return viz_data
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}", exc_info=True)
        return {'errors': {'general': str(e)}}

if __name__ == '__main__':
    app.run(debug=True)