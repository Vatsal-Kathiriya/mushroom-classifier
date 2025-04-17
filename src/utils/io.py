import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_file(obj, filepath):
    """Save an object to disk"""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    joblib.dump(obj, filepath)
    print(f"Saved to: {filepath}")

def load_file(filepath, default=None, required=False):
    """Load an object from disk with error handling
    
    Parameters:
        filepath (str): Path to the file to load
        default: Default value to return if loading fails
        required (bool): Whether the file is required (raises exception if True)
    
    Returns:
        The loaded object or default value
    """
    try:
        obj = joblib.load(filepath)
        print(f"Successfully loaded: {filepath}")
        return obj
    except (FileNotFoundError, IOError) as e:
        message = f"Error loading {filepath}: {str(e)}"
        if required:
            raise FileNotFoundError(message)
        else:
            print(message)
            return default

def extract_feature_names(preprocessor):
    """Extract feature names from a column transformer"""
    feature_names = []
    
    if not hasattr(preprocessor, 'transformers_'):
        return feature_names
        
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            # Numerical features stay as they are
            feature_names.extend(cols)
        elif name == 'cat' and hasattr(transformer, 'get_feature_names_out'):
            # Categorical features get expanded with OneHotEncoder
            feature_names.extend(transformer.get_feature_names_out(cols))
        elif name == 'cat':
            # Fallback if get_feature_names_out is not available
            for col in cols:
                if hasattr(transformer, 'categories_'):
                    categories = transformer.categories_[cols.index(col)]
                    for cat in categories:
                        feature_names.append(f"{col}_{cat}")
                else:
                    feature_names.append(col)
    
    return feature_names

def get_original_feature_names(encoded_features):
    """Extract original feature names from encoded feature names"""
    original_features = [feat.split('_')[0] if '_' in feat else feat for feat in encoded_features]
    unique_features = list(dict.fromkeys(original_features))
    return unique_features