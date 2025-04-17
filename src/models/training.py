from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os

class ModelTrainer:
    """Model training class for mushroom classification"""
    
    def __init__(self, random_state=42, param_config=None):
        self.random_state = random_state
        self.models = {}
        self.results_df = None
        self.best_model = None
        self.best_model_name = None
        
    def train_models(self, X_train, y_train):
        """Train multiple models on the dataset"""
        self.models = {}
        
        # Define model configurations
        model_configs = {
            'Logistic Regression': (LogisticRegression, 
                                  {'random_state': self.random_state, 'max_iter': 1000}),
            'Random Forest': (RandomForestClassifier, 
                             {'random_state': self.random_state, 'n_estimators': 100}),
            'Gradient Boosting': (GradientBoostingClassifier, 
                                 {'random_state': self.random_state}),
            'Gaussian Naive Bayes': (GaussianNB, {}),
            'Bernoulli Naive Bayes': (BernoulliNB, {}),
            'SVM': (SVC, {'random_state': self.random_state, 'probability': True}),
            'Neural Network': (MLPClassifier, 
                              {'random_state': self.random_state, 'max_iter': 500}),
            'Decision Tree': (DecisionTreeClassifier, {'random_state': self.random_state}),
            'KNN': (KNeighborsClassifier, {'n_neighbors': 5})
        }
        
        # Train each model
        for name, (model_class, params) in model_configs.items():
            print(f"Training {name}...")
            try:
                model = model_class(**params)
                model.fit(X_train, y_train)
                self.models[name] = model
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
        
        # Try to train XGBoost if available
        try:
            from xgboost import XGBClassifier
            print("Training XGBoost...")
            model = XGBClassifier(random_state=self.random_state)
            model.fit(X_train, y_train)
            self.models['XGBoost'] = model
        except ImportError:
            print("XGBoost not available. Skipping.")
        
        return self.models
        
    def evaluate_models(self, X_train, X_test, y_train, y_test, cv=5):
        """Evaluate all models and return results DataFrame"""
        results = []
        
        for name, model in self.models.items():
            # Calculate scores
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            
            # Cross-validation score
            try:
                cv_score = np.mean(cross_val_score(model, X_train, y_train, cv=cv))
            except Exception as e:
                print(f"Warning: CV scoring failed for {name}: {str(e)}")
                cv_score = np.nan
            
            # ROC AUC score
            try:
                from sklearn.metrics import roc_auc_score
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_pred_proba)
                else:
                    auc = np.nan
            except Exception as e:
                print(f"Warning: ROC AUC calculation failed for {name}: {str(e)}")
                auc = np.nan
            
            results.append({
                'Model': name,
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy,
                'CV Score': cv_score,
                'ROC AUC': auc
            })
        
        # Convert to DataFrame and sort by test accuracy
        self.results_df = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)
        
        # Set best model
        if len(self.results_df) > 0:
            self.best_model_name = self.results_df.iloc[0]['Model']
            self.best_model = self.models[self.best_model_name]
        
        return self.results_df
        
    def tune_model(self, model_name=None, X_train=None, y_train=None, cv=3, n_jobs=-1):
        """Tune the best model or a specified model"""
        if model_name is None:
            if self.best_model_name is None:
                print("No best model found. Run evaluate_models first.")
                return None
            model_name = self.best_model_name
            model = self.best_model
        else:
            if model_name not in self.models:
                print(f"Model {model_name} not found.")
                return None
            model = self.models[model_name]
        
        print(f"Tuning {model_name}...")
        
        # Get parameter grid
        param_grid = get_param_grid(model_name)
        if not param_grid:
            print(f"No parameter grid defined for {model_name}. Skipping tuning.")
            return model
        
        # Run grid search
        try:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=n_jobs,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
            
            # Update model in dictionary
            tuned_model = grid_search.best_estimator_
            self.models[model_name] = tuned_model
            
            # Update best model if this is the best model
            if model_name == self.best_model_name:
                self.best_model = tuned_model
            
            return tuned_model
        
        except Exception as e:
            print(f"Error during hyperparameter tuning for {model_name}: {e}")
            return model
    
    def save_models(self, directory='models'):
        """Save all models to disk"""
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            filename = name.lower().replace(' ', '_') + '.pkl'
            filepath = os.path.join(directory, filename)
            try:
                joblib.dump(model, filepath)
                print(f"Model saved to {filepath}")
            except Exception as e:
                print(f"Error saving model {name}: {str(e)}")
        
        # Save best model separately
        if self.best_model is not None:
            best_model_path = os.path.join(directory, 'best_model.pkl')
            best_model_name_path = os.path.join(directory, 'best_model_name.pkl')
            try:
                joblib.dump(self.best_model, best_model_path)
                joblib.dump(self.best_model_name, best_model_name_path)
                print(f"Saved best model ({self.best_model_name}) to {best_model_path}")
            except Exception as e:
                print(f"Error saving best model: {str(e)}")
        
        return directory

# Keep original functions for backward compatibility
def get_param_grid(model_name):
    """Return appropriate parameter grid for a given model type"""
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'saga'],
            'penalty': ['l1', 'l2']
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        },
        'SVM': {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.1],
            'kernel': ['rbf']
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001]
        },
        'Decision Tree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski', 'manhattan']
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
    }
    
    return param_grids.get(model_name, {})

def plot_model_comparison(results_df, save_path='model_comparison.png'):
    """Plot a comparison of model performance metrics"""
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(results_df))
    width = 0.2
    
    # Plot metrics as grouped bar chart
    plt.bar(x - width*1.5, results_df['Train Accuracy'], width, label='Train Accuracy')
    plt.bar(x - width/2, results_df['Test Accuracy'], width, label='Test Accuracy')
    plt.bar(x + width/2, results_df['CV Score'], width, label='CV Score')
    plt.bar(x + width*1.5, results_df['ROC AUC'], width, label='ROC AUC')
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.xticks(x, results_df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.ylim(0.9, 1.01)  # Adjust to focus on high accuracy region
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path