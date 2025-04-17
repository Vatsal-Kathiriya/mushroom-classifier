import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_curve, auc)
from sklearn.model_selection import cross_val_score, learning_curve
from .visualization import EDIBLE_COLOR, POISONOUS_COLOR, save_plot

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['edible', 'poisonous']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)"
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=['edible', 'poisonous'],
                yticklabels=['edible', 'poisonous'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'confusion_matrix.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return cm

def plot_roc_curve(y_test, y_pred_proba, save_path=None):
    """Plot ROC curve"""
    if y_pred_proba is None:
        print("Warning: No probability predictions available. Cannot plot ROC curve.")
        return None
        
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'roc_curve.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return roc_auc

def plot_coefficients(model, feature_names, top_n=20, save_path=None, show=False):
    """Plot model coefficients (for logistic regression)"""
    if not hasattr(model, 'coef_'):
        print("Warning: Model does not have coefficients. Cannot plot.")
        return None
        
    coefs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    })
    
    coefs['Abs_Coefficient'] = np.abs(coefs['Coefficient'])
    coefs = coefs.sort_values('Abs_Coefficient', ascending=False).head(top_n)
    coefs = coefs.sort_values('Coefficient', ascending=True)
    
    # Red for positive (predicts poisonous), green for negative (predicts edible)
    colors = [POISONOUS_COLOR if c > 0 else EDIBLE_COLOR for c in coefs['Coefficient']]
    
    plt.figure(figsize=(12, 8))
    plt.barh(coefs['Feature'], coefs['Coefficient'], color=colors)
    plt.title('Logistic Regression Coefficients', fontsize=15)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    elif not show:
        save_plot('logistic_regression_coefficients.png')
    
    if show:
        return plt
    else:
        plt.close()
        return coefs

def plot_learning_curve(model, X, y, cv=5, scoring='accuracy', save_path=None):
    """Plot learning curve showing model performance vs training set size"""
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring, n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color=EDIBLE_COLOR, label=f'Training {scoring}')
    plt.plot(train_sizes, test_mean, 'o-', color=POISONOUS_COLOR, label=f'Cross-validation {scoring}')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color=EDIBLE_COLOR)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color=POISONOUS_COLOR)
    
    plt.title('Learning Curve', fontsize=14)
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel(f'{scoring.capitalize()}', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'learning_curve.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return pd.DataFrame({
        'Train Size': train_sizes,
        f'Train {scoring}': train_mean,
        f'Train Std': train_std,
        f'CV {scoring}': test_mean,
        f'CV Std': test_std
    })

def try_shap_analysis(model, X_train, feature_names, max_display=20, save_path=None):
    """Try to create SHAP summary plot (if shap package is available)"""
    try:
        import shap
        
        if hasattr(model, 'predict_proba'):
            explainer = shap.Explainer(model)
            shap_values = explainer(X_train[:500] if X_train.shape[0] > 500 else X_train)
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, feature_names=feature_names, max_display=max_display, show=False)
            
            if save_path is None:
                save_path = 'shap_summary.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return shap_values
            
    except ImportError:
        print("SHAP package not available. Install with 'pip install shap' for model interpretation.")
    except Exception as e:
        print(f"Error in SHAP analysis: {str(e)}")
    
    return None