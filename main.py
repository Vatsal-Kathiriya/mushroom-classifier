import os
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import from the refactored modules
from src.data.processing import DataProcessor
from src.models.training import ModelTrainer
from src.visualization.core import VisualizationManager
from src.models.feature_selection import FeatureSelector
from src.utils.io import save_file, load_file
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

class MushroomClassifier:
    """Main class for mushroom classifier project"""
    
    def __init__(self, args):
        self.args = args
        self.data_processor = DataProcessor(random_state=42)
        self.model_trainer = ModelTrainer(random_state=42)
        self.viz_manager = VisualizationManager()
        self.feature_selector = FeatureSelector()
        
        # Setup directory structure
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = ['models', 'data/processed', 'visualizations', 'logs']
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
    
    def run_phase1(self):
        """Phase 1: Data processing, model training and evaluation"""
        logger.info("===== Running Phase 1: Training =====")
        
        try:
            # Step 1: Load data
            logger.info("Step 1: Loading data...")
            data_path = 'data/processed/secondary_data_processed.csv' if self.args.processed_data else 'data/secondary_data.csv'
            logger.info(f"Using data from: {data_path}")
            
            df = self.data_processor.load_data(data_path, delimiter=';')
            if df is None:
                logger.error("Failed to load data. Exiting Phase 1.")
                return None
            
            # Step 2: Explore data and create visualizations
            logger.info("\nStep 2: Exploring data...")
            self._explore_data(df)
            
            # Step 3: Preprocess data
            logger.info("\nStep 3: Preprocessing data...")
            preprocessed = self.data_processor.preprocess(df, test_size=0.2)
            
            X_train = preprocessed['X_train']
            X_test = preprocessed['X_test']
            y_train = preprocessed['y_train']
            y_test = preprocessed['y_test']
            preprocessor = preprocessed['preprocessor']
            feature_names = preprocessed['feature_names']
            
            # Step 4: Train models
            logger.info("\nStep 4: Training models...")
            models = self.model_trainer.train_models(X_train, y_train)
            
            # Step 5: Evaluate models
            logger.info("\nStep 5: Evaluating models...")
            results_df = self.model_trainer.evaluate_models(
                X_train, X_test, y_train, y_test)
            logger.info("\nModel Performance Comparison:")
            logger.info(results_df)
            
            # Save results for visualization
            joblib.dump(results_df, 'models/model_results.pkl')
            
            # Plot model comparison
            self.viz_manager.plot_model_comparison(results_df)
            
            # Step 6: Hyperparameter tuning of best model
            logger.info("\nStep 6: Tuning best model...")
            best_model = self.model_trainer.tune_model(X_train=X_train, y_train=y_train)
            
            # Step 7: Evaluate tuned model
            logger.info("\nStep 7: Evaluating tuned model...")
            metrics = self.viz_manager.evaluate_and_visualize_model(
                best_model, X_test, y_test, feature_names)
            
            # Step 8: Save models and artifacts
            logger.info("\nStep 8: Saving models and artifacts...")
            self.model_trainer.save_models()
            
            # Save preprocessor and feature names
            joblib.dump(preprocessor, 'models/preprocessor.pkl')
            joblib.dump(feature_names, 'models/feature_names.pkl')
            
            # Save feature map for web app
            feature_map = self.data_processor.create_feature_map(df)
            joblib.dump(feature_map, 'models/feature_map.pkl')
            
            logger.info("Phase 1 completed successfully!")
            
            # Return key components for phase 2
            return {
                'df': df,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'best_model': best_model,
                'preprocessor': preprocessor,
                'feature_names': feature_names
            }
            
        except Exception as e:
            logger.error(f"Error in Phase 1: {str(e)}", exc_info=True)
            return None
    
    def _explore_data(self, df):
        """Create exploratory visualizations of the dataset"""
        logger.info(f"Samples: {df.shape[0]}, Features: {df.shape[1]}")
        logger.info(f"Class distribution:\n{df['class'].value_counts()}")
        
        # Create basic visualizations
        self.viz_manager.plot_distribution(df, feature=None)  # Class distribution
        
        numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.drop('class').tolist()
        
        if numerical_features:
            self.viz_manager.plot_numerical_features(df, numerical_features)
            self.viz_manager.plot_correlation_matrix(df, numerical_features)
        
        if categorical_features:
            self.viz_manager.plot_categorical_features(df, categorical_features)
    
    def run_phase2(self, phase1_results=None):
        """Phase 2: Feature selection and model refinement"""
        logger.info("\n===== Running Phase 2: Feature Selection and Refinement =====")
        
        try:
            # If phase1_results is provided, use it directly
            if phase1_results is not None:
                df = phase1_results['df']
                X_train = phase1_results['X_train']
                X_test = phase1_results['X_test']
                y_train = phase1_results['y_train']
                y_test = phase1_results['y_test']
                best_model = phase1_results['best_model']
                preprocessor = phase1_results['preprocessor']
                feature_names = phase1_results['feature_names']
            else:
                # Load saved models from Phase 1
                logger.info("Loading models from Phase 1...")
                data_path = 'data/processed/secondary_data_processed.csv' if self.args.processed_data else 'data/secondary_data.csv'
                df = self.data_processor.load_data(data_path, delimiter=';')
                
                best_model = load_file('models/best_model.pkl')
                preprocessor = load_file('models/preprocessor.pkl')
                feature_names = load_file('models/feature_names.pkl')
                
                # Ensure required components are loaded
                if any(item is None for item in [df, best_model, preprocessor, feature_names]):
                    logger.error("Required files from Phase 1 not found. Run Phase 1 first.")
                    return None
                
                # Preprocess data again to get train/test split
                logger.info("Preprocessing data...")
                preprocessed = self.data_processor.preprocess(df, test_size=0.2)
                X_train = preprocessed['X_train']
                X_test = preprocessed['X_test']
                y_train = preprocessed['y_train']
                y_test = preprocessed['y_test']
            
            # Step 1: Feature selection
            logger.info("\nStep 1: Selecting important features...")
            feature_selection_results = self.feature_selector.select_features(
                df, best_model, X_train, X_test, y_train, y_test, feature_names,
                methods=['permutation', 'importance', 'mutual_info']
            )
            
            # Step 2: Train model with selected features
            logger.info("\nStep 2: Training model with selected features...")
            refined_model = self.feature_selector.train_with_selected_features(
                best_model, X_train, X_test, y_train, y_test, 
                feature_selection_results['feature_indices']
            )
            
            # Step 3: Evaluate refined model
            logger.info("\nStep 3: Evaluating refined model...")
            metrics = self.viz_manager.evaluate_and_visualize_model(
                refined_model, 
                feature_selection_results['X_test_selected'], 
                y_test,
                feature_selection_results['selected_feature_names']
            )
            
            # Step 4: Generate learning curves
            logger.info("\nStep 4: Generating learning curves...")
            self.viz_manager.plot_learning_curve(refined_model, 
                                               feature_selection_results['X_train_selected'], 
                                               y_train)
            
            # Step 5: Save refined model and feature selection results
            logger.info("\nStep 5: Saving refined model and results...")
            joblib.dump(refined_model, 'models/refined_feature_model.pkl')
            joblib.dump(feature_selection_results, 'models/feature_selection_performance.pkl')
            joblib.dump(feature_selection_results['selected_features'], 'models/top_features.pkl')
            
            logger.info("Phase 2 completed successfully!")
            return refined_model
            
        except Exception as e:
            logger.error(f"Error in Phase 2: {str(e)}", exc_info=True)
            return None
    
    def run_web_app(self):
        """Run the Flask web application"""
        logger.info("\nStarting the web application...")
        logger.info("Access at http://127.0.0.1:5000")
        
        # Check if required files exist
        required_files = [
            'models/refined_feature_model.pkl',
            'models/preprocessor.pkl',
            'models/feature_map.pkl',
            'models/top_features.pkl'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            logger.warning("\nSome required files are missing:")
            for file in missing_files:
                logger.warning(f" - {file}")
            logger.warning("Consider running Phase 2 first: python main.py --phase 2")
        
        # Import and run the Flask app
        try:
            from app import app
            app.run(debug=True)
        except Exception as e:
            logger.error(f"Error starting web application: {str(e)}")

def setup_argparse():
    """Setup command line argument parsing"""
    parser = argparse.ArgumentParser(description='Mushroom Classifier')
    parser.add_argument('--phase', type=int, default=2, choices=[1, 2],
                      help='Phase to run (1: training, 2: refinement)')
    parser.add_argument('--run-app', action='store_true',
                      help='Run the web application')
    parser.add_argument('--generate-processed-csv', action='store_true',
                      help='Generate processed CSV from raw data')
    parser.add_argument('--processed-data', action='store_true',
                      help='Use processed data (from data/processed/)')
    parser.add_argument('--skip-phase1', action='store_true',
                      help='Skip Phase 1 when running Phase 2')
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip model training steps in Phase 1 if models exist')
    parser.add_argument('--param-config', type=str, default=None,
                      help='Path to custom hyperparameter configuration YAML file')
    return parser.parse_args()

def main():
    """Main entry point"""
    # Setup argument parsing
    args = setup_argparse()
    
    # Create classifier instance
    classifier = MushroomClassifier(args)
    
    # Generate processed CSV if requested
    if args.generate_processed_csv:
        logger.info("Generating processed CSV...")
        processor = DataProcessor()
        df = processor.load_data('data/secondary_data.csv', delimiter=';')
        if df is not None:
            df_processed = processor.impute_missing_values(df)
            os.makedirs('data/processed', exist_ok=True)
            output_path = 'data/processed/secondary_data_processed.csv'
            df_processed.to_csv(output_path, index=False, sep=';')
            logger.info(f"Processed CSV saved to {output_path}")

    # Run web app if requested
    if args.run_app:
        classifier.run_web_app()
        return

    # Determine if we should run Phase 1
    run_phase1 = (args.phase == 1 or (args.phase == 2 and not args.skip_phase1))
    
    # Run selected phases
    if run_phase1:
        phase1_results = classifier.run_phase1()
        
        # Run Phase 2 if requested and Phase 1 was successful
        if args.phase == 2 and phase1_results is not None:
            classifier.run_phase2(phase1_results)
    elif args.phase == 2:
        # Run only Phase 2
        classifier.run_phase2()

if __name__ == "__main__":
    main()