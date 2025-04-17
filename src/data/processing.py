import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataProcessor:
    """Data processing class for mushroom classification data"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def load_data(self, file_path, delimiter=';'):
        """Load the mushroom dataset with error handling"""
        try:
            df = pd.read_csv(file_path, delimiter=delimiter)
            print(f"Successfully loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def impute_missing_values(self, df):
        """Impute missing values in the dataframe"""
        print("Imputing missing values...")
        df = df.replace('', np.nan)

        missing_before = df.isnull().sum().sum()
        print(f"Total missing values before imputation: {missing_before}")

        # Skip imputation if no missing values
        if missing_before == 0:
            print("No missing values to impute.")
            return df

        # Impute numerical columns with median
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in numerical_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  Imputed numerical column '{col}' with median ({median_val})")

        # Impute categorical columns with mode
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        for col in categorical_cols:
            if col == 'class' and 'class' in df.columns:
                 continue
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"  Imputed categorical column '{col}' with mode ('{mode_val}')")

        missing_after = df.isnull().sum().sum()
        print(f"Missing values after imputation: {missing_after}")
        
        return df
        
    def preprocess(self, df, test_size=0.2):
        """Complete preprocessing pipeline for mushroom dataset"""
        # Impute missing values
        df_imputed = self.impute_missing_values(df.copy())

        # Prepare features and target
        if 'class' not in df_imputed.columns:
            raise ValueError("Target column 'class' not found in the dataframe.")

        y = df_imputed['class'].map({'e': 0, 'p': 1})
        X = df_imputed.drop('class', axis=1)

        # Identify column types
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X.select_dtypes(include='object').columns.tolist()
        print(f"\nNumerical columns: {numerical_cols}")
        print(f"Categorical columns: {categorical_cols}")

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ],
            remainder='passthrough'
        )

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Fit and transform
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Extract feature names
        feature_names = self._extract_feature_names(preprocessor, numerical_cols, categorical_cols)

        print(f"Processed data shapes: X_train {X_train_processed.shape}, X_test {X_test_processed.shape}")

        return {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_test': y_test,
            'preprocessor': preprocessor,
            'feature_names': feature_names
        }
        
    def _extract_feature_names(self, preprocessor, numerical_cols, categorical_cols):
        """Extract feature names from column transformer"""
        feature_names = []
        
        # Add numerical features
        feature_names.extend(numerical_cols)
        
        # Add one-hot encoded features
        cat_transformer = preprocessor.named_transformers_['cat']
        if hasattr(cat_transformer, 'get_feature_names_out'):
            cat_features = cat_transformer.get_feature_names_out(categorical_cols)
            feature_names.extend(cat_features)
        else:
            # Fallback for older scikit-learn versions
            for col in categorical_cols:
                cats = cat_transformer.categories_[categorical_cols.index(col)]
                for cat in cats:
                    feature_names.append(f"{col}_{cat}")
                    
        return feature_names
        
    def create_feature_map(self, df):
        """Create a map of features and their possible values for the webapp"""
        feature_map = {}
        
        for col in df.columns:
            if col == 'class':
                continue
                
            if df[col].dtype.name in ['int64', 'float64']:
                feature_map[col] = {
                    'type': 'numerical',
                    'values': [float(df[col].min()), float(df[col].max())]
                }
            else:
                feature_map[col] = {
                    'type': 'categorical',
                    'values': sorted(df[col].unique().tolist())
                }
        
        return feature_map