#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle
import logging
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wifi_ml.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WiFiMLPredictor:
    def __init__(self, model_dir: str = 'models'):
        """Initialize the WiFi ML Predictor.
        
        Args:
            model_dir: Directory to store model files
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
        # Initialize models
        self.categorical_models: Dict[str, MLPClassifier] = {}
        self.numerical_models: Dict[str, MLPRegressor] = {}
        
        # Store common values for categorical features
        self.common_values = {
            'ssid': None,
            'bssid': None
        }
        
        # Define column order
        self.column_order = [
            'ssid', 'bssid', 'signal_strength', 'channel', 'frequency',
            'security_type', 'phy_rate', 'client_count', 'retransmission_count',
            'lost_packets', 'airtime_ms', 'day_of_week', 'month', 'day',
            'minutes_since_midnight', 'timestamp'
        ]
        
        # Define feature types
        self.categorical_features = [
            'ssid', 'bssid'
        ]
        self.numerical_features = [
            'signal_strength', 'channel', 'frequency', 'phy_rate', 'client_count',
            'retransmission_count', 'lost_packets', 'airtime_ms',
            'day_of_week', 'month', 'day', 'minutes_since_midnight'
        ]
        
        # Define target variables
        self.categorical_targets = []  # No categorical targets
        self.numerical_targets = [
            'signal_strength', 'client_count', 'retransmission_count',
            'lost_packets', 'airtime_ms', 'frequency'
        ]
        
        # Define feature columns (excluding targets)
        self.feature_columns = {
            'categorical': [f for f in self.categorical_features if f not in self.categorical_targets],
            'numerical': [f for f in self.numerical_features if f not in self.numerical_targets]
        }
        
        # Log feature columns for debugging
        logger.debug(f"Categorical features: {self.feature_columns['categorical']}")
        logger.debug(f"Numerical features: {self.feature_columns['numerical']}")
        
        # Initialize performance metrics
        self.performance_history: Dict[str, List[float]] = {
            'numerical_mse': []
        }
        
        # Load existing models if available
        self._load_models()

    def _load_models(self):
        """Load existing models and preprocessing components if available."""
        try:
            # Load feature columns if available
            feature_columns_path = os.path.join(self.model_dir, 'feature_columns.json')
            if os.path.exists(feature_columns_path):
                with open(feature_columns_path, 'r') as f:
                    saved_feature_columns = json.load(f)
                # Update feature columns if they match
                if saved_feature_columns == self.feature_columns:
                    logger.info("Loaded existing feature columns configuration")
                else:
                    logger.warning("Feature columns configuration has changed. Will reinitialize scaler.")
                    self.scaler = StandardScaler()

            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded existing scaler with {len(self.scaler.mean_)} features")

            # Load label encoders
            for feature in self.categorical_features:
                encoder_path = os.path.join(self.model_dir, f'encoder_{feature}.pkl')
                if os.path.exists(encoder_path):
                    with open(encoder_path, 'rb') as f:
                        self.label_encoders[feature] = pickle.load(f)
                    logger.info(f"Loaded existing encoder for {feature}")

            # Load categorical models
            for target in self.categorical_targets:
                model_path = os.path.join(self.model_dir, f'cat_model_{target}.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.categorical_models[target] = pickle.load(f)
                    logger.info(f"Loaded existing categorical model for {target}")

            # Load numerical models
            for target in self.numerical_targets:
                model_path = os.path.join(self.model_dir, f'num_model_{target}.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.numerical_models[target] = pickle.load(f)
                    logger.info(f"Loaded existing numerical model for {target}")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error("Stack trace:", exc_info=True)
            self._initialize_new_models()

    def _initialize_new_models(self):
        """Initialize new models if none exist."""
        # Initialize categorical models
        for target in self.categorical_targets:
            self.categorical_models[target] = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1,
                warm_start=True,
                random_state=42
            )
            logger.info(f"Initialized new categorical model for {target}")

        # Initialize numerical models
        for target in self.numerical_targets:
            self.numerical_models[target] = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=1,
                warm_start=True,
                random_state=42
            )
            logger.info(f"Initialized new numerical model for {target}")

    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Preprocess the input data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (numerical_features, categorical_features)
        """
        try:
            # Log available columns
            logger.debug(f"Available columns: {data.columns.tolist()}")
            
            # Create a copy of the data to avoid modifying the original
            data = data.copy()
            
            # Ensure all required columns exist
            missing_columns = [col for col in self.column_order if col not in data.columns]
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
                # Add missing columns with default values
                for col in missing_columns:
                    if col in self.categorical_features:
                        data[col] = 'unknown'
                    elif col in self.numerical_features:
                        data[col] = 0
                    elif col == 'timestamp':
                        data[col] = pd.Timestamp.now()
            
            # Ensure columns are in the correct order
            data = data.reindex(columns=self.column_order)
            
            # Log data after reindexing
            logger.debug(f"Columns after reindexing: {data.columns.tolist()}")
            logger.debug(f"Data shape: {data.shape}")
            
            # Convert timestamp to datetime if it's not already
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Process categorical features (excluding targets)
            categorical_data = {}
            for feature in self.feature_columns['categorical']:
                try:
                    if feature not in self.label_encoders:
                        self.label_encoders[feature] = LabelEncoder()
                        # Handle potential NaN values
                        unique_values = data[feature].fillna('unknown').unique()
                        self.label_encoders[feature].fit(unique_values)
                        logger.debug(f"Fitted encoder for {feature} with classes: {unique_values}")
                    
                    # Transform with handling of NaN values
                    categorical_data[feature] = self.label_encoders[feature].transform(
                        data[feature].fillna('unknown')
                    )
                    logger.debug(f"Successfully processed categorical feature: {feature}")
                except Exception as e:
                    logger.error(f"Error processing categorical feature {feature}: {str(e)}")
                    logger.error(f"Data for {feature}:\n{data[feature].head()}")
                    raise

            # Process numerical features (excluding targets)
            try:
                # Get numerical features excluding targets
                numerical_features = [f for f in self.numerical_features if f not in self.numerical_targets]
                logger.debug(f"Processing numerical features: {numerical_features}")
                
                numerical_data = data[numerical_features].fillna(0).values
                logger.debug(f"Numerical data shape before scaling: {numerical_data.shape}")
                
                # Check if we need to reinitialize the scaler
                if hasattr(self.scaler, 'mean_'):
                    expected_features = len(self.scaler.mean_)
                    actual_features = numerical_data.shape[1]
                    if expected_features != actual_features:
                        logger.warning(f"Feature count mismatch: expected {expected_features}, got {actual_features}. Reinitializing scaler.")
                        self.scaler = StandardScaler()
                
                if not hasattr(self.scaler, 'mean_'):
                    self.scaler.fit(numerical_data)
                    logger.debug("Fitted new scaler")
                    logger.debug(f"Scaler feature count: {len(self.scaler.mean_)}")
                
                numerical_data = self.scaler.transform(numerical_data)
                logger.debug(f"Numerical data shape after scaling: {numerical_data.shape}")
            except Exception as e:
                logger.error(f"Error processing numerical features: {str(e)}")
                logger.error(f"Numerical features being processed: {numerical_features}")
                logger.error(f"Numerical data head:\n{data[numerical_features].head()}")
                raise

            return numerical_data, categorical_data

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            logger.error(f"DataFrame info:\n{data.info()}")
            logger.error(f"DataFrame head:\n{data.head()}")
            raise

    def _save_models(self):
        """Save all models and preprocessing components."""
        try:
            # Save scaler
            with open(os.path.join(self.model_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.debug(f"Saved scaler with {len(self.scaler.mean_)} features")

            # Save feature columns for reference
            with open(os.path.join(self.model_dir, 'feature_columns.json'), 'w') as f:
                json.dump(self.feature_columns, f)
            logger.debug("Saved feature columns configuration")

            # Save label encoders
            for feature, encoder in self.label_encoders.items():
                with open(os.path.join(self.model_dir, f'encoder_{feature}.pkl'), 'wb') as f:
                    pickle.dump(encoder, f)
                logger.debug(f"Saved encoder for {feature}")

            # Save categorical models
            for target, model in self.categorical_models.items():
                with open(os.path.join(self.model_dir, f'cat_model_{target}.pkl'), 'wb') as f:
                    pickle.dump(model, f)
                logger.debug(f"Saved categorical model for {target}")

            # Save numerical models
            for target, model in self.numerical_models.items():
                with open(os.path.join(self.model_dir, f'num_model_{target}.pkl'), 'wb') as f:
                    pickle.dump(model, f)
                logger.debug(f"Saved numerical model for {target}")

            logger.info("Successfully saved all models and components")

        except Exception as e:
            logger.error(f"Error saving models: {e}")
            logger.error("Stack trace:", exc_info=True)

    def update(self, data: pd.DataFrame):
        """Update models with new data.
        
        Args:
            data: New data for model update
        """
        try:
            # Log input data shape and columns
            logger.debug(f"Updating models with data shape: {data.shape}")
            logger.debug(f"Input data columns: {data.columns.tolist()}")
            
            # Update common values for categorical features
            for feature in self.categorical_features:
                if feature in data.columns:
                    value_counts = data[feature].value_counts()
                    if not value_counts.empty:
                        self.common_values[feature] = value_counts.index[0]
                        logger.debug(f"Updated common value for {feature}: {self.common_values[feature]}")
            
            # Preprocess data
            numerical_data, categorical_data = self._preprocess_data(data)
            
            # Update numerical models
            for target in self.numerical_targets:
                if target in data.columns:
                    try:
                        logger.debug(f"Updating numerical model for {target}")
                        
                        # Create feature matrix
                        X = numerical_data
                        if categorical_data:
                            X = np.column_stack([X] + list(categorical_data.values()))
                        
                        logger.debug(f"Feature matrix shape for {target}: {X.shape}")
                        
                        y = data[target].fillna(0).values
                        
                        if target not in self.numerical_models:
                            self.numerical_models[target] = MLPRegressor(
                                hidden_layer_sizes=(100, 50),
                                max_iter=1,
                                warm_start=True,
                                random_state=42
                            )
                            logger.debug(f"Initialized new numerical model for {target}")
                        
                        # Fit or update model
                        if not hasattr(self.numerical_models[target], 'n_outputs_'):
                            logger.debug(f"First fit for {target} model")
                            self.numerical_models[target].fit(X, y)
                        else:
                            logger.debug(f"Partial fit for {target} model")
                            self.numerical_models[target].partial_fit(X, y)
                        
                        # Calculate and log MSE
                        y_pred = self.numerical_models[target].predict(X)
                        mse = mean_squared_error(y, y_pred)
                        self.performance_history['numerical_mse'].append(mse)
                        logger.info(f"Updated numerical model for {target}, MSE: {mse:.4f}")
                    except Exception as e:
                        logger.error(f"Error updating numerical model for {target}: {str(e)}")
                        logger.error(f"Data for {target}:\n{data[target].head()}")
                        continue

            # Save models periodically
            self._save_models()

        except Exception as e:
            logger.error(f"Error updating models: {str(e)}")
            logger.error(f"DataFrame info:\n{data.info()}")
            logger.error(f"DataFrame head:\n{data.head()}")
            logger.error("Stack trace:", exc_info=True)

    def predict(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions using the trained models.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Dictionary of predictions for each target variable
        """
        try:
            # Preprocess data
            numerical_data, categorical_data = self._preprocess_data(data)
            
            predictions = {}
            
            # Make numerical predictions
            for target in self.numerical_targets:
                try:
                    if target not in self.numerical_models:
                        logger.warning(f"No model available for {target}")
                        continue
                        
                    # Create feature matrix
                    X = numerical_data
                    if categorical_data:
                        X = np.column_stack([X] + list(categorical_data.values()))
                    
                    logger.debug(f"Feature matrix shape for {target} prediction: {X.shape}")
                    
                    # Check if model is fitted
                    if not hasattr(self.numerical_models[target], 'n_outputs_'):
                        logger.warning(f"Model for {target} is not fitted yet")
                        continue
                    
                    predictions[target] = self.numerical_models[target].predict(X)
                    logger.debug(f"Successfully made predictions for {target}")
                except Exception as e:
                    logger.error(f"Error making numerical prediction for {target}: {str(e)}")
                    logger.error(f"Feature matrix shape: {X.shape if 'X' in locals() else 'Not created'}")
                    continue

            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            logger.error(f"DataFrame info:\n{data.info()}")
            logger.error(f"DataFrame head:\n{data.head()}")
            return {}

    def get_performance_metrics(self) -> Dict[str, List[float]]:
        """Get the performance history of the models.
        
        Returns:
            Dictionary containing performance metrics history
        """
        return self.performance_history

    def predict_for_datetime(self, date: datetime, output_file: str = 'predictions.csv') -> None:
        """Make predictions for a specific date and time.
        
        Args:
            date: datetime object for which to make predictions
            output_file: Path to save the predictions CSV
        """
        try:
            # Create a DataFrame with the specified date
            data = pd.DataFrame({
                'day_of_week': [date.weekday()],
                'month': [date.month],
                'day': [date.day],
                'minutes_since_midnight': [date.hour * 60 + date.minute],
                'timestamp': [date]
            })
            
            # Add common values for categorical features
            for feature in self.categorical_features:
                if self.common_values[feature] is not None:
                    data[feature] = self.common_values[feature]
                else:
                    logger.warning(f"No common value available for {feature}")
                    return
            
            # Add default values for numerical features
            data['signal_strength'] = 0
            data['channel'] = 0
            data['frequency'] = 0
            data['phy_rate'] = 0
            data['client_count'] = 0
            data['retransmission_count'] = 0
            data['lost_packets'] = 0
            data['airtime_ms'] = 0
            
            # Make predictions
            predictions = self.predict(data)
            
            if not predictions:
                logger.error("No predictions were made")
                return
            
            # Create a DataFrame with the predictions
            results = pd.DataFrame({
                'timestamp': [date],
                'day_of_week': [date.weekday()],
                'month': [date.month],
                'day': [date.day],
                'minutes_since_midnight': [date.hour * 60 + date.minute]
            })
            
            # Add predictions to the results
            for target, preds in predictions.items():
                results[f'predicted_{target}'] = preds
            
            # Save to CSV
            results.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to {output_file}")
            
            # Print the predictions
            logger.info("\nPredictions:")
            for target, preds in predictions.items():
                logger.info(f"{target}: {preds[0]:.2f}")
                
        except Exception as e:
            logger.error(f"Error making predictions for datetime: {str(e)}")
            logger.error("Stack trace:", exc_info=True)

def main():
    """Main function to demonstrate usage."""
    # Initialize predictor
    predictor = WiFiMLPredictor()
    
    # Example: Load and process data in batches
    try:
        # Read data in chunks
        for chunk in pd.read_csv('wifi_data.csv', chunksize=1000):
            # Log chunk information
            logger.info(f"Processing chunk with shape: {chunk.shape}")
            logger.debug(f"Chunk columns: {chunk.columns.tolist()}")
            logger.debug(f"Chunk head:\n{chunk.head()}")
            
            # Ensure timestamp column is properly parsed
            if 'timestamp' in chunk.columns:
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
            
            # Update models with new data
            predictor.update(chunk)
            
            # Make predictions
            predictions = predictor.predict(chunk)
            
            # Log some example predictions
            logger.info("Example predictions:")
            for target, preds in predictions.items():
                logger.info(f"{target}: {preds[:5]}")  # Show first 5 predictions
            
            # Get and log performance metrics
            metrics = predictor.get_performance_metrics()
            logger.info("Current performance metrics:")
            for metric, values in metrics.items():
                if values:
                    logger.info(f"{metric}: {values[-1]:.4f}")

        # Example: Make predictions for a specific date and time
        target_date = datetime.now()
        predictor.predict_for_datetime(target_date, 'predictions.csv')

    except Exception as e:
        logger.error(f"Error in main processing loop: {str(e)}")
        logger.error("Stack trace:", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WiFi ML Predictor')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--predict', action='store_true', help='Make predictions for current time')
    parser.add_argument('--date', type=str, help='Date for predictions (YYYY-MM-DD HH:MM)')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output file for predictions')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    if args.predict or args.date:
        predictor = WiFiMLPredictor()
        if args.date:
            try:
                target_date = datetime.strptime(args.date, '%Y-%m-%d %H:%M')
            except ValueError:
                logger.error("Invalid date format. Use YYYY-MM-DD HH:MM")
                exit(1)
        else:
            target_date = datetime.now()
        
        predictor.predict_for_datetime(target_date, args.output)
    else:
        main() 