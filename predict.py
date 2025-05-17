#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path
import logging
from typing import Tuple, Dict, List
import json


class WiFiChannelPredictor:
    def __init__(self, model_path: str = 'wifi_model.joblib', scaler_path: str = 'scaler.joblib'):
        """
        Initialize the WiFi Channel Predictor.
        
        Args:
            model_path (str): Path to save/load the trained model
            scaler_path (str): Path to save/load the feature scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['day_of_week', 'month', 'day', 'minutes_since_midnight', 'channel']
        self.target_columns = [
            'avg_signal_strength', 'network_count', 'total_client_count',
            'avg_retransmission_count', 'avg_lost_packets', 'avg_airtime'
        ]
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load and preprocess the data from CSV.
        
        Args:
            csv_path (str): Path to the input CSV file
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        self.logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Verify required columns exist
        required_columns = self.feature_columns + self.target_columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data for model training.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed features and targets
        """
        self.logger.info("Preprocessing data")
        
        # Split features and targets
        X = df[self.feature_columns].values
        y = df[self.target_columns].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y

    def create_model(self) -> MLPRegressor:
        """
        Create and configure the neural network model.
        
        Returns:
            MLPRegressor: Configured neural network model
        """
        self.logger.info("Creating neural network model")
        
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            random_state=42
        )
        
        return model

    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the neural network model.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training targets
        """
        self.logger.info("Training model")
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train model
        self.model = self.create_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        self._evaluate_model(y_test, y_pred)
        
        # Save model and scaler
        self._save_model()

    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Evaluate the model performance.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
        """
        self.logger.info("Evaluating model performance")
        
        # Calculate metrics for each target variable
        metrics = {}
        for i, col in enumerate(self.target_columns):
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            metrics[col] = {'MSE': mse, 'R2': r2}
        
        # Print evaluation results
        print("\nModel Evaluation Results:")
        print("-" * 50)
        for col, scores in metrics.items():
            print(f"\n{col}:")
            print(f"  MSE: {scores['MSE']:.4f}")
            print(f"  R2 Score: {scores['R2']:.4f}")
        
        # Overall R2 score
        overall_r2 = r2_score(y_true, y_pred)
        print(f"\nOverall R2 Score: {overall_r2:.4f}")

    def _save_model(self) -> None:
        """Save the trained model and scaler."""
        self.logger.info("Saving model and scaler")
        
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def load_saved_model(self) -> None:
        """Load the saved model and scaler."""
        self.logger.info("Loading saved model and scaler")
        
        if not Path(self.model_path).exists() or not Path(self.scaler_path).exists():
            raise FileNotFoundError("Model or scaler files not found")
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def predict(self, day_of_week: int, month: int, day: int,
                minutes_since_midnight: int, channel: int) -> Dict[str, float]:
        """
        Make predictions for a specific time and channel.
        
        Args:
            day_of_week (int): Day of the week (1-7)
            month (int): Month (1-12)
            day (int): Day of the month (1-31)
            minutes_since_midnight (int): Minutes since midnight (0-1439)
            channel (int): Wi-Fi channel number
            
        Returns:
            Dict[str, float]: Predicted parameters
        """
        if self.model is None:
            self.load_saved_model()
        
        # Create input array
        X = np.array([[day_of_week, month, day, minutes_since_midnight, channel]])
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        y_pred = self.model.predict(X_scaled)[0]
        
        # Create result dictionary
        result = dict(zip(self.target_columns, y_pred))
        
        return result

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Update the model with new data using partial fit.
        
        Args:
            X (np.ndarray): New training features
            y (np.ndarray): New training targets
        """
        self.logger.info("Updating model with new data")
        
        if self.model is None:
            self.load_saved_model()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Update model
        self.model.partial_fit(X_scaled, y)
        
        # Save updated model
        self._save_model()


def main():
    # Example usage
    predictor = WiFiChannelPredictor()
    
    try:
        # Load and preprocess data
        df = predictor.load_data('aggregated_wifi_data.csv')
        X, y = predictor.preprocess_data(df)
        
        # Train model
        predictor.train_model(X, y)
        
        # Example prediction
        prediction = predictor.predict(
            day_of_week=1,
            month=1,
            day=1,
            minutes_since_midnight=720,  # Noon
            channel=1
        )
        
        print("\nExample Prediction:")
        print(json.dumps(prediction, indent=2))
        
    except Exception as e:
        predictor.logger.error(f"Error: {str(e)}")


if __name__ == '__main__':
    main() 