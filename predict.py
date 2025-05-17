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
from datetime import datetime


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

    def predict_multiple_channels(self, day_of_week: int, month: int, day: int,
                                minutes_since_midnight: int,
                                channels_2_4ghz: List[int] = None,
                                channels_5ghz: List[int] = None,
                                output_file: str = None) -> pd.DataFrame:
        """
        Make predictions for multiple channels and save results to CSV.
        
        Args:
            day_of_week (int): Day of the week (1-7)
            month (int): Month (1-12)
            day (int): Day of the month (1-31)
            minutes_since_midnight (int): Minutes since midnight (0-1439)
            channels_2_4ghz (List[int]): List of 2.4 GHz channels to predict
            channels_5ghz (List[int]): List of 5 GHz channels to predict
            output_file (str): Path to save the predictions CSV file
            
        Returns:
            pd.DataFrame: DataFrame containing predictions for all channels
        """
        if self.model is None:
            self.load_saved_model()
        
        # Default channels if none provided
        if channels_2_4ghz is None:
            channels_2_4ghz = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        if channels_5ghz is None:
            channels_5ghz = [36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 
                           116, 120, 124, 128, 132, 136, 140, 144, 149, 153, 
                           157, 161, 165]
        
        # Combine all channels
        all_channels = channels_2_4ghz + channels_5ghz
        
        # Create results list
        results = []
        
        # Make predictions for each channel
        for channel in all_channels:
            prediction = self.predict(
                day_of_week=day_of_week,
                month=month,
                day=day,
                minutes_since_midnight=minutes_since_midnight,
                channel=channel
            )
            
            # Add channel and time information
            prediction['channel'] = channel
            prediction['day_of_week'] = day_of_week
            prediction['month'] = month
            prediction['day'] = day
            prediction['minutes_since_midnight'] = minutes_since_midnight
            
            # Convert minutes to time string
            hours = minutes_since_midnight // 60
            minutes = minutes_since_midnight % 60
            prediction['time'] = f"{hours:02d}:{minutes:02d}"
            
            # Add band information
            prediction['band'] = '2.4 GHz' if channel in channels_2_4ghz else '5 GHz'
            
            results.append(prediction)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns
        columns = [
            'channel', 'band', 'day_of_week', 'month', 'day', 'time',
            'minutes_since_midnight', 'avg_signal_strength', 'network_count',
            'total_client_count', 'avg_retransmission_count', 'avg_lost_packets',
            'avg_airtime'
        ]
        df = df[columns]
        
        # Save to CSV if output file specified
        if output_file:
            # Create timestamp for filename if not provided
            if output_file == 'auto':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f'wifi_predictions_{timestamp}.csv'
            
            self.logger.info(f"Saving predictions to {output_file}")
            df.to_csv(output_file, index=False)
        
        return df


def main():
    # Example usage
    predictor = WiFiChannelPredictor()
    
    try:
        # Load and preprocess data
        df = predictor.load_data('aggregated_wifi_data.csv')
        X, y = predictor.preprocess_data(df)
        
        # Train model
        predictor.train_model(X, y)
        
        # Example: Predict for all channels at a specific time
        predictions_df = predictor.predict_multiple_channels(
            day_of_week=1,      # Monday
            month=1,            # January
            day=1,              # 1st day
            minutes_since_midnight=720,  # Noon (12:00)
            output_file='wifi_predictions.csv'
        )
        
        # Print summary of predictions
        print("\nPrediction Summary:")
        print("-" * 50)
        print(f"Total channels predicted: {len(predictions_df)}")
        print(f"2.4 GHz channels: {len(predictions_df[predictions_df['band'] == '2.4 GHz'])}")
        print(f"5 GHz channels: {len(predictions_df[predictions_df['band'] == '5 GHz'])}")
        print("\nSample predictions:")
        print(predictions_df.head())
        
    except Exception as e:
        predictor.logger.error(f"Error: {str(e)}")


if __name__ == '__main__':
    main() 