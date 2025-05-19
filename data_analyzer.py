#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import argparse


class WiFiChannelAnalyzer:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the WiFi Channel Analyzer.
        
        Args:
            weights (Dict[str, float], optional): Custom weights for channel load calculation.
                Default weights are:
                - signal_strength: -1.0 (negative because lower is better)
                - network_count: 2.0
                - client_count: 3.0
                - retransmission_count: 2.0
                - lost_packets: 1.0
                - airtime: 2.0
        """
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Default weights for channel load calculation
        self.default_weights = {
            'signal_strength': -1.0,  # Negative because lower is better
            'network_count': 2.0,
            'client_count': 3.0,
            'retransmission_count': 2.0,
            'lost_packets': 1.0,
            'airtime': 2.0
        }
        
        # Use custom weights if provided
        self.weights = weights if weights is not None else self.default_weights
        
        # Channel spacing requirements (in MHz)
        self.channel_spacing = {
            '2.4 GHz': 5,  # 5 MHz spacing for 2.4 GHz
            '5 GHz': 20    # 20 MHz spacing for 5 GHz
        }
        
        # Load thresholds
        self.load_thresholds = {
            'low': 0.3,    # Below 30% load
            'medium': 0.6,  # Below 60% load
            'high': 1.0     # Above 60% load
        }
        
        # Stability threshold (variance in load score)
        self.stability_threshold = 0.2  # 20% variance threshold

    def load_data(self, file_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load prediction data from one or multiple CSV files.
        
        Args:
            file_paths (List[str]): List of paths to CSV files containing predictions
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping point names to DataFrames
        """
        self.logger.info(f"Loading data from {len(file_paths)} files")
        
        data_dict = {}
        for file_path in file_paths:
            try:
                # Get point name from filename
                point_name = Path(file_path).stem
                
                # Load data
                df = pd.read_csv(file_path)
                
                # Verify required columns
                required_columns = [
                    'channel', 'band', 'day_of_week', 'month', 'day',
                    'time', 'minutes_since_midnight', 'avg_signal_strength',
                    'network_count', 'total_client_count', 'avg_retransmission_count',
                    'avg_lost_packets', 'avg_airtime'
                ]
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns in {file_path}: {missing_columns}")
                
                data_dict[point_name] = df
                self.logger.info(f"Successfully loaded data for point: {point_name}")
                
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {str(e)}")
                raise
        
        return data_dict

    def calculate_channel_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate channel metrics for a single point.
        
        Args:
            df (pd.DataFrame): DataFrame containing predictions
            
        Returns:
            pd.DataFrame: DataFrame with calculated metrics for each channel
        """
        self.logger.info("Calculating channel metrics")
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Calculate load scores for each row using .loc
        df.loc[:, 'load_score'] = df.apply(self._calculate_load_score, axis=1)
        
        # Group by channel and calculate metrics
        metrics = df.groupby(['channel', 'band']).agg({
            'avg_signal_strength': 'mean',
            'network_count': 'mean',
            'total_client_count': 'mean',
            'avg_retransmission_count': 'mean',
            'avg_lost_packets': 'mean',
            'avg_airtime': 'mean',
            'load_score': ['mean', 'std']  # Use std instead of var for better stability
        }).reset_index()
        
        # Flatten the multi-level columns
        metrics.columns = ['channel', 'band', 
                         'avg_signal_strength', 'network_count', 'total_client_count',
                         'avg_retransmission_count', 'avg_lost_packets', 'avg_airtime',
                         'load_score_mean', 'load_score_std']
        
        # Calculate stability (1 - normalized standard deviation)
        # Normalize std by mean to get coefficient of variation
        metrics['stability'] = 1 - (metrics['load_score_std'] / metrics['load_score_mean']).fillna(0)
        
        # Ensure stability is between 0 and 1
        metrics['stability'] = metrics['stability'].clip(0, 1)
        
        return metrics

    def _calculate_load_score(self, row: pd.Series) -> float:
        """
        Calculate load score for a single prediction.
        
        Args:
            row (pd.Series): Single row of prediction data
            
        Returns:
            float: Calculated load score
        """
        # Normalize signal strength (convert to 0-1 scale where 0 is best)
        signal_score = (row['avg_signal_strength'] + 100) / 100  # Assuming -100 dBm is worst
        
        # Calculate weighted sum
        score = (
            self.weights['signal_strength'] * signal_score +
            self.weights['network_count'] * row['network_count'] +
            self.weights['client_count'] * row['total_client_count'] +
            self.weights['retransmission_count'] * row['avg_retransmission_count'] +
            self.weights['lost_packets'] * row['avg_lost_packets'] +
            self.weights['airtime'] * (row['avg_airtime'] / 1000)  # Convert to seconds
        )
        
        return score

    def analyze_point(self, df: pd.DataFrame) -> Dict:
        """
        Analyze a single point and provide channel recommendations.
        
        Args:
            df (pd.DataFrame): DataFrame containing predictions for a point
            
        Returns:
            Dict: Analysis results including recommendations
        """
        self.logger.info("Analyzing point data")
        
        # Create a copy and sort by time
        df = df.copy()
        df = df.sort_values('minutes_since_midnight')
        
        # Calculate metrics for each time period
        time_periods = []
        current_period = {
            'start_time': df.iloc[0]['minutes_since_midnight'],
            'end_time': None,
            'recommendations': {}
        }
        
        # Minimum duration for a channel recommendation (in minutes)
        min_duration = 60  # 1 hour minimum
        
        # Analyze each band separately
        for band in ['2.4 GHz', '5 GHz']:
            # Create a copy of the band-specific DataFrame
            band_df = df[df['band'] == band].copy()
            if band_df.empty:
                continue
                
            # Calculate metrics for each time point
            time_metrics = []
            for time in band_df['minutes_since_midnight'].unique():
                time_df = band_df[band_df['minutes_since_midnight'] == time].copy()
                metrics = self.calculate_channel_metrics(time_df)
                time_metrics.append({
                    'time': time,
                    'metrics': metrics
                })
            
            # Find stable channel periods
            current_channel = None
            period_start = None
            period_metrics = None
            
            for i, time_point in enumerate(time_metrics):
                metrics = time_point['metrics']
                if metrics.empty:
                    continue
                
                # Create a copy of metrics for sorting
                metrics = metrics.copy()
                
                # Sort by combined score (using stability in the calculation)
                metrics.loc[:, 'combined_score'] = (
                    metrics['load_score_mean'] * (1 - metrics['stability'])
                )
                metrics = metrics.sort_values('combined_score')
                
                best_channel = int(metrics.iloc[0]['channel'])
                best_score = float(metrics.iloc[0]['combined_score'])
                
                if current_channel is None:
                    # First time point
                    current_channel = best_channel
                    period_start = time_point['time']
                    period_metrics = metrics.iloc[0]
                elif best_channel != current_channel:
                    # Channel change detected
                    if time_point['time'] - period_start >= min_duration:
                        # Only record change if period was long enough
                        time_periods.append({
                            'band': band,
                            'start_time': period_start,
                            'end_time': time_point['time'],
                            'channel': current_channel,
                            'load_score': float(period_metrics['load_score_mean']),
                            'stability': float(period_metrics['stability']),
                            'metrics': {
                                'avg_signal_strength': float(period_metrics['avg_signal_strength']),
                                'network_count': float(period_metrics['network_count']),
                                'client_count': float(period_metrics['total_client_count']),
                                'retransmission_count': float(period_metrics['avg_retransmission_count']),
                                'lost_packets': float(period_metrics['avg_lost_packets']),
                                'airtime': float(period_metrics['avg_airtime'])
                            }
                        })
                        current_channel = best_channel
                        period_start = time_point['time']
                        period_metrics = metrics.iloc[0]
                    else:
                        # Ignore short-term changes
                        continue
            
            # Add final period
            if current_channel is not None:
                time_periods.append({
                    'band': band,
                    'start_time': period_start,
                    'end_time': band_df['minutes_since_midnight'].max(),
                    'channel': current_channel,
                    'load_score': float(period_metrics['load_score_mean']),
                    'stability': float(period_metrics['stability']),
                    'metrics': {
                        'avg_signal_strength': float(period_metrics['avg_signal_strength']),
                        'network_count': float(period_metrics['network_count']),
                        'client_count': float(period_metrics['total_client_count']),
                        'retransmission_count': float(period_metrics['avg_retransmission_count']),
                        'lost_packets': float(period_metrics['avg_lost_packets']),
                        'airtime': float(period_metrics['avg_airtime'])
                    }
                })
        
        return {'time_periods': time_periods}

    def analyze_multiple_points(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze multiple points and provide combined recommendations.
        
        Args:
            data_dict (Dict[str, pd.DataFrame]): Dictionary mapping point names to DataFrames
            
        Returns:
            Dict: Combined analysis results
        """
        self.logger.info(f"Analyzing {len(data_dict)} points")
        
        # Analyze each point
        point_analyses = {}
        for point_name, df in data_dict.items():
            point_analyses[point_name] = self.analyze_point(df)
        
        # Check for channel conflicts
        conflicts = self._detect_conflicts(point_analyses)
        
        # Resolve conflicts and provide final recommendations
        final_recommendations = self._resolve_conflicts(point_analyses, conflicts)
        
        return {
            'point_analyses': point_analyses,
            'conflicts': conflicts,
            'final_recommendations': final_recommendations
        }

    def _detect_conflicts(self, point_analyses: Dict) -> List[Dict]:
        """
        Detect channel conflicts between points.
        
        Args:
            point_analyses (Dict): Dictionary of point analyses
            
        Returns:
            List[Dict]: List of detected conflicts
        """
        conflicts = []
        points = list(point_analyses.keys())
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                point1, point2 = points[i], points[j]
                
                # Get current recommendations for each point
                point1_periods = point_analyses[point1]['time_periods']
                point2_periods = point_analyses[point2]['time_periods']
                
                # Group periods by band
                point1_bands = {}
                point2_bands = {}
                
                for period in point1_periods:
                    band = period['band']
                    if band not in point1_bands:
                        point1_bands[band] = []
                    point1_bands[band].append(period)
                
                for period in point2_periods:
                    band = period['band']
                    if band not in point2_bands:
                        point2_bands[band] = []
                    point2_bands[band].append(period)
                
                # Check for conflicts in each band
                for band in ['2.4 GHz', '5 GHz']:
                    if band not in point1_bands or band not in point2_bands:
                        continue
                    
                    # Get current channels for each point
                    point1_channel = int(point1_bands[band][0]['channel'])  # Use first period
                    point2_channel = int(point2_bands[band][0]['channel'])  # Use first period
                    
                    # Check if channels are too close
                    if self._channels_too_close(point1_channel, point2_channel, band):
                        conflicts.append({
                            'point1': point1,
                            'point2': point2,
                            'band': band,
                            'channel1': point1_channel,
                            'channel2': point2_channel
                        })
        
        return conflicts

    def _channels_too_close(self, channel1: int, channel2: int, band: str) -> bool:
        """
        Check if two channels are too close in frequency.
        
        Args:
            channel1 (int): First channel number
            channel2 (int): Second channel number
            band (str): Frequency band
            
        Returns:
            bool: True if channels are too close
        """
        spacing = self.channel_spacing[band]
        return abs(channel1 - channel2) < spacing

    def _resolve_conflicts(self, point_analyses: Dict, conflicts: List[Dict]) -> Dict:
        """
        Resolve channel conflicts and provide final recommendations.
        
        Args:
            point_analyses (Dict): Dictionary of point analyses
            conflicts (List[Dict]): List of detected conflicts
            
        Returns:
            Dict: Final recommendations with conflict resolutions
        """
        final_recommendations = {}
        
        # Start with current recommendations
        for point, analysis in point_analyses.items():
            final_recommendations[point] = {}
            for period in analysis['time_periods']:
                band = period['band']
                if band not in final_recommendations[point]:
                    final_recommendations[point][band] = {
                        'recommended_channel': int(period['channel'])
                    }
        
        # Resolve conflicts
        for conflict in conflicts:
            point1, point2 = conflict['point1'], conflict['point2']
            band = conflict['band']
            
            # Get alternative channels for both points
            alt_channels1 = self._get_alternative_channels(
                final_recommendations[point1][band]['recommended_channel'],
                band,
                final_recommendations
            )
            alt_channels2 = self._get_alternative_channels(
                final_recommendations[point2][band]['recommended_channel'],
                band,
                final_recommendations
            )
            
            # Choose best alternative
            if alt_channels1 and alt_channels2:
                # Use the alternative with better load score
                if alt_channels1[0]['load_score'] < alt_channels2[0]['load_score']:
                    final_recommendations[point1][band]['recommended_channel'] = int(alt_channels1[0]['channel'])
                else:
                    final_recommendations[point2][band]['recommended_channel'] = int(alt_channels2[0]['channel'])
        
        return final_recommendations

    def _get_alternative_channels(self, current_channel: int, band: str,
                                current_recommendations: Dict) -> List[Dict]:
        """
        Get alternative channels that don't conflict with current recommendations.
        
        Args:
            current_channel (int): Current channel number
            band (str): Frequency band
            current_recommendations (Dict): Current channel recommendations
            
        Returns:
            List[Dict]: List of alternative channels with their metrics
        """
        # Get all channels for the band
        if band == '2.4 GHz':
            channels = list(range(1, 14))
        else:  # 5 GHz
            channels = [36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112,
                       116, 120, 124, 128, 132, 136, 140, 144, 149, 153,
                       157, 161, 165]
        
        # Filter out channels that conflict with current recommendations
        valid_channels = []
        for channel in channels:
            if channel == current_channel:
                continue
            
            # Check if channel conflicts with any current recommendation
            conflicts = False
            for point, recommendations in current_recommendations.items():
                if band in recommendations:
                    if self._channels_too_close(channel, recommendations[band]['recommended_channel'], band):
                        conflicts = True
                        break
            
            if not conflicts:
                valid_channels.append({
                    'channel': channel,
                    'load_score': 0.0  # Placeholder load score
                })
        
        return valid_channels

    def save_analysis(self, analysis_results: Dict, output_path: str) -> None:
        """
        Save analysis results to a file.
        
        Args:
            analysis_results (Dict): Analysis results
            output_path (str): Path to save the results
        """
        self.logger.info(f"Saving analysis results to {output_path}")
        
        # Convert to JSON-serializable format
        serializable_results = json.loads(json.dumps(analysis_results, default=str))
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info("Analysis results saved successfully")

    def print_summary(self, analysis_results: Dict) -> None:
        """
        Print a summary of the analysis results.
        
        Args:
            analysis_results (Dict): Analysis results
        """
        print("\nWi-Fi Channel Analysis Summary")
        print("=" * 50)
        
        # Print point analyses
        for point, analysis in analysis_results['point_analyses'].items():
            print(f"\nPoint: {point}")
            print("-" * 30)
            
            # Group periods by band
            band_periods = {}
            for period in analysis['time_periods']:
                band = period['band']
                if band not in band_periods:
                    band_periods[band] = []
                band_periods[band].append(period)
            
            # Print periods for each band
            for band, periods in band_periods.items():
                print(f"\n{band}:")
                print("-" * 20)
                
                for period in periods:
                    start_time = f"{int(period['start_time']) // 60:02d}:{int(period['start_time']) % 60:02d}"
                    end_time = f"{int(period['end_time']) // 60:02d}:{int(period['end_time']) % 60:02d}"
                    duration = (int(period['end_time']) - int(period['start_time'])) // 60  # in hours
                    
                    print(f"\nPeriod: {start_time} - {end_time} (Duration: {duration} hours)")
                    print(f"  Channel: {int(period['channel'])}")
                    print(f"  Load Score: {float(period['load_score']):.2f}")
                    print(f"  Stability: {float(period['stability']):.2f}")
                    print("  Metrics:")
                    print(f"    Signal Strength: {float(period['metrics']['avg_signal_strength']):.1f} dBm")
                    print(f"    Networks: {float(period['metrics']['network_count']):.1f}")
                    print(f"    Clients: {float(period['metrics']['client_count']):.1f}")
                    print(f"    Retransmissions: {float(period['metrics']['retransmission_count']):.1f}")
                    print(f"    Lost Packets: {float(period['metrics']['lost_packets']):.1f}")
                    print(f"    Airtime: {float(period['metrics']['airtime']):.1f} ms")
        
        # Print conflicts if any
        if analysis_results['conflicts']:
            print("\nChannel Conflicts:")
            print("-" * 30)
            for conflict in analysis_results['conflicts']:
                print(f"\nConflict between {conflict['point1']} and {conflict['point2']}:")
                print(f"  Band: {conflict['band']}")
                print(f"  Channels: {int(conflict['channel1'])} and {int(conflict['channel2'])}")
        
        # Print final recommendations
        print("\nFinal Recommendations:")
        print("-" * 30)
        for point, recommendations in analysis_results['final_recommendations'].items():
            print(f"\n{point}:")
            for band, band_rec in recommendations.items():
                print(f"  {band}: Channel {int(band_rec['recommended_channel'])}")

    def generate_test_data(self, num_points: int = 3) -> Dict[str, pd.DataFrame]:
        """
        Generate random but realistic test data for analysis.
        
        Args:
            num_points (int): Number of points to generate (between 3 and 6)
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping point names to DataFrames
        """
        self.logger.info(f"Generating test data for {num_points} points")
        
        # Ensure num_points is between 3 and 6
        num_points = max(3, min(6, num_points))
        
        # Define possible channels for each band
        channels_2_4 = list(range(1, 14))  # 2.4 GHz channels
        channels_5 = [36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112,
                     116, 120, 124, 128, 132, 136, 140, 144, 149, 153,
                     157, 161, 165]  # 5 GHz channels
        
        # Generate data for each point
        data_dict = {}
        for i in range(num_points):
            point_name = f"test_point_{i+1}"
            rows = []
            
            # Generate 4-6 random transition times throughout the day
            num_transitions = np.random.randint(4, 7)
            transition_times = sorted(np.random.choice(range(0, 1440, 5), num_transitions, replace=False))
            transition_times = [0] + transition_times + [1440]  # Add start and end of day
            
            # Generate data for each time period
            for j in range(len(transition_times) - 1):
                start_time = transition_times[j]
                end_time = transition_times[j + 1]
                
                # Generate random offset for start and end times (within 5 minutes)
                start_offset = np.random.randint(-5, 6)
                end_offset = np.random.randint(-5, 6)
                start_time = max(0, min(1439, start_time + start_offset))
                end_time = max(0, min(1439, end_time + end_offset))
                
                # Generate base metrics for 2.4 GHz band
                base_signal_2_4 = float(np.random.normal(-65, 10))  # Mean -65 dBm, std 10
                base_networks_2_4 = int(np.random.randint(1, 5))
                base_clients_2_4 = int(np.random.randint(1, 15))
                base_retrans_2_4 = int(np.random.randint(5, 30))
                base_lost_2_4 = int(np.random.randint(1, 10))
                base_airtime_2_4 = int(np.random.randint(100, 500))
                
                # Generate base metrics for 5 GHz band
                base_signal_5 = float(np.random.normal(-55, 8))  # Mean -55 dBm, std 8
                base_networks_5 = int(np.random.randint(1, 8))
                base_clients_5 = int(np.random.randint(1, 40))
                base_retrans_5 = int(np.random.randint(5, 25))
                base_lost_5 = int(np.random.randint(1, 8))
                base_airtime_5 = int(np.random.randint(200, 600))
                
                # Initialize random walk parameters for smooth transitions
                clients_2_4 = base_clients_2_4
                clients_5 = base_clients_5
                signal_2_4 = base_signal_2_4
                signal_5 = base_signal_5
                
                # Generate data points for this period
                for minute in range(start_time, end_time, 5):  # 5-minute intervals
                    # Smooth random walk for total client count
                    total_clients_2_4 = max(1, min(30, clients_2_4 + np.random.normal(0, 0.5)))
                    total_clients_5 = max(1, min(50, clients_5 + np.random.normal(0, 0.8)))
                    
                    # Smooth random walk for signal strength
                    signal_2_4 = max(-100, min(-30, signal_2_4 + np.random.normal(0, 0.3)))
                    # More gradual signal changes for 5 GHz
                    signal_5 = max(-100, min(-30, signal_5 + np.random.normal(0, 0.15)))  # Reduced variation
                    
                    # Generate overlapping client distributions for 2.4 GHz
                    channels_2_4_clients = {}
                    remaining_clients = total_clients_2_4
                    
                    # First, distribute clients to primary channel
                    primary_channel = int(np.random.choice(channels_2_4))
                    primary_clients = int(total_clients_2_4 * np.random.uniform(0.6, 0.8))  # Increased primary channel share
                    channels_2_4_clients[primary_channel] = primary_clients
                    remaining_clients -= primary_clients
                    
                    # Then distribute remaining clients to nearby channels
                    max_attempts = 5  # Reduced attempts
                    attempts = 0
                    while remaining_clients > 0 and attempts < max_attempts:
                        # Get all possible nearby channels
                        possible_channels = []
                        for offset in [-1, 1]:  # Only immediate neighbors
                            nearby_channel = primary_channel + offset
                            if 1 <= nearby_channel <= 13 and nearby_channel not in channels_2_4_clients:
                                possible_channels.append(nearby_channel)
                        
                        if possible_channels:
                            # Choose a random channel from available ones
                            nearby_channel = np.random.choice(possible_channels)
                            channel_clients = min(remaining_clients, int(np.random.uniform(1, 3)))  # Reduced max clients per secondary channel
                            channels_2_4_clients[nearby_channel] = channel_clients
                            remaining_clients -= channel_clients
                        
                        attempts += 1
                    
                    # If there are still remaining clients, add them to the primary channel
                    if remaining_clients > 0:
                        channels_2_4_clients[primary_channel] += remaining_clients
                    
                    # Generate overlapping client distributions for 5 GHz
                    channels_5_clients = {}
                    remaining_clients = total_clients_5
                    
                    # First, distribute clients to primary channel
                    primary_channel = int(np.random.choice(channels_5))
                    primary_clients = int(total_clients_5 * np.random.uniform(0.7, 0.9))  # Increased primary channel share
                    channels_5_clients[primary_channel] = primary_clients
                    remaining_clients -= primary_clients
                    
                    # Then distribute remaining clients to nearby channels
                    max_attempts = 3  # Reduced attempts
                    attempts = 0
                    while remaining_clients > 0 and attempts < max_attempts:
                        # Get all possible nearby channels
                        possible_channels = []
                        for offset in [-20, 20]:  # Only immediate neighbors
                            nearby_channel = primary_channel + offset
                            if nearby_channel in channels_5 and nearby_channel not in channels_5_clients:
                                possible_channels.append(nearby_channel)
                        
                        if possible_channels:
                            # Choose a random channel from available ones
                            nearby_channel = np.random.choice(possible_channels)
                            channel_clients = min(remaining_clients, int(np.random.uniform(1, 4)))  # Reduced max clients per secondary channel
                            channels_5_clients[nearby_channel] = channel_clients
                            remaining_clients -= channel_clients
                        
                        attempts += 1
                    
                    # If there are still remaining clients, add them to the primary channel
                    if remaining_clients > 0:
                        channels_5_clients[primary_channel] += remaining_clients
                    
                    # Generate data points for each channel with clients
                    for channel, channel_clients in channels_2_4_clients.items():
                        # Calculate correlated metrics based on client count
                        networks = max(1, int(base_networks_2_4 + (channel_clients / 5)))
                        retrans = max(5, int(base_retrans_2_4 * (1 + channel_clients / 20)))
                        lost = max(1, int(base_lost_2_4 * (1 + channel_clients / 15)))
                        airtime = max(100, int(base_airtime_2_4 * (1 + channel_clients / 10)))
                        
                        # Add small variation to signal strength for each channel
                        channel_signal = signal_2_4 + np.random.normal(0, 2)
                        
                        rows.append({
                            'channel': channel,
                            'band': '2.4 GHz',
                            'day_of_week': int(np.random.randint(0, 7)),
                            'month': int(np.random.randint(1, 13)),
                            'day': int(np.random.randint(1, 29)),
                            'time': f"{minute//60:02d}:{minute%60:02d}",
                            'minutes_since_midnight': int(minute),
                            'avg_signal_strength': float(channel_signal),
                            'network_count': int(networks),
                            'total_client_count': int(channel_clients),
                            'avg_retransmission_count': int(retrans),
                            'avg_lost_packets': int(lost),
                            'avg_airtime': int(airtime)
                        })
                    
                    for channel, channel_clients in channels_5_clients.items():
                        # Calculate correlated metrics based on client count
                        networks = max(1, int(base_networks_5 + (channel_clients / 8)))
                        retrans = max(5, int(base_retrans_5 * (1 + channel_clients / 30)))
                        lost = max(1, int(base_lost_5 * (1 + channel_clients / 25)))
                        airtime = max(200, int(base_airtime_5 * (1 + channel_clients / 20)))
                        
                        # Add very small variation to signal strength for each channel
                        # More gradual changes for 5 GHz
                        channel_signal = signal_5 + np.random.normal(0, 0.5)  # Reduced variation
                        
                        rows.append({
                            'channel': channel,
                            'band': '5 GHz',
                            'day_of_week': int(np.random.randint(0, 7)),
                            'month': int(np.random.randint(1, 13)),
                            'day': int(np.random.randint(1, 29)),
                            'time': f"{minute//60:02d}:{minute%60:02d}",
                            'minutes_since_midnight': int(minute),
                            'avg_signal_strength': float(channel_signal),
                            'network_count': int(networks),
                            'total_client_count': int(channel_clients),
                            'avg_retransmission_count': int(retrans),
                            'avg_lost_packets': int(lost),
                            'avg_airtime': int(airtime)
                        })
            
            # Create DataFrame and add to dictionary
            df = pd.DataFrame(rows)
            data_dict[point_name] = df
            
            # Save test data to CSV
            output_file = f"{point_name}.csv"
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved test data to {output_file}")
        
        return data_dict


def main():
    parser = argparse.ArgumentParser(
        description='Wi-Fi Channel Analyzer - Analyze and optimize channel selection.'
    )
    
    parser.add_argument('input_files', nargs='*',
                       help='One or more CSV files containing prediction data')
    parser.add_argument('--output', '-o', default='analysis_results.json',
                       help='Output file path (default: analysis_results.json)')
    parser.add_argument('--weights', '-w', type=json.loads,
                       help='Custom weights for channel load calculation (JSON format)')
    parser.add_argument('--test', '-t', action='store_true',
                       help='Generate and analyze test data')
    parser.add_argument('--test-points', type=int, default=3,
                       help='Number of test points to generate (3-6, default: 3)')
    
    args = parser.parse_args()
    
    try:
        # Create analyzer instance
        analyzer = WiFiChannelAnalyzer(weights=args.weights)
        
        if args.test:
            # Generate and analyze test data
            data_dict = analyzer.generate_test_data(args.test_points)
        else:
            # Load and analyze real data
            if not args.input_files:
                parser.error("No input files provided. Use --test for test mode or provide input files.")
            data_dict = analyzer.load_data(args.input_files)
        
        # Analyze data
        analysis_results = analyzer.analyze_multiple_points(data_dict)
        
        # Save results
        analyzer.save_analysis(analysis_results, args.output)
        
        # Print summary
        analyzer.print_summary(analysis_results)
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main()) 