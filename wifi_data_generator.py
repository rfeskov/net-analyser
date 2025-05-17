#!/usr/bin/env python3

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

@dataclass
class NetworkConfig:
    """Configuration for a simulated network."""
    ssid: str
    bssid: str
    channel: int
    frequency: str  # '2.4' or '5'
    security_type: str
    baseline_clients: int
    peak_clients: int
    baseline_rssi: float
    rssi_variation: float
    baseline_phy_rate: int
    phy_rate_variation: int

class WiFiDataGenerator:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the data generator with configuration."""
        self.configs = self._load_configs(config_path)
        self.networks = {}  # Store network states
        
    def _load_configs(self, config_path: Optional[str]) -> Dict[str, NetworkConfig]:
        """Load network configurations from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                return {
                    name: NetworkConfig(**config)
                    for name, config in config_data.items()
                }
        
        # Default configurations for different network types
        return {
            'office_main': NetworkConfig(
                ssid="Office-Network",
                bssid="00:11:22:33:44:55",
                channel=36,
                frequency="5",
                security_type="WPA2",
                baseline_clients=20,
                peak_clients=50,
                baseline_rssi=-55,
                rssi_variation=10,
                baseline_phy_rate=866000,
                phy_rate_variation=200000
            ),
            'office_guest': NetworkConfig(
                ssid="Office-Guest",
                bssid="00:11:22:33:44:66",
                channel=1,
                frequency="2.4",
                security_type="WPA2",
                baseline_clients=5,
                peak_clients=15,
                baseline_rssi=-65,
                rssi_variation=15,
                baseline_phy_rate=144000,
                phy_rate_variation=100000
            )
        }
    
    def _generate_temporal_pattern(self, hour: float, is_workday: bool = True) -> float:
        """Generate a temporal pattern for the given hour (0-23)."""
        if not is_workday:
            # Weekend pattern: lower baseline with small variations
            return 0.3 + 0.1 * np.sin(hour * np.pi / 12)
        
        # Workday pattern
        if 8 <= hour < 9:  # Morning arrival
            return 0.3 + 0.7 * (hour - 8)
        elif 9 <= hour < 12:  # Morning work
            return 1.0
        elif 12 <= hour < 13:  # Lunch break
            return 0.7
        elif 13 <= hour < 17:  # Afternoon work
            return 1.0
        elif 17 <= hour < 18:  # Evening departure
            return 1.0 - 0.7 * (hour - 17)
        else:  # Off hours
            return 0.2
    
    def _generate_meeting_pattern(self, hour: float) -> float:
        """Generate meeting patterns during work hours."""
        if 9 <= hour < 17:  # Work hours
            # Simulate meetings at 10, 11, 14, and 15
            meeting_hours = [10, 11, 14, 15]
            for meeting_hour in meeting_hours:
                if abs(hour - meeting_hour) < 0.5:  # 30-minute meetings
                    return 1.5  # 50% increase during meetings
        return 1.0
    
    def _generate_anomaly(self, probability: float = 0.01) -> float:
        """Generate random anomalies with given probability."""
        if np.random.random() < probability:
            return np.random.choice([0.5, 2.0])  # 50% decrease or 100% increase
        return 1.0
    
    def _calculate_metrics(self, network: NetworkConfig, hour: float, 
                         is_workday: bool = True, current_time: datetime = None) -> Dict:
        """Calculate network metrics for the given hour."""
        # Base temporal pattern
        temporal_factor = self._generate_temporal_pattern(hour, is_workday)
        
        # Meeting pattern
        meeting_factor = self._generate_meeting_pattern(hour)
        
        # Random anomaly
        anomaly_factor = self._generate_anomaly()
        
        # Calculate client count
        base_clients = network.baseline_clients
        peak_clients = network.peak_clients
        client_factor = temporal_factor * meeting_factor * anomaly_factor
        client_count = int(base_clients + (peak_clients - base_clients) * client_factor)
        
        # Calculate RSSI (inversely proportional to client count)
        rssi_factor = 1.0 - (client_count / peak_clients) * 0.3  # Up to 30% degradation
        rssi = network.baseline_rssi * rssi_factor + np.random.normal(0, network.rssi_variation)
        rssi = max(-100, min(-30, rssi))  # Clamp to realistic range
        
        # Calculate PHY rate (decreases with client count and lower RSSI)
        phy_factor = (1.0 - (client_count / peak_clients) * 0.4) * (1.0 - abs(rssi) / 100 * 0.3)
        phy_rate = int(network.baseline_phy_rate * phy_factor + 
                      np.random.normal(0, network.phy_rate_variation))
        phy_rate = max(1000, min(6000000, phy_rate))  # Clamp to realistic range
        
        # Calculate retransmissions and lost packets
        error_probability = (client_count / peak_clients) * 0.1 + (abs(rssi) / 100) * 0.2
        retransmissions = int(np.random.poisson(error_probability * 100))
        lost_packets = int(np.random.poisson(error_probability * 50))
        
        # Calculate airtime (increases with client count and errors)
        airtime_factor = (client_count / peak_clients) * 0.8 + (retransmissions / 100) * 0.2
        airtime_ms = int(airtime_factor * 1000 + np.random.normal(0, 100))
        airtime_ms = max(0, min(1000, airtime_ms))  # Clamp to realistic range
        
        # Calculate time-related features
        minutes_since_midnight = current_time.hour * 60 + current_time.minute
        
        return {
            'ssid': network.ssid,
            'bssid': network.bssid,
            'signal_strength': int(rssi),
            'channel': network.channel,
            'frequency': network.frequency,
            'security_type': network.security_type,
            'phy_rate': phy_rate,
            'client_count': client_count,
            'retransmission_count': retransmissions,
            'lost_packets': lost_packets,
            'airtime_ms': airtime_ms,
            'day_of_week': current_time.weekday(),  # 0-6 (Monday-Sunday)
            'month': current_time.month,  # 1-12
            'day': current_time.day,  # 1-31
            'minutes_since_midnight': minutes_since_midnight
        }
    
    def generate_dataset(self, start_date: datetime, end_date: datetime, 
                        interval_minutes: int = 1) -> pd.DataFrame:
        """Generate a dataset for the specified time range."""
        records = []
        current = start_date
        
        while current <= end_date:
            is_workday = current.weekday() < 5  # Monday-Friday
            hour = current.hour + current.minute / 60
            
            for network in self.configs.values():
                metrics = self._calculate_metrics(network, hour, is_workday, current)
                metrics['timestamp'] = current
                records.append(metrics)
            
            current += timedelta(minutes=interval_minutes)
        
        return pd.DataFrame(records)
    
    def save_to_sqlite(self, df: pd.DataFrame, db_path: str):
        """Save the generated dataset to SQLite database."""
        with sqlite3.connect(db_path) as conn:
            df.to_sql('networks', conn, if_exists='append', index=False)
            logger.info(f"Saved {len(df)} records to {db_path}")
    
    def save_to_csv(self, df: pd.DataFrame, csv_path: str):
        """Save the generated dataset to CSV file."""
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df)} records to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Wi-Fi Network Data Generator")
    parser.add_argument("--config", help="Path to network configuration JSON file")
    parser.add_argument("--start-date", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-01-02", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=int, default=1, help="Data collection interval in minutes")
    parser.add_argument("--output-db", help="Output SQLite database path")
    parser.add_argument("--output-csv", help="Output CSV file path")
    args = parser.parse_args()
    
    # Initialize generator
    generator = WiFiDataGenerator(args.config)
    
    # Generate dataset
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    df = generator.generate_dataset(start_date, end_date, args.interval)
    
    # Save output
    if args.output_db:
        generator.save_to_sqlite(df, args.output_db)
    if args.output_csv:
        generator.save_to_csv(df, args.output_csv)
    
    # Display summary
    print("\nDataset Summary:")
    print(f"Time Range: {start_date} to {end_date}")
    print(f"Total Records: {len(df)}")
    print(f"Networks: {len(generator.configs)}")
    print("\nMetrics Summary:")
    print(df.describe())

if __name__ == "__main__":
    main() 