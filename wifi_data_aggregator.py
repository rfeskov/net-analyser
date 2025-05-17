#!/usr/bin/env python3

import pandas as pd
import argparse
from pathlib import Path


def aggregate_wifi_data(input_file: str, output_file: str) -> None:
    """
    Aggregate Wi-Fi network monitoring data by channel and time.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
    """
    # Read the input CSV file
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Group by channel and time-related columns
    print("Aggregating data...")
    grouped = df.groupby([
        'channel',
        'day_of_week',
        'month',
        'day',
        'minutes_since_midnight'
    ]).agg({
        'signal_strength': 'mean',
        'bssid': 'nunique',
        'client_count': 'sum',
        'retransmission_count': 'mean',
        'lost_packets': 'mean',
        'airtime_ms': 'mean'
    }).reset_index()
    
    # Rename columns to match the required output format
    grouped = grouped.rename(columns={
        'signal_strength': 'avg_signal_strength',
        'bssid': 'network_count',
        'client_count': 'total_client_count',
        'retransmission_count': 'avg_retransmission_count',
        'lost_packets': 'avg_lost_packets',
        'airtime_ms': 'avg_airtime'
    })
    
    # Ensure columns are in the correct order
    columns = [
        'channel',
        'avg_signal_strength',
        'network_count',
        'total_client_count',
        'avg_retransmission_count',
        'avg_lost_packets',
        'avg_airtime',
        'day_of_week',
        'month',
        'day',
        'minutes_since_midnight'
    ]
    grouped = grouped[columns]
    
    # Save the aggregated data to a new CSV file
    print(f"Saving aggregated data to {output_file}...")
    grouped.to_csv(output_file, index=False)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate Wi-Fi network monitoring data by channel and time.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input CSV file containing Wi-Fi network monitoring data'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Path to save the output CSV file with aggregated data'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_file).is_file():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    aggregate_wifi_data(args.input_file, args.output_file)


if __name__ == '__main__':
    main() 