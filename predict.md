# Wi-Fi Channel Predictor

A command-line tool for predicting Wi-Fi channel parameters based on time and date.

## Quick Start

Basic usage with default settings:
```bash
python predict.py --date 2024-03-15 --time 14:30
```

## Command Line Arguments

### Required Arguments
- `--date`: Date in YYYY-MM-DD format
- `--time`: Time in HH:MM format (24-hour)

### Optional Arguments
- `--model`: Path to trained model file (default: wifi_model.joblib)
- `--output`: Output CSV file path (default: auto-generated with timestamp)
- `--train-data`: Path to training data CSV file (default: aggregated_wifi_data.csv)
- `--retrain`: Retrain the model before making predictions

## Examples

1. Make predictions with custom output file:
```bash
python predict.py --date 2024-03-15 --time 14:30 --output predictions.csv
```

2. Use a different model:
```bash
python predict.py --date 2024-03-15 --time 14:30 --model my_model.joblib
```

3. Retrain the model:
```bash
python predict.py --date 2024-03-15 --time 14:30 --retrain
```

4. Use custom training data:
```bash
python predict.py --date 2024-03-15 --time 14:30 --retrain --train-data my_data.csv
```

## Output Format

The script generates a CSV file with the following columns:
- `channel`: Wi-Fi channel number
- `band`: '2.4 GHz' or '5 GHz'
- `day_of_week`: Day of the week (1-7)
- `month`: Month (1-12)
- `day`: Day of the month (1-31)
- `time`: Time in HH:MM format
- `minutes_since_midnight`: Minutes since midnight (0-1439)
- `avg_signal_strength`: Predicted signal strength (dBm)
- `network_count`: Predicted number of networks
- `total_client_count`: Predicted total clients
- `avg_retransmission_count`: Predicted retransmissions
- `avg_lost_packets`: Predicted lost packets
- `avg_airtime`: Predicted airtime (ms)

## Supported Channels

### 2.4 GHz Band
Channels 1-14

### 5 GHz Band
Channels 36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 149, 153, 157, 161, 165 