# Wi-Fi Channel Predictor

A command-line tool for predicting Wi-Fi channel parameters based on time and date.

## Quick Start

1. Train the model:
```bash
python predict.py train --train-data your_data.csv
```

2. Make predictions:
```bash
python predict.py predict --date 2024-03-15 --time 14:30
```

## Commands

### Train Command
Train a new model:
```bash
python predict.py train --train-data <path_to_data> [--model <model_path>]
```

Arguments:
- `--train-data`: Path to training data CSV file (required)
- `--model`: Path to save the trained model (default: wifi_model.joblib)

### Predict Command
Make predictions using a trained model:
```bash
python predict.py predict --date <YYYY-MM-DD> --time <HH:MM> [--model <model_path>] [--output <output_path>]
```

Arguments:
- `--date`: Date in YYYY-MM-DD format (required)
- `--time`: Time in HH:MM format (24-hour) (required)
- `--model`: Path to the trained model file (default: wifi_model.joblib)
- `--output`: Output CSV file path (default: auto-generated with timestamp)

## Examples

1. Train a new model:
```bash
python predict.py train --train-data wifi_data.csv --model my_model.joblib
```

2. Make predictions with custom output file:
```bash
python predict.py predict --date 2024-03-15 --time 14:30 --output predictions.csv
```

3. Use a different model:
```bash
python predict.py predict --date 2024-03-15 --time 14:30 --model my_model.joblib
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

## Channel Handling

The model handles both existing and missing channels:
- For channels present in the training data: Uses actual measurements
- For missing channels: Assumes ideal conditions:
  - Signal strength: -100 dBm (very weak)
  - Network count: 0
  - Client count: 0
  - Retransmissions: 0
  - Lost packets: 0
  - Airtime: 0

## Supported Channels

### 2.4 GHz Band
Channels 1-14

### 5 GHz Band
Channels 36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 149, 153, 157, 161, 165 