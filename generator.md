# WiFi Data Generator Documentation

The `wifi_data_generator.py` script is a tool for generating synthetic WiFi network data for testing and analysis purposes. It creates realistic network patterns with configurable parameters and temporal variations.

## Prerequisites

- Python 3.6 or higher
- Required Python packages:
  - pandas
  - numpy
  - sqlite3
  - datetime
  - random
  - json

## Basic Usage

```bash
python3 wifi_data_generator.py [options]
```

## Command Line Options

### Output Format Options
- `--sqlite`: Save output to SQLite database (default)
- `--csv`: Save output to CSV file
- `--output OUTPUT`: Specify output file name (default: 'wifi_data.db' for SQLite or 'wifi_data.csv' for CSV)

### Data Generation Options
- `--networks NETWORKS`: Number of networks to generate (default: 50)
- `--samples SAMPLES`: Number of samples to generate (default: 1000)
- `--config CONFIG`: Path to configuration file (default: 'config.json')

### Time Range Options
- `--start START`: Start date (YYYY-MM-DD, default: 2024-01-01)
- `--end END`: End date (YYYY-MM-DD, default: 2024-12-31)
- `--interval INTERVAL`: Time interval between samples in minutes (default: 15)

## Configuration File

The script uses a JSON configuration file (`config.json` by default) to define network patterns and characteristics. Here's an example configuration:

```json
{
    "networks": {
        "ssid_patterns": ["Home_%d", "Office_%d", "Guest_%d"],
        "security_types": ["WPA2", "WPA3", "Open"],
        "channels": {
            "2.4": [1, 6, 11],
            "5": [36, 40, 44, 48]
        }
    },
    "temporal_patterns": {
        "weekday": {
            "morning": {"mean": -50, "std": 5},
            "afternoon": {"mean": -60, "std": 8},
            "evening": {"mean": -55, "std": 6}
        },
        "weekend": {
            "morning": {"mean": -45, "std": 4},
            "afternoon": {"mean": -50, "std": 7},
            "evening": {"mean": -48, "std": 5}
        }
    }
}
```

### Configuration Parameters

#### Networks Section
- `ssid_patterns`: List of SSID patterns with %d for numbering
- `security_types`: List of security types to use
- `channels`: Available channels for 2.4GHz and 5GHz bands

#### Temporal Patterns Section
- `weekday`: Signal strength patterns for weekdays
  - `morning`: 6:00-12:00
  - `afternoon`: 12:00-18:00
  - `evening`: 18:00-24:00
- `weekend`: Signal strength patterns for weekends
  - Same time periods as weekdays

Each time period includes:
- `mean`: Average signal strength in dBm
- `std`: Standard deviation of signal strength

## Output Data

The generated data includes the following fields:

### Network Information
- `ssid`: Network name
- `bssid`: MAC address
- `channel`: Channel number
- `frequency`: Frequency band (2.4 or 5 GHz)
- `security_type`: Network security type

### Signal Information
- `signal_strength`: Signal strength in dBm
- `snr`: Signal-to-noise ratio
- `noise_level`: Background noise level

### Time Information
- `timestamp`: Date and time of the sample
- `day_of_week`: Day of week (0-6, where 0 is Sunday)
- `month`: Month (1-12)
- `day`: Day of month (1-31)
- `minutes_since_midnight`: Time in minutes since midnight

## Examples

1. Generate data with default settings:
```bash
python3 wifi_data_generator.py
```

2. Generate data for a specific time range:
```bash
python3 wifi_data_generator.py --start 2024-03-01 --end 2024-03-31
```

3. Generate data with custom number of networks and samples:
```bash
python3 wifi_data_generator.py --networks 100 --samples 2000
```

4. Generate data and save to CSV:
```bash
python3 wifi_data_generator.py --csv --output my_data.csv
```

5. Use a custom configuration file:
```bash
python3 wifi_data_generator.py --config my_config.json
```

## Notes

- The script generates realistic temporal patterns with higher signal strengths during peak hours
- Weekend patterns typically show higher signal strengths than weekdays
- The data includes realistic variations in signal strength based on time of day
- All generated MAC addresses are valid and follow the standard format
- The script ensures no duplicate BSSIDs are generated 