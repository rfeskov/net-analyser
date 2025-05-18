# Wi-Fi Channel Analyzer

# Basic usage
python data_analyzer.py predictions.csv

# Multiple points
python data_analyzer.py point1.csv point2.csv

# Custom output and weights
python data_analyzer.py predictions.csv -o results.json -w '{"signal_strength": -1.5, "network_count": 2.5}'

A Python tool for analyzing Wi-Fi channel predictions and providing optimization recommendations.

## Features

- Analyzes one or multiple prediction data points
- Calculates channel load scores based on multiple metrics
- Detects and resolves channel conflicts
- Provides channel recommendations for both 2.4 GHz and 5 GHz bands
- Supports custom weighting of metrics
- Generates detailed analysis reports

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Analyze a single prediction file:
```bash
python data_analyzer.py predictions.csv
```

Analyze multiple prediction files:
```bash
python data_analyzer.py point1_predictions.csv point2_predictions.csv
```

### Command Line Options

- `input_files`: One or more CSV files containing prediction data (required)
- `--output`, `-o`: Output file path (default: analysis_results.json)
- `--weights`, `-w`: Custom weights for channel load calculation (JSON format)

### Custom Weights

You can customize the weights used for channel load calculation using the `--weights` option:

```bash
python data_analyzer.py predictions.csv --weights '{"signal_strength": -1.5, "network_count": 2.5, "client_count": 3.5, "retransmission_count": 2.0, "lost_packets": 1.5, "airtime": 2.5}'
```

Default weights:
- Signal Strength: -1.0 (negative because lower is better)
- Network Count: 2.0
- Client Count: 3.0
- Retransmission Count: 2.0
- Lost Packets: 1.0
- Airtime: 2.0

## Input Format

The input CSV files should contain the following columns:

- `channel`: Wi-Fi channel number
- `band`: Frequency band (2.4 GHz or 5 GHz)
- `day_of_week`: Day of the week (1-7)
- `month`: Month (1-12)
- `day`: Day of the month (1-31)
- `time`: Time in HH:MM format
- `minutes_since_midnight`: Minutes since midnight (0-1439)
- `avg_signal_strength`: Average signal strength in dBm
- `network_count`: Number of networks on the channel
- `total_client_count`: Total number of clients
- `avg_retransmission_count`: Average number of retransmissions
- `avg_lost_packets`: Average number of lost packets
- `avg_airtime`: Average airtime in milliseconds

## Output Format

The analyzer generates a JSON file containing:

1. Point Analyses:
   - Recommended channels for each band
   - Load scores and stability metrics
   - Detailed channel metrics

2. Channel Conflicts:
   - List of detected conflicts between points
   - Affected channels and bands

3. Final Recommendations:
   - Resolved channel assignments
   - Conflict-free channel selections

## Channel Selection Criteria

The analyzer uses the following criteria for channel selection:

1. Load Score:
   - Calculated using weighted metrics
   - Lower score indicates better channel

2. Stability:
   - Measures variance in load score
   - More stable channels are preferred

3. Channel Spacing:
   - 2.4 GHz: Minimum 5 MHz spacing
   - 5 GHz: Minimum 20 MHz spacing

4. Load Classification:
   - Low: < 30% load
   - Medium: 30-60% load
   - High: > 60% load

## Example Output

```json
{
  "point_analyses": {
    "point1": {
      "2.4 GHz": {
        "recommended_channel": 1,
        "load_score": 0.25,
        "stability": 0.85,
        "load_classification": "low",
        "metrics": {
          "avg_signal_strength": -65.5,
          "network_count": 2.0,
          "client_count": 5.0,
          "retransmission_count": 1.2,
          "lost_packets": 0.5,
          "airtime": 150.0
        }
      }
    }
  },
  "conflicts": [
    {
      "point1": "point1",
      "point2": "point2",
      "band": "2.4 GHz",
      "channel1": 1,
      "channel2": 3
    }
  ],
  "final_recommendations": {
    "point1": {
      "2.4 GHz": {
        "recommended_channel": 6
      }
    }
  }
}
```

## Error Handling

The analyzer includes comprehensive error handling for:
- Missing or invalid input files
- Incorrect data formats
- Missing required columns
- Invalid channel numbers
- JSON parsing errors

## Contributing

Feel free to submit issues and enhancement requests! 