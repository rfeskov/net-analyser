# Wi-Fi Network Analyzer

A comprehensive tool for analyzing and optimizing Wi-Fi networks, providing detailed insights into network performance, security, and channel optimization.

## Features

- **Network Scanning**: Detect and analyze all available Wi-Fi networks
- **Band Analysis**: Separate analysis for 2.4 GHz and 5 GHz bands
- **Channel Congestion Analysis**: Identify crowded channels and suggest optimal ones
- **Security Analysis**: Check for weak or outdated security protocols
- **Signal Strength Analysis**: Identify networks with poor signal quality
- **DFS Channel Detection**: Special handling for 5 GHz DFS channels
- **Color-coded Output**: Easy-to-read analysis with color indicators
- **Recommendations**: Actionable suggestions for network optimization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/wifi-analyzer.git
cd wifi-analyzer
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Network Scan
```bash
sudo python3 wifi_analyzer.py
```
This will display a list of all detected Wi-Fi networks with basic information.

### Detailed Analysis with Recommendations
```bash
sudo python3 wifi_analyzer.py --recommend
```
This will perform a comprehensive analysis and provide optimization recommendations.

## Analysis Metrics

### Congestion Score
- **0-5**: Good (minimal interference)
- **5-10**: Moderate (some interference)
- **>10**: High congestion (significant interference)

### Signal Strength
- **>-50 dBm**: Excellent signal
- **-50 to -70 dBm**: Good signal
- **<-70 dBm**: Weak signal

### Security Levels
- **WPA3**: Most secure
- **WPA2**: Good security
- **WPA**: Basic security
- **WEP**: Insecure
- **Open**: No security

### Channel Analysis
- **2.4 GHz**: Focus on non-overlapping channels (1, 6, 11)
- **5 GHz**: Analysis includes DFS channels and channel width considerations

## Output Colors
- **Green**: Good conditions
- **Yellow**: Warning or DFS channels
- **Red**: Critical issues or high congestion

## Requirements
- Python 3.6+
- NetworkManager (`nmcli`)
- Sudo privileges for scanning
- Colorama for colored output
- Pandas and NumPy for analysis

## Troubleshooting

### Common Issues
1. **Permission Denied**: Ensure you're running with sudo
2. **No Networks Found**: Check if your Wi-Fi interface is properly configured
3. **Missing Dependencies**: Install all required packages

### Debug Mode
For detailed logging, run with the debug flag:
```bash
sudo python3 wifi_analyzer.py --debug
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

# Wi-Fi Network Data Generator

A Python script for generating synthetic Wi-Fi network data with realistic temporal patterns and metric dependencies. This tool is designed to create training datasets for machine learning models that analyze Wi-Fi network performance.

## Features

- Generates realistic Wi-Fi network metrics with temporal patterns
- Simulates workday/weekend patterns and meeting schedules
- Models metric interdependencies (RSSI, PHY Rate, Client Count, etc.)
- Supports multiple network configurations
- Outputs to SQLite database or CSV format
- Configurable parameters via JSON configuration file

## Prerequisites

- Python 3.6 or higher
- Required Python packages:
  ```bash
  pip install numpy pandas
  ```

## Installation

1. Clone the repository or download the script files:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Ensure the script is executable:
   ```bash
   chmod +x wifi_data_generator.py
   ```

## Configuration

The script can be configured using a JSON file. A sample configuration file (`network_config.json`) is provided with the following network types:

- Office Main Network (5 GHz)
- Office Guest Network (2.4 GHz)
- Conference Room Network (5 GHz)
- Cafe Network (2.4 GHz)

Each network configuration includes:
- Basic network info (SSID, BSSID, channel, frequency)
- Security settings
- Baseline and peak client counts
- Signal strength parameters
- PHY rate parameters

Example configuration:
```json
{
    "office_main": {
        "ssid": "Office-Network",
        "bssid": "00:11:22:33:44:55",
        "channel": 36,
        "frequency": "5",
        "security_type": "WPA2",
        "baseline_clients": 20,
        "peak_clients": 50,
        "baseline_rssi": -55,
        "rssi_variation": 10,
        "baseline_phy_rate": 866000,
        "phy_rate_variation": 200000
    }
}
```

## Usage

### Basic Usage

Generate data for a single day with default settings:
```bash
python wifi_data_generator.py --output-db wifi_data.db
```

### Advanced Usage

Generate data with custom configuration and parameters:
```bash
python wifi_data_generator.py \
    --config network_config.json \
    --start-date 2024-01-01 \
    --end-date 2024-01-07 \
    --interval 5 \
    --output-db wifi_data.db \
    --output-csv wifi_data.csv
```

### Command Line Arguments

- `--config`: Path to network configuration JSON file
- `--start-date`: Start date in YYYY-MM-DD format (default: 2024-01-01)
- `--end-date`: End date in YYYY-MM-DD format (default: 2024-01-02)
- `--interval`: Data collection interval in minutes (default: 1)
- `--output-db`: Output SQLite database path
- `--output-csv`: Output CSV file path

### Output Format

The generated data includes the following metrics:
- Timestamp
- SSID
- BSSID
- Signal Strength (RSSI)
- Channel
- Frequency
- Security Type
- PHY Rate
- Client Count
- Retransmission Count
- Lost Packets
- Airtime (ms)

## Temporal Patterns

The script simulates various temporal patterns:

1. **Workday Pattern**:
   - Morning arrival (8:00-9:00)
   - Morning work (9:00-12:00)
   - Lunch break (12:00-13:00)
   - Afternoon work (13:00-17:00)
   - Evening departure (17:00-18:00)
   - Off hours (18:00-8:00)

2. **Weekend Pattern**:
   - Lower baseline activity
   - Small variations throughout the day

3. **Meeting Patterns**:
   - Scheduled meetings at 10:00, 11:00, 14:00, and 15:00
   - 30-minute duration
   - 50% increase in activity during meetings

4. **Anomalies**:
   - Random anomalies with 1% probability
   - Can cause 50% decrease or 100% increase in metrics

## Metric Dependencies

The script models realistic dependencies between metrics:

1. **RSSI (Signal Strength)**:
   - Decreases with higher client count
   - Includes random variations
   - Range: -100 to -30 dBm

2. **PHY Rate**:
   - Decreases with higher client count
   - Decreases with lower RSSI
   - Range: 1-6000 Mbps

3. **Client Count**:
   - Follows temporal patterns
   - Increases during meetings
   - Affects other metrics

4. **Error Metrics**:
   - Retransmissions and lost packets increase with:
     - Higher client count
     - Lower signal strength
   - Airtime increases with:
     - Higher client count
     - More retransmissions

## Examples

### Generate One Week of Data
```bash
python wifi_data_generator.py \
    --config network_config.json \
    --start-date 2024-01-01 \
    --end-date 2024-01-07 \
    --output-db week_data.db
```

### Generate Data with 5-Minute Intervals
```bash
python wifi_data_generator.py \
    --interval 5 \
    --output-db five_min_data.db
```

### Generate Data for Multiple Networks
```bash
python wifi_data_generator.py \
    --config network_config.json \
    --output-csv all_networks.csv
```

## Troubleshooting

1. **Database Errors**:
   - Ensure the output directory is writable
   - Check if the database file is not locked by another process

2. **Configuration Errors**:
   - Verify JSON syntax in configuration file
   - Ensure all required parameters are present
   - Check parameter value ranges

3. **Memory Issues**:
   - For long time ranges, consider using larger intervals
   - Monitor system memory usage
   - Use CSV output for very large datasets

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 