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