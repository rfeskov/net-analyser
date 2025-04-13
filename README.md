# Wi-Fi Network Scanner

A Python program for scanning available Wi-Fi networks on macOS and Linux systems. This program provides detailed information about nearby networks including SSID, BSSID, signal strength, channel, frequency, and encryption type.

## Features

- Detects and lists all available Wi-Fi networks
- Shows detailed network information:
  - SSID (Network name)
  - BSSID (MAC address)
  - Signal strength (in dBm)
  - Channel number
  - Frequency (2.4 GHz or 5 GHz)
  - Encryption type (WEP, WPA, WPA2, WPA3)
- Supports filtering networks by encryption type
- Works on both macOS and Linux systems
- Uses the Wireless Diagnostics framework on macOS for enhanced capabilities

## Requirements

- Python 3.6 or higher
- macOS 10.15 (Catalina) or higher for Wireless Diagnostics support
- Linux systems with `iwlist` command available
- Root/sudo privileges for scanning networks

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/wifi-scanner.git
cd wifi-scanner
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the scanner with sudo/root privileges:

```bash
sudo python3 wifi_scanner.py
```

### Output Format

The program displays networks in a table format:
```
SSID                           Signal     Channel    Frequency  Security   
--------------------------------------------------------------------------------
MyNetwork                      -65 dBm    36         5.0        WPA2       
Neighbor's WiFi               -72 dBm    1          2.4        WPA3       
```

## Wireless Diagnostics Framework

On macOS, this program uses the Wireless Diagnostics framework (`wdutil`) to access detailed Wi-Fi information. This provides several advantages:

1. More accurate signal strength measurements
2. Detailed BSSID information
3. Precise channel and frequency data
4. Better encryption type detection
5. Access to hidden networks

### Troubleshooting

If you encounter permission issues:

1. Ensure you're running the program with sudo:
```bash
sudo python3 wifi_scanner.py
```

2. Check if Wireless Diagnostics is available:
```bash
/usr/sbin/wdutil info
```

3. If you get a "command not found" error, ensure you're running macOS 10.15 or higher.

4. For Linux systems, ensure the `iwlist` command is available:
```bash
which iwlist
```

## Error Handling

The program includes comprehensive error handling for common issues:
- Insufficient permissions
- Missing system utilities
- Network interface detection failures
- Parsing errors

## License

This project is licensed under the MIT License - see the LICENSE file for details. 