# Wi-Fi Network Scanner

A Python-based Wi-Fi network scanner that works on both macOS and Linux systems. This tool scans for available Wi-Fi networks and displays detailed information about each network, including SSID, BSSID (MAC address), signal strength, channel, frequency, and encryption type.

## Features

- Automatic OS detection (macOS/Linux)
- Native CoreWLAN integration for macOS
- Dynamic wireless interface detection
- Detailed network information display
- Filtering by encryption type
- Error handling for permissions and scanning issues
- Clean, tabular output format

## Requirements

### macOS
- Python 3.6 or higher
- pyobjc-framework-CoreWLAN (for native Wi-Fi scanning)
- scapy (for advanced network analysis)

### Linux
- Python 3.6 or higher
- `wireless-tools` package (provides `iwlist` command)
- Root/sudo privileges for scanning

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd wifi-scanner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make the script executable:
```bash
chmod +x wifi_scanner.py
```

### Linux Dependencies
On Linux systems, install the required wireless tools:
```bash
# Debian/Ubuntu
sudo apt-get install wireless-tools

# Fedora/RHEL
sudo dnf install wireless-tools
```

## Usage

Run the scanner with sudo/root privileges:

```bash
# Basic scan
sudo python3 wifi_scanner.py

# Filter networks by encryption type
sudo python3 wifi_scanner.py --filter WPA2
sudo python3 wifi_scanner.py --filter Open

# Enable debug logging
sudo python3 wifi_scanner.py --debug
```

Available encryption type filters:
- WPA3
- WPA2
- WPA
- WEP
- Open

## Output Format

The program displays results in a table format with the following columns:
- SSID: Network name
- BSSID: MAC address of the access point
- Signal: Signal strength in dBm
- Channel: Wi-Fi channel number
- Freq: Frequency in GHz
- Security: Encryption type

## Implementation Details

### macOS
The scanner uses Apple's CoreWLAN framework through PyObjC bindings to:
- Access Wi-Fi interface information
- Perform native network scanning
- Get detailed network information including:
  - SSID and BSSID
  - Signal strength (RSSI)
  - Channel and frequency
  - Security type (WPA/WPA2/WPA3)

### Linux
On Linux systems, the scanner uses:
- `iwlist` for network scanning
- `iw` for interface detection
- Direct hardware access for advanced features

## Error Handling

The program includes error handling for:
- Insufficient permissions (requires sudo/root)
- Failed network scans
- Missing system dependencies
- Interface detection issues

## Notes

- On macOS, the program uses the native CoreWLAN framework for optimal performance
- The scanner can detect networks that are currently in range
- Signal strength is measured in dBm (typically negative values)
- Some networks may not broadcast all information

## Troubleshooting

### macOS Issues
- Make sure your Wi-Fi is turned on
- Verify that the CoreWLAN framework is accessible
- For more detailed Wi-Fi diagnostics, use the Wireless Diagnostics app:
  ```bash
  open -a "Wireless Diagnostics"
  ```

### Linux Issues
- Ensure you have the wireless-tools package installed
- Make sure you're running the script with sudo/root privileges
- Check that your wireless interface is enabled and not in monitor mode

## License

This project is licensed under the MIT License - see the LICENSE file for details. 