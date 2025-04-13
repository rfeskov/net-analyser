# Wi-Fi Network Scanner

A Python-based Wi-Fi network scanner that works on both macOS and Linux systems. This tool scans for available Wi-Fi networks and displays detailed information about each network, including SSID, BSSID (MAC address), signal strength, channel, frequency, and encryption type.

## Features

- Automatic OS detection (macOS/Linux)
- Dynamic wireless interface detection
- Detailed network information display
- Filtering by encryption type
- Error handling for permissions and scanning issues
- Clean, tabular output format

## Requirements

### macOS
- Python 3.6 or higher
- `system_profiler` command (built into macOS) - primary scanning method
- Note: The scanner uses system_profiler to get real-time information about available Wi-Fi networks

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

2. Make the script executable:
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

## Error Handling

The program includes error handling for:
- Insufficient permissions (requires sudo/root)
- Failed network scans
- Unexpected command output formats
- Missing system dependencies

## Notes

- On macOS, the program uses the `system_profiler` command to get real-time information about available Wi-Fi networks
- The scanner can detect networks that are currently in range
- On Linux, it uses the `iwlist` command from wireless-tools for active scanning
- The program automatically detects the wireless interface
- Some networks may not broadcast all information
- Signal strength is measured in dBm (typically negative values)

## Troubleshooting

### macOS Issues
- If you're not seeing any networks, make sure your Wi-Fi is turned on
- The scanner uses system_profiler which requires sudo privileges
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