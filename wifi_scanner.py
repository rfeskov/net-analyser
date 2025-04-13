#!/usr/bin/env python3

import subprocess
import re
import platform
import argparse
from typing import List, Dict, Optional
import sys
import os
import logging
import json
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EncryptionType(Enum):
    WPA3 = "WPA3"
    WPA2 = "WPA2"
    WPA = "WPA"
    WEP = "WEP"
    OPEN = "Open"
    UNKNOWN = "Unknown"

@dataclass
class NetworkInfo:
    ssid: str
    bssid: str
    signal_strength: int
    channel: int
    frequency: float
    encryption: EncryptionType

class WiFiScanner:
    def __init__(self):
        self.os_type = platform.system()
        logger.info(f"Detected OS: {self.os_type}")
        self.interface = self._detect_interface()
        logger.info(f"Using interface: {self.interface}")
        self.airport_path = self._find_airport_path() if self.os_type == "Darwin" else None
        if self.os_type == "Darwin":
            logger.info(f"Using airport utility at: {self.airport_path}")

    def _find_airport_path(self) -> str:
        """Find the path to the airport utility on macOS."""
        # Common locations for the airport utility
        possible_paths = [
            "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
            "/usr/local/bin/airport",
            "/usr/bin/airport"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found airport utility at: {path}")
                return path
                
        logger.warning("Airport utility not found in common locations, searching system...")
        # If not found in common locations, try to find it using find command
        try:
            result = subprocess.run(
                ["find", "/System", "-name", "airport", "-type", "f", "-perm", "+111"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.stdout.strip():
                path = result.stdout.strip().split('\n')[0]
                logger.info(f"Found airport utility using find command at: {path}")
                return path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error searching for airport utility: {e}")
            
        logger.warning("Could not find airport utility, falling back to command name")
        return "airport"  # Fallback to just the command name

    def _detect_interface(self) -> str:
        """Detect the default wireless interface based on OS."""
        if self.os_type == "Darwin":  # macOS
            try:
                result = subprocess.run(
                    ["networksetup", "-listallhardwareports"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.debug(f"Hardware ports output:\n{result.stdout}")
                # Look for Wi-Fi interface in the output
                for line in result.stdout.split('\n'):
                    if "Wi-Fi" in line:
                        # Next line should contain the interface name
                        next_line = result.stdout.split('\n')[result.stdout.split('\n').index(line) + 1]
                        interface = next_line.split()[-1]
                        logger.info(f"Found Wi-Fi interface: {interface}")
                        return interface
            except subprocess.CalledProcessError as e:
                logger.error(f"Error detecting interface: {e}")
            logger.warning("Using default macOS interface: en0")
            return "en0"  # Default macOS Wi-Fi interface
        else:  # Linux
            try:
                result = subprocess.run(
                    ["iw", "dev"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # Extract the first wireless interface name
                match = re.search(r"Interface\s+(\w+)", result.stdout)
                if match:
                    interface = match.group(1)
                    logger.info(f"Found Wi-Fi interface: {interface}")
                    return interface
            except subprocess.CalledProcessError as e:
                logger.error(f"Error detecting interface: {e}")
            logger.warning("Using default Linux interface: wlan0")
            return "wlan0"  # Default Linux Wi-Fi interface

    def _check_permissions(self) -> bool:
        """Check if we have necessary permissions to scan networks."""
        if self.os_type == "Darwin":  # macOS
            try:
                # Try networksetup
                result = subprocess.run(
                    ["networksetup", "-listpreferredwirelessnetworks", self.interface],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Error accessing networksetup: {e}")
                return False
        else:  # Linux
            try:
                result = subprocess.run(
                    ["iwlist", "scanning"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Error accessing iwlist: {e}")
                return False

    def _parse_macos_output(self, output: str) -> List[NetworkInfo]:
        """Parse the output of system_profiler command on macOS."""
        networks = []
        current_network = None
        in_wifi_section = False
        in_networks_section = False
        
        logger.debug("Starting to parse system_profiler output")
        logger.debug(f"Raw output:\n{output}")
        
        # Process each line
        for line in output.split('\n'):
            line_content = line.rstrip()
            if not line_content:
                continue
            
            # Check if we're in the Wi-Fi section
            if "Wi-Fi:" in line_content:
                in_wifi_section = True
                continue
            
            if not in_wifi_section:
                continue
            
            # Check if we're in the Local Wi-Fi Networks section
            if "Local Wi-Fi Networks:" in line_content:
                in_networks_section = True
                continue
            
            if not in_networks_section:
                continue
            
            # Get the indentation level
            indent = len(line_content) - len(line_content.lstrip())
            
            # Network SSID (12 spaces indentation)
            if indent == 12 and line_content.endswith(':'):
                if current_network:
                    networks.append(NetworkInfo(
                        ssid=current_network['ssid'],
                        bssid=current_network['bssid'],
                        signal_strength=current_network['signal'],
                        channel=current_network['channel'],
                        frequency=current_network['frequency'],
                        encryption=current_network['encryption']
                    ))
                
                current_network = {
                    'ssid': line_content[12:-1],  # Remove indentation and colon
                    'bssid': 'Unknown',
                    'signal': -100,
                    'channel': 1,
                    'frequency': 2.4,
                    'encryption': EncryptionType.UNKNOWN
                }
                logger.debug(f"Found network: {current_network['ssid']}")
            
            # Network properties (14 spaces indentation)
            elif indent == 14 and current_network:
                if "PHY Mode:" in line_content:
                    phy_mode = line_content.split("PHY Mode:")[1].strip()
                    if any(x in phy_mode for x in ['802.11a', '802.11ac', '802.11ax']):
                        current_network['frequency'] = 5.0
                elif "Channel:" in line_content:
                    try:
                        channel_info = line_content.split("Channel:")[1].strip()
                        channel = int(channel_info.split()[0])
                        current_network['channel'] = channel
                        if '5GHz' in channel_info:
                            current_network['frequency'] = 5.0
                        elif channel > 14:
                            current_network['frequency'] = 5.0
                    except (ValueError, IndexError):
                        pass
                elif "Security:" in line_content:
                    security = line_content.split("Security:")[1].strip()
                    if security == "None":
                        current_network['encryption'] = EncryptionType.OPEN
                    elif "WPA3" in security:
                        current_network['encryption'] = EncryptionType.WPA3
                    elif "WPA2" in security:
                        current_network['encryption'] = EncryptionType.WPA2
                    elif "WPA" in security:
                        current_network['encryption'] = EncryptionType.WPA
                    elif "WEP" in security:
                        current_network['encryption'] = EncryptionType.WEP
            
            # End of Wi-Fi section
            elif indent == 0 and line_content.endswith(':'):
                in_wifi_section = False
                in_networks_section = False
        
        # Add the last network if exists
        if current_network:
            networks.append(NetworkInfo(
                ssid=current_network['ssid'],
                bssid=current_network['bssid'],
                signal_strength=current_network['signal'],
                channel=current_network['channel'],
                frequency=current_network['frequency'],
                encryption=current_network['encryption']
            ))
        
        logger.info(f"Finished parsing, found {len(networks)} networks")
        return networks

    def _parse_linux_output(self, output: str) -> List[NetworkInfo]:
        """Parse the output of Linux iwlist command."""
        networks = []
        current_network = {}
        
        logger.debug(f"Raw iwlist output:\n{output}")
        
        for line in output.split('\n'):
            line = line.strip()
            
            # New network entry
            if "Cell" in line:
                if current_network:
                    try:
                        networks.append(NetworkInfo(
                            ssid=current_network.get('ssid', ''),
                            bssid=current_network.get('bssid', ''),
                            signal_strength=int(current_network.get('signal', -100)),
                            channel=int(current_network.get('channel', 1)),
                            frequency=float(current_network.get('frequency', 2.4)),
                            encryption=current_network.get('encryption', EncryptionType.UNKNOWN)
                        ))
                    except (ValueError, KeyError) as e:
                        logger.error(f"Error creating NetworkInfo: {e}")
                        pass
                current_network = {}
            
            # Extract network information
            try:
                if "ESSID" in line:
                    current_network['ssid'] = re.search(r'ESSID:"([^"]*)"', line).group(1)
                elif "Address" in line:
                    current_network['bssid'] = line.split("Address: ")[1]
                elif "Signal level" in line:
                    current_network['signal'] = re.search(r'Signal level=(-?\d+)', line).group(1)
                elif "Channel" in line:
                    current_network['channel'] = re.search(r'Channel (\d+)', line).group(1)
                elif "Frequency" in line:
                    current_network['frequency'] = float(re.search(r'(\d+\.\d+)', line).group(1))
                elif "IE: IEEE 802.11i" in line:
                    if "WPA3" in line:
                        current_network['encryption'] = EncryptionType.WPA3
                    elif "WPA2" in line:
                        current_network['encryption'] = EncryptionType.WPA2
                    elif "WPA" in line:
                        current_network['encryption'] = EncryptionType.WPA
                elif "WEP" in line:
                    current_network['encryption'] = EncryptionType.WEP
                elif "key:off" in line:
                    current_network['encryption'] = EncryptionType.OPEN
            except Exception as e:
                logger.error(f"Error parsing line '{line}': {e}")
        
        # Add the last network if exists
        if current_network:
            try:
                networks.append(NetworkInfo(
                    ssid=current_network.get('ssid', ''),
                    bssid=current_network.get('bssid', ''),
                    signal_strength=int(current_network.get('signal', -100)),
                    channel=int(current_network.get('channel', 1)),
                    frequency=float(current_network.get('frequency', 2.4)),
                    encryption=current_network.get('encryption', EncryptionType.UNKNOWN)
                ))
            except (ValueError, KeyError) as e:
                logger.error(f"Error creating NetworkInfo: {e}")
                pass
        
        logger.info(f"Found {len(networks)} networks")
        return networks

    def scan_networks(self) -> List[NetworkInfo]:
        """Scan for available Wi-Fi networks."""
        if not self._check_permissions():
            logger.error("Insufficient permissions to scan networks")
            return []

        try:
            if self.os_type == "Darwin":  # macOS
                logger.info("Scanning networks using system_profiler...")
                result = subprocess.run(
                    ["system_profiler", "SPAirPortDataType"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return self._parse_macos_output(result.stdout)
            else:  # Linux
                logger.info("Scanning networks using iwlist...")
                result = subprocess.run(
                    ["iwlist", self.interface, "scan"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return self._parse_linux_output(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error scanning networks: {e}")
            return []

def display_networks(networks: List[NetworkInfo], encryption_filter: Optional[EncryptionType] = None):
    """Display networks in a formatted table."""
    # Filter networks if encryption type is specified
    if encryption_filter:
        networks = [n for n in networks if n.encryption == encryption_filter]
    
    if not networks:
        logger.warning("No networks found matching the criteria.")
        return

    # Print header
    print("\n{:<32} {:<17} {:<8} {:<8} {:<8} {:<8}".format(
        "SSID", "BSSID", "Signal", "Channel", "Freq", "Security"
    ))
    print("-" * 85)

    # Print network information
    for network in networks:
        print("{:<32} {:<17} {:<8} {:<8} {:<8.1f} {:<8}".format(
            network.ssid[:32],
            network.bssid,
            f"{network.signal_strength} dBm",
            network.channel,
            network.frequency,
            network.encryption.value
        ))

def main():
    parser = argparse.ArgumentParser(description="Wi-Fi Network Scanner")
    parser.add_argument(
        "--filter",
        choices=[e.value for e in EncryptionType],
        help="Filter networks by encryption type"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    scanner = WiFiScanner()
    networks = scanner.scan_networks()
    
    encryption_filter = None
    if args.filter:
        encryption_filter = EncryptionType(args.filter)
    
    display_networks(networks, encryption_filter)

if __name__ == "__main__":
    main() 