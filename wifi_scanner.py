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
        """Check if the program has necessary permissions."""
        if self.os_type == "Darwin":
            try:
                # Check if we can run system_profiler
                logger.info("Testing system_profiler access...")
                result = subprocess.run(
                    ["system_profiler", "SPAirPortDataType"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info("system_profiler test successful")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Error accessing system_profiler: {e}")
                return False
        else:
            try:
                logger.info(f"Testing iwlist access with command: iwlist {self.interface} scan")
                result = subprocess.run(["iwlist", self.interface, "scan"], capture_output=True, check=True)
                logger.info("iwlist test successful")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Error accessing iwlist: {e}")
                return False

    def _parse_macos_output(self, output: str) -> List[NetworkInfo]:
        """Parse the output of macOS system_profiler command."""
        networks = []
        seen_networks = set()  # To track unique networks
        
        logger.debug(f"Raw system_profiler output:\n{output}")
        
        # Parse the output to extract network information
        lines = output.strip().split('\n')
        current_network = {}
        in_local_networks = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and headers
            if not line or line.startswith("Wi-Fi:") or line.startswith("Software Versions:"):
                continue
                
            # Start of Local Wi-Fi Networks section
            if "Local Wi-Fi Networks:" in line:
                in_local_networks = True
                continue
                
            # Skip if not in Local Wi-Fi Networks section
            if not in_local_networks:
                continue
                
            # New network entry
            if line.endswith(":"):
                if current_network:
                    try:
                        # Create a unique key for the network
                        network_key = f"{current_network.get('ssid', '')}_{current_network.get('channel', 1)}_{current_network.get('frequency', 2.4)}"
                        
                        # Only add if we haven't seen this network before
                        if network_key not in seen_networks:
                            seen_networks.add(network_key)
                            networks.append(NetworkInfo(
                                ssid=current_network.get('ssid', ''),
                                bssid=current_network.get('bssid', 'Unknown'),
                                signal_strength=current_network.get('signal', -100),
                                channel=int(current_network.get('channel', 1)),
                                frequency=float(current_network.get('frequency', 2.4)),
                                encryption=current_network.get('encryption', EncryptionType.UNKNOWN)
                            ))
                    except (ValueError, KeyError) as e:
                        logger.error(f"Error creating NetworkInfo: {e}")
                        pass
                current_network = {'ssid': line[:-1]}  # Remove the colon
                continue
            
            # Extract network information
            if "PHY Mode:" in line:
                # Skip this line as it's not directly useful for our purposes
                continue
            elif "Channel:" in line:
                try:
                    channel_match = re.search(r'Channel (\d+)', line)
                    if channel_match:
                        current_network['channel'] = int(channel_match.group(1))
                        
                        # Extract frequency from the channel
                        if "5GHz" in line:
                            current_network['frequency'] = 5.0
                        elif "2GHz" in line:
                            current_network['frequency'] = 2.4
                        else:
                            # Calculate approximate frequency based on channel
                            channel = int(channel_match.group(1))
                            if channel >= 36:  # 5GHz channels start at 36
                                current_network['frequency'] = 5.0
                            else:
                                current_network['frequency'] = 2.4
                except (ValueError, IndexError) as e:
                    logger.error(f"Error parsing channel line '{line}': {e}")
            elif "Security:" in line:
                security = line.split("Security:")[1].strip()
                if "WPA2/WPA3" in security or "WPA3" in security:
                    current_network['encryption'] = EncryptionType.WPA3
                elif "WPA2" in security:
                    current_network['encryption'] = EncryptionType.WPA2
                elif "WPA" in security:
                    current_network['encryption'] = EncryptionType.WPA
                elif "WEP" in security:
                    current_network['encryption'] = EncryptionType.WEP
                elif "None" in security:
                    current_network['encryption'] = EncryptionType.OPEN
                else:
                    current_network['encryption'] = EncryptionType.UNKNOWN
            elif "Signal:" in line:
                try:
                    signal_match = re.search(r'Signal:\s*(-?\d+)', line)
                    if signal_match:
                        current_network['signal'] = int(signal_match.group(1))
                except (ValueError, IndexError) as e:
                    logger.error(f"Error parsing signal line '{line}': {e}")
        
        # Add the last network if exists
        if current_network:
            try:
                # Create a unique key for the network
                network_key = f"{current_network.get('ssid', '')}_{current_network.get('channel', 1)}_{current_network.get('frequency', 2.4)}"
                
                # Only add if we haven't seen this network before
                if network_key not in seen_networks:
                    seen_networks.add(network_key)
                    networks.append(NetworkInfo(
                        ssid=current_network.get('ssid', ''),
                        bssid=current_network.get('bssid', 'Unknown'),
                        signal_strength=current_network.get('signal', -100),
                        channel=int(current_network.get('channel', 1)),
                        frequency=float(current_network.get('frequency', 2.4)),
                        encryption=current_network.get('encryption', EncryptionType.UNKNOWN)
                    ))
            except (ValueError, KeyError) as e:
                logger.error(f"Error creating NetworkInfo: {e}")
                pass
        
        logger.info(f"Found {len(networks)} networks")
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
            logger.error("Insufficient permissions. Please run with sudo/root privileges.")
            sys.exit(1)

        try:
            if self.os_type == "Darwin":
                logger.info("Scanning networks using system_profiler...")
                result = subprocess.run(
                    ["system_profiler", "SPAirPortDataType"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return self._parse_macos_output(result.stdout)
            else:
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
            logger.error(f"Command output: {e.stdout}")
            logger.error(f"Error output: {e.stderr}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            sys.exit(1)

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