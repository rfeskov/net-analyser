#!/usr/bin/env python3

import subprocess
import re
import sys
import argparse
from typing import List, Dict, Optional
import os
import logging
from dataclasses import dataclass
from enum import Enum
from scapy.all import *
from scapy.layers.dot11 import Dot11, Dot11Beacon, Dot11Elt, Dot11WEP

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add a stream handler with more verbose output
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class EncryptionType(Enum):
    NONE = "Open"
    WEP = "WEP"
    WPA = "WPA"
    WPA2 = "WPA2"
    WPA3 = "WPA3"
    WPA2_WPA3 = "WPA2/WPA3"
    UNKNOWN = "Unknown"

@dataclass
class NetworkInfo:
    ssid: str
    bssid: Optional[str]
    signal_strength: Optional[int]
    channel: Optional[int]
    frequency: Optional[str]
    security_type: Optional[str]

    def __str__(self) -> str:
        return f"{self.ssid:<32} {self.bssid or 'N/A':<17} {self.signal_strength or 'N/A':<8} {self.channel or 'N/A':<8} {self.frequency or 'N/A':<6} {self.security_type or 'N/A'}"

class WiFiScanner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.interface = None
        self.debug = False
        self.setup_logging()
        logger.debug("WiFiScanner initialized")
        self._detect_interface()
        self._setup_linux()

    def setup_logging(self):
        """Set up logging configuration."""
        # Remove any existing handlers to prevent duplication
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True
        )
        self.logger = logging.getLogger(__name__)

    def _detect_interface(self) -> str:
        """Detect the default wireless interface on Linux."""
        logger.debug("Starting interface detection")
        try:
            # Try iw command first
            result = subprocess.run(
                ["iw", "dev"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"iw dev output: {result.stdout}")
            # Extract the first wireless interface name
            match = re.search(r"Interface\s+(\w+)", result.stdout)
            if match:
                interface = match.group(1)
                logger.info(f"Found Wi-Fi interface with iw: {interface}")
                self.interface = interface
                return interface
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Error detecting interface with iw: {e}")
        
        try:
            # Try nmcli as fallback
            result = subprocess.run(
                ["nmcli", "device", "show"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"nmcli output: {result.stdout}")
            # Look for wireless devices
            for line in result.stdout.split('\n'):
                if "wifi" in line.lower() and "device" in line.lower():
                    interface = line.split(':')[1].strip()
                    logger.info(f"Found Wi-Fi interface with nmcli: {interface}")
                    self.interface = interface
                    return interface
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Error detecting interface with nmcli: {e}")
        
        try:
            # Try ip command as last resort
            result = subprocess.run(
                ["ip", "link", "show"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"ip link show output: {result.stdout}")
            # Look for wireless interfaces (usually start with wl)
            for line in result.stdout.split('\n'):
                if "wl" in line and "@" not in line:
                    interface = line.split(':')[1].strip()
                    logger.info(f"Found Wi-Fi interface with ip: {interface}")
                    self.interface = interface
                    return interface
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Error detecting interface with ip: {e}")
        
        logger.warning("Using default Linux interface: wlan0")
        self.interface = "wlan0"
        return "wlan0"  # Default Linux Wi-Fi interface

    def _check_permissions(self) -> bool:
        """Check if we have necessary permissions to scan networks."""
        try:
            # Check if we can access the wireless interface
            result = subprocess.run(['iwconfig', self.interface], capture_output=True, text=True)
            if result.returncode != 0:
                return False
            
            # Check if we can set the channel
            result = subprocess.run(['iw', 'dev', self.interface, 'set', 'channel', '1'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking permissions: {e}")
            return False

    def _parse_nmcli_output(self, output: str) -> List[NetworkInfo]:
        """Parse the output of nmcli command."""
        networks = []
        
        logger.debug(f"Raw nmcli output:\n{output}")
        
        for line in output.split('\n'):
            if not line.strip():
                continue
                
            try:
                # Split the line into parts, handling escaped colons in MAC addresses
                parts = []
                current_part = []
                i = 0
                while i < len(line):
                    if line[i] == ':' and (i == 0 or line[i-1] != '\\'):
                        parts.append(''.join(current_part))
                        current_part = []
                    else:
                        current_part.append(line[i])
                    i += 1
                parts.append(''.join(current_part))
                
                if len(parts) >= 6:
                    ssid = parts[0]
                    bssid = parts[1].replace('\\:', ':')  # Unescape colons in MAC address
                    channel = int(parts[2])
                    rate = parts[3]
                    signal = int(parts[4])
                    security = parts[5]
                    
                    # Determine frequency based on channel
                    frequency = '2.4'
                    if channel > 14:
                        # Convert channel to frequency in MHz
                        if 36 <= channel <= 64:  # UNII-1 and UNII-2A
                            frequency = str(5180 + (channel - 36) * 20)
                        elif 100 <= channel <= 140:  # UNII-2C
                            frequency = str(5500 + (channel - 100) * 20)
                        elif 149 <= channel <= 165:  # UNII-3
                            frequency = str(5745 + (channel - 149) * 20)
                        else:
                            frequency = '5'  # Fallback to just '5' if channel is unknown
                    
                    # Create network info
                    network = NetworkInfo(
                        ssid=ssid,
                        bssid=bssid,
                        signal_strength=signal,
                        channel=channel,
                        frequency=frequency,
                        security_type=security or EncryptionType.NONE.value
                    )
                    networks.append(network)
                    logger.debug(f"Found network: {ssid} ({bssid})")
            except Exception as e:
                logger.error(f"Error parsing line '{line}': {e}")
                continue
        
        return networks

    def _scan_with_scapy(self) -> List[NetworkInfo]:
        """Scan for Wi-Fi networks using scapy."""
        networks = []
        seen_networks = set()

        def packet_handler(pkt):
            if pkt.haslayer(Dot11Beacon):
                # Extract BSSID
                bssid = pkt[Dot11].addr2
                
                # Extract SSID
                ssid = None
                for element in pkt[Dot11Elt:]:
                    if element.ID == 0:  # SSID element
                        try:
                            ssid = element.info.decode()
                        except:
                            ssid = str(element.info)
                        break
                
                if not ssid or ssid in seen_networks:
                    return
                
                seen_networks.add(ssid)
                
                # Extract channel
                channel = None
                logger.debug(f"Attempting to extract channel for network {ssid}")
                for element in pkt[Dot11Elt:]:
                    logger.debug(f"Processing element ID {element.ID} for network {ssid}")
                    if element.ID == 3:  # DS Parameter Set element
                        try:
                            channel = ord(element.info)
                            logger.debug(f"Found channel {channel} from DS Parameter Set for network {ssid}")
                        except Exception as e:
                            logger.debug(f"Failed to extract channel from DS Parameter Set for network {ssid}: {e}")
                    elif element.ID == 1:  # Channel element
                        try:
                            channel = ord(element.info)
                            logger.debug(f"Found channel {channel} from Channel element for network {ssid}")
                        except Exception as e:
                            logger.debug(f"Failed to extract channel from Channel element for network {ssid}: {e}")
                    elif element.ID == 36:  # Supported Channels element
                        try:
                            # First byte is first channel number
                            channel = ord(element.info[0])
                            logger.debug(f"Found channel {channel} from Supported Channels for network {ssid}")
                        except Exception as e:
                            logger.debug(f"Failed to extract channel from Supported Channels for network {ssid}: {e}")
                
                if channel is None:
                    logger.debug(f"Could not find channel information for network {ssid}")
                    # Try to get channel from RadioTap header
                    if hasattr(pkt, 'Channel'):
                        try:
                            channel = pkt.Channel
                            logger.debug(f"Found channel {channel} from RadioTap header for network {ssid}")
                        except Exception as e:
                            logger.debug(f"Failed to extract channel from RadioTap header for network {ssid}: {e}")
                
                # Extract signal strength from RadioTap header
                signal_strength = None
                if hasattr(pkt, 'dBm_AntSignal'):
                    signal_strength = pkt.dBm_AntSignal
                    logger.debug(f"Found signal strength {signal_strength} dBm for network {ssid} using dBm_AntSignal")
                elif hasattr(pkt, 'notdecoded'):
                    try:
                        # Try different offsets for RSSI
                        for offset in range(-4, 0):
                            try:
                                rssi = -(256-ord(pkt.notdecoded[offset:offset+1]))
                                if -100 <= rssi <= 0:  # Valid RSSI range
                                    signal_strength = rssi
                                    logger.debug(f"Found signal strength {signal_strength} dBm for network {ssid} using notdecoded")
                                    break
                            except (IndexError, TypeError):
                                continue
                    except:
                        logger.debug(f"Failed to extract signal strength for network {ssid}")
                
                # Determine frequency based on channel
                frequency = '2.4'
                if channel and channel > 14:
                    frequency = '5'
                    logger.debug(f"Network {ssid} is on 5 GHz band (channel {channel})")
                
                # Determine encryption type
                security_type = EncryptionType.UNKNOWN.value
                if pkt.haslayer(Dot11WEP):
                    security_type = EncryptionType.WEP.value
                    logger.debug(f"Network {ssid} uses WEP encryption")
                else:
                    # Check for WPA/WPA2/WPA3
                    rsn = None
                    for element in pkt[Dot11Elt:]:
                        if element.ID == 48:  # RSN element
                            rsn = element
                            break
                    
                    if rsn:
                        # Parse RSN capabilities
                        if len(rsn.info) > 2:
                            version = rsn.info[0]
                            if version == 3:
                                security_type = EncryptionType.WPA3.value
                            elif version == 2:
                                security_type = EncryptionType.WPA2.value
                            elif version == 1:
                                security_type = EncryptionType.WPA.value
                            logger.debug(f"Network {ssid} uses {security_type} encryption")
                
                # Create network info
                network = NetworkInfo(
                    ssid=ssid,
                    bssid=bssid,
                    signal_strength=signal_strength,
                    channel=channel,
                    frequency=frequency,
                    security_type=security_type
                )
                networks.append(network)
                logger.debug(f"Found network: {ssid} ({bssid}) - Signal: {signal_strength} dBm, Channel: {channel}, Security: {security_type}")

        try:
            # Start sniffing for 5 seconds
            logger.info("Starting Wi-Fi scan with scapy...")
            sniff(iface=self.interface, prn=packet_handler, timeout=5, store=0)
            logger.info(f"Found {len(networks)} networks using scapy")
            return networks
        except Exception as e:
            logger.error(f"Error scanning with scapy: {e}")
            return []

    def scan_networks(self) -> List[NetworkInfo]:
        """Scan for available Wi-Fi networks."""
        logger.debug("Starting network scan")
        
        try:
            # Try nmcli first
            cmd = ["nmcli", "-t", "-f", "SSID,BSSID,CHAN,RATE,SIGNAL,SECURITY", "device", "wifi", "list"]
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.debug("Parsing nmcli output")
                networks = self._parse_nmcli_output(result.stdout)
                if networks:
                    logger.info(f"Found {len(networks)} networks using nmcli")
                    return networks
            
            # If nmcli fails or finds no networks, try with scapy
            logger.warning("No networks found with nmcli, trying scapy...")
            return self._scan_with_scapy()
                
        except Exception as e:
            logger.error(f"Error scanning networks: {str(e)}")
            # Try with scapy as fallback
            return self._scan_with_scapy()

    def _setup_linux(self):
        """Set up the scanner for Linux systems."""
        logger.debug(f"Setting up Linux interface: {self.interface}")
        if not self.interface:
            logger.error("No interface detected")
            raise Exception("No wireless interface detected")
            
        try:
            # First, try to stop NetworkManager
            logger.debug("Attempting to stop NetworkManager")
            subprocess.run(['systemctl', 'stop', 'NetworkManager'], capture_output=True)
            logger.info("Stopped NetworkManager")
            
            # Then stop wpa_supplicant
            logger.debug("Attempting to stop wpa_supplicant")
            subprocess.run(['systemctl', 'stop', 'wpa_supplicant'], capture_output=True)
            logger.info("Stopped wpa_supplicant")
            
            # Kill any processes that might interfere
            logger.debug("Running airmon-ng check kill")
            subprocess.run(['airmon-ng', 'check', 'kill'], capture_output=True)
            
            # Set interface to monitor mode
            logger.debug(f"Attempting to set {self.interface} to monitor mode")
            subprocess.run(['airmon-ng', 'start', self.interface], capture_output=True)
            logger.info(f"Set {self.interface} to monitor mode")
            
            # Verify interface is in monitor mode
            logger.debug("Verifying monitor mode")
            result = subprocess.run(['iwconfig', self.interface], capture_output=True, text=True)
            logger.debug(f"iwconfig output: {result.stdout}")
            if 'Mode:Monitor' not in result.stdout:
                raise Exception(f"Failed to set {self.interface} to monitor mode")
            
            # Set interface up
            logger.debug("Setting interface up")
            subprocess.run(['ip', 'link', 'set', self.interface, 'up'], capture_output=True)
            logger.info(f"Set {self.interface} up")
            
            # Verify interface is up
            logger.debug("Verifying interface state")
            result = subprocess.run(['ip', 'link', 'show', self.interface], capture_output=True, text=True)
            logger.debug(f"ip link show output: {result.stdout}")
            if 'state UP' not in result.stdout:
                raise Exception(f"Failed to bring {self.interface} up")
            
            logger.info(f"Successfully configured {self.interface}")
            
        except Exception as e:
            logger.error(f"Error setting up interface: {e}")
            # Try to restore NetworkManager
            logger.debug("Attempting to restore NetworkManager")
            subprocess.run(['systemctl', 'start', 'NetworkManager'], capture_output=True)
            raise

    def __del__(self):
        """Cleanup when the scanner is destroyed."""
        try:
            # Restore NetworkManager
            subprocess.run(['systemctl', 'start', 'NetworkManager'], capture_output=True)
            logger.info("Restored NetworkManager")
            
            # Restore wpa_supplicant
            subprocess.run(['systemctl', 'start', 'wpa_supplicant'], capture_output=True)
            logger.info("Restored wpa_supplicant")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def display_networks(networks: List[NetworkInfo], encryption_filter: Optional[str] = None) -> None:
    """Display the list of networks in a formatted table."""
    if encryption_filter:
        networks = [n for n in networks if n.security_type.lower() == encryption_filter.lower()]

    # Sort by signal strength (None values last) then by SSID
    networks.sort(key=lambda x: (x.signal_strength if x.signal_strength is not None else -1000, x.ssid))

    # Print header
    print(f"{'SSID':<32} {'BSSID':<17} {'Signal':<8} {'Channel':<8} {'Freq':<6} {'Security'}")
    print("-" * 85)

    # Print each network
    for network in networks:
        # Handle special characters in SSID by using unicode normalization
        import unicodedata
        ssid = unicodedata.normalize('NFKC', network.ssid)[:32] if network.ssid.strip() else "[NO SSID]"
        bssid = network.bssid if network.bssid else "N/A"
        signal = f"{network.signal_strength} dBm" if network.signal_strength is not None else "N/A"
        channel = str(network.channel) if network.channel else "N/A"
        freq = f"{network.frequency}" if network.frequency else "N/A"
        security = network.security_type if network.security_type else "NONE"

        print(f"{ssid:<32} {bssid:<17} {signal:<8} {channel:<8} {freq:<6} {security}")

    print(f"\nTotal networks found: {len(networks)}")

def check_sudo():
    """Check if the script is running with sudo privileges."""
    return os.geteuid() == 0

def main():
    parser = argparse.ArgumentParser(description="Linux Wi-Fi Network Scanner")
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
    parser.add_argument(
        "--interface",
        "-i",
        help="Specify the wireless interface to use"
    )
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Check for sudo privileges
    if not check_sudo():
        logger.error("This script requires sudo privileges to scan networks.")
        logger.error("Please run with: sudo python3 wifi_scanner.py")
        sys.exit(1)

    scanner = WiFiScanner()
    if args.interface:
        scanner.interface = args.interface
        logger.info(f"Using specified interface: {scanner.interface}")
    
    networks = scanner.scan_networks()
    
    encryption_filter = None
    if args.filter:
        encryption_filter = args.filter
    
    display_networks(networks, encryption_filter)

if __name__ == "__main__":
    main() 