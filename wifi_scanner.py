#!/usr/bin/env python3

import platform
import subprocess
import logging
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
from wireless_diagnostics import WirelessDiagnostics, NetworkInfo

class EncryptionType(Enum):
    NONE = "None"
    WEP = "WEP"
    WPA = "WPA"
    WPA2 = "WPA2"
    WPA3 = "WPA3"
    UNKNOWN = "Unknown"

class WiFiScanner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.wd = WirelessDiagnostics()
        
    def _check_permissions(self) -> bool:
        """Check if we have necessary permissions to scan networks."""
        try:
            # Try to get interface info to check permissions
            self.wd.get_interface_info()
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Permission error: {e}")
            return False

    def scan_networks(self, encryption_filter: Optional[EncryptionType] = None) -> List[NetworkInfo]:
        """Scan for available Wi-Fi networks."""
        if not self._check_permissions():
            raise PermissionError("Insufficient permissions to scan networks")

        try:
            # Get scan results
            scan_output = self.wd.get_scan_info()
            networks = self.wd.parse_scan_results(scan_output)
            
            # Apply encryption filter if specified
            if encryption_filter:
                networks = [net for net in networks 
                          if net.encryption_type == encryption_filter.value]
            
            return networks
            
        except Exception as e:
            self.logger.error(f"Error scanning networks: {e}")
            raise

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        scanner = WiFiScanner()
        networks = scanner.scan_networks()
        
        print("\nAvailable Wi-Fi Networks:")
        print("-" * 80)
        print(f"{'SSID':<30} {'Signal':<10} {'Channel':<10} {'Frequency':<10} {'Security':<10}")
        print("-" * 80)
        
        for net in networks:
            print(f"{net.ssid:<30} {net.signal_strength:>3} dBm   {net.channel:<10} "
                  f"{net.frequency:<10.1f} {net.encryption_type:<10}")
            
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 