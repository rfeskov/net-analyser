import subprocess
import json
import logging
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

@dataclass
class NetworkInfo:
    ssid: str
    bssid: str
    signal_strength: int
    channel: int
    frequency: float
    encryption_type: str

class WirelessDiagnostics:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def _run_wdutil(self, command: str) -> str:
        """Run wdutil command and return its output."""
        try:
            result = subprocess.run(
                ['/usr/sbin/wdutil', command],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running wdutil: {e}")
            self.logger.error(f"Error output: {e.stderr}")
            raise

    def get_interface_info(self) -> str:
        """Get information about the Wi-Fi interface."""
        return this._run_wdutil('info')

    def get_scan_info(self) -> str:
        """Get information about available networks."""
        return this._run_wdutil('scan')

    def get_diagnostics(self) -> str:
        """Get detailed diagnostics information."""
        return this._run_wdutil('diagnose')

    def parse_scan_results(self, output: str) -> List[NetworkInfo]:
        """Parse the scan results into NetworkInfo objects."""
        networks = []
        current_network = None
        
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if 'SSID:' in line:
                if current_network:
                    networks.append(current_network)
                ssid = line.split('SSID:')[1].strip()
                current_network = NetworkInfo(
                    ssid=ssid,
                    bssid='',
                    signal_strength=-100,
                    channel=1,
                    frequency=2.4,
                    encryption_type='Unknown'
                )
            elif current_network:
                if 'BSSID:' in line:
                    current_network.bssid = line.split('BSSID:')[1].strip()
                elif 'RSSI:' in line:
                    try:
                        current_network.signal_strength = int(line.split('RSSI:')[1].strip())
                    except ValueError:
                        pass
                elif 'Channel:' in line:
                    try:
                        current_network.channel = int(line.split('Channel:')[1].strip())
                    except ValueError:
                        pass
                elif 'Frequency:' in line:
                    try:
                        freq_str = line.split('Frequency:')[1].strip()
                        current_network.frequency = float(freq_str.replace('GHz', ''))
                    except ValueError:
                        pass
                elif 'Security:' in line:
                    current_network.encryption_type = line.split('Security:')[1].strip()
        
        if current_network:
            networks.append(current_network)
            
        return networks 