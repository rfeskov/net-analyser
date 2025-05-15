#!/usr/bin/env python3

import logging
from scapy.all import *
from scapy.layers.dot11 import *
from typing import Dict, Set, Optional
from collections import defaultdict
import time
from threading import Thread, Event
import subprocess

logger = logging.getLogger(__name__)

# 2.4 GHz channel to frequency mapping
CHANNEL_TO_FREQ_24 = {
    1: 2412, 2: 2417, 3: 2422, 4: 2427, 5: 2432,
    6: 2437, 7: 2442, 8: 2447, 9: 2452, 10: 2457,
    11: 2462, 12: 2467, 13: 2472, 14: 2484
}

# 5 GHz channel to frequency mapping
CHANNEL_TO_FREQ_5 = {
    36: 5180, 40: 5200, 44: 5220, 48: 5240,
    52: 5260, 56: 5280, 60: 5300, 64: 5320,
    100: 5500, 104: 5520, 108: 5540, 112: 5560,
    116: 5580, 120: 5600, 124: 5620, 128: 5640,
    132: 5660, 136: 5680, 140: 5700,
    149: 5745, 153: 5765, 157: 5785, 161: 5805,
    165: 5825
}

# Create reverse mappings
FREQ_TO_CHANNEL_24 = {freq: chan for chan, freq in CHANNEL_TO_FREQ_24.items()}
FREQ_TO_CHANNEL_5 = {freq: chan for chan, freq in CHANNEL_TO_FREQ_5.items()}

def get_channel_from_freq(freq: int) -> tuple[Optional[int], str]:
    """Convert frequency to channel number and determine band."""
    if freq in FREQ_TO_CHANNEL_24:
        return FREQ_TO_CHANNEL_24[freq], '2.4'
    elif freq in FREQ_TO_CHANNEL_5:
        return FREQ_TO_CHANNEL_5[freq], '5'
    return None, 'Unknown'

class WiFiMonitor:
    def __init__(self, interface: str):
        """Initialize the Wi-Fi monitor with the specified interface."""
        self.interface = interface
        self.stop_event = Event()
        self.frame_stats = defaultdict(lambda: defaultdict(int))
        self.client_stats = defaultdict(lambda: defaultdict(int))
        self.sequence_numbers = defaultdict(set)
        self.retry_counts = defaultdict(int)
        self.airtime_stats = defaultdict(int)
        self.phy_rates = defaultdict(list)
        self.rssi_values = defaultdict(list)
        self.network_info = {}  # Store network information
        
        # Set up monitor mode
        self._setup_monitor_mode()
    
    def _setup_monitor_mode(self):
        """Configure the wireless interface for monitor mode."""
        try:
            # Check if interface exists
            result = subprocess.run(['ip', 'link', 'show', self.interface], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Interface {self.interface} not found")
            
            # Check if interface is already in monitor mode
            result = subprocess.run(['iwconfig', self.interface], capture_output=True, text=True)
            if 'Mode:Monitor' in result.stdout:
                logger.info(f"Interface {self.interface} already in monitor mode")
                return
            
            # Stop NetworkManager and wpa_supplicant
            subprocess.run(['systemctl', 'stop', 'NetworkManager'], capture_output=True)
            subprocess.run(['systemctl', 'stop', 'wpa_supplicant'], capture_output=True)
            logger.info("Stopped network services")
            
            # Kill any processes that might interfere
            subprocess.run(['airmon-ng', 'check', 'kill'], capture_output=True)
            
            # Put interface in monitor mode
            subprocess.run(['airmon-ng', 'start', self.interface], capture_output=True)
            logger.info(f"Started monitor mode on {self.interface}")
            
            # Verify monitor mode
            result = subprocess.run(['iwconfig', self.interface], capture_output=True, text=True)
            if 'Mode:Monitor' not in result.stdout:
                raise Exception(f"Failed to set {self.interface} to monitor mode")
            
            # Set interface up
            subprocess.run(['ip', 'link', 'set', self.interface, 'up'], capture_output=True)
            logger.info(f"Set {self.interface} up")
            
            # Verify interface is up
            result = subprocess.run(['ip', 'link', 'show', self.interface], capture_output=True, text=True)
            if 'state UP' not in result.stdout:
                raise Exception(f"Failed to bring {self.interface} up")
            
            logger.info(f"Successfully configured {self.interface} for monitoring")
            
        except Exception as e:
            logger.error(f"Error setting up monitor mode: {e}")
            # Try to restore network services
            subprocess.run(['systemctl', 'start', 'NetworkManager'], capture_output=True)
            subprocess.run(['systemctl', 'start', 'wpa_supplicant'], capture_output=True)
            raise
    
    def __del__(self):
        """Cleanup when the monitor is destroyed."""
        try:
            # Stop the capture
            self.stop()
            
            # Restore network services
            subprocess.run(['systemctl', 'start', 'NetworkManager'], capture_output=True)
            subprocess.run(['systemctl', 'start', 'wpa_supplicant'], capture_output=True)
            logger.info("Restored network services")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _process_frame(self, pkt):
        """Process a captured frame and extract metrics."""
        if not pkt.haslayer(Dot11):
            return
        
        try:
            # Extract basic frame information
            frame_type = pkt.type
            subtype = pkt.subtype
            
            # Get BSSID based on frame type
            if frame_type == 0:  # Management frame
                if subtype == 8:  # Beacon frame
                    bssid = pkt.addr2
                    # Extract SSID from beacon
                    ssid = None
                    for element in pkt[Dot11Elt:]:
                        if element.ID == 0:  # SSID element
                            try:
                                ssid = element.info.decode()
                            except:
                                ssid = str(element.info)
                            break
                    
                    if ssid and bssid:
                        self.network_info[bssid] = {
                            'ssid': ssid,
                            'channel': None,
                            'frequency': None,
                            'security_type': None
                        }
                        
                        # Extract channel and frequency
                        channel = None
                        frequency = None
                        logger.debug(f"Attempting to extract channel for network {ssid}")
                        for element in pkt[Dot11Elt:]:
                            logger.debug(f"Processing element ID {element.ID} for network {ssid}")
                            if element.ID == 3:  # DS Parameter Set
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
                        
                        # If we have a channel, determine frequency
                        if channel is not None:
                            if channel in CHANNEL_TO_FREQ_24:
                                frequency = str(CHANNEL_TO_FREQ_24[channel])
                            elif channel in CHANNEL_TO_FREQ_5:
                                frequency = str(CHANNEL_TO_FREQ_5[channel])
                        
                        # If we still don't have frequency, try to get it from RadioTap
                        if frequency is None and hasattr(pkt, 'ChannelFrequency'):
                            try:
                                freq = pkt.ChannelFrequency
                                channel, band = get_channel_from_freq(freq)
                                frequency = str(freq)
                                logger.debug(f"Found frequency {freq} MHz (channel {channel}) from RadioTap for network {ssid}")
                            except Exception as e:
                                logger.debug(f"Failed to extract frequency from RadioTap for network {ssid}: {e}")
                        
                        # Determine band based on frequency
                        band = '2.4'
                        if frequency:
                            freq = int(frequency)
                            if freq >= 5000:  # 5 GHz starts at 5000 MHz
                                band = '5'
                        
                        self.network_info[bssid]['channel'] = channel
                        self.network_info[bssid]['frequency'] = band
                        
                        # Extract security type
                        security = []
                        for element in pkt[Dot11Elt:]:
                            if element.ID == 48:  # RSN
                                security.append("WPA2")
                            elif element.ID == 221 and element.info.startswith(b'\x00P\xf2\x01\x01\x00'):
                                security.append("WPA")
                        self.network_info[bssid]['security_type'] = "/".join(security) if security else "Open"
                        logger.debug(f"Found security type {self.network_info[bssid]['security_type']} for network {ssid}")
            
            # Get BSSID for all frame types
            bssid = pkt.addr2 if pkt.addr2 else pkt.addr3
            if not bssid:
                return
            
            # Get RSSI from RadioTap header
            if hasattr(pkt, 'dBm_AntSignal'):
                rssi = pkt.dBm_AntSignal
                if -100 <= rssi <= 0:  # Valid RSSI range
                    self.rssi_values[bssid].append(rssi)
                    logger.debug(f"Found RSSI {rssi} dBm for BSSID {bssid} using dBm_AntSignal")
            elif hasattr(pkt, 'notdecoded'):
                # Try different offsets for RSSI
                for offset in range(-4, 0):
                    try:
                        rssi = -(256-ord(pkt.notdecoded[offset:offset+1]))
                        if -100 <= rssi <= 0:  # Valid RSSI range
                            self.rssi_values[bssid].append(rssi)
                            logger.debug(f"Found RSSI {rssi} dBm for BSSID {bssid} using notdecoded")
                            break
                    except (IndexError, TypeError):
                        continue
            
            # Extract PHY rate from RadioTap header
            if hasattr(pkt, 'Rate'):
                rate = pkt.Rate * 500  # Convert to kbps
                if 1000 <= rate <= 6000000:  # Valid rate range (1-6000 Mbps)
                    self.phy_rates[bssid].append(rate)
                    logger.debug(f"Found PHY rate {rate} kbps for BSSID {bssid} using Rate")
            elif hasattr(pkt, 'notdecoded'):
                try:
                    # Try different offsets for rate
                    for offset in range(-2, 0):
                        rate_bytes = pkt.notdecoded[offset:offset+1]
                        if rate_bytes:
                            rate = ord(rate_bytes) * 500  # Convert to kbps
                            if 1000 <= rate <= 6000000:  # Valid rate range (1-6000 Mbps)
                                self.phy_rates[bssid].append(rate)
                                logger.debug(f"Found PHY rate {rate} kbps for BSSID {bssid} using notdecoded")
                                break
                except (IndexError, TypeError):
                    pass
            
            # Track sequence numbers for lost packet detection
            if hasattr(pkt, 'SC') and pkt.SC is not None:
                try:
                    seq_num = pkt.SC >> 4
                    self.sequence_numbers[bssid].add(seq_num)
                    logger.debug(f"Tracked sequence number {seq_num} for BSSID {bssid}")
                except (TypeError, AttributeError):
                    logger.debug(f"Failed to extract sequence number for BSSID {bssid}")
            
            # Check for retry flag
            if hasattr(pkt, 'FCfield') and pkt.FCfield is not None:
                if pkt.FCfield & 0x8:  # Retry flag
                    self.retry_counts[bssid] += 1
                    logger.debug(f"Incremented retry count for BSSID {bssid}")
            
            # Calculate airtime
            if hasattr(pkt, 'duration') and pkt.duration is not None:
                self.airtime_stats[bssid] += pkt.duration
                logger.debug(f"Added {pkt.duration} to airtime for BSSID {bssid}")
            
            # Track clients
            if pkt.addr1 and pkt.addr1 != 'ff:ff:ff:ff:ff:ff':
                self.client_stats[bssid][pkt.addr1] += 1
                logger.debug(f"Tracked client {pkt.addr1} for BSSID {bssid}")
            
            # Update frame statistics
            frame_type_str = f"{frame_type}_{subtype}"
            self.frame_stats[bssid][frame_type_str] += 1
            logger.debug(f"Updated frame stats for BSSID {bssid}: {frame_type_str}")
            
        except Exception as e:
            logger.debug(f"Error processing frame: {e}")
            # Don't log every error to avoid spam
    
    def _channel_hopper(self):
        """Hop between channels to cover the entire spectrum."""
        channels = list(range(1, 14)) + [36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 149, 153, 157, 161, 165]
        current_channel = 0
        
        while not self.stop_event.is_set():
            try:
                channel = channels[current_channel]
                subprocess.run(['iw', 'dev', self.interface, 'set', 'channel', str(channel)], 
                             capture_output=True, check=True)
                logger.debug(f"Switched to channel {channel}")
                current_channel = (current_channel + 1) % len(channels)
                time.sleep(0.1)  # Short delay between channel hops
            except Exception as e:
                logger.error(f"Error during channel hopping: {e}")
                time.sleep(1)
    
    def start_capture(self):
        """Start capturing and analyzing frames."""
        # Start channel hopping in a separate thread
        hopper_thread = Thread(target=self._channel_hopper)
        hopper_thread.daemon = True
        hopper_thread.start()
        
        try:
            # Start sniffing with a timeout to allow channel hopping
            sniff(iface=self.interface, prn=self._process_frame, store=0, 
                  stop_filter=lambda _: self.stop_event.is_set())
        except Exception as e:
            logger.error(f"Error during frame capture: {e}")
        finally:
            self.stop_event.set()
    
    def get_metrics(self) -> Dict[str, Dict]:
        """Get collected metrics for all networks."""
        metrics = {}
        
        for bssid in self.frame_stats.keys():
            # Get network info
            network_info = self.network_info.get(bssid, {})
            
            # Calculate average RSSI
            avg_rssi = sum(self.rssi_values[bssid]) / len(self.rssi_values[bssid]) if self.rssi_values[bssid] else None
            
            # Calculate average PHY rate
            avg_phy_rate = sum(self.phy_rates[bssid]) / len(self.phy_rates[bssid]) if self.phy_rates[bssid] else None
            
            # Estimate lost packets from sequence number gaps
            seq_nums = sorted(self.sequence_numbers[bssid])
            lost_packets = 0
            if len(seq_nums) > 1:
                for i in range(len(seq_nums) - 1):
                    if seq_nums[i+1] - seq_nums[i] > 1:
                        lost_packets += seq_nums[i+1] - seq_nums[i] - 1
            
            metrics[bssid] = {
                'ssid': network_info.get('ssid', 'Unknown'),
                'channel': network_info.get('channel', 1),
                'frequency': network_info.get('frequency', 'Unknown'),
                'security_type': network_info.get('security_type', 'Unknown'),
                'phy_rate': int(avg_phy_rate) if avg_phy_rate else None,
                'client_count': len(self.client_stats[bssid]),
                'retransmission_count': self.retry_counts[bssid],
                'lost_packets': lost_packets,
                'airtime_ms': self.airtime_stats[bssid],
                'frame_stats': dict(self.frame_stats[bssid]),
                'avg_rssi': avg_rssi
            }
            
            logger.debug(f"Metrics for {bssid}: {metrics[bssid]}")
        
        return metrics
    
    def stop(self):
        """Stop the frame capture and channel hopping."""
        self.stop_event.set()
        
        try:
            # Restore interface to managed mode
            subprocess.run(['airmon-ng', 'stop', self.interface], capture_output=True)
            logger.info(f"Restored {self.interface} to managed mode")
        except Exception as e:
            logger.error(f"Error restoring interface: {e}") 