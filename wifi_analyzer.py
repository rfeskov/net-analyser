#!/usr/bin/env python3

import subprocess
import re
import sys
import os
import argparse
import time
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from colorama import init, Fore, Style
from dataclasses import dataclass
from enum import Enum
import logging
from wifi_scanner import WiFiScanner, NetworkInfo, EncryptionType
from wifi_db import WiFiDatabase
from wifi_monitor import WiFiMonitor
from threading import Thread
from datetime import datetime

# Initialize colorama
init()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# 5 GHz channel definitions
FREQ_TO_CHANNEL = {
    5180: 36, 5200: 40, 5220: 44, 5240: 48,  # UNII-1
    5260: 52, 5280: 56, 5300: 60, 5320: 64,  # UNII-2A (DFS)
    5340: 68, 5360: 72, 5380: 76, 5400: 80,  # UNII-2A Extended
    5420: 84, 5440: 88, 5460: 92, 5480: 96,  # UNII-2A Extended
    5500: 100, 5520: 104, 5540: 108, 5560: 112,  # UNII-2C (DFS)
    5580: 116, 5600: 120, 5620: 124, 5640: 128,  # UNII-2C (DFS)
    5660: 132, 5680: 136, 5700: 140,  # UNII-2C (DFS)
    5745: 149, 5765: 153, 5785: 157, 5805: 161,  # UNII-3
    5825: 165  # UNII-3
}

# DFS channels (require radar detection)
DFS_CHANNELS = list(range(52, 65)) + list(range(100, 141))

# Channel width definitions (MHz)
CHANNEL_WIDTHS = {
    20: 1,  # Number of channels occupied
    40: 2,
    80: 4
}

@dataclass
class ChannelAnalysis:
    channel: int
    networks: List[NetworkInfo]
    signal_strength_avg: float
    congestion_score: float
    recommendation: str
    is_dfs: bool = False
    channel_width: int = 20

@dataclass
class BandAnalysis:
    frequency: str
    channels: Dict[int, ChannelAnalysis]
    total_networks: int
    congestion_score: float
    recommendation: str

@dataclass
class SecurityAnalysis:
    network: NetworkInfo
    issues: List[str]
    recommendations: List[str]

class AnalysisResult:
    def __init__(self):
        self.band_analysis: Dict[str, BandAnalysis] = {}
        self.security_issues: List[SecurityAnalysis] = []
        self.weak_signals: List[NetworkInfo] = []
        self.recommendations: List[str] = []

def get_5ghz_channel(frequency: str) -> Optional[int]:
    """Convert frequency in MHz to 5 GHz channel number."""
    try:
        logger.debug(f"Converting frequency: {frequency}")
        # Convert frequency string to integer (it's already in MHz)
        freq = int(frequency)
        logger.debug(f"Converted to integer: {freq}")
        
        # Look up channel in our dictionary
        if freq in FREQ_TO_CHANNEL:
            channel = FREQ_TO_CHANNEL[freq]
            logger.debug(f"Found channel {channel} for frequency {freq}")
            return channel
        else:
            logger.warning(f"Unknown 5 GHz frequency: {freq} MHz")
            return None
    except (ValueError, AttributeError) as e:
        logger.warning(f"Error converting frequency {frequency} to channel: {e}")
        return None

def is_dfs_channel(channel: int) -> bool:
    """Check if a channel is in the DFS range."""
    return channel in DFS_CHANNELS

def get_5ghz_overlapping_channels(channel: int, width: int = 20) -> List[int]:
    """Get list of channels that overlap with the given 5 GHz channel and width."""
    if width not in CHANNEL_WIDTHS:
        return [channel]
    
    # Get all valid channels
    valid_channels = sorted(FREQ_TO_CHANNEL.values())
    
    # Find the index of the channel in the list
    try:
        center_idx = valid_channels.index(channel)
    except ValueError:
        logger.warning(f"Channel {channel} not found in valid channels list")
        return [channel]
    
    # Calculate the range of channels to include based on width
    num_channels = CHANNEL_WIDTHS[width]
    start_idx = max(0, center_idx - (num_channels // 2))
    end_idx = min(len(valid_channels), center_idx + (num_channels // 2) + 1)
    
    return valid_channels[start_idx:end_idx]

def get_overlapping_channels(channel: int) -> List[int]:
    """Get list of channels that overlap with the given 2.4 GHz channel."""
    # In 2.4 GHz, each channel is 5 MHz wide but requires 20 MHz bandwidth
    # Channels overlap with adjacent channels
    if channel == 1:
        return [1, 2, 3, 4, 5]
    elif channel == 6:
        return [4, 5, 6, 7, 8]
    elif channel == 11:
        return [9, 10, 11, 12, 13]
    return [channel]

def analyze_channel_congestion(networks: List[NetworkInfo]) -> Dict[str, BandAnalysis]:
    """Analyze channel congestion in both 2.4 GHz and 5 GHz bands."""
    analysis = {}
    
    # Group networks by frequency band
    band_networks = {
        '2.4': [n for n in networks if n.frequency == '2.4'],
        '5': [n for n in networks if n.frequency != '2.4']  # Changed this line
    }
    
    logger.debug(f"2.4 GHz networks: {len(band_networks['2.4'])}")
    logger.debug(f"5 GHz networks: {len(band_networks['5'])}")
    
    for band, band_nets in band_networks.items():
        if not band_nets:
            continue
            
        if band == '2.4':
            # Existing 2.4 GHz analysis
            target_channels = [1, 6, 11]
            channel_analysis = {}
            
            for channel in target_channels:
                overlapping_channels = get_overlapping_channels(channel)
                channel_networks = [n for n in band_nets if n.channel in overlapping_channels]
                
                if channel_networks:
                    signals = [n.signal_strength for n in channel_networks if n.signal_strength is not None]
                    avg_signal = np.mean(signals) if signals else -100
                    congestion_score = len(channel_networks) * (1 + (avg_signal / 100))
                    
                    recommendation = (
                        "Highly congested" if congestion_score > 5 else
                        "Moderately congested" if congestion_score > 3 else
                        "Good channel choice"
                    )
                    
                    channel_analysis[channel] = ChannelAnalysis(
                        channel=channel,
                        networks=channel_networks,
                        signal_strength_avg=avg_signal,
                        congestion_score=congestion_score,
                        recommendation=recommendation
                    )
        else:
            # 5 GHz analysis
            channel_analysis = {}
            used_channels = set()
            
            # First, collect all used channels
            for network in band_nets:
                logger.debug(f"Processing 5 GHz network: {network.ssid}, frequency: {network.frequency}, channel: {network.channel}")
                channel = get_5ghz_channel(network.frequency)
                if channel is not None:
                    used_channels.add(channel)
                    logger.debug(f"Found 5 GHz network on channel {channel}: {network.ssid}")
            
            logger.debug(f"Total unique 5 GHz channels found: {len(used_channels)}")
            logger.debug(f"5 GHz channels: {sorted(used_channels)}")
            
            # Analyze each used channel
            for channel in sorted(used_channels):
                # Get overlapping channels based on assumed width (default 20 MHz)
                overlapping_channels = get_5ghz_overlapping_channels(channel)
                channel_networks = [n for n in band_nets 
                                  if get_5ghz_channel(n.frequency) in overlapping_channels]
                
                logger.debug(f"Analyzing channel {channel}:")
                logger.debug(f"  - Overlapping channels: {overlapping_channels}")
                logger.debug(f"  - Networks on this channel: {len(channel_networks)}")
                
                if channel_networks:
                    signals = [n.signal_strength for n in channel_networks if n.signal_strength is not None]
                    avg_signal = np.mean(signals) if signals else -100
                    congestion_score = len(channel_networks) * (1 + (avg_signal / 100))
                    
                    logger.debug(f"  - Average signal: {avg_signal:.1f} dBm")
                    logger.debug(f"  - Congestion score: {congestion_score:.2f}")
                    
                    is_dfs = is_dfs_channel(channel)
                    recommendation = (
                        "DFS channel - requires radar detection" if is_dfs else
                        "Highly congested" if congestion_score > 5 else
                        "Moderately congested" if congestion_score > 3 else
                        "Good channel choice"
                    )
                    
                    channel_analysis[channel] = ChannelAnalysis(
                        channel=channel,
                        networks=channel_networks,
                        signal_strength_avg=avg_signal,
                        congestion_score=congestion_score,
                        recommendation=recommendation,
                        is_dfs=is_dfs
                    )
        
        # Calculate band-wide metrics
        total_networks = len(band_nets)
        avg_congestion = np.mean([ca.congestion_score for ca in channel_analysis.values()]) if channel_analysis else 0
        
        band_recommendation = (
            f"Consider using {'5 GHz' if band == '2.4' else '2.4 GHz'} "
            f"if possible" if avg_congestion > 4 else
            "Good band choice"
        )
        
        analysis[band] = BandAnalysis(
            frequency=band,
            channels=channel_analysis,
            total_networks=total_networks,
            congestion_score=avg_congestion,
            recommendation=band_recommendation
        )
    
    return analysis

def analyze_security(networks: List[NetworkInfo]) -> List[SecurityAnalysis]:
    """Analyze security settings of networks, with special attention to 5 GHz."""
    security_issues = []
    
    for network in networks:
        issues = []
        recommendations = []
        
        # Check for weak encryption
        if network.security_type:
            if "WEP" in network.security_type:
                issues.append("Using outdated WEP encryption")
                recommendations.append("Upgrade to WPA2 or WPA3")
            elif "WPA" in network.security_type and "WPA2" not in network.security_type:
                issues.append("Using WPA (not WPA2/WPA3)")
                recommendations.append("Upgrade to WPA2 or WPA3")
        
        # Check for open networks
        if not network.security_type or network.security_type == "Open":
            issues.append("Network is open (no encryption)")
            recommendations.append("Enable WPA2 or WPA3 encryption")
        
        # Additional warning for 5 GHz networks with weak security
        if network.frequency == '5' and any(issue.startswith("Using") for issue in issues):
            issues.append("Weak security on 5 GHz network")
            recommendations.append("5 GHz networks should use WPA2/WPA3 for optimal security")
        
        if issues:
            security_issues.append(SecurityAnalysis(
                network=network,
                issues=issues,
                recommendations=recommendations
            ))
    
    return security_issues

def analyze_signal_strength(networks: List[NetworkInfo]) -> List[NetworkInfo]:
    """Identify networks with weak signal strength."""
    WEAK_SIGNAL_THRESHOLD = -70
    return [n for n in networks 
            if n.signal_strength is not None 
            and n.signal_strength < WEAK_SIGNAL_THRESHOLD]

def generate_recommendations(analysis: AnalysisResult) -> List[str]:
    """Generate overall recommendations based on analysis."""
    recommendations = []
    
    # Band recommendations
    if analysis.band_analysis:
        # Get congestion scores for both bands
        band_scores = {
            band: band_data.congestion_score 
            for band, band_data in analysis.band_analysis.items()
        }
        
        # Find the best channel in each band
        best_channels = {}
        for band, band_data in analysis.band_analysis.items():
            if band_data.channels:
                best_channel = min(band_data.channels.values(), 
                                 key=lambda x: x.congestion_score)
                best_channels[band] = best_channel
        
        # Compare congestion scores and provide practical recommendations
        if band_scores['2.4'] > 10 and band_scores['5'] > 10:
            # Both bands are highly congested
            recommendations.append(
                "Both 2.4 GHz and 5 GHz bands are highly congested. "
                "Consider using the following channels:"
            )
            for band, channel in best_channels.items():
                recommendations.append(
                    f"  - {band} GHz band: Channel {channel.channel} "
                    f"(congestion score: {channel.congestion_score:.2f})"
                )
        else:
            # One band is better than the other
            better_band = min(band_scores.items(), key=lambda x: x[1])[0]
            worse_band = '5' if better_band == '2.4' else '2.4'
            
            recommendations.append(
                f"The {better_band} GHz band has lower congestion. "
                f"Best channel: {best_channels[better_band].channel} "
                f"(congestion score: {best_channels[better_band].congestion_score:.2f})"
            )
            
            if best_channels[worse_band].congestion_score < 10:
                recommendations.append(
                    f"Alternatively, you could use the {worse_band} GHz band on channel "
                    f"{best_channels[worse_band].channel} "
                    f"(congestion score: {best_channels[worse_band].congestion_score:.2f})"
                )
        
        # Add DFS channel warnings if applicable
        for band, channel in best_channels.items():
            if channel.is_dfs:
                recommendations.append(
                    f"Note: Channel {channel.channel} in {band} GHz band is a DFS channel "
                    "and requires radar detection"
                )
    
    # Security recommendations
    if analysis.security_issues:
        recommendations.append(
            f"Found {len(analysis.security_issues)} networks with security issues. "
            "Consider upgrading to WPA2/WPA3 encryption."
        )
    
    # Signal strength recommendations
    if analysis.weak_signals:
        recommendations.append(
            f"Found {len(analysis.weak_signals)} networks with weak signal strength "
            "(below -70 dBm). Consider adjusting antenna position or using a signal booster."
        )
    
    return recommendations

def display_analysis(analysis: AnalysisResult):
    """Display analysis results with color-coded output."""
    print("\n=== Wi-Fi Network Analysis ===\n")
    
    # Display metrics explanation
    print(f"{Fore.CYAN}Analysis Metrics:{Style.RESET_ALL}")
    print("Congestion Score: Higher is worse (0-5: Good, 5-10: Moderate, >10: High congestion)")
    print("Signal Strength: Higher is better (>-50 dBm: Excellent, -50 to -70 dBm: Good, <-70 dBm: Weak)")
    print("DFS Channels: Yellow indicates channels requiring radar detection")
    print("Security: Red indicates outdated or missing encryption")
    print("\n" + "="*50 + "\n")
    
    # Display band analysis
    for band, band_data in analysis.band_analysis.items():
        print(f"{Fore.CYAN}{band} GHz Band Analysis:{Style.RESET_ALL}")
        print(f"Total networks: {band_data.total_networks}")
        print(f"Average congestion: {band_data.congestion_score:.2f}")
        print(f"Recommendation: {band_data.recommendation}\n")
        
        if band_data.channels:
            print(f"{'Channel':<8} {'Networks':<10} {'Avg Signal':<12} {'Congestion':<12} {'DFS':<5} {'Recommendation'}")
            print("-" * 70)
            
            for channel, data in band_data.channels.items():
                # Color coding
                if data.is_dfs:
                    channel_color = Fore.YELLOW
                elif data.congestion_score > 5:
                    channel_color = Fore.RED
                elif data.congestion_score > 3:
                    channel_color = Fore.YELLOW
                else:
                    channel_color = Fore.GREEN
                
                print(f"{channel_color}{channel:<8}{Style.RESET_ALL} {len(data.networks):<10} "
                      f"{data.signal_strength_avg:.1f} dBm{'':<4} "
                      f"{data.congestion_score:.2f}{'':<12} "
                      f"{'Yes' if data.is_dfs else 'No':<5} "
                      f"{data.recommendation}")
        print()
    
    # Display security issues
    if analysis.security_issues:
        print(f"{Fore.RED}Security Issues:{Style.RESET_ALL}")
        for issue in analysis.security_issues:
            print(f"\nNetwork: {issue.network.ssid} ({issue.network.bssid})")
            for i, problem in enumerate(issue.issues):
                print(f"  {Fore.RED}•{Style.RESET_ALL} {problem}")
                print(f"  {Fore.GREEN}→{Style.RESET_ALL} {issue.recommendations[i]}")
        print()
    
    # Display weak signals
    if analysis.weak_signals:
        print(f"{Fore.YELLOW}Weak Signals:{Style.RESET_ALL}")
        for network in analysis.weak_signals:
            print(f"  • {network.ssid} ({network.bssid}): {network.signal_strength} dBm")
        print()
    
    # Display recommendations
    if analysis.recommendations:
        print(f"{Fore.CYAN}Overall Recommendations:{Style.RESET_ALL}")
        for rec in analysis.recommendations:
            print(f"  • {rec}")
        print()

def record_analysis_to_db(analysis: AnalysisResult, db: WiFiDatabase):
    """Record analysis results to the database."""
    timestamp = datetime.now()
    day_of_week = timestamp.weekday()  # 0-6 (Monday-Sunday)
    month = timestamp.month  # 1-12
    day = timestamp.day  # 1-31
    minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
    
    for band, band_data in analysis.band_analysis.items():
        for channel, channel_data in band_data.channels.items():
            db.record_analysis(
                band=band,
                channel=channel,
                networks_count=len(channel_data.networks),
                avg_signal=channel_data.signal_strength_avg,
                congestion_score=channel_data.congestion_score,
                is_dfs=channel_data.is_dfs,
                day_of_week=day_of_week,
                month=month,
                day=day,
                minutes_since_midnight=minutes_since_midnight
            )

def main():
    parser = argparse.ArgumentParser(description="Advanced Wi-Fi Network Analyzer")
    parser.add_argument("--recommend", action="store_true", help="Perform detailed analysis and provide recommendations")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--record", action="store_true", help="Record data to SQLite database every minute")
    parser.add_argument("--db-path", default="wifi_data.db", help="Path to SQLite database file")
    parser.add_argument("--interface", "-i", help="Specify the wireless interface to use")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Check for sudo privileges
    if os.geteuid() != 0:
        logger.error("This script requires sudo privileges to scan networks.")
        logger.error("Please run with: sudo python3 wifi_analyzer.py")
        sys.exit(1)

    try:
        # Initialize database if recording is enabled
        db = WiFiDatabase(args.db_path) if args.record else None
        
        # Use existing scanner
        scanner = WiFiScanner()
        if args.interface:
            scanner.interface = args.interface
        
        if args.record:
            logger.info(f"Starting continuous recording to {args.db_path}")
            
            # Initialize Wi-Fi monitor
            monitor = WiFiMonitor(scanner.interface)
            
            # Start frame capture in a separate thread
            monitor_thread = Thread(target=monitor.start_capture)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            try:
                while True:
                    # Get basic network information
                    networks = scanner.scan_networks()
                    
                    # Get detailed metrics from frame capture
                    metrics = monitor.get_metrics()
                    
                    # Update network information with metrics
                    for network in networks:
                        if network.bssid in metrics:
                            network_metrics = metrics[network.bssid]
                            # Update network info with metrics
                            network.signal_strength = network_metrics.get('avg_rssi')
                            network.channel = network_metrics.get('channel')
                            network.security_type = network_metrics.get('security_type')
                    
                    # Record data to database
                    db.record_networks(networks, metrics)
                    
                    if args.recommend:
                        analysis = AnalysisResult()
                        analysis.band_analysis = analyze_channel_congestion(networks)
                        analysis.security_issues = analyze_security(networks)
                        analysis.weak_signals = analyze_signal_strength(networks)
                        analysis.recommendations = generate_recommendations(analysis)
                        
                        record_analysis_to_db(analysis, db)
                        display_analysis(analysis)
                    else:
                        from wifi_scanner import display_networks
                        display_networks(networks)
                    
                    time.sleep(60)  # Wait for 1 minute
            except KeyboardInterrupt:
                logger.info("Recording stopped by user")
                monitor.stop()
        else:
            networks = scanner.scan_networks()
            
            if args.recommend:
                analysis = AnalysisResult()
                analysis.band_analysis = analyze_channel_congestion(networks)
                analysis.security_issues = analyze_security(networks)
                analysis.weak_signals = analyze_signal_strength(networks)
                analysis.recommendations = generate_recommendations(analysis)
                
                display_analysis(analysis)
            else:
                from wifi_scanner import display_networks
                display_networks(networks)
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 