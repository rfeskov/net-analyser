#!/usr/bin/env python3

import subprocess
import re
import sys
import argparse
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from colorama import init, Fore, Style
from dataclasses import dataclass
from enum import Enum
import logging
from wifi_scanner import WiFiScanner, NetworkInfo, EncryptionType

# Initialize colorama
init()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

@dataclass
class ChannelAnalysis:
    channel: int
    networks: List[NetworkInfo]
    signal_strength_avg: float
    congestion_score: float
    recommendation: str

@dataclass
class SecurityAnalysis:
    network: NetworkInfo
    issues: List[str]
    recommendations: List[str]

class AnalysisResult:
    def __init__(self):
        self.channel_analysis: Dict[int, ChannelAnalysis] = {}
        self.security_issues: List[SecurityAnalysis] = []
        self.weak_signals: List[NetworkInfo] = []
        self.recommendations: List[str] = []

def analyze_channel_congestion(networks: List[NetworkInfo]) -> Dict[int, ChannelAnalysis]:
    """Analyze channel congestion in 2.4 GHz band."""
    # Focus on 2.4 GHz channels 1, 6, 11
    target_channels = [1, 6, 11]
    analysis = {}
    
    for channel in target_channels:
        # Get networks on this channel and overlapping channels
        overlapping_channels = get_overlapping_channels(channel)
        channel_networks = [n for n in networks 
                          if n.channel in overlapping_channels 
                          and n.frequency == '2.4']
        
        if channel_networks:
            # Calculate average signal strength
            signals = [n.signal_strength for n in channel_networks if n.signal_strength is not None]
            avg_signal = np.mean(signals) if signals else -100
            
            # Calculate congestion score (higher is worse)
            congestion_score = len(channel_networks) * (1 + (avg_signal / 100))
            
            # Generate recommendation
            if congestion_score > 5:
                recommendation = f"Highly congested. Consider switching to 5 GHz if possible."
            elif congestion_score > 3:
                recommendation = f"Moderately congested. Monitor performance."
            else:
                recommendation = f"Good channel choice."
            
            analysis[channel] = ChannelAnalysis(
                channel=channel,
                networks=channel_networks,
                signal_strength_avg=avg_signal,
                congestion_score=congestion_score,
                recommendation=recommendation
            )
    
    return analysis

def get_overlapping_channels(channel: int) -> List[int]:
    """Get list of channels that overlap with the given channel."""
    # In 2.4 GHz, each channel is 5 MHz wide but requires 20 MHz bandwidth
    # Channels overlap with adjacent channels
    if channel == 1:
        return [1, 2, 3, 4, 5]
    elif channel == 6:
        return [4, 5, 6, 7, 8]
    elif channel == 11:
        return [9, 10, 11, 12, 13]
    return [channel]

def analyze_security(networks: List[NetworkInfo]) -> List[SecurityAnalysis]:
    """Analyze security settings of networks."""
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
    
    # Channel recommendations
    if analysis.channel_analysis:
        best_channel = min(analysis.channel_analysis.values(), 
                          key=lambda x: x.congestion_score)
        recommendations.append(
            f"Recommended channel: {best_channel.channel} "
            f"(congestion score: {best_channel.congestion_score:.2f})"
        )
    
    # Security recommendations
    if analysis.security_issues:
        recommendations.append(
            f"Found {len(analysis.security_issues)} networks with security issues"
        )
    
    # Signal strength recommendations
    if analysis.weak_signals:
        recommendations.append(
            f"Found {len(analysis.weak_signals)} networks with weak signal strength"
        )
    
    return recommendations

def display_analysis(analysis: AnalysisResult):
    """Display analysis results with color-coded output."""
    print("\n=== Wi-Fi Network Analysis ===\n")
    
    # Display channel analysis
    if analysis.channel_analysis:
        print(f"{Fore.CYAN}Channel Analysis:{Style.RESET_ALL}")
        print(f"{'Channel':<8} {'Networks':<10} {'Avg Signal':<12} {'Congestion':<12} {'Recommendation'}")
        print("-" * 60)
        
        for channel, data in analysis.channel_analysis.items():
            color = Fore.RED if data.congestion_score > 5 else Fore.YELLOW if data.congestion_score > 3 else Fore.GREEN
            print(f"{channel:<8} {len(data.networks):<10} "
                  f"{data.signal_strength_avg:.1f} dBm{'':<4} "
                  f"{color}{data.congestion_score:.2f}{Style.RESET_ALL:<12} "
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
        print(f"{Fore.CYAN}Recommendations:{Style.RESET_ALL}")
        for rec in analysis.recommendations:
            print(f"  • {rec}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Advanced Wi-Fi Network Analyzer")
    parser.add_argument("--recommend", action="store_true", help="Perform detailed analysis and provide recommendations")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Check for sudo privileges
    if os.geteuid() != 0:
        logger.error("This script requires sudo privileges to scan networks.")
        logger.error("Please run with: sudo python3 wifi_analyzer.py")
        sys.exit(1)

    try:
        # Use existing scanner
        scanner = WiFiScanner()
        networks = scanner.scan_networks()
        
        if args.recommend:
            # Perform detailed analysis
            analysis = AnalysisResult()
            analysis.channel_analysis = analyze_channel_congestion(networks)
            analysis.security_issues = analyze_security(networks)
            analysis.weak_signals = analyze_signal_strength(networks)
            analysis.recommendations = generate_recommendations(analysis)
            
            # Display results
            display_analysis(analysis)
        else:
            # Display basic scan results
            from wifi_scanner import display_networks
            display_networks(networks)
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 