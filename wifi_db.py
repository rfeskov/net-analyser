#!/usr/bin/env python3

import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Set
from wifi_scanner import NetworkInfo

logger = logging.getLogger(__name__)

class WiFiDatabase:
    def __init__(self, db_path: str = "wifi_data.db"):
        """Initialize the database connection and create tables if they don't exist."""
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create networks table with additional metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS networks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ssid TEXT NOT NULL,
                    bssid TEXT,
                    signal_strength INTEGER,
                    channel INTEGER,
                    frequency TEXT,
                    security_type TEXT,
                    phy_rate INTEGER,
                    client_count INTEGER,
                    retransmission_count INTEGER,
                    lost_packets INTEGER,
                    airtime_ms INTEGER,
                    timestamp DATETIME NOT NULL,
                    day_of_week INTEGER,
                    month INTEGER,
                    day INTEGER,
                    minutes_since_midnight INTEGER
                )
            ''')
            
            # Create analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    band TEXT NOT NULL,
                    channel INTEGER NOT NULL,
                    networks_count INTEGER NOT NULL,
                    avg_signal_strength REAL,
                    congestion_score REAL,
                    is_dfs BOOLEAN,
                    timestamp DATETIME NOT NULL,
                    day_of_week INTEGER,
                    month INTEGER,
                    day INTEGER,
                    minutes_since_midnight INTEGER
                )
            ''')
            
            # Create frame statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS frame_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bssid TEXT NOT NULL,
                    frame_type TEXT NOT NULL,
                    frame_count INTEGER NOT NULL,
                    avg_rssi REAL,
                    avg_phy_rate REAL,
                    retry_count INTEGER,
                    timestamp DATETIME NOT NULL
                )
            ''')
            
            # Create client tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS clients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mac_address TEXT NOT NULL,
                    bssid TEXT NOT NULL,
                    first_seen DATETIME NOT NULL,
                    last_seen DATETIME NOT NULL,
                    frame_count INTEGER NOT NULL,
                    avg_rssi REAL,
                    avg_phy_rate REAL
                )
            ''')
            
            conn.commit()
    
    def record_networks(self, networks: List[NetworkInfo], metrics: Dict[str, Dict] = None):
        """Record network information to the database with additional metrics."""
        timestamp = datetime.now()
        day_of_week = timestamp.weekday()  # 0-6 (Monday-Sunday)
        month = timestamp.month  # 1-12
        day = timestamp.day  # 1-31
        minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for network in networks:
                # Get metrics for this network if available
                network_metrics = metrics.get(network.bssid, {}) if metrics else {}
                
                cursor.execute('''
                    INSERT INTO networks (
                        ssid, bssid, signal_strength, channel,
                        frequency, security_type, phy_rate,
                        client_count, retransmission_count,
                        lost_packets, airtime_ms, timestamp,
                        day_of_week, month, day, minutes_since_midnight
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    network.ssid,
                    network.bssid,
                    network.signal_strength,
                    network.channel,
                    network.frequency,
                    network.security_type,
                    network_metrics.get('phy_rate'),
                    network_metrics.get('client_count'),
                    network_metrics.get('retransmission_count'),
                    network_metrics.get('lost_packets'),
                    network_metrics.get('airtime_ms'),
                    timestamp,
                    day_of_week,
                    month,
                    day,
                    minutes_since_midnight
                ))
            
            conn.commit()
            logger.info(f"Recorded {len(networks)} networks to database")
    
    def record_frame_stats(self, bssid: str, frame_type: str, frame_count: int,
                          avg_rssi: float, avg_phy_rate: float, retry_count: int):
        """Record frame statistics to the database."""
        timestamp = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO frame_stats (
                    bssid, frame_type, frame_count,
                    avg_rssi, avg_phy_rate, retry_count,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                bssid,
                frame_type,
                frame_count,
                avg_rssi,
                avg_phy_rate,
                retry_count,
                timestamp
            ))
            
            conn.commit()
            logger.info(f"Recorded frame stats for {bssid}, type: {frame_type}")
    
    def update_client(self, mac_address: str, bssid: str, rssi: float, phy_rate: float):
        """Update client information in the database."""
        timestamp = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if client exists
            cursor.execute('''
                SELECT id, frame_count, avg_rssi, avg_phy_rate
                FROM clients
                WHERE mac_address = ? AND bssid = ?
            ''', (mac_address, bssid))
            
            result = cursor.fetchone()
            
            if result:
                # Update existing client
                client_id, frame_count, avg_rssi, avg_phy_rate = result
                new_frame_count = frame_count + 1
                new_avg_rssi = ((avg_rssi * frame_count) + rssi) / new_frame_count
                new_avg_phy_rate = ((avg_phy_rate * frame_count) + phy_rate) / new_frame_count
                
                cursor.execute('''
                    UPDATE clients
                    SET frame_count = ?,
                        avg_rssi = ?,
                        avg_phy_rate = ?,
                        last_seen = ?
                    WHERE id = ?
                ''', (new_frame_count, new_avg_rssi, new_avg_phy_rate, timestamp, client_id))
            else:
                # Insert new client
                cursor.execute('''
                    INSERT INTO clients (
                        mac_address, bssid, first_seen, last_seen,
                        frame_count, avg_rssi, avg_phy_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (mac_address, bssid, timestamp, timestamp, 1, rssi, phy_rate))
            
            conn.commit()
            logger.info(f"Updated client {mac_address} for network {bssid}")
    
    def get_recent_networks(self, minutes: int = 60) -> List[dict]:
        """Get network records from the last specified minutes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM networks
                WHERE timestamp >= datetime('now', ?)
                ORDER BY timestamp DESC
            ''', (f'-{minutes} minutes',))
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_recent_analysis(self, minutes: int = 60) -> List[dict]:
        """Get analysis records from the last specified minutes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM analysis
                WHERE timestamp >= datetime('now', ?)
                ORDER BY timestamp DESC
            ''', (f'-{minutes} minutes',))
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_network_metrics(self, bssid: str, minutes: int = 60) -> Dict:
        """Get comprehensive metrics for a specific network."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get latest network info
            cursor.execute('''
                SELECT * FROM networks
                WHERE bssid = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (bssid,))
            
            network_info = dict(zip([col[0] for col in cursor.description], cursor.fetchone()))
            
            # Get frame statistics
            cursor.execute('''
                SELECT frame_type, SUM(frame_count) as total_frames,
                       AVG(avg_rssi) as avg_rssi,
                       AVG(avg_phy_rate) as avg_phy_rate,
                       SUM(retry_count) as total_retries
                FROM frame_stats
                WHERE bssid = ? AND timestamp >= datetime('now', ?)
                GROUP BY frame_type
            ''', (bssid, f'-{minutes} minutes'))
            
            frame_stats = {}
            for row in cursor.fetchall():
                frame_stats[row[0]] = {
                    'total_frames': row[1],
                    'avg_rssi': row[2],
                    'avg_phy_rate': row[3],
                    'total_retries': row[4]
                }
            
            # Get client information
            cursor.execute('''
                SELECT COUNT(*) as active_clients,
                       AVG(avg_rssi) as avg_client_rssi,
                       AVG(avg_phy_rate) as avg_client_phy_rate
                FROM clients
                WHERE bssid = ? AND last_seen >= datetime('now', ?)
            ''', (bssid, f'-{minutes} minutes'))
            
            client_info = dict(zip([col[0] for col in cursor.description], cursor.fetchone()))
            
            return {
                'network_info': network_info,
                'frame_stats': frame_stats,
                'client_info': client_info
            }
    
    def record_analysis(self, band: str, channel: int, networks_count: int,
                       avg_signal: float, congestion_score: float, is_dfs: bool,
                       day_of_week: int, month: int, day: int, minutes_since_midnight: int):
        """Record analysis results to the database."""
        timestamp = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO analysis (
                    band, channel, networks_count,
                    avg_signal_strength, congestion_score,
                    is_dfs, timestamp, day_of_week,
                    month, day, minutes_since_midnight
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                band, channel, networks_count,
                avg_signal, congestion_score,
                is_dfs, timestamp, day_of_week,
                month, day, minutes_since_midnight
            ))
            
            conn.commit()
            logger.info(f"Recorded analysis for {band} GHz band, channel {channel}") 