#!/usr/bin/env python3

import sqlite3
import logging
from datetime import datetime
from typing import List
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
            
            # Create networks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS networks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ssid TEXT NOT NULL,
                    bssid TEXT,
                    signal_strength INTEGER,
                    channel INTEGER,
                    frequency TEXT,
                    security_type TEXT,
                    timestamp DATETIME NOT NULL
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
                    timestamp DATETIME NOT NULL
                )
            ''')
            
            conn.commit()
    
    def record_networks(self, networks: List[NetworkInfo]):
        """Record network information to the database."""
        timestamp = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for network in networks:
                cursor.execute('''
                    INSERT INTO networks (
                        ssid, bssid, signal_strength, channel,
                        frequency, security_type, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    network.ssid,
                    network.bssid,
                    network.signal_strength,
                    network.channel,
                    network.frequency,
                    network.security_type,
                    timestamp
                ))
            
            conn.commit()
            logger.info(f"Recorded {len(networks)} networks to database")
    
    def record_analysis(self, band: str, channel: int, networks_count: int,
                       avg_signal: float, congestion_score: float, is_dfs: bool):
        """Record channel analysis information to the database."""
        timestamp = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO analysis (
                    band, channel, networks_count, avg_signal_strength,
                    congestion_score, is_dfs, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                band,
                channel,
                networks_count,
                avg_signal,
                congestion_score,
                is_dfs,
                timestamp
            ))
            
            conn.commit()
            logger.info(f"Recorded analysis for {band} GHz band, channel {channel}")
    
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