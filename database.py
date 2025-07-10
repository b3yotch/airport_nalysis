# database.py
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlightDatabase:
    def __init__(self, db_path: str = "flight_data.db"):
        self.db_path = db_path
        self.local = threading.local()
        self._create_tables()
        
    @property
    def conn(self):
        """Thread-safe connection"""
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(self.db_path)
            self.local.conn.row_factory = sqlite3.Row
        return self.local.conn
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database operations"""
        cursor = self.conn.cursor()
        try:
            yield cursor
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()
    
    def _create_tables(self):
        """Create all necessary tables"""
        with self.get_cursor() as cursor:
            # Flights table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS flights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT NOT NULL,
                airline TEXT,
                flight_number TEXT,
                departure_airport TEXT,
                arrival_airport TEXT,
                departure_time TEXT,
                arrival_time TEXT,
                scheduled_departure TEXT,
                scheduled_arrival TEXT,
                status TEXT,
                aircraft_type TEXT,
                raw_data TEXT
            )''')
            
            # Price history table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                route TEXT NOT NULL,
                departure_date DATE,
                price REAL,
                currency TEXT DEFAULT 'USD',
                airline TEXT,
                class TEXT,
                data_source TEXT,
                seats_available INTEGER
            )''')
            
            # Route statistics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS route_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                route TEXT NOT NULL,
                flight_count INTEGER DEFAULT 0,
                avg_price REAL,
                min_price REAL,
                max_price REAL,
                total_seats INTEGER,
                load_factor REAL,
                UNIQUE(date, route)
            )''')
            
            # Airport demand table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS airport_demand (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                airport_code TEXT NOT NULL,
                arrivals_count INTEGER DEFAULT 0,
                departures_count INTEGER DEFAULT 0,
                avg_delay_minutes REAL,
                demand_score REAL
            )''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_flights_airports ON flights(departure_airport, arrival_airport)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_flights_timestamp ON flights(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_route ON price_history(route, departure_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_route_stats ON route_stats(date, route)')
            
            logger.info("Database tables created successfully")
    
    def insert_flights(self, flights_data: List[Dict]) -> int:
        """Insert multiple flight records"""
        if not flights_data:
            return 0
            
        with self.get_cursor() as cursor:
            inserted = 0
            for flight in flights_data:
                try:
                    cursor.execute('''
                    INSERT INTO flights (
                        source, airline, flight_number, departure_airport, 
                        arrival_airport, departure_time, arrival_time,
                        scheduled_departure, scheduled_arrival, status, 
                        aircraft_type, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        flight.get('source'),
                        flight.get('airline'),
                        flight.get('flight_number'),
                        flight.get('departure_airport'),
                        flight.get('arrival_airport'),
                        flight.get('departure_time'),
                        flight.get('arrival_time'),
                        flight.get('scheduled_departure'),
                        flight.get('scheduled_arrival'),
                        flight.get('status'),
                        flight.get('aircraft_type'),
                        json.dumps(flight)
                    ))
                    inserted += 1
                except Exception as e:
                    logger.error(f"Error inserting flight: {e}")
                    
        logger.info(f"Inserted {inserted} flight records")
        return inserted
    
    def insert_price_data(self, price_data: List[Dict]) -> int:
        """Insert price history records"""
        if not price_data:
            return 0
            
        with self.get_cursor() as cursor:
            inserted = 0
            for price in price_data:
                try:
                    route = f"{price.get('departure', '')}-{price.get('arrival', '')}"
                    cursor.execute('''
                    INSERT INTO price_history (
                        route, departure_date, price, currency, airline, 
                        class, data_source, seats_available
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        route,
                        price.get('departure_date'),
                        price.get('price'),
                        price.get('currency', 'USD'),
                        price.get('airline'),
                        price.get('class', 'Economy'),
                        price.get('source'),
                        price.get('seats_available')
                    ))
                    inserted += 1
                except Exception as e:
                    logger.error(f"Error inserting price: {e}")
                    
        logger.info(f"Inserted {inserted} price records")
        return inserted
    
    def update_route_statistics(self, date: datetime = None):
        """Calculate and update route statistics"""
        if date is None:
            date = datetime.now().date()
            
        with self.get_cursor() as cursor:
            # Get flight counts by route
            cursor.execute('''
            SELECT 
                departure_airport || '-' || arrival_airport as route,
                COUNT(*) as flight_count,
                AVG(CASE WHEN status = 'delayed' THEN 1 ELSE 0 END) as delay_rate
            FROM flights
            WHERE DATE(timestamp) = ?
            GROUP BY route
            ''', (date,))
            
            flight_stats = cursor.fetchall()
            
            for stat in flight_stats:
                route = stat['route']
                
                # Get price statistics for the route
                cursor.execute('''
                SELECT 
                    AVG(price) as avg_price,
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    SUM(seats_available) as total_seats
                FROM price_history
                WHERE route = ? AND DATE(timestamp) = ?
                ''', (route, date))
                
                price_stat = cursor.fetchone()
                
                # Insert or update route statistics
                cursor.execute('''
                INSERT OR REPLACE INTO route_stats 
                (date, route, flight_count, avg_price, min_price, max_price, total_seats)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    date,
                    route,
                    stat['flight_count'],
                    price_stat['avg_price'] if price_stat else None,
                    price_stat['min_price'] if price_stat else None,
                    price_stat['max_price'] if price_stat else None,
                    price_stat['total_seats'] if price_stat else None
                ))
    
    def get_route_statistics(self, 
                           start_date: datetime = None, 
                           end_date: datetime = None,
                           routes: List[str] = None) -> pd.DataFrame:
        """Get route statistics for analysis"""
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=7)
            
        query = '''
        SELECT 
            date,
            route,
            flight_count,
            avg_price,
            min_price,
            max_price,
            total_seats,
            load_factor
        FROM route_stats
        WHERE date BETWEEN ? AND ?
        '''
        
        params = [start_date.date(), end_date.date()]
        
        if routes:
            placeholders = ','.join(['?' for _ in routes])
            query += f' AND route IN ({placeholders})'
            params.extend(routes)
            
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_popular_routes(self, limit: int = 10, days_back: int = 7) -> pd.DataFrame:
        """Get most popular routes by flight frequency"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        query = '''
        SELECT 
            departure_airport || '-' || arrival_airport as route,
            COUNT(*) as flight_count,
            COUNT(DISTINCT airline) as airline_count,
            AVG(CASE WHEN status IN ('on_time', 'landed') THEN 1 ELSE 0 END) as on_time_rate
        FROM flights
        WHERE timestamp BETWEEN ? AND ?
        GROUP BY route
        ORDER BY flight_count DESC
        LIMIT ?
        '''
        
        return pd.read_sql_query(query, self.conn, params=[start_date, end_date, limit])
    
    def get_price_trends(self, route: str, days_back: int = 30) -> pd.DataFrame:
        """Get price trends for a specific route"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        query = '''
        SELECT 
            DATE(timestamp) as date,
            AVG(price) as avg_price,
            MIN(price) as min_price,
            MAX(price) as max_price,
            COUNT(*) as sample_size,
            airline
        FROM price_history
        WHERE route = ? AND timestamp BETWEEN ? AND ?
        GROUP BY DATE(timestamp), airline
        ORDER BY date, airline
        '''
        
        return pd.read_sql_query(query, self.conn, params=[route, start_date, end_date])
    
    def get_demand_analysis(self, airports: List[str] = None) -> pd.DataFrame:
        """Analyze demand patterns for airports"""
        query = '''
        WITH airport_activity AS (
            SELECT 
                airport,
                SUM(arrivals) as total_arrivals,
                SUM(departures) as total_departures,
                AVG(daily_flights) as avg_daily_flights
            FROM (
                SELECT 
                    departure_airport as airport,
                    0 as arrivals,
                    COUNT(*) as departures,
                    COUNT(*) / COUNT(DISTINCT DATE(timestamp)) as daily_flights
                FROM flights
                WHERE timestamp >= datetime('now', '-30 days')
                GROUP BY departure_airport
                
                UNION ALL
                
                SELECT 
                    arrival_airport as airport,
                    COUNT(*) as arrivals,
                    0 as departures,
                    COUNT(*) / COUNT(DISTINCT DATE(timestamp)) as daily_flights
                FROM flights
                WHERE timestamp >= datetime('now', '-30 days')
                GROUP BY arrival_airport
            )
            GROUP BY airport
        )
        SELECT 
            airport,
            total_arrivals,
            total_departures,
            total_arrivals + total_departures as total_traffic,
            avg_daily_flights,
            CASE 
                WHEN avg_daily_flights > 100 THEN 'Very High'
                WHEN avg_daily_flights > 50 THEN 'High'
                WHEN avg_daily_flights > 20 THEN 'Medium'
                ELSE 'Low'
            END as demand_level
        FROM airport_activity
        '''
        
        if airports:
            placeholders = ','.join(['?' for _ in airports])
            query += f' WHERE airport IN ({placeholders})'
            return pd.read_sql_query(query, self.conn, params=airports)
        
        return pd.read_sql_query(query, self.conn)
    
    def get_peak_travel_times(self, airport: str = None) -> pd.DataFrame:
        """Identify peak travel times"""
        query = '''
        SELECT 
            strftime('%H', departure_time) as hour_of_day,
            strftime('%w', departure_time) as day_of_week,
            COUNT(*) as flight_count,
            AVG(CASE WHEN status = 'delayed' THEN 1 ELSE 0 END) as delay_rate
        FROM flights
        WHERE departure_time IS NOT NULL
        '''
        
        params = []
        if airport:
            query += ' AND (departure_airport = ? OR arrival_airport = ?)'
            params = [airport, airport]
            
        query += ' GROUP BY hour_of_day, day_of_week'
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Remove old data to manage database size"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self.get_cursor() as cursor:
            cursor.execute('DELETE FROM flights WHERE timestamp < ?', (cutoff_date,))
            cursor.execute('DELETE FROM price_history WHERE timestamp < ?', (cutoff_date,))
            cursor.execute('DELETE FROM route_stats WHERE date < ?', (cutoff_date.date(),))
            cursor.execute('DELETE FROM airport_demand WHERE timestamp < ?', (cutoff_date,))
            
            # Vacuum to reclaim disk space
            cursor.execute('VACUUM')
            
        logger.info(f"Cleaned up data older than {days_to_keep} days")
    
    def close(self):
        """Close database connection"""
        if hasattr(self.local, 'conn'):
            self.local.conn.close()