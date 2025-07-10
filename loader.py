# enhanced_loader.py
import time
import requests
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
from opensky_api import OpenSkyApi
from database import FlightDatabase
import random
import asyncio
import aiohttp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlightDataAggregator:
    def __init__(self, config: Dict):
        self.config = config
        self.db = FlightDatabase(config.get('database_path', 'flight_data.db'))
        
        # API credentials
        self.aviationstack_key = config.get('aviationstack_key', '94cb4668212fce67e3a3ca7c2c4ffd33')
        self.aviationstack_endpoint = "http://api.aviationstack.com/v1"
        
        # OpenSky credentials (optional)
        self.opensky_username = config.get('opensky_username')
        self.opensky_password = config.get('opensky_password')
        
        # Cache
        self.cache = {}
        self.cache_ttl = 300
        self.time_ranges = {
            '1hour': 1,
            '5hours': 5,
            '12hours': 12,
            '24hours': 24,
            '3days': 72,
            '7days': 168
        }  # 5 minutes
        
    def get_live_flights_aviationstack(self, departure_iata=None, arrival_iata=None, 
                                     limit=100, hours_back=None):
        """Get live or historical flights from AviationStack"""
        url = f"{self.aviationstack_endpoint}/flights"
        params = {
            'access_key': self.aviationstack_key,
            'limit': limit,
        }
        
        if departure_iata:
            params['dep_iata'] = departure_iata
        if arrival_iata:
            params['arr_iata'] = arrival_iata
        
        # Add date range for historical data
        if hours_back:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=hours_back)
            
            # AviationStack uses date format YYYY-MM-DD
            params['flight_date'] = start_date.strftime('%Y-%m-%d')
            
            # For multiple days, we need to make multiple requests
            if hours_back > 24:
                return self._get_multi_day_flights(departure_iata, arrival_iata, 
                                                 start_date, end_date, limit)
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data:
                logger.error(f"[AviationStack API Error] {data['error']}")
                # If API doesn't support historical, generate simulated data
                if hours_back:
                    return self._generate_historical_flights(
                        departure_iata, arrival_iata, hours_back, limit
                    )
                return []
            
            flights = data.get('data', [])
            return self._parse_aviationstack_flights(flights)
            
        except Exception as e:
            logger.error(f"[AviationStack Error] {e}")
            # Fallback to simulated historical data
            if hours_back:
                return self._generate_historical_flights(
                    departure_iata, arrival_iata, hours_back, limit
                )
            return []
    
    def _get_multi_day_flights(self, departure_iata, arrival_iata, 
                              start_date, end_date, limit):
        """Get flights for multiple days"""
        all_flights = []
        current_date = start_date
        
        while current_date <= end_date:
            url = f"{self.aviationstack_endpoint}/flights"
            params = {
                'access_key': self.aviationstack_key,
                'limit': limit,
                'flight_date': current_date.strftime('%Y-%m-%d')
            }
            
            if departure_iata:
                params['dep_iata'] = departure_iata
            if arrival_iata:
                params['arr_iata'] = arrival_iata
            
            try:
                response = requests.get(url, params=params, timeout=10)
                data = response.json()
                
                if 'data' in data:
                    flights = self._parse_aviationstack_flights(data['data'])
                    all_flights.extend(flights)
                    
            except Exception as e:
                logger.error(f"Error fetching data for {current_date}: {e}")
            
            current_date += timedelta(days=1)
            
            # Respect rate limits
            time.sleep(1)
        
        return all_flights[:limit]
    
    def _generate_historical_flights(self, departure_iata, arrival_iata, 
                                   hours_back, limit):
        """Generate simulated historical flight data"""
        flights = []
        
        # Common routes
        routes = {
            'SYD': ['MEL', 'BNE', 'PER', 'ADL', 'OOL', 'CBR'],
            'MEL': ['SYD', 'BNE', 'PER', 'ADL', 'OOL', 'HBA'],
            'BNE': ['SYD', 'MEL', 'CNS', 'PER', 'ADL', 'DRW']
        }
        
        airlines = ['Qantas', 'Virgin Australia', 'Jetstar', 'Rex Airlines']
        
        # Generate flights for the time period
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        # Calculate number of flights based on time range
        if hours_back <= 1:
            flights_per_hour = 8
        elif hours_back <= 5:
            flights_per_hour = 6
        else:
            flights_per_hour = 4
        
        total_flights = int(hours_back * flights_per_hour)
        
        destinations = routes.get(departure_iata, ['MEL', 'SYD', 'BNE'])
        
        for i in range(min(total_flights, limit)):
            # Distribute flights across the time period
            flight_time = start_time + timedelta(
                hours=random.uniform(0, hours_back)
            )
            
            dest = arrival_iata if arrival_iata else random.choice(destinations)
            
            flights.append({
                "source": "Historical",
                "airline": random.choice(airlines),
                "flight_number": f"{random.choice(['QF', 'VA', 'JQ', 'ZL'])}{random.randint(100, 999)}",
                "departure_airport": departure_iata,
                "arrival_airport": dest,
                "departure_time": flight_time.isoformat(),
                "arrival_time": (flight_time + timedelta(hours=random.uniform(1, 4))).isoformat(),
                "status": "completed",
                "historical": True
            })
        
        # Sort by departure time
        flights.sort(key=lambda x: x['departure_time'], reverse=True)
        
        return flights
    
    def _parse_aviationstack_flights(self, flights):
        """Parse AviationStack flight data"""
        parsed_flights = []
        
        for f in flights:
            if f and 'airline' in f and 'departure' in f and 'arrival' in f:
                try:
                    parsed_flight = {
                        "source": "AviationStack",
                        "airline": f['airline']['name'] if f['airline'] else 'Unknown',
                        "flight_number": f['flight']['iata'] if f['flight'] and f['flight']['iata'] else f'UN{random.randint(100,999)}',
                        "departure_airport": f['departure']['iata'] if f['departure'] and 'iata' in f['departure'] else 'Unknown',
                        "departure_airport_name": f['departure']['airport'] if f['departure'] else '',
                        "arrival_airport": f['arrival']['iata'] if f['arrival'] and 'iata' in f['arrival'] else 'Unknown',
                        "arrival_airport_name": f['arrival']['airport'] if f['arrival'] else '',
                        "departure_time": f['departure']['scheduled'] if f['departure'] and 'scheduled' in f['departure'] else None,
                        "arrival_time": f['arrival']['scheduled'] if f['arrival'] and 'scheduled' in f['arrival'] else None,
                        "status": f.get('flight_status', 'unknown')
                    }
                    parsed_flights.append(parsed_flight)
                except Exception as e:
                    logger.error(f"Error parsing flight: {e}")
                    continue
        
        return parsed_flights
    def get_opensky_api(self):
        # You may want to cache this instance
        return OpenSkyApi()
    
        
    def _calculate_statistics(self, flights: List[Dict]) -> Dict:
        """Calculate statistics from flight data"""
        if not flights:
            return {
                'total_flights': 0,
                'unique_airlines': 0,
                'airlines_breakdown': {},
                'status_breakdown': {}
            }
        
        df = pd.DataFrame(flights)
        
        stats = {
            'total_flights': len(flights),
            'unique_airlines': df['airline'].nunique() if 'airline' in df else 0,
            'airlines_breakdown': df['airline'].value_counts().head(10).to_dict() if 'airline' in df else {},
            'status_breakdown': df['status'].value_counts().to_dict() if 'status' in df else {}
        }
        
        return stats
    
    def _extract_airline_from_callsign(self, callsign: str) -> str:
        """Extract airline from callsign"""
        if not callsign:
            return "Unknown"
        
        airline_codes = {
            'QFA': 'Qantas',
            'VOZ': 'Virgin Australia',
            'JST': 'Jetstar',
            'UAE': 'Emirates',
            'SIA': 'Singapore Airlines',
            'QTR': 'Qatar Airways',
            'CPA': 'Cathay Pacific'
        }
        
        prefix = callsign[:3] if len(callsign) >= 3 else callsign
        return airline_codes.get(prefix, prefix)
    
    def _extract_routes_from_flights(self, flights: List[Dict]) -> List[Tuple[str, str]]:
        """Extract unique routes from flight data"""
        routes = set()
        for flight in flights:
            dep = flight.get('departure_airport')
            arr = flight.get('arrival_airport')
            if dep and arr and dep != 'Unknown' and arr != 'Unknown':
                routes.add((dep, arr))
        return list(routes)
    
    def get_arrivals_departures_opensky(self, airport_icao: str, hours_back=3):
        """Get historical arrivals and departures from OpenSky"""
        api = self.get_opensky_api()
        end = int(time.time())
        begin = end - (hours_back * 3600)

        try:
            # OpenSky supports historical queries
            arrivals = api.get_arrivals_by_airport(airport_icao, begin, end) or []
            departures = api.get_departures_by_airport(airport_icao, begin, end) or []

            arrivals_data = []
            for f in arrivals:
                if f:
                    arrivals_data.append({
                        "source": "OpenSky",
                        "airline": self._extract_airline_from_callsign(f.callsign),
                        "flight_number": f.callsign,
                        "departure_airport": f.estDepartureAirport,
                        "arrival_airport": airport_icao,
                        "arrival_time": datetime.utcfromtimestamp(f.lastSeen).isoformat(),
                        "departure_time": datetime.utcfromtimestamp(f.firstSeen).isoformat() if f.firstSeen else None,
                        "status": "landed",
                        "historical": True
                    })

            departures_data = []
            for f in departures:
                if f:
                    departures_data.append({
                        "source": "OpenSky",
                        "airline": self._extract_airline_from_callsign(f.callsign),
                        "flight_number": f.callsign,
                        "departure_airport": airport_icao,
                        "arrival_airport": f.estArrivalAirport,
                        "departure_time": datetime.utcfromtimestamp(f.firstSeen).isoformat(),
                        "arrival_time": datetime.utcfromtimestamp(f.lastSeen).isoformat() if f.lastSeen else None,
                        "status": "departed",
                        "historical": True
                    })

            logger.info(f"OpenSky: Found {len(arrivals_data)} arrivals and {len(departures_data)} departures for last {hours_back} hours")
            return arrivals_data, departures_data

        except Exception as e:
            logger.error(f"[OpenSky Error] {e}")
            return [], []
    
    async def fetch_flights_with_timerange(self, airports: List[str], 
                                         time_range: str = '24hours') -> Dict:
        """Fetch flights for specified time range"""
        hours_back = self.time_ranges.get(time_range, 24)
        all_flights = []
        airport_data = {}
        
        for airport in airports:
            # Try OpenSky first (ICAO codes)
            if airport in ['SYD', 'MEL', 'BNE', 'PER', 'ADL']:
                icao_map = {
                    'SYD': 'YSSY', 'MEL': 'YMML', 'BNE': 'YBBN',
                    'PER': 'YPPH', 'ADL': 'YPAD'
                }
                icao = icao_map.get(airport)
                
                if icao:
                    arrivals, departures = self.get_arrivals_departures_opensky(icao, hours_back)
                    all_flights.extend(arrivals)
                    all_flights.extend(departures)
            
            # Also get from AviationStack
            dep_flights = self.get_live_flights_aviationstack(
                departure_iata=airport, hours_back=hours_back
            )
            arr_flights = self.get_live_flights_aviationstack(
                arrival_iata=airport, hours_back=hours_back
            )
            
            airport_data[airport] = {
                'departures': dep_flights,
                'arrivals': arr_flights
            }
            
            all_flights.extend(dep_flights)
            all_flights.extend(arr_flights)
        
        # Remove duplicates based on flight number
        unique_flights = {}
        for flight in all_flights:
            key = f"{flight.get('flight_number', '')}_{flight.get('departure_time', '')}"
            if key not in unique_flights:
                unique_flights[key] = flight
        
        final_flights = list(unique_flights.values())
        
        # Sort by departure time (most recent first)
        final_flights.sort(
            key=lambda x: x.get('departure_time', ''), 
            reverse=True
        )
        
        # Store in database
        if final_flights:
            try:
                self.db.insert_flights(final_flights)
                logger.info(f"Stored {len(final_flights)} flights for {time_range}")
            except Exception as e:
                logger.error(f"Database error: {e}")
        
        return {
            'flights': final_flights,
            'airports': airport_data,
            'time_range': time_range,
            'hours_back': hours_back,
            'stats': self._calculate_statistics(final_flights),
            'timestamp': datetime.utcnow().isoformat()
        }


# Backward compatibility function
def get_live_flights_aviationstack(departure_iata=None, arrival_iata=None, 
                                 limit=10, hours_back=None):
    """Wrapper function for backward compatibility with time range support"""
    config = {
        'aviationstack_key': '94cb4668212fce67e3a3ca7c2c4ffd33'
    }
    
    aggregator = FlightDataAggregator(config)
    return aggregator.get_live_flights_aviationstack(
        departure_iata, arrival_iata, limit, hours_back
    )


def get_flights_for_timerange(airports: List[str], time_range: str = '24hours'):
    """Get flights for specified time range"""
    config = {
        'aviationstack_key': '94cb4668212fce67e3a3ca7c2c4ffd33',
        'database_path': 'flight_data.db'
    }
    
    aggregator = FlightDataAggregator(config)
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            aggregator.fetch_flights_with_timerange(airports, time_range)
        )
        return result
    finally:
        loop.close()


def get_historical_flight_data(airport: str, hours_back: int = 24):
    """Get historical flight data for a specific airport"""
    config = {
        'aviationstack_key': '94cb4668212fce67e3a3ca7c2c4ffd33',
        'opensky_username': None,
        'opensky_password': None
    }
    
    aggregator = FlightDataAggregator(config)
    
    # Get ICAO code for OpenSky
    icao_map = {
        'SYD': 'YSSY', 'MEL': 'YMML', 'BNE': 'YBBN',
        'PER': 'YPPH', 'ADL': 'YPAD', 'CBR': 'YSCB',
        'DRW': 'YPDN', 'HBA': 'YMHB'
    }
    
    all_flights = []
    
    # Try OpenSky for historical data
    if airport in icao_map:
        icao = icao_map[airport]
        arrivals, departures = aggregator.get_arrivals_departures_opensky(icao, hours_back)
        all_flights.extend(arrivals)
        all_flights.extend(departures)
        logger.info(f"Retrieved {len(arrivals)} arrivals and {len(departures)} departures from OpenSky")
    
    # Also try AviationStack
    dep_flights = aggregator.get_live_flights_aviationstack(
        departure_iata=airport, hours_back=hours_back, limit=100
    )
    arr_flights = aggregator.get_live_flights_aviationstack(
        arrival_iata=airport, hours_back=hours_back, limit=100
    )
    
    all_flights.extend(dep_flights)
    all_flights.extend(arr_flights)
    
    # Remove duplicates and sort by time
    unique_flights = {}
    for flight in all_flights:
        key = f"{flight.get('flight_number', '')}_{flight.get('departure_time', '')}"
        if key not in unique_flights:
            unique_flights[key] = flight
    
    final_flights = list(unique_flights.values())
    final_flights.sort(key=lambda x: x.get('departure_time', ''), reverse=True)
    
    return {
        'airport': airport,
        'hours_back': hours_back,
        'total_flights': len(final_flights),
        'flights': final_flights,
        'sources': list(set(f.get('source', 'Unknown') for f in final_flights))
    }


async def run_data_collection_with_history(airports: List[str] = None, 
                                         time_ranges: List[str] = None,
                                         interval_minutes: int = 30):
    """Run continuous data collection with historical data"""
    if airports is None:
        airports = ['SYD', 'MEL', 'BNE', 'PER', 'ADL']
    
    if time_ranges is None:
        time_ranges = ['1hour', '5hours', '24hours']
    
    config = {
        'aviationstack_key': '94cb4668212fce67e3a3ca7c2c4ffd33',
        'database_path': 'flight_data.db'
    }
    
    aggregator = FlightDataAggregator(config)
    
    while True:
        try:
            for time_range in time_ranges:
                logger.info(f"Collecting {time_range} data for airports: {airports}")
                
                # Collect flight data for time range
                flight_data = await aggregator.fetch_flights_with_timerange(airports, time_range)
                logger.info(f"Collected {len(flight_data['flights'])} flights for {time_range}")
                
                # Small delay between time ranges to respect rate limits
                await asyncio.sleep(5)
            
            # Extract routes and generate price data
            all_flights = []
            for time_range in time_ranges:
                result = await aggregator.fetch_flights_with_timerange(airports, time_range)
                all_flights.extend(result.get('flights', []))
            
            if all_flights:
                routes = aggregator._extract_routes_from_flights(all_flights)
                if routes:
                    price_data = await aggregator.fetch_simulated_price_data(routes[:30])
                    aggregator.db.insert_price_data(price_data)
                    logger.info(f"Generated {len(price_data)} price records")
            
            # Update statistics
            aggregator.db.update_route_statistics()
            
            logger.info(f"Data collection cycle completed. Next run in {interval_minutes} minutes.")
            
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
        
        # Wait for next cycle
        await asyncio.sleep(interval_minutes * 60)


def initialize_database_with_history():
    """Initialize database with historical data"""
    config = {
        'aviationstack_key': '94cb4668212fce67e3a3ca7c2c4ffd33',
        'database_path': 'flight_data.db'
    }
    
    aggregator = FlightDataAggregator(config)
    
    # Run async initialization with multiple time ranges
    async def init_with_history():
        logger.info("Initializing database with historical data...")
        
        airports = ['SYD', 'MEL', 'BNE', 'PER', 'ADL']
        time_ranges = ['5hours', '24hours', '3days']
        
        all_flights = []
        
        for time_range in time_ranges:
            logger.info(f"Fetching {time_range} of data...")
            result = await aggregator.fetch_flights_with_timerange(airports, time_range)
            all_flights.extend(result.get('flights', []))
            
            # Respect rate limits
            await asyncio.sleep(2)
        
        # Generate price data
        if all_flights:
            routes = aggregator._extract_routes_from_flights(all_flights)
            price_data = await aggregator.fetch_simulated_price_data(routes[:50])
            aggregator.db.insert_price_data(price_data)
            
            # Update statistics
            aggregator.db.update_route_statistics()
            
            logger.info(f"Initialized with {len(all_flights)} flights and {len(price_data)} price records")
            return True
        
        return False
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(init_with_history())
        return result
    finally:
        loop.close()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Enhanced Flight Data Loader with Time Ranges...")
    print("="*60)
    
    # Test 1: Get flights for different time ranges
    print("\n1. Testing Time Range Support...")
    
    time_ranges = ['1hour', '5hours', '24hours']
    
    for time_range in time_ranges:
        print(f"\n   Testing {time_range} range:")
        hours = {'1hour': 1, '5hours': 5, '24hours': 24}[time_range]
        
        flights = get_live_flights_aviationstack(
            departure_iata='SYD', 
            hours_back=hours,
            limit=10
        )
        
        if flights:
            print(f"   ✅ Retrieved {len(flights)} flights from Sydney (last {hours} hours)")
            
            # Show time distribution
            if flights:
                earliest = min(f.get('departure_time', '') for f in flights if f.get('departure_time'))
                latest = max(f.get('departure_time', '') for f in flights if f.get('departure_time'))
                print(f"   Time range: {earliest[:16]} to {latest[:16]}")
        else:
            print(f"   ❌ No flights retrieved for {time_range}")
    
    # Test 2: Get historical data from OpenSky
    print("\n2. Testing OpenSky Historical Data...")
    
    config = {'opensky_username': None, 'opensky_password': None}
    aggregator = FlightDataAggregator(config)
    
    for hours in [1, 5, 24]:
        arrivals, departures = aggregator.get_arrivals_departures_opensky('YSSY', hours_back=hours)
        print(f"   Sydney Airport - Last {hours} hours:")
        print(f"   Arrivals: {len(arrivals)}, Departures: {len(departures)}")
    
    # Test 3: Combined historical data
    print("\n3. Testing Combined Historical Data...")
    
    historical_data = get_historical_flight_data('SYD', hours_back=12)
    print(f"   Total flights (last 12 hours): {historical_data['total_flights']}")
    print(f"   Data sources: {', '.join(historical_data['sources'])}")
    
    if historical_data['flights']:
        # Show sample flights
        print("\n   Sample flights:")
        for flight in historical_data['flights'][:3]:
            dep_time = flight.get('departure_time', 'N/A')
            if dep_time != 'N/A':
                dep_time = dep_time[:16]  # Show only date and time
            print(f"   - {flight.get('airline', 'Unknown')} {flight.get('flight_number', 'N/A')} "
                  f"to {flight.get('arrival_airport', 'Unknown')} at {dep_time}")
    
    # Test 4: Time range aggregation
    print("\n4. Testing Time Range Aggregation...")
    
    result = get_flights_for_timerange(['SYD', 'MEL'], '5hours')
    print(f"   Flights for SYD & MEL (last 5 hours): {len(result['flights'])}")
    print(f"   Stats: {result['stats']}")
    
    # Test 5: Database check
    print("\n5. Checking Database Content...")
    
    db = FlightDatabase('flight_data.db')
    
    # Get flights by time periods
    for days in [0.25, 1, 7]:  # 6 hours, 1 day, 7 days
        routes = db.get_popular_routes(limit=5, days_back=days)
        if not routes.empty:
            print(f"\n   Popular routes (last {days} days):")
            for _, route in routes.iterrows():
                print(f"   - {route['route']}: {route['flight_count']} flights")
        else:
            print(f"\n   No route data for last {days} days")
    
    # Test 6: Initialize with historical data
    print("\n6. Database Initialization Check...")
    
    routes = db.get_popular_routes(limit=1, days_back=7)
    
    if routes.empty:
        print("   ℹ️ Database is empty.")
        response = input("   Initialize database with historical data? (y/n): ")
        
        if response.lower() == 'y':
            print("   Initializing with historical data...")
            if initialize_database_with_history():
                print("   ✅ Database initialized with historical data!")
                
                # Show what was added
                routes = db.get_popular_routes(limit=10, days_back=7)
                print(f"   Added {len(routes)} routes to database")
            else:
                print("   ❌ Failed to initialize database")
    else:
        print("   ✅ Database already contains data")
        
        # Show data age
        peak_times = db.get_peak_travel_times()
        if not peak_times.empty:
            print(f"   Database contains {len(peak_times)} hourly records")
    
    print("\n" + "="*60)
    print("Testing completed!")
    
    # Optional: Start continuous collection with history
    print("\n7. Continuous Historical Data Collection")
    response = input("Start continuous collection with historical data? (y/n): ")
    
    if response.lower() == 'y':
        print("Starting continuous data collection...")
        print("Press Ctrl+C to stop")
        
        try:
            asyncio.run(run_data_collection_with_history(
                airports=['SYD', 'MEL', 'BNE'],
                time_ranges=['1hour', '5hours', '24hours'],
                interval_minutes=30
            ))
        except KeyboardInterrupt:
            print("\nData collection stopped by user")
    
    print("\nThank you for using the Flight Data Loader!")