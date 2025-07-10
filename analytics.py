# analytics.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import google.generativeai as genai
from groq import Groq
from database import FlightDatabase
from config import Config
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlightAnalytics:
    """Advanced analytics for flight and airline booking data"""
    
    def __init__(self, db_path: str = None):
        self.config = Config()
        self.db = FlightDatabase(db_path or self.config.DATABASE_PATH)
        
        # Initialize AI provider
        self.ai_provider = self.config.AI_PROVIDER
        
        if self.ai_provider == 'gemini' and self.config.GEMINI_API_KEY:
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(self.config.GEMINI_MODEL)
        elif self.ai_provider == 'groq' and self.config.GROQ_API_KEY:
            self.groq_client = Groq(api_key=self.config.GROQ_API_KEY)
        
        # Initialize scalers for ML models
        self.price_scaler = StandardScaler()
        self.demand_scaler = StandardScaler()
        
    def calculate_demand_score(self, 
                             airport: str = None, 
                             route: str = None, 
                             days_back: int = 30) -> Dict:
        """
        Calculate demand score based on multiple factors:
        - Flight frequency
        - Load factor (simulated)
        - Price volatility
        - Seasonal patterns
        """
        if route:
            return self._calculate_route_demand_score(route, days_back)
        elif airport:
            return self._calculate_airport_demand_score(airport, days_back)
        else:
            return self._calculate_overall_demand_scores(days_back)
    
    def _calculate_route_demand_score(self, route: str, days_back: int) -> Dict:
        """Calculate demand score for a specific route"""
        # Get route statistics
        stats = self.db.get_route_statistics(
            start_date=datetime.now() - timedelta(days=days_back),
            routes=[route]
        )
        
        if stats.empty:
            return {'route': route, 'demand_score': 0, 'factors': {}}
        
        # Get price data
        price_data = self.db.get_price_trends(route, days_back)
        
        # Calculate factors
        factors = {
            'flight_frequency': float(stats['flight_count'].sum()),
            'price_volatility': float(price_data['avg_price'].std()) if not price_data.empty else 0,
            'avg_price': float(price_data['avg_price'].mean()) if not price_data.empty else 0,
            'price_range': float(price_data['max_price'].max() - price_data['min_price'].min()) if not price_data.empty else 0
        }
        
        # Calculate weighted demand score (0-100)
        weights = self.config.DEMAND_SCORE_WEIGHTS
        
        # Normalize factors
        freq_score = min(factors['flight_frequency'] / 10, 100)  # Assume 10 flights/day is max
        volatility_score = min(factors['price_volatility'] / 50, 100)  # Higher volatility = higher demand
        stability_score = 100 - volatility_score  # Lower volatility = more stable market
        capacity_score = min(factors['avg_price'] / 500, 100)  # Higher prices indicate high demand
        
        demand_score = (
            weights['flight_frequency'] * freq_score +
            weights['price_volatility'] * volatility_score +
            weights['price_stability'] * stability_score +
            weights['capacity'] * capacity_score
        )
        
        return {
            'route': route,
            'demand_score': round(demand_score, 2),
            'factors': factors,
            'interpretation': self._interpret_demand_score(demand_score)
        }
    
    def _calculate_airport_demand_score(self, airport: str, days_back: int) -> Dict:
        """Calculate demand score for a specific airport"""
        demand_data = self.db.get_demand_analysis([airport])
        
        if demand_data.empty:
            return {'airport': airport, 'demand_score': 0, 'factors': {}}
        
        airport_data = demand_data.iloc[0]
        
        # Calculate demand score based on traffic
        total_traffic = airport_data['total_traffic']
        daily_avg = airport_data['avg_daily_flights']
        
        # Normalize to 0-100 scale
        traffic_score = min(total_traffic / 1000, 100)  # Assume 1000 flights/month is high
        daily_score = min(daily_avg / 50, 100)  # Assume 50 flights/day is high
        
        demand_score = (traffic_score * 0.6 + daily_score * 0.4)
        
        return {
            'airport': airport,
            'demand_score': round(demand_score, 2),
            'factors': {
                'total_traffic': int(total_traffic),
                'avg_daily_flights': round(daily_avg, 2),
                'demand_level': airport_data['demand_level']
            },
            'interpretation': self._interpret_demand_score(demand_score)
        }
    
    def _calculate_overall_demand_scores(self, days_back: int) -> Dict:
        """Calculate demand scores for all airports and routes"""
        airports = self.db.get_demand_analysis()
        routes = self.db.get_popular_routes(limit=50, days_back=days_back)
        
        results = {
            'airports': [],
            'routes': [],
            'summary': {}
        }
        
        # Process airports
        for _, airport in airports.iterrows():
            score = self._calculate_airport_demand_score(airport['airport'], days_back)
            results['airports'].append(score)
        
        # Process routes
        for _, route in routes.iterrows():
            score = self._calculate_route_demand_score(route['route'], days_back)
            results['routes'].append(score)
        
        # Summary statistics
        if results['airports']:
            airport_scores = [a['demand_score'] for a in results['airports']]
            results['summary']['avg_airport_demand'] = round(np.mean(airport_scores), 2)
            results['summary']['top_demand_airports'] = sorted(
                results['airports'], 
                key=lambda x: x['demand_score'], 
                reverse=True
            )[:5]
        
        if results['routes']:
            route_scores = [r['demand_score'] for r in results['routes']]
            results['summary']['avg_route_demand'] = round(np.mean(route_scores), 2)
            results['summary']['top_demand_routes'] = sorted(
                results['routes'], 
                key=lambda x: x['demand_score'], 
                reverse=True
            )[:5]
        
        return results
    
    def _interpret_demand_score(self, score: float) -> str:
        """Interpret demand score into categories"""
        if score >= 80:
            return "Very High Demand"
        elif score >= 60:
            return "High Demand"
        elif score >= 40:
            return "Moderate Demand"
        elif score >= 20:
            return "Low Demand"
        else:
            return "Very Low Demand"
    
    def predict_price_trends(self, 
                           route: str, 
                           days_ahead: int = 7,
                           use_ml: bool = True) -> Dict:
        """Predict price trends using historical data and ML models"""
        # Get historical price data
        historical_data = self.db.get_price_trends(route, days_back=90)
        
        if historical_data.empty or len(historical_data) < 10:
            return {
                'route': route,
                'predictions': [],
                'error': 'Insufficient historical data'
            }
        
        if use_ml and self.config.ENABLE_ML_PREDICTIONS:
            return self._ml_price_prediction(route, historical_data, days_ahead)
        else:
            return self._statistical_price_prediction(route, historical_data, days_ahead)
    
    def _ml_price_prediction(self, route: str, historical_data: pd.DataFrame, days_ahead: int) -> Dict:
        """Use machine learning for price prediction"""
        try:
            # Prepare features
            df = historical_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Create features
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
            df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
            
            # Aggregate by date (multiple airlines)
            daily_avg = df.groupby('date').agg({
                'avg_price': 'mean',
                'min_price': 'min',
                'max_price': 'max',
                'day_of_week': 'first',
                'day_of_month': 'first',
                'month': 'first',
                'days_since_start': 'first'
            }).reset_index()
            
            # Prepare for ML
            features = ['day_of_week', 'day_of_month', 'month', 'days_since_start']
            X = daily_avg[features]
            y = daily_avg['avg_price']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Make future predictions
            last_date = daily_avg['date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead)
            
            future_features = pd.DataFrame({
                'day_of_week': future_dates.dayofweek,
                'day_of_month': future_dates.day,
                'month': future_dates.month,
                'days_since_start': [(d - daily_avg['date'].min()).days for d in future_dates]
            })
            
            predictions = model.predict(future_features)
            
            # Calculate confidence intervals (simplified)
            std_dev = daily_avg['avg_price'].std()
            
            prediction_results = []
            for i, (date, pred) in enumerate(zip(future_dates, predictions)):
                prediction_results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_price': round(float(pred), 2),
                    'lower_bound': round(float(pred - std_dev), 2),
                    'upper_bound': round(float(pred + std_dev), 2),
                    'confidence': 0.8 - (i * 0.05)  # Decreasing confidence over time
                })
            
            return {
                'route': route,
                'predictions': prediction_results,
                'model_performance': {
                    'mae': round(mae, 2),
                    'r2_score': round(r2, 2)
                },
                'historical_avg': round(float(y.mean()), 2),
                'trend': self._calculate_trend(daily_avg['avg_price'])
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return self._statistical_price_prediction(route, historical_data, days_ahead)
    
    def _statistical_price_prediction(self, route: str, historical_data: pd.DataFrame, days_ahead: int) -> Dict:
        """Simple statistical prediction as fallback"""
        df = historical_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        daily_avg = df.groupby('date')['avg_price'].mean().reset_index()
        
        # Simple moving average
        ma_7 = daily_avg['avg_price'].rolling(window=7).mean().iloc[-1]
        ma_30 = daily_avg['avg_price'].rolling(window=30).mean().iloc[-1]
        
        # Trend calculation
        trend = self._calculate_trend(daily_avg['avg_price'])
        trend_factor = 1.01 if trend == 'increasing' else 0.99 if trend == 'decreasing' else 1.0
        
        predictions = []
        base_price = ma_7 if not np.isnan(ma_7) else daily_avg['avg_price'].mean()
        
        for i in range(days_ahead):
            date = datetime.now() + timedelta(days=i+1)
            predicted_price = base_price * (trend_factor ** i)
            
            predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'predicted_price': round(float(predicted_price), 2),
                'method': 'statistical'
            })
        
        return {
            'route': route,
            'predictions': predictions,
            'historical_avg': round(float(daily_avg['avg_price'].mean()), 2),
            'trend': trend
        }
    
    def _calculate_trend(self, prices: pd.Series) -> str:
        """Calculate price trend"""
        if len(prices) < 2:
            return 'stable'
        
        x = np.arange(len(prices))
        y = prices.values
        
        # Linear regression
        slope, _ = np.polyfit(x, y, 1)
        
        if slope > 0.5:
            return 'increasing'
        elif slope < -0.5:
            return 'decreasing'
        else:
            return 'stable'
    
    def cluster_routes(self, n_clusters: int = None) -> Dict:
        """Cluster routes based on demand characteristics"""
        if n_clusters is None:
            n_clusters = self.config.CLUSTER_COUNT
            
        # Get route data
        routes = self.db.get_popular_routes(limit=100, days_back=30)
        
        if routes.empty:
            return {'clusters': {}, 'error': 'No route data available', 'n_clusters': 0, 'total_routes_analyzed': 0}
        
        # Prepare features for clustering
        features = []
        route_names = []
        
        for _, route in routes.iterrows():
            route_name = route['route']
            price_data = self.db.get_price_trends(route_name, days_back=30)
            
            if not price_data.empty:
                features.append([
                    route['flight_count'],
                    route['airline_count'],
                    price_data['avg_price'].mean(),
                    price_data['avg_price'].std(),  # Price volatility
                    route['on_time_rate'] if 'on_time_rate' in route else 0.8
                ])
                route_names.append(route_name)
        
        if len(features) < n_clusters:
            n_clusters = len(features)
        
        if not features:
            return {'clusters': [], 'error': 'Insufficient data for clustering'}
        
        # Standardize features
        X = np.array(features)
        X_scaled = self.demand_scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Organize results
        clusters = {}
        for i, (route, label) in enumerate(zip(route_names, cluster_labels)):
            if label not in clusters:
                clusters[label] = {
                    'routes': [],
                    'characteristics': {
                        'avg_flight_count': 0,
                        'avg_airline_count': 0,
                        'avg_price': 0,
                        'avg_volatility': 0,
                        'avg_on_time_rate': 0
                    }
                }
            
            clusters[label]['routes'].append({
                'route': route,
                'flight_count': features[i][0],
                'airline_count': features[i][1],
                'avg_price': round(features[i][2], 2),
                'price_volatility': round(features[i][3], 2),
                'on_time_rate': round(features[i][4], 2)
            })
        
        # Calculate cluster characteristics
        for label, cluster in clusters.items():
            routes_in_cluster = cluster['routes']
            n = len(routes_in_cluster)
            
            cluster['characteristics'] = {
                'avg_flight_count': round(sum(r['flight_count'] for r in routes_in_cluster) / n, 1),
                'avg_airline_count': round(sum(r['airline_count'] for r in routes_in_cluster) / n, 1),
                'avg_price': round(sum(r['avg_price'] for r in routes_in_cluster) / n, 2),
                'avg_volatility': round(sum(r['price_volatility'] for r in routes_in_cluster) / n, 2),
                'avg_on_time_rate': round(sum(r['on_time_rate'] for r in routes_in_cluster) / n, 2)
            }
            
            # Interpret cluster
            cluster['interpretation'] = self._interpret_cluster(cluster['characteristics'])
        
        return {
        'clusters': clusters,  # This should be a dict, not a list
        'n_clusters': n_clusters,
        'total_routes_analyzed': len(route_names)
    }

    def _interpret_cluster(self, characteristics: Dict) -> str:
        """Interpret cluster characteristics"""
        flight_count = characteristics['avg_flight_count']
        price = characteristics['avg_price']
        volatility = characteristics['avg_volatility']
        
        if flight_count > 50 and price > 300:
            return "High-demand premium routes"
        elif flight_count > 50 and price <= 300:
            return "High-volume budget routes"
        elif flight_count <= 50 and price > 300:
            return "Low-frequency premium routes"
        elif volatility > 50:
            return "Volatile market routes"
        else:
            return "Stable regional routes"
    
    def generate_ai_insights(self, data_summary: Dict) -> str:
        """Generate AI-powered insights using Gemini or Groq"""
        if not self.config.ENABLE_AI_INSIGHTS:
            return self._generate_rule_based_insights(data_summary)
        
        try:
            prompt = f"""
            As an aviation market analyst, analyze this airline booking data for Australian hostel operators:
            
            {json.dumps(data_summary, indent=2)}
            
            Provide insights on:
            1. Which routes show the highest demand and why they're important for hostels
            2. Seasonal patterns and their impact on hostel bookings
            3. Price trends and budget traveler opportunities
            4. Recommendations for hostel locations based on flight traffic
            5. Emerging travel patterns and opportunities
            
            Focus on actionable insights that would help hostel operators make strategic decisions.
            Keep the response concise and practical.
            """
            
            if self.ai_provider == 'gemini':
                return self._generate_gemini_insights(prompt)
            elif self.ai_provider == 'groq':
                return self._generate_groq_insights(prompt)
            else:
                return self._generate_rule_based_insights(data_summary)
                
        except Exception as e:
            logger.error(f"AI insights generation error: {e}")
            return self._generate_rule_based_insights(data_summary)
    
    def _generate_gemini_insights(self, prompt: str) -> str:
        """Generate insights using Google Gemini"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def _generate_groq_insights(self, prompt: str) -> str:
        """Generate insights using Groq"""
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert aviation and hospitality market analyst."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.config.GROQ_MODEL,
                temperature=0.7,
                max_tokens=1000
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
    
    def _generate_rule_based_insights(self, data_summary: Dict) -> str:
        """Generate insights without AI API"""
        insights = []
        
        # Analyze popular routes
        if 'popular_routes' in data_summary:
            top_routes = data_summary['popular_routes'][:5]
            insights.append("**Top Flight Routes:**")
            for route in top_routes:
                insights.append(f"- {route['route']}: {route['flight_count']} flights")
        
        # Analyze demand scores
        if 'demand_analysis' in data_summary:
            high_demand = [a for a in data_summary['demand_analysis'] 
                          if a.get('demand_score', 0) > 60]
            if high_demand:
                insights.append("\n**High Demand Locations:**")
                for location in high_demand[:3]:
                    insights.append(f"- {location.get('airport', location.get('route'))}: "
                                  f"Demand Score {location['demand_score']}")
        
        # Price insights
        if 'price_trends' in data_summary:
            insights.append("\n**Price Trends:**")
            increasing = [r for r, t in data_summary['price_trends'].items() 
                         if t.get('trend') == 'increasing']
            decreasing = [r for r, t in data_summary['price_trends'].items() 
                         if t.get('trend') == 'decreasing']
            
            if increasing:
                insights.append(f"- Rising prices on {len(increasing)} routes")
            if decreasing:
                insights.append(f"- Falling prices on {len(decreasing)} routes (opportunity for budget travelers)")
        
        # Recommendations
        insights.append("\n**Recommendations for Hostel Operators:**")
        insights.append("1. Focus on locations with high flight frequency for steady customer flow")
        insights.append("2. Monitor routes with decreasing prices to anticipate budget traveler influx")
        insights.append("3. Consider seasonal patterns when planning capacity and pricing")
        insights.append("4. Target airports with 'High' or 'Very High' demand scores")
        insights.append("5. Develop partnerships with budget airlines serving high-volume routes")
        
        return "\n".join(insights)
    
    def visualize_demand_heatmap(self) -> str:
        """Create a demand heatmap and return as base64 string"""
        try:
            # Get demand data
            demand_data = self.db.get_demand_analysis()
            
            if demand_data.empty:
                return None
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Prepare data for heatmap
            airports = demand_data['airport'].tolist()[:10]  # Top 10 airports
            metrics = ['total_arrivals', 'total_departures', 'avg_daily_flights']
            
            data_matrix = []
            for metric in metrics:
                if metric in demand_data.columns:
                    data_matrix.append(demand_data[metric].tolist()[:10])
            
            # Create heatmap
            sns.heatmap(data_matrix, 
                       xticklabels=airports,
                       yticklabels=metrics,
                       cmap='YlOrRd',
                       annot=True,
                       fmt='.0f')
            
            plt.title('Airport Demand Heatmap')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return None
    
    def get_comprehensive_analysis(self, 
                                 airports: List[str] = None,
                                 routes: List[str] = None,
                                 days_back: int = 30) -> Dict:
        """Get comprehensive analysis combining all analytics"""
        analysis = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_days': days_back,
            'demand_scores': {},
            'price_predictions': {},
            'route_clusters': {},
            'insights': '',
            'visualizations': {}
        }
        
        try:
            # Calculate demand scores
            if airports or routes:
                if airports:
                    for airport in airports:
                        analysis['demand_scores'][airport] = self.calculate_demand_score(
                            airport=airport, days_back=days_back
                        )
                if routes:
                    for route in routes:
                        analysis['demand_scores'][route] = self.calculate_demand_score(
                            route=route, days_back=days_back
                        )
            else:
                analysis['demand_scores'] = self.calculate_demand_score(days_back=days_back)
            
            # Get price predictions for top routes
            popular_routes = self.db.get_popular_routes(limit=5, days_back=days_back)
            for _, route in popular_routes.iterrows():
                route_name = route['route']
                analysis['price_predictions'][route_name] = self.predict_price_trends(
                    route_name, days_ahead=7
                )
            
            # Cluster routes
            analysis['route_clusters'] = self.cluster_routes()
            
            # Generate AI insights
            summary_data = {
                'demand_analysis': analysis['demand_scores'],
                'popular_routes': popular_routes.to_dict('records') if not popular_routes.empty else [],
                'price_trends': analysis['price_predictions']
            }
            analysis['insights'] = self.generate_ai_insights(summary_data)
            
            # Generate visualizations
            heatmap = self.visualize_demand_heatmap()
            if heatmap:
                analysis['visualizations']['demand_heatmap'] = heatmap
            
            # Add summary statistics
            analysis['summary'] = self._generate_summary_stats(analysis)
            
        except Exception as e:
            logger.error(f"Comprehensive analysis error: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _generate_summary_stats(self, analysis: Dict) -> Dict:
        """Generate summary statistics from analysis"""
        summary = {
            'total_routes_analyzed': 0,
            'high_demand_routes': [],
            'price_trend_summary': {
                'increasing': 0,
                'decreasing': 0,
                'stable': 0
            }
        }
        
        # Count high demand routes
        if 'demand_scores' in analysis:
            if 'routes' in analysis['demand_scores']:
                routes = analysis['demand_scores']['routes']
                summary['total_routes_analyzed'] = len(routes)
                summary['high_demand_routes'] = [
                    r['route'] for r in routes 
                    if r.get('demand_score', 0) > 60
                ][:5]
        
        # Summarize price trends
        if 'price_predictions' in analysis:
            for route, prediction in analysis['price_predictions'].items():
                if 'trend' in prediction:
                    trend = prediction['trend']
                    if trend in summary['price_trend_summary']:
                        summary['price_trend_summary'][trend] += 1
        
        return summary
    
    def export_analysis_report(self, analysis: Dict, format: str = 'json') -> bytes:
        """Export analysis report in various formats"""
        if format == 'json':
            return json.dumps(analysis, indent=2).encode('utf-8')
        
        elif format == 'csv':
            # Convert to CSV format
            rows = []
            
            # Add demand scores
            if 'demand_scores' in analysis and 'routes' in analysis['demand_scores']:
                for route in analysis['demand_scores']['routes']:
                    rows.append({
                        'type': 'demand_score',
                        'identifier': route['route'],
                        'score': route['demand_score'],
                        'interpretation': route['interpretation']
                    })
            
            df = pd.DataFrame(rows)
            return df.to_csv(index=False).encode('utf-8')
        
        else:
            raise ValueError(f"Unsupported format: {format}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize analytics
    analytics = FlightAnalytics()
    
    print("Testing Flight Analytics...")
    print("="*50)
    
    # Test demand scoring
    print("\n1. Testing Demand Scoring")
    print("-"*30)
    
    # Test route demand
    route_demand = analytics.calculate_demand_score(route='SYD-MEL', days_back=30)
    print(f"Route SYD-MEL demand: {route_demand}")
    
    # Test airport demand
    airport_demand = analytics.calculate_demand_score(airport='SYD', days_back=30)
    print(f"\nAirport SYD demand: {airport_demand}")
    
    # Test overall demand
    overall_demand = analytics.calculate_demand_score(days_back=7)
    print(f"\nTop 3 high-demand routes:")
    if 'summary' in overall_demand and 'top_demand_routes' in overall_demand['summary']:
        for route in overall_demand['summary']['top_demand_routes'][:3]:
            print(f"  - {route['route']}: {route['demand_score']}")
    
    # Test price predictions
    print("\n2. Testing Price Predictions")
    print("-"*30)
    
    price_pred = analytics.predict_price_trends('SYD-MEL', days_ahead=7, use_ml=True)
    print(f"Price predictions for SYD-MEL:")
    if 'predictions' in price_pred:
        for pred in price_pred['predictions'][:3]:
            print(f"  - {pred['date']}: ${pred['predicted_price']}")
        print(f"  - Trend: {price_pred.get('trend', 'unknown')}")
        if 'model_performance' in price_pred:
            print(f"  - Model R²: {price_pred['model_performance'].get('r2_score', 'N/A')}")
    
    # Test route clustering
    print("\n3. Testing Route Clustering")
    print("-"*30)
    
    clusters = analytics.cluster_routes(n_clusters=3)
    if 'clusters' in clusters:
        for cluster_id, cluster_data in clusters['clusters'].items():
            print(f"\nCluster {cluster_id}: {cluster_data['interpretation']}")
            print(f"  - Routes: {len(cluster_data['routes'])}")
            print(f"  - Avg price: ${cluster_data['characteristics']['avg_price']}")
            print(f"  - Avg flights: {cluster_data['characteristics']['avg_flight_count']}")
    
    # Test AI insights
    print("\n4. Testing Insights Generation")
    print("-"*30)
    
    sample_data = {
        'popular_routes': [
            {'route': 'SYD-MEL', 'flight_count': 120},
            {'route': 'SYD-BNE', 'flight_count': 80},
            {'route': 'MEL-BNE', 'flight_count': 60}
        ],
        'demand_analysis': [
            {'airport': 'SYD', 'demand_score': 85},
            {'airport': 'MEL', 'demand_score': 80},
            {'airport': 'BNE', 'demand_score': 70}
        ],
        'price_trends': {
            'SYD-MEL': {'trend': 'increasing', 'avg_price': 250},
            'SYD-BNE': {'trend': 'stable', 'avg_price': 280},
            'MEL-BNE': {'trend': 'decreasing', 'avg_price': 220}
        }
    }
    
    insights = analytics.generate_ai_insights(sample_data)
    print("Generated Insights:")
    print(insights[:500] + "..." if len(insights) > 500 else insights)
    
    print("\n" + "="*50)
    print("Analytics testing completed!")