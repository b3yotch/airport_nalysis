# main.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio
import json
from typing import Dict, List
import base64
import logging
# Import custom modules
from loader import FlightDataAggregator, get_live_flights_aviationstack
from analytics import FlightAnalytics
from database import FlightDatabase
from config import Config
from loader import get_flights_for_timerange


# Add this helper function near the top of main.py after imports
def safe_get_clusters(cluster_results):
    """Safely extract cluster data from results"""
    if not cluster_results:
        return {}
    
    clusters = cluster_results.get('clusters', {})
    
    # If clusters is accidentally a list, convert it to a dict
    if isinstance(clusters, list):
        return {i: cluster for i, cluster in enumerate(clusters)}
    
    return clusters
# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
    st.session_state.last_update = None
    st.session_state.selected_airports = ['SYD', 'MEL']
    st.session_state.selected_routes = []
    st.session_state.analysis_hours = 24 # Initialize with 24 hours

# Initialize components
@st.cache_resource
def init_components():
    """Initialize analytics and data components"""
    config = Config.get_config_dict()
    analytics = FlightAnalytics()
    aggregator = FlightDataAggregator(config)
    db = FlightDatabase(Config.DATABASE_PATH)
    return analytics, aggregator, db

analytics, aggregator, db = init_components()

# Sidebar
# Sidebar
with st.sidebar:
    st.header("🔍 Analysis Filters")
    
    # Add time range selector for real-time data
    col1, col2 = st.columns(2)
    
    with col1:
        # Historical analysis period
        analysis_period = st.selectbox(
            "Historical Analysis",
            options=[7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days",
            key="sidebar_analysis_period_1"
        )
    
    with col2:
        # Real-time data range
        realtime_range = st.selectbox(
            "Real-time Range",
            options=['1hour', '5hours', '24hours', '3days', '7days'],
            index=2,
            format_func=lambda x: {
                '1hour': 'Last 1 hour',
                '5hours': 'Last 5 hours', 
                '24hours': 'Last 24 hours',
                '3days': 'Last 3 days',
                '7days': 'Last 7 days'
            }[x],
            key="sidebar_realtime_range_1"
        )
    
    st.session_state.analysis_period = analysis_period
    st.session_state.realtime_range = realtime_range
    
    # Airport selection
    selected_airports = st.multiselect(
        "Select Airports",
        options=list(Config.AUSTRALIAN_AIRPORTS.keys()),
        default=st.session_state.selected_airports,
        help="Select airports to analyze"
    )
    st.session_state.selected_airports = selected_airports
    
    # Route selection
    popular_routes = db.get_popular_routes(limit=20, days_back=analysis_period)  # Fixed here too
    if not popular_routes.empty:
        route_options = popular_routes['route'].tolist()
        selected_routes = st.multiselect(
            "Select Routes",
            options=route_options,
            default=st.session_state.selected_routes[:5] if st.session_state.selected_routes else route_options[:3],
            help="Select specific routes to analyze"
        )
        st.session_state.selected_routes = selected_routes
    
    
    
    # Refresh data button
    col1, col2 = st.columns(2)
    with col1:
        # Historical analysis period
        analysis_period = st.selectbox(
            "Historical Analysis",
            options=[7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days",
            key="sidebar_analysis_period_2"
        )
    
    with col2:
        # Real-time data range
        realtime_range = st.selectbox(
            "Real-time Range",
            options=['1hour', '5hours', '24hours', '3days', '7days'],
            index=2,
            format_func=lambda x: {
                '1hour': 'Last 1 hour',
                '5hours': 'Last 5 hours', 
                '24hours': 'Last 24 hours',
                '3days': 'Last 3 days',
                '7days': 'Last 7 days'
            }[x],
            key="sidebar_realtime_range_2"
        )
    
    st.session_state.analysis_period = analysis_period
    st.session_state.realtime_range = realtime_range
    # Feature toggles
    st.header("⚙️ Features")
    show_predictions = st.checkbox("Show Price Predictions", value=Config.ENABLE_ML_PREDICTIONS)
    show_ai_insights = st.checkbox("Show AI Insights", value=Config.ENABLE_AI_INSIGHTS)
    show_clustering = st.checkbox("Show Route Clustering", value=True)

# Main content
st.title(f"✈️ {Config.APP_TITLE}")
st.markdown(Config.APP_SUBTITLE)

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "💰 Price Analysis", "🎯 Demand Analysis", "🤖 AI Insights", "📈 Reports"])

# Tab 1: Dashboard
with tab1:
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Get current flight data
    @st.cache_data(ttl=300)
    def get_flight_stats():
        stats = {}
        
        # Get flights based on selected time range
        time_range = st.session_state.get('realtime_range', '24hours')
        
        # Fetch real-time data with time range
        from loader import get_flights_for_timerange  # Fixed import
        
        flight_data = get_flights_for_timerange(
            st.session_state.selected_airports, 
            time_range
        )
        
        if flight_data and 'flights' in flight_data:
            stats['total_flights'] = len(flight_data['flights'])
            stats['time_range'] = time_range
            stats['data_sources'] = list(set(f.get('source', 'Unknown') 
                                        for f in flight_data['flights']))
        else:
            stats['total_flights'] = 0
            stats['time_range'] = time_range
            stats['data_sources'] = []
        
        # Get route statistics from database
        flights_df = db.get_popular_routes(limit=100, days_back=analysis_period)
        if not flights_df.empty:
            stats['total_routes'] = len(flights_df)
            stats['avg_flights_per_route'] = round(
                flights_df['flight_count'].sum() / len(flights_df), 1
            )
        else:
            stats['total_routes'] = 0
            stats['avg_flights_per_route'] = 0
        
        # Get demand scores
        demand_analysis = analytics.calculate_demand_score(days_back=analysis_period)
        if 'summary' in demand_analysis and 'avg_route_demand' in demand_analysis['summary']:
            stats['avg_demand_score'] = demand_analysis['summary']['avg_route_demand']
        else:
            stats['avg_demand_score'] = 0
        
        return stats
    
    stats = get_flight_stats()
    
    with col1:
        st.metric(
            label="Total Flights",
            value=f"{stats['total_flights']:,}",
            help=f"Flights in {stats.get('time_range', 'selected period')}"  # Updated help text
        )
    
    with col2:
        st.metric(
            label="Active Routes",
            value=stats['total_routes'],
            help="Number of unique routes"
        )
    
    with col3:
        st.metric(
            label="Avg Flights/Route",
            value=stats['avg_flights_per_route'],
            help="Average flights per route"
        )
    
    with col4:
        st.metric(
            label="Avg Demand Score",
            value=f"{stats['avg_demand_score']:.1f}",
            help="Average demand score (0-100)"
        )
    
    # Popular routes visualization
    st.subheader("🛫 Most Popular Routes")
    
    routes_df = db.get_popular_routes(limit=10, days_back=analysis_period)  # Fixed: was analysis_hours
    if not routes_df.empty:
        fig = px.bar(
            routes_df,
            x='route',
            y='flight_count',
            color='airline_count',
            title="Top 10 Routes by Flight Frequency",
            labels={'flight_count': 'Number of Flights', 'airline_count': 'Airlines'},
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No route data available for the selected period")
    
    # Airport demand heatmap
    st.subheader("🔥 Airport Demand Heatmap")
    
    demand_df = db.get_demand_analysis(selected_airports if selected_airports else None)
    if not demand_df.empty:
        # Create heatmap data
        metrics = ['total_arrivals', 'total_departures', 'avg_daily_flights']
        heatmap_data = []
        
        for _, row in demand_df.iterrows():
            for metric in metrics:
                if metric in row:
                    heatmap_data.append({
                        'Airport': row['airport'],
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': row[metric]
                    })
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            fig = px.density_heatmap(
                heatmap_df,
                x='Airport',
                y='Metric',
                z='Value',
                color_continuous_scale='YlOrRd',
                title="Airport Activity Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No demand data available")
    
    # Real-time flight data section
    st.subheader(f"✈️ Real-time Flights ({st.session_state.get('realtime_range', '24hours')})")
    
    # Import at module level or inside function
    from loader import get_flights_for_timerange  # Added import
    
    # Get real-time flight data
    realtime_data = get_flights_for_timerange(
        selected_airports if selected_airports else ['SYD'],
        st.session_state.get('realtime_range', '24hours')  # Added .get() for safety
    )
    
    if realtime_data and 'flights' in realtime_data and realtime_data['flights']:
        # Create DataFrame
        flights_df = pd.DataFrame(realtime_data['flights'])
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Flights", len(flights_df))
        with col2:
            unique_airlines = flights_df['airline'].nunique() if 'airline' in flights_df else 0
            st.metric("Airlines Operating", unique_airlines)
        with col3:
            sources = list(set(f.get('source', 'Unknown') for f in realtime_data['flights']))
            st.metric("Data Sources", len(sources))
        
        # Show recent flights
        with st.expander("View Recent Flights"):
            display_cols = ['airline', 'flight_number', 'departure_airport', 
                           'arrival_airport', 'departure_time', 'status']
            available_cols = [col for col in display_cols if col in flights_df.columns]
            
            if available_cols:
                st.dataframe(
                    flights_df[available_cols].head(20),
                    use_container_width=True
                )
            else:
                st.write("No flight details available")
    else:
        st.info(f"No flight data available for {st.session_state.get('realtime_range', '24hours')}")

# Tab 2: Price Analysis
# Tab 2: Price Analysis
with tab2:
    st.header("💰 Price Analysis & Predictions")
    
    # Ensure selected_routes is always defined
    if 'selected_routes' not in st.session_state:
        selected_routes = []
    else:
        selected_routes = st.session_state.selected_routes

    if selected_routes:
        route_selector = st.selectbox(
            "Select Route for Price Analysis",
            options=selected_routes,
            key="price_route_selector"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Price trends chart
            price_data = db.get_price_trends(route_selector, days_back=analysis_period)
            
            if not price_data.empty:
                # Group by date and airline
                price_data['date'] = pd.to_datetime(price_data['date'])
                
                fig = go.Figure()
                
                # Add traces for each airline
                for airline in price_data['airline'].unique():
                    airline_data = price_data[price_data['airline'] == airline]
                    fig.add_trace(go.Scatter(
                        x=airline_data['date'],
                        y=airline_data['avg_price'],
                        mode='lines+markers',
                        name=airline,
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
                
                fig.update_layout(
                    title=f"Price Trends for {route_selector}",
                    xaxis_title="Date",
                    yaxis_title="Average Price (AUD)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical price data available for this route")
        
        with col2:
            # Price statistics
            if not price_data.empty:
                st.subheader("📊 Price Statistics")
                
                current_avg = price_data['avg_price'].mean()
                min_price = price_data['min_price'].min()
                max_price = price_data['max_price'].max()
                volatility = price_data['avg_price'].std()
                
                st.metric("Average Price", f"${current_avg:.2f}")
                st.metric("Price Range", f"${min_price:.2f} - ${max_price:.2f}")
                st.metric("Volatility", f"${volatility:.2f}")
        
        # Price predictions
        if show_predictions:
            st.subheader("🔮 Price Predictions")
            
            with st.spinner("Generating price predictions..."):
                predictions = analytics.predict_price_trends(
                    route_selector,
                    days_ahead=7,
                    use_ml=Config.ENABLE_ML_PREDICTIONS
                )
            
            if 'predictions' in predictions and predictions['predictions']:
                # Create prediction chart
                pred_df = pd.DataFrame(predictions['predictions'])
                pred_df['date'] = pd.to_datetime(pred_df['date'])
                
                fig = go.Figure()
                
                # Add historical data
                if not price_data.empty:
                    recent_data = price_data.groupby('date')['avg_price'].mean().reset_index()
                    fig.add_trace(go.Scatter(
                        x=recent_data['date'],
                        y=recent_data['avg_price'],
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                
                # Add predictions
                fig.add_trace(go.Scatter(
                    x=pred_df['date'],
                    y=pred_df['predicted_price'],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=8)
                ))
                
                # Add confidence intervals if available
                if 'lower_bound' in pred_df.columns:
                    fig.add_trace(go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['upper_bound'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(255,0,0,0)',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['lower_bound'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(255,0,0,0)',
                        name='Confidence Interval',
                        fillcolor='rgba(255,0,0,0.2)'
                    ))
                
                fig.update_layout(
                    title=f"Price Predictions for {route_selector}",
                    xaxis_title="Date",
                    yaxis_title="Price (AUD)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show prediction metrics
                if 'model_performance' in predictions:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Trend", predictions.get('trend', 'Unknown').title())
                    with col2:
                        if 'r2_score' in predictions['model_performance']:
                            st.metric("Model R²", f"{predictions['model_performance']['r2_score']:.3f}")
                    with col3:
                        if 'mae' in predictions['model_performance']:
                            st.metric("MAE", f"${predictions['model_performance']['mae']:.2f}")
            else:
                st.warning("Unable to generate predictions for this route")
    else:
        st.info("Please select routes in the sidebar to view price analysis")

# Tab 3: Demand Analysis
with tab3:
    st.header("🎯 Demand Analysis")
    
    # Demand scores overview
    demand_results = analytics.calculate_demand_score(days_back=analysis_period)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✈️ Airport Demand Scores")
        
        if 'airports' in demand_results:
            airport_demands = pd.DataFrame(demand_results['airports'])
            
            if not airport_demands.empty:
                # Sort by demand score
                airport_demands = airport_demands.sort_values('demand_score', ascending=False)
                
                fig = px.bar(
                    airport_demands.head(10),
                    x='airport',
                    y='demand_score',
                    color='demand_score',
                    color_continuous_scale='RdYlGn',
                    title="Top 10 Airports by Demand Score",
                    labels={'demand_score': 'Demand Score'}
                )
                
                fig.update_traces(text=airport_demands['interpretation'].head(10), textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed view
                with st.expander("View Detailed Airport Metrics"):
                    st.dataframe(
                        airport_demands[['airport', 'demand_score', 'interpretation', 'factors']],
                        use_container_width=True
                    )
    
    with col2:
        st.subheader("🛫 Route Demand Scores")
        
        if 'routes' in demand_results:
            route_demands = pd.DataFrame(demand_results['routes'])
            
            if not route_demands.empty:
                # Sort by demand score
                route_demands = route_demands.sort_values('demand_score', ascending=False)
                
                fig = px.bar(
            route_demands.head(10),
            x='route',
            y='demand_score',
            color='demand_score',
            color_continuous_scale='RdYlGn',
            title="Top 10 Routes by Demand Score",
            labels={'demand_score': 'Demand Score'},
            hover_data=['demand_score']  # Add any available columns
        )
                
                if 'interpretation' in route_demands.columns:
                    fig.update_traces(
                        customdata=route_demands['interpretation'].head(10),
                        hovertemplate='<b>%{x}</b><br>Demand Score: %{y}<br>%{customdata}<extra></extra>'
                    )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    # Route clustering
    cluster_results = {}
    # Fix for Tab 3: Demand Analysis - Route clustering section
# Replace the clustering section in Tab 3 with this:

    # Route clustering
    if show_clustering:
        st.subheader("🎯 Route Clustering Analysis")
        
        with st.spinner("Performing cluster analysis..."):
            cluster_results = analytics.cluster_routes(n_clusters=5)
        
        if cluster_results and 'clusters' in cluster_results:
            # Display cluster summary
            cluster_summary = []
            
            # Check if clusters is a dictionary
            if isinstance(cluster_results['clusters'], dict):
                clusters = safe_get_clusters(cluster_results)
                for cluster_id, cluster_data in clusters.items():
                    cluster_summary.append({
                        'Cluster': f"Cluster {cluster_id}",
                        'Type': cluster_data['interpretation'],
                        'Routes': len(cluster_data['routes']),
                        'Avg Flight Count': cluster_data['characteristics']['avg_flight_count'],
                        'Avg Price': f"${cluster_data['characteristics']['avg_price']:.2f}",
                        'Avg Airlines': cluster_data['characteristics']['avg_airline_count']
                    })
            else:
                st.warning("Cluster data format is incorrect")
                st.stop()
            
            if cluster_summary:
                cluster_df = pd.DataFrame(cluster_summary)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.dataframe(cluster_df, use_container_width=True)
                
                with col2:
                    # Visualize clusters
                    fig = px.scatter(
                        cluster_df,
                        x='Avg Flight Count',
                        y='Avg Airlines',
                        size='Routes',
                        color='Type',
                        title="Route Clusters Visualization",
                        hover_data=['Cluster', 'Avg Price']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed cluster view
                with st.expander("View Detailed Cluster Information"):
                    cluster_options = list(cluster_results['clusters'].keys()) if isinstance(cluster_results['clusters'], dict) else []
                    if cluster_options:
                        selected_cluster = st.selectbox(
                            "Select Cluster",
                            options=cluster_options,
                            format_func=lambda x: f"Cluster {x}: {cluster_results['clusters'][x]['interpretation']}"
                        )
                        
                        if selected_cluster is not None:
                            cluster_routes = pd.DataFrame(cluster_results['clusters'][selected_cluster]['routes'])
                            st.dataframe(cluster_routes, use_container_width=True)
                    else:
                        st.info("No cluster details available")
        else:
            st.info("No clustering data available. This might be due to insufficient route data.")

# Tab 4: AI Insights
with tab4:
    st.header("🤖 AI-Powered Insights")
    
    if show_ai_insights:
        # Prepare data for AI analysis
        insight_data = {
            'popular_routes': db.get_popular_routes(limit=10, days_back=analysis_period).to_dict('records'),
            'demand_analysis': demand_results.get('airports', [])[:10] if 'airports' in demand_results else [],
            'price_trends': {}
        }
        
        # Add price trends for top routes
        for route in selected_routes[:5]:
            price_pred = analytics.predict_price_trends(route, hours_ahead=7)
            if 'trend' in price_pred:
                insight_data['price_trends'][route] = {
                    'trend': price_pred['trend'],
                    'avg_price': price_pred.get('historical_avg', 0)
                }
        
        with st.spinner("Generating AI insights..."):
            ai_insights = analytics.generate_ai_insights(insight_data)
        
        # Display insights in a nice format
        st.markdown("### 📊 Market Analysis")
        
        # Parse and display insights
        if ai_insights:
            # Create columns for better layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(ai_insights)
            
            with col2:
                # Key takeaways box
                st.info(
                    "**🔑 Key Takeaways**\n\n"
                    "• Monitor high-demand routes for hostel placement\n"
                    "• Track price trends for budget traveler patterns\n"
                    "• Consider seasonal variations in planning\n"
                    "• Focus on airports with consistent traffic"
                )
        
        # Additional analysis cards
        st.markdown("### 📈 Detailed Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏆 Best Performing Routes**")
            if insight_data['popular_routes']:
                for i, route in enumerate(insight_data['popular_routes'][:3]):
                    st.write(f"{i+1}. {route['route']} ({route['flight_count']} flights)")
        
        with col2:
            st.markdown("**💰 Price Opportunities**")
            decreasing_prices = [r for r, d in insight_data['price_trends'].items() 
                               if d['trend'] == 'decreasing']
            if decreasing_prices:
                for route in decreasing_prices[:3]:
                    st.write(f"• {route} ⬇️")
            else:
                st.write("No significant price drops detected")
        
        with col3:
            st.markdown("**🔥 High Demand Airports**")
            high_demand_airports = [a for a in insight_data['demand_analysis'] 
                                  if a.get('demand_score', 0) > 70]
            for airport in high_demand_airports[:3]:
                st.write(f"• {airport['airport']} ({airport['demand_score']:.0f})")
    else:
        st.info("Enable AI Insights in the sidebar to view this section")

# Tab 5: Reports
# Tab 5: Reports
with tab5:
    st.header("📈 Export Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Generate Comprehensive Report")
        
        report_type = st.selectbox(
            "Select Report Type",
            options=["Full Analysis", "Demand Report", "Price Report", "Executive Summary"]
        )
        
        export_format = st.selectbox(
            "Export Format",
            options=["JSON", "CSV", "Excel"] if Config.ENABLE_EXPORT else ["JSON"],
            disabled=not Config.ENABLE_EXPORT
        )
        
        include_visualizations = st.checkbox("Include Visualizations", value=True)
        include_ai_insights = st.checkbox("Include AI Insights", value=show_ai_insights)
        
        if st.button("Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                # Get comprehensive analysis
                if report_type == "Full Analysis":
                    report_data = analytics.get_comprehensive_analysis(
                        airports=selected_airports,
                        routes=selected_routes,
                        days_back=analysis_period
                    )
                
                elif report_type == "Demand Report":
                    report_data = {
                        'report_type': 'Demand Analysis',
                        'generated_at': datetime.now().isoformat(),
                        'period_hours': analysis_period,
                        'demand_scores': analytics.calculate_demand_score(days_back=analysis_period),
                        'clusters': analytics.cluster_routes() if show_clustering else None
                    }
                
                elif report_type == "Price Report":
                    report_data = {
                        'report_type': 'Price Analysis',
                        'generated_at': datetime.now().isoformat(),
                        'period_hours': analysis_period,
                        'routes': {}
                    }
                    
                    for route in selected_routes:
                        report_data['routes'][route] = {
                            'historical_prices': db.get_price_trends(route, analysis_period).to_dict('records'),
                            'predictions': analytics.predict_price_trends(route, hours_ahead=7)
                        }
                
                else:  # Executive Summary
                    report_data = {
                        'report_type': 'Executive Summary',
                        'generated_at': datetime.now().isoformat(),
                        'period_hours': analysis_period,
                        'key_metrics': stats,
                        'top_routes': db.get_popular_routes(limit=5, days_back=analysis_period).to_dict('records'),
                        'summary': {}
                    }
                    
                    # Add summary insights
                    demand_analysis = analytics.calculate_demand_score(days_back=analysis_period)
                    if 'summary' in demand_analysis:
                        report_data['summary'] = demand_analysis['summary']
                
                # Add AI insights if requested
                if include_ai_insights and report_data:
                    report_data['ai_insights'] = ai_insights if 'ai_insights' in locals() else analytics.generate_ai_insights(report_data)
                
                # Store in session state
                st.session_state.generated_report = report_data
                st.success("Report generated successfully!")
    
    with col2:
        st.subheader("📥 Download Report")
        
        if 'generated_report' in st.session_state:
            report_data = st.session_state.generated_report
            
            # Display report preview
            with st.expander("Preview Report"):
                st.json(report_data, expanded=False)
            
            # Export functionality
            if export_format == "JSON":
                report_json = json.dumps(report_data, indent=2)
                st.download_button(
                    label="Download JSON Report",
                    data=report_json,
                    file_name=f"flight_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            elif export_format == "CSV":
                # Convert to CSV format
                try:
                    # Flatten the data for CSV
                    csv_data = []
                    
                    if 'demand_scores' in report_data and 'routes' in report_data['demand_scores']:
                        for route in report_data['demand_scores']['routes']:
                            csv_data.append({
                                'Type': 'Route Demand',
                                'Identifier': route['route'],
                                'Score': route['demand_score'],
                                'Interpretation': route['interpretation']
                            })
                    
                    if 'routes' in report_data:  # Price report
                        for route, data in report_data['routes'].items():
                            if 'predictions' in data and 'predictions' in data['predictions']:
                                for pred in data['predictions']['predictions']:
                                    csv_data.append({
                                        'Type': 'Price Prediction',
                                        'Identifier': route,
                                        'Date': pred['date'],
                                        'Predicted Price': pred['predicted_price']
                                    })
                    
                    if csv_data:
                        df = pd.DataFrame(csv_data)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="Download CSV Report",
                            data=csv,
                            file_name=f"flight_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No data available for CSV export")
                        
                except Exception as e:
                    st.error(f"Error generating CSV: {str(e)}")
            
            elif export_format == "Excel":
                # Convert to Excel format
                try:
                    import io
                    output = io.BytesIO()
                    
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Summary sheet
                        summary_data = {
                            'Metric': ['Report Type', 'Generated At', 'Analysis Period'],
                            'Value': [
                                report_data.get('report_type', 'Unknown'),
                                report_data.get('generated_at', ''),
                                f"{report_data.get('period_hours', 0)} hours"
                            ]
                        }
                        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                        
                        # Add other sheets based on report type
                        if 'demand_scores' in report_data and 'routes' in report_data['demand_scores']:
                            routes_df = pd.DataFrame(report_data['demand_scores']['routes'])
                            routes_df.to_excel(writer, sheet_name='Route Demand', index=False)
                        
                        if 'demand_scores' in report_data and 'airports' in report_data['demand_scores']:
                            airports_df = pd.DataFrame(report_data['demand_scores']['airports'])
                            airports_df.to_excel(writer, sheet_name='Airport Demand', index=False)
                    
                    output.seek(0)
                    
                    st.download_button(
                        label="Download Excel Report",
                        data=output,
                        file_name=f"flight_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating Excel: {str(e)}")
        else:
            st.info("Generate a report first to download")
    
    # Report history
    st.subheader("📚 Recent Reports")
    
    # This would typically load from a database
    report_history = [
        {
            'date': (datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M'),
            'type': ['Full Analysis', 'Demand Report', 'Price Report'][i % 3],
            'status': 'Completed'
        }
        for i in range(5)
    ]
    
    history_df = pd.DataFrame(report_history)
    st.dataframe(history_df, use_container_width=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Data Sources**")
    st.caption("• AviationStack API")
    st.caption("• Simulated pricing data")

with col2:
    st.markdown("**AI Provider**")
    st.caption(f"• {Config.AI_PROVIDER.title()}")
    if Config.AI_PROVIDER == 'gemini':
        st.caption(f"• Model: {Config.GEMINI_MODEL}")
    else:
        st.caption(f"• Model: {Config.GROQ_MODEL}")

with col3:
    st.markdown("**Status**")
    if st.session_state.last_update:
        time_diff = datetime.now() - st.session_state.last_update
        if time_diff.seconds < 300:
            st.caption("🟢 Data is fresh")
        else:
            st.caption("🟡 Consider refreshing")
    else:
        st.caption("🔴 No data loaded")

# Background data collection (optional)
if Config.ENABLE_REAL_TIME_DATA and 'data_collection_started' not in st.session_state:
    @st.cache_resource
    def start_background_collection():
        """Start background data collection"""
        # This would typically run in a separate process
        # For now, we'll just mark it as started
        return True
    
    st.session_state.data_collection_started = start_background_collection()

# Error handling
if 'error' in st.session_state:
    st.error(f"An error occurred: {st.session_state.error}")
    if st.button("Clear Error"):
        del st.session_state.error

# Debug mode
if Config.DEBUG:
    with st.expander("🐛 Debug Information"):
        st.write("Session State:", st.session_state)
        st.write("Config:", Config.get_config_dict())
        
        # Validate configuration
        warnings = Config.validate_config()
        if warnings:
            st.warning("Configuration warnings:")
            for warning in warnings:
                st.write(f"• {warning}")

# Auto-refresh functionality (optional)
if st.checkbox("Enable auto-refresh", value=False, key="auto_refresh"):
    refresh_interval = st.number_input("Refresh interval (minutes)", min_value=5, max_value=60, value=30)
    
    # This would typically use JavaScript for true auto-refresh
    st.info(f"Page will refresh every {refresh_interval} minutes")
    
    # Add JavaScript for auto-refresh
    st.markdown(
        f"""
        <script>
        setTimeout(function() {{
            window.location.reload();
        }}, {refresh_interval * 60 * 1000});
        </script>
        """,
        unsafe_allow_html=True
    )

# Help section
with st.sidebar:
    st.markdown("---")
    with st.expander("ℹ️ Help & Documentation"):
        st.markdown("""
        **Using the Dashboard:**
        
        1. **Select Airports/Routes**: Use the filters to choose specific locations
        2. **Analyze Data**: Navigate through tabs to view different analyses
        3. **Generate Reports**: Create and export comprehensive reports
        4. **AI Insights**: Enable AI features for advanced analysis
        
        **Key Features:**
        - Real-time flight data analysis
        - Price trend predictions
        - Demand scoring algorithm
        - Route clustering
        - AI-powered insights
        - Export capabilities
        
        **For Hostel Operators:**
        - Focus on high-demand routes for customer flow
        - Monitor price trends for budget travelers
        - Use clustering to identify market segments
        - Export reports for strategic planning
        """)

# Main execution
if __name__ == "__main__":
    # Log startup
    logger = logging.getLogger(__name__)
    logger.info(f"Streamlit app started at {datetime.now()}")
    
    # Validate environment
    if not Config.AVIATIONSTACK_KEY:
        st.error("⚠️ AviationStack API key not configured. Please check your configuration.")
    
    # Show welcome message on first load
    if 'welcomed' not in st.session_state:
        st.balloons()
        st.session_state.welcomed = True