
# Australian Flight Market Analysis Platform - Documentation

## Executive Summary
This platform provides real-time airline booking market analysis specifically designed for hostel operators in Australia. By analyzing flight patterns, pricing trends, and demand metrics, it helps hostel businesses make data-driven decisions about location, capacity, and pricing strategies.

## Table of Contents
- Project Overview
- Technical Architecture
- Setup Instructions
- Usage Guide
- Key Features
- API Documentation
- Business Value

## Project Overview

### Problem Statement
Hostel operators in Australia need to understand airline booking trends to:

- Identify high-traffic routes for customer acquisition
- Predict demand patterns for capacity planning
- Monitor price trends to anticipate budget traveler influx
- Make strategic decisions about property locations

### Solution
A comprehensive web application that:

- Collects real-time and historical flight data (up to 7 days)
- Analyzes demand patterns using machine learning
- Provides AI-powered insights and recommendations
- Offers interactive visualizations and exportable reports

## Technical Architecture

### System Components

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Web Interface │     │  Data Sources   │     │   AI Services   │
│   (Streamlit)   │     │  - AviationStack│     │  - Gemini API   │
│    main.py      │────▶│  - OpenSky API  │     │  - Groq API     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Analytics Engine                          │
│                        (analytics.py)                            │
│  • Demand Scoring  • Price Predictions  • Route Clustering      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                  │
│  ┌──────────────┐        ┌──────────────┐                      │
│  │ Data Loader  │────────▶│   Database   │                      │
│  │enhanced_loader.py│     │ database.py  │                      │
│  └──────────────┘        └──────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

### Core Modules

- **main.py**: Web interface (Streamlit dashboard)
- **enhanced_loader.py**: Data collection and aggregation
- **database.py**: SQLite database backend
- **analytics.py**: Flight demand scoring, price prediction, clustering
- **config.py**: Central config and environment loader

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip
- 2GB+ disk space

### Installation

```bash
git clone https://github.com/your-repo/flight-analysis-project
cd flight-analysis-project
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file:

```env
AVIATIONSTACK_KEY=your_key
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
AI_PROVIDER=gemini
DEBUG=false
```

### Initialize the Database

```bash
python init_data.py
```

Or use:

```python
from enhanced_loader import initialize_database_with_history
initialize_database_with_history()
```

### Run the App

```bash
streamlit run main.py
```

Go to: [http://localhost:8501](http://localhost:8501)

## Usage Guide

### Sidebar Controls

- **Analysis Period**: 7/14/30/60/90 days
- **Airport Selection**: Australian airports
- **Route Selection**: Specific route filters
- **Feature Toggles**: ML/AI/clustering toggle

### Main Tabs

- 📊 Dashboard
- 💰 Price Analysis
- 🎯 Demand Analysis
- 🤖 AI Insights
- 📈 Reports

## Typical Workflows

### 1. Market Analysis
1. Select airports (e.g. SYD, MEL)
2. Set to 30 days
3. View Dashboard & AI Insights
4. Export summary

### 2. Price Trend Analysis
1. Choose route
2. View historical & predicted prices
3. Review confidence intervals

### 3. Route Opportunity
1. Demand Analysis + Clustering
2. Check stable high-volume routes
3. Export insights

## Key Features

### 1. Multi-Source Integration
- AviationStack
- OpenSky
- Ranges: 1h, 5h, 1d, 3d, 7d

### 2. Demand Scoring

```python
Demand Score = 0.4 * Flight Frequency + 0.2 * Price Volatility + 0.2 * Price Stability + 0.2 * Capacity
```

- 80-100: Very High
- 60-79: High
- 40-59: Medium
- 20-39: Low
- 0-19: Very Low

### 3. ML-Based Price Forecasting

- Model: Random Forest
- Output: 7-day forecast with R² score

### 4. Route Clustering

- K-means segments:
  - Premium routes
  - Budget routes
  - Volatile markets
  - Stable connections

### 5. AI-Powered Insights

- Gemini/Groq text generation
- Seasonal pattern detection
- Route improvement suggestions

### 6. Report Export

- JSON, CSV, Excel
- Full report or tab-based exports

## API Documentation

### Data Collection

```python
from enhanced_loader import get_live_flights_aviationstack

flights = get_live_flights_aviationstack(
    departure_iata="SYD",
    arrival_iata="MEL",
    hours_back=24
)
```

### Analytics

```python
from analytics import FlightAnalytics
analytics = FlightAnalytics()
analytics.calculate_demand_score("SYD-MEL", 30)
analytics.predict_price_trends("SYD-MEL", 7, use_ml=True)
analytics.cluster_routes(n_clusters=5)
```

### DB Access

```python
from database import FlightDatabase
db = FlightDatabase()
db.get_popular_routes(10, 7)
db.get_price_trends("SYD-MEL", 30)
db.get_demand_analysis(["SYD", "MEL"])
```

## Business Value

### For Hostel Operators

#### 📍 Strategic Location Planning
- High-traffic airport routes
- Seasonal travel patterns

#### 💵 Dynamic Pricing
- Adjust hostel rates by demand
- React to budget flight influx

#### 👥 Capacity Management
- Forecast busy arrival periods
- Optimize resources
