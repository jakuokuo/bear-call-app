import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Bear Call Spread Analyzer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e2e;
    }
    .stat-box {
        background: linear-gradient(135deg, #313244 0%, #45475a 100%);
        border-left: 3px solid #6366f1;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .stat-label {
        color: #a6adc8;
        font-size: 14px;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 24px;
        font-weight: bold;
    }
    .positive { color: #22c55e; }
    .negative { color: #ef4444; }
    .neutral { color: #f59e0b; }
    .premium { color: #6366f1; }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    # Use relative path for deployment (place CSV in same folder or 'data' subfolder)
    import os
    # Try relative path first, then absolute path for local development
    if os.path.exists('August Peak Settles 2020-2025.csv'):
        return pd.read_csv('August Peak Settles 2020-2025.csv')
    elif os.path.exists('data/August Peak Settles 2020-2025.csv'):
        return pd.read_csv('data/August Peak Settles 2020-2025.csv')
    else:
        return pd.read_csv('S:/Jasmine - Copy/C/CAISO Intertie Correlations - Copy/Power Options/August Peak Settles 2020-2025.csv')

data = load_data()
years = [2020, 2021, 2022, 2023, 2024, 2025]

# Sidebar for inputs
st.sidebar.header("📈 Option Parameters")

strike_price_short_call = st.sidebar.number_input(
    "Short Call Strike ($)", 
    min_value=0.0, 
    max_value=500.0, 
    value=90.0,
    step=5.0
)

strike_price_long_call = st.sidebar.number_input(
    "Long Call Strike ($)", 
    min_value=0.0, 
    max_value=500.0, 
    value=120.0,
    step=5.0
)

premium_short_call = st.sidebar.number_input(
    "Short Call Premium ($)", 
    min_value=0.0, 
    max_value=100.0, 
    value=12.50,
    step=0.50
)

premium_long_call = st.sidebar.number_input(
    "Long Call Premium ($)", 
    min_value=0.0, 
    max_value=100.0, 
    value=6.50,
    step=0.50
)

contract_multiplier = st.sidebar.number_input(
    "Contract Multiplier", 
    min_value=1, 
    max_value=1000, 
    value=400,
    step=100
)

st.sidebar.markdown("---")
selected_year = st.sidebar.selectbox("Select Year", years, index=5)

# Calculations
net_premium = premium_short_call - premium_long_call
breakeven_price = strike_price_short_call + net_premium
max_loss = (strike_price_long_call - strike_price_short_call) - net_premium

# Filter data for selected year
year_data = data[data['Year'] == selected_year]['ICE MID-C'].dropna().values

def call_payoff(sT, strike_price, premium):
    return np.where(sT > strike_price, sT - strike_price, 0) - premium

def bear_call_payoff_at_price(settle_price):
    long_call_pnl = max(settle_price - strike_price_long_call, 0) - premium_long_call
    short_call_pnl = premium_short_call - max(settle_price - strike_price_short_call, 0)
    return long_call_pnl + short_call_pnl

def calculate_stats(prices):
    if len(prices) == 0:
        return {
            'days_below_short_strike': 0,
            'days_between_strikes': 0,
            'days_above_long_strike': 0,
            'total_payout': 0
        }
    
    hours_to_days = 1 / 16
    
    hours_below_short = np.sum(prices < strike_price_short_call)
    hours_between = np.sum((prices >= strike_price_short_call) & (prices < strike_price_long_call))
    hours_above_long = np.sum(prices >= strike_price_long_call)
    
    days_below_short = hours_below_short * hours_to_days
    days_between = hours_between * hours_to_days
    days_above_long = hours_above_long * hours_to_days
    
    payout_below = days_below_short * contract_multiplier * net_premium
    payout_above = days_above_long * contract_multiplier * (-max_loss)
    
    prices_between = prices[(prices >= strike_price_short_call) & (prices < strike_price_long_call)]
    payout_between = 0
    for price in prices_between:
        payoff_per_unit = bear_call_payoff_at_price(price)
        payout_between += payoff_per_unit * contract_multiplier * hours_to_days
    
    total_payout = payout_below + payout_between + payout_above
    
    return {
        'days_below_short_strike': round(days_below_short, 1),
        'days_between_strikes': round(days_between, 1),
        'days_above_long_strike': round(days_above_long, 1),
        'total_payout': round(total_payout, 2)
    }

stats = calculate_stats(year_data)

# -------------------------------
# Calculate payout for a given year with custom net premium (for EV calculation)
# -------------------------------
def calculate_payout_with_premium(year_prices, custom_net_premium):
    """Calculate total payout for a year using a custom net premium spread."""
    if len(year_prices) == 0:
        return 0
    
    hours_to_days = 1 / 16
    spread_width = strike_price_long_call - strike_price_short_call
    custom_max_loss = spread_width - custom_net_premium
    
    hours_below_short = np.sum(year_prices < strike_price_short_call)
    hours_above_long = np.sum(year_prices >= strike_price_long_call)
    
    days_below_short = hours_below_short * hours_to_days
    days_above_long = hours_above_long * hours_to_days
    
    payout_below = days_below_short * contract_multiplier * custom_net_premium
    payout_above = days_above_long * contract_multiplier * (-custom_max_loss)
    
    prices_between = year_prices[(year_prices >= strike_price_short_call) & (year_prices < strike_price_long_call)]
    payout_between = 0
    for price in prices_between:
        intrinsic_loss = price - strike_price_short_call
        payoff_per_unit = custom_net_premium - intrinsic_loss
        payout_between += payoff_per_unit * contract_multiplier * hours_to_days
    
    return payout_below + payout_between + payout_above

# Calculate EV for different premium spreads (average across all years)
premium_spreads = [3, 4, 5, 6, 7]
ev_by_premium = {}

for premium_spread in premium_spreads:
    total_payout_all_years = 0
    for year in years:
        year_prices = data[data['Year'] == year]['ICE MID-C'].dropna().values
        total_payout_all_years += calculate_payout_with_premium(year_prices, premium_spread)
    avg_payout = total_payout_all_years / len(years)
    ev_by_premium[premium_spread] = round(avg_payout, 2)

# Create payoff curves
if len(year_data) > 0:
    lo, hi = year_data.min() - 10, year_data.max() + 10
else:
    lo, hi = 20, 150

sT = np.arange(int(lo), int(hi) + 1, 1)
long_call_payoff = call_payoff(sT, strike_price_long_call, premium_long_call)
short_call_payoff = call_payoff(sT, strike_price_short_call, premium_short_call) * -1
bear_call_payoff = long_call_payoff + short_call_payoff

# Main content
st.title("Bear Call Spread Analyzer")
st.markdown(f"### August {selected_year} Settlement Distribution")

# Create the plot
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Histogram
fig.add_trace(
    go.Histogram(
        x=year_data,
        name=f'Aug {selected_year} Settles',
        opacity=0.3,
        marker_color='#6366f1',
        nbinsx=30,
        hovertemplate='Price: $%{x}<br>Count: %{y}<extra></extra>'
    ),
    secondary_y=True
)

# Payoff curves
fig.add_trace(
    go.Scatter(x=sT, y=bear_call_payoff, mode='lines', name='Bear Call Spread',
               line=dict(color='#3b82f6', width=3)),
    secondary_y=False
)
fig.add_trace(
    go.Scatter(x=sT, y=long_call_payoff, mode='lines', name='Long Call',
               line=dict(color='#ef4444', width=2, dash='dash')),
    secondary_y=False
)
fig.add_trace(
    go.Scatter(x=sT, y=short_call_payoff, mode='lines', name='Short Call',
               line=dict(color='#22c55e', width=2, dash='dash')),
    secondary_y=False
)

# Reference lines
fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=1, opacity=0.5)
fig.add_vline(x=breakeven_price, line_dash="dash", line_color="#f59e0b", line_width=2, opacity=0.8,
              annotation_text=f"Breakeven: ${breakeven_price:.2f}", annotation_position="top")

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='#1e1e2e',
    plot_bgcolor='#1e1e2e',
    hovermode='x unified',
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(30,30,46,0.8)'),
    height=500,
    margin=dict(l=60, r=60, t=40, b=60)
)

fig.update_xaxes(title_text='ICE Mid-C Peak Settlement Price ($/MWh)', gridcolor='#313244')
fig.update_yaxes(title_text='Profit & Loss ($)', secondary_y=False, gridcolor='#313244', range=[-50, 50])
fig.update_yaxes(title_text='Settlement Frequency', secondary_y=True, showgrid=False)

# Display plot
st.plotly_chart(fig, use_container_width=True)

# Stats display
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-label">Days Below ${strike_price_short_call:.0f}</div>
        <div class="stat-value positive">{stats['days_below_short_strike']}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-label">Days ${strike_price_short_call:.0f}-${strike_price_long_call:.0f}</div>
        <div class="stat-value neutral">{stats['days_between_strikes']}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-label">Days Above ${strike_price_long_call:.0f}</div>
        <div class="stat-value negative">{stats['days_above_long_strike']}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    payout_class = "positive" if stats['total_payout'] >= 0 else "negative"
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-label">Total Payout</div>
        <div class="stat-value {payout_class}">${stats['total_payout']:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

# EV Section
st.markdown("---")
st.markdown(f"### 📊 Sell ${strike_price_short_call:.0f} / Buy ${strike_price_long_call:.0f} Expected Value")

ev_cols = st.columns(5)

for i, premium_spread in enumerate(premium_spreads):
    ev_value = ev_by_premium[premium_spread]
    ev_class = "positive" if ev_value >= 0 else "negative"
    with ev_cols[i]:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">${premium_spread} Premium Spread</div>
            <div class="stat-value {ev_class}">${ev_value:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

# Option details
st.markdown("---")
st.markdown("### 📋 Option Details")

detail_col1, detail_col2, detail_col3 = st.columns(3)

with detail_col1:
    st.metric("Short Call Strike", f"${strike_price_short_call:.2f}")
    st.metric("Short Call Premium", f"${premium_short_call:.2f}")

with detail_col2:
    st.metric("Long Call Strike", f"${strike_price_long_call:.2f}")
    st.metric("Long Call Premium", f"${premium_long_call:.2f}")

with detail_col3:
    st.metric("Breakeven Price", f"${breakeven_price:.2f}")
    st.metric("Net Premium", f"${net_premium:.2f}")
    st.metric("Max Loss", f"${max_loss:.2f}")
