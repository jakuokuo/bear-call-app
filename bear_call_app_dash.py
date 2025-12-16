# Dash version of the Bear Call Spread Analyzer
# -------------------------------------------------
# Single-file, deployable Dash app
# Run with: python bear_call_dash_app.py
# Then open http://127.0.0.1:8050

import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import os

# =========================
# Color Theme (matching Streamlit app)
# =========================
COLORS = {
    'background': '#1e1e2e',
    'card_bg': '#313244',
    'card_bg_end': '#45475a',
    'accent': '#6366f1',
    'text': '#cdd6f4',
    'text_muted': '#a6adc8',
    'positive': '#22c55e',
    'negative': '#ef4444',
    'neutral': '#f59e0b',
    'grid': '#313244',
    'histogram': '#6366f1',
    'line_spread': '#3b82f6',
    'line_long': '#ef4444',
    'line_short': '#22c55e',
}

# Month name mapping
MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

MONTH_ABBREV = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

# Premium spreads to calculate EV for
PREMIUM_SPREADS = [3, 4, 5, 6, 7]

# =========================
# Data loading
# =========================

def load_data():
    # Try different file paths for the full Mid-C data
    paths_to_try = [
        'Mid-C Peak Settles 2020-2025.csv',
        'data/Mid-C Peak Settles 2020-2025.csv',
        'S:/Jasmine - Copy/C/CAISO Intertie Correlations - Copy/Power Options/Mid-C Peak Settles 2020-2025.csv',
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Parse the Price Date column to extract month
            df['Price Date'] = pd.to_datetime(df['Price Date'], format='%m/%d/%Y')
            df['Month'] = df['Price Date'].dt.month
            return df
    
    raise FileNotFoundError("Mid-C Peak Settles 2020-2025.csv not found")

DATA = load_data()
YEARS = sorted(DATA['Year'].unique())
MONTHS = list(range(1, 13))  # 1-12

# =========================
# Option math
# =========================

def call_payoff(sT, strike, premium):
    return np.where(sT > strike, sT - strike, 0) - premium


def bear_call_payoff(price, k_short, k_long, p_short, p_long):
    long_call = max(price - k_long, 0) - p_long
    short_call = p_short - max(price - k_short, 0)
    return long_call + short_call


def black76_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (S * norm.cdf(d1) - K * norm.cdf(d2))


def black76_vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return S * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, S, K, T, r):
    sigma = 0.5
    for _ in range(50):
        model = black76_call(S, K, T, r, sigma)
        vega = black76_vega(S, K, T, r, sigma)
        if vega < 1e-8:
            break
        sigma += (price - model) / vega
        sigma = max(0.01, min(5.0, sigma))
    return sigma


def calculate_payout_with_premium(prices, k_short, k_long, custom_net_premium, mult):
    """Calculate total payout for given prices using a custom net premium spread."""
    if len(prices) == 0:
        return 0
    
    hours_to_days = 1 / 16
    spread_width = k_long - k_short
    custom_max_loss = spread_width - custom_net_premium
    
    hours_below_short = np.sum(prices < k_short)
    hours_above_long = np.sum(prices >= k_long)
    
    days_below_short = hours_below_short * hours_to_days
    days_above_long = hours_above_long * hours_to_days
    
    payout_below = days_below_short * mult * custom_net_premium
    payout_above = days_above_long * mult * (-custom_max_loss)
    
    # Calculate payout for prices between strikes
    prices_between = prices[(prices >= k_short) & (prices < k_long)]
    payout_between = 0
    for price in prices_between:
        intrinsic_loss = price - k_short
        payoff_per_unit = custom_net_premium - intrinsic_loss
        payout_between += payoff_per_unit * mult * hours_to_days
    
    return payout_below + payout_between + payout_above


# =========================
# Helper: Stat Box Component
# =========================

def stat_box(label, value, color_class='neutral'):
    color_map = {
        'positive': COLORS['positive'],
        'negative': COLORS['negative'],
        'neutral': COLORS['neutral'],
        'accent': COLORS['accent'],
    }
    value_color = color_map.get(color_class, COLORS['text'])
    
    return html.Div(
        style={
            'background': f"linear-gradient(135deg, {COLORS['card_bg']} 0%, {COLORS['card_bg_end']} 100%)",
            'borderLeft': f"3px solid {COLORS['accent']}",
            'padding': '20px',
            'borderRadius': '8px',
            'marginBottom': '10px',
        },
        children=[
            html.Div(label, style={'color': COLORS['text_muted'], 'fontSize': '14px', 'marginBottom': '4px'}),
            html.Div(value, style={'fontSize': '24px', 'fontWeight': 'bold', 'color': value_color}),
        ]
    )

# =========================
# App
# =========================

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Bear Call Spread Analyzer</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: ''' + COLORS['background'] + ''';
                color: ''' + COLORS['text'] + ''';
            }
            .container-fluid {
                background-color: ''' + COLORS['background'] + ''';
            }
            .form-control, .Select-control, .Select-menu-outer {
                background-color: ''' + COLORS['card_bg'] + ''' !important;
                border-color: ''' + COLORS['card_bg_end'] + ''' !important;
                color: ''' + COLORS['text'] + ''' !important;
            }
            .form-control:focus {
                border-color: ''' + COLORS['accent'] + ''' !important;
                box-shadow: 0 0 0 0.2rem rgba(99, 102, 241, 0.25) !important;
            }
            input[type="number"] {
                background-color: ''' + COLORS['card_bg'] + ''' !important;
                border: 1px solid ''' + COLORS['card_bg_end'] + ''' !important;
                color: ''' + COLORS['text'] + ''' !important;
                padding: 8px 12px;
                border-radius: 4px;
                width: 100%;
                margin-bottom: 10px;
            }
            input[type="number"]:focus {
                border-color: ''' + COLORS['accent'] + ''' !important;
                outline: none;
            }
            label {
                color: ''' + COLORS['text_muted'] + ''';
                margin-bottom: 4px;
                display: block;
            }
            h2, h4, h5 {
                color: ''' + COLORS['text'] + ''';
            }
            hr {
                border-color: ''' + COLORS['card_bg_end'] + ''';
            }
            .sidebar-panel {
                background: linear-gradient(135deg, ''' + COLORS['card_bg'] + ''' 0%, ''' + COLORS['card_bg_end'] + ''' 100%);
                padding: 20px;
                border-radius: 8px;
                border-left: 3px solid ''' + COLORS['accent'] + ''';
            }
            .btn-primary {
                background-color: ''' + COLORS['accent'] + ''' !important;
                border-color: ''' + COLORS['accent'] + ''' !important;
            }
            .btn-primary:hover {
                background-color: #5558e3 !important;
                border-color: #5558e3 !important;
            }
            .Select-value-label {
                color: ''' + COLORS['text'] + ''' !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container(fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '20px'}, children=[
    html.H2("📊 Bear Call Spread Analyzer – Mid-C Power", className="my-3", style={'color': COLORS['text']}),
    
    # Dynamic title that updates with month/year selection
    html.H4(id='chart-title', className="mb-3", style={'color': COLORS['text_muted']}),

    dbc.Row([
        dbc.Col([
            html.Div(className="sidebar-panel", children=[
                html.H5("📈 Option Parameters", style={'color': COLORS['text'], 'marginBottom': '20px'}),
                dbc.Label("Short Call Strike ($)"),
                dcc.Input(id='k_short', type='number', value=90, step=5),
                dbc.Label("Long Call Strike ($)"),
                dcc.Input(id='k_long', type='number', value=120, step=5),
                dbc.Label("Short Call Premium ($)"),
                dcc.Input(id='p_short', type='number', value=12.5, step=0.5),
                dbc.Label("Long Call Premium ($)"),
                dcc.Input(id='p_long', type='number', value=6.5, step=0.5),
                dbc.Label("Contract Multiplier"),
                dcc.Input(id='mult', type='number', value=400, step=100),
                html.Hr(),
                dbc.Label("Select Month"),
                dcc.Dropdown(
                    id='month', 
                    options=[{'label': MONTH_NAMES[m], 'value': m} for m in MONTHS], 
                    value=8,  # Default to August
                    style={'backgroundColor': COLORS['card_bg'], 'marginBottom': '10px'}
                ),
                dbc.Label("Select Year"),
                dcc.Dropdown(
                    id='year', 
                    options=[{'label': y, 'value': y} for y in YEARS], 
                    value=YEARS[-1],
                    style={'backgroundColor': COLORS['card_bg']}
                ),
            ])
        ], width=3),

        dbc.Col([
            dcc.Graph(id='payoff-graph'),
        ], width=9)
    ]),

    html.Hr(),

    # Stats row
    dbc.Row([
        dbc.Col(html.Div(id='stats-1'), width=3),
        dbc.Col(html.Div(id='stats-2'), width=3),
        dbc.Col(html.Div(id='stats-3'), width=3),
        dbc.Col(html.Div(id='stats-4'), width=3),
    ]),

    html.Hr(),

    # EV Section
    html.H4(id='ev-title', style={'color': COLORS['text']}),
    html.P(id='ev-subtitle', style={'color': COLORS['text_muted'], 'marginBottom': '15px'}),
    dbc.Row([
        dbc.Col(html.Div(id='ev-1'), width=True),
        dbc.Col(html.Div(id='ev-2'), width=True),
        dbc.Col(html.Div(id='ev-3'), width=True),
        dbc.Col(html.Div(id='ev-4'), width=True),
        dbc.Col(html.Div(id='ev-5'), width=True),
    ]),

    html.Hr(),

    html.H4("📈 Implied Volatility Calculator (Black-76)", style={'color': COLORS['text']}),
    dbc.Row([
        dbc.Col([
            html.Div(className="sidebar-panel", children=[
                html.Div("Market Inputs", style={'color': COLORS['text'], 'fontWeight': 'bold', 'marginBottom': '15px'}),
                dbc.Label("Option Market Price ($)"),
                dcc.Input(id='iv_price', type='number', value=5, step=0.25),
                dbc.Label("Futures Price ($)"),
                dcc.Input(id='iv_fut', type='number', value=72, step=1),
                dbc.Label("Strike Price ($)"),
                dcc.Input(id='iv_k', type='number', value=120, step=5),
            ])
        ], width=4),
        dbc.Col([
            html.Div(className="sidebar-panel", children=[
                html.Div("Time & Rate", style={'color': COLORS['text'], 'fontWeight': 'bold', 'marginBottom': '15px'}),
                dbc.Label("Days to Expiry"),
                dcc.Input(id='iv_days', type='number', value=30, step=1),
                dbc.Label("Risk-Free Rate (%)"),
                dcc.Input(id='iv_r', type='number', value=3.6, step=0.1),
                html.Br(),
                dbc.Button("Calculate Implied Volatility", id='iv_btn', color='primary', style={'marginTop': '10px', 'width': '100%'})
            ])
        ], width=4),
        dbc.Col(html.Div(id='iv_out'), width=4)
    ])
])

# =========================
# Callbacks
# =========================

@app.callback(
    Output('payoff-graph', 'figure'),
    Output('stats-1', 'children'),
    Output('stats-2', 'children'),
    Output('stats-3', 'children'),
    Output('stats-4', 'children'),
    Output('chart-title', 'children'),
    Output('ev-title', 'children'),
    Output('ev-subtitle', 'children'),
    Output('ev-1', 'children'),
    Output('ev-2', 'children'),
    Output('ev-3', 'children'),
    Output('ev-4', 'children'),
    Output('ev-5', 'children'),
    Input('k_short', 'value'),
    Input('k_long', 'value'),
    Input('p_short', 'value'),
    Input('p_long', 'value'),
    Input('mult', 'value'),
    Input('month', 'value'),
    Input('year', 'value'),
)
def update_graph(k_short, k_long, p_short, p_long, mult, month, year):
    # Filter data by both month and year
    filtered_data = DATA[(DATA['Year'] == year) & (DATA['Month'] == month)]
    prices = filtered_data['ICE MID-C'].dropna().values
    
    month_name = MONTH_NAMES[month]
    month_abbrev = MONTH_ABBREV[month]

    net_prem = p_short - p_long
    max_loss = (k_long - k_short) - net_prem
    breakeven = k_short + net_prem
    
    # Calculate EV for each premium spread (average across all years for this month)
    ev_values = []
    for premium_spread in PREMIUM_SPREADS:
        total_payout_all_years = 0
        years_with_data = 0
        for y in YEARS:
            year_month_prices = DATA[(DATA['Year'] == y) & (DATA['Month'] == month)]['ICE MID-C'].dropna().values
            if len(year_month_prices) > 0:
                total_payout_all_years += calculate_payout_with_premium(
                    year_month_prices, k_short, k_long, premium_spread, mult
                )
                years_with_data += 1
        
        if years_with_data > 0:
            avg_payout = total_payout_all_years / years_with_data
        else:
            avg_payout = 0
        ev_values.append(round(avg_payout, 2))
    
    # Create EV stat boxes
    ev_boxes = []
    for i, (premium_spread, ev_value) in enumerate(zip(PREMIUM_SPREADS, ev_values)):
        ev_color = 'positive' if ev_value >= 0 else 'negative'
        ev_boxes.append(stat_box(f"${premium_spread} Premium Spread", f"${ev_value:,.2f}", ev_color))

    # Handle case where no data exists for the selected month/year
    if len(prices) == 0:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['background'],
            height=500,
            annotations=[{
                'text': f'No data available for {month_name} {year}',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'showarrow': False,
                'font': {'size': 20, 'color': COLORS['text_muted']}
            }]
        )
        return (
            fig,
            stat_box(f"Days Below ${k_short}", "N/A", 'neutral'),
            stat_box(f"Days ${k_short}-${k_long}", "N/A", 'neutral'),
            stat_box(f"Days Above ${k_long}", "N/A", 'neutral'),
            stat_box("Total Payout", "N/A", 'neutral'),
            f"{month_name} {year} Settlement Distribution",
            f"📊 Sell ${k_short:.0f} / Buy ${k_long:.0f} Expected Value",
            f"Average payout for {month_name} across {YEARS[0]}-{YEARS[-1]}",
            *ev_boxes,
        )

    lo = min(prices.min(), k_short) - 10
    hi = max(prices.max(), k_long) + 10
    sT = np.arange(int(lo), int(hi) + 1)

    # Calculate individual payoffs
    long_call_payoff = call_payoff(sT, k_long, p_long)
    short_call_payoff = call_payoff(sT, k_short, p_short) * -1
    bear_call_payoff_curve = long_call_payoff + short_call_payoff

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=prices, 
            nbinsx=30, 
            opacity=0.3, 
            name=f"{month_abbrev} {year} Settles",
            marker_color=COLORS['histogram'],
            hovertemplate='Price: $%{x}<br>Count: %{y}<extra></extra>'
        ), 
        secondary_y=True
    )
    
    # Payoff curves
    fig.add_trace(
        go.Scatter(
            x=sT, 
            y=bear_call_payoff_curve, 
            name="Bear Call Spread",
            line=dict(color=COLORS['line_spread'], width=3)
        ), 
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=sT, 
            y=long_call_payoff, 
            name="Long Call",
            line=dict(color=COLORS['line_long'], width=2, dash='dash')
        ), 
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=sT, 
            y=short_call_payoff, 
            name="Short Call",
            line=dict(color=COLORS['line_short'], width=2, dash='dash')
        ), 
        secondary_y=False
    )
    
    # Reference lines
    fig.add_vline(x=breakeven, line_dash='dash', line_color=COLORS['neutral'], line_width=2,
                  annotation_text=f"Breakeven: ${breakeven:.2f}", annotation_position="top")
    fig.add_hline(y=0, line_color='white', line_width=1, opacity=0.5)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99, 
            bgcolor='rgba(30,30,46,0.8)'
        ),
        margin=dict(l=60, r=60, t=40, b=60)
    )
    fig.update_xaxes(title_text='ICE Mid-C Peak Settlement Price ($/MWh)', gridcolor=COLORS['grid'])
    fig.update_yaxes(title_text='Profit & Loss ($)', secondary_y=False, gridcolor=COLORS['grid'], range=[-50, 50])
    fig.update_yaxes(title_text='Settlement Frequency', secondary_y=True, showgrid=False)

    hours_to_days = 1 / 16
    days_below = np.sum(prices < k_short) * hours_to_days
    days_between = np.sum((prices >= k_short) & (prices < k_long)) * hours_to_days
    days_above = np.sum(prices >= k_long) * hours_to_days

    payout = (
        days_below * mult * net_prem
        + days_above * mult * (-max_loss)
    )
    
    payout_color = 'positive' if payout >= 0 else 'negative'

    return (
        fig,
        stat_box(f"Days Below ${k_short}", f"{days_below:.1f}", 'positive'),
        stat_box(f"Days ${k_short}-${k_long}", f"{days_between:.1f}", 'neutral'),
        stat_box(f"Days Above ${k_long}", f"{days_above:.1f}", 'negative'),
        stat_box("Total Payout", f"${payout:,.2f}", payout_color),
        f"{month_name} {year} Settlement Distribution",
        f"📊 Sell ${k_short:.0f} / Buy ${k_long:.0f} Expected Value",
        f"Average payout for {month_name} across {YEARS[0]}-{YEARS[-1]}",
        *ev_boxes,
    )


@app.callback(
    Output('iv_out', 'children'),
    Input('iv_btn', 'n_clicks'),
    Input('iv_price', 'value'),
    Input('iv_fut', 'value'),
    Input('iv_k', 'value'),
    Input('iv_days', 'value'),
    Input('iv_r', 'value'),
    Input('month', 'value'),
)
def update_iv(n, price, fut, k, days, r, month):
    if not n:
        return stat_box("Implied Volatility", "Click Calculate", 'accent')
    T = days / 365
    r_decimal = r / 100
    iv = implied_vol(price, fut, k, T, r_decimal)
    
    # Calculate 1σ and 2σ moves
    one_std_move = fut * iv * np.sqrt(T)
    two_std_move = 2 * one_std_move
    
    # Calculate price ranges
    one_std_low = fut - one_std_move
    one_std_high = fut + one_std_move
    two_std_low = fut - two_std_move
    two_std_high = fut + two_std_move
    
    # Calculate historical percentage of hours above 2σ upper bound for this month
    month_name = MONTH_NAMES[month]
    all_month_prices = DATA[DATA['Month'] == month]['ICE MID-C'].dropna().values
    
    if len(all_month_prices) > 0:
        hours_above_2std = np.sum(all_month_prices > two_std_high)
        total_hours = len(all_month_prices)
        pct_above_2std = (hours_above_2std / total_hours) * 100
        
        # Convert to days for display
        hours_to_days = 1 / 16
        days_above_2std = hours_above_2std * hours_to_days
    else:
        pct_above_2std = 0
        days_above_2std = 0
    
    return html.Div([
        stat_box("Implied Volatility (Annual)", f"{iv*100:.1f}%", 'accent'),
        stat_box("1σ Move (68%)", f"±${one_std_move:.2f}", 'neutral'),
        stat_box("1σ Price Range", f"${one_std_low:.2f} - ${one_std_high:.2f}", 'positive'),
        stat_box("2σ Move (95%)", f"±${two_std_move:.2f}", 'neutral'),
        stat_box("2σ Price Range", f"${two_std_low:.2f} - ${two_std_high:.2f}", 'negative'),
        stat_box(
            f"Historical % Above ${two_std_high:.2f}",
            f"{pct_above_2std:.1f}% ({days_above_2std:.1f} days)",
            'negative' if pct_above_2std > 5 else 'positive'
        ),
        html.Div(
            f"Based on {month_name} settlements {YEARS[0]}-{YEARS[-1]}",
            style={'color': COLORS['text_muted'], 'fontSize': '12px', 'marginTop': '10px'}
        ),
    ])


if __name__ == '__main__':
    app.run_server(debug=True)
