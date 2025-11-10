import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize
import statsmodels.api as sm

# Page configuration
st.set_page_config(
    page_title="Quantitative Portfolio Analysis",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e2e;
    }
    [data-testid="stSidebar"] {
        background-color: #2b2b3d;
    }
    h1, h2, h3, p, label {
        color: #ffffff !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #ffffff;
    }
    .metric-card {
        background-color: #2d2d3f;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #3d3d5f;
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #ff4757;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ff6b7a;
    }
    div[data-baseweb="select"] > div {
        background-color: #2d2d3f;
        color: white;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #2d2d3f;
        color: white;
        border: 1px solid #3d3d5f;
        border-radius: 8px;
    }
    .stNumberInput>div>div>input {
        background-color: #2d2d3f;
        color: white;
    }
    [data-testid="stExpander"] {
        background-color: #2d2d3f;
        border: 1px solid #3d3d5f;
        border-radius: 8px;
    }
    .stDataFrame {
        background-color: #2d2d3f;
    }
    div[data-testid="stMarkdownContainer"] > p {
        color: #d1d5db;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Overview'
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'prices' not in st.session_state:
    st.session_state.prices = None
if 'returns' not in st.session_state:
    st.session_state.returns = None
if 'capm_results' not in st.session_state:
    st.session_state.capm_results = None
if 'valid_tickers' not in st.session_state:
    st.session_state.valid_tickers = []
if 'market_ticker' not in st.session_state:
    st.session_state.market_ticker = 'SPY'
if 'risk_free_rate' not in st.session_state:
    st.session_state.risk_free_rate = 0.042
if 'investment_amount' not in st.session_state:
    st.session_state.investment_amount = 10000
if 'years' not in st.session_state:
    st.session_state.years = 5

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🏦 Navigation")
    
    nav_buttons = {
        '🏠 Overview': 'Overview',
        '📊 CAPM Analysis': 'CAPM Analysis',
        '🥧 Portfolio Composition': 'Portfolio Composition',
        '⚡ Performance Metrics': 'Performance Metrics',
        '💰 Margin Analysis': 'Margin Analysis'
    }
    
    for label, page in nav_buttons.items():
        if st.button(label, key=f"nav_{page}", use_container_width=True):
            st.session_state.page = page
            st.rerun()
    
    st.markdown("---")
    st.markdown("## 🚀 Custom Analysis Stats")
    
    # Display stats
    total_stocks = len(st.session_state.valid_tickers) if st.session_state.data_loaded else 0
    st.metric("Total Stocks", total_stocks)
    
    data_period = f"{st.session_state.years} Years"
    st.metric("Data Period", data_period)
    st.markdown("📈 **Monthly**")
    
    sig_alphas = 0
    if st.session_state.capm_results is not None:
        sig_alphas = st.session_state.capm_results['Significant'].sum()
    st.metric("Significant Alphas", int(sig_alphas))
    
    # Show confidence level if data loaded
    if st.session_state.data_loaded and hasattr(st.session_state, 'confidence_level'):
        st.markdown("---")
        st.markdown(f"**Confidence Level:** {st.session_state.confidence_level}%")

# Main content based on current page
current_page = st.session_state.page

# ============================================================================
# OVERVIEW PAGE
# ============================================================================
if current_page == 'Overview':
    st.markdown("# 🏦 Quantitative Portfolio Analysis")
    st.markdown("""
    Transform raw stock data into actionable investment insights using advanced quantitative finance techniques. This professional-grade portfolio 
    analytics platform combines modern portfolio theory, capital asset pricing models, and statistical optimization to help you construct 
    data-driven investment portfolios with superior risk-adjusted returns.
    
    **What You Can Do:**
    - 📊 **Analyze Any Stocks** - Input your own tickers and get instant CAPM analysis with alpha/beta calculations
    - 🎯 **Discover Alpha** - Identify stocks with statistically significant excess returns beyond market expectations
    - 🔬 **Factor Analysis** - Extend beyond basic CAPM with Fama-French multi-factor models to understand what drives returns
    - 💎 **Optimize Portfolios** - Build efficient portfolios that maximize returns for your chosen risk level
    - 📈 **Visualize Performance** - Interactive charts showing correlations, efficient frontiers, and historical trends
    - 💰 **Margin Analysis** - Understand leverage effects and margin requirements for your investment strategy
    
    Built for investors, analysts, and finance students who want to move beyond gut feelings and make quantitatively-informed decisions.
    """)
    
    st.markdown("---")
    st.markdown("## 🎯 Stock Selection")
    
    with st.expander("📊 Configure Your Analysis", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            default_tickers = "AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, AMD, PG, V"
            stock_input = st.text_area(
                "Enter stock tickers (comma-separated)",
                value=default_tickers,
                height=100
            )
        
        with col2:
            market_ticker = st.text_input("Market Index Ticker", value="SPY")
            years = st.slider("Years of Historical Data", 1, 10, 5)
        
        col3, col4, col5 = st.columns(3)
        with col3:
            risk_free = st.number_input("Risk Free Rate (%)", min_value=0.0, max_value=10.0, value=4.20, step=0.1)
        with col4:
            investment = st.number_input("Investment Amount ($)", min_value=1000, max_value=10000000, value=10000, step=1000)
        with col5:
            confidence_level = st.selectbox(
                "Confidence Level",
                options=[90, 95, 99],
                index=1,
                help="Statistical confidence level for alpha significance"
            )
        
        if st.button("🚀 Analyze Stocks", type="primary", use_container_width=True):
            tickers = [t.strip().upper() for t in stock_input.split(',') if t.strip()]
            
            if len(tickers) == 0:
                st.error("❌ Please enter at least one stock ticker.")
            else:
                with st.spinner('📥 Downloading data and performing analysis...'):
                    # Download data
                    all_tickers = tickers + [market_ticker]
                    prices_df = pd.DataFrame()
                    failed = []
                    
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=years*365)
                    
                    progress = st.progress(0)
                    status = st.empty()
                    
                    for i, ticker in enumerate(all_tickers):
                        status.text(f"Downloading {ticker}... ({i+1}/{len(all_tickers)})")
                        try:
                            # Method 1: Try using Ticker object
                            ticker_obj = yf.Ticker(ticker)
                            hist = ticker_obj.history(start=start_date, end=end_date, interval="1mo")
                            
                            if not hist.empty and 'Close' in hist.columns:
                                prices_df[ticker] = hist['Close']
                            else:
                                # Method 2: Try download function as fallback
                                try:
                                    data = yf.download(ticker, start=start_date, end=end_date, interval="1mo", progress=False, ignore_tz=True)
                                    if not data.empty:
                                        if 'Close' in data.columns:
                                            prices_df[ticker] = data['Close']
                                        elif isinstance(data.columns, pd.MultiIndex):
                                            prices_df[ticker] = data['Close'][ticker] if ticker in data['Close'].columns else data['Close'].iloc[:, 0]
                                    else:
                                        failed.append(ticker)
                                except:
                                    failed.append(ticker)
                        except Exception as e:
                            failed.append(ticker)
                        progress.progress((i + 1) / len(all_tickers))
                    
                    progress.empty()
                    status.empty()
                    
                    # Show download summary
                    if failed:
                        if len(failed) == len(all_tickers):
                            st.error(f"❌ Could not download any data. This might be due to:")
                            st.markdown("""
                            - **Network connectivity issues**
                            - **Yahoo Finance API temporarily unavailable**
                            - **Invalid ticker symbols**
                            
                            **Troubleshooting:**
                            1. Check your internet connection
                            2. Verify ticker symbols are correct (e.g., AAPL not APPLE)
                            3. Try again in a few moments
                            4. Try with fewer tickers first (e.g., just AAPL, MSFT)
                            """)
                        else:
                            st.warning(f"⚠️ Could not download: {', '.join(failed)}")
                    
                    # Check if we have enough data
                    valid_stock_tickers = [t for t in tickers if t in prices_df.columns]
                    
                    if market_ticker not in prices_df.columns:
                        st.error(f"❌ Market index '{market_ticker}' data not available. Please check the ticker.")
                    elif len(valid_stock_tickers) == 0:
                        st.error("❌ No valid stock data downloaded. Please check ticker symbols.")
                    else:
                        # Calculate log returns
                        log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
                        
                        # CAPM Analysis
                        r_f_monthly = risk_free / 100 / 12
                        alpha_threshold = 1 - (confidence_level / 100)
                        results = pd.DataFrame(columns=['Alpha', 'Beta', 'T-Stat', 'P-Value', 'Significant'])
                        
                        for ticker in valid_stock_tickers:
                            excess_stock = log_returns[ticker] - r_f_monthly
                            excess_market = log_returns[market_ticker] - r_f_monthly
                            
                            X = sm.add_constant(excess_market)
                            model = sm.OLS(excess_stock, X).fit()
                            
                            results.loc[ticker] = [
                                model.params['const'],
                                model.params[market_ticker],
                                model.tvalues['const'],
                                model.pvalues['const'],
                                model.pvalues['const'] < alpha_threshold
                            ]
                        
                        # Industry classification
                        industry_map = {
                            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
                            'AMZN': 'Consumer Cyclical', 'TSLA': 'Consumer Cyclical',
                            'NVDA': 'Technology', 'META': 'Technology', 'AMD': 'Technology',
                            'PG': 'Consumer Defensive', 'V': 'Financial Services',
                            'JPM': 'Financial Services', 'BAC': 'Financial Services',
                            'WMT': 'Consumer Defensive', 'JNJ': 'Healthcare',
                            'UNH': 'Healthcare', 'XOM': 'Energy', 'CVX': 'Energy',
                            'COST': 'Consumer Defensive', 'HD': 'Consumer Cyclical',
                            'LLY': 'Healthcare', 'AVGO': 'Technology', 'PLTR': 'Technology'
                        }
                        
                        industries = [industry_map.get(t, 'Other') for t in valid_stock_tickers]
                        industry_counts = pd.Series(industries).value_counts()
                        dominant_industry = industry_counts.idxmax()
                        industry_pct = (industry_counts.iloc[0] / len(valid_stock_tickers)) * 100
                        
                        # Store in session state
                        st.session_state.prices = prices_df
                        st.session_state.returns = log_returns
                        st.session_state.capm_results = results
                        st.session_state.valid_tickers = valid_stock_tickers
                        st.session_state.market_ticker = market_ticker
                        st.session_state.risk_free_rate = risk_free / 100
                        st.session_state.investment_amount = investment
                        st.session_state.years = years
                        st.session_state.confidence_level = confidence_level
                        st.session_state.dominant_industry = dominant_industry
                        st.session_state.industry_pct = industry_pct
                        st.session_state.data_loaded = True
                        
                        st.success(f"✅ Analysis completed! {len(valid_stock_tickers)} stocks analyzed.")
                        st.rerun()
    
    # Display summary metrics
    if st.session_state.data_loaded:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        # Get industry info
        industry_text = ""
        if hasattr(st.session_state, 'dominant_industry'):
            industry_text = f" • {st.session_state.industry_pct:.0f}% {st.session_state.dominant_industry}"
        
        with col1:
            total = len(st.session_state.valid_tickers) + 1
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #9ca3af; font-size: 14px; margin: 0;">Total Assets</p>
                <h1 style="margin: 10px 0;">{total} Stocks</h1>
                <p style="color: #10b981; margin: 0;">↑ +{st.session_state.market_ticker}{industry_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sig = int(st.session_state.capm_results['Significant'].sum())
            total_analyzed = len(st.session_state.capm_results)
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #9ca3af; font-size: 14px; margin: 0;">Significant Alpha</p>
                <h1 style="margin: 10px 0;">{sig}</h1>
                <p style="color: #10b981; margin: 0;"><span style="color: #10b981;">↑</span> out of {total_analyzed}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Calculate best Sharpe ratio from CAPM results
            if st.session_state.capm_results is not None and len(st.session_state.capm_results) > 0:
                returns = st.session_state.returns
                r_f = st.session_state.risk_free_rate / 12
                
                best_sharpe = -999
                best_stock = None
                
                for ticker in st.session_state.valid_tickers:
                    if ticker in returns.columns:
                        stock_return = returns[ticker].mean()
                        stock_vol = returns[ticker].std()
                        if stock_vol > 0:
                            sharpe = (stock_return - r_f) / stock_vol
                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_stock = ticker
                
                if best_stock:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color: #9ca3af; font-size: 14px; margin: 0;">Best Sharpe</p>
                        <h1 style="margin: 10px 0;">{best_sharpe:.4f}</h1>
                        <p style="color: #10b981; margin: 0;"><span style="color: #10b981;">↑</span> {best_stock}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color: #9ca3af; font-size: 14px; margin: 0;">Best Sharpe</p>
                        <h1 style="margin: 10px 0;">-</h1>
                        <p style="color: #f59e0b; margin: 0;"><span style="color: #f59e0b;">↑</span> Not calculated</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <p style="color: #9ca3af; font-size: 14px; margin: 0;">Best Sharpe</p>
                    <h1 style="margin: 10px 0;">-</h1>
                    <p style="color: #f59e0b; margin: 0;"><span style="color: #f59e0b;">↑</span> Not calculated</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #9ca3af; font-size: 14px; margin: 0;">Portfolio Value</p>
                <h1 style="margin: 10px 0;">${st.session_state.investment_amount:,}</h1>
                <p style="color: #10b981; margin: 0;"><span style="color: #10b981;">↑</span> Configured</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Industry Analysis
        if hasattr(st.session_state, 'dominant_industry'):
            st.markdown("---")
            st.info(f"📊 **Portfolio Composition:** Your portfolio is **{st.session_state.industry_pct:.0f}% {st.session_state.dominant_industry}** stocks. This indicates a concentrated exposure to the {st.session_state.dominant_industry} sector.")
        
        
        # Project Summary
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("## 📚 Project Summary")
            st.markdown("This project applies **quantitative techniques** in portfolio analysis to:")
            st.markdown("""
            - 📊 Fetch historical monthly price data for selected stocks from Yahoo Finance
            - 📈 Calculate monthly log returns, variance-covariance matrices, and statistical metrics
            - 💹 Use CAPM to estimate each asset's sensitivity to market returns (beta) and excess return (alpha)
            - 🔬 Extend analysis with Fama-French multi-factor models (SMB, HML, MOM)
            - ⚖️ Construct active and passive portfolios using the Single Index Model (SIM)
            - 🎯 Compare expected performance using Sharpe Ratio metrics
            - 🧮 Optimize portfolio weights for maximum risk-adjusted returns
            - 💰 Analyze margin requirements and leverage effects on portfolio performance
            """)
        
        with col2:
            st.markdown("## 🔧 Methodology")
            
            st.markdown("**Data Collection**")
            st.markdown("- Yahoo Finance API\n- Customizable time periods\n- Real-time data fetching")
            
            st.markdown("**Risk Analysis**")
            st.markdown("- CAPM Regression\n- Fama-French 3-Factor\n- Momentum Factor\n- Statistical Significance Testing")
            
            st.markdown("**Portfolio Optimization**")
            st.markdown("- Mean-Variance Analysis\n- Sharpe Ratio Maximization\n- Margin Constraints\n- Scipy SLSQP Optimizer")
        
        # Stock Analysis Results
        st.markdown("---")
        st.markdown("## 📊 Stock Analysis Results")
        
        display_df = st.session_state.capm_results.copy()
        display_df.index.name = 'Stock'
        display_df = display_df.reset_index()
        display_df['Alpha'] = display_df['Alpha'].apply(lambda x: f"{x:.4f}")
        display_df['Beta'] = display_df['Beta'].round(3)
        display_df['P-Value'] = display_df['P-Value'].round(3)
        display_df['Significant'] = display_df['Significant'].apply(lambda x: 'Yes' if x else 'No')
        
        st.dataframe(display_df, use_container_width=True, height=400)

# ============================================================================
# CAPM ANALYSIS PAGE
# ============================================================================
elif current_page == 'CAPM Analysis':
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Overview page first.")
    else:
        st.markdown("# Capital Asset Pricing Model Results")
        
        tab1, tab2 = st.tabs(["📊 Alpha & Beta", "🔬 Fama-French Extension"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            results = st.session_state.capm_results
            
            with col1:
                st.markdown("### Alpha by Stock (CAPM)")
                
                alpha_data = results['Alpha'].copy()
                colors = ['#10b981' if x > 0 else '#ef4444' for x in alpha_data.values]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=alpha_data.index,
                        y=alpha_data.values,
                        marker_color=colors,
                        text=[f"{x:.4f}" for x in alpha_data.values],
                        textposition='outside',
                        textfont=dict(size=10)
                    )
                ])
                fig.update_layout(
                    plot_bgcolor='#1e1e2e',
                    paper_bgcolor='#1e1e2e',
                    font=dict(color='white'),
                    xaxis=dict(title='Stock', gridcolor='#3d3d5f'),
                    yaxis=dict(title='Alpha', gridcolor='#3d3d5f'),
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Beta by Stock (Market Sensitivity)")
                
                beta_data = results['Beta'].copy()
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=beta_data.index,
                        y=beta_data.values,
                        marker_color='#3b82f6',
                        text=[f"{x:.2f}" for x in beta_data.values],
                        textposition='outside',
                        textfont=dict(size=10)
                    )
                ])
                fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                             annotation_text="Market Beta = 1.0")
                fig.update_layout(
                    plot_bgcolor='#1e1e2e',
                    paper_bgcolor='#1e1e2e',
                    font=dict(color='white'),
                    xaxis=dict(title='Stock', gridcolor='#3d3d5f'),
                    yaxis=dict(title='Beta', gridcolor='#3d3d5f'),
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical Significance
            st.markdown("---")
            st.markdown("## 🎯 Statistical Significance (95% Confidence)")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### P-Values for Alpha Significance")
                
                pval_data = results['P-Value'].copy()
                colors = ['#10b981' if x < 0.05 else '#ef4444' for x in pval_data.values]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=pval_data.index,
                        y=pval_data.values,
                        marker_color=colors,
                        text=[f"{x:.3f}" for x in pval_data.values],
                        textposition='outside',
                        textfont=dict(size=10)
                    )
                ])
                fig.add_hline(y=0.05, line_dash="dash", line_color="white",
                             annotation_text="Significance Level (0.05)")
                fig.update_layout(
                    plot_bgcolor='#1e1e2e',
                    paper_bgcolor='#1e1e2e',
                    font=dict(color='white'),
                    xaxis=dict(title='Stock', gridcolor='#3d3d5f'),
                    yaxis=dict(title='P-Value', gridcolor='#3d3d5f'),
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Key Findings")
                
                sig_count = results['Significant'].sum()
                total_count = len(results)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h2>Significant Alphas</h2>
                    <h1 style="color: #10b981;">{int(sig_count)}</h1>
                    <p style="color: #3b82f6;">↑ out of {total_count}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                sig_stocks = results[results['Significant'] == True].index.tolist()
                if sig_stocks:
                    stock_list = ', '.join(sig_stocks)
                    st.info(f"**{stock_list}** {'is' if len(sig_stocks) == 1 else 'are'} the only stock{'s' if len(sig_stocks) > 1 else ''} with statistically significant alpha at the 95% confidence level.")
                else:
                    st.info("No stocks have statistically significant alpha at the 95% confidence level.")
        
        with tab2:
            st.info("🔬 Fama-French factor analysis requires uploading the Fama-French factor CSV file.")
            
            uploaded_file = st.file_uploader("Upload Fama-French Factors CSV", type=['csv'])
            
            if uploaded_file is not None:
                ff_data = pd.read_csv(uploaded_file)
                st.success("✅ Fama-French data loaded!")
                st.dataframe(ff_data.head(10), use_container_width=True)

# ============================================================================
# PORTFOLIO COMPOSITION PAGE
# ============================================================================
elif current_page == 'Portfolio Composition':
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Overview page first.")
    else:
        st.markdown("# 🥧 Portfolio Composition")
        st.markdown("## Optimized Tangent Portfolio")
        
        st.markdown("### 💡 Select stocks to include in your optimized portfolio")
        
        tickers = st.session_state.valid_tickers
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Choose stocks for portfolio optimization**")
            selected_stocks = st.multiselect(
                "Select stocks",
                options=tickers,
                default=tickers[:4] if len(tickers) >= 4 else tickers,
                label_visibility="collapsed"
            )
        
        with col2:
            investment_amount = st.number_input(
                "Total Investment Amount ($)",
                min_value=1000,
                value=st.session_state.investment_amount,
                step=1000
            )
        
        st.markdown("### ⚙️ Optimization Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            min_weight = st.slider("Minimum Weight (%)", 0, 20, 5) / 100
        with col2:
            opt_method = st.selectbox("Optimization Method", 
                                     ["Maximum Sharpe Ratio", "Minimum Variance"])
        with col3:
            margin_pct = st.slider("Initial Margin (%)", 25, 100, 50)
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🎯 Optimize Portfolio", type="primary", use_container_width=True):
            if len(selected_stocks) >= 2:
                try:
                    returns = st.session_state.returns
                    opt_returns = returns[selected_stocks].dropna()
                    cov_matrix = opt_returns.cov().values
                    expected_returns = opt_returns.mean().values
                    r_f = st.session_state.risk_free_rate / 12
                    
                    n = len(selected_stocks)
                    
                    if opt_method == "Maximum Sharpe Ratio":
                        def objective(weights):
                            ret = np.dot(weights, expected_returns)
                            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                            if vol == 0:
                                return 1e10
                            return -(ret - r_f) / vol
                        
                        constraints = [
                            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                            {'type': 'ineq', 'fun': lambda w: w - min_weight}
                        ]
                        bounds = [(min_weight, 1) for _ in range(n)]
                    else:
                        def objective(weights):
                            return np.dot(weights.T, np.dot(cov_matrix, weights))
                        
                        constraints = [
                            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                            {'type': 'ineq', 'fun': lambda w: w - min_weight}
                        ]
                        bounds = [(min_weight, 1) for _ in range(n)]
                    
                    initial = np.ones(n) / n
                    result = minimize(objective, initial, method='SLSQP',
                                    bounds=bounds, constraints=constraints)
                    
                    if result.success:
                        weights = result.x
                        
                        # Calculate metrics
                        port_return = np.dot(weights, expected_returns)
                        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                        port_volatility = np.sqrt(port_variance)
                        sharpe = (port_return - r_f) / port_volatility
                        
                        # Store results
                        st.session_state.portfolio_weights = weights
                        st.session_state.portfolio_stocks = selected_stocks
                        st.session_state.portfolio_return = port_return
                        st.session_state.portfolio_volatility = port_volatility
                        st.session_state.portfolio_sharpe = sharpe
                        st.session_state.portfolio_investment = investment_amount
                        st.session_state.portfolio_margin = margin_pct / 100
                        
                        st.success("✅ Portfolio optimized successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Optimization failed. Try different settings.")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please select at least 2 stocks.")
        
        # Display results if available
        if hasattr(st.session_state, 'portfolio_weights'):
            st.markdown("---")
            st.markdown("## 📊 Optimized Portfolio Allocation")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=st.session_state.portfolio_stocks,
                    values=st.session_state.portfolio_weights,
                    hole=0.4,
                    marker=dict(colors=px.colors.qualitative.Set3),
                    textinfo='label+percent',
                    textfont=dict(size=12)
                )])
                fig.update_layout(
                    plot_bgcolor='#1e1e2e',
                    paper_bgcolor='#1e1e2e',
                    font=dict(color='white'),
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Allocation table
                allocation_df = pd.DataFrame({
                    'Stock': st.session_state.portfolio_stocks,
                    'Weight (%)': st.session_state.portfolio_weights * 100,
                    'Investment ($)': st.session_state.portfolio_weights * st.session_state.portfolio_investment,
                    'Margin Required ($)': st.session_state.portfolio_weights * st.session_state.portfolio_investment * st.session_state.portfolio_margin
                })
                st.dataframe(allocation_df.round(2), use_container_width=True, height=400)
            
            # Performance metrics
            st.markdown("---")
            st.markdown("## 📈 Portfolio Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Monthly Return", f"{st.session_state.portfolio_return*100:.2f}%")
            with col2:
                st.metric("Annual Return", f"{st.session_state.portfolio_return*12*100:.2f}%")
            with col3:
                st.metric("Monthly Volatility", f"{st.session_state.portfolio_volatility*100:.2f}%")
            with col4:
                st.metric("Sharpe Ratio", f"{st.session_state.portfolio_sharpe:.4f}")
            
            # Efficient Frontier
            st.markdown("---")
            st.markdown("## 🎯 Efficient Frontier")
            
            returns = st.session_state.returns
            selected = st.session_state.portfolio_stocks
            opt_returns = returns[selected].dropna()
            cov_matrix = opt_returns.cov().values
            expected_returns = opt_returns.mean().values
            r_f = st.session_state.risk_free_rate / 12
            
            # Generate random portfolios
            n_portfolios = 1000
            returns_array = []
            volatility_array = []
            sharpe_array = []
            
            np.random.seed(42)
            for _ in range(n_portfolios):
                weights = np.random.random(len(selected))
                weights /= np.sum(weights)
                
                ret = np.dot(weights, expected_returns)
                vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe = (ret - r_f) / vol
                
                returns_array.append(ret * 100)
                volatility_array.append(vol * 100)
                sharpe_array.append(sharpe)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=volatility_array,
                y=returns_array,
                mode='markers',
                marker=dict(
                    size=5,
                    color=sharpe_array,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                name='Random Portfolios',
                hovertemplate='Volatility: %{x:.2f}%<br>Return: %{y:.2f}%'
            ))
            
            fig.add_trace(go.Scatter(
                x=[st.session_state.portfolio_volatility * 100],
                y=[st.session_state.portfolio_return * 100],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star', line=dict(color='white', width=2)),
                name='Optimal Portfolio',
                hovertemplate='Optimal<br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%'
            ))
            
            fig.update_layout(
                title='Efficient Frontier',
                plot_bgcolor='#1e1e2e',
                paper_bgcolor='#1e1e2e',
                font=dict(color='white'),
                xaxis=dict(title='Volatility (%)', gridcolor='#3d3d5f'),
                yaxis=dict(title='Expected Return (%)', gridcolor='#3d3d5f'),
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PERFORMANCE METRICS PAGE
# ============================================================================
elif current_page == 'Performance Metrics':
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Overview page first.")
    else:
        st.markdown("# ⚡ Performance Metrics")
        
        # Price Evolution
        st.markdown("## 📈 Price Evolution")
        
        prices = st.session_state.prices
        tickers = st.session_state.valid_tickers + [st.session_state.market_ticker]
        
        # Normalize prices
        normalized = prices[tickers].copy()
        for col in normalized.columns:
            normalized[col] = (normalized[col] / normalized[col].iloc[0]) * 100
        
        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(go.Scatter(
                x=normalized.index,
                y=normalized[ticker],
                mode='lines',
                name=ticker,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='Normalized Price Evolution (Base = 100)',
            plot_bgcolor='#1e1e2e',
            paper_bgcolor='#1e1e2e',
            font=dict(color='white'),
            xaxis=dict(title='Date', gridcolor='#3d3d5f'),
            yaxis=dict(title='Normalized Price', gridcolor='#3d3d5f'),
            height=500,
            hovermode='x unified',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns Distribution
        st.markdown("---")
        st.markdown("## 📊 Returns Distribution")
        
        returns = st.session_state.returns
        selected_ticker = st.selectbox("Select stock to view distribution", tickers)
        
        fig = go.Figure(data=[go.Histogram(
            x=returns[selected_ticker] * 100,
            nbinsx=30,
            marker_color='#3b82f6',
            name=selected_ticker
        )])
        
        fig.update_layout(
            title=f'{selected_ticker} Monthly Returns Distribution',
            plot_bgcolor='#1e1e2e',
            paper_bgcolor='#1e1e2e',
            font=dict(color='white'),
            xaxis=dict(title='Monthly Return (%)', gridcolor='#3d3d5f'),
            yaxis=dict(title='Frequency', gridcolor='#3d3d5f'),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Matrix
        st.markdown("---")
        st.markdown("## 🔗 Correlation Matrix")
        
        corr = returns[tickers].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Correlation Matrix',
            plot_bgcolor='#1e1e2e',
            paper_bgcolor='#1e1e2e',
            font=dict(color='white'),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MARGIN ANALYSIS PAGE
# ============================================================================
elif current_page == 'Margin Analysis':
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data from the Overview page first.")
    elif not hasattr(st.session_state, 'portfolio_weights'):
        st.warning("⚠️ Please optimize a portfolio from the Portfolio Composition page first.")
    else:
        st.markdown("# 💰 Margin Analysis")
        
        st.markdown("## 💵 Leverage Effects on Portfolio Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Current Portfolio (Base Case)")
            
            base_investment = st.session_state.portfolio_investment
            base_return = st.session_state.portfolio_return
            base_volatility = st.session_state.portfolio_volatility
            base_sharpe = st.session_state.portfolio_sharpe
            
            st.metric("Investment", f"${base_investment:,}")
            st.metric("Monthly Return", f"{base_return*100:.2f}%")
            st.metric("Monthly Volatility", f"{base_volatility*100:.2f}%")
            st.metric("Sharpe Ratio", f"{base_sharpe:.4f}")
        
        with col2:
            st.markdown("### With 50% Margin")
            
            leverage = 1 / 0.5  # 2x leverage
            leveraged_investment = base_investment * leverage
            leveraged_return = base_return * leverage
            leveraged_volatility = base_volatility * leverage
            leveraged_sharpe = base_sharpe  # Sharpe ratio unchanged with leverage
            
            st.metric("Effective Investment", f"${leveraged_investment:,}", 
                     delta=f"+${leveraged_investment-base_investment:,}")
            st.metric("Monthly Return", f"{leveraged_return*100:.2f}%",
                     delta=f"+{(leveraged_return-base_return)*100:.2f}%")
            st.metric("Monthly Volatility", f"{leveraged_volatility*100:.2f}%",
                     delta=f"+{(leveraged_volatility-base_volatility)*100:.2f}%")
            st.metric("Sharpe Ratio", f"{leveraged_sharpe:.4f}")
        
        st.markdown("---")
        st.markdown("## 📊 Margin Requirement Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 50% Initial Margin")
            
            margin_50 = 0.5
            allocation = pd.DataFrame({
                'Stock': st.session_state.portfolio_stocks,
                'Weight (%)': st.session_state.portfolio_weights * 100,
                'Investment ($)': st.session_state.portfolio_weights * base_investment,
                'Margin Required ($)': st.session_state.portfolio_weights * base_investment * margin_50
            })
            
            st.dataframe(allocation.round(2), use_container_width=True, height=300)
            
            total_margin = allocation['Margin Required ($)'].sum()
            st.metric("Total Margin Required", f"${total_margin:,.2f}")
        
        with col2:
            st.markdown("### 75% Initial Margin")
            
            margin_75 = 0.75
            allocation_75 = pd.DataFrame({
                'Stock': st.session_state.portfolio_stocks,
                'Weight (%)': st.session_state.portfolio_weights * 100,
                'Investment ($)': st.session_state.portfolio_weights * base_investment,
                'Margin Required ($)': st.session_state.portfolio_weights * base_investment * margin_75
            })
            
            st.dataframe(allocation_75.round(2), use_container_width=True, height=300)
            
            total_margin_75 = allocation_75['Margin Required ($)'].sum()
            st.metric("Total Margin Required", f"${total_margin_75:,.2f}")
        
        # Sharpe Ratio Comparison
        st.markdown("---")
        st.markdown("## 📊 Sharpe Ratio with Different Margin Levels")
        
        margin_levels = [0.25, 0.50, 0.75, 1.00]
        sharpe_ratios = []
        
        for margin in margin_levels:
            leverage_factor = 1 / margin
            adjusted_sharpe = base_sharpe  # Sharpe remains same theoretically
            sharpe_ratios.append(adjusted_sharpe)
        
        fig = go.Figure(data=[go.Bar(
            x=[f"{int(m*100)}%" for m in margin_levels],
            y=sharpe_ratios,
            marker_color='#3b82f6',
            text=[f"{s:.4f}" for s in sharpe_ratios],
            textposition='outside'
        )])
        
        fig.update_layout(
            title='Sharpe Ratio vs Margin Requirement',
            plot_bgcolor='#1e1e2e',
            paper_bgcolor='#1e1e2e',
            font=dict(color='white'),
            xaxis=dict(title='Initial Margin (%)', gridcolor='#3d3d5f'),
            yaxis=dict(title='Sharpe Ratio', gridcolor='#3d3d5f'),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("ℹ️ Note: Higher leverage increases both returns and risk proportionally, leaving the Sharpe ratio theoretically unchanged. However, in practice, margin calls and financing costs may affect actual performance.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280;'>
    <p>📊 Quantitative Portfolio Analysis Dashboard</p>
    <p>Built with Streamlit • Data from Yahoo Finance • <a href='https://github.com/RonithReddy08/quantitative-investment-analysis' style='color: #3b82f6;'>Github Repository</a></p>
</div>
""", unsafe_allow_html=True)