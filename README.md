# Quantitative Investment Analysis

This project applies quantitative finance techniques to analyze, evaluate, and optimize a portfolio of U.S. stocks using Python. Key concepts include CAPM, Fama-French factor models, Mean-Variance Optimization, Single Index Model (SIM), and Sharpe Ratio evaluation. The project also integrates margin constraints and explores both theoretical and practical aspects of portfolio construction.

---

## Project Objectives

* Analyze monthly stock returns using financial models
* Estimate alpha, beta, and assess statistical significance
* Construct portfolios using the Single Index Model (SIM)
* Optimize for minimum variance and maximum Sharpe Ratio
* Incorporate margin-based investment constraints
* Compare portfolio performance with the market (SPY)

---

## Datasets & Tools

* **Data Source**: Yahoo Finance (via `yfinance` API)
* **Time Period**: Last 5 years (monthly data)
* **Stocks Analyzed**:

  * AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, AMD, PG, V
  * SPY as market index
* **Tools/Libraries**: `pandas`, `numpy`, `yfinance`, `scipy`, `statsmodels`, `matplotlib`

---

## Methodology

### 1. Data Collection & Preprocessing

* Fetched monthly adjusted close prices
* Computed logarithmic returns
* Cleaned and structured data for modeling

### 2. CAPM Analysis

* Computed alpha and beta using the CAPM formula
* Performed OLS regression of excess returns
* Assessed statistical significance of alpha values

### 3. Fama-French & Momentum Model

* Extended CAPM by including SMB, HML, and MOM factors
* Evaluated improvement in alpha estimation

### 4. Single Index Model Portfolio Construction

* Calculated tracking error and alpha-based active weights
* Combined with beta-based passive weights to form overall portfolio

### 5. Sharpe Ratio Comparison

* Computed Sharpe Ratios for:

  * SIM Portfolio
  * SPY (Market)

### 6. Optimization

* **Minimum Variance Portfolio**:

  * Used `scipy.optimize.minimize` to allocate weights under constraints
* **Tangency Portfolio**:

  * Maximized Sharpe Ratio
  * Calculated risk and return statistics

### 7. Margin Analysis

* Adjusted weights for 50% and 75% margin constraints
* Computed total capital and margin requirements
* Re-evaluated Sharpe Ratios under margin assumptions

---

## Key Insights

* NVDA showed statistically significant alpha under CAPM
* Most stocks' alphas were not significant, supporting market efficiency
* Fama-French factors improved model fit but didn’t consistently enhance alpha significance
* SIM portfolio offered diversification benefits but lower Sharpe Ratio than SPY
* Optimized tangency portfolios provided higher risk-adjusted returns under specific margin assumptions

---

## Requirements

* Python 3.8+
* pandas
* numpy
* yfinance
* scipy
* statsmodels
* matplotlib

---

