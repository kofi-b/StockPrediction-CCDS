import yfinance as yf
import pandas as pd
import os
from loguru import logger
from stock_prediction.config import PROCESSED_DATA_DIR, REPORTS_DIR

# Configuration
input_base_dir = PROCESSED_DATA_DIR / "feature_engineering"
output_base_dir = REPORTS_DIR / "financial_metrics"
tickers = ['AAPL', 'TSLA', 'META', 'AMZN', 'MSFT', 'NVDA', 'CSCO', 'MA']

# Create output base directory
output_base_dir.mkdir(parents=True, exist_ok=True)

# CAPM and DCF parameters
risk_free_rate = 0.02  # 10-year Treasury yield approximation
market_return = 0.08  # Historical S&P 500 return approximation
growth_rate = 0.05  # Assumed growth rate for DCF

def calculate_capm_dcf(ticker):
    """Calculate CAPM and DCF for a given ticker."""
    # Create ticker-specific output directory
    output_dir = output_base_dir / ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load engineered data
    input_file = input_base_dir / ticker / f"{ticker}_engineered_data.csv"
    if not input_file.exists():
        raise FileNotFoundError(f"Engineered data file not found: {input_file}")
    df = pd.read_csv(input_file)

    # Fetch beta from Yahoo Finance
    yf_ticker = yf.Ticker(ticker)
    beta = yf_ticker.info.get('beta', 1.2)  # Default to 1.2 if unavailable

    # Extract latest pe_ratio and forward_eps
    if 'pe_ratio' not in df.columns or 'forward_eps' not in df.columns:
        raise ValueError(f"Missing 'pe_ratio' or 'forward_eps' in data for {ticker}")
    pe_ratio = df['pe_ratio'].iloc[-1]
    forward_eps = df['forward_eps'].iloc[-1]

    # CAPM calculation
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)

    # DCF calculation
    discount_rate = expected_return
    try:
        intrinsic_value = (forward_eps * (1 + growth_rate)) / (discount_rate - growth_rate)
    except ZeroDivisionError:
        intrinsic_value = float('nan')  # Handle cases where discount_rate <= growth_rate

    # Store results
    results = {
        'ticker': ticker,
        'beta': beta,
        'expected_return': expected_return,
        'pe_ratio': pe_ratio,
        'forward_eps': forward_eps,
        'intrinsic_value': intrinsic_value
    }

    # Save individual notes
    with open(output_dir / f"{ticker}_financial_notes.txt", 'w') as f:
        f.write(f"Financial Metrics for {ticker}:\n")
        f.write(f"- Beta: {beta:.2f}\n")
        f.write(f"- CAPM Expected Return: {expected_return:.4f}\n")
        f.write(f"- P/E Ratio: {pe_ratio:.2f}\n")
        f.write(f"- Forward EPS: {forward_eps:.2f}\n")
        f.write(f"- DCF Intrinsic Value: {intrinsic_value:.2f}\n")
        f.write("- Assumptions:\n")
        f.write(f"  - Risk-Free Rate: {risk_free_rate}\n")
        f.write(f"  - Market Return: {market_return}\n")
        f.write(f"  - Growth Rate: {growth_rate}\n")

    logger.info(f"Financial metrics calculated for {ticker}. Results saved in {output_dir}.")
    return results

def main():
    """Calculate financial metrics for all tickers and save results."""
    all_results = []
    for ticker in tickers:
        try:
            results = calculate_capm_dcf(ticker)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")

    # Save combined results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_base_dir / "all_tickers_financial_metrics.csv", index=False)
    logger.info("All financial metrics processed and saved.")

if __name__ == "__main__":
    main()