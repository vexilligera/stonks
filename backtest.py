#!/usr/bin/env python3
"""
Backtest for Mean Reversion Strategy.

Tests the MA-based mean reversion strategy on 2 years of historical hourly data.
Properly tracks capital across all symbols with a shared capital pool.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_config(config_path: str = "config.json") -> dict:
    """Load config from file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def fetch_hourly_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly data for a symbol."""
    try:
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval='1h',
            progress=False
        )
        if not df.empty:
            # Flatten MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            # Ensure columns are simple values, not DataFrames
            df = df.copy()
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = df[col].values.flatten()
            df['Symbol'] = symbol
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


def backtest_ma_strategy(
    symbols: list[str],
    start_date: str,
    end_date: str,
    ma_window: int = 40,
    buy_threshold: float = 0.03,
    take_profit: float = 0.05,
    notional_per_trade: float = 500.0,
    max_positions_per_symbol: int = 10,
    initial_capital: float = 25000.0,
) -> dict:
    """
    Backtest the Mean Reversion Strategy on historical data.
    
    Strategy:
    - BUY when price is X% below MA
    - SELL when price is X% above entry AND MA derivative <= 0
    
    Uses a SHARED capital pool across all symbols.
    """
    
    print("=" * 60)
    print("MEAN REVERSION STRATEGY BACKTEST")
    print("=" * 60)
    print(f"Period: {start_date} to {end_date}")
    print(f"Symbols: {len(symbols)}")
    print(f"MA Window: {ma_window} hours")
    print(f"Buy Threshold: -{buy_threshold*100:.1f}% below MA")
    print(f"Take Profit: +{take_profit*100:.1f}% above entry")
    print(f"Notional per Trade: ${notional_per_trade:.0f}")
    print(f"Max Positions per Symbol: {max_positions_per_symbol}")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print("=" * 60)
    print()
    
    # Fetch all data in parallel
    print("Fetching historical data...")
    all_data = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_hourly_data, s, start_date, end_date): s for s in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            df = future.result()
            if not df.empty and len(df) >= ma_window:
                all_data[symbol] = df
                print(f"  ✓ {symbol}: {len(df)} bars")
            else:
                print(f"  ✗ {symbol}: insufficient data")
    
    print(f"\nLoaded data for {len(all_data)}/{len(symbols)} symbols")
    print()
    
    # Precompute indicators for each symbol
    print("Computing indicators...")
    for symbol, df in all_data.items():
        df['avg_price'] = (df['Open'] + df['Close']) / 2
        df['ma'] = df['avg_price'].rolling(window=ma_window).mean()
        df['ma_prev'] = df['ma'].shift(1)
        df['ma_deriv'] = df['ma'] - df['ma_prev']
        df['deviation'] = (df['Close'] - df['ma']) / df['ma']
        all_data[symbol] = df
    
    # Combine all data into a single timeline
    print("Building unified timeline...")
    combined_rows = []
    for symbol, df in all_data.items():
        for timestamp, row in df.iterrows():
            if pd.notna(row['ma']) and pd.notna(row['ma_deriv']):
                combined_rows.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'price': row['Close'],
                    'ma': row['ma'],
                    'ma_deriv': row['ma_deriv'],
                    'deviation': row['deviation'],
                })
    
    timeline = pd.DataFrame(combined_rows)
    timeline = timeline.sort_values('timestamp').reset_index(drop=True)
    print(f"Timeline: {len(timeline)} data points")
    print()
    
    # Simulation state
    capital = initial_capital
    positions = {}  # symbol -> list of {entry_price, shares, entry_time}
    all_trades = []
    equity_curve = []
    
    print("Running backtest simulation...")
    
    # Process timeline chronologically
    for idx, row in timeline.iterrows():
        timestamp = row['timestamp']
        symbol = row['symbol']
        price = row['price']
        ma_deriv = row['ma_deriv']
        deviation = row['deviation']
        
        # Initialize position list for symbol
        if symbol not in positions:
            positions[symbol] = []
        
        # --- CHECK SELL SIGNALS FIRST ---
        positions_to_remove = []
        for pos in positions[symbol]:
            gain = (price - pos['entry_price']) / pos['entry_price']
            
            # Sell when: gain >= take_profit AND ma_derivative <= 0
            if gain >= take_profit and ma_deriv <= 0:
                pnl = (price - pos['entry_price']) * pos['shares']
                pnl_pct = gain * 100
                hold_hours = (timestamp - pos['entry_time']).total_seconds() / 3600
                
                all_trades.append({
                    'symbol': symbol,
                    'entry_time': pos['entry_time'],
                    'exit_time': timestamp,
                    'entry_price': pos['entry_price'],
                    'exit_price': price,
                    'shares': pos['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'hold_hours': hold_hours,
                    'notional': pos['entry_price'] * pos['shares'],
                })
                
                # Return capital
                capital += price * pos['shares']
                positions_to_remove.append(pos)
        
        for pos in positions_to_remove:
            positions[symbol].remove(pos)
        
        # --- CHECK BUY SIGNALS ---
        if deviation <= -buy_threshold:
            num_positions = len(positions[symbol])
            if num_positions < max_positions_per_symbol:
                shares = max(1, int(notional_per_trade / price))
                cost = price * shares
                
                # Only buy if we have enough capital
                if capital >= cost:
                    positions[symbol].append({
                        'entry_price': price,
                        'shares': shares,
                        'entry_time': timestamp,
                    })
                    capital -= cost
        
        # Track equity periodically (every 100 rows to save memory)
        if idx % 100 == 0:
            position_value = sum(
                price * p['shares'] 
                for sym_positions in positions.values() 
                for p in sym_positions
                if sym_positions  # Skip empty lists
            )
            # Get current prices for all positions
            total_position_value = 0
            for sym, sym_positions in positions.items():
                if sym_positions and sym in all_data:
                    # Find closest price for this symbol at this timestamp
                    sym_df = all_data[sym]
                    if timestamp in sym_df.index:
                        sym_price = sym_df.loc[timestamp, 'Close']
                    else:
                        # Use last known price before this timestamp
                        mask = sym_df.index <= timestamp
                        if mask.any():
                            sym_price = sym_df.loc[mask, 'Close'].iloc[-1]
                        else:
                            sym_price = sym_positions[0]['entry_price']
                    total_position_value += sum(sym_price * p['shares'] for p in sym_positions)
            
            equity_curve.append({
                'timestamp': timestamp,
                'capital': capital,
                'position_value': total_position_value,
                'total_equity': capital + total_position_value,
                'num_positions': sum(len(p) for p in positions.values()),
            })
    
    # Close remaining open positions at last price (mark to market)
    print("Marking open positions to market...")
    for symbol, sym_positions in positions.items():
        if sym_positions and symbol in all_data:
            last_price = all_data[symbol]['Close'].iloc[-1]
            last_time = all_data[symbol].index[-1]
            
            for pos in sym_positions:
                gain = (last_price - pos['entry_price']) / pos['entry_price']
                pnl = (last_price - pos['entry_price']) * pos['shares']
                hold_hours = (last_time - pos['entry_time']).total_seconds() / 3600
                
                all_trades.append({
                    'symbol': symbol,
                    'entry_time': pos['entry_time'],
                    'exit_time': last_time,
                    'entry_price': pos['entry_price'],
                    'exit_price': last_price,
                    'shares': pos['shares'],
                    'pnl': pnl,
                    'pnl_pct': gain * 100,
                    'hold_hours': hold_hours,
                    'notional': pos['entry_price'] * pos['shares'],
                    'status': 'OPEN',
                })
    
    # Compute final equity
    final_position_value = 0
    for symbol, sym_positions in positions.items():
        if sym_positions and symbol in all_data:
            last_price = all_data[symbol]['Close'].iloc[-1]
            final_position_value += sum(last_price * p['shares'] for p in sym_positions)
    
    final_equity = capital + final_position_value
    
    # Compute results
    if not all_trades:
        print("\n⚠️  No trades executed!")
        return {'trades': [], 'stats': {}}
    
    df_trades = pd.DataFrame(all_trades)
    df_equity = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame()
    
    # Statistics
    total_trades = len(df_trades)
    closed_trades = df_trades[~df_trades.get('status', pd.Series([''] * len(df_trades))).str.contains('OPEN', na=False)]
    open_trades = df_trades[df_trades.get('status', pd.Series([''] * len(df_trades))).str.contains('OPEN', na=False)]
    
    winning_trades = closed_trades[closed_trades['pnl'] > 0]
    losing_trades = closed_trades[closed_trades['pnl'] <= 0]
    
    total_pnl = closed_trades['pnl'].sum()
    realized_pnl = closed_trades['pnl'].sum()
    unrealized_pnl = open_trades['pnl'].sum() if len(open_trades) > 0 else 0
    
    win_rate = len(winning_trades) / len(closed_trades) * 100 if len(closed_trades) > 0 else 0
    
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    
    avg_hold_hours = closed_trades['hold_hours'].mean() if len(closed_trades) > 0 else 0
    avg_gain_pct = closed_trades['pnl_pct'].mean() if len(closed_trades) > 0 else 0
    
    # Profit factor
    gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max drawdown from equity curve
    if len(df_equity) > 0:
        df_equity['peak'] = df_equity['total_equity'].cummax()
        df_equity['drawdown'] = (df_equity['total_equity'] - df_equity['peak']) / df_equity['peak'] * 100
        max_drawdown = df_equity['drawdown'].min()
    else:
        max_drawdown = 0
    
    total_return = (final_equity - initial_capital) / initial_capital * 100
    
    # Print results
    print()
    print("=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print()
    print(f"📊 PERFORMANCE SUMMARY")
    print(f"   Initial Capital:    ${initial_capital:>12,.2f}")
    print(f"   Final Cash:         ${capital:>12,.2f}")
    print(f"   Position Value:     ${final_position_value:>12,.2f}")
    print(f"   Final Equity:       ${final_equity:>12,.2f}")
    print(f"   Total Return:       {total_return:>12.2f}%")
    print()
    print(f"💰 P&L BREAKDOWN")
    print(f"   Realized P&L:       ${realized_pnl:>12,.2f}")
    print(f"   Unrealized P&L:     ${unrealized_pnl:>12,.2f}")
    print(f"   Total P&L:          ${realized_pnl + unrealized_pnl:>12,.2f}")
    print()
    print(f"📈 TRADE STATISTICS")
    print(f"   Total Trades:       {total_trades:>12}")
    print(f"   Closed Trades:      {len(closed_trades):>12}")
    print(f"   Open Positions:     {len(open_trades):>12}")
    print(f"   Winning Trades:     {len(winning_trades):>12}")
    print(f"   Losing Trades:      {len(losing_trades):>12}")
    print(f"   Win Rate:           {win_rate:>12.1f}%")
    print()
    print(f"📉 RISK METRICS")
    print(f"   Avg Win:            ${avg_win:>12,.2f}")
    print(f"   Avg Loss:           ${avg_loss:>12,.2f}")
    print(f"   Avg Gain %:         {avg_gain_pct:>12.2f}%")
    print(f"   Profit Factor:      {profit_factor:>12.2f}")
    print(f"   Max Drawdown:       {max_drawdown:>12.2f}%")
    print()
    print(f"⏱️  TIMING")
    print(f"   Avg Hold Time:      {avg_hold_hours:>12.1f} hours ({avg_hold_hours/24:.1f} days)")
    print()
    print(f"📋 PER-SYMBOL BREAKDOWN (Top 10 by P&L)")
    print("-" * 60)
    
    symbol_pnl = df_trades.groupby('symbol')['pnl'].agg(['sum', 'count']).sort_values('sum', ascending=False)
    for symbol, row in symbol_pnl.head(10).iterrows():
        pnl_str = f"${row['sum']:>10,.2f}"
        trades_str = f"{int(row['count']):>3} trades"
        print(f"   {symbol:<6} {pnl_str}  ({trades_str})")
    
    print()
    print("=" * 60)
    
    # Save detailed results
    df_trades.to_csv('backtest_trades.csv', index=False)
    if len(df_equity) > 0:
        df_equity.to_csv('backtest_equity.csv', index=False)
    print(f"\n💾 Saved: backtest_trades.csv, backtest_equity.csv")
    
    return {
        'trades': df_trades,
        'equity': df_equity,
        'stats': {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_trades': total_trades,
            'closed_trades': len(closed_trades),
            'open_positions': len(open_trades),
            'win_rate': win_rate,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_return': total_return,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'avg_hold_hours': avg_hold_hours,
        }
    }


if __name__ == "__main__":
    # Load config
    config = load_config("config.json")
    
    # Use config values or defaults
    symbols = config.get('symbols', [
        "AAPL", "AMD", "IBM", "ORCL", "AMZN", "GE", "INTC", "MSFT",
        "AVGO", "GOOGL", "TSLA", "CSCO", "META", "NVDA", "QCOM",
        "TSM", "PLTR", "ASML", "MU"
    ])
    
    # Backtest period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    results = backtest_ma_strategy(
        symbols=symbols,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        ma_window=config.get('ma_window', 40),
        buy_threshold=config.get('buy_threshold', 0.03),
        take_profit=config.get('take_profit', 0.05),
        notional_per_trade=config.get('notional_per_trade', 500.0),
        max_positions_per_symbol=config.get('max_positions_per_symbol', 10),
        initial_capital=25000.0,
    )
