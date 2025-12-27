import os
import time
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from utils import (
    get_order_info, get_current_time, is_market_hours, get_positions, get_open_orders,
    open_limit_order, cancel_order, get_buying_power, account
)


class MeanReversionStrategy:
    """
    Mean Reversion Strategy based on Moving Average.
    
    Strategy:
    - Compute MA using (open + close) / 2 of hourly bars
    - BUY when price is X% below MA
    - SELL when price is X% above entry cost AND MA derivative <= 0 (trend weakening)
    - Allow multiple positions per symbol
    
    Persists trade records to disk for recovery on restart.
    """
    
    def __init__(
        self,
        symbols: list[str] = [
            "AAPL", "AMD", "IBM", "ORCL", "AMZN", "GE", "INTC", "MSFT",
            "AVGO", "GOOGL", "TSLA", "CSCO", "META", "NVDA", "QCOM",
            "TSM", "PLTR", "ASML", "MU"
        ],
        ma_window: int = 40,  # 40 hours
        buy_threshold: float = 0.03,  # Buy when price is 3% below MA
        take_profit: float = 0.05,  # Sell when price is 5% above entry cost
        notional_per_trade: float = 500.0,  # Dollars per trade
        max_positions_per_symbol: int = 10,  # Max positions per symbol
        data_dir: str = "./ma_strategy_data",  # Directory for persistent storage
        execution_interval: int = 3600,  # Execution interval in seconds
    ):
        self.symbols = symbols
        self.ma_window = ma_window
        self.buy_threshold = buy_threshold
        self.take_profit = take_profit
        self.notional_per_trade = notional_per_trade
        self.max_positions_per_symbol = max_positions_per_symbol
        self.data_dir = data_dir
        self.execution_interval = execution_interval
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # State tracking
        self.prev_ma = {}  # symbol -> previous MA value (for derivative)
        self.price_history = {}  # symbol -> list of (open, close) tuples
        
        # Position metadata: symbol -> list of {entry_price, entry_time, shares, order_id}
        self.position_entries = {}
        
        # Pending orders: order_id -> {symbol, side, shares, limit_price, placed_time, ...}
        self.pending_orders = {}
        
        # Completed trades history
        self.completed_trades = []
        
        # Load state from disk
        self.load_state()
        
        print(f"[MA_STRATEGY] Initialized with {len(symbols)} symbols")
        print(f"[MA_STRATEGY] MA window: {ma_window}h, Buy: -{buy_threshold*100:.1f}%, TP: +{take_profit*100:.1f}%")
        print(f"[MA_STRATEGY] Data dir: {self.data_dir}")
        print(f"[MA_STRATEGY] Loaded {len(self.pending_orders)} pending orders, "
              f"{sum(len(e) for e in self.position_entries.values())} position entries, "
              f"{len(self.completed_trades)} completed trades")
    
    def _get_state_file(self) -> str:
        """Get the path to the state file."""
        return os.path.join(self.data_dir, "state.json")
    
    def save_state(self):
        """Save current state to disk."""
        state = {
            'position_entries': {},
            'pending_orders': self.pending_orders,
            'completed_trades': self.completed_trades,
            'last_saved': datetime.now().isoformat(),
        }
        
        # Convert position_entries (datetime objects need serialization)
        for symbol, entries in self.position_entries.items():
            state['position_entries'][symbol] = []
            for entry in entries:
                entry_copy = entry.copy()
                if isinstance(entry_copy.get('entry_time'), datetime):
                    entry_copy['entry_time'] = entry_copy['entry_time'].isoformat()
                state['position_entries'][symbol].append(entry_copy)
        
        try:
            with open(self._get_state_file(), 'w') as f:
                json.dump(state, f, indent=2)
            print(f"[MA_STRATEGY] State saved to {self._get_state_file()}")
        except Exception as e:
            print(f"[MA_STRATEGY] ERROR saving state: {e}")
    
    def load_state(self):
        """Load state from disk."""
        state_file = self._get_state_file()
        if not os.path.exists(state_file):
            print(f"[MA_STRATEGY] No state file found, starting fresh")
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Restore pending_orders
            self.pending_orders = state.get('pending_orders', {})
            
            # Restore completed_trades
            self.completed_trades = state.get('completed_trades', [])
            
            # Restore position_entries (convert datetime strings back)
            self.position_entries = {}
            for symbol, entries in state.get('position_entries', {}).items():
                self.position_entries[symbol] = []
                for entry in entries:
                    if 'entry_time' in entry and isinstance(entry['entry_time'], str):
                        entry['entry_time'] = datetime.fromisoformat(entry['entry_time'])
                    self.position_entries[symbol].append(entry)
            
            last_saved = state.get('last_saved', 'unknown')
            print(f"[MA_STRATEGY] Loaded state from {state_file} (last saved: {last_saved})")
            
        except Exception as e:
            print(f"[MA_STRATEGY] ERROR loading state: {e}")
    
    def check_pending_orders(self) -> tuple[list, list, list]:
        """
        Check status of pending orders and update position_entries for filled orders.
        
        Returns:
            Tuple of (filled_buy_orders, filled_sell_orders, cancelled_orders)
        """
        filled_buys = []
        filled_sells = []
        cancelled = []
        orders_to_remove = []
        
        for order_id, order_info in list(self.pending_orders.items()):
            # Get current order status from broker
            current_status = get_order_info(order_id)
            
            if current_status is None:
                print(f"[MA_STRATEGY] Could not get status for order {order_id}")
                continue
            
            state = current_status.get('state', '')
            symbol = order_info['symbol']
            side = order_info['side']
            
            if state == 'filled':
                filled_qty = float(current_status.get('cumulative_quantity', current_status.get('quantity', 0)))
                avg_price = float(current_status.get('average_price', order_info['limit_price']))
                
                if side == 'buy':
                    # Add to position_entries
                    if symbol not in self.position_entries:
                        self.position_entries[symbol] = []
                    
                    self.position_entries[symbol].append({
                        'entry_price': avg_price,
                        'entry_time': datetime.now(),
                        'shares': int(filled_qty),
                        'order_id': order_id,
                        'entry_ma': order_info.get('entry_ma'),
                    })
                    
                    filled_buys.append({
                        'order_id': order_id,
                        'symbol': symbol,
                        'shares': int(filled_qty),
                        'avg_price': avg_price,
                    })
                    print(f"[MA_STRATEGY] BUY ORDER FILLED: {symbol} {int(filled_qty)} shares @ ${avg_price:.2f}")
                    
                elif side == 'sell':
                    # Record completed trade
                    entry_info = order_info.get('entry_info', {})
                    entry_price = entry_info.get('entry_price', avg_price)
                    pnl = (avg_price - entry_price) * filled_qty
                    pnl_pct = ((avg_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                    
                    self.completed_trades.append({
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'exit_price': avg_price,
                        'shares': int(filled_qty),
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'entry_time': entry_info.get('entry_time'),
                        'exit_time': datetime.now().isoformat(),
                        'order_id': order_id,
                    })
                    
                    filled_sells.append({
                        'order_id': order_id,
                        'symbol': symbol,
                        'shares': int(filled_qty),
                        'avg_price': avg_price,
                        'pnl': pnl,
                    })
                    print(f"[MA_STRATEGY] SELL ORDER FILLED: {symbol} {int(filled_qty)} shares @ ${avg_price:.2f}, PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                
                orders_to_remove.append(order_id)
                
            elif state in ('cancelled', 'failed', 'rejected'):
                cancelled.append({
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'state': state,
                })
                print(f"[MA_STRATEGY] Order {state.upper()}: {side} {symbol} (order_id: {order_id})")
                orders_to_remove.append(order_id)
        
        # Remove processed orders
        for order_id in orders_to_remove:
            del self.pending_orders[order_id]
        
        # Save state if anything changed
        if orders_to_remove:
            self.save_state()
        
        return filled_buys, filled_sells, cancelled
    
    def fetch_hourly_data(self, current_time: datetime) -> dict[str, pd.DataFrame]:
        """
        Fetch hourly OHLC data for all symbols.
        
        Args:
            current_time: Current datetime
            
        Returns:
            Dict of symbol -> DataFrame with hourly data
        """
        import yfinance as yf
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Need enough data for MA calculation (ma_window + buffer)
        lookback_days = (self.ma_window // 7) + 10  # ~7 trading hours per day
        start_date = current_time - timedelta(days=lookback_days)
        
        def fetch_symbol(symbol):
            try:
                df = yf.download(
                    symbol,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=(current_time + timedelta(days=1)).strftime('%Y-%m-%d'),
                    interval='1h',
                    progress=False
                )
                if df.empty:
                    return symbol, None
                return symbol, df
            except Exception as e:
                print(f"[MA_STRATEGY] Error fetching {symbol}: {e}")
                return symbol, None
        
        results = {}
        with ThreadPoolExecutor(max_workers=min(20, len(self.symbols))) as executor:
            futures = {executor.submit(fetch_symbol, s): s for s in self.symbols}
            for future in as_completed(futures):
                symbol, df = future.result()
                if df is not None and len(df) >= self.ma_window:
                    results[symbol] = df
        
        print(f"[MA_STRATEGY] Fetched data for {len(results)}/{len(self.symbols)} symbols")
        return results
    
    def compute_ma(self, df: pd.DataFrame) -> tuple[float, float]:
        """
        Compute current MA and MA derivative from hourly data.
        
        Uses (open + close) / 2 for MA calculation.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            (current_ma, ma_derivative)
        """
        if len(df) < self.ma_window:
            return np.nan, 0
        
        # Use (open + close) / 2 for MA
        # Handle both multi-level and single-level column names
        try:
            if isinstance(df.columns, pd.MultiIndex):
                open_prices = df['Open'].values.flatten()
                close_prices = df['Close'].values.flatten()
            else:
                open_prices = df['Open'].values
                close_prices = df['Close'].values
        except KeyError:
            # Try lowercase
            open_prices = df['open'].values
            close_prices = df['close'].values
        
        avg_prices = (open_prices + close_prices) / 2
        
        # Compute MA for last ma_window bars
        current_ma = np.mean(avg_prices[-self.ma_window:])
        
        # Compute previous MA (one bar ago)
        if len(avg_prices) > self.ma_window:
            prev_ma = np.mean(avg_prices[-(self.ma_window + 1):-1])
            ma_derivative = current_ma - prev_ma
        else:
            ma_derivative = 0
        
        return current_ma, ma_derivative
    
    def get_current_price(self, df: pd.DataFrame) -> float:
        """Get the most recent close price."""
        try:
            if isinstance(df.columns, pd.MultiIndex):
                return float(df['Close'].values[-1].flatten()[0])
            else:
                return float(df['Close'].values[-1])
        except:
            return float(df['close'].values[-1])
    
    def sync_position_entries(self, open_positions: dict):
        """
        Sync position_entries with actual open positions.
        Remove entries for positions that no longer exist.
        """
        position_symbols = set(open_positions.keys())
        
        # Remove entries for closed positions
        for symbol in list(self.position_entries.keys()):
            if symbol not in position_symbols:
                print(f"[MA_STRATEGY] Position closed: {symbol}, removing {len(self.position_entries[symbol])} entries")
                del self.position_entries[symbol]
    
    def run(self):
        """
        Execute one iteration of the mean reversion strategy.
        """
        current_time = get_current_time()
        print(f"\n[MA_STRATEGY] ===== Run at {current_time} =====")
        
        # Check if market is open
        if not is_market_hours(current_time):
            print(f"[MA_STRATEGY] Market closed, skipping")
            return
        
        # First, check pending orders from previous runs
        print(f"[MA_STRATEGY] Checking {len(self.pending_orders)} pending orders...")
        filled_buys, filled_sells, cancelled = self.check_pending_orders()
        if filled_buys or filled_sells or cancelled:
            print(f"[MA_STRATEGY] Order updates: {len(filled_buys)} buys filled, "
                  f"{len(filled_sells)} sells filled, {len(cancelled)} cancelled")
        
        # Fetch hourly data
        hourly_data = self.fetch_hourly_data(current_time)
        
        if not hourly_data:
            print(f"[MA_STRATEGY] No data available")
            return
        
        # Get current positions and open orders from broker
        open_positions = get_positions()
        open_orders = get_open_orders()
        
        # Sync position entries with actual positions (handle external changes)
        self.sync_position_entries(open_positions)
        
        # Build set of symbols with pending orders (from broker)
        pending_symbols = set(order['symbol'] for order in open_orders)
        # Also include our locally tracked pending orders
        for order_info in self.pending_orders.values():
            pending_symbols.add(order_info['symbol'])
        
        # Track actions this iteration
        new_buy_orders = []
        new_sell_orders = []
        
        for symbol, df in hourly_data.items():
            current_price = self.get_current_price(df)
            current_ma, ma_derivative = self.compute_ma(df)
            
            if np.isnan(current_ma):
                continue
            
            # Calculate deviation from MA
            deviation = (current_price - current_ma) / current_ma
            
            # --- CHECK SELL SIGNALS FIRST ---
            # Sell when: price >= entry + take_profit AND ma_derivative <= 0
            if symbol in self.position_entries and len(self.position_entries[symbol]) > 0:
                if ma_derivative <= 0:  # Only sell when trend is weakening
                    entries_to_sell = []
                    
                    for i, entry in enumerate(self.position_entries[symbol]):
                        entry_price = entry['entry_price']
                        shares = entry['shares']
                        gain = (current_price - entry_price) / entry_price
                        
                        if gain >= self.take_profit:
                            # Place sell order (if no pending order for this symbol)
                            if symbol not in pending_symbols:
                                order_result = open_limit_order(symbol, current_price, shares, "sell")
                                
                                if order_result and 'id' in order_result:
                                    order_id = order_result['id']
                                    
                                    # Track as pending order
                                    self.pending_orders[order_id] = {
                                        'symbol': symbol,
                                        'side': 'sell',
                                        'shares': shares,
                                        'limit_price': current_price,
                                        'placed_time': current_time.isoformat(),
                                        'entry_info': {
                                            'entry_price': entry_price,
                                            'entry_time': entry['entry_time'].isoformat() if isinstance(entry['entry_time'], datetime) else entry['entry_time'],
                                            'shares': shares,
                                        },
                                    }
                                    
                                    entries_to_sell.append(i)
                                    pending_symbols.add(symbol)  # Prevent multiple orders
                                    
                                    new_sell_orders.append({
                                        'symbol': symbol,
                                        'price': current_price,
                                        'shares': shares,
                                        'gain_pct': gain * 100,
                                        'order_id': order_id,
                                    })
                                    print(f"[MA_STRATEGY] SELL ORDER PLACED: {symbol} {shares} shares @ ${current_price:.2f}, "
                                          f"gain: {gain*100:.2f}%, MA deriv: {ma_derivative:.4f}, order_id: {order_id}")
                    
                    # Remove entries that have pending sell orders (in reverse order)
                    for i in reversed(entries_to_sell):
                        self.position_entries[symbol].pop(i)
                    
                    if symbol in self.position_entries and len(self.position_entries[symbol]) == 0:
                        del self.position_entries[symbol]
            
            # --- CHECK BUY SIGNALS ---
            # Buy when price is X% below MA
            if deviation <= -self.buy_threshold:
                # Check position limits (count both confirmed and pending)
                confirmed_positions = len(self.position_entries.get(symbol, []))
                pending_buy_positions = sum(1 for o in self.pending_orders.values() 
                                           if o['symbol'] == symbol and o['side'] == 'buy')
                total_positions = confirmed_positions + pending_buy_positions
                
                if total_positions < self.max_positions_per_symbol:
                    if symbol not in pending_symbols:
                        # Calculate shares
                        shares = max(1, int(self.notional_per_trade / current_price))
                        
                        # Calculate order cost and check buying power
                        buy_price = current_price * 0.999
                        order_cost = buy_price * shares
                        buying_power = get_buying_power()
                        
                        if buying_power < order_cost:
                            print(f"[MA_STRATEGY] SKIP BUY {symbol}: Insufficient buying power "
                                  f"(${buying_power:.2f} < ${order_cost:.2f} needed)")
                            continue
                        
                        # Place buy order
                        order_result = open_limit_order(symbol, buy_price, shares, "buy")
                        
                        if order_result and 'id' in order_result:
                            order_id = order_result['id']
                            
                            # Track as pending order (NOT in position_entries yet)
                            self.pending_orders[order_id] = {
                                'symbol': symbol,
                                'side': 'buy',
                                'shares': shares,
                                'limit_price': current_price,
                                'placed_time': current_time.isoformat(),
                                'entry_ma': current_ma,
                                'deviation': deviation,
                            }
                            
                            pending_symbols.add(symbol)  # Prevent multiple orders
                            
                            new_buy_orders.append({
                                'symbol': symbol,
                                'price': current_price,
                                'shares': shares,
                                'deviation_pct': deviation * 100,
                                'order_id': order_id,
                            })
                            print(f"[MA_STRATEGY] BUY ORDER PLACED: {symbol} {shares} shares @ ${current_price:.2f}, "
                                  f"deviation: {deviation*100:.2f}%, MA: ${current_ma:.2f}, order_id: {order_id}")
        
        # Save state after processing
        self.save_state()
        
        # Summary
        total_entries = sum(len(entries) for entries in self.position_entries.values())
        print(f"\n[MA_STRATEGY] Summary: {len(new_buy_orders)} buy orders, {len(new_sell_orders)} sell orders placed")
        print(f"[MA_STRATEGY] Confirmed positions: {total_entries} across {len(self.position_entries)} symbols")
        print(f"[MA_STRATEGY] Pending orders: {len(self.pending_orders)}")
        print(f"[MA_STRATEGY] Completed trades: {len(self.completed_trades)}")
    
    def get_trade_stats(self) -> dict:
        """
        Get statistics on completed trades.
        
        Returns:
            Dictionary with trade statistics
        """
        if not self.completed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'avg_pnl_pct': 0,
            }
        
        total = len(self.completed_trades)
        winning = sum(1 for t in self.completed_trades if t.get('pnl', 0) > 0)
        losing = total - winning
        total_pnl = sum(t.get('pnl', 0) for t in self.completed_trades)
        avg_pnl = total_pnl / total
        avg_pnl_pct = sum(t.get('pnl_pct', 0) for t in self.completed_trades) / total
        
        return {
            'total_trades': total,
            'winning_trades': winning,
            'losing_trades': losing,
            'win_rate': winning / total * 100 if total > 0 else 0,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_pnl_pct': avg_pnl_pct,
        }
    
    def print_status(self):
        """Print current strategy status."""
        print("\n" + "=" * 60)
        print("MEAN REVERSION STRATEGY STATUS")
        print("=" * 60)
        
        # Pending orders
        print(f"\nPending Orders ({len(self.pending_orders)}):")
        for order_id, info in self.pending_orders.items():
            print(f"  {info['side'].upper()} {info['symbol']}: {info['shares']} shares @ ${info['limit_price']:.2f}")
            print(f"    Order ID: {order_id}, Placed: {info['placed_time']}")
        
        # Open positions
        total_entries = sum(len(e) for e in self.position_entries.values())
        print(f"\nOpen Positions ({total_entries} entries across {len(self.position_entries)} symbols):")
        for symbol, entries in self.position_entries.items():
            for i, entry in enumerate(entries):
                print(f"  {symbol} #{i+1}: {entry['shares']} shares @ ${entry['entry_price']:.2f}")
                print(f"    Entry: {entry.get('entry_time', 'unknown')}")
        
        # Trade stats
        stats = self.get_trade_stats()
        print(f"\nCompleted Trades ({stats['total_trades']}):")
        print(f"  Win Rate: {stats['win_rate']:.1f}% ({stats['winning_trades']}/{stats['total_trades']})")
        print(f"  Total P&L: ${stats['total_pnl']:.2f}")
        print(f"  Avg P&L: ${stats['avg_pnl']:.2f} ({stats['avg_pnl_pct']:+.2f}%)")
        
        print("=" * 60)
    
    def cancel_pending_orders(self):
        """Cancel all pending orders tracked by this strategy."""
        cancelled = 0
        for order_id in list(self.pending_orders.keys()):
            try:
                cancel_order(order_id)
                del self.pending_orders[order_id]
                cancelled += 1
            except Exception as e:
                print(f"[MA_STRATEGY] Error cancelling order {order_id}: {e}")
        
        if cancelled > 0:
            self.save_state()
            print(f"[MA_STRATEGY] Cancelled {cancelled} pending orders")


class TradingBot:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.account = account
        self.strategy = MeanReversionStrategy()

    def start(self):
        self.account.init(self.config_path)
        while True:
            self.account.login()
            self.strategy.run()
            time.sleep(self.strategy.execution_interval)


if __name__ == "__main__":
    config_path = "config.json"
    bot = TradingBot(config_path)
    bot.start()
