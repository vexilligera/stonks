from datetime import datetime
from typing import List, Optional
import pytz
import pandas as pd
import yfinance as yf
import json
import robin_stocks.robinhood as rh

ET = pytz.timezone('US/Eastern')


class Account:
    def __init__(self, backtest: bool = False):
        self.cash = 0
        self.backtest = backtest
        self.broker = ""
        self.username = ""
        self.password = ""

    def init(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.broker = self.config["broker"]
        self.username = self.config["username"]
        self.password = self.config["password"]

    def login(self, pickle_path: str = "./ma_strategy_data"):
        if self.broker == "robinhood":
            rh.login(
                self.username, self.password, expiresIn=864000, pickle_path=pickle_path,
            )
            print(f"[INFO] Logged in to Robinhood")
        else:
            raise ValueError(f"Unsupported broker: {self.broker}")

    def set_cash(self, cash: float):
        self.cash = cash

account = Account()


def get_current_time() -> datetime:
    """Get current time in Eastern timezone."""
    return datetime.now(ET)


def get_sp500_symbols() -> List[str]:
    """
    Fetch S&P 500 stock symbols from Wikipedia.
    Falls back to hardcoded list if fetch fails.
    """
    try:
        import requests
        from io import StringIO
        
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        tables = pd.read_html(StringIO(response.text))
        symbols = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        symbols = [s.strip() for s in symbols if s and isinstance(s, str)]
        print(f"[INFO] Fetched {len(symbols)} S&P 500 symbols")
        return sorted(symbols)
    except Exception as e:
        print(f"[WARN] Failed to fetch S&P 500: {e}")
        return _get_sp500_fallback()


def _get_sp500_fallback() -> List[str]:
    """Fallback hardcoded S&P 500 symbols."""
    return [
        "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AIG",
        "AMAT", "AMD", "AMGN", "AMZN", "AVGO", "AXP", "BA", "BAC", "BK", "BKNG",
        "BLK", "BMY", "C", "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST",
        "CRM", "CSCO", "CVS", "CVX", "DE", "DHR", "DIS", "DOW", "DUK", "EMR",
        "EXC", "F", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS",
        "HD", "HON", "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KO", "LIN",
        "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "META", "MMM",
        "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NOW", "NVDA", "ORCL",
        "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO",
        "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS",
        "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM",
    ]


def is_market_hours(dt: datetime = None) -> bool:
    """
    Check if given datetime is during regular market hours.
    
    Market hours: 9:30 AM - 4:00 PM ET, weekdays only.
    
    Args:
        dt: datetime to check (default: current time)
        
    Returns:
        True if during market hours
    """
    if dt is None:
        dt = get_current_time()
    
    if dt.tzinfo is None:
        dt = ET.localize(dt)
    else:
        dt = dt.astimezone(ET)
    
    # Check if weekday (0=Monday, 6=Sunday)
    if dt.weekday() >= 5:
        return False
    
    # Check time
    market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= dt < market_close


def get_prices_by_time_range(
    symbol: str,
    start_datetime: datetime,
    end_datetime: Optional[datetime] = None,
    interval: str = "5m",
) -> pd.DataFrame:
    """
    Get prices for a symbol within a time range from yfinance (live data).
    
    Args:
        symbol: Stock ticker symbol
        start_datetime: Start datetime. If naive, assumes Eastern Time.
        end_datetime: End datetime (inclusive). If None, returns only the bar at start_datetime.
        interval: Data interval (e.g., "5m", "1m", "1h")
        
    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume
        Returns empty DataFrame if no data found.
    """
    # Handle timezone
    if start_datetime.tzinfo is None:
        start_datetime = ET.localize(start_datetime)
    else:
        start_datetime = start_datetime.astimezone(ET)
    
    if end_datetime is None:
        end_datetime = start_datetime
    elif end_datetime.tzinfo is None:
        end_datetime = ET.localize(end_datetime)
    else:
        end_datetime = end_datetime.astimezone(ET)
    
    # Calculate period needed - yfinance uses period for recent data
    # Add buffer to ensure we have enough data
    now = datetime.now(ET)
    days_back = (now - start_datetime).days + 2  # Add buffer
    
    # yfinance 5m data limited to ~60 days
    if days_back > 59:
        days_back = 59
    
    period = f"{days_back}d"
    
    try:
        # Download data from yfinance
        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            prepost=False,
            progress=False,
        )
        
        if df.empty:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        
        # Handle MultiIndex columns (yfinance returns this even for single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten by taking just the first level (Price names)
            df.columns = [col[0] for col in df.columns]
        
        # Reset index to get datetime as column
        df = df.reset_index()
        
        # Rename columns to lowercase
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        
        # Ensure datetime column name is correct
        if "date" in df.columns and "datetime" not in df.columns:
            df = df.rename(columns={"date": "datetime"})
        
        # Convert datetime to Eastern Time
        if df["datetime"].dt.tz is None:
            df["datetime"] = df["datetime"].dt.tz_localize("UTC")
        df["datetime"] = df["datetime"].dt.tz_convert(ET)
        
        # Filter to time range
        if end_datetime == start_datetime:
            # Single bar - find the exact or nearest bar
            mask = df["datetime"] == start_datetime
            if not mask.any():
                # Find the closest bar at or before start_datetime
                before_mask = df["datetime"] <= start_datetime
                if before_mask.any():
                    df = df[before_mask].tail(1)
                else:
                    return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
            else:
                df = df[mask]
        else:
            # Range query
            mask = (df["datetime"] >= start_datetime) & (df["datetime"] <= end_datetime)
            df = df[mask]
        
        # Select and order columns
        result = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
        result = result.sort_values("datetime").reset_index(drop=True)
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch {symbol}: {e}")
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])


def get_price_at_time(
    symbol: str,
    dt: datetime,
    price_type: str = "close"
) -> Optional[float]:
    """
    Get a single price for a symbol at a specific time from yfinance.
    
    Args:
        symbol: Stock ticker symbol
        dt: datetime to get price for
        price_type: "open", "high", "low", "close", or "volume"
        
    Returns:
        Price value or None if not found
    """
    df = get_prices_by_time_range(symbol, dt, dt, "5m")
    
    if df.empty:
        return None
    
    if price_type not in df.columns:
        return None
    
    return float(df.iloc[0][price_type])


def list_all_symbols():
    # TODO: list all symbols from nyse
    pass


def open_limit_order(symbol: str, price: float, quantity: int, side: str = "buy") -> Optional[dict]:
    """
    Place a limit order.
    
    Args:
        symbol: Stock ticker symbol
        price: Limit price
        quantity: Number of shares
        side: "buy" or "sell"
        
    Returns:
        Order response dict with 'id' field, or None if failed
    """
    if account.broker == "robinhood":
        try:
            if side == "buy":
                result = rh.orders.order_buy_limit(
                    symbol=symbol,
                    quantity=quantity,
                    limitPrice=price,
                    timeInForce="gfd",  # Good for day
                )
            elif side == "sell":
                result = rh.orders.order_sell_limit(
                    symbol=symbol,
                    quantity=quantity,
                    limitPrice=price,
                    timeInForce="gfd",
                )
            else:
                raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")
            
            if result and "id" in result:
                print(f"[ORDER] {side.upper()} {quantity} {symbol} @ ${price:.2f} (id: {result['id'][:8]}...)")
                return result
            else:
                print(f"[ERROR] Order failed: {result}")
                return None
        except Exception as e:
            print(f"[ERROR] Failed to place order: {e}")
            return None
    else:
        raise ValueError(f"Unsupported broker: {account.broker}")


def open_market_order(symbol: str, quantity: int, side: str = "buy") -> Optional[dict]:
    """
    Place a market order.
    
    Args:
        symbol: Stock ticker symbol
        quantity: Number of shares
        side: "buy" or "sell"
        
    Returns:
        Order response dict with 'id' field, or None if failed
    """
    if account.broker == "robinhood":
        try:
            if side == "buy":
                result = rh.orders.order_buy_market(
                    symbol=symbol,
                    quantity=quantity,
                    timeInForce="gfd",
                )
            elif side == "sell":
                result = rh.orders.order_sell_market(
                    symbol=symbol,
                    quantity=quantity,
                    timeInForce="gfd",
                )
            else:
                raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")
            
            if result and "id" in result:
                print(f"[ORDER] {side.upper()} {quantity} {symbol} @ MARKET (id: {result['id'][:8]}...)")
                return result
            else:
                print(f"[ERROR] Order failed: {result}")
                return None
        except Exception as e:
            print(f"[ERROR] Failed to place order: {e}")
            return None
    else:
        raise ValueError(f"Unsupported broker: {account.broker}")


def cancel_order(order_id: str) -> bool:
    """
    Cancel an order by ID.
    
    Args:
        order_id: Order ID
        
    Returns:
        True if cancelled successfully
    """
    if account.broker == "robinhood":
        try:
            result = rh.orders.cancel_stock_order(order_id)
            if result:
                print(f"[CANCEL] Order {order_id[:8]}... cancelled")
                return True
            return False
        except Exception as e:
            print(f"[ERROR] Failed to cancel order: {e}")
            return False
    else:
        raise ValueError(f"Unsupported broker: {account.broker}")


def cancel_all_orders() -> int:
    """
    Cancel all open orders.
    
    Returns:
        Number of orders cancelled
    """
    if account.broker == "robinhood":
        try:
            result = rh.orders.cancel_all_stock_orders()
            count = len(result) if result else 0
            print(f"[CANCEL] Cancelled {count} orders")
            return count
        except Exception as e:
            print(f"[ERROR] Failed to cancel orders: {e}")
            return 0
    else:
        raise ValueError(f"Unsupported broker: {account.broker}")


def get_open_orders() -> List[dict]:
    """
    Get all open orders.
    
    Returns:
        List of order dicts
    """
    if account.broker == "robinhood":
        try:
            orders = rh.orders.get_all_open_stock_orders()
            return orders if orders else []
        except Exception as e:
            print(f"[ERROR] Failed to get open orders: {e}")
            return []
    else:
        raise ValueError(f"Unsupported broker: {account.broker}")


def get_order_info(order_id: str) -> Optional[dict]:
    """
    Get information for a single order by ID.
    
    Args:
        order_id: The order ID from a placed order
        
    Returns:
        Order info dict with keys like 'state', 'filled_quantity', 'average_price', etc.
        States: 'queued', 'unconfirmed', 'confirmed', 'partially_filled', 'filled', 
                'cancelled', 'pending_cancel', 'failed', 'rejected'
    """
    if account.broker == "robinhood":
        try:
            order_info = rh.orders.get_stock_order_info(order_id)
            return order_info if order_info else None
        except Exception as e:
            print(f"[ERROR] Failed to get order info for {order_id}: {e}")
            return None
    else:
        raise ValueError(f"Unsupported broker: {account.broker}")


def get_positions() -> dict[str, dict]:
    """
    Get current stock positions.
    
    Returns:
        Dictionary of symbol -> position info dict with keys:
        - quantity: float (number of shares)
        - average_buy_price: float
        - equity: float (current value)
        - percent_change: float
        - equity_change: float
    """
    if account.broker == "robinhood":
        try:
            holdings = rh.account.build_holdings()
            return holdings if holdings else {}
        except Exception as e:
            print(f"[ERROR] Failed to get positions: {e}")
            return {}
    else:
        raise ValueError(f"Unsupported broker: {account.broker}")


def get_position(symbol: str) -> Optional[dict]:
    """
    Get position for a specific symbol.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Position dict or None if no position
    """
    positions = get_positions()
    return positions.get(symbol)


def get_account_info() -> dict:
    """
    Get account information including buying power.
    
    Returns:
        Dict with keys: buying_power, cash, portfolio_value
    """
    if account.broker == "robinhood":
        try:
            profile = rh.profiles.load_account_profile()
            portfolio = rh.profiles.load_portfolio_profile()
            
            return {
                "buying_power": float(profile.get("buying_power", 0)),
                "cash": float(profile.get("cash", 0)),
                "portfolio_value": float(portfolio.get("equity", 0)),
                "extended_hours_equity": float(portfolio.get("extended_hours_equity", 0) or 0),
            }
        except Exception as e:
            print(f"[ERROR] Failed to get account info: {e}")
            return {"buying_power": 0, "cash": 0, "portfolio_value": 0}
    else:
        raise ValueError(f"Unsupported broker: {account.broker}")


def get_buying_power() -> float:
    """Get current buying power."""
    info = get_account_info()
    return info.get("buying_power", 0)


def get_quote(symbol: str) -> Optional[dict]:
    """
    Get current quote for a symbol.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Dict with keys: price, bid, ask, volume, etc.
    """
    if account.broker == "robinhood":
        try:
            quote = rh.stocks.get_stock_quote_by_symbol(symbol)
            if quote:
                return {
                    "symbol": symbol,
                    "price": float(quote.get("last_trade_price", 0)),
                    "bid": float(quote.get("bid_price", 0) or 0),
                    "ask": float(quote.get("ask_price", 0) or 0),
                    "bid_size": int(quote.get("bid_size", 0) or 0),
                    "ask_size": int(quote.get("ask_size", 0) or 0),
                    "volume": int(quote.get("volume", 0) or 0),
                    "previous_close": float(quote.get("previous_close", 0) or 0),
                }
            return None
        except Exception as e:
            print(f"[ERROR] Failed to get quote for {symbol}: {e}")
            return None
    else:
        raise ValueError(f"Unsupported broker: {account.broker}")


def get_last_price(symbol: str) -> Optional[float]:
    """
    Get last trade price for a symbol.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Last trade price or None
    """
    if account.broker == "robinhood":
        try:
            prices = rh.stocks.get_latest_price(symbol)
            if prices and prices[0]:
                return float(prices[0])
            return None
        except Exception as e:
            print(f"[ERROR] Failed to get price for {symbol}: {e}")
            return None
    else:
        raise ValueError(f"Unsupported broker: {account.broker}")


def get_all_symbols(date: Optional[datetime] = None) -> List[str]:
    """
    Get all available symbols from online sources.
    Fetches symbols from NASDAQ and NYSE exchanges.
    In backtest mode, this will be patched to return symbols from dataset.
    
    Args:
        date: Ignored in live mode
        
    Returns:
        List of symbol strings
    """
    try:
        import requests
        from io import StringIO
        
        symbols = set()
        
        # Fetch NASDAQ symbols
        try:
            nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
            nasdaq_response = requests.get(nasdaq_url, timeout=10)
            if nasdaq_response.status_code == 200:
                nasdaq_df = pd.read_csv(StringIO(nasdaq_response.text), sep='|')
                # Filter for stocks (not ETFs, not test symbols)
                nasdaq_stocks = nasdaq_df[
                    (nasdaq_df['ETF'] == 'N') & 
                    (nasdaq_df['Test Issue'] == 'N') &
                    (~nasdaq_df['Symbol'].str.contains(r'[\$\^]', regex=True, na=False))
                ]
                symbols.update(nasdaq_stocks['Symbol'].tolist())
                print(f"[INFO] Fetched {len(nasdaq_stocks)} NASDAQ symbols")
        except Exception as e:
            print(f"[WARN] Failed to fetch NASDAQ symbols: {e}")
        
        # Fetch NYSE symbols (alternative method using FTP data)
        try:
            # Use the same NASDAQ trader file which includes NYSE
            # The 'NASDAQ Traded' file actually includes all exchanges
            pass  # Already included above
        except Exception as e:
            print(f"[WARN] Failed to fetch NYSE symbols: {e}")
        
        # If we got symbols, return sorted list
        if symbols:
            symbol_list = sorted(list(symbols))
            print(f"[INFO] Total unique symbols fetched: {len(symbol_list)}")
            return symbol_list
        
        # Fallback to S&P 500 if online fetch fails
        print("[WARN] Online symbol fetch failed, falling back to S&P 500")
        return get_sp500_symbols()
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch symbols online: {e}")
        # Fallback to S&P 500
        return get_sp500_symbols()