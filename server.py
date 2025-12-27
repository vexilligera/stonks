#!/usr/bin/env python3
"""
Simple HTTP server to control the trading bot via a web interface.
"""

import http.server
import json
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from http.server import HTTPServer
from urllib.parse import parse_qs, urlparse

# Bot process management
bot_process = None
bot_lock = threading.Lock()
bot_start_time = None
bot_logs = []
MAX_LOGS = 100


def get_bot_status():
    """Get the current status of the trading bot."""
    global bot_process, bot_start_time
    with bot_lock:
        if bot_process is None:
            return {"running": False, "pid": None, "uptime": None}
        
        # Check if process is still running
        poll = bot_process.poll()
        if poll is not None:
            # Process has terminated
            bot_process = None
            bot_start_time = None
            return {"running": False, "pid": None, "uptime": None}
        
        uptime = None
        if bot_start_time:
            uptime = str(datetime.now() - bot_start_time).split('.')[0]
        
        return {
            "running": True,
            "pid": bot_process.pid,
            "uptime": uptime
        }


def start_bot():
    """Start the trading bot as a subprocess."""
    global bot_process, bot_start_time, bot_logs
    with bot_lock:
        if bot_process is not None and bot_process.poll() is None:
            return {"success": False, "message": "Bot is already running"}
        
        try:
            # Start the bot as a subprocess
            # Use -u flag for unbuffered output so logs appear in real-time
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            bot_process = subprocess.Popen(
                [sys.executable, "-u", "main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=env,
                text=True,
                bufsize=1
            )
            bot_start_time = datetime.now()
            bot_logs = []
            
            # Start a thread to read logs
            log_thread = threading.Thread(target=read_bot_logs, daemon=True)
            log_thread.start()
            
            return {"success": True, "message": f"Bot started with PID {bot_process.pid}"}
        except Exception as e:
            return {"success": False, "message": str(e)}


def read_bot_logs():
    """Read logs from the bot subprocess."""
    global bot_process, bot_logs
    try:
        while bot_process and bot_process.stdout:
            line = bot_process.stdout.readline()
            if not line:
                break
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {line.rstrip()}"
            bot_logs.append(log_entry)
            if len(bot_logs) > MAX_LOGS:
                bot_logs = bot_logs[-MAX_LOGS:]
    except:
        pass


def stop_bot():
    """Stop the trading bot subprocess."""
    global bot_process, bot_start_time
    with bot_lock:
        if bot_process is None:
            return {"success": False, "message": "Bot is not running"}
        
        try:
            bot_process.terminate()
            bot_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            bot_process.kill()
            bot_process.wait()
        
        pid = bot_process.pid
        bot_process = None
        bot_start_time = None
        return {"success": True, "message": f"Bot stopped (PID {pid})"}


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def get_config():
    """Read the config.json file."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return {"success": True, "config": json.load(f)}
    except FileNotFoundError:
        return {"success": False, "message": "Config file not found"}
    except json.JSONDecodeError as e:
        return {"success": False, "message": f"Invalid JSON: {e}"}


def save_config(config_data):
    """Save config to config.json file."""
    try:
        # Validate the config structure for Mean Reversion Strategy
        required_fields = ['username', 'password', 'symbols']
        for field in required_fields:
            if field not in config_data:
                return {"success": False, "message": f"Missing required field: {field}"}
        
        # Ensure correct types
        if not isinstance(config_data.get('symbols'), list):
            return {"success": False, "message": "symbols must be a list"}
        
        # Validate numeric fields
        numeric_fields = {
            'ma_window': int,
            'buy_threshold': float,
            'take_profit': float,
            'notional_per_trade': float,
            'max_positions_per_symbol': int,
        }
        for field, field_type in numeric_fields.items():
            if field in config_data:
                try:
                    config_data[field] = field_type(config_data[field])
                except (ValueError, TypeError):
                    return {"success": False, "message": f"{field} must be a {field_type.__name__}"}
        
        # Set defaults if not provided
        config_data.setdefault('broker', 'robinhood')
        config_data.setdefault('data_dir', './ma_strategy_data')
        config_data.setdefault('ma_window', 40)
        config_data.setdefault('buy_threshold', 0.03)
        config_data.setdefault('take_profit', 0.05)
        config_data.setdefault('notional_per_trade', 500.0)
        config_data.setdefault('max_positions_per_symbol', 10)
        
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        return {"success": True, "message": "Config saved successfully"}
    except Exception as e:
        return {"success": False, "message": str(e)}


class TradingBotHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for the trading bot control interface."""
    
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == "/" or parsed.path == "/index.html":
            self.serve_file("index.html", "text/html")
        elif parsed.path == "/api/status":
            self.send_json(get_bot_status())
        elif parsed.path == "/api/logs":
            self.send_json({"logs": bot_logs[-50:]})
        elif parsed.path == "/api/config":
            self.send_json(get_config())
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        parsed = urlparse(self.path)
        
        if parsed.path == "/api/start":
            result = start_bot()
            self.send_json(result)
        elif parsed.path == "/api/stop":
            result = stop_bot()
            self.send_json(result)
        elif parsed.path == "/api/config":
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8')
                config_data = json.loads(body)
                result = save_config(config_data)
                self.send_json(result)
            except json.JSONDecodeError:
                self.send_json({"success": False, "message": "Invalid JSON"})
            except Exception as e:
                self.send_json({"success": False, "message": str(e)})
        else:
            self.send_error(404, "Not Found")
    
    def serve_file(self, filename, content_type):
        try:
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
            with open(filepath, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404, "File not found")
    
    def send_json(self, data):
        content = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(content))
        self.end_headers()
        self.wfile.write(content)
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass


def run_server(port=8080):
    """Run the HTTP server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, TradingBotHandler)
    httpd.allow_reuse_address = True
    print(f"🚀 Trading Bot Control Panel running at http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    shutdown_event = threading.Event()
    
    def signal_handler(sig, frame):
        print("\nShutting down server...")
        shutdown_event.set()
        # Shutdown must be called from a different thread
        threading.Thread(target=httpd.shutdown, daemon=True).start()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_bot()
        httpd.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trading Bot Control Server")
    parser.add_argument("--port", type=int, default=5188, help="Port to run the server on")
    args = parser.parse_args()
    run_server(args.port)

