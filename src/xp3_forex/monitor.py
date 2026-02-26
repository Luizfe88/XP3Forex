#!/usr/bin/env python3
"""
XP3 PRO FOREX - Simple Monitor

Real-time monitoring of trading signals and bot status.
"""

import sys
import os
import time
import signal
import re
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from xp3_forex.config.settings import settings

def start_monitor():
    """Start the monitor process"""
    try:
        monitor = SimpleMonitor(log_file=settings.get_log_file())
        monitor.run()
    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")

class SimpleMonitor:
    """Simple monitor for XP3 PRO FOREX"""
    
    def __init__(self, log_file: Path = None):
        self.log_file = log_file or settings.get_log_file()
        self.running = True
        self.last_position = 0
        self.signal_buffer = deque(maxlen=100)
        
        # Signal patterns
        self.patterns = {
            'signal': re.compile(r'(BUY|SELL).*?(@|price).*?(\d+\.\d+)'),
            'veto': re.compile(r'(VETO|MOTIVO|REJEITADO)', re.IGNORECASE),
            'adx': re.compile(r'ADX.*?[:\s](\d+\.?\d*)', re.IGNORECASE),
            'rsi': re.compile(r'RSI.*?[:\s](\d+\.?\d*)', re.IGNORECASE),
            'ema': re.compile(r'EMA.*?[:\s](\d+\.?\d*)', re.IGNORECASE),
            'profit': re.compile(r'(profit|lucro|ganho).*?(\d+\.?\d*)', re.IGNORECASE),
            'loss': re.compile(r'(loss|perda|prejuízo).*?(\d+\.?\d*)', re.IGNORECASE),
        }
        
        # Colors for terminal output
        self.colors = {
            'buy': '\033[92m',      # Green
            'sell': '\033[91m',     # Red
            'veto': '\033[93m',     # Yellow
            'profit': '\033[92m',   # Green
            'loss': '\033[91m',     # Red
            'info': '\033[94m',     # Blue
            'reset': '\033[0m'      # Reset
        }
        
        # Register signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Monitor stopped by signal {signum}")
        self.running = False
    
    def colorize(self, text: str, color: str) -> str:
        """Add color to text"""
        return f"{self.colors.get(color, '')}{text}{self.colors['reset']}"
    
    def analyze_log_line(self, line: str) -> Dict[str, any]:
        """Analyze a log line for trading signals"""
        result = {
            'type': 'info',
            'text': line.strip(),
            'signals': {},
            'highlight': False
        }
        
        # Check for BUY/SELL signals
        signal_match = self.patterns['signal'].search(line)
        if signal_match:
            order_type = signal_match.group(1)
            price = signal_match.group(3)
            result['type'] = 'signal'
            result['signals']['order_type'] = order_type
            result['signals']['price'] = float(price)
            result['highlight'] = True
        
        # Check for vetos
        veto_match = self.patterns['veto'].search(line)
        if veto_match:
            result['type'] = 'veto'
            result['highlight'] = True
        
        # Check for technical indicators
        adx_match = self.patterns['adx'].search(line)
        if adx_match:
            result['signals']['adx'] = float(adx_match.group(1))
        
        rsi_match = self.patterns['rsi'].search(line)
        if rsi_match:
            result['signals']['rsi'] = float(rsi_match.group(1))
        
        ema_match = self.patterns['ema'].search(line)
        if ema_match:
            result['signals']['ema'] = float(ema_match.group(1))
        
        # Check for profit/loss
        profit_match = self.patterns['profit'].search(line)
        if profit_match:
            result['type'] = 'profit'
            result['signals']['profit'] = float(profit_match.group(2))
            result['highlight'] = True
        
        loss_match = self.patterns['loss'].search(line)
        if loss_match:
            result['type'] = 'loss'
            result['signals']['loss'] = float(loss_match.group(2))
            result['highlight'] = True
        
        return result
    
    def format_output(self, analysis: Dict[str, any]) -> str:
        """Format the output for display"""
        text = analysis['text']
        
        if analysis['highlight']:
            if analysis['type'] == 'signal':
                order_type = analysis['signals'].get('order_type', '')
                price = analysis['signals'].get('price', 0)
                color = 'buy' if order_type == 'BUY' else 'sell'
                return self.colorize(f"🎯 {order_type} @ {price}", color)
            
            elif analysis['type'] == 'veto':
                return self.colorize(f"🚫 VETO: {text}", 'veto')
            
            elif analysis['type'] == 'profit':
                profit = analysis['signals'].get('profit', 0)
                return self.colorize(f"💰 PROFIT: {profit}", 'profit')
            
            elif analysis['type'] == 'loss':
                loss = analysis['signals'].get('loss', 0)
                return self.colorize(f"📉 LOSS: {loss}", 'loss')
        
        # Add technical indicators
        signals = analysis['signals']
        if signals:
            indicators = []
            if 'adx' in signals:
                indicators.append(f"ADX:{signals['adx']:.1f}")
            if 'rsi' in signals:
                indicators.append(f"RSI:{signals['rsi']:.1f}")
            if 'ema' in signals:
                indicators.append(f"EMA:{signals['ema']:.1f}")
            
            if indicators:
                text += f" [{', '.join(indicators)}]"
        
        return text
    
    def run(self):
        """Run the monitor"""
        print("🚀 XP3 PRO FOREX Monitor")
        print("=" * 50)
        print(f"📡 Reading logs from: {self.log_file}")
        print("🛑 Press Ctrl+C to exit")
        print("=" * 50)
        
        while self.running:
            try:
                # Check if log file exists
                if not Path(self.log_file).exists():
                    print(f"⏳ Waiting for log file: {self.log_file}")
                    time.sleep(5)
                    continue
                
                # Read new log entries
                with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    # Move to last position
                    f.seek(self.last_position)
                    
                    # Read new lines
                    new_lines = f.readlines()
                    self.last_position = f.tell()
                
                # Process new lines
                for line in new_lines:
                    analysis = self.analyze_log_line(line)
                    output = self.format_output(analysis)
                    
                    if analysis['highlight']:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] {output}")
                    
                    # Store signals
                    if analysis['type'] in ['signal', 'veto']:
                        self.signal_buffer.append({
                            'timestamp': datetime.now(),
                            'type': analysis['type'],
                            'data': analysis['signals']
                        })
                
                # Small delay to avoid high CPU usage
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(1)
        
        print(f"\n📊 Summary:")
        print(f"   Signals captured: {len(self.signal_buffer)}")
        print(f"   Monitor stopped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="XP3 PRO FOREX Simple Monitor")
    parser.add_argument(
        "--log-file", "-l",
        type=str,
        help="Path to log file"
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="XP3 Monitor v4.2.0"
    )
    
    args = parser.parse_args()
    
    try:
        monitor = SimpleMonitor(log_file=args.log_file)
        monitor.run()
    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())