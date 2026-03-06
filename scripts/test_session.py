import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from datetime import datetime, timedelta
from xp3_forex.strategies.session_analyzer import get_active_session_name

print("Testing Every Hour:")
base = datetime.strptime("2026-03-05 00:00:00", "%Y-%m-%d %H:%M:%S")
for i in range(24):
    test_time = base + timedelta(hours=i)
    session = get_active_session_name(test_time)
    print(f"{test_time.strftime('%H:%M')} -> {session}")
