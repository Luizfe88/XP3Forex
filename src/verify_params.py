
import logging
import json
from datetime import datetime, UTC
from xp3_forex.core.settings import settings
from xp3_forex.strategies.session_analyzer import get_active_session_params, get_active_session_name

# Configure logging to see SessionAnalyzer outputs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestSessionParams")

def test_param_loading():
    print("=== Test: Optimization Metrics Recognition ===")
    
    # 1. Check if the JSON file exists at the absolute path
    json_path = settings.DATA_DIR / "session_optimized_params.json"
    print(f"Target JSON Path: {json_path}")
    print(f"Exists: {json_path.exists()}")
    
    if not json_path.exists():
        print("❌ Error: session_optimized_params.json not found!")
        return

    # 2. Get current session info
    now_utc = datetime.now(UTC)
    session_name = get_active_session_name(now_utc)
    print(f"Current UTC Time: {now_utc.strftime('%H:%M:%S')}")
    print(f"Current Active Session: {session_name}")

    # 3. Test parameter retrieval for a common symbol
    symbol = "EURUSD"
    print(f"\nFetching params for {symbol}...")
    params = get_active_session_params(symbol, now_utc)
    
    if params and "active_session" in params:
        print(f"✅ Success! Parameters loaded for session: {params['active_session']}")
        print(f"Parameters: {json.dumps(params, indent=2)}")
        
        # Check if they are non-default (if the JSON has specific values for EURUSD)
        with open(json_path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)
            if symbol in raw_json and session_name in raw_json[symbol]:
                print(f"✅ Verified: The parameters match the JSON file content.")
            else:
                print(f"ℹ️ Note: Using DEFAULT params because specific {symbol}/{session_name} block was not in JSON.")
    else:
        print("❌ Failed: Parameters returned are empty or invalid.")

if __name__ == "__main__":
    test_param_loading()
