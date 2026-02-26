import json
import sys
from datetime import datetime

import asset_selection as selector


def main():
    date_str = None
    if len(sys.argv) >= 2:
        date_str = sys.argv[1]
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    result = selector.run_daily_update(date_str)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

