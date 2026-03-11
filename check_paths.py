
import os
path = r"C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe"
print(f"Path: {path}")
print(f"Exists: {os.path.exists(path)}")

alt_path = r"C:\MetaTrader 5 Terminal\terminal64.exe"
print(f"Alt Path: {alt_path}")
print(f"Exists: {os.path.exists(alt_path)}")

# Also check for other MT5 installations
prog_files = r"C:\Program Files"
if os.path.exists(prog_files):
    print(f"\nSearching in {prog_files}...")
    for item in os.listdir(prog_files):
        if "MetaTrader" in item:
            print(f"Found: {item}")
