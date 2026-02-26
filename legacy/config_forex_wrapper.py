"""
XP3 PRO FOREX - Config Wrapper (Legacy Compatibility)

This module provides backward compatibility for code that imports config_forex
while using the new src-layout architecture.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    # Try to import from new structure
    from xp3_forex.core.config import *
    print("✅ Config wrapper - usando nova arquitetura src-layout")
except ImportError as e:
    print(f"⚠️ Falha ao importar nova config: {e}")
    # Fallback to legacy config
    try:
        import config_forex as _legacy_config
        # Re-export all variables from legacy config
        for attr in dir(_legacy_config):
            if not attr.startswith('_'):
                globals()[attr] = getattr(_legacy_config, attr)
        print("⚠️ Usando config legada como fallback")
    except ImportError as fallback_error:
        print(f"❌ Erro no fallback de config: {fallback_error}")
        raise