import sys
import os
import numpy as np

# Add the project directory to sys.path
sys.path.append(r"c:\Users\luizf\Documents\xp3forex")
# Add the legacy directory to sys.path
sys.path.append(r"c:\Users\luizf\Documents\xp3forex\legacy")

# Mock config_forex and utils_forex before importing otimizador_semanal_forex
import config_mock
sys.modules['config_forex'] = config_mock
import utils_mock
sys.modules['utils_forex'] = utils_mock

import otimizador_semanal_forex as opt

def run_test():
    try:
        x = np.array([1.0, 1.1, 1.2, 1.15, 1.18, 1.22] * 100, dtype=np.float64)
        r1 = opt.ema_numba(x, 3)
        r2 = opt.ema_numpy(x, 3)

        print(f"Type of opt.ema_numba: {type(opt.ema_numba)}")
        print(f"Are results close? {np.allclose(r1, r2)}")

        from numba.core.registry import CPUDispatcher
        if isinstance(opt.ema_numba, CPUDispatcher):
            print("✅ Numba sanity test passed: ema_numba is a CPUDispatcher.")
        else:
            print("❌ Numba sanity test failed: ema_numba is NOT a CPUDispatcher.")
            sys.exit(1)
            
        if np.allclose(r1, r2):
            print("✅ Results match.")
        else:
            print("❌ Results do NOT match.")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error during sanity test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()
