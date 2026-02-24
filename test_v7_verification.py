import sys
import pandas as pd
import numpy as np
import config_forex as config
import optimizer_optuna_forex as optimizer
# import otimizador_semanal_forex as otimizador # AVOID RAY IMPORT

# Force reduced settings for test
config.OPTUNA_N_TRIALS = 2
config.OPTUNA_TIMEOUT = 60

print("Testing V7 Pipeline (No Ray)...")

def load_data_mock(symbol):
    try:
        path = f"dukascopy_data/{symbol}_M15.csv"
        df = pd.read_csv(path, names=['time', 'open', 'high', 'low', 'close', 'volume'], header=0)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        # Mock Pip/Tick
        return {
            "df": df,
            "pip_size": 0.0001,
            "tick_value": 1.0, # Simplified
            "spread": 1.5,
            "news_mask": np.zeros(len(df), dtype=bool),
            "source": "MOCK"
        }
    except Exception as e:
        print(f"Error loading mock data: {e}")
        return None

# Mock data loading
data = load_data_mock("AUDUSD")
if data:
    print(f"Data loaded: {len(data['df'])} candles")
    
    # Use small subset
    df_train = data['df'].iloc[-5000:-1000]
    data_train = data.copy()
    data_train['df'] = df_train
    # Fix news mask slicing
    full_news_mask = data['news_mask']
    data_train['news_mask'] = full_news_mask[-5000:-1000]

    print("Training ML...")
    model, probs = optimizer.train_ml_model(df_train)
    data_train['ml_confidence'] = probs
    print(f"Model trained. Probs shape: {probs.shape}")
    
    print("Optimizing...")
    res = optimizer.optimize_with_optuna(data_train, n_trials=2, timeout=60)
    print("Best params:", res)
    
    if not res['best_params']:
        print("Optimization failed to find valid params")
        sys.exit(1)

    print("Predicting OOS...")
    df_test = data['df'].iloc[-1000:]
    probs_oos = optimizer.predict_ml_model(model, df_test)
    
    print("Backtesting OOS...")
    data_oos = data.copy()
    data_oos['df'] = df_test
    data_oos['news_mask'] = full_news_mask[-1000:]
    data_oos['ml_confidence'] = probs_oos
    
    metrics, _, _ = optimizer.run_backtest_with_params(data_oos, res['best_params'])
    print("Metrics OOS:", metrics)
    print("Test Complete ✅")

else:
    print("Could not load data for AUDUSD")
