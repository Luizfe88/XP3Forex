"""
ML Model Management for XP3 PRO FOREX
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)

def train_ml_model(df):
    """Treina um modelo RandomForest para prever direção do próximo movimento"""
    try:
        if len(df) < 100:
            return None, np.full(len(df), 0.5)
            
        # Feature engineering simples
        data = df.copy()
        data['returns'] = data['close'].pct_change()
        data['target'] = (data['returns'].shift(-1) > 0).astype(int)
        
        # Features básicas
        for p in [5, 14, 21]:
            data[f'rsi_{p}'] = _calculate_rsi(data['close'], p)
            data[f'mom_{p}'] = data['close'].pct_change(p)
        
        data = data.dropna()
        if len(data) < 50:
            return None, np.full(len(df), 0.5)
            
        features = [col for col in data.columns if 'rsi' in col or 'mom' in col]
        X = data[features]
        y = data['target']
        
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)
        
        return model, predict_ml_model(model, df)
    except Exception as e:
        logger.error(f"Erro no treino ML: {e}")
        return None, np.full(len(df), 0.5)

def predict_ml_model(model, df):
    """Gera probabilidades de confiança para o dataset"""
    if model is None:
        return np.full(len(df), 0.5)
        
    try:
        data = df.copy()
        for p in [5, 14, 21]:
            data[f'rsi_{p}'] = _calculate_rsi(data['close'], p)
            data[f'mom_{p}'] = data['close'].pct_change(p)
            
        features = [col for col in data.columns if 'rsi' in col or 'mom' in col]
        X = data[features].fillna(0.5)
        
        # Probabilidade da classe 1 (subida)
        probs = model.predict_proba(X)[:, 1]
        return probs
    except Exception:
        return np.full(len(df), 0.5)

def _calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
