# download_dukascopy.py
"""
🌐 DUKASCOPY DATA DOWNLOADER
Baixa dados históricos tick-by-tick do Dukascopy
Converte para candles M15
Salva em CSV para o otimizador
"""

import os
import requests
import lzma
import struct
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
import time
from tqdm import tqdm

# ===========================
# CONFIGURAÇÕES
# ===========================
CACHE_DIR = Path("dukascopy_data")
CACHE_DIR.mkdir(exist_ok=True)

BASE_URL = "https://datafeed.dukascopy.com/datafeed"

# Símbolos suportados
DUKASCOPY_SYMBOLS = {
    # Forex Majors
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "USDCHF": "USDCHF",
    "AUDUSD": "AUDUSD",
    "USDCAD": "USDCAD",
    "NZDUSD": "NZDUSD",

    # Forex Crosses
    "EURJPY": "EURJPY",
    "GBPJPY": "GBPJPY",
    "EURGBP": "EURGBP",
    "AUDJPY": "AUDJPY",
    "EURAUD": "EURAUD",
    "CADJPY": "CADJPY",
    "GBPAUD": "GBPAUD",
    "GBPCAD": "GBPCAD",
    "AUDCAD": "AUDCAD",
    "AUDNZD": "AUDNZD",

    # Metais
    "XAUUSD": "XAUUSD",
    "XAGUSD": "XAGUSD",
    "XAUJPY": "XAUJPY",
    "XAGEUR": "XAGEUR",
    "XAGAUD": "XAGAUD",
    "XAUGBP": "XAUGBP",
    "XAUAUD": "XAUAUD",
    "XAUEUR": "XAUEUR",
    "XAUCHF": "XAUCHF",
}

# Período padrão (últimos 12 meses)
MONTHS_BACK = 12

# ===========================
# FUNÇÕES AUXILIARES
# ===========================
def download_tick_data(symbol: str, year: int, month: int, day: int, hour: int) -> Optional[bytes]:
    """
    Baixa dados tick de 1 hora do Dukascopy

    Args:
        symbol: Par (ex: EURUSD)
        year, month, day, hour: Data/hora

    Returns:
        Bytes comprimidos ou None se falhar
    """
    # Dukascopy usa mês 0-11
    url = f"{BASE_URL}/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.content
        return None
    except Exception as e:
        return None


def decompress_bi5(data: bytes, symbol: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Descomprime arquivo .bi5 do Dukascopy

    Args:
        data: Bytes comprimidos
        symbol: Nome do símbolo (para determinar divisor)

    Returns:
        Lista de (timestamp_ms, ask, bid, ask_volume, bid_volume)
    """
    try:
        decompressed = lzma.decompress(data)
    except Exception as e:
        return []

    # Determina divisor de preço
    if "JPY" in symbol:
        divisor = 1000.0
    else:
        divisor = 100000.0

    ticks = []
    chunk_size = 20  # Cada tick = 20 bytes

    for i in range(0, len(decompressed), chunk_size):
        if i + chunk_size > len(decompressed):
            break

        chunk = decompressed[i:i+chunk_size]

        try:
            # Formato: timestamp (4 bytes), ask (4), bid (4), ask_vol (4), bid_vol (4)
            timestamp_ms = struct.unpack('>I', chunk[0:4])[0]
            ask_raw = struct.unpack('>I', chunk[4:8])[0]
            bid_raw = struct.unpack('>I', chunk[8:12])[0]
            ask_volume = struct.unpack('>f', chunk[12:16])[0]
            bid_volume = struct.unpack('>f', chunk[16:20])[0]

            # Converte para preço real
            ask = ask_raw / divisor
            bid = bid_raw / divisor

            ticks.append((timestamp_ms, ask, bid, ask_volume, bid_volume))

        except Exception as e:
            continue

    return ticks


def ticks_to_m15_candles(ticks: List[Tuple], symbol: str, base_time: datetime) -> pd.DataFrame:
    """
    Converte ticks para candles M15

    Args:
        ticks: Lista de (timestamp_ms, ask, bid, ask_vol, bid_vol)
        symbol: Nome do símbolo
        base_time: Hora base (início da hora)

    Returns:
        DataFrame com candles M15
    """
    if not ticks:
        return pd.DataFrame()

    # Cria DataFrame de ticks
    df_ticks = pd.DataFrame(
        ticks,
        columns=['timestamp_ms', 'ask', 'bid', 'ask_volume', 'bid_volume']
    )

    # Calcula preço médio e spread
    df_ticks['mid'] = (df_ticks['ask'] + df_ticks['bid']) / 2
    df_ticks['spread'] = (df_ticks['ask'] - df_ticks['bid']) * 10000  # em pips

    # Timestamp absoluto
    df_ticks['time'] = base_time + pd.to_timedelta(df_ticks['timestamp_ms'], unit='ms')

    # Agrupa em candles de 15 minutos
    df_ticks.set_index('time', inplace=True)

    candles = df_ticks['mid'].resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })

    # Tick volume (número de ticks)
    candles['tick_volume'] = df_ticks['mid'].resample('15T').count()

    # Spread médio
    candles['spread'] = df_ticks['spread'].resample('15T').mean()

    # Remove candles vazios
    candles = candles[candles['tick_volume'] > 0].copy()

    # Reordena colunas
    candles = candles[['open', 'high', 'low', 'close', 'tick_volume', 'spread']]

    return candles.reset_index()


def download_symbol_range(symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """
    Baixa dados de um símbolo para um período

    Args:
        symbol: Par Forex (ex: EURUSD)
        start_date: Data inicial
        end_date: Data final

    Returns:
        DataFrame com candles M15 ou None se falhar
    """
    if symbol not in DUKASCOPY_SYMBOLS:
        print(f"❌ {symbol} não disponível no Dukascopy")
        return None

    duka_symbol = DUKASCOPY_SYMBOLS[symbol]
    all_candles = []

    current = start_date
    total_hours = int((end_date - start_date).total_seconds() / 3600)

    print(f"\n📥 Baixando {symbol}...")
    print(f"   Período: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')} ({total_hours} horas)")

    downloaded = 0
    failed = 0

    pbar = tqdm(total=total_hours, desc=f"  {symbol}", leave=False)

    while current < end_date:
        data = download_tick_data(
            duka_symbol,
            current.year,
            current.month - 1,  # Dukascopy usa 0-11
            current.day,
            current.hour
        )

        if data:
            ticks = decompress_bi5(data, symbol)
            if ticks:
                candles = ticks_to_m15_candles(ticks, symbol, current)
                if not candles.empty:
                    all_candles.append(candles)
                    downloaded += 1
        else:
            failed += 1

        current += timedelta(hours=1)
        pbar.update(1)

        # Rate limit (Dukascopy permite ~10 req/s)
        time.sleep(0.1)

    pbar.close()

    if not all_candles:
        print(f"❌ {symbol}: Nenhum dado baixado")
        return None

    df_final = pd.concat(all_candles, ignore_index=True)
    df_final = df_final.sort_values('time').reset_index(drop=True)

    # Remove duplicatas
    df_final = df_final.drop_duplicates(subset=['time'], keep='first')

    print(f"✅ {symbol}: {len(df_final)} candles M15 ({downloaded} horas OK, {failed} falharam)")

    return df_final


def save_to_csv(df: pd.DataFrame, symbol: str) -> Path:
    """
    Salva DataFrame em CSV
    """
    output_file = CACHE_DIR / f"{symbol}_M15.csv"
    df.to_csv(output_file, index=False)
    return output_file


def main():
    print("="*80)
    print("🌐 DOWNLOAD DE DADOS HISTÓRICOS - DUKASCOPY")
    print("="*80)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=MONTHS_BACK * 30)

    print(f"📅 Período: {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")
    print(f"📊 Timeframe: M15 (15 minutos)")
    print(f"💾 Destino: {CACHE_DIR.absolute()}")
    print(f"📍 Símbolos: {len(DUKASCOPY_SYMBOLS)}")
    print("="*80 + "\n")

    success_count = 0
    fail_count = 0
    total_candles = 0

    for symbol in sorted(DUKASCOPY_SYMBOLS.keys()):
        df = download_symbol_range(symbol, start_date, end_date)

        if df is not None and len(df) > 0:
            output_file = save_to_csv(df, symbol)
            success_count += 1
            total_candles += len(df)
        else:
            fail_count += 1

    print("\n" + "="*80)
    print("📊 RESUMO DO DOWNLOAD")
    print("="*80)
    print(f"✅ Sucesso: {success_count} símbolos")
    print(f"❌ Falha: {fail_count} símbolos")
    print(f"📊 Total de candles: {total_candles:,}")
    print(f"💾 Arquivos salvos em: {CACHE_DIR.absolute()}")
    print("="*80)

    # Lista arquivos criados
    print("\n📁 Arquivos criados:")
    for csv_file in sorted(CACHE_DIR.glob("*_M15.csv")):
        size_mb = csv_file.stat().st_size / (1024 * 1024)
        print(f"  • {csv_file.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
