import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd

from paths import FINAL_DATASET_CSV, RAW_DATA_DIR

# загрузка (weekly return CSVs in data/raw/)
btc = pd.read_csv(RAW_DATA_DIR / "btc_weekly_returns.csv", parse_dates=["date"]).set_index("date")
eth = pd.read_csv(RAW_DATA_DIR / "eth_returns_rub.csv", parse_dates=["date"]).set_index("date")
gold = pd.read_csv(RAW_DATA_DIR / "gold_returns_rub.csv", parse_dates=["date"]).set_index("date")
ruonia = pd.read_csv(RAW_DATA_DIR / "ruonia_returns_weekly.csv", parse_dates=["date"]).set_index("date")
moex = pd.read_csv(RAW_DATA_DIR / "moex_weekly_returns.csv", parse_dates=["date"]).set_index("date")
ofz = pd.read_csv(RAW_DATA_DIR / "ofz_weekly_returns.csv", parse_dates=["date"]).set_index("date")

# объединение через inner join
df = pd.concat([
    btc,
    eth,
    gold,
    ruonia,
    moex,
    ofz
], axis=1, join="inner")

# проверка
print(df.head())
print("Rows:", len(df))
print(df.isna().sum())

# сохранение
df.to_csv(FINAL_DATASET_CSV)