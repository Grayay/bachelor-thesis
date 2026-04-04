import numpy as np
import pandas as pd

from paths_local import INCOMING, OUT_WEEKLY

# === 1. Загрузка OFZ ===
ofz = pd.read_csv(INCOMING / "ofz_index.csv")

ofz.columns = ofz.columns.str.strip()

ofz = ofz.rename(columns={
    "Дата": "date",
    "Цена": "price"
})

ofz["date"] = pd.to_datetime(ofz["date"], dayfirst=True)
ofz = ofz.set_index("date")
ofz = ofz.sort_index()

ofz["price"] = ofz["price"].astype(str).str.replace(",", "")
ofz["price"] = ofz["price"].astype(float)

ofz = ofz[["price"]]
ofz = ofz.rename(columns={"price": "ofz_index"})

ofz["ofz_return"] = np.log(ofz["ofz_index"] / ofz["ofz_index"].shift(1))

ofz = ofz.dropna()

eth = pd.read_csv(OUT_WEEKLY / "eth_weekly_returns_usd.csv", parse_dates=["date"])
eth = eth.set_index("date")

common_index = ofz.index.intersection(eth.index)
ofz = ofz.loc[common_index]

result = ofz[["ofz_return"]]

print(result.head())
print("Rows:", len(result))

result.to_csv(OUT_WEEKLY / "ofz_weekly_returns.csv")
