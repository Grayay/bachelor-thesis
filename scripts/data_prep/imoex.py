import numpy as np
import pandas as pd

from paths_local import INCOMING, OUT_WEEKLY

# === 1. Загрузка MOEX ===
moex = pd.read_csv(INCOMING / "moex_index.csv")

moex.columns = moex.columns.str.strip()

moex = moex.rename(columns={
    "Дата": "date",
    "Цена": "price"
})

moex["date"] = pd.to_datetime(moex["date"], dayfirst=True)
moex = moex.set_index("date")
moex = moex.sort_index()

moex["price"] = (
    moex["price"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

moex = moex[["price"]]
moex = moex.rename(columns={"price": "moex_index"})

moex["moex_return"] = np.log(moex["moex_index"] / moex["moex_index"].shift(1))

moex = moex.dropna()

eth = pd.read_csv(OUT_WEEKLY / "eth_returns_rub.csv", parse_dates=["date"])
eth = eth.set_index("date")

common_index = moex.index.intersection(eth.index)
moex = moex.loc[common_index]

result = moex[["moex_return"]]

print(result.head())
print("Rows:", len(result))

result.to_csv(OUT_WEEKLY / "moex_weekly_returns.csv")
