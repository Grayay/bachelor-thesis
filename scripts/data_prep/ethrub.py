import numpy as np
import pandas as pd

from paths_local import INCOMING, OUT_WEEKLY

# =========================
# 1. ETH (сырой файл)
# =========================
eth = pd.read_csv(INCOMING / "eth_usd.csv")

eth.columns = eth.columns.str.strip()

eth = eth.rename(columns={
    "Дата": "date",
    "Цена": "price"
})

eth["date"] = pd.to_datetime(eth["date"], dayfirst=True)
eth = eth.set_index("date").sort_index()

eth["price"] = (
    eth["price"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

eth = eth.rename(columns={"price": "eth_usd"})


# =========================
# 2. USD/RUB
# =========================
fx = pd.read_csv(INCOMING / "usd_rub.csv")

fx.columns = fx.columns.str.strip()

fx = fx.rename(columns={
    "Дата": "date",
    "Цена": "price"
})

fx["date"] = pd.to_datetime(fx["date"], dayfirst=True)
fx = fx.set_index("date").sort_index()

fx["price"] = (
    fx["price"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

fx = fx.rename(columns={"price": "usd_rub"})


# =========================
# 3. WEEKLY (воскресенье)
# =========================
eth = eth.resample("W-SUN").last()
fx = fx.resample("W-SUN").last()


# =========================
# 4. ОБЪЕДИНЕНИЕ
# =========================
df = pd.concat([eth, fx], axis=1, join="inner")


# =========================
# 5. ETH в RUB
# =========================
df["eth_rub"] = df["eth_usd"] * df["usd_rub"]


# =========================
# 6. ЛОГ-ДОХОДНОСТИ
# =========================
df["eth_return"] = np.log(df["eth_rub"] / df["eth_rub"].shift(1))


# =========================
# 7. УДАЛЕНИЕ NaN
# =========================
result = df[["eth_return"]].dropna()


# =========================
# 8. ПРОВЕРКА
# =========================
print(result.head())
print("Rows:", len(result))

result.to_csv(OUT_WEEKLY / "eth_returns_rub.csv")
