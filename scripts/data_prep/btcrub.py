import numpy as np
import pandas as pd

from paths_local import INCOMING, OUT_WEEKLY

# =========================
# 1. Загрузка BTC
# =========================
btc = pd.read_csv(INCOMING / "btc_usd.csv")

btc.columns = btc.columns.str.strip()

btc = btc.rename(columns={
    "Дата": "date",
    "Цена": "price"
})

btc["date"] = pd.to_datetime(btc["date"], dayfirst=True)
btc = btc.set_index("date").sort_index()

btc["price"] = (
    btc["price"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

btc = btc.rename(columns={"price": "btc_usd"})


# =========================
# 2. Загрузка USD/RUB
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
# 3. WEEKLY (ВОСКРЕСЕНЬЕ)
# =========================
btc = btc.resample("W-SUN").last()
fx = fx.resample("W-SUN").last()


# =========================
# 4. ОБЪЕДИНЕНИЕ
# =========================
df = pd.concat([btc, fx], axis=1, join="inner")


# =========================
# 5. BTC в RUB
# =========================
df["btc_rub"] = df["btc_usd"] * df["usd_rub"]


# =========================
# 6. Лог-доходности
# =========================
df["btc_return"] = np.log(df["btc_rub"] / df["btc_rub"].shift(1))


# =========================
# 7. Удаление NaN
# =========================
result = df[["btc_return"]].dropna()


# =========================
# 8. Вывод
# =========================
print(result.head())
print("Rows:", len(result))

result.to_csv(OUT_WEEKLY / "btc_weekly_returns.csv")
