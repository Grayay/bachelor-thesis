import numpy as np
import pandas as pd

from paths_local import INCOMING, OUT_WEEKLY

# =========================
# 1. GOLD (сырой файл)
# =========================
gold = pd.read_csv(INCOMING / "xau_usd.csv")

gold.columns = gold.columns.str.strip()

gold = gold.rename(columns={
    "Дата": "date",
    "Цена": "price"
})

gold["date"] = pd.to_datetime(gold["date"], dayfirst=True)
gold = gold.set_index("date").sort_index()

gold["price"] = (
    gold["price"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)
gold = gold.rename(columns={"price": "gold_usd"})


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
gold = gold.resample("W-SUN").last()
fx = fx.resample("W-SUN").last()


# =========================
# 4. ОБЪЕДИНЕНИЕ
# =========================
df = pd.concat([gold, fx], axis=1, join="inner")


# =========================
# 5. GOLD в RUB
# =========================
df["gold_rub"] = df["gold_usd"] * df["usd_rub"]


# =========================
# 6. ЛОГ-ДОХОДНОСТИ
# =========================
df["gold_return"] = np.log(df["gold_rub"] / df["gold_rub"].shift(1))


# =========================
# 7. УДАЛЕНИЕ NaN
# =========================
result = df[["gold_return"]].dropna()


# =========================
# 8. ПРОВЕРКА
# =========================
print(result.head())
print("Rows:", len(result))

result.to_csv(OUT_WEEKLY / "gold_returns_rub.csv")
