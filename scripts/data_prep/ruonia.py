import numpy as np
import pandas as pd

from paths_local import INCOMING, OUT_WEEKLY

# =========================
# 1. Загрузка
# =========================
df = pd.read_excel(INCOMING / "ruonia_daily.xlsx")

df.columns = df.columns.str.strip()

df = df.rename(columns={
    "DT": "date",
    "ruo": "ruonia"
})

df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()


# =========================
# 2. Дневная доходность
# =========================
df["daily_return"] = df["ruonia"] / 100 / 365


# =========================
# 3. Индекс (как цена)
# =========================
df["index"] = (1 + df["daily_return"]).cumprod()


# =========================
# 4. WEEKLY (воскресенье)
# =========================
weekly_index = df["index"].resample("W-SUN").last()


# =========================
# 5. Лог-доходности
# =========================
weekly_return = np.log(weekly_index / weekly_index.shift(1))


# =========================
# 6. Финал
# =========================
result = weekly_return.to_frame(name="ruonia_return").dropna()


# =========================
# 7. Проверка
# =========================
print(result.head())
print("Rows:", len(result))

result.to_csv(OUT_WEEKLY / "ruonia_returns_weekly.csv")
