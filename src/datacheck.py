import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import skew, kurtosis

from paths import FINAL_DATASET_CSV


# =========================
# 1. ЗАГРУЗКА
# =========================
df = pd.read_csv(FINAL_DATASET_CSV)

# правильная дата
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")
df = df.sort_index()

print("=== DATA SHAPE ===")
print(df.shape)


# =========================
# 2. СТРУКТУРА
# =========================
print("\n=== NaN ===")
print(df.isna().sum())

print("\n=== DUPLICATES ===")
print("Duplicate dates:", df.index.duplicated().sum())

print("\n=== TIME STEP ===")
diff = df.index.to_series().diff().dropna()
print(diff.value_counts().head())


# =========================
# 3. БАЗОВАЯ СТАТИСТИКА
# =========================
stats = pd.DataFrame({
    "mean": df.mean(),
    "std": df.std(),
    "min": df.min(),
    "max": df.max(),
    "skew": df.apply(skew),
    "kurtosis": df.apply(kurtosis)
})

print("\n=== BASIC STATS ===")
print(stats)

print("\n=== EXTREME RETURNS (>30%) ===")
print((df.abs() > 0.3).sum())


# =========================
# 4. ADF
# =========================
print("\n=== ADF TEST ===")

for col in df.columns:
    result = adfuller(df[col].dropna())
    print(f"{col}: p-value = {result[1]:.5f}")


# =========================
# 5. LJUNG-BOX
# =========================
print("\n=== LJUNG-BOX TEST ===")

for col in df.columns:
    lb = acorr_ljungbox(df[col].dropna(), lags=[10], return_df=True)
    print(f"{col}: p-value = {lb['lb_pvalue'].values[0]:.5f}")


# =========================
# 6. ARCH
# =========================
print("\n=== ARCH TEST ===")

for col in df.columns:
    arch_test = het_arch(df[col].dropna())
    print(f"{col}: p-value = {arch_test[1]:.5f}")


# =========================
# 7. КОРРЕЛЯЦИИ
# =========================
corr = df.corr()

print("\n=== CORRELATION MATRIX ===")
print(corr)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()


# =========================
# 8. ГРАФИКИ
# =========================
df.plot(subplots=True, figsize=(10,8), title="Time Series")
plt.tight_layout()
plt.show()

df.plot(kind="box", figsize=(8,6))
plt.title("Returns Distribution")
plt.show()


# =========================
# 9. ДОП ПРОВЕРКИ
# =========================
print("\n=== ZERO VARIANCE CHECK ===")
print(df.std() == 0)

print("\n=== NEAR-ZERO VALUES ===")
print((df.abs() < 1e-6).sum())

print("\n=== OFZ RETURN SORTED ===")
print(df["ofz_return"].sort_values().head())
print(df["ofz_return"].sort_values().tail())
