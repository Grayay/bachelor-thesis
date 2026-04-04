# Raw and intermediate weekly inputs

## Processed panel (repository default)

The merged weekly log-return panel used by the backtests is:

- `../final_dataset.csv`

Regenerate it only if you rebuild from single-asset series:

```bash
python merge_dataset.py
```

That script expects these **weekly** CSV files in `data/raw/`:

| File | Contents |
|------|-----------|
| `btc_weekly_returns.csv` | BTC/RUB weekly log return |
| `eth_returns_rub.csv` | ETH/RUB weekly log return |
| `gold_returns_rub.csv` | Gold/RUB weekly log return |
| `ruonia_returns_weekly.csv` | RUONIA weekly log return |
| `moex_weekly_returns.csv` | MOEX weekly log return |
| `ofz_weekly_returns.csv` | OFZ weekly log return |

## Vendor exports (not committed)

Place original broker / exchange files under `incoming/` using the filenames expected by `scripts/data_prep/` (see repository `README.md`). Rename your sources to match, or adjust paths inside those scripts.

**Note:** `scripts/data_prep/bonds.py` still aligns the OFZ series to `eth_weekly_returns_usd.csv` in `data/raw/`, matching the original thesis pipeline. Provide that file if you rebuild OFZ returns the same way.
