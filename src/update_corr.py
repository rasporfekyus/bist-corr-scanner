# src/update_corr.py
from pathlib import Path
from datetime import date
import time

import numpy as np
import pandas as pd
import yfinance as yf

MARKET_TICKER = "^XU030"   # BIST 30 (Yahoo). İstersen "^XU100" yapabilirsin.
NEUTRAL_MODE = "index"     # "index" veya "mean"


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TICKERS_FILE = BASE_DIR / "tickers.txt"


def load_tickers() -> list[str]:
    if not TICKERS_FILE.exists():
        raise FileNotFoundError(f"tickers.txt bulunamadı: {TICKERS_FILE}")

    tickers = []
    for line in TICKERS_FILE.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tickers.append(s)

    if not tickers:
        raise ValueError("tickers.txt boş görünüyor.")
    return tickers


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    series_list = []
    failed = []

    for t in tickers:
        last_err = None

        for attempt in range(4):  # 4 deneme
            try:
                df = yf.download(
                    t,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )

                if df is None or df.empty:
                    raise ValueError("empty dataframe")

                if "Close" not in df.columns:
                    raise ValueError(f"Close not in columns: {list(df.columns)}")

                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                if not isinstance(close, pd.Series):
                    raise ValueError(f"Close is not Series: {type(close)}")

                close.name = t
                series_list.append(close)
                break

            except Exception as e:
                last_err = e
                time.sleep(2 + attempt)  # 2,3,4,5 sn

        else:
            failed.append((t, repr(last_err)))

    if failed:
        print("SKIP edilenler:")
        for t, err in failed:
            print(f" - {t}: {err}")

    if not series_list:
        raise RuntimeError("Hiçbir ticker indirilemedi. İnternet/Yahoo kaynaklı olabilir.")

    prices = pd.concat(series_list, axis=1).sort_index()
    prices.columns.name = None  # sonradan 'Ticker' çakışması olmasın
    return prices


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).dropna(how="all")


def market_neutralize(rets: pd.DataFrame):
    """
    Basit market-neutral: r_i_neutral = r_i - beta_i * r_mkt
    r_mkt: aynı gün tüm hisselerin eşit ağırlıklı ortalama getirisi
    """
    X = rets.dropna(axis=1, how="all").copy()

    # Market proxy: cross-sectional mean
    mkt = X.mean(axis=1).rename("_mkt")

    # Ortak günleri kullan (NaN temizliği)
    df = pd.concat([X, mkt], axis=1).dropna(how="any")
    Xc = df[X.columns]
    mktc = df["_mkt"]

    var_m = float(mktc.var())
    if not np.isfinite(var_m) or var_m == 0.0:
        raise ValueError("Market serisinin varyansı 0/NaN çıktı, neutralize edemiyorum.")

    # beta_i = cov(r_i, r_mkt) / var(r_mkt)
    covs = Xc.apply(lambda s: s.cov(mktc))
    betas = covs / var_m

    # residual = r_i - beta_i * r_mkt  (vektörize)
    resid = Xc.values - (mktc.values[:, None] * betas.values[None, :])
    rets_neutral = pd.DataFrame(resid, index=Xc.index, columns=Xc.columns)

    return rets_neutral, betas

def corr_pairs(rets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = rets.corr()

    # isim çakışması olmasın
    corr.index.name = "a"
    corr.columns.name = "b"

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    pairs = (
        corr.where(mask)
        .stack()
        .rename("corr")
        .rename_axis(index=["a", "b"])
        .reset_index()
    )

    pairs["abs"] = pairs["corr"].abs()
    return corr, pairs


def main():
    DATA_DIR.mkdir(exist_ok=True)

    tickers = load_tickers()
    start = "2024-01-01"
    end = date.today().strftime("%Y-%m-%d")

    prices = download_prices(tickers, start=start, end=end)
    rets = to_returns(prices)

    # HAM
    corr, pairs = corr_pairs(rets)
    pairs = pairs.sort_values("corr", ascending=False).reset_index(drop=True)

    # MARKET-NEUTRAL
    rets_neutral, betas = market_neutralize(rets)
    corr_n, pairs_n = corr_pairs(rets_neutral)
    pairs_n = pairs_n.sort_values("corr", ascending=False).reset_index(drop=True)

    # Kaydet
    prices.to_parquet(DATA_DIR / "prices.parquet")
    rets.to_parquet(DATA_DIR / "returns.parquet")
    corr.to_parquet(DATA_DIR / "corr_matrix.parquet")
    pairs.to_parquet(DATA_DIR / "pairs.parquet", index=False)

    rets_neutral.to_parquet(DATA_DIR / "returns_neutral.parquet")
    corr_n.to_parquet(DATA_DIR / "corr_matrix_neutral.parquet")
    pairs_n.to_parquet(DATA_DIR / "pairs_neutral.parquet", index=False)

    print(f"OK -> data/ klasörüne yazıldı. ham pairs: {len(pairs)} | neutral pairs: {len(pairs_n)}")


if __name__ == "__main__":
    main()