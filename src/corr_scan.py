import argparse
import pandas as pd
import numpy as np
import yfinance as yf

def get_prices(tickers, start, end):
    # Adj Close genelde temettü/bölünme düzeltilidir
    df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)["Adj Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(how="all")
    return df

def to_returns(prices, kind="log"):
    prices = prices.sort_index()
    if kind == "log":
        rets = np.log(prices).diff()
    else:
        rets = prices.pct_change()
    return rets.dropna(how="all")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", required=True, help="örn: AKBNK.IS THYAO.IS KCHOL.IS")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--returns", choices=["log", "pct"], default="log")
    p.add_argument("--out", default="corr_matrix.xlsx")
    args = p.parse_args()

    prices = get_prices(args.tickers, args.start, args.end)
    rets = to_returns(prices, kind=args.returns)

    # Korelasyon matrisi
    corr = rets.corr()

    # En yüksek/En düşük korelasyon çiftleri (kendisi hariç)
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .dropna()
        .sort_values(ascending=False)
    )

    print("\nTop + korelasyon (ilk 10):")
    print(pairs.head(10))

    print("\nTop - korelasyon (ilk 10):")
    print(pairs.tail(10))

    # Excel'e yaz
    with pd.ExcelWriter(args.out) as w:
        prices.to_excel(w, sheet_name="prices")
        rets.to_excel(w, sheet_name="returns")
        corr.to_excel(w, sheet_name="corr_matrix")
        pairs.rename("corr").to_frame().to_excel(w, sheet_name="pair_list")

    print(f"\nKaydedildi: {args.out}")

if __name__ == "__main__":
    main()
