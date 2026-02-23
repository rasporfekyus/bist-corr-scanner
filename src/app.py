# src/app.py
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="BIST Korelasyon Paneli", layout="wide")
st.title("Korelasyon Tarayıcı")

mode = st.radio("Mod", ["Ham", "Market-neutral"], horizontal=True)

pairs_path = DATA_DIR / ("pairs_neutral.parquet" if mode == "Market-neutral" else "pairs.parquet")
if not pairs_path.exists():
    st.error(f"{pairs_path.name} bulunamadı. Önce terminalde: python src/update_corr.py")
    st.stop()


pairs_path = DATA_DIR / ("pairs_neutral.parquet" if mode == "Market-neutral" else "pairs.parquet")
pairs = pd.read_parquet(pairs_path)

pairs["abs"] = pairs["corr"].abs()

search = st.text_input("Ara (ticker yaz):", "")

pos_thr = st.slider("Pozitif korelasyon eşiği (>=)", 0.0, 1.0, 0.60, 0.01)
neg_thr = st.slider("Negatif korelasyon eşiği (<=)", -1.0, 0.0, -0.20, 0.01)
top_n = st.slider("Gösterilecek adet", 5, 300, 150, 5)

df = pairs.copy()

if search.strip():
    s = search.upper().strip()
    df = df[df["a"].str.contains(s) | df["b"].str.contains(s)]

col1, col2 = st.columns(2)

with col1:
    st.subheader("En Pozitif Korelasyonlar")
    pos = df[df["corr"] >= pos_thr].sort_values("corr", ascending=False).head(top_n)
    if pos.empty:
        st.info("Bu eşikte pozitif korelasyon yok.")
    else:
        pos = pos.copy()
        pos["pair"] = pos["a"] + " / " + pos["b"]
        h = max(350, 18 * len(pos) + 120)
        fig = px.bar(pos, x="corr", y="pair", orientation="h", height=h, text="corr")
        fig.update_traces(texttemplate="%{text:.2f}", marker_color="green")
        fig.update_layout(yaxis_title="", xaxis_title="corr")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("En Negatif Korelasyonlar")
    neg = df[df["corr"] <= neg_thr].sort_values("corr").head(top_n)
    if neg.empty:
        st.info("Bu eşikte negatif korelasyon yok.")
    else:
        neg = neg.copy()
        neg["pair"] = neg["a"] + " / " + neg["b"]
        h = max(350, 18 * len(neg) + 120)
        fig = px.bar(neg, x="corr", y="pair", orientation="h", height=h, text="corr")
        fig.update_traces(texttemplate="%{text:.2f}", marker_color="red")
        fig.update_layout(yaxis_title="", xaxis_title="corr")
        st.plotly_chart(fig, use_container_width=True)