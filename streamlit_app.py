import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json, os
from bandits import compare_policies

st.title("Snappx — TRL-3 / Run 1 (Streamlit)")

with st.sidebar:
    st.header("Parametri simulazione")
    users = st.slider("Utenti simulati", 100, 10000, 3000, 100)
    horizon = st.slider("Durata drop (sec)", 300, 3600, 900, 60)
    drops = st.slider("Numero drop", 2, 6, 3, 1)
    seeds = st.slider("Seeds (medie)", 1, 50, 20, 1)
    seed0 = st.number_input("Seed di partenza", 0, 100000, 0, 1)

rows = []
for s in range(seed0, seed0+seeds):
    out = compare_policies(users=users, horizon_s=horizon, drops_k=drops, seed=s)
    for pol, m in out.items():
        rows.append({"seed": s, "policy": pol, **m})
df = pd.DataFrame(rows)

st.subheader("Per-seed results")
st.dataframe(df, use_container_width=True)

g = df.groupby("policy").mean(numeric_only=True)
uplift = (g.loc["thompson","CTR"]/g.loc["random","CTR"] - 1.0)*100.0 if g.loc["random","CTR"]>0 else float("nan")

st.subheader("Sintesi")
st.write(
    f"**CTR medio** — Random: `{g.loc['random','CTR']:.4f}` · Thompson: `{g.loc['thompson','CTR']:.4f}`  \n"
    f"**Uplift CTR (Thompson vs Random):** `{uplift:.2f}%`"
)

fig1 = plt.figure()
df.pivot(index="seed", columns="policy", values="CTR").sort_index().plot(marker='o')
plt.title("Learning curve (CTR per seed)")
plt.xlabel("seed"); plt.ylabel("CTR"); plt.tight_layout()
st.pyplot(fig1)

fig2 = plt.figure()
g["CTR"].plot(kind="bar")
plt.title("CTR medio — Random vs Thompson")
plt.ylabel("CTR"); plt.xticks(rotation=0); plt.tight_layout()
st.pyplot(fig2)

st.markdown("---")
st.subheader("Run-1: output salvati (se presenti nel repo)")

if os.path.exists("metrics.json"):
    with open("metrics.json") as f: st.json(json.load(f))

cols = st.columns(2)
if os.path.exists("plots_learning_curve.png"):
    cols[0].image("plots_learning_curve.png", caption="Learning curve (salvata)")
if os.path.exists("plots_overall_rate.png"):
    cols[1].image("plots_overall_rate.png", caption="CTR medio (salvato)")

if os.path.exists("run1_3000u_20s.csv"):
    df_saved = pd.read_csv("run1_3000u_20s.csv")
    st.download_button("Scarica CSV run salvato", df_saved.to_csv(index=False), "run1_3000u_20s.csv", "text/csv")
