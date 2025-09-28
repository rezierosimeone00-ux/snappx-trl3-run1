import streamlit as st
import pandas as pd
import os
from bandits import compare_policies

st.title("Snappx — TRL-3 / Run 1 (Streamlit)")

# ==============================
# SEZIONE 1 — SIMULAZIONE LIVE
# ==============================
with st.sidebar:
    st.header("Parametri simulazione")
    users = st.slider("Utenti simulati", 100, 10000, 3000, 100)
    horizon = st.slider("Durata drop (sec)", 300, 3600, 900, 60)
    drops = st.slider("Numero drop", 2, 6, 3, 1)
    seeds = st.slider("Seeds (medie)", 1, 50, 20, 1)
    seed0 = st.number_input("Seed di partenza", 0, 100000, 0, 1)

rows = []
for s in range(seed0, seed0 + seeds):
    out = compare_policies(users=users, horizon_s=horizon, drops_k=drops, seed=s)
    for pol, m in out.items():
        rows.append({"seed": s, "policy": pol, **m})
df = pd.DataFrame(rows)

st.subheader("Per-seed results (Simulazione live)")
st.dataframe(df, use_container_width=True)

g = df.groupby("policy").mean(numeric_only=True)
if "random" in g.index and "thompson" in g.index:
    uplift = (g.loc["thompson","CTR"]/g.loc["random","CTR"] - 1.0)*100.0 if g.loc["random","CTR"] > 0 else float("nan")
    st.metric("Uplift Thompson vs Random", f"{uplift:.2f}%")

# ==============================
# SEZIONE 2 — RUN SALVATO
# ==============================
st.header("📂 Run salvato (da CSV + grafici)")

csv_path = "outputs/streamlit_trl_3_r1/run.csv"

if os.path.exists(csv_path):
    df_saved = pd.read_csv(csv_path)

    for c in ["views","tokens","redemptions","CTR"]:
        if c in df_saved.columns:
            df_saved[c] = pd.to_numeric(df_saved[c], errors="coerce")

    st.subheader("Per-seed results (da file salvato)")
    st.dataframe(df_saved.head(20), use_container_width=True)

    st.download_button(
        "Scarica CSV run salvato",
        df_saved.to_csv(index=False),
        "run1_3000u_20s.csv",
        "text/csv"
    )

    # Mostra i grafici salvati
    plot_lc = os.path.join("outputs/streamlit_trl_3_r1", "plots_learning_curve.png")
    plot_or = os.path.join("outputs/streamlit_trl_3_r1", "plots_overall_rate.png")

    if os.path.exists(plot_lc):
        st.image(plot_lc, caption="Learning Curve", use_container_width=True)

    if os.path.exists(plot_or):
        st.image(plot_or, caption="Overall CTR", use_container_width=True)

else:
    st.warning("Nessun file run.csv trovato in outputs/streamlit_trl_3_r1/")

