
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bandits import compare_policies

st.title("Snappx — TRL‑3 / Run 1")
st.subheader("Random vs Thompson con Urgenza (Countdown) e Stock limitato")

with st.sidebar:
    st.header("Parametri")
    users = st.slider("Utenti simulati", 100, 5000, 1000, 100)
    horizon_s = st.slider("Durata drop (secondi)", 300, 3600, 900, 60)
    k = st.slider("Numero di drop", 2, 6, 3, 1)
    seed = st.number_input("Seed", 0, 999999, 42, 1)

res = compare_policies(users=users, horizon_s=horizon_s, seed=seed, k=k)

df = pd.DataFrame(res).T
st.write("**Risultati**", df)

st.write("**Confronto metriche chiave**")
ax = df[["CTR","conversion_given_token","utilization_stock"]].plot(kind="bar")
plt.xticks(rotation=0)
st.pyplot(plt.gcf())
plt.clf()

st.caption("Nota: in TRL‑3/Run 1 consideriamo 'token' ≈ 'redemption' per semplicità. La separazione avverrà nei run successivi.")
