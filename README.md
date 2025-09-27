
# Snappx — TRL‑3 / Run 1

**Obiettivo**: confrontare politica **Random** vs **Thompson Sampling (Beta‑Bernoulli)** su un feed di drop con **urgenza** (countdown) e **scarsità** (stock).

## Contenuti
- `bandits.py` — modulo con la logica di simulazione condiviso
- `snappx_trl3_run1.ipynb` — notebook Colab‑ready
- `streamlit_app.py` — app Streamlit per esplorazione interattiva
- `requirements.txt` — dipendenze Streamlit

## Esecuzione (Colab)
1. Carica l'intera cartella o i singoli file su Colab.
2. Apri `snappx_trl3_run1.ipynb` e **Esegui tutto**.

## Esecuzione (Streamlit, locale)
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Struttura metrica
- `CTR` = token / views
- `conversion_given_token` = redemptions / token (≈1 in Run 1 per semplicità)
- `utilization_stock` = token venduti / stock totale

## Note
- Il modello di urgenza amplifica la conversione verso la fine del countdown.
- Nei run successivi (TRL‑3/Run 2–3) si separeranno **token** e **redemption**, si introdurranno **wallet polarizzati**, **baseline fraud** e **ledger**.
