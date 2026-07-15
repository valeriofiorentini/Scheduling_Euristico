import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scheduling import (
    Job,
    calcola_utilita_minima,
    genera_istanza_casuale,
    iterated_local_search,
    ricerca_locale_scambio,
    simulated_annealing,
    tabu_search,
)

st.set_page_config(page_title="Scheduling - Ricerca Locale", layout="centered")

st.title("Scheduling: massimizzazione dell'utilità minima")
st.caption("Ricerca locale basata su scambio a coppie (pairwise swap)")

with st.expander("Genera istanza casuale"):
    st.caption("a_j ∈ [1, 5], b_j ∈ [10, 200], p_j ∈ [1, 15], r_j ∈ [0, 2·n_job]")
    col_n, col_seed, col_btn = st.columns([1, 1, 1])
    n_jobs_casuali = col_n.number_input("Numero di job", min_value=2, max_value=500, value=50)
    seed_casuale = col_seed.number_input("Seed", min_value=0, value=0, step=1)
    if col_btn.button("Genera", type="primary"):
        nuovi_jobs = genera_istanza_casuale(int(n_jobs_casuali), seed=int(seed_casuale))
        st.session_state.df_jobs = pd.DataFrame(
            [
                {
                    "id": job.id,
                    "processing_j": round(job.processing_j, 3),
                    "release_time_j": round(job.release_time_j, 3),
                    "coefficiente_a_j": round(job.coefficiente_a_j, 3),
                    "coefficiente_b_j": round(job.coefficiente_b_j, 3),
                }
                for job in nuovi_jobs.values()
            ]
        )
        st.session_state.sequenza_default = list(nuovi_jobs.keys())
        st.rerun()

st.subheader("Job")

if "df_jobs" not in st.session_state:
    jobs_iniziali = genera_istanza_casuale(50, seed=0)
    st.session_state.df_jobs = pd.DataFrame(
        [
            {
                "id": job.id,
                "processing_j": round(job.processing_j, 3),
                "release_time_j": round(job.release_time_j, 3),
                "coefficiente_a_j": round(job.coefficiente_a_j, 3),
                "coefficiente_b_j": round(job.coefficiente_b_j, 3),
            }
            for job in jobs_iniziali.values()
        ]
    )

df_jobs = st.data_editor(
    st.session_state.df_jobs,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "id": st.column_config.NumberColumn("id", step=1),
        "processing_j": st.column_config.NumberColumn("p_j (processamento)", step=0.5),
        "release_time_j": st.column_config.NumberColumn("r_j (rilascio)", step=0.5),
        "coefficiente_a_j": st.column_config.NumberColumn("a_j", step=0.1),
        "coefficiente_b_j": st.column_config.NumberColumn("b_j", step=1.0),
    },
    key="editor_jobs",
)

colonne_richieste = ["id", "processing_j", "release_time_j", "coefficiente_a_j", "coefficiente_b_j"]

righe_incomplete = df_jobs[colonne_richieste].isna().any(axis=1).sum()
df_jobs = df_jobs.dropna(subset=colonne_richieste)

if righe_incomplete:
    st.warning(f"{righe_incomplete} riga/e incomplete sono state ignorate. Compila tutti i campi per includerle.")

if df_jobs.empty:
    st.warning("Aggiungi almeno un job compilando tutti i campi della tabella.")
    st.stop()

df_jobs["id"] = df_jobs["id"].astype(int)

if df_jobs["id"].duplicated().any():
    st.error("Ci sono id duplicati tra i job. Correggi la tabella per continuare.")
    st.stop()

jobs_by_id = {
    int(row.id): Job(
        id=int(row.id),
        processing_j=row.processing_j,
        release_time_j=row.release_time_j,
        coefficiente_a_j=row.coefficiente_a_j,
        coefficiente_b_j=row.coefficiente_b_j,
    )
    for row in df_jobs.itertuples()
}

id_disponibili = list(jobs_by_id.keys())

st.subheader("Sequenza iniziale")
sequenza_default = st.session_state.get("sequenza_default", id_disponibili)
sequenza_default = [j for j in sequenza_default if j in id_disponibili] + [
    j for j in id_disponibili if j not in sequenza_default
]

sequenza_iniziale = st.multiselect(
    "Ordine di esecuzione dei job (id)",
    options=id_disponibili,
    default=sequenza_default,
)
st.session_state.sequenza_default = sequenza_iniziale

if len(sequenza_iniziale) != len(id_disponibili):
    st.warning("Seleziona tutti i job, ognuno una sola volta, per definire la sequenza iniziale.")
    st.stop()

obiettivo_iniziale = calcola_utilita_minima(sequenza_iniziale, jobs_by_id)

col1, col2 = st.columns(2)
col1.metric("Sequenza iniziale", " → ".join(str(j) for j in sequenza_iniziale))
col2.metric("Obiettivo iniziale", f"{obiettivo_iniziale:.2f}")

st.subheader("Algoritmo")
algoritmo = st.selectbox(
    "Metodo di ottimizzazione",
    [
        "Ricerca locale (pairwise swap)",
        "Tabu Search",
        "Simulated Annealing",
        "Iterated Local Search",
    ],
)

parametri = {}
if algoritmo == "Tabu Search":
    col_a, col_b = st.columns(2)
    parametri["iterazioni"] = col_a.number_input("Iterazioni", min_value=1, value=100)
    parametri["tabu_tenure"] = col_b.number_input("Tabu tenure", min_value=1, value=5)
elif algoritmo == "Simulated Annealing":
    col_a, col_b, col_c = st.columns(3)
    parametri["iterazioni"] = col_a.number_input("Iterazioni", min_value=1, value=500)
    parametri["temperatura_iniziale"] = col_b.number_input("Temperatura iniziale", min_value=0.01, value=10.0)
    parametri["raffreddamento"] = col_c.number_input("Raffreddamento", min_value=0.01, max_value=0.999, value=0.95)
elif algoritmo == "Iterated Local Search":
    col_a, col_b = st.columns(2)
    parametri["iterazioni"] = col_a.number_input("Iterazioni", min_value=1, value=20)
    parametri["num_scambi_perturbazione"] = col_b.number_input("Scambi per perturbazione", min_value=1, value=2)

if algoritmo == "Ricerca locale (pairwise swap)":
    sequenza_ottima, valore_ottimo = ricerca_locale_scambio(sequenza_iniziale, jobs_by_id)
elif algoritmo == "Tabu Search":
    sequenza_ottima, valore_ottimo = tabu_search(sequenza_iniziale, jobs_by_id, **parametri)
elif algoritmo == "Simulated Annealing":
    sequenza_ottima, valore_ottimo = simulated_annealing(sequenza_iniziale, jobs_by_id, **parametri)
else:
    sequenza_ottima, valore_ottimo = iterated_local_search(sequenza_iniziale, jobs_by_id, **parametri)

st.subheader(f"Risultato: {algoritmo}")
col3, col4 = st.columns(2)
col3.metric("Sequenza ottima locale", " → ".join(str(j) for j in sequenza_ottima))
col4.metric("Valore obiettivo", f"{valore_ottimo:.2f}", delta=f"{valore_ottimo - obiettivo_iniziale:+.2f}")

# dettaglio passo-passo della sequenza ottima
dettaglio = []
tempo_corrente = 0.0
for job_id in sequenza_ottima:
    job = jobs_by_id[job_id]
    inizio = max(tempo_corrente, job.release_time_j)
    tempo_corrente = inizio + job.processing_j
    utilita = job.coefficiente_b_j - job.coefficiente_a_j * tempo_corrente
    dettaglio.append(
        {
            "job": job_id,
            "inizio": inizio,
            "completamento (C_j)": tempo_corrente,
            "utilità (b_j - a_j*C_j)": round(utilita, 3),
        }
    )

st.dataframe(pd.DataFrame(dettaglio), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Confronto tra algoritmi")
st.caption("Tutti e quattro i metodi eseguiti sulla stessa sequenza iniziale")

risultati = {
    "Ricerca locale": ricerca_locale_scambio(sequenza_iniziale, jobs_by_id),
    "Tabu Search": tabu_search(sequenza_iniziale, jobs_by_id),
    "Simulated Annealing": simulated_annealing(sequenza_iniziale, jobs_by_id, seed=42),
    "Iterated Local Search": iterated_local_search(sequenza_iniziale, jobs_by_id, seed=42),
}

nomi = list(risultati.keys())
valori = [risultati[nome][1] for nome in nomi]
sequenze = [" → ".join(str(j) for j in risultati[nome][0]) for nome in nomi]

# palette categorica fissa (blu, aqua, giallo, verde) — un colore per algoritmo, ordine stabile
colori = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]

fig = go.Figure(
    data=[
        go.Bar(
            x=nomi,
            y=valori,
            marker_color=colori[: len(nomi)],
            text=[f"{v:.2f}" for v in valori],
            textposition="outside",
            hovertext=[f"{nome}<br>Sequenza: {seq}<br>Utilità minima: {v:.2f}"
                       for nome, seq, v in zip(nomi, sequenze, valori)],
            hoverinfo="text",
        )
    ]
)
fig.add_hline(
    y=obiettivo_iniziale,
    line_dash="dash",
    line_color="#52514e",
    annotation_text="Obiettivo iniziale",
    annotation_position="top left",
)
fig.update_layout(
    yaxis_title="Utilità minima",
    xaxis_title=None,
    showlegend=False,
    margin=dict(t=40, b=20),
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
    font_color="#0b0b0b",
)

st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    pd.DataFrame(
        {
            "Algoritmo": nomi,
            "Sequenza ottima": sequenze,
            "Utilità minima": [round(v, 3) for v in valori],
            "Miglioramento vs iniziale": [round(v - obiettivo_iniziale, 3) for v in valori],
        }
    ),
    use_container_width=True,
    hide_index=True,
)
