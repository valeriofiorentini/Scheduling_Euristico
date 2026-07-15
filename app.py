import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scheduling import (
    calcola_utilita_minima,
    genera_istanza_casuale,
    iterated_local_search,
    maxmin_greedy,
    ricerca_locale_scambio,
    simulated_annealing,
    tabu_search,
)

st.set_page_config(page_title="Scheduling - Ricerca Locale", layout="centered")

st.title("Scheduling: massimizzazione dell'utilità minima")
st.caption(
    "Confronto tra Ricerca locale (pairwise swap), Tabu Search, Simulated Annealing, "
    "Iterated Local Search e MaxMinGreedy (ottimo esatto) su 10 istanze casuali indipendenti."
)

st.subheader("Confronto su 10 istanze casuali")
st.caption(
    "Problema 1|uj|umin (nessuna release date, come in Nicosia-Pacifici-Pferschy 2026). "
    "Genera 10 istanze indipendenti (seed 0-9) con — "
    "a_j ∈ [0.1,2] (frazionario), b_j ∈ [200,2000], p_j ∈ [1,15] — "
    "ed esegue tutti gli algoritmi su ciascuna, incluso l'ottimo esatto MaxMinGreedy."
)

n_jobs_multi = st.number_input("Job per istanza", min_value=2, max_value=500, value=50, key="n_jobs_multi")

with st.expander("Parametri degli algoritmi"):
    col_t1, col_t2 = st.columns(2)
    tabu_iterazioni = col_t1.number_input("Tabu Search — iterazioni", min_value=1, value=200)
    tabu_tenure = col_t2.number_input("Tabu Search — tabu tenure", min_value=1, value=10)

    col_s1, col_s2, col_s3 = st.columns(3)
    sa_iterazioni = col_s1.number_input("Simulated Annealing — iterazioni", min_value=1, value=2000)
    sa_temperatura = col_s2.number_input("Simulated Annealing — temperatura iniziale", min_value=0.01, value=20.0)
    sa_raffreddamento = col_s3.number_input(
        "Simulated Annealing — raffreddamento", min_value=0.01, max_value=0.999, value=0.995,
    )

    col_i1, col_i2 = st.columns(2)
    ils_iterazioni = col_i1.number_input("Iterated Local Search — iterazioni", min_value=1, value=30)
    ils_scambi = col_i2.number_input("Iterated Local Search — scambi per perturbazione", min_value=1, value=4)


@st.cache_data(show_spinner="Genero e confronto le 10 istanze...")
def confronta_10_istanze(
    n_jobs, tabu_iterazioni, tabu_tenure, sa_iterazioni, sa_temperatura, sa_raffreddamento,
    ils_iterazioni, ils_scambi,
):
    righe_multi = []
    righe_dettaglio = []

    for i in range(10):
        jobs_istanza = genera_istanza_casuale(int(n_jobs), seed=i)
        seq_iniziale = list(jobs_istanza.keys())
        valore_iniziale = calcola_utilita_minima(seq_iniziale, jobs_istanza)

        seq_locale, v_locale = ricerca_locale_scambio(seq_iniziale, jobs_istanza)
        seq_tabu, v_tabu = tabu_search(
            seq_iniziale, jobs_istanza, iterazioni=int(tabu_iterazioni), tabu_tenure=int(tabu_tenure),
        )
        seq_sa, v_sa = simulated_annealing(
            seq_iniziale, jobs_istanza, iterazioni=int(sa_iterazioni),
            temperatura_iniziale=sa_temperatura, raffreddamento=sa_raffreddamento, seed=i,
        )
        seq_ils, v_ils = iterated_local_search(
            seq_iniziale, jobs_istanza, iterazioni=int(ils_iterazioni),
            num_scambi_perturbazione=int(ils_scambi), seed=i,
        )
        seq_greedy, v_greedy = maxmin_greedy(jobs_istanza)

        righe_multi.append({
            "istanza": i + 1,
            "Ricerca locale": v_locale,
            "Tabu Search": v_tabu,
            "Simulated Annealing": v_sa,
            "Iterated Local Search": v_ils,
            "MaxMinGreedy (ottimo)": v_greedy,
        })

        for nome_algo, seq_finale, v_finale in [
            ("Ricerca locale", seq_locale, v_locale),
            ("Tabu Search", seq_tabu, v_tabu),
            ("Simulated Annealing", seq_sa, v_sa),
            ("Iterated Local Search", seq_ils, v_ils),
            ("MaxMinGreedy (ottimo)", seq_greedy, v_greedy),
        ]:
            righe_dettaglio.append({
                "istanza": i + 1,
                "algoritmo": nome_algo,
                "valore iniziale": round(valore_iniziale, 3),
                "valore finale": round(v_finale, 3),
                "miglioramento": round(v_finale - valore_iniziale, 3),
            })

    return pd.DataFrame(righe_multi), pd.DataFrame(righe_dettaglio)


df_multi, df_dettaglio = confronta_10_istanze(
    int(n_jobs_multi), tabu_iterazioni, tabu_tenure, sa_iterazioni, sa_temperatura, sa_raffreddamento,
    ils_iterazioni, ils_scambi,
)

algoritmi_multi = [
    "Ricerca locale", "Tabu Search", "Simulated Annealing", "Iterated Local Search", "MaxMinGreedy (ottimo)",
]
colori_multi = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]

fig_multi = go.Figure()
for nome, colore in zip(algoritmi_multi, colori_multi):
    fig_multi.add_trace(
        go.Bar(
            x=df_multi["istanza"],
            y=df_multi[nome],
            name=nome,
            marker_color=colore,
        )
    )
fig_multi.update_layout(
    barmode="group",
    xaxis_title="Istanza",
    yaxis_title="Utilità minima",
    xaxis=dict(tickmode="linear", dtick=1),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=60, b=20),
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
    font_color="#0b0b0b",
)
st.plotly_chart(fig_multi, use_container_width=True)

st.dataframe(df_multi.round(2), use_container_width=True, hide_index=True)

migliori = df_multi[algoritmi_multi].max(axis=1)
st.markdown("**Quante volte ciascun metodo eguaglia il migliore trovato:**")
vittorie = {
    nome: int((df_multi[nome].sub(migliori).abs() < 1e-6).sum())
    for nome in algoritmi_multi
}
st.dataframe(
    pd.DataFrame({"Algoritmo": list(vittorie.keys()), "Istanze vinte (su 10)": list(vittorie.values())}),
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.subheader("Dettaglio per istanza e algoritmo")
st.caption(
    "Per ogni istanza: sequenza iniziale (casuale) e relativo valore obiettivo, "
    "sequenza finale trovata da ciascun algoritmo e relativo valore obiettivo."
)

istanza_scelta = st.selectbox("Filtra per istanza", ["Tutte"] + [str(i) for i in range(1, 11)])
df_filtrato = df_dettaglio if istanza_scelta == "Tutte" else df_dettaglio[df_dettaglio["istanza"] == int(istanza_scelta)]

st.dataframe(df_filtrato, use_container_width=True, hide_index=True)
