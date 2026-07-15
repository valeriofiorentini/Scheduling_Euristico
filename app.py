import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scheduling import (
    genera_istanza_casuale,
    iterated_local_search,
    ricerca_locale_scambio,
    simulated_annealing,
    tabu_search,
)

st.set_page_config(page_title="Scheduling - Ricerca Locale", layout="centered")

st.title("Scheduling: massimizzazione dell'utilità minima")
st.caption(
    "Confronto tra Ricerca locale (pairwise swap), Tabu Search, Simulated Annealing "
    "e Iterated Local Search su 10 istanze casuali indipendenti."
)

st.subheader("Confronto su 10 istanze casuali")
st.caption(
    "Genera 10 istanze indipendenti (seed 0-9) con — "
    "a_j ∈ [1,5] (frazionario), b_j ∈ [10,200], p_j ∈ [1,15] — "
    "ed esegue tutti e 4 gli algoritmi su ciascuna."
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
    for i in range(10):
        jobs_istanza = genera_istanza_casuale(int(n_jobs), seed=i)
        seq_istanza = list(jobs_istanza.keys())

        _, v_locale = ricerca_locale_scambio(seq_istanza, jobs_istanza)
        _, v_tabu = tabu_search(
            seq_istanza, jobs_istanza, iterazioni=int(tabu_iterazioni), tabu_tenure=int(tabu_tenure),
        )
        _, v_sa = simulated_annealing(
            seq_istanza, jobs_istanza, iterazioni=int(sa_iterazioni),
            temperatura_iniziale=sa_temperatura, raffreddamento=sa_raffreddamento, seed=i,
        )
        _, v_ils = iterated_local_search(
            seq_istanza, jobs_istanza, iterazioni=int(ils_iterazioni),
            num_scambi_perturbazione=int(ils_scambi), seed=i,
        )

        righe_multi.append({
            "istanza": i + 1,
            "Ricerca locale": v_locale,
            "Tabu Search": v_tabu,
            "Simulated Annealing": v_sa,
            "Iterated Local Search": v_ils,
        })

    return pd.DataFrame(righe_multi)


df_multi = confronta_10_istanze(
    int(n_jobs_multi), tabu_iterazioni, tabu_tenure, sa_iterazioni, sa_temperatura, sa_raffreddamento,
    ils_iterazioni, ils_scambi,
)

algoritmi_multi = ["Ricerca locale", "Tabu Search", "Simulated Annealing", "Iterated Local Search"]
colori_multi = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]

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
