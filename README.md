# Scheduling Euristico

Algoritmo di ricerca locale (scambio a coppie) per la massimizzazione dell'utilità minima in un problema di scheduling, con interfaccia interattiva in Streamlit.

🔗 **App live**: https://scheduling-euristico-fiorentini-valerio.streamlit.app/

## Problema

Dati n job, ciascuno con tempo di processamento `p_j`, tempo di rilascio `r_j` e coefficienti `a_j`, `b_j`, l'utilità di un job è:

```
z_j = b_j - a_j * C_j
```

dove `C_j` è il tempo di completamento. L'obiettivo è trovare la sequenza σ che massimizza `min_j z_j`.

## Algoritmi implementati (`scheduling.py`)

- **Ricerca locale (pairwise swap)** — scambia coppie di job finché non trova più miglioramenti.
- **Tabu Search** — accetta mosse peggiorative evitando di tornare sulle ultime sequenze visitate.
- **Simulated Annealing** — accetta mosse peggiorative con probabilità decrescente nel tempo.
- **Iterated Local Search** — perturba l'ottimo locale e ripete la ricerca.

## Avvio locale

```bash
uv run streamlit run app.py
```

## Deploy su Streamlit Community Cloud

1. Collega questo repository su [share.streamlit.io](https://share.streamlit.io).
2. File principale: `app.py`.
3. Le dipendenze vengono lette da `requirements.txt`.
