import logging

from scheduling import (
    genera_istanza_casuale,
    iterated_local_search,
    ricerca_locale_scambio,
    simulated_annealing,
    tabu_search,
)

logger = logging.getLogger(__name__)

N_ISTANZE = 10
N_JOBS = 50


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    risultati = []

    for i in range(N_ISTANZE):
        seed = i
        jobs_by_id = genera_istanza_casuale(N_JOBS, seed=seed, con_release_date=True)
        sequenza_iniziale = list(jobs_by_id.keys())

        _, valore_locale = ricerca_locale_scambio(sequenza_iniziale, jobs_by_id)
        _, valore_tabu = tabu_search(sequenza_iniziale, jobs_by_id, iterazioni=200, tabu_tenure=10)
        _, valore_sa = simulated_annealing(
            sequenza_iniziale, jobs_by_id, iterazioni=2000, temperatura_iniziale=20.0,
            raffreddamento=0.995, seed=seed,
        )
        _, valore_ils = iterated_local_search(
            sequenza_iniziale, jobs_by_id, iterazioni=30, num_scambi_perturbazione=4, seed=seed,
        )

        migliore = max(valore_locale, valore_tabu, valore_sa, valore_ils)

        logger.info(
            "Istanza %2d | Locale: %8.2f | Tabu: %8.2f | SA: %8.2f | ILS: %8.2f | Migliore: %8.2f",
            i + 1, valore_locale, valore_tabu, valore_sa, valore_ils, migliore,
        )

        risultati.append({
            "istanza": i + 1,
            "locale": valore_locale,
            "tabu": valore_tabu,
            "sa": valore_sa,
            "ils": valore_ils,
            "migliore": migliore,
        })

    logger.info("")
    logger.info("Riepilogo — quante volte ciascun metodo eguaglia il migliore trovato:")
    for chiave, nome in [("locale", "Ricerca locale"), ("tabu", "Tabu Search"),
                          ("sa", "Simulated Annealing"), ("ils", "Iterated Local Search")]:
        vittorie = sum(1 for r in risultati if abs(r[chiave] - r["migliore"]) < 1e-6)
        logger.info("  %-22s %d/%d istanze", nome, vittorie, N_ISTANZE)

    return risultati


if __name__ == "__main__":
    main()
