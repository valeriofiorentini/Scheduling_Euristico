import logging
import math
import random

logger = logging.getLogger(__name__)


class Job:
    def __init__(self, id, processing_j, release_time_j, coefficiente_a_j, coefficiente_b_j):
        self.id = id
        self.processing_j = processing_j          # p_j: tempo di processamento
        self.release_time_j = release_time_j      # r_j: tempo di rilascio
        self.coefficiente_a_j = coefficiente_a_j   # a_j
        self.coefficiente_b_j = coefficiente_b_j   # b_j


def genera_istanza_casuale(n_jobs, seed=None):
    """Genera un'istanza casuale di n_jobs job.

    a_j uniforme in [1, 5] (frazionario), b_j uniforme in [10, 200],
    p_j uniforme in [1, 15], r_j uniforme in [0, 2*n_jobs] per distribuire
    i rilasci su un orizzonte proporzionato al numero di job.
    """
    rng = random.Random(seed)
    jobs_by_id = {}
    for job_id in range(1, n_jobs + 1):
        jobs_by_id[job_id] = Job(
            id=job_id,
            processing_j=rng.uniform(1, 15),
            release_time_j=rng.uniform(0, 2 * n_jobs),
            #coefficiente_a_j=rng.uniform(1, 5),
            #@todo overfitting
            #coefficiente_a_j=rng.uniform(0.1, 1),
            coefficiente_a_j=rng.uniform(0.1, 2),
            #coefficiente_b_j=rng.uniform(10, 200),
            #@todo overfitting
            coefficiente_b_j=rng.uniform(200, 2000),
        )
    return jobs_by_id


def calcola_utilita_minima(sequenza, jobs_by_id):
    """simula l'esecuzione della sequenza e restituisce l'utilita' minima tra i job."""
    tempo_corrente = 0.0
    utilita_minima = float("inf")

    for job_id in sequenza:
        job = jobs_by_id[job_id]
        tempo_corrente = max(tempo_corrente, job.release_time_j) + job.processing_j
        utilita = job.coefficiente_b_j - job.coefficiente_a_j * tempo_corrente
        utilita_minima = min(utilita_minima, utilita)

    return utilita_minima


def ricerca_locale_scambio(sequenza_iniziale, jobs_by_id):
    """ricerca locale con scambi a coppie (pairwise swap).

    continua a esplorare finche non trova piu' alcun miglioramento
    dell'utilita' minima; se lo trova, adotta la nuova sequenza e riparte.
    """
    sequenza_corrente = list(sequenza_iniziale)
    utilita_corrente = calcola_utilita_minima(sequenza_corrente, jobs_by_id)

    migliorata = True
    while migliorata:
        migliorata = False
        n = len(sequenza_corrente)

        for i in range(n):
            for k in range(i + 1, n):
                candidata = list(sequenza_corrente)
                candidata[i], candidata[k] = candidata[k], candidata[i]
                utilita_candidata = calcola_utilita_minima(candidata, jobs_by_id)

                if utilita_candidata > utilita_corrente:
                    sequenza_corrente = candidata
                    utilita_corrente = utilita_candidata
                    migliorata = True

        # ricomincia la scansione degli scambi dalla sequenza aggiornata

    return sequenza_corrente, utilita_corrente


def vicini_scambio(sequenza):
    """genera tutte le sequenze ottenibili con uno scambio a coppie."""
    n = len(sequenza)
    for i in range(n):
        for k in range(i + 1, n):
            candidata = list(sequenza)
            candidata[i], candidata[k] = candidata[k], candidata[i]
            yield candidata


def tabu_search(sequenza_iniziale, jobs_by_id, iterazioni=100, tabu_tenure=5):
    """Tabu Search come la ricerca locale, ma accetta anche mosse peggiorative
    per uscire dai minimi locali, evitando pero di tornare su sequenze visitate di recente (tabu list).
    """
    sequenza_corrente = list(sequenza_iniziale)
    utilita_corrente = calcola_utilita_minima(sequenza_corrente, jobs_by_id)

    migliore_sequenza = list(sequenza_corrente)
    migliore_utilita = utilita_corrente

    tabu_list = []  # coda di sequenze (tuple) vietate temporaneamente

    for _ in range(iterazioni):
        vicini = list(vicini_scambio(sequenza_corrente))
        migliore_vicino = None
        migliore_valore_vicino = float("-inf")

        for candidata in vicini:
            if tuple(candidata) in tabu_list:
                continue
            valore = calcola_utilita_minima(candidata, jobs_by_id)
            if valore > migliore_valore_vicino:
                migliore_valore_vicino = valore
                migliore_vicino = candidata

        if migliore_vicino is None:
            break  # tutti i vicini sono tabu

        sequenza_corrente = migliore_vicino
        utilita_corrente = migliore_valore_vicino

        tabu_list.append(tuple(sequenza_corrente))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        if utilita_corrente > migliore_utilita:
            migliore_sequenza = list(sequenza_corrente)
            migliore_utilita = utilita_corrente

    return migliore_sequenza, migliore_utilita


def simulated_annealing(sequenza_iniziale, jobs_by_id, iterazioni=500,
                         temperatura_iniziale=10.0, raffreddamento=0.95, seed=None):
    """Simulated Annealing: accetta uno scambio peggiorativo con probabilita'
    exp(delta / temperatura), con la temperatura che decresce ad ogni iterazione.
    """
    rng = random.Random(seed)

    sequenza_corrente = list(sequenza_iniziale)
    utilita_corrente = calcola_utilita_minima(sequenza_corrente, jobs_by_id)

    migliore_sequenza = list(sequenza_corrente)
    migliore_utilita = utilita_corrente

    temperatura = temperatura_iniziale
    n = len(sequenza_corrente)

    for _ in range(iterazioni):
        if n < 2:
            break

        i, k = rng.sample(range(n), 2)
        candidata = list(sequenza_corrente)
        candidata[i], candidata[k] = candidata[k], candidata[i]
        utilita_candidata = calcola_utilita_minima(candidata, jobs_by_id)

        delta = utilita_candidata - utilita_corrente
        if delta > 0 or rng.random() < math.exp(delta / max(temperatura, 1e-9)):
            sequenza_corrente = candidata
            utilita_corrente = utilita_candidata

            if utilita_corrente > migliore_utilita:
                migliore_sequenza = list(sequenza_corrente)
                migliore_utilita = utilita_corrente

        temperatura *= raffreddamento

    return migliore_sequenza, migliore_utilita


def perturba(sequenza, num_scambi, rng):
    """perturbazione casuale forte, effettua piu' scambi casuali di coppie."""
    perturbata = list(sequenza)
    n = len(perturbata)
    for _ in range(num_scambi):
        if n < 2:
            break
        i, k = rng.sample(range(n), 2)
        perturbata[i], perturbata[k] = perturbata[k], perturbata[i]
    return perturbata


def iterated_local_search(sequenza_iniziale, jobs_by_id, iterazioni=20,
                           num_scambi_perturbazione=2, seed=None):
    """Iterated Local Search: raggiunto un ottimo locale, applica una
    perturbazione casuale (piu di uno scambio) e riparte con la ricerca
    locale, tenendo traccia della migliore soluzione trovata.
    """
    rng = random.Random(seed)

    sequenza_corrente, utilita_corrente = ricerca_locale_scambio(sequenza_iniziale, jobs_by_id)
    migliore_sequenza = list(sequenza_corrente)
    migliore_utilita = utilita_corrente

    for _ in range(iterazioni):
        perturbata = perturba(sequenza_corrente, num_scambi_perturbazione, rng)
        sequenza_locale, utilita_locale = ricerca_locale_scambio(perturbata, jobs_by_id)

        if utilita_locale > migliore_utilita:
            migliore_sequenza = list(sequenza_locale)
            migliore_utilita = utilita_locale

        # criterio di accettazione: si riparte sempre dal nuovo ottimo locale
        sequenza_corrente, utilita_corrente = sequenza_locale, utilita_locale

    return migliore_sequenza, migliore_utilita


def maxmin_greedy(jobs_by_id):
    """MaxMinGreedy (Nicosia, Pacifici, Pferschy 2026): algoritmo esatto O(n^2)
    per 1|uj|umin, cioe' senza release date. Costruisce la sequenza a ritroso:
    a ogni passo assegna all'ultima posizione libera il job con l'utilita'
    piu' alta se completasse esattamente al tempo T corrente, poi riduce T
    del suo processing time e ripete per la posizione precedente.

    Garanzia di ottimalita' valida solo se tutti i release_time_j sono 0;
    con release date il problema e' NP-hard e questo greedy resta euristico.
    """
    da_schedulare = set(jobs_by_id.keys())
    tempo_residuo = sum(job.processing_j for job in jobs_by_id.values())

    n = len(jobs_by_id)
    sequenza = [None] * n

    for posizione in range(n - 1, -1, -1):
        job_scelto = max(
            da_schedulare,
            key=lambda job_id: (
                jobs_by_id[job_id].coefficiente_b_j
                - jobs_by_id[job_id].coefficiente_a_j * tempo_residuo
            ),
        )
        sequenza[posizione] = job_scelto
        tempo_residuo -= jobs_by_id[job_scelto].processing_j
        da_schedulare.remove(job_scelto)

    valore = calcola_utilita_minima(sequenza, jobs_by_id)
    return sequenza, valore


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    jobs = [
        Job(id=1, processing_j=1, release_time_j=0, coefficiente_a_j=1.0, coefficiente_b_j=10.0),
        Job(id=2, processing_j=1, release_time_j=2, coefficiente_a_j=0.0, coefficiente_b_j=20.0),
        Job(id=3, processing_j=1, release_time_j=2, coefficiente_a_j=0.5, coefficiente_b_j=5.0),
        Job(id=4, processing_j=1, release_time_j=0, coefficiente_a_j=2.0, coefficiente_b_j=50.0),
    ]
    jobs_by_id = {job.id: job for job in jobs}

    sequenza_iniziale = [1, 2, 4, 3]
    obiettivo_iniziale = calcola_utilita_minima(sequenza_iniziale, jobs_by_id)

    logger.info("Sequenza iniziale: %s", sequenza_iniziale)
    logger.info("Obiettivo iniziale: %s", obiettivo_iniziale)
    logger.info("Risultato ricerca locale")

    sequenza_ottima, valore_ottimo = ricerca_locale_scambio(sequenza_iniziale, jobs_by_id)
    logger.info("Sequenza ottima locale: %s", sequenza_ottima)
    logger.info("Valore obiettivo: %s", valore_ottimo)
