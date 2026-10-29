# training/metrics.py

import numpy as np


def compute_metrics(eval_pred):
    """
    Calcola le metriche di valutazione.
    Il Trainer la chiama automaticamente alla fine di ogni epoca.

    Parametri:
        eval_pred: tupla (logits, labels) prodotta dal Trainer
            - logits: predizioni grezze shape [num_esempi, num_classes]
            - labels: etichette vere shape [num_esempi]

    Restituisce:
        dizionario con le metriche, il Trainer lo stampa nel log
    """

    logits, labels = eval_pred
    # spacchetta la tupla nei due componenti
    # logits è un numpy array, il Trainer converte i tensori PyTorch
    # in numpy automaticamente prima di passarli a compute_metrics

    predictions = np.argmax(logits, axis=-1)
    # np.argmax trova l'indice del valore massimo lungo l'ultima dimensione
    # axis=-1 significa "ultima dimensione", cioè quella delle classi
    # logits shape [872, 2] → predictions shape [872]
    # ogni valore è 0 o 1, la classe predetta per ogni frase
    # stesso concetto di logits.argmax(dim=-1) che usavamo in PyTorch

    accuracy = (predictions == labels).mean()
    # confronta ogni predizione con la sua etichetta vera
    # (predictions == labels) → array di True/False
    # .mean() calcola la media, True=1, False=0
    # risultato: percentuale di predizioni corrette

    return {"accuracy": float(accuracy)}
    # float() converte da numpy scalar a float Python
    # il Trainer si aspetta un dizionario con stringhe come chiavi
    # il nome "accuracy" apparirà nel log durante il training:
    # "eval_accuracy": 0.9105


# ------------------------------------------------------------------
# Test rapido
# ------------------------------------------------------------------

if __name__ == "__main__":

    # simuliamo logits e labels per verificare che la funzione funzioni
    logits_finti = np.array(
        [
            [0.1, 0.9],  # predice classe 1, corretto se label=1
            [0.8, 0.2],  # predice classe 0, corretto se label=0
            [0.3, 0.7],  # predice classe 1, sbagliato se label=0
            [0.6, 0.4],  # predice classe 0, corretto se label=0
        ]
    )
    labels_finti = np.array([1, 0, 0, 0])

    risultato = compute_metrics((logits_finti, labels_finti))
    print(f"Accuracy: {risultato['accuracy']:.2f}")
    # atteso: 3/4 = 0.75, tre predizioni corrette su quattro
