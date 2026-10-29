"""
pipeline principale: carica i sample, interroga ollama con
zero-shot e few-shot, salva i risultati

uso:
    python pipeline.py        → salva nel prossimo run disponibile
    python pipeline.py 2      → salva in results/outputs_run2.json (forzato)
"""

import json
import os
import sys
import requests

from prompts import zero_shot_prompt, few_shot_prompt

# endpoint locale di ollama per le richieste di generazione
OLLAMA_URL = "http://localhost:11434/api/generate"
# modello utilizzato per la classificazione
MODEL = "gemma2:2b"

# percorso dei dati di input contenenti i sample da classificare
SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "data", "samples.json")
# cartella dove salvare i risultati delle run
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def next_run_id() -> int:
    """trova il prossimo run_id disponibile contando i file esistenti di output"""
    if not os.path.exists(RESULTS_DIR):
        return 1
    existing = [
        f
        for f in os.listdir(RESULTS_DIR)
        if f.startswith("outputs_run") and f.endswith(".json")
    ]
    return len(existing) + 1


def query_ollama(prompt: str, model: str = MODEL) -> str:
    """invia il prompt al modello ollama e recupera la risposta"""
    # richiesta POST all'API di ollama con stream disabilitato per risposta completa
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def parse_label(text: str) -> str:
    """normalizza l'output del modello, ollama non risponde sempre con una parola sola"""
    # converte in maiuscolo e estrae l'etichetta dalla risposta
    upper = text.strip().upper()
    if "FAKE" in upper:
        return "FAKE"
    if "REAL" in upper:
        return "REAL"
    # se non riconosce l'etichetta la marca come sconosciuta
    return "UNKNOWN"


def run_pipeline(run_id: int):
    """esegue la pipeline di classificazione per una run specifica"""
    output_path = os.path.join(RESULTS_DIR, f"outputs_run{run_id}.json")

    # carica i sample dal file di input
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # crea la cartella di output se non esiste
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    # elabora ogni sample con entrambe le strategie
    for i, sample in enumerate(samples):
        text = sample["text"]
        ground_truth = sample["label_str"]

        print(f"[{i+1}/{len(samples)}] ground truth: {ground_truth}")

        # ottiene le risposte dal modello per zero-shot e few-shot
        zs_raw = query_ollama(zero_shot_prompt(text))
        fs_raw = query_ollama(few_shot_prompt(text))

        # normalizza le risposte in etichette standardizzate
        zs_label = parse_label(zs_raw)
        fs_label = parse_label(fs_raw)

        print(f"  zero-shot → {zs_label}  |  few-shot → {fs_label}")

        # salva il risultato con metadati e correttezza per entrambe le strategie
        results.append(
            {
                "id": i,
                "text_preview": text[:200],
                "ground_truth": ground_truth,
                "zero_shot": {
                    "raw": zs_raw,
                    "label": zs_label,
                    "correct": zs_label == ground_truth,
                },
                "few_shot": {
                    "raw": fs_raw,
                    "label": fs_label,
                    "correct": fs_label == ground_truth,
                },
            }
        )

    # serializza i risultati in JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nrisultati salvati in {output_path}")
    return results


if __name__ == "__main__":
    run_id = int(sys.argv[1]) if len(sys.argv) > 1 else next_run_id()
    print(f"run {run_id}")
    run_pipeline(run_id)
