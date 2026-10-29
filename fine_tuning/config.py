import os
from datetime import datetime

# ── model ─────────────────────────────────────────────────────────────
MODEL = "distilbert-base-uncased"
NUM_LABELS = 2  # FAKE o REAL


# ── dataset ───────────────────────────────────────────────────────────
DATASET = "GonzaloA/fake_news"
DATASET_CONFIG = None  # nessun sottoconfig
TEXT_COL = "text"
LABEL_COL = "label"  # etichette 0=FAKE, 1=REAL
MAX_LENGTH = 256  # lunghezza massima dei testi (tokenizzati) info rilevanti nelle prime righe del testo


# ── training ──────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
SAVE_STRATEGY = "epoch"


# ── campionamento per la demo ─────────────────────────────────────────
MAX_TRAIN_SAMPLES = 1000
MAX_EVAL_SAMPLES = 500


# ── configurazioni iperparametri da testare ───────────────────────────
CONFIGS = [
    {"name": "baseline", "lr": 2e-5, "batch": 16, "epochs": 3, "wd": 0.01},
    {
        "name": "high_lr",
        "lr": 5e-5,
        "batch": 16,
        "epochs": 3,
        "wd": 0.01,
    },  # imparare più velocemente migliora o peggiora i risultati?
    {
        "name": "more_epochs",
        "lr": 2e-5,
        "batch": 16,
        "epochs": 5,
        "wd": 0.01,
    },  # più tempo al modello migliora i risultati o overfitting?
    {
        "name": "small_batch",
        "lr": 2e-5,
        "batch": 8,
        "epochs": 3,
        "wd": 0.01,
    },  # aggiornare i pesi più spesso cambia qualcosa?
]


# ── output ────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "checkpoints",
    f"{MODEL}_lr{LEARNING_RATE}_bs{TRAIN_BATCH_SIZE}_ep{EPOCHS}_{timestamp}",
)


# ── evaluation ────────────────────────────────────────────────────────
EVAL_STRATEGY = "epoch"
EVAL_STEPS = 100
LOGGING_STEPS = 10


# ── reproducibility ───────────────────────────────────────────────────
SEED = 42
