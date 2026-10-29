# training/trainer.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import (
    OUTPUT_DIR,
    EPOCHS,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    EVAL_STRATEGY,
    SAVE_STRATEGY,
    LOGGING_STEPS,
    SEED,
)

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding


def build_training_args():
    """
    Configura gli iperparametri del training.
    Tutto viene da config.py, nessun numero hardcoded qui.
    """
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        # cartella dove salvare i checkpoint e i log
        # viene creata automaticamente se non esiste
        num_train_epochs=EPOCHS,
        # numero di volte che il modello vede tutto il training set
        # 3 epoche sono sufficienti per SST-2 con un modello pre-addestrato
        # più epoche → rischio overfitting
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        # quante frasi per batch durante il training
        # "per_device" significa per GPU/CPU, se hai 2 GPU il batch totale
        # sarebbe TRAIN_BATCH_SIZE * 2
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        # batch size per la valutazione, può essere più grande del training
        # perché non calcoliamo i gradienti → meno memoria
        learning_rate=LEARNING_RATE,
        # quanto modificare i pesi ad ogni step
        # 2e-5 è il valore raccomandato dal paper BERT per il fine-tuning
        # troppo alto → catastrophic forgetting del pre-training
        # troppo basso → convergenza lentissima
        eval_strategy=EVAL_STRATEGY,
        # quando valutare, "epoch" significa alla fine di ogni epoca
        # alternativa: "steps" valuta ogni N step
        save_strategy=SAVE_STRATEGY,
        # quando salvare i checkpoint, coerente con eval_strategy
        # "epoch" salva alla fine di ogni epoca
        save_total_limit=2,
        # tenere solo gli ultimi due checkpoint; evita di riempire il disco
        load_best_model_at_end=True,
        # alla fine del training carica il modello con la migliore
        # val accuracy, non necessariamente l'ultimo checkpoint
        # è l'early stopping implicito che cercavamo in train.py
        logging_steps=LOGGING_STEPS,
        # stampa la loss ogni LOGGING_STEPS step
        # utile per monitorare il training in tempo reale
        seed=SEED,
        # fissa il seed per la riproducibilità
        # stesso seed = stessi risultati ad ogni run
    )


def build_trainer(model, tokenized_dataset, tokenizer, compute_metrics):
    """
    Costruisce il Trainer con tutti gli ingredienti.

    Parametri:
        model:             il modello caricato da model.py
        tokenized_dataset: il dataset tokenizzato da dataset.py
        tokenizer:         serve per il DataCollator
        compute_metrics:   funzione di valutazione da metrics.py
    """

    training_args = build_training_args()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # il DataCollator si occupa del padding dinamico per ogni batch
    # porta tutte le sequenze del batch alla lunghezza della più lunga
    # più efficiente che paddare tutto a MAX_LENGTH in anticipo,
    # un batch di frasi corte non viene inutilmente allungato

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        # usiamo validation, non test, per valutare durante il training
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    return trainer


# ------------------------------------------------------------------
# Test rapido
# ------------------------------------------------------------------

if __name__ == "__main__":
    args = build_training_args()
    print(f"Output dir:      {args.output_dir}")
    print(f"Epoche:          {args.num_train_epochs}")
    print(f"Batch size:      {args.per_device_train_batch_size}")
    print(f"Learning rate:   {args.learning_rate}")
    print(f"Eval strategy:   {args.eval_strategy}")
