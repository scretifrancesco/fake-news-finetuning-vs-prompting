# Progetto finale, corso LLM e AI Generativa 2026

**Autori:** Francesco Rolando & Francesco Screti  
**Repository:** github.com/francescorolando/llm_fake_news

---

## Parte 1: Fine-tuning

### Modello scelto e motivazione

**distilbert-base-uncased**, versione distillata di BERT con il 40% di parametri in meno e il 97% delle performance originali. Scelto per la leggerezza (66.9M parametri), la velocità di training su hardware consumer e le performance competitive per task di classificazione testuale. L'architettura encoder-only è ideale per task di classificazione in cui non è necessaria la generazione di testo.

### Dataset scelto e motivazione

**GonzaloA/fake_news** (Hugging Face), dataset di articoli in inglese con etichette binarie: `0 = FAKE`, `1 = REAL`. Scelto per la disponibilità pubblica, il bilanciamento delle classi e la rilevanza pratica del task. Sono stati usati i primi 1000 esempi dello split di training e i primi 500 di quello di validation, con seed 42 per la riproducibilità del training.

### Risultati ottenuti

| Configurazione  | LR   | Batch | Epochs | Accuracy   |
| --------------- | ---- | ----- | ------ | ---------- |
| baseline        | 2e-5 | 16    | 3      | 0.9780     |
| high_lr         | 5e-5 | 16    | 3      | 0.9740     |
| more_epochs     | 2e-5 | 16    | 5      | 0.9780     |
| **small_batch** | 2e-5 | **8** | 3      | **0.9800** |

La configurazione migliore è **small_batch** con accuracy 0.9800, ma le differenze tra le configurazioni sono minime (al massimo 3 predizioni su 500). Il learning rate elevato non porta benefici. In tutte e quattro le configurazioni il checkpoint con la validation loss più bassa è quello della seconda epoca, per cui aumentare le epoche non cambia il risultato.

### Difficoltà incontrate

- **Overfitting precoce**: Data l'elevata densità di segnale del dataset, il modello tendeva a convergere già alla seconda epoca. È stato necessario monitorare attentamente la validation loss per individuare il punto di saturazione ed evitare la memorizzazione del noise del training set.
- **Ottimizzazione della lunghezza delle sequenze**: Bilanciare il parametro `max_length` per includere sufficiente contesto dagli articoli (spesso molto lunghi) senza eccedere i limiti di memoria della GPU durante i passaggi di backpropagation.

---

## Parte 2: Pipeline Ollama

### Task scelto e motivazione

**Classificazione binaria di fake news**, stesso task della Parte 1, scelto deliberatamente per rendere possibile un confronto diretto tra fine-tuning e prompting sullo stesso problema. Questo permette di valutare quanto un LLM generalista senza training specifico si avvicini alle performance di un modello fine-tuned.

### Strategie di prompting usate

**Zero-shot**. Il modello riceve solo l'istruzione e il testo da classificare, senza esempi. Il prompt richiede esplicitamente una risposta di una sola parola (FAKE o REAL).

**Few-shot**. Il modello riceve l'istruzione seguita da 4 esempi etichettati (2 FAKE, 2 REAL) prima del testo da classificare. Gli esempi sono stati scelti per coprire stili diversi di notizie false e reali.

Il confronto tra le due strategie sullo stesso input è implementato in `pipeline.py` e analizzato in `comparator.py`.

### Esempio di input/output

**Input** (run 3):

> "History is once again being made thanks to President Obama. On Wednesday, Obama nominated Abid Riaz Qureshi to serve as federal judge, making him the first Muslim ever nominated for a federal judgeship in United States history."

| Strategia | Output raw | Label | Corretto |
| --------- | ---------- | ----- | -------- |
| Zero-shot | `FAKE`     | FAKE  | ✓        |
| Few-shot  | `REAL`     | REAL  | ✗        |

Ground truth: **FAKE**

**Input:**

> "hillary clinton campaign still whining about the fbi november — the hillary clinton whineaton continues after having mounted..."

| Strategia | Output raw | Label | Corretto |
| --------- | ---------- | ----- | -------- |
| Zero-shot | `FAKE`     | FAKE  | ✗        |
| Few-shot  | `FAKE`     | FAKE  | ✗        |

Ground truth: **REAL**

### Considerazioni sui risultati

Valori medi su 4 run, ciascuna su 20 articoli (10 FAKE e 10 REAL).

| Strategia | Accuracy  | Accuracy FAKE | Accuracy REAL |
| --------- | --------- | ------------- | ------------- |
| Zero-shot | **0.938** | 1.00          | 0.88          |
| Few-shot  | 0.913     | 0.98          | 0.85          |

**Zero-shot supera few-shot (0.94 vs 0.91)**, ma la differenza media corrisponde a circa mezza predizione su 20 e va letta con cautela. Gli esempi del few-shot sono stilisticamente stereotipati (titoli sensazionalistici, teorie del complotto esplicite), ed è plausibile che questo spieghi il caso dell'articolo id 1, dal tono istituzionale ma falso, che nella run 3 il few-shot classifica come reale.

**L'errore condiviso** (articolo Clinton) mostra un limite strutturale del prompting: articoli reali scritti con linguaggio aggressivo o partigiano vengono classificati come fake da entrambe le strategie. Questo tipo di errore richiederebbe esempi di addestramento specifici.

**Confronto con la Parte 1:**

| Approccio             | Accuracy  |
| --------------------- | --------- |
| Zero-shot (gemma2:2b) | 0.938     |
| Few-shot (gemma2:2b)  | 0.913     |
| DistilBERT fine-tuned | **0.980** |

Il fine-tuning rimane superiore, ma il gap è contenuto considerando che gemma2:2b non ha ricevuto alcun training specifico sul task. Il confronto è indicativo, perché DistilBERT è valutato su 500 articoli e gemma2:2b su 20. Il prompting è una baseline solida quando non si dispone di dati etichettati.

---

## Presentazione finale

### 1. Introduzione al progetto

**Task scelto:** classificazione binaria di fake news (FAKE/REAL). Scelto per la rilevanza pratica, la disponibilità di dataset pubblici e la possibilità di confrontare i due approcci sullo stesso problema.

**Dataset:** GonzaloA/fake_news (Hugging Face). Scelto per il bilanciamento delle classi, la qualità delle etichette e le dimensioni adeguate per un fine-tuning su hardware consumer.

**Modelli:**

- Parte 1: distilbert-base-uncased, encoder-only leggero e performante per classificazione
- Parte 2: gemma2:2b via Ollama, LLM locale senza dipendenze da API esterne

### 2. Approccio e sviluppo

**Come abbiamo affrontato il problema:**
L'approccio è stato iterativo: nella Parte 1 abbiamo ottimizzato un classificatore specializzato tramite grid search sugli iperparametri; nella Parte 2 abbiamo testato la robustezza "out-of-the-box" di un LLM generalista, analizzando come diverse strutture di prompt influenzino il processo decisionale.

**Difficoltà teoriche:**

- **Generalizzazione vs Specializzazione**: Analizzare perché un modello compatto (66M parametri) fine-tuned superi un modello da 2B parametri in un task verticale.
- **Bias indotto dal prompting**: Comprendere il motivo del calo di performance nel few-shot dovuto all'ancoraggio stilistico del modello agli esempi forniti.

**Difficoltà pratiche:**

- **Normalizzazione dell'output**: La tendenza del modello a fornire spiegazioni discorsive ha richiesto l'implementazione di un parser robusto per ricondurre l'output alle label FAKE/REAL.
- **Sincronizzazione dei checkpoint**: Assicurare la coerenza tra i log di training e i file di stato del modello per una corretta visualizzazione delle curve di loss.

**Uso degli LLM come supporto:**
_Claude_ è stato utilizzato per la progettazione dell'architettura della pipeline e per il ragionamento strutturato sulla risoluzione dei problemi di parsing; _Copilot_ per la generazione rapida di codice boilerplate per l'interfaccia Streamlit.

**Cosa ha funzionato bene:** Claude per ragionamento strutturato su problemi complessi e spiegazioni dettagliate; Copilot per generazione rapida di boilerplate.

**Cosa non ha funzionato:** Alcune spiegazioni sugli aspetti di configurazione dei path erano inizialmente generiche e hanno richiesto un affinamento manuale basato sulla struttura specifica della repository.

**Qualcosa spiegato in modo sbagliato o fuorviante:** Nella prima versione del codice di visualizzazione, le metriche di training e validation venivano allineate in modo errato a causa di una diversa frequenza di logging. Il problema è stato risolto revisionando manualmente la logica di estrazione dal file `trainer_state.json`.

### 3. Risultati

**Plot delle loss:** disponibili nell'app Streamlit, sezione Fine-tuning. Per ogni configurazione viene mostrato l'andamento di training loss e validation loss per step, letti dai file `trainer_state.json` nei checkpoint.

**Esempi concreti di input/output DistilBERT:**

Input: _"Breaking: new study links 5G towers to memory loss in laboratory rats, scientists demand immediate shutdown."_
→ Predizione: **FAKE** (confidenza: 0.96)

Input: _"The European Central Bank held interest rates steady on Thursday, citing easing inflation and stable growth projections."_
→ Predizione: **REAL** (confidenza: 0.94)

**Schema pipeline Ollama:**

```
samples.json
     ↓
pipeline.py
     ├── zero_shot_prompt(text) → POST /api/generate → parse_label() → FAKE/REAL
     └── few_shot_prompt(text)  → POST /api/generate → parse_label() → FAKE/REAL
     ↓
results/outputs_run{N}.json
     ↓
comparator.py → metriche, stabilità multi-run, disaccordi
```

**Confronto strategie di prompting:**

| Strategia | Accuracy  | FAKE | REAL |
| --------- | --------- | ---- | ---- |
| Zero-shot | **0.938** | 1.00 | 0.88 |
| Few-shot  | 0.913     | 0.98 | 0.85 |

Zero-shot supera few-shot di poco (media su 4 run). La qualità degli esempi nel few-shot è determinante: esempi stereotipati introducono bias stilistici che penalizzano articoli ambigui.
