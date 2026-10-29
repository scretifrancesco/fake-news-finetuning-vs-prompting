# llm_fake_news

Fake news classification using two approaches: fine-tuning of DistilBERT and a local LLM inference pipeline via Ollama with zero-shot and few-shot prompting strategies.

Built for the LLM and GenAI Course 2026, University of Turin.

**Authors:** Francesco Rolando & Francesco Screti

---

## Project Structure

```
llm_fake_news/
├── README.md
├── PROGETTO.md               # course assignment report (Italian)
├── app.py                    # Streamlit dashboard + live demo
├── .streamlit/config.toml
├── fine_tuning/              # Part 1: DistilBERT fine-tuning
│   ├── config.py
│   ├── train.py
│   ├── requirements.txt
│   ├── TRAINING_SUMMARY.md
│   ├── data/
│   │   └── dataset.py
│   ├── model/
│   │   └── model.py
│   ├── training/
│   │   ├── metrics.py
│   │   └── trainer.py
│   └── results/              # checkpoints (not versioned)
└── ollama_pipeline/          # Part 2: Ollama pipeline
    ├── README.md
    ├── prompts.py
    ├── pipeline.py
    ├── comparator.py
    ├── data/
    │   └── samples.json
    └── results/
        └── outputs_run*.json
```

---

## Setup

### Dependencies

```bash
pip install -r fine_tuning/requirements.txt
```

### Ollama

Install Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama pull gemma2:2b
```

### DistilBERT Checkpoints

Checkpoints are not versioned due to size (~3GB). Regenerate by running:

```bash
cd fine_tuning
python train.py
```

Training takes approximately 90 minutes. Without checkpoints, the app works but the fine-tuning section and DistilBERT live demo will not be available.

---

## Usage

### Ollama Pipeline

```bash
# in a separate terminal
ollama serve

# run the pipeline (repeat for multiple runs)
cd ollama_pipeline
python pipeline.py

# analyze results
python comparator.py
python comparator.py --runs 3
```

### Streamlit App

```bash
streamlit run app.py
```

---

## Results

### DistilBERT fine-tuning

`distilbert-base-uncased` on [`GonzaloA/fake_news`](https://huggingface.co/datasets/GonzaloA/fake_news), trained on the first 1,000 training examples and evaluated on the first 500 validation examples (seed 42, max length 256).

| Config      | Learning rate | Batch | Epochs | Val. accuracy |
| ----------- | ------------- | ----- | ------ | ------------- |
| baseline    | 2e-5          | 16    | 3      | 0.978         |
| high_lr     | 5e-5          | 16    | 3      | 0.974         |
| more_epochs | 2e-5          | 16    | 5      | 0.978         |
| small_batch | 2e-5          | 8     | 3      | **0.980**     |

### Ollama pipeline

`gemma2:2b` on 20 samples from the same dataset, 4 runs per strategy.

| Strategy  | Mean accuracy | Min  | Max  |
| --------- | ------------- | ---- | ---- |
| Zero-shot | 0.94          | 0.90 | 0.95 |
| Few-shot  | 0.91          | 0.85 | 0.95 |

The two parts are evaluated on sets of different size (500 vs 20 examples), so the comparison between fine-tuning and prompting is indicative. On 20 examples a difference of 0.05 corresponds to a single prediction.
