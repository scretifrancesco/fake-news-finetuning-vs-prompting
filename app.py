"""
Streamlit dashboard: fake news detection, fine-tuning vs prompting
run with: streamlit run app.py
Ollama must be running locally for the live demo
"""

import os
import sys
import json
import requests
import torch
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# importa i prompt dalla pipeline ollama senza __init__.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ollama_pipeline"))
from ollama_pipeline.prompts import zero_shot_prompt, few_shot_prompt

# ── configurazione pagina ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fake News: Fine-tuning vs Prompting",
    page_icon="📰",
    layout="wide",  # TODO : valutare se lasciarlo o meno
    initial_sidebar_state="collapsed",
)

# ── tema chiaro e stile ────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* forza tema chiaro */
[data-testid="stAppViewContainer"] { background: #f8f9fa; }
[data-testid="stHeader"] { background: #f8f9fa; }

/* tab bar come navigazione principale */
[data-testid="stTabs"] button {
    font-size: 0.95rem;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    color: #555;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #1a1a2e;
    border-bottom: 3px solid #e63946;
}

/* card metriche */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 1rem;
}

/* badge risultato */
.badge-fake {
    background: #ffe0e0;
    color: #c0392b;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
}
.badge-real {
    background: #e0f5e0;
    color: #27ae60;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
}
.badge-unknown {
    background: #f0f0f0;
    color: #888;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
}

/* sezione highlight */
.info-box {
    background: #eef2ff;
    border-left: 4px solid #4361ee;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}
.warn-box {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}

/* Pulsante Classifica: testo bianco e sfondo rosso, selettore robusto */
.stButton > button:first-child {
    color: #fff !important;
    background-color: #e63946 !important;
    border: none !important;
    font-weight: 700;
}
.stButton > button:first-child:hover, .stButton > button:first-child:focus {
    color: #fff !important;
    background-color: #b71c1c !important;
    border: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── dati configurazioni fine-tuning ───────────────────────────────────────────

CONFIGS = [
    {
        "name": "baseline",
        "lr": 2e-5,
        "batch": 16,
        "epochs": 3,
        "wd": 0.01,
        "accuracy": 0.9780,
    },
    {
        "name": "high_lr",
        "lr": 5e-5,
        "batch": 16,
        "epochs": 3,
        "wd": 0.01,
        "accuracy": 0.9740,
    },
    {
        "name": "more_epochs",
        "lr": 2e-5,
        "batch": 16,
        "epochs": 5,
        "wd": 0.01,
        "accuracy": 0.9780,
    },
    {
        "name": "small_batch",
        "lr": 2e-5,
        "batch": 8,
        "epochs": 3,
        "wd": 0.01,
        "accuracy": 0.9800,
    },
]

COLORS = {
    "primary": "#1a1a2e",
    "accent": "#e63946",
    "best": "#b3ffb3",
    "default": "#d0d0d0",
    "zero_shot": "#4361ee",
    "few_shot": "#f4a261",
    "distilbert": "#2a9d8f",
}

# ── helper: trova checkpoint ───────────────────────────────────────────────────


def find_checkpoint_dir(config_name: str) -> str | None:
    base = os.path.join("fine_tuning", "results", config_name)
    if not os.path.exists(base):
        return None
    checkpoints = [d for d in os.listdir(base) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    return os.path.join(base, checkpoints[-1])


# ── caching ────────────────────────────────────────────────────────────────────


# fine-tuned model on the Hugging Face Hub, used when no local checkpoint is available
HF_MODEL_ID = os.environ.get("FAKE_NEWS_MODEL", "")


@st.cache_resource(show_spinner="Loading DistilBERT...")
def load_distilbert():
    ckpt = find_checkpoint_dir("small_batch")
    if ckpt is not None and os.path.exists(os.path.join(ckpt, "model.safetensors")):
        source = ckpt
    elif HF_MODEL_ID:
        source = HF_MODEL_ID
    else:
        raise FileNotFoundError(
            "fine-tuned model not found: run fine_tuning/train.py "
            "or set FAKE_NEWS_MODEL to a Hugging Face model id"
        )
    model = AutoModelForSequenceClassification.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model.eval()
    return model, tokenizer


@st.cache_data(show_spinner=False)
def load_trainer_state(config_name: str) -> dict | None:
    ckpt = find_checkpoint_dir(config_name)
    if ckpt is None:
        return None
    path = os.path.join(ckpt, "trainer_state.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_ollama_runs() -> list[list]:
    results_dir = os.path.join("ollama_pipeline", "results")
    if not os.path.exists(results_dir):
        return []
    files = sorted(
        f
        for f in os.listdir(results_dir)
        if f.startswith("outputs_run") and f.endswith(".json")
    )
    runs = []
    for f in files:
        with open(os.path.join(results_dir, f)) as j:
            runs.append(json.load(j))
    return runs


def parse_label(text: str) -> str:
    upper = text.strip().upper()
    if "FAKE" in upper:
        return "FAKE"
    if "REAL" in upper:
        return "REAL"
    return "UNKNOWN"


def distilbert_predict(text: str) -> str:
    model, tokenizer = load_distilbert()
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return "FAKE" if pred == 0 else "REAL"


def ollama_predict(prompt: str) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "gemma2:2b", "prompt": prompt, "stream": False},
        timeout=30,
    )
    resp.raise_for_status()
    return parse_label(resp.json()["response"])


def label_badge(label: str) -> str:
    cls = {"FAKE": "badge-fake", "REAL": "badge-real"}.get(label, "badge-unknown")
    return f'<span class="{cls}">{label}</span>'


# ── navigazione con tabs ───────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📋 Overview",
        "🧠 Fine-tuning",
        "🤖 Ollama pipeline",
        "⚖️ Comparison",
        "🎯 Live demo",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: INTRODUZIONE
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.title("📰 Fake News Detection: Fine-tuning vs Prompting")
    st.markdown(
        "**Automatic classification of news articles as FAKE or REAL** with two approaches, fine-tuning an encoder-only model and prompting a local LLM."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("The project")
        st.markdown(
            """
        <div class="info-box">
        <b>Task:</b> binary classification, each article is labelled <b>FAKE</b> or <b>REAL</b>.<br><br>
        <b>Dataset:</b> <a href="https://huggingface.co/datasets/GonzaloA/fake_news">GonzaloA/fake_news</a> on Hugging Face,
        a balanced dataset of English articles labelled 0 = FAKE, 1 = REAL.<br><br>
        <b>Why this task?</b> Fake news classification is a real-world problem with a public,
        well-structured dataset, and it allows a direct comparison of the two
        strategies on the same problem.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.subheader("The two approaches")
        st.markdown("""
        | | Part 1 | Part 2 |
        |---|---|---|
        | Model | DistilBERT | gemma2:2b |
        | Strategy | Fine-tuning | Prompting |
        | Training | Yes (1,000 examples) | No |
        | Accuracy | **0.980** | 0.94 (zero-shot, mean of 4 runs) |
        """)

    with col2:
        st.subheader("The models")
        st.markdown("""
        **DistilBERT** (Part 1)
        - Distilled version of BERT, 40% fewer parameters and 97% of its performance
        - Encoder-only architecture, well suited to text classification
        - Pretrained on general text, specialised through fine-tuning
        - 66.9M parameters, all trainable

        **gemma2:2b** (Part 2)
        - Google model with 2 billion parameters
        - Runs locally through Ollama, no external API
        - Used without task-specific training, through prompting only
        """)

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("⚠️ Challenges")
        st.markdown("""
        **Theoretical**
        - **Generalisation vs specialisation**: understanding why a compact fine-tuned model (66M parameters) outperforms a 2B-parameter model on a narrow task.
        - **Prompt-induced bias**: understanding the few-shot drop, caused by the model anchoring on the style of the examples.

        **Practical**
        - **Output normalisation**: the model tends to answer with explanations, so a robust parser maps its output to the FAKE/REAL labels.
        - **Checkpoint consistency**: keeping training logs and model state files aligned so that loss curves are plotted correctly.
        """)

    with col4:
        st.subheader("🤖 Use of LLMs as assistants")
        st.markdown("""
        - _Claude_ helped design the pipeline architecture and reason through parsing issues.
        - _Copilot_ generated boilerplate code for the Streamlit interface.

        **What worked:** Claude for structured reasoning on complex problems and detailed explanations, Copilot for fast boilerplate.

        **What did not:** some explanations about path configuration were generic at first and had to be adapted by hand to the structure of the repository.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.title("🧠 Fine-tuning DistilBERT")
    st.markdown(
        "Fine-tuning `distilbert-base-uncased` for binary fake news classification with 4 hyperparameter configurations."
    )

    st.divider()

    # ── SEZIONE: STRUTTURA DEL CODICE ──
    st.subheader("Code structure")

    st.markdown("""
    The code adapts a fine-tuning script for SST-2 (sentiment analysis), replacing the dataset
    and adding support for 4 hyperparameter configurations compared in sequence.
    """)

    col_struct1, col_struct2, col_struct3 = st.columns(3)

    with col_struct1:
        st.markdown(
            """
        <div class="card" style="background:#fff; border:1px solid #e8e4df; border-radius:12px; padding:16px; margin:8px 0">
        <b style="font-size:0.85rem; color:#7a7470; display:block; margin-bottom:8px">config.py</b>
        <div style="font-size:0.85rem; color:#7a7470; line-height:1.5">
        Control panel with every project constant: model, dataset and the 4 hyperparameter configurations.
        No hardcoded numbers elsewhere.
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="card" style="background:#fff; border:1px solid #e8e4df; border-radius:12px; padding:16px; margin:8px 0">
        <b style="font-size:0.85rem; color:#7a7470; display:block; margin-bottom:8px">data/dataset.py</b>
        <div style="font-size:0.85rem; color:#7a7470; line-height:1.5">
        Downloads the dataset from Hugging Face, tokenizes the articles and keeps 1,000 training examples.
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_struct2:
        st.markdown(
            """
        <div class="card" style="background:#fff; border:1px solid #e8e4df; border-radius:12px; padding:16px; margin:8px 0">
        <b style="font-size:0.85rem; color:#7a7470; display:block; margin-bottom:8px">model/model.py</b>
        <div style="font-size:0.85rem; color:#7a7470; line-height:1.5">
        Loads pretrained DistilBERT with a classification head with 2 outputs (FAKE/REAL).
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="card" style="background:#fff; border:1px solid #e8e4df; border-radius:12px; padding:16px; margin:8px 0">
        <b style="font-size:0.85rem; color:#7a7470; display:block; margin-bottom:8px">training/metrics.py</b>
        <div style="font-size:0.85rem; color:#7a7470; line-height:1.5">
        Computes accuracy. The Trainer calls it at the end of every epoch.
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_struct3:
        st.markdown(
            """
        <div class="card" style="background:#fff; border:1px solid #e8e4df; border-radius:12px; padding:16px; margin:8px 0">
        <b style="font-size:0.85rem; color:#7a7470; display:block; margin-bottom:8px">training/trainer.py</b>
        <div style="font-size:0.85rem; color:#7a7470; line-height:1.5">
        Builds the Hugging Face Trainer from dataset, model and metric, and handles checkpoint saving.
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="card" style="background:#fff; border:1px solid #e8e4df; border-radius:12px; padding:16px; margin:8px 0">
        <b style="font-size:0.85rem; color:#7a7470; display:block; margin-bottom:8px">train.py</b>
        <div style="font-size:0.85rem; color:#7a7470; line-height:1.5">
        Entry point. Loads the data once, then loops over the 4 configurations, training a fresh model for each.
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── SEZIONE: IPERPARAMETRI ──
    st.subheader("The hyperparameters")

    st.markdown("""
    | Parameter | Meaning | Values tested |
    |---|---|---|
    | **lr** (learning rate) | The step size of each weight update. **2e-5** gives small, stable steps. **5e-5** learns faster but may overshoot the optimum. | `2e-5` vs `5e-5` |
    | **batch** | How many articles the model sees before updating its weights. **Batch 16** averages the error over 16 articles. **Batch 8** updates twice as often, on fewer examples. | `16` vs `8` |
    | **epochs** | How many times the model sees the whole training set. With 1,000 articles and 3 epochs, it sees 3,000 articles in total. More epochs raise the risk of memorisation. | `3` vs `5` |
    | **wd** (weight decay) | Regularisation that shrinks the weights at every step to limit overfitting. | `0.01` (fixed) |
    """)

    st.divider()

    # metriche riassuntive
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "DistilBERT")
    c2.metric("Parameters", "66.9M")
    c3.metric("Best accuracy", "0.9800")
    c4.metric("Best config", "small_batch")

    st.divider()
    st.subheader("Configurations tested")

    df_cfg = pd.DataFrame(CONFIGS)
    best_idx = df_cfg["accuracy"].idxmax()

    def highlight_best(row):
        return [
            "background-color: #b3ffb3" if row.name == best_idx else "" for _ in row
        ]

    st.dataframe(
        df_cfg[["name", "lr", "batch", "epochs", "wd", "accuracy"]]
        .style.apply(highlight_best, axis=1)
        .format({"lr": "{:.0e}", "accuracy": "{:.4f}"}),
        width="stretch",
    )

    # bar chart accuracy
    fig_acc = go.Figure(
        [
            go.Bar(
                x=[c["name"] for c in CONFIGS],
                y=[c["accuracy"] for c in CONFIGS],
                marker_color=[
                    COLORS["best"] if i == best_idx else COLORS["default"]
                    for i in range(len(CONFIGS))
                ],
                text=[f"{c['accuracy']:.4f}" for c in CONFIGS],
                textposition="outside",
            )
        ]
    )
    fig_acc.update_layout(
        title="Validation accuracy by configuration",
        yaxis=dict(title="Accuracy", range=[0.96, 0.985]),
        xaxis_title="Configuration",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380,
    )
    st.plotly_chart(fig_acc, width="stretch")

    st.divider()
    st.subheader("Loss during training")

    for cfg in CONFIGS:
        state = load_trainer_state(cfg["name"])
        if state is None:
            st.warning(f"trainer_state.json not found for {cfg['name']}")
            continue

        log = state.get("log_history", [])

        # separa correttamente training loss e validation loss
        train_logs = [
            (x["step"], x["loss"]) for x in log if "loss" in x and "eval_loss" not in x
        ]
        eval_logs = [(x["step"], x["eval_loss"]) for x in log if "eval_loss" in x]

        if not train_logs:
            st.warning(f"No training loss data for {cfg['name']}")
            continue

        fig_loss = go.Figure()
        fig_loss.add_trace(
            go.Scatter(
                x=[s for s, _ in train_logs],
                y=[l for _, l in train_logs],
                mode="lines+markers",
                name="Training loss",
                line=dict(color=COLORS["zero_shot"]),
            )
        )
        if eval_logs:
            fig_loss.add_trace(
                go.Scatter(
                    x=[s for s, _ in eval_logs],
                    y=[l for _, l in eval_logs],
                    mode="lines+markers",
                    name="Validation loss",
                    line=dict(color=COLORS["accent"], dash="dash"),
                )
            )
        fig_loss.update_layout(
            title=f"{cfg['name']}, lr={cfg['lr']:.0e}, batch={cfg['batch']}, epochs={cfg['epochs']}",
            xaxis_title="Step",
            yaxis_title="Loss",
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=320,
        )
        st.plotly_chart(fig_loss, width="stretch")

    st.divider()
    st.subheader("Effect of each hyperparameter")

    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown("**Learning rate**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Config": ["baseline (2e-5)", "high_lr (5e-5)"],
                    "Accuracy": [0.9780, 0.9740],
                    "Δ": ["-", "-0.004"],
                }
            ),
            width="stretch",
        )
        st.markdown(
            '<div class="warn-box">A higher learning rate does not help, 0.974 against 0.978 for the baseline (2 predictions out of 500).</div>',
            unsafe_allow_html=True,
        )
    with cb:
        st.markdown("**Batch size**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Config": ["baseline (16)", "small_batch (8)"],
                    "Accuracy": [0.9780, 0.9800],
                    "Δ": ["-", "+0.002"],
                }
            ),
            width="stretch",
        )
        st.markdown(
            '<div class="info-box">A smaller batch means more frequent weight updates. The gain is minimal (1 prediction out of 500).</div>',
            unsafe_allow_html=True,
        )
    with cc:
        st.markdown("**Epochs**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Config": ["baseline (3)", "more_epochs (5)"],
                    "Accuracy": [0.9780, 0.9780],
                    "Δ": ["-", "="],
                }
            ),
            width="stretch",
        )
        st.markdown(
            '<div class="warn-box">More epochs do not change the result. In every configuration the checkpoint with the lowest validation loss is the one from epoch 2.</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: PIPELINE OLLAMA
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.title("🤖 Ollama pipeline")
    st.markdown(
        "Fake news classification with **gemma2:2b** running locally, using two prompting strategies, zero-shot and few-shot."
    )

    st.divider()

    # schema pipeline
    st.subheader("Pipeline overview")
    st.markdown(
        """
    <div class="info-box">
    <b>Input</b> → article text from <code>data/samples.json</code><br><br>
    <b>Zero-shot</b>: instruction + text → <code>POST localhost:11434/api/generate</code> → raw answer → <code>parse_label()</code> → FAKE / REAL<br><br>
    <b>Few-shot</b>: instruction + 4 labelled examples + text → same pipeline → FAKE / REAL<br><br>
    <b>Output</b> → saved to <code>results/outputs_run{N}.json</code> with prediction, raw label and correctness for each sample
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("""
    **Why `parse_label()`?**
    The model does not always answer with a single word, even when asked to.
    Answers such as `"The label is FAKE."` or `"fake"` are normalised by searching
    for `FAKE` or `REAL` in the upper-cased text.
    """)

    st.divider()

    all_runs = load_ollama_runs()
    if not all_runs:
        st.error("No outputs_run*.json file found in ollama_pipeline/results/")
    else:
        # selettore run
        run_labels = [f"Run {i+1}" for i in range(len(all_runs))]
        selected_run_idx = 0
        if len(all_runs) > 1:
            selected_run_label = st.selectbox("Select a run", run_labels)
            selected_run_idx = run_labels.index(selected_run_label)

        run = all_runs[selected_run_idx]
        df = pd.DataFrame(run)
        df["zs_label"] = df["zero_shot"].apply(lambda x: x["label"])
        df["fs_label"] = df["few_shot"].apply(lambda x: x["label"])
        df["zs_correct"] = df["zero_shot"].apply(lambda x: x["correct"])
        df["fs_correct"] = df["few_shot"].apply(lambda x: x["correct"])

        # metriche
        st.subheader("Metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        zs_acc = df["zs_correct"].mean()
        fs_acc = df["fs_correct"].mean()
        zs_fake = df[df["ground_truth"] == "FAKE"]["zs_correct"].mean()
        zs_real = df[df["ground_truth"] == "REAL"]["zs_correct"].mean()
        fs_fake = df[df["ground_truth"] == "FAKE"]["fs_correct"].mean()
        fs_real = df[df["ground_truth"] == "REAL"]["fs_correct"].mean()
        m1.metric("Zero-shot accuracy", f"{zs_acc:.2f}")
        m2.metric("Zero-shot FAKE", f"{zs_fake:.2f}")
        m3.metric("Zero-shot REAL", f"{zs_real:.2f}")
        m4.metric("Few-shot accuracy", f"{fs_acc:.2f}")
        m5.metric("Few-shot FAKE", f"{fs_fake:.2f}")
        m6.metric("Few-shot REAL", f"{fs_real:.2f}")

        # bar chart metriche
        fig_m = go.Figure(
            [
                go.Bar(
                    name="Zero-shot",
                    x=["Overall", "FAKE", "REAL"],
                    y=[zs_acc, zs_fake, zs_real],
                    marker_color=COLORS["zero_shot"],
                ),
                go.Bar(
                    name="Few-shot",
                    x=["Overall", "FAKE", "REAL"],
                    y=[fs_acc, fs_fake, fs_real],
                    marker_color=COLORS["few_shot"],
                ),
            ]
        )
        fig_m.update_layout(
            barmode="group",
            title="Accuracy by strategy and class",
            yaxis=dict(title="Accuracy", range=[0.7, 1.05]),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=340,
        )
        st.plotly_chart(fig_m, width="stretch")

        st.divider()

        # tabella sample per sample
        st.subheader("Results by sample")
        display_df = df[
            [
                "id",
                "text_preview",
                "ground_truth",
                "zs_label",
                "fs_label",
                "zs_correct",
                "fs_correct",
            ]
        ].copy()
        display_df.columns = [
            "ID",
            "Text",
            "Ground truth",
            "Zero-shot",
            "Few-shot",
            "ZS ✓",
            "FS ✓",
        ]

        filter_col1, filter_col2 = st.columns(2)
        filter_gt = filter_col1.selectbox(
            "Filter by ground truth", ["All", "FAKE", "REAL"]
        )
        filter_err = filter_col2.selectbox(
            "Filter by errors",
            [
                "All",
                "Zero-shot errors only",
                "Few-shot errors only",
                "Both wrong",
            ],
        )

        filtered = display_df.copy()
        if filter_gt != "All":
            filtered = filtered[filtered["Ground truth"] == filter_gt]
        if filter_err == "Zero-shot errors only":
            filtered = filtered[~filtered["ZS ✓"]]
        elif filter_err == "Few-shot errors only":
            filtered = filtered[~filtered["FS ✓"]]
        elif filter_err == "Both wrong":
            filtered = filtered[~filtered["ZS ✓"] & ~filtered["FS ✓"]]

        st.dataframe(filtered, width="stretch")

        st.divider()

        # casi di disaccordo
        st.subheader("Disagreements between the two strategies")
        disag = df[df["zs_label"] != df["fs_label"]]
        if disag.empty:
            st.success("The two strategies agree on every sample.")
        else:
            st.markdown(
                f"**{len(disag)} case(s)** where zero-shot and few-shot disagree:"
            )
            for _, row in disag.iterrows():
                winner = (
                    "zero-shot"
                    if row["zs_correct"]
                    else ("few-shot" if row["fs_correct"] else "neither")
                )
                st.markdown(
                    f"""
                <div class="warn-box">
                <b>ID {int(row['id'])}, GT: {row['ground_truth']}</b><br>
                Zero-shot: <b>{row['zs_label']}</b> &nbsp;|&nbsp; Few-shot: <b>{row['fs_label']}</b> &nbsp;|&nbsp; Correct: <b>{winner}</b><br><br>
                <i>{row['text_preview'][:200]}...</i>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # stabilità multi-run
        if len(all_runs) > 1:
            st.divider()
            st.subheader(f"Stability across {len(all_runs)} runs")

            zs_accs = [
                pd.DataFrame(r)["zero_shot"].apply(lambda x: x["correct"]).mean()
                for r in all_runs
            ]
            fs_accs = [
                pd.DataFrame(r)["few_shot"].apply(lambda x: x["correct"]).mean()
                for r in all_runs
            ]

            sr1, sr2 = st.columns(2)
            sr1.metric(
                "Zero-shot mean ± std",
                f"{pd.Series(zs_accs).mean():.3f} ± {pd.Series(zs_accs).std():.3f}",
            )
            sr2.metric(
                "Few-shot mean ± std",
                f"{pd.Series(fs_accs).mean():.3f} ± {pd.Series(fs_accs).std():.3f}",
            )

            fig_runs = go.Figure()
            fig_runs.add_trace(
                go.Scatter(
                    x=list(range(1, len(all_runs) + 1)),
                    y=zs_accs,
                    mode="lines+markers",
                    name="Zero-shot",
                    line=dict(color=COLORS["zero_shot"]),
                )
            )
            fig_runs.add_trace(
                go.Scatter(
                    x=list(range(1, len(all_runs) + 1)),
                    y=fs_accs,
                    mode="lines+markers",
                    name="Few-shot",
                    line=dict(color=COLORS["few_shot"]),
                )
            )
            fig_runs.update_layout(
                title="Accuracy by run",
                xaxis_title="Run",
                yaxis_title="Accuracy",
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=300,
            )
            st.plotly_chart(fig_runs, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: CONFRONTO
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.title("⚖️ Comparing the approaches")
    st.markdown("Fine-tuning vs prompting, and when to use each.")

    st.divider()

    # tabella comparativa
    comp_data = {
        "Approach": ["DistilBERT (best)", "Zero-shot gemma2", "Few-shot gemma2"],
        "Accuracy": [0.9800, 0.9375, 0.9125],
        "Acc. FAKE": ["-", "1.00", "0.98"],
        "Acc. REAL": ["-", "0.88", "0.85"],
        "Training": ["Yes", "No", "No"],
        "Resources": ["High", "Low", "Low"],
    }
    st.dataframe(pd.DataFrame(comp_data), width="stretch")
    st.caption("DistilBERT on 500 validation articles, gemma2 as the mean of 4 runs on 20 articles.")

    # bar chart confronto
    fig_comp = go.Figure(
        [
            go.Bar(
                x=comp_data["Approach"],
                y=comp_data["Accuracy"],
                marker_color=[
                    COLORS["distilbert"],
                    COLORS["zero_shot"],
                    COLORS["few_shot"],
                ],
                text=[f"{a:.2f}" for a in comp_data["Accuracy"]],
                textposition="outside",
            )
        ]
    )
    fig_comp.update_layout(
        title="Accuracy comparison",
        yaxis=dict(title="Accuracy", range=[0.85, 1.0]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=360,
    )
    st.plotly_chart(fig_comp, width="stretch")

    st.divider()

    st.subheader("Examples")

    ex1, ex2 = st.columns(2)
    with ex1:
        st.markdown(
            """
        <div class="warn-box">
        <b>Run 3: zero-shot is right, few-shot is wrong</b><br><br>
        <i>"History is once again being made thanks to President Obama. On Wednesday,
        Obama nominated Abid Riaz Qureshi to serve as..."</i><br><br>
        Ground truth: <b>FAKE</b><br>
        Zero-shot: ✅ FAKE<br>
        Few-shot: ❌ REAL<br><br>
        In run 3 the few-shot prompt labelled this article REAL, plausibly because its institutional tone
        resembled the real examples in the prompt. The examples introduced a stylistic bias
        instead of helping.
        </div>
        """,
            unsafe_allow_html=True,
        )
    with ex2:
        st.markdown(
            """
        <div class="warn-box">
        <b>Shared error: both strategies are wrong in all 4 runs</b><br><br>
        <i>"hillary clinton campaign still whining about the fbi november —
        the hillary clinton whineaton continues after having m..."</i><br><br>
        Ground truth: <b>REAL</b><br>
        Zero-shot: ❌ FAKE<br>
        Few-shot: ❌ FAKE<br><br>
        Aggressive, partisan language that resembles fake content in style,
        although the article is real. Correcting this kind of error requires task-specific training.
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("When to use each approach")

    ca, cb = st.columns(2)
    with ca:
        st.markdown(
            """
        <div class="info-box">
        <b>Fine-tuning</b> when:<br>
        • you have enough labelled data (even a few hundred examples)<br>
        • you need the highest possible accuracy<br>
        • the task does not change (production)<br>
        • you have resources for training<br>
        • consistent answers are critical
        </div>
        """,
            unsafe_allow_html=True,
        )
    with cb:
        st.markdown(
            """
        <div class="info-box">
        <b>Prompting</b> when:<br>
        • you have no labelled data<br>
        • you need results right away<br>
        • the task changes often<br>
        • you want to prototype quickly<br>
        • an accuracy around 0.94 is enough
        </div>
        """,
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: DEMO LIVE
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.title("🎯 Live demo")
    st.markdown(
        "Paste an article and compare the predictions of the three classifiers in real time."
    )

    st.markdown(
        """
    <div class="warn-box">
    ⚠️ Ollama must be running locally (<code>ollama serve</code>) with the <code>gemma2:2b</code> model installed.
    DistilBERT is loaded from <code>fine_tuning/results/small_batch/</code> or, if missing, from the Hugging Face model set in <code>FAKE_NEWS_MODEL</code>.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # esempi rapidi
    st.markdown("**Quick examples:**")
    ex_col1, ex_col2 = st.columns(2)
    example_fake = "Scientists confirm that drinking bleach cures cancer. The government has been hiding this information for decades to protect pharmaceutical companies."
    example_real = "The Federal Reserve raised interest rates by 0.25 percentage points on Wednesday, citing persistent inflation concerns and a resilient labor market."

    if ex_col1.button("📋 Insert FAKE example"):
        st.session_state["demo_text"] = example_fake
    if ex_col2.button("📋 Insert REAL example"):
        st.session_state["demo_text"] = example_real

    user_text = st.text_area(
        "Article text:",
        value=st.session_state.get("demo_text", ""),
        height=150,
        placeholder="Paste the text of a news article here...",
    )

    if st.button("🔍 Classify", type="primary") and user_text.strip():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### DistilBERT")
            st.caption("fine-tuned on GonzaloA/fake_news")
            with st.spinner("classifying..."):
                try:
                    label = distilbert_predict(user_text)
                    st.markdown(label_badge(label), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")

        with col2:
            st.markdown("### Zero-shot")
            st.caption("gemma2:2b, no examples")
            with st.spinner("querying Ollama..."):
                try:
                    label = ollama_predict(zero_shot_prompt(user_text))
                    st.markdown(label_badge(label), unsafe_allow_html=True)
                except Exception as e:
                    st.error(
                        "Ollama is not available, make sure `ollama serve` is running"
                    )

        with col3:
            st.markdown("### Few-shot")
            st.caption("gemma2:2b, 4 examples in the prompt")
            with st.spinner("querying Ollama..."):
                try:
                    label = ollama_predict(few_shot_prompt(user_text))
                    st.markdown(label_badge(label), unsafe_allow_html=True)
                except Exception as e:
                    st.error(
                        "Ollama is not available, make sure `ollama serve` is running"
                    )
