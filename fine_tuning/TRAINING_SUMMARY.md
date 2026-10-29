# Training summary

## Objective

Fine-tuning of `distilbert-base-uncased` for binary classification of news articles as fake or real.

## Dataset

- Source: [`GonzaloA/fake_news`](https://huggingface.co/datasets/GonzaloA/fake_news) on Hugging Face
- Labels: `0 = FAKE`, `1 = REAL`
- Subsets: the first 1,000 examples of the `train` split and the first 500 of the `validation` split
- Input: the `text` column; `title` is dropped
- Tokenization: `distilbert-base-uncased` tokenizer, truncation at 256 tokens, dynamic padding via `DataCollatorWithPadding`

## Model

- `distilbert-base-uncased`, an encoder-only model with a sequence classification head (2 labels)
- Loaded in `model/model.py` through `AutoModelForSequenceClassification`
- Each configuration starts from a fresh pretrained model

## Training pipeline

`train.py` runs the following steps:

1. Fix the random seed (42) for Python, NumPy and PyTorch
2. Load and tokenize the dataset once (`data/dataset.py`)
3. For each of the 4 hyperparameter configurations, build `TrainingArguments` and a `Trainer`, train, then evaluate on the validation subset
4. Print a summary of the accuracy of each configuration

Evaluation and checkpointing run at the end of every epoch. Only one checkpoint is kept per configuration, and at the end of training the `Trainer` reloads the checkpoint with the lowest validation loss. The metric is accuracy (`training/metrics.py`).

## Hyperparameter configurations

| Config      | Learning rate | Batch | Epochs | Weight decay |
| ----------- | ------------- | ----- | ------ | ------------ |
| baseline    | 2e-5          | 16    | 3      | 0.01         |
| high_lr     | 5e-5          | 16    | 3      | 0.01         |
| more_epochs | 2e-5          | 16    | 5      | 0.01         |
| small_batch | 2e-5          | 8     | 3      | 0.01         |

## Results

Validation accuracy of the reloaded checkpoint:

| Config      | Accuracy   |
| ----------- | ---------- |
| baseline    | 0.9780     |
| high_lr     | 0.9740     |
| more_epochs | 0.9780     |
| small_batch | **0.9800** |

The best configuration is `small_batch` (batch size 8, learning rate 2e-5).

## Checkpoints

Saved under `fine_tuning/results/<config>/` (not versioned, about 3 GB in total). In all four configurations the checkpoint selected by validation loss is the one from epoch 2.

## Notes

- Evaluation uses the validation split, which also drives checkpoint selection. No held-out test set was used.
- Since `more_epochs` also selects its epoch 2 checkpoint, its result does not reflect the model after 5 epochs.
- The differences between configurations are small (at most 3 predictions out of 500).
