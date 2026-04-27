# Siamese Embedding Experiments

Train and analyze Siamese embedding networks with contrastive loss on 256x256 grayscale images. The project includes multiple lightweight VGG-style embedding variants, cross-validated training, and analysis scripts for AUC and model efficiency.

## Project layout

- `src/train.py`: main training entry point (single model or all models).
- `src/experiment.py`: cross-validation experiment runner, results aggregation, and model checkpointing.
- `src/trainer.py`: training loop and evaluation (ROC AUC, optimal threshold).
- `src/dataset.py`: Siamese pair sampling dataset and dataloader helper.
- `src/contrastive_loss.py`: contrastive loss implementation.
- `src/models/`: VGG-style embedding network variants and base model.
- `src/embedding_projection.py`: t-SNE projection for embeddings from saved checkpoints.
- `src/visualize_data.py`: class mean images and per-class grid visualization.
- `src/analysis1.py`: model-size vs AUC analysis, CSV export, and Pareto plot.
- `src/analysis2.py`: embedding dimension effect analysis.
- `src/analysis3.py`: pairs-per-epoch effect analysis.
- `src/env.py`: output and data paths.

## Data

- Expected data file: `data/siam_xkiszk00.npz`.
- Arrays inside: `x_train` and `y_train`.
- Images are treated as single-channel and reshaped to `[N, 1, 256, 256]` during training.

## Install

```bash
pip install -r requirements.txt
```

## Train

Train a single model (default uses `VGGEmbedNet1`):

```bash
python -m src.train
```

Train a specific model and embedding dimension:

```bash
python -m src.train --model VGGEmbedNet5 --emb_dim 32
```

Train all models:

```bash
python -m src.train --model all
```

Key flags (see `src/train.py`):

- `--emb_dim`: embedding size.
- `--batch_size`: batch size.
- `--num_epochs`: epochs per fold.
- `--pairs_per_epoch`: number of pairs sampled per epoch.
- `--lr`: learning rate.
- `--runs_per_model`: repeated CV runs per model.
- `--note`: label suffix for result file names.

## Outputs

- Logs: `logs/train_*.log`
- Results JSON: `results/*.json`
- Best checkpoints per model: `models/{model}_emb{dim}.pt`
- Best AUC registry: `models/best_auc.json`
- Model MACs/params registry: `results/model_stats.json`

## Analysis

Generate AUC vs MACs plots and Pareto front:

```bash
python -m src.analysis1 --folder results --prefix <prefix>
```

Analyze embedding dimension effect:

```bash
python -m src.analysis2 --folder results --prefix <prefix>
```

Analyze pairs-per-epoch effect:

```bash
python -m src.analysis3 --folder results --prefix <prefix>
```

## Embedding projection

Create t-SNE plots for selected checkpoints:

```bash
python -m src.embedding_projection \
  --models VGGEmbedNet1 VGGEmbedNet5 \
  --model_paths models/VGGEmbedNet1_emb64.pt models/VGGEmbedNet5_emb64.pt \
  --emb_dims 64 64 \
  --save_path comparison.png
```

## Data visualization

```bash
python -m src.visualize_data
```

This saves class mean images and per-class grids under `data/`.

## Notes

- Cross-validation uses `StratifiedKFold(n_splits=5)`.
- Pair sampling is balanced 1:1 between positive and negative pairs.
- All models use a normalized embedding in `BaseModel.embed()`.

