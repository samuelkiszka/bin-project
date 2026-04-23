python3 -m src.train \
    --emb_dim 64 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 10 \
    --num_workers 2 \
    --pairs_per_epoch 100 \
    --model "baseline" \
    --runs_per_model 1 \