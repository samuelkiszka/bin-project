##
#  Train models for the report
##

#python3 -m src.train \
#    --emb_dim 32 \
#    --batch_size 32 \
#    --lr 0.0003 \
#    --num_epochs 30 \
#    --num_workers 2 \
#    --pairs_per_epoch 1000 \
#    --model "VGGEmbedNet8" \
##    --note "ex1" \
#    --runs_per_model 1 \


##
#  Generate graphs and tables for the report
##

#python3 -m src.analysis1 \
#    --folder "results" \
#    --prefix "ex1" \
#
#python3 -m src.analysis2 \
#    --folder "results" \
#    --prefix "ex2" \
#
#python3 -m src.analysis3 \
#    --folder "results" \
#    --prefix "ex3" \


##
#  Generate embedding visualizations for the report
##

#python3 -m src.plot \
#    --models "VGGEmbedNet1" "VGGEmbedNet6" "VGGEmbedNet8" "VGGEmbedNet13" \
#    --model_paths "models/VGGEmbedNet1_emb32.pt" "models/VGGEmbedNet6_emb32.pt" "models/VGGEmbedNet8_emb32.pt" "models/VGGEmbedNet13_emb4.pt" \
#    --emb_dims 32 32 32 4\