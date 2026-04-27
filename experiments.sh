# Run this script to execute all experiments.

###
# Experiment 1: VGGEmbedNet Architecture Comparison
#
# Compare 5 different VGGEmbedNet architectures with different MCUs and trainable parameter counts.
###

#"VGGEmbedNet1": {
#    "macs": 302645248,
#    "params": 699424
#},
echo ""
echo ""
echo "Running Experiment 1: VGGEmbedNet Architecture Comparison"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet1" \
    --note "ex1" \
    --runs_per_model 3 \

#"VGGEmbedNet3": {
#    "macs": 47255680,
#    "params": 20896
#},
echo ""
echo ""
echo "Running Experiment 1: VGGEmbedNet Architecture Comparison"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet3" \
    --note "ex1" \
    --runs_per_model 3 \

#"VGGEmbedNet6": {
#    "macs": 113216,
#    "params": 5856
#},
echo ""
echo ""
echo "Running Experiment 1: VGGEmbedNet Architecture Comparison"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet6" \
    --note "ex1" \
    --runs_per_model 3 \

#"VGGEmbedNet8": {
#    "macs": 7568,
#    "params": 624
#},
echo ""
echo ""
echo "Running Experiment 1: VGGEmbedNet Architecture Comparison"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet8" \
    --note "ex1" \
    --runs_per_model 3 \

#"VGGEmbedNet13": {
#    "macs": 168,
#    "params": 28
#}
echo ""
echo ""
echo "Running Experiment 1: VGGEmbedNet Architecture Comparison"
python3 -m src.train \
    --emb_dim 4 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet13" \
    --note "ex1" \
    --runs_per_model 3 \

###
# Experiment 2: Embedding Dimension Comparison
#
# Compare the performance of VGGEmbedNet1, VGGEmbedNet6 and VGGEmbedNet13  with different embedding dimensions (4, 16, 32, 64).
###

##
#  VGGEmbedNet1
##
echo ""
echo ""
echo "Running Experiment 2: Embedding Dimension Comparison - VGGEmbedNet1"
python3 -m src.train \
    --emb_dim 4 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet1" \
    --note "ex2" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 2: Embedding Dimension Comparison - VGGEmbedNet1"
python3 -m src.train \
    --emb_dim 16 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet1" \
    --note "ex2" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 2: Embedding Dimension Comparison - VGGEmbedNet1"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet1" \
    --note "ex2" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 2: Embedding Dimension Comparison - VGGEmbedNet1"
python3 -m src.train \
    --emb_dim 64 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet1" \
    --note "ex2" \
    --runs_per_model 3 \

##
#  VGGEmbedNet6
##
echo ""
echo ""
echo "Running Experiment 2: Embedding Dimension Comparison - VGGEmbedNet6"
python3 -m src.train \
    --emb_dim 4 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet6" \
    --note "ex2" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 2: Embedding Dimension Comparison - VGGEmbedNet6"
python3 -m src.train \
    --emb_dim 16 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet6" \
    --note "ex2" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 2: Embedding Dimension Comparison - VGGEmbedNet6"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet6" \
    --note "ex2" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 2: Embedding Dimension Comparison - VGGEmbedNet6"
python3 -m src.train \
    --emb_dim 64 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet6" \
    --note "ex2" \
    --runs_per_model 3 \

##
#  VGGEmbedNet13
##
echo ""
echo ""
echo "Running Experiment 2: Embedding Dimension Comparison - VGGEmbedNet13"
python3 -m src.train \
    --emb_dim 4 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet13" \
    --note "ex2" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 2: Embedding Dimension Comparison - VGGEmbedNet13"
python3 -m src.train \
    --emb_dim 16 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet13" \
    --note "ex2" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 2: Embedding Dimension Comparison - VGGEmbedNet13"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet13" \
    --note "ex2" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 2: Embedding Dimension Comparison - VGGEmbedNet13"
python3 -m src.train \
    --emb_dim 64 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet13" \
    --note "ex2" \
    --runs_per_model 3 \

###
# Experiment 3: Training samples per epoch comparison
#
# Compare the performance of VGGEmbedNet1, VGGEmbedNet6 and VGGEmbedNet13  with different pair count per epoch (1000, 500, 100, 32).
###

##
#  VGGEmbedNet1
##
echo ""
echo ""
echo "Running Experiment 3: Training samples per epoch comparison - VGGEmbedNet1"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet1" \
    --note "ex3" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 3: Training samples per epoch comparison - VGGEmbedNet1"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 500 \
    --model "VGGEmbedNet1" \
    --note "ex3" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 3: Training samples per epoch comparison - VGGEmbedNet1"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 100 \
    --model "VGGEmbedNet1" \
    --note "ex3" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 3: Training samples per epoch comparison - VGGEmbedNet1"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 32 \
    --model "VGGEmbedNet1" \
    --note "ex3" \
    --runs_per_model 3 \

##
#  VGGEmbedNet6
##
echo ""
echo ""
echo "Running Experiment 3: Training samples per epoch comparison - VGGEmbedNet6"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet6" \
    --note "ex3" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 3: Training samples per epoch comparison - VGGEmbedNet6"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 500 \
    --model "VGGEmbedNet6" \
    --note "ex3" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 3: Training samples per epoch comparison - VGGEmbedNet6"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 100 \
    --model "VGGEmbedNet6" \
    --note "ex3" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 3: Training samples per epoch comparison - VGGEmbedNet6"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 32 \
    --model "VGGEmbedNet6" \
    --note "ex3" \
    --runs_per_model 3 \

##
#  VGGEmbedNet13
##
echo ""
echo ""
echo "Running Experiment 3: Training samples per epoch comparison - VGGEmbedNet13"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 1000 \
    --model "VGGEmbedNet13" \
    --note "ex3" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 3: Training samples per epoch comparison - VGGEmbedNet13"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 500 \
    --model "VGGEmbedNet13" \
    --note "ex3" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 3: Training samples per epoch comparison - VGGEmbedNet13"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 100 \
    --model "VGGEmbedNet13" \
    --note "ex3" \
    --runs_per_model 3 \

echo ""
echo ""
echo "Running Experiment 3: Training samples per epoch comparison - VGGEmbedNet13"
python3 -m src.train \
    --emb_dim 32 \
    --batch_size 32 \
    --lr 0.0003 \
    --num_epochs 30 \
    --num_workers 2 \
    --pairs_per_epoch 32 \
    --model "VGGEmbedNet13" \
    --note "ex3" \
    --runs_per_model 3 \
