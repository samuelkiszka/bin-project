## Ideas
- generate new random training pairs every epoch in contrastive learning
- validation/test set should be fixed to evaluate the model performance on the same data
- 5fold cross validation to evaluate the model performance on different data splits


## Plan
- [ ] implement pipeline to run training on full data on metacentrum or other cluster (merlin?)
- [ ] utilize tensorboard for logging and visualization
- [ ] add proper logging
- [ ] add proper checkpointing
- [ ] add proper evaluation
- [ ] add proper visualization
- [ ] add proper documentation



# Measurement
- accuracy
- ROC
- AUC
- precision
- DET curves


# Experiments
- [ ] different model sizes (different MCUs and learnable parameter counts)
- [ ] different embedding sizes (32, 64, 128, 256)
- [ ] different number of training pairs per epoch (1000, 5000, 10000)
