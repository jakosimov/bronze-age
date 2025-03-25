#!/bin/zsh

# Run the experiments
python train_bronze.py
cp results.csv results_stone_age_model.csv
cp results2.csv results_stone_age_model2.csv
python train_bronze2.py
cp results.csv results_bronze_age_model.csv
cp results2.csv results_bronze_age_model2.csv