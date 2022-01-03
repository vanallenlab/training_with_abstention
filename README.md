# Training with abstention
Code supporting manuscript doi: https://doi.org/10.1101/2021.09.14.460365. Here, in this repo, are helper scripts that inform how the code is organized and provide details on how the training data was pre-processed (image augmentation, label balancing, cross-validation etc.)

The methods used to train the model are maintained in this repo: https://github.com/vanallenlab/mc_lightning_public


This repo has four python files. 

1. pred_grade_kirc_dro.py
2. pred_gs_test_num_hosps.py
3. pred_healthy_dro_wrst_hsp.py
4. pred_healthy_source_hetero.py

The first three, are used to measure the performance of models trained with abstention and compare these performances with ERM models- 1. compares the models' performances in predciting the grade (2 vs. 4) of clear cell Renal Cell Carcinoma, 2. compares the models' performances of predicting the Gleason score (6 vs. 8 or greater) in prostate adenocarcinoma and 3. compares the models' performances in identifying whether a tile comes from a slide containing tumor vs. one that doesn't in lung adenocarcinoma. The data for these tasks is taken from TCGA. For more details, please refer to the manuscript.

The program in 4. performs the same task as program 3- identifying whether a tile comes from a slide containing tumor vs. one that doesn't in lung adenocarcinoma, but where the training data and validation are each taken from one hospital. The choice of hospital used in training and validation is ablated. 

