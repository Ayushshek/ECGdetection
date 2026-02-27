ECG Anomaly Detection using CNN & LSTM Autoencoders
Project Overview

This project implements and compares CNN and LSTM autoencoders for anomaly detection on ECG (electrocardiogram) heartbeat signals.
The goal is to detect abnormal heartbeats using a reconstruction-based anomaly detection approach:

The models are trained only on normal heartbeats.
They learn a compressed (latent) representation of normal ECG patterns.
During testing, both normal and abnormal signals are passed through the autoencoder.
If a signal has a high reconstruction error (i.e., the model fails to accurately reconstruct it), it is classified as abnormal using a predefined threshold.

This approach allows anomaly detection without directly training on abnormal examples.
The project also compares:
CNN-based sequence modeling (local pattern detection)
LSTM-based sequence modeling (temporal dependency learning)

Performance is evaluated using reconstruction error statistics and classification metrics.

Data Source

Dataset used:
Kaggle – Heartbeat Dataset
https://www.kaggle.com/datasets/shayanfazeli/heartbeat

Specifically, this project uses the PTB Diagnostic ECG Database (PTBDB) subset, which contains:

ptbdb_normal.csv → Normal heartbeats
ptbdb_abnormal.csv → Abnormal heartbeats

Data Description
Each row represents a single ECG heartbeat.
Each heartbeat consists of 187 time-step values (1D time-series signal).
The dataset is preprocessed and normalized.
For anomaly detection:
    Normal heartbeats are split into:
          70% training
          15% validation
          15% test

    Abnormal heartbeats are used only for evaluation.

Project Structure & File Descriptions
/Cnn_autoencoder.py
Contains the full implementation of the CNN Autoencoder, including:
Data loading and preprocessing
Train/validation/test split
CNN encoder and decoder architecture
Reconstruction error computation
Threshold-based anomaly classification
Evaluation metrics

/cnnmodelresults.txt
Records experimental results for different CNN configurations, including:
Kernel size variations
Latent dimension changes
Reconstruction error statistics
Normal vs abnormal misclassification rates
Overall detection performance

/conclusion.txt
Summarizes findings from the experiments, including:
Performance comparison between CNN and LSTM autoencoders
Impact of latent dimension and architecture choices
Observations on anomaly separation
Final conclusions about which architecture performed better and why

/lstm_autoencoder_ptbdb_latent8_big.py
Contains the implementation of the LSTM Autoencoder, including:
LSTM-based encoder and decoder
Sequence modeling using recurrent layers
Reconstruction-based anomaly detection
Evaluation on normal test data and abnormal data
This file reflects the larger-capacity LSTM configuration tested during experimentation.

/lstmresults
Stores evaluation results from the LSTM autoencoder experiments, including:
Threshold value used
Mean reconstruction errors (normal vs abnormal)
Misclassification rates
Overall anomaly detection performance
