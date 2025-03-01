# Overview

This repository provides a Python script for processing EEG data, extracting features, and classifying data into two categories: Caffeine and Placebo. The code processes MATLAB .mat files and applies machine learning techniques to distinguish between experimental conditions.

# Features

- Data Loading and Preprocessing: Handles .mat files and reshapes EEG data for further analysis.
- Feature Extraction: Computes power spectral density (PSD) to derive key EEG signal features.
- Filtering: Applies band-pass filtering (1–40 Hz) to the EEG signals.
- Classification: Implements a Random Forest Classifier to differentiate between the two categories.
- Evaluation: Provides metrics like accuracy, precision, recall, F1-score, and a confusion matrix visualization.

# Required Libraries:

- numpy
- scipy
- mne
- scikit-learn
- matplotlib
- seaborn
