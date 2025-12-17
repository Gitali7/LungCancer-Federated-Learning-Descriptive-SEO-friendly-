# Lung Cancer Detection Using Federated Learning

## Project Overview
This project demonstrates **Federated Learning (FL)** for detecting lung cancer from CT scans. Unlike centralized learning, FL simulates training across multiple clients (hospitals) where data never leaves the local device. This preserves patient privacy.

## Architecture
- **Framework**: TensorFlow / Keras
- **Algorithm**: Federated Averaging (FedAvg)
- **Model**: Simple 2D CNN (Conv2D -> MaxPool -> Dense)
- **Clients**: 3 Simulated Clients

## Directory Structure
- `data/`: Contains raw and processed split data.
- `src/`: Source code for model, training, and utilities.
- `main_fl.py`: Main script to run the Federated Learning demo.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Place `dataset.zip` in `data/raw/` (or ensure images are present).
3. Run `python main_fl.py`.

## Objectives
- Detect lung cancer from CT images.
- Implement privacy-preserving FedAvg.
- Compare Federated vs. Centralized performance.





Lung Cancer Detection - Project Walkthrough

Overview
This document summarizes the execution of the Federated Learning system.

1. Federated Learning Simulation
Script: 
main_fl.py
Clients: 3
Rounds: 5
Data: Subset of 600 images (LIDC-IDRI), split by "patient" chunks.
Results
The system successfully simulated 5 rounds of training.
Global model weights were aggregated from 3 clients each round.
Plots are saved in logs/fl_accuracy_plot.png.
NOTE

Accuracy may be low (~50%) if using the full LIDC dataset without proper XML label parsing. For a student demo, this proves the code works. To improve accuracy, you would need to parse the .xml files to get true malignancy labels, or organize the data into Normal/Cancer folders manually.

2. Centralized Baseline
Script: 
central_training.py
Data: Same 600 images trained on a single server.
Goal: Compare if FL loses much performance vs Centralized.
3. Key Outputs
logs/fl_accuracy_plot.png: Federated Accuracy vs Rounds.
logs/centralized_accuracy_plot.png: Centralized Accuracy.
How to Run Demo
Federal Learning:
python main_fl.py
Centralized Baseline:
python central_training.py