KANvsMLP: Knowledge-Augmented Networks vs. Multi-Layer Perceptron on MNIST
==========================================================================

Project Overview
----------------
This project compares the performance of a Knowledge-Augmented Network (KAN) against a conventional Multi-Layer Perceptron (MLP) on the MNIST dataset.  
Both models are implemented in PyTorch, trained and validated using 5-fold cross-validation, and benchmarked across a grid of hyperparameters.

The KAN model introduces spline-based feature transformations that enhance the representational capability beyond standard linear layers.  
The MLP serves as the baseline for comparison under identical experimental conditions.

Core Features
--------------
- End-to-end training pipeline for both KAN and MLP models.
- Configurable architecture: variable hidden layer sizes, learning rates, and batch sizes.
- 5-fold cross-validation to measure model generalization.
- Automatic CSV logging for:
  - Hidden layer configuration
  - Batch size
  - Learning rate
  - Total trainable parameters
  - Average validation accuracy
- GPU support check (via `test_gpu.py`).

Files Included
--------------
1. **kanvsmlp_mnist.py**  
   - Main training script for both KAN and MLP models.  
   - Implements the following classes:
     - `KANLayer`: Defines a spline-based layer with layer normalization.
     - `KAN`: Multi-layer KAN network.
     - `MLP`: Standard multi-layer perceptron.
   - Conducts grid search over learning rates, batch sizes, and hidden layer configurations.
   - Outputs `kan_results.csv` and `mlp_results.csv` with performance metrics.

2. **test_gpu.py**  
   - Simple utility to confirm GPU availability using PyTorch:
     ```
     import torch
     print(f"Number of GPUs: {torch.cuda.device_count()}")
     ```

3. **.gitignore**  
   - Standard Git ignore file for Python projects.

Dependencies
------------
- PyTorch
- torchvision
- scikit-learn
- numpy
- csv

