# Fashion-MNIST CNN Classification & Error Analysis

A Computer Vision project using PyTorch to train, evaluate, and analyze a Convolutional Neural Network on the Fashion-MNIST dataset.

The project focuses not only on overall accuracy, but also on per-class performance, confusion matrix analysis, failure cases, and controlled data augmentation experiments.

## Project Goals

- Build a CNN for multiclass image classification
- Create a clear Train / Validation / Test workflow
- Evaluate performance using accuracy, precision, recall, and F1-score
- Analyze misclassified samples and confusion patterns
- Compare baseline training with different data augmentation strategies

## Dataset

The project uses the Fashion-MNIST dataset, which contains grayscale images of 10 clothing categories.

The training data was split into:
- 50,000 training samples
- 10,000 validation samples

The official test set contains:
- 10,000 unseen test samples

## Model

The baseline model is a Convolutional Neural Network built with PyTorch.

It includes:
- Two convolutional layers
- ReLU activations
- Max pooling
- Dropout
- Fully connected classification layers

## Experiments

### Baseline Model

The baseline CNN achieved:

**Test Accuracy: 90.51%**

Per-class analysis showed strong performance on categories such as:
- Trouser
- Bag
- Sandal
- Ankle boot

The most challenging class was:

**Shirt — F1-score: 0.7331**

### Strong Data Augmentation

Applied:
- Random rotation up to ±10°
- Translation up to 8%

Result:

**Test Accuracy: 88.81%**

Performance decreased across all classes, with the largest drop observed in the Shirt category.

### Mild Data Augmentation

Applied:
- Random rotation up to ±5°
- Translation up to 3%

Result:

**Test Accuracy: 90.04%**

Mild augmentation recovered most of the performance lost in the stronger augmentation experiment, but did not outperform the baseline model.

Pullover showed a small F1 improvement, while Shirt remained the most challenging category.

## Key Findings

- The baseline model achieved the best overall performance.
- Strong augmentation was too aggressive for this training setup.
- Mild augmentation performed better than strong augmentation, but still slightly below the baseline.
- Per-class metrics revealed weaknesses that were not visible from overall accuracy alone.
- Error analysis showed that visually similar clothing categories are harder for the model to distinguish.

## Evaluation

The project includes:

- Classification report
- Precision, Recall, and F1-score
- Confusion matrix
- Misclassified image analysis
- Baseline vs augmentation comparison

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Notebook

Main notebook:

`fashion_mnist_cnn_error_analysis.ipynb`

## Future Improvements

Possible next steps:

- Train for more epochs
- Compare deeper CNN architectures
- Evaluate transfer learning approaches
- Run experiments across multiple random seeds
- Explore additional augmentation strategies
- Test the model on a more complex real-world Computer Vision dataset
