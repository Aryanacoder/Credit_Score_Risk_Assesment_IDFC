# Credit Card Default Prediction

## Overview

This repository contains the implementation of a machine learning model to predict the likelihood of credit card default for customers. The project aims to develop a "Behaviour Score" for credit card holders, helping financial institutions proactively manage portfolio risk and minimize losses due to defaults.

Credit card default is a critical issue for banks and financial institutions, leading to significant financial losses and reputational damage. By leveraging machine learning techniques, this project addresses the limitations of traditional statistical models and provides accurate predictions based on historical data.

---

## Problem Statement

Bank A issues credit cards to eligible customers and uses advanced machine learning models for eligibility, limit assignments, and interest rate optimization. To further enhance its risk management framework, the bank seeks to create a "Behaviour Score" that predicts the probability of default for existing credit card customers.

---

## Goal

The goal of this project is to build a predictive model that estimates the likelihood of credit card default using historical data. The model will focus on customers whose accounts are open and not past due.

---

## Machine Learning Pipeline

### 1. **Data Preprocessing**
- **Handling Missing Values**:
  - Removed columns with more than 60% missing values (reduced features from 1216 to 1199).
  - Filled remaining missing values with median values.
- **Class Imbalance**:
  - Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
- **Feature Scaling**:
  - Standardized features using `StandardScaler`.
- **Dimensionality Reduction**:
  - Reduced features from 1199 to 50 components using PCA (Principal Component Analysis).

### 2. **Model Development**
Several machine learning models were tested, including:
- Logistic Regression (Final Model)
- Random Forest
- XGBoost
- Gradient Boosting
- Support Vector Classifier (SVC)
- Naive Bayes

**Logistic Regression** was chosen as the final model due to its interpretability, computational efficiency, and strong performance metrics.

### 3. **Model Evaluation**
- **Metrics Used**:
  - AUC-ROC: Measures discriminative ability.
  - Precision, Recall, F1-score: Detailed classification metrics.
- **Results**:
  - Logistic Regression achieved an AUC-ROC score of **0.8126**, indicating good predictive performance.

### 4. **Prediction**
The model generates predictions on validation data and outputs a CSV file containing `account_number` and `Predicted Probability`.

---

## Results

### Key Metrics
| Metric                | Value   |
|-----------------------|---------|
| AUC-ROC              | 0.8126  |
| Overall Accuracy      | 93%     |
| Precision (Class 1)   | 0.06    |
| Recall (Class 1)      | 0.27    |
| F1-score (Class 1)    | 0.10    |

### ROC Curve
![ROC Curve](https://cdn.mathpix.com/cropped/2025_04_03_795285e221b244727241g-12.jpg?height=886&width=1080&top_left_y=731&top_lefte shows significant discriminative ability, with an AUC well above random guessing (AUC = 0.5).

---

## Repository Structure

```plaintext
credit-card-default-prediction/
├── data/
│   ├── raw/                # Original dataset files
│   └── processed/          # Preprocessed datasets
├── docs/
│   ├── data_dictionary.md  # Feature descriptions
│   └── model_card.md       # Detailed model documentation
├── notebooks/
│   ├── eda.ipynb           # Exploratory Data Analysis
│   ├── preprocessing.ipynb # Data preprocessing steps
│   ├── training.ipynb      # Model training workflows
│   └── evaluation.ipynb    # Model evaluation and visualization
├── src/
│   ├── data/
│   │   ├── preprocess.py   # Data cleaning functions
│   │   └── smote.py        # SMOTE implementation for class balancing
│   ├── models/
│   │   ├── train.py        # Model training script
│   │   └── evaluate.py     # Model evaluation script
├── tests/                  # Unit tests for code validation
├── LICENSE                 # Project license (MIT)
├── README.md               # Project overview and instructions
├── requirements.txt        # Python dependencies list
└── predicted_probabilities.csv # Final prediction results
```

---

## Installation

### Prerequisites
Ensure you have Python installed on your system (version >=3.8). Install the required libraries using:

```bash
pip install -r requirements.txt
```

### Clone Repository

```bash
git clone https://github.com/Aryanacoder/credit-card-default-prediction.git
cd credit-card-default-prediction
```

---

## Usage

### Data Preprocessing

Run the preprocessing script to clean and prepare the data:

```bash
python src/data/preprocess.py --input data/raw/dev_data.csv --output data/processed/train.csv
```

### Model Training

Train the logistic regression model:

```bash
python src/models/train.py --train-data data/processed/train.csv --output models/logistic_regression.pkl
```

### Model Evaluation

Evaluate the trained model on test data:

```bash
python src/models/evaluate.py --model models/logistic_regression.pkl --test-data data/processed/test.csv
```

### Generate Predictions

Generate predictions on validation data:

```bash
python src/models/predict.py --model models/logistic_regression.pkl --validation-data data/processed/validation.csv --output predicted_probabilities.csv
```

---

## Future Work

1. **Feature Importance Analysis**:
   Use SHAP values or other techniques to interpret feature contributions.
2. **Advanced Ensemble Models**:
   Explore stacking or boosting methods for improved performance.
3. **Web Application**:
   Build a FastAPI-based web app for real-time predictions.
4. **Bias Detection**:
   Implement fairness metrics to ensure unbiased predictions across demographics.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Special thanks to Convolve Hackathon organizers for providing the dataset and problem statement.

Libraries used:
- pandas, numpy: Data manipulation and analysis.
- scikit-learn: Machine learning algorithms.
- imbalanced-learn: SMOTE implementation.
- matplotlib: Visualization tools.

---

This README provides a comprehensive overview of your project, making it easy for potential employers or collaborators to understand your work and its significance! Let me know if you'd like further customization or enhancements!

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/30676447/d93b1c50-6488-4cc0-96e4-97eaf5af7a9b/Documentation.pdf

---
Answer from Perplexity: pplx.ai/share
