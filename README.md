# Fraud Detection in Financial Transactions

This repository contains an implementation of a fraud detection system for credit card transactions using machine learning. The system employs both anomaly detection (Isolation Forest) and classification (XGBoost) approaches to identify fraudulent transactions with high precision and recall.

## Features

- Data preprocessing and feature normalization
- Class imbalance handling using SMOTE
- Dual-model approach combining Isolation Forest and XGBoost
- Comprehensive model evaluation using precision, recall, and F1 score
- Real-time fraud detection function

## Requirements

- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - imbalanced-learn
  - xgboost

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Darwee4/fraud-detection.git
   cd fraud-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your credit card transaction data (CSV format) in the project directory
2. Run the fraud detection system:
   ```python
   python fraud_detection.py
   ```
3. To detect fraud on a new transaction:
   ```python
   from fraud_detection import FraudDetectionSystem
   
   system = FraudDetectionSystem('creditcard.csv')
   transaction = [...]  # Your transaction features
   is_fraud = system.detect_fraud(transaction)
   print(f"Fraudulent: {is_fraud}")
   ```

## Model Performance

The system provides the following metrics on test data:
- Isolation Forest: Precision, Recall, F1 Score
- XGBoost: Precision, Recall, F1 Score

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

[MIT License](LICENSE)
