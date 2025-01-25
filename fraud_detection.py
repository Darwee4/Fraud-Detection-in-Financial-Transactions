import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    def __init__(self, data_path):
        """Initialize fraud detection system with data path"""
        self.data = self.load_data(data_path)
        self.preprocess_data()
        self.handle_imbalance()
        self.train_models()
        
    def load_data(self, path):
        """Load and inspect credit card transaction data"""
        df = pd.read_csv(path)
        print(f"Data loaded with {len(df)} transactions")
        print(f"Fraud percentage: {df['Class'].mean()*100:.2f}%")
        return df
        
    def preprocess_data(self):
        """Preprocess data: handle missing values, normalize features"""
        # Drop time column and separate features/target
        self.X = self.data.drop(['Time', 'Class'], axis=1)
        self.y = self.data['Class']
        
        # Normalize features
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)
            
    def handle_imbalance(self):
        """Handle class imbalance using SMOTE"""
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        print(f"After SMOTE - Fraud percentage: {self.y_train.mean()*100:.2f}%")
        
    def train_models(self):
        """Train both Isolation Forest and XGBoost models"""
        # Isolation Forest for anomaly detection
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.01,
            random_state=42
        )
        self.iso_forest.fit(self.X_train)
        
        # XGBoost for classification
        self.xgb = XGBClassifier(
            scale_pos_weight=len(self.y_train[self.y_train==0])/len(self.y_train[self.y_train==1]),
            random_state=42
        )
        self.xgb.fit(self.X_train, self.y_train)
        
    def evaluate_models(self):
        """Evaluate models using precision, recall and F1 score"""
        # Isolation Forest evaluation
        iso_preds = self.iso_forest.predict(self.X_test)
        iso_preds = np.where(iso_preds == -1, 1, 0)  # Convert to binary labels
        iso_metrics = precision_recall_fscore_support(self.y_test, iso_preds, average='binary')
        
        # XGBoost evaluation
        xgb_preds = self.xgb.predict(self.X_test)
        xgb_metrics = precision_recall_fscore_support(self.y_test, xgb_preds, average='binary')
        
        print("\nModel Evaluation:")
        print(f"Isolation Forest - Precision: {iso_metrics[0]:.4f}, Recall: {iso_metrics[1]:.4f}, F1: {iso_metrics[2]:.4f}")
        print(f"XGBoost - Precision: {xgb_metrics[0]:.4f}, Recall: {xgb_metrics[1]:.4f}, F1: {xgb_metrics[2]:.4f}")
        
    def detect_fraud(self, transaction):
        """Detect if a transaction is fraudulent using both models"""
        # Preprocess transaction
        transaction = self.scaler.transform([transaction])
        
        # Get predictions
        iso_pred = self.iso_forest.predict(transaction)
        xgb_pred = self.xgb.predict(transaction)
        
        # Return consensus prediction
        return int(iso_pred == -1 or xgb_pred == 1)

if __name__ == "__main__":
    # Example usage
    system = FraudDetectionSystem('creditcard.csv')
    system.evaluate_models()
    
    # Example transaction detection
    sample_transaction = [0, -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 0.239599, 0.098698, 0.363787, 0.090794, -0.551600, -0.617801, -0.991390, -0.311169, 1.468177, -0.470401, 0.207971, 0.025791, 0.403993, 0.251412, -0.018307, 0.277838, -0.110474, 0.066928, 0.128539, -0.189115, 0.133558, -0.021053, 149.62]
    print(f"\nFraud detection for sample transaction: {system.detect_fraud(sample_transaction[1:])}")
