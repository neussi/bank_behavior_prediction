import os
import pickle
import pandas as pd
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

# 1. Custom PyTorch Neural Network Classifier Wrapper
if HAS_TORCH:
    class PyTorchMLP(nn.Module):
        def __init__(self, input_dim):
            super(PyTorchMLP, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            return self.net(x)

    class SklearnPyTorchClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, epochs=20, batch_size=64, lr=0.005, input_dim=None):
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr
            self.input_dim = input_dim
            self.model = None
            self.classes_ = np.array([0, 1])
            
        def fit(self, X, y):
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y).view(-1, 1)
            
            self.input_dim = X.shape[1]
            self.model = PyTorchMLP(self.input_dim)
            
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            self.model.train()
            for epoch in range(self.epochs):
                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            return self
            
        def predict_proba(self, X):
            X_tensor = torch.FloatTensor(X)
            self.model.eval()
            with torch.no_grad():
                probs = self.model(X_tensor).numpy()
            return np.hstack([1 - probs, probs])
            
        def predict(self, X):
            probs = self.predict_proba(X)
            return (probs[:, 1] >= 0.5).astype(int)
else:
    class SklearnPyTorchClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, epochs=20, batch_size=64, lr=0.005, input_dim=None):
            self.classes_ = np.array([0, 1])
            
        def fit(self, X, y):
            return self
            
        def predict_proba(self, X):
            probs = np.zeros((X.shape[0], 1))
            return np.hstack([1 - probs, probs])
            
        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

def train_and_evaluate():
    print("Loading Bank Marketing dataset...")
    # Load dataset
    data_path = "dataset/bank.csv"
    if not os.path.exists(data_path):
        data_path = "dataset/bank-full.csv"
        
    df = pd.read_csv(data_path, sep=";")
    
    # Map target column to binary
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    X = df.drop(columns=['y'])
    y = df['y']
    
    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical features: {num_cols}")
    print(f"Categorical features: {cat_cols}")
    
    # Preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit preprocessor on train data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Processed feature matrix dimensions: {X_train_processed.shape}")
    
    # Train classifiers
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_train_processed, y_train)
    
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, max_depth=12)
    rf_model.fit(X_train_processed, y_train)
    
    if HAS_TORCH:
        print("Training Custom PyTorch MLP...")
        pytorch_model = SklearnPyTorchClassifier(epochs=30, batch_size=128, lr=0.005, input_dim=X_train_processed.shape[1])
        pytorch_model.fit(X_train_processed, y_train)
    else:
        print("PyTorch not installed. Using secondary Logistic Regression as fallback for ensemble...")
        pytorch_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=99)
        pytorch_model.fit(X_train_processed, y_train)
    
    # Create Voting Ensemble
    print("Building Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ('lr', lr_model),
            ('rf', rf_model),
            ('pytorch', pytorch_model)
        ],
        voting='soft'
    )
    ensemble.fit(X_train_processed, y_train)
    
    # Evaluate models
    models = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "PyTorch MLP": pytorch_model,
        "Voting Ensemble": ensemble
    }
    
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test_processed)
        probs = model.predict_proba(X_test_processed)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        }
        
        print(f"\n--- {name} Evaluation ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        
    # Save the models, preprocessor, features metadata
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    
    artifacts = {
        "preprocessor": preprocessor,
        "logistic_regression": lr_model,
        "random_forest": rf_model,
        "pytorch_mlp": pytorch_model,
        "ensemble": ensemble,
        "features": {
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "categorical_options": {col: list(X[col].unique()) for col in cat_cols}
        },
        "metrics": results
    }
    
    with open(os.path.join(output_dir, "bank_model_assets.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
        
    print("\nSaved all model assets to models/bank_model_assets.pkl")

if __name__ == "__main__":
    train_and_evaluate()
