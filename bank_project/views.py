import os
import pickle
import pandas as pd
import numpy as np
import sys
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.base import BaseEstimator, ClassifierMixin

class DummyTorchObject:
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        pass
    def __reduce__(self):
        return (DummyTorchObject, ())
    def __getattr__(self, name):
        return DummyTorchObject()
    def __call__(self, *args, **kwargs):
        return DummyTorchObject()

# Define classes to match the pickled model's references
class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dim=64, epochs=15, batch_size=64, lr=0.005):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y):
        return self
        
    def predict_proba(self, X):
        try:
            import torch
            if not hasattr(torch, 'FloatTensor'):
                raise ImportError("torch is not fully loaded")
            self.model.eval()
            X_tensor = torch.FloatTensor(X)
            with torch.no_grad():
                prob1 = self.model(X_tensor).numpy().flatten()
            prob0 = 1.0 - prob1
            return np.column_stack([prob0, prob1])
        except Exception:
            prob1 = np.ones(X.shape[0]) * 0.5
            prob0 = 1.0 - prob1
            return np.column_stack([prob0, prob1])
            
    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.5).astype(int)

class CustomVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights if weights else [1.0] * len(estimators)
        
    def fit(self, X, y):
        return self
        
    def predict_proba(self, X):
        probs = []
        total_weight = 0
        for (name, clf), weight in zip(self.estimators.items(), self.weights):
            if name in ['pytorch', 'PyTorchMLP', 'pytorch_mlp']:
                try:
                    p = clf.predict_proba(X)
                    if np.all(p[:, 1] == 0.5):
                        continue
                    probs.append(p * weight)
                    total_weight += weight
                except Exception:
                    continue
            else:
                probs.append(clf.predict_proba(X) * weight)
                total_weight += weight
        if total_weight == 0:
            return np.zeros((X.shape[0], 2))
        return sum(probs) / total_weight
        
    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.5).astype(int)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__':
            if name == 'PyTorchMLPClassifier':
                return PyTorchMLPClassifier
            elif name == 'CustomVotingClassifier':
                return CustomVotingClassifier
        if 'torch' in module or 'torch' in name:
            return DummyTorchObject
        try:
            return super().find_class(module, name)
        except Exception:
            return DummyTorchObject

# Load model assets
ASSETS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'bank_model_assets.pkl')
model_assets = None

def load_assets():
    global model_assets
    if model_assets is None and os.path.exists(ASSETS_PATH):
        with open(ASSETS_PATH, 'rb') as f:
            model_assets = CustomUnpickler(f).load()
    return model_assets

def index(request):
    assets = load_assets()
    options = {}
    metrics = {}
    if assets:
        options = assets['features']['categorical_options']
        metrics = assets.get('metrics', {})
    return render(request, 'index.html', {'options': options, 'metrics': metrics})

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        assets = load_assets()
        if not assets:
            return JsonResponse({'error': 'Modèle non disponible. Veuillez exécuter le notebook dentraînement.'}, status=500)
        
        preprocessor = assets['preprocessor']
        ensemble = assets['ensemble']
        features_meta = assets['features']
        
        # Get POST fields
        input_data = {}
        for col in features_meta['num_cols']:
            val = request.POST.get(col, 0)
            input_data[col] = [float(val)]
            
        for col in features_meta['cat_cols']:
            val = request.POST.get(col, '')
            input_data[col] = [val]
            
        df_input = pd.DataFrame(input_data)
        
        # Process and predict
        try:
            X_proc = preprocessor.transform(df_input)
            prob = ensemble.predict_proba(X_proc)[0, 1]
            prediction = int(prob >= 0.5)
            
            # Predict with individual models for comparison
            prob_lr = assets['logistic_regression'].predict_proba(X_proc)[0, 1]
            prob_rf = assets['random_forest'].predict_proba(X_proc)[0, 1]
            prob_mlp = assets['pytorch_mlp'].predict_proba(X_proc)[0, 1]
            
            return JsonResponse({
                'probability': float(prob),
                'prediction': 'Oui' if prediction == 1 else 'Non',
                'breakdown': {
                    'lr': float(prob_lr),
                    'rf': float(prob_rf),
                    'mlp': float(prob_mlp)
                }
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
            
    return JsonResponse({'error': 'Méthode non autorisée'}, status=405)

@csrf_exempt
def batch_predict(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        assets = load_assets()
        if not assets:
            return JsonResponse({'error': 'Modèle non disponible.'}, status=500)
            
        preprocessor = assets['preprocessor']
        ensemble = assets['ensemble']
        
        csv_file = request.FILES['csv_file']
        try:
            df = pd.read_csv(csv_file, sep=';')
            if 'y' in df.columns:
                df = df.drop(columns=['y'])
                
            X_proc = preprocessor.transform(df)
            probs = ensemble.predict_proba(X_proc)[:, 1]
            preds = (probs >= 0.5).astype(int)
            
            results = []
            for i, row in df.head(50).iterrows():
                row_dict = row.to_dict()
                row_dict['score'] = float(probs[i])
                row_dict['prediction'] = 'Oui' if preds[i] == 1 else 'Non'
                results.append(row_dict)
                
            return JsonResponse({
                'results': results,
                'total': len(df),
                'yes_count': int(np.sum(preds)),
                'no_count': int(len(df) - np.sum(preds))
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
            
    return JsonResponse({'error': 'Requête invalide'}, status=400)

from django.core.mail import send_mail
from django.conf import settings

@csrf_exempt
def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name', '')
        email = request.POST.get('email', '')
        subject = request.POST.get('subject', '')
        message = request.POST.get('message', '')
        
        if not name or not email or not message:
            return JsonResponse({'error': 'Veuillez remplir tous les champs obligatoires.'}, status=400)
            
        full_message = f"Message de {name} ({email}) :\n\n{message}"
        try:
            send_mail(
                subject=f"[Contact Plateforme] {subject or 'Nouveau Message'}",
                message=full_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=['npe.techs@gmail.com'],
                fail_silently=False,
            )
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Méthode non autorisée'}, status=405)

