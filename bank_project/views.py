import os
import pickle
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Load model assets
ASSETS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'bank_model_assets.pkl')
model_assets = None

def load_assets():
    global model_assets
    if model_assets is None and os.path.exists(ASSETS_PATH):
        with open(ASSETS_PATH, 'rb') as f:
            model_assets = pickle.load(f)
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
