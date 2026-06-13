# BankScoreAI - Prédiction du Comportement des Clients Bancaires

**BankScoreAI** est une application web intelligente basée sur Django permettant de prédire l'intention d'achat d'un prospect (souscription à un dépôt à terme) lors de campagnes de marketing direct. Ce projet combine des algorithmes de Machine Learning (Régression Logistique, Forêt Aléatoire, et un Perceptron Multicouche PyTorch) sous forme d'un modèle d'ensemble de vote pondéré pour fournir des scores de propension très précis.

---

## Fonctionnalités Clés

1. **Évaluation Individuelle en Temps Réel** : Un formulaire ergonomique complet (données démographiques, statut financier, historique de contact) permettant de calculer instantanément le score de propension d'un client.
2. **Jauge Interactive & Visualisation** : Affichage dynamique du score de propension via une jauge circulaire interactive en SVG.
3. **Décomposition par Modèle (Breakdown)** : Comparaison des probabilités prédites individuellement par chaque modèle (Régression Logistique, Forêt Aléatoire, MLP).
4. **Analyse Batch (Fichier CSV)** : Drag-and-drop ou sélection d'un fichier CSV formaté pour évaluer des milliers de profils clients simultanément avec synthèse statistique (nombre de souscriptions prévues vs refus).
5. **Aérodynamisme Visuel** : Interface haut de gamme sous thème sombre construite en Tailwind CSS avec une police moderne (Outfit).

---

## Stack Technique

- **Framework Web** : Django 6.0+
- **Frontend** : HTML5, Tailwind CSS, Javascript (Vanilla / Fetch API)
- **Machine Learning** : Scikit-Learn, Pandas, NumPy, PyTorch (MLP Classifier)
- **Visualisation** : Chart.js, SVG animés
- **Base de Données** : SQLite (pour le stockage local et le suivi d'audit)

---

## Performances des Modèles

Les métriques d'évaluation sur le jeu de données test sont les suivantes :

| Modèle | Exactitude (Accuracy) | Précision | Rappel (Recall) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Régression Logistique** | 82.32% | 0.3716 | 0.7788 | 0.5031 | 0.8908 |
| **Forêt Aléatoire** | 88.07% | 0.4867 | 0.7019 | 0.5748 | 0.9095 |
| **Voting Ensemble** | 83.31% | 0.3842 | 0.7500 | 0.5081 | 0.9008 |

---

## Installation et Lancement Local

### Prerequisites
- Python 3.10+
- Pip & Virtualenv

### Étapes d'installation

1. **Activer l'environnement virtuel et installer les dépendances** :
   ```bash
   source ../venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Générer le modèle et les métadonnées** :
   Exécutez le script d'entraînement pour générer l'artéfact `bank_model_assets.pkl` contenant le pipeline de prétraitement et les modèles :
   ```bash
   python train_bank.py
   ```

3. **Appliquer les migrations de base de données** :
   ```bash
   python manage.py migrate
   ```

4. **Lancer le serveur de développement** :
   ```bash
   python manage.py runserver 0.0.0.0:8001
   ```

Accédez à l'application sur [http://localhost:8001/](http://localhost:8001/).

---

## Structure du Projet

```
├── bank_project/           # Configuration globale Django (URLs, settings, WSGI)
│   ├── settings.py
│   ├── urls.py
│   └── views.py            # Logique d'inférence et API REST
├── dataset/                # Fichiers de données (bank.csv, bank-full.csv)
├── models/                 # Artéfact sérialisé (bank_model_assets.pkl)
├── static/                 # Fichiers statiques et assets graphiques
├── templates/
│   └── index.html          # Interface utilisateur Tailwind CSS
├── train_bank.py           # Script d'entraînement des modèles de classification
├── bank_prediction.ipynb   # Notebook Jupyter d'EDA et d'explications mathématiques
└── manage.py
```
