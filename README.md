# Bank Behavior Prediction

A production-ready machine learning platform for predicting bank client subscription to term deposits, built on an ensemble of interpretable classifiers and exposed through a premium Django web interface.

**Production URL:** https://bank-behavior-prediction.vercel.app  
**Repository:** https://github.com/neussi/bank_behavior_prediction

---

## Platform Overview

The platform analyzes 17 client and campaign features from the UCI Bank Marketing dataset to predict whether a client will subscribe to a term deposit. It combines three classification models — Logistic Regression, Random Forest, and a PyTorch MLP — into a soft-voting ensemble that achieves superior generalization over any single model.

The web interface provides four dedicated sections:

| Route | Section | Description |
|-------|---------|-------------|
| `/` | Home | Platform overview, key metrics, model summary |
| `/predictor/` | Simulateur Unitaire | Real-time single-client prediction with gauge visualization |
| `/batch/` | Traitement en Lot | CSV bulk upload for campaign-scale scoring |
| `/analytics/` | Analytique | Diagnostic plots: ROC curve, confusion matrix, correlation heatmap |
| `/contact/` | Contact | SMTP contact form (Gmail backend) |

---

## Project Structure

```
bank_behavior_prediction/
|
+-- bank_project/               Django project configuration
|   +-- settings.py             Application settings (WhiteNoise, SMTP, CORS)
|   +-- urls.py                 URL routing
|   +-- views.py                All view functions and prediction logic
|   +-- wsgi.py                 WSGI entry point (Vercel serverless)
|   +-- asgi.py                 ASGI entry point
|
+-- templates/                  Jinja2 HTML templates
|   +-- base.html               Master layout (Tailwind CSS, Outfit font)
|   +-- home.html               Landing page with metrics dashboard
|   +-- predict.html            Single prediction form and result gauge
|   +-- batch.html              Batch CSV upload and results table
|   +-- analytics.html          Diagnostic charts and model evaluation
|   +-- contact.html            Contact form with AJAX submission
|
+-- static/
|   +-- images/                 Pre-generated diagnostic plots (PNG)
|
+-- staticfiles/                Collected static assets (WhiteNoise, Vercel)
|
+-- dataset/
|   +-- bank.csv                UCI Bank Marketing dataset (small split)
|   +-- bank-full.csv           Full dataset (45,211 samples)
|   +-- bank-names.txt          Feature description file
|
+-- models/
|   +-- bank_model_assets.pkl   Serialized models, preprocessor, metrics
|
+-- docs/
|   +-- images/                 High-resolution analysis plots for documentation
|
+-- train_bank.py               Offline training pipeline script
+-- bank_prediction.ipynb       Full Jupyter analysis and training notebook
+-- requirements.txt            Python dependencies
+-- vercel.json                 Vercel deployment configuration
+-- manage.py                   Django management CLI
+-- .gitignore
```

---

## AI Model Architecture

Three classifiers are trained and combined into a soft-voting ensemble:

```
Input Features (17 variables)
        |
        v
+---------------------------------------+
|          Preprocessing Pipeline       |
|  - StandardScaler (numerical)         |
|  - OneHotEncoder (categorical)        |
+---------------------------------------+
        |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
+---------------+   +---------------+   +---------------+
| Logistic      |   | Random Forest |   | PyTorch MLP   |
| Regression    |   | 200 estimators|   | 3-layer NN    |
| C=1.0, L2     |   | max_depth=15  |   | 256-128-64    |
+---------------+   +---------------+   +---------------+
        |                   |                   |
        +-------------------+-------------------+
                            |
                    +---------------+
                    | Soft Voting   |
                    | Ensemble      |
                    | (mean proba)  |
                    +---------------+
                            |
                    P(subscription)  [0.0 - 1.0]
```

### PyTorch MLP Architecture

```
Layer       Type          Units    Activation
-----       ----          -----    ----------
Input       Linear        n_feat   -
Hidden 1    Linear        256      ReLU + Dropout(0.3)
Hidden 2    Linear        128      ReLU + Dropout(0.3)
Hidden 3    Linear         64      ReLU
Output      Linear          2      Softmax
```

Training: Adam optimizer, lr=0.001, Binary Cross-Entropy loss, 50 epochs, batch size 64.

---

## Dataset

**Source:** UCI Machine Learning Repository - Bank Marketing Dataset  
**Samples:** 45,211 client records  
**Target:** `y` - subscription to term deposit (binary: yes/no)  
**Class imbalance:** approximately 88.3% no / 11.7% yes  
**Features:** 16 input variables across demographic, financial, and campaign categories

---

## Local Development

```bash
# Clone repository
git clone https://github.com/neussi/bank_behavior_prediction.git
cd bank_behavior_prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train models (requires dataset)
python train_bank.py

# Run development server
python manage.py runserver
```

---

## Deployment

The application is deployed on **Vercel** using the `@vercel/python` runtime with WhiteNoise serving all static assets directly through the WSGI handler.

**Environment variables required on Vercel:**

| Variable | Description |
|----------|-------------|
| `EMAIL_HOST_PASSWORD` | Gmail App Password for SMTP |

**Configuration file:** `vercel.json` routes all traffic through `bank_project/wsgi.py`.

---

## Dependencies

| Package | Role |
|---------|------|
| `django>=5.0` | Web framework |
| `scikit-learn` | Logistic Regression, Random Forest, preprocessing |
| `numpy`, `pandas` | Data manipulation |
| `whitenoise` | Static file serving in production |
| `django-cors-headers` | Cross-origin request handling |

---

## Contact

**Institution:** Ecole Nationale Superieure Polytechnique de Yaounde (ENSPY)  
**Level:** Master 2 - Intelligence Artificielle  
**Contact:** npe.techs@gmail.com | +237 650 970 526
