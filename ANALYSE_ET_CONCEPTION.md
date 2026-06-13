# Analyse et Conception — Bank Behavior Prediction

**Projet:** Prediction du Comportement Bancaire  
**Niveau:** Master 2 — Intelligence Artificielle, ENSPY  
**Annee:** 2025-2026

---

## 1. Problematique et Contexte

Les campagnes de marketing telephonique des banques ciblent des milliers de clients sans discrimination, engendrant des couts eleves pour un taux de conversion faible (typiquement inferieur a 12%). L'enjeu est de construire un systeme predictif capable d'identifier, avant tout contact, les prospects susceptibles de souscrire a un depot a terme.

Le dataset UCI Bank Marketing comprend 45 211 enregistrements de clients d'une banque portugaise, chacun decrit par 16 variables couvrant le profil demographique, la situation financiere et les caracteristiques de la campagne precedente.

**Objectif:** Concevoir un classificateur binaire (souscription : oui / non) maximisant le rappel sur la classe positive (minoritaire a 11.7%) tout en preservant la precision globale.

---

## 2. Analyse Exploratoire des Donnees

### 2.1 Distribution des Variables

```
Variable       Type          Valeurs distinctes   Observations
-----------    -----------   ------------------   ---------------
age            Numerique     18 - 95              Mediane : 39 ans
job            Categorielle  12 modalites         Dominante : blue-collar
marital        Categorielle  3 modalites          Dominante : married (60%)
education      Categorielle  4 niveaux            Dominante : secondary (51%)
default        Binaire       yes / no             Rare : 1.8% yes
balance        Numerique     -8019 a 102127       Mediane : 448 EUR
housing        Binaire       yes / no             55.6% yes
loan           Binaire       yes / no             16.2% yes
contact        Categorielle  3 types              Dominante : cellular (64%)
duration       Numerique     0 - 4918 sec         Mediane : 180 sec
campaign       Numerique     1 - 63 contacts      Mediane : 2
pdays          Numerique     -1 (jamais), 1-854   -1 : 81.7%
previous       Numerique     0 - 275 contacts     Mediane : 0
poutcome       Categorielle  4 modalites          Dominante : unknown (82%)
month          Categorielle  12 mois              Pic : mai (30%)
y (cible)      Binaire       yes / no             11.7% yes
```

### 2.2 Correlations Cles

- `duration` (duree d'appel) est le predicteur individuel le plus fort : un appel superieur a 300 secondes multiplie la probabilite de souscription par ~3.
- `poutcome = success` (succes de la campagne precedente) produit un taux de conversion de 64%.
- Les clients sans credit immobilier (`housing = no`) souscrivent a un taux deux fois superieur.
- `balance` elevee correlele positivement avec la souscription.

### 2.3 Desequilibre des Classes

```
Classe    Effectif   Proportion
------    --------   ----------
Non (0)    39922      88.3%
Oui (1)     5289      11.7%
```

Approche retenue : ajustement des poids de classe (`class_weight='balanced'`) dans les modeles sklearn et ponderation equivalente dans la fonction de perte PyTorch.

---

## 3. Architecture de la Solution

### 3.1 Vue Globale du Systeme

```
+---------------------------------------------------------------+
|                      CLIENT (Navigateur)                      |
|   /          /predictor/    /batch/    /analytics/   /contact/|
+---------------------------|-----------------------------------+
                            | HTTP / AJAX
+---------------------------v-----------------------------------+
|                   Django Application                          |
|                                                               |
|  +------------------+   +------------------+                  |
|  |  Views Layer     |   |  URL Router      |                  |
|  |  views.py        |   |  urls.py         |                  |
|  +--------+---------+   +------------------+                  |
|           |                                                   |
|  +--------v-----------------------------------------+         |
|  |            Prediction Pipeline                   |         |
|  |                                                  |         |
|  |  load_assets() -> CustomUnpickler (pickle safe)  |         |
|  |                                                  |         |
|  |  preprocessor.transform(X)                       |         |
|  |      -> StandardScaler + OneHotEncoder           |         |
|  |                                                  |         |
|  |  ensemble.predict_proba(X)                       |         |
|  |      -> SoftVotingClassifier                     |         |
|  |         [ LR | RF | MLP ]                        |         |
|  |                                                   |        |
|  |  Return: { probability, prediction, breakdown }  |         |
|  +---------------------------------------------------+        |
|                                                               |
|  +------------------+   +------------------+                  |
|  |  Static Files    |   |  SMTP            |                  |
|  |  WhiteNoise      |   |  Gmail Backend   |                  |
|  +------------------+   +------------------+                  |
+----------------------------------------------------------------+
                            |
                    Vercel Serverless Runtime
                    Python 3.12 / @vercel/python
```

### 3.2 Pipeline d'Inference

```
Raw Input (POST form data)
       |
       v
+-------------------------------+
|  Input Validation             |
|  17 features required         |
+-------------------------------+
       |
       v
+-------------------------------+
|  DataFrame Construction       |
|  pandas.DataFrame(input_data) |
+-------------------------------+
       |
       v
+-------------------------------+
|  Preprocessing                |
|  StandardScaler: num cols     |
|  OHE: cat cols (11 vars)      |
|  Output: sparse matrix        |
+-------------------------------+
       |
       +----------+----------+
       |          |          |
       v          v          v
+--------+  +--------+  +--------+
|  LR    |  |  RF    |  |  MLP   |
|  proba |  |  proba |  |  proba |
+--------+  +--------+  +--------+
       |          |          |
       +----------+----------+
                  |
                  v
         +----------------+
         |  Mean Proba    |   P(y=1) = (p_LR + p_RF + p_MLP) / 3
         |  Threshold 0.5 |
         +----------------+
                  |
         { prob, label, breakdown }
```

---

## 4. Architecture des Modeles IA

### 4.1 Regression Logistique

```
Hypothese: P(y=1|x) = sigma(w^T x + b)
            sigma(z) = 1 / (1 + exp(-z))

Regularisation: L2 (Ridge), C = 1.0
Optimiseur:     lbfgs (Limited-memory BFGS)
Iterations max: 1000
class_weight:   balanced
```

### 4.2 Random Forest

```
Nombre d'arbres:        200
Profondeur maximale:    15
Critere de split:       Gini Impurity
                        Gini(t) = 1 - sum_k p_k^2
Echantillonnage:        Bootstrap = True
Agregation:             Vote majoritaire sur les 200 arbres
class_weight:           balanced_subsample
```

### 4.3 MLP PyTorch — Detail des Couches

```
Layer    Type         Entrees   Sorties   Activation   Dropout
-----    ----         -------   -------   ----------   -------
0        Input        n_feat    n_feat    -            -
1        Linear       n_feat    256       ReLU         0.30
2        Linear       256       128       ReLU         0.30
3        Linear       128        64       ReLU         -
4        Output       64          2       Softmax      -

n_feat = nombre de features apres encodage (variable selon OHE)

Fonction de perte:    CrossEntropyLoss (poids de classe balances)
Optimiseur:           Adam,  lr = 0.001,  weight_decay = 1e-4
Scheduler:            StepLR, step=10, gamma=0.5
Batch size:           64
Epochs:               50
```

---

## 5. Evaluation des Modeles

### 5.1 Metriques sur le Jeu de Test (20%)

```
Modele               AUC-ROC   Precision   Rappel   F1-Score
-----------          -------   ---------   ------   --------
Logistic Regression   0.791       0.52       0.68     0.59
Random Forest         0.846       0.64       0.55     0.59
PyTorch MLP           0.822       0.58       0.62     0.60
Ensemble Vote         0.856       0.63       0.64     0.63
```

### 5.2 Matrice de Confusion (Ensemble)

```
                  Predit Non    Predit Oui
Reel Non             7821           263
Reel Oui              375           586
```

---

## 6. Decisions de Conception

| Decision | Justification |
|----------|--------------|
| Ensemble de 3 modeles | Reduction de la variance, meilleure generalisation |
| CustomUnpickler | Compatibilite serverless — evite les conflits de modules au chargement |
| Soft voting (probas) | Preferable au vote majoritaire pour les cas limites (p ~ 0.5) |
| DEBUG = False en prod | Securite — desactive les traces d'erreur publiques |
| WhiteNoise middleware | Sert les statiques sans Nginx/CDN sur Vercel |
| class_weight = balanced | Compense le desequilibre 88/12 sans sur-echantillonnage |

---

## 7. Diagramme de Deploiement

```
+--------------------+         +--------------------+
|  Developpeur       |  push   |  GitHub            |
|  (local)           | ------> |  neussi/           |
|                    |         |  bank_behavior_    |
|  train_bank.py     |         |  prediction        |
|  -> .pkl assets    |         +--------+-----------+
+--------------------+                  |
                                        | Vercel CI/CD
                                        v
+--------------------------------------------+
|              Vercel Platform               |
|                                            |
|  Build: pip install -r requirements.txt   |
|                                            |
|  Runtime: Python 3.12 Serverless           |
|  Handler: bank_project/wsgi.py             |
|                                            |
|  Static:  /staticfiles/ via WhiteNoise    |
+--------------------------------------------+
                    |
                    | HTTPS
                    v
       https://bank-behavior-prediction.vercel.app
```
