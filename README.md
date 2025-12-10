# ✈️ Tunisair Flight Delay Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)  
![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-orange)  
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)  
![Tableau](https://img.shields.io/badge/Tableau-Visualization-blue)  
![Status](https://img.shields.io/badge/Project-Complete-brightgreen)

---

## Project Snapshot — a short story

I explored Tunisair's flight logs to understand why flights get delayed and whether we could predict these delays. This case study blends exploratory storytelling with applied machine learning: we clean the data, craft features that capture temporal and route patterns, test models (Logistic Regression and Random Forest), and deploy predictions for visualization (Tableau) and ad-hoc inference (Streamlit).

Each code snippet below has been chosen to illustrate a key step in the analysis and decision-making process.

---

## Repository 

```
tunisair-flight-delay-analysis/
│── README.md
│── requirements.txt
│── images/Flight_ERD.png
│── data/
│   ├── Tunisair_flights_dataset.csv        
│   ├── airports2.csv                        
│   └── flight_delay_with_predictions.csv    
│── models/
│   ├── tunisair_best_model.joblib           
│   └── tunisair_delay_regressor.joblib     
│── images/
│   ├── TunisairDelayPredictor.gif           
│   └── TableauPublic-flight_delay.gif       
│── tableau/
│   └── flight_delay_visual.twb              
```


---

## The analytical journey (story + code)

### 1) Loading the data — first glance
We begin by loading the raw flight logs and parsing the timestamp columns to set expectations:

```python
import pandas as pd

df = pd.read_csv('data/Tunisair_flights_dataset.csv', parse_dates=['scheduled_dep', 'actual_dep', 'scheduled_arr', 'actual_arr'])
df.head()
```

**Narrative:** At this stage I check for missing timestamps, impossible times, and columns needed for modeling.

---

### 2) Cleaning & sanity checks
Clip unrealistic delays and add human-friendly time features for modeling:

```python
df = df[(df['dep_delay'] >= -60) & (df['dep_delay'] <= 1440)]
df['day_of_week'] = df['scheduled_dep'].dt.day_name()
```

**Narrative:** This removes clear data-entry errors and creates features like `day_of_week` that are highly predictive of delay patterns.

---

### 3) Feature engineering — route history
Capture recent route performance via rolling averages (e.g., last 30 flights on that origin→destination):

```python
df = df.sort_values(['origin','destination','scheduled_dep'])
df['route_avg_delay_30'] = df.groupby(['origin','destination'])['dep_delay'].transform(lambda x: x.rolling(30, min_periods=1).mean())
```

**Narrative:** Route-level rolling averages are strong short-term predictors: delays often cascade along routes with recent problems.

---

### 4) Modeling — comparing classifiers
We trained both Logistic Regression and Random Forest. Random Forest was chosen as the production model.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
```

**Narrative:** Logistic Regression is an interpretable baseline. Random Forest captures non-linear interactions between schedule, route, and aircraft features.

---

### 5) Evaluation — classification report & ROC-AUC
We used precision/recall/f1 and ROC-AUC to compare models:

```python
from sklearn.metrics import classification_report, roc_auc_score

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC (RF):", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))
```

**Narrative:** Use ROC-AUC together with class-level metrics to balance operational trade-offs: false positives (unnecessary alerts) vs false negatives (missed delays).

---

### 6) Exporting predictions for Tableau
After generating predictions, export the enriched dataset for Tableau:

```python
df[['flight_id','scheduled_dep','origin','destination','predicted_dep_delay']].to_csv('data/flight_delay_with_predictions.csv', index=False)
```

**Narrative:** The export acts as a single source-of-truth for interactive dashboards and operational reporting.

---

## Model results (summary)

- **Models tested:** Logistic Regression, Random Forest  
- **Best model:** Random Forest (ROC-AUC ≈ 0.7895)

**Logistic Regression — classification snapshot**
```
precision    recall  f1-score   support
0       0.75      0.67      0.71     14376
1       0.66      0.74      0.70     12583
accuracy                           0.70     26959
ROC-AUC (LR): 0.7822
```

**Random Forest — classification snapshot**
```
precision    recall  f1-score   support
0       0.73      0.72      0.72     14376
1       0.68      0.70      0.69     12583
accuracy                           0.71     26959
ROC-AUC (RF): 0.7895
```

**Interpretation:** Random Forest offered a modest but meaningful improvement in ROC-AUC and balanced precision/recall across classes, so it was saved as `tunisair_best_model.joblib`.

---

## Operational recommendations

- Monitor chronic-delay origin→destination pairs and allocate reserve crew on those routes.  
- Use the Streamlit app to run ad-hoc predictions for scheduled flights and inform passengers proactively.  
- Enrich the data with weather and ATC event features to further improve accuracy.

---

## How to reproduce

1. Clone repo and add real data/models to the `data/` and `models/` folders.  
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the notebook:
```bash
jupyter notebook flight_analysis.ipynb
```
4. Run the Streamlit app:
```bash
streamlit run app/tunisair_app.py
```
5. Open the Tableau workbook: `tableau/flight_delay_visual.twb`.

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
joblib
streamlit
matplotlib
seaborn
plotly
```

---

## Author

**Paul Egeonu**  
_Data Analyst | Data Scientist_  
[LinkedIn](https://www.linkedin.com/in/paul-egeonu) | [GitHub](https://github.com/Paul-Egeonu)
