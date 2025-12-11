# ✈️ Tunisair Flight Delay Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)  
![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-orange)  
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)  
![Tableau](https://img.shields.io/badge/Tableau-Visualization-blue)  
![Status](https://img.shields.io/badge/Project-Complete-brightgreen)

---

## 📘 Project Overview — A Short Story

Flight delays are disruptive, expensive, and unpredictable. Tunisair provides thousands of flight entries each year — but what patterns explain these delays, and can we predict them?
This project combines exploratory analysis, feature engineering, machine learning, and interactive reporting to understand why Tunisair flights are delayed and to build a model that predicts those delays before they occur.

You’ll find:
- Cleaned and enriched flight records
- Route-level temporal features
- Machine learning models (Logistic Regression & Random Forest)
- Exported prediction dataset for BI tools
- An interactive Streamlit delay predictor
- A Tableau dashboard showing operational insights
---

## 📂 Repository Structure

```
tunisair-flight-delay-analysis/
│── README.md
│── requirements.txt
│── images/
│   ├── Flight_ERD.png
│   ├── TunisairDelayPredictor.gif           
│   └── TableauPublic-flight_delay.gif
│
│── data/
│   ├── Tunisair_flights_dataset.csv        
│   ├── airports2.csv                        
│   └── flight_delay_with_predictions.csv    
│
│── models/
│   ├── tunisair_best_model.joblib           
│   └── tunisair_delay_regressor.joblib     
│
│── tableau/
│   └── flight_delay_visual.twb
            
```


---
## 🛫 Data Used
### 1. Tunisair Flight Dataset (Tunisair_flights_dataset.csv)
Includes:
- scheduled_dep, actual_dep, scheduled_arr, actual_arr
- origin, destination, flight_id
- dep_delay (target variable, minutes)
- Aircraft & route information

### 2. Airports Metadata (airports2.csv)
Used to enhance route context:
- Airport name, IATA code
- Country, region, location

### 3. Predictions Export (flight_delay_with_predictions.csv)
Output from the ML model for Tableau dashboards.

---

## 🧭 Analytical Workflow
### 1) Initial Load & Timestamp Parsing

```python
df = pd.read_csv(
    'data/Tunisair_flights_dataset.csv',
    parse_dates=['scheduled_dep','actual_dep','scheduled_arr','actual_arr']
)
df.head()

```

**This allows detection of missing timestamps, impossible sequences, timezone inconsistencies, etc.** 

---

### 2) Data Cleaning
Removing extreme or impossible values:

```python
df = df[(df['dep_delay'] >= -60) & (df['dep_delay'] <= 1440)]
df['day_of_week'] = df['scheduled_dep'].dt.day_name()
```

**Why?**
- Negative delays below –60 mins indicate data entry errors
- Delays over 24 hours (1440 mins) are operationally invalid
- Weekday is a strong feature for predicting operational congestion

---

### 3) Feature Engineering (The Secret Sauce)
 **Rolling route performance (last 30 flights)**

```python
df = df.sort_values(['origin','destination','scheduled_dep'])
df['route_avg_delay_30'] = df.groupby(
    ['origin','destination']
)['dep_delay'].transform(lambda x: x.rolling(30, min_periods=1).mean())
```

Why this matters:
Delays tend to propagate through routes due to crew cycles, airport congestion, and aircraft availability.

**Additional engineered features:**
- Hour of day / month / year
- Busy-hour flag
- Weekend flag
- Short-haul vs long-haul route
- Seasonal traffic patterns

---

### 4) Modeling — comparing classifiers
Models tested:
| Model                        | Purpose                | Notes                           |
| ---------------------------- | ---------------------- | ------------------------------- |
| **Logistic Regression**      | interpretable baseline | Captures linear patterns        |
| **Random Forest Classifier** | production model       | Handles non-linear interactions |


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


## 🧠 Key Insights from the Analysis
### Operational Findings:
- Certain origin→destination routes have persistent delay histories
- Early morning flights tend to depart on time
- Congestion peaks on Wednesdays and Fridays
- High route-level rolling averages are strong red flags
- Significant delays often cascade across subsequent flights

### Business Recommendations:
1. Deploy reserve crew & aircraft on high-risk routes
2. Monitor rolling route delays to detect early cascade patterns
3. Proactive passenger communication for predicted delay flights
4. Enrich the model with:
- Weather data
- ATC event logs
- Airport-level congestion metrics

---

## How to reproduce

1. Clone repo.  
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
