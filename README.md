# EV Station Maintenance Predictor

A machine learning project that predicts whether an EV charging station requires maintenance, based on daily usage, downtime hours, and maintenance cost data.

## Live Demo

Open `web/index.html` in any browser — no server needed.  
Or deploy instantly via [Netlify Drop](https://app.netlify.com/drop) by dragging the `web/` folder.

---

## Project Structure

```
ev-maintenance-predictor/
├── data/
│   └── raw_station_data.csv       # 500-row training dataset
├── notebook/
│   └── Proj_1_4_Final.ipynb       # Model training & evaluation notebook
├── web/
│   └── index.html                 # Self-contained web app (models baked in)
├── docs/
│   └── confidence_explanation.md  # How confidence is calculated
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

| Column | Description |
|---|---|
| `Station_ID` | Unique station identifier |
| `Daily_Usage` | Charging sessions per day (80–700) |
| `Downtime_Hours` | Hours offline per day (0–8) |
| `Maintenance_Cost` | Repair cost this month in ₹ (500–4500) |
| `Requires_Maintenance` | Target label: 0 = No, 1 = Yes |

- **500 rows**, 260 class-0 / 240 class-1 (balanced)
- Generated with realistic distributions matching real EV station operating patterns

---

## Models

Both models are trained on an 80/20 train-test split (`random_state=42`).

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 100% | StandardScaler applied; unscaled coefs embedded in web app |
| Random Forest | 100% | 50 trees, max_depth=6; all trees exported to JS |

**Ensemble**: Final confidence = (LR probability + RF vote ratio) ÷ 2  
**Threshold**: ≥ 50% → Maintenance Required

### Feature weights (Logistic Regression)

| Feature | Coefficient | Relative importance |
|---|---|---|
| Downtime Hours | 2.0237 | **Strongest** |
| Maintenance Cost | 0.0036 | Moderate |
| Daily Usage | 0.0015 | Weakest |

---

## Web App Features

- Live risk gauge that updates as sliders move
- Per-prediction confidence breakdown (LR vs RF scores)
- Feature influence bars showing what drove the prediction
- Prediction history table and chart
- Fully offline — no backend, no API calls

---

## Running the Notebook

```bash
pip install -r requirements.txt
jupyter notebook notebook/Proj_1_4_Final.ipynb
```

Make sure `data/raw_station_data.csv` is accessible. The notebook reads it as:
```python
df = pd.read_csv("../data/raw_station_data.csv")
```

---



## Author

Pednekar Atharva Pramod
