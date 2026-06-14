#  Airline Passenger Satisfaction - ML Pipeline

End-to-end machine learning pipeline for predicting airline passenger satisfaction.

- **Source:** [Kaggle - Real Airline Passenger Satisfaction Dataset](https://www.kaggle.com/datasets/rustam32/real-airline-passenger-satisfaction-dataset)
- **Samples:** 57,514 | **Features:** 145 | **Target:** `liked` (0 = Not Satisfied, 1 = Satisfied)

---

## 1. Problem Description

**Domain:** Airport & Airline Services  
**Problem type:** Binary Classification  
**Target variable:** `liked` — whether a passenger was satisfied with their overall airport experience.  
This is a useful prediction task because understanding what drives passenger satisfaction allows airports to prioritize service improvements.

---

## 2. Dataset Description

| Property | Value |
|----------|-------|
| Total samples | 57,514 |
| Total features | 145 |
| Categorical features | 14 |
| Numerical features | 131 |
| Missing values | 0 (some `<null>` strings in categoricals) |
| Class balance | 50% / 50% |

### Feature Categories
- **Process info:** process, flight_type, connection, ticket info
- **Ratings (~60 columns):** checkin, security, food, parking, restrooms, etc.
- **Demographics:** gender, age_group, nationality, education, household_income
- **Trip info:** trip_purpose, traveling_alone, number_of_companions
- **Applicability flags (~60 columns):** binary indicators (0/1) showing if a rating was applicable

### Target Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| 0 - Not Satisfied | 28,757 | 50% |
| 1 - Satisfied | 28,757 | 50% |

---

## 3. Preprocessing Approach

**Split first, preprocess second** — all statistics derived from training set only.

| Step | Strategy | Reason |
|------|----------|--------|
| Split | 80/10/10 stratified | Preserve class balance |
| Drop columns | 59 `_is_applicable` + `flight_type` + `gender` | Not statistically significant (chi-square p>0.05) |
| Missing values | Numerical→median, Categorical→mode (from train only) | `<null>` strings replaced with NaN first |
| Outliers | IQR Winsorizing (bounds from train only) | Caps extreme values without removing rows |
| Encoding | LabelEncoder for all categoricals | Handles unseen labels in val/test |
| Time columns | `connection_wait_time` & `arrival_lead_time` converted from seconds to minutes | More interpretable scale |
| Scaling | StandardScaler (fitted on train only) | Required for Logistic Regression & Neural Network |

---

## 4. Feature Engineering

4 new features derived from EDA insights:

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `comfort_score` | mean(boarding_lounge_comfort, thermal_comfort, acoustic_comfort) | Top correlated features with liked (r=0.55) |
| `cleanliness_score` | mean(overall_airport_cleanliness, restroom_cleanliness, restroom_maintenance) | Passengers evaluate cleanliness holistically |
| `price_quality_score` | mean(food_beverage_price_quality, retail_price_quality, parking_value_for_money) | Lowest rated services — key pain point |
| `queue_score` | mean(checkin_queue_wait_time, security_queue_wait_time) | Wait times affect overall satisfaction |

---

## 5. PCA Insights

- **61 components** needed to explain 90% of total variance
- First 2 components show partial separation between satisfied/not satisfied passengers
- Features with highest loadings on PC1: `boarding_lounge_comfort`, `overall_airport_cleanliness`, `restrooms`
- Consistent with EDA correlation findings

![PCA Scree Plot](SRC/plots/pca_scree.png)
![PCA 2D Projection](SRC/plots/pca_2d.png)

---

## 6. Model Comparison

All models evaluated on the **test set only** (5,752 samples).

| Metric | Random Forest | Logistic Regression | Neural Network |
|--------|--------------|---------------------|----------------|
| Accuracy | 80.44% | 80.75% | **81.61%** |
| Precision | 77.35% | 78.90% | **78.95%** |
| Recall | 86.09% | 83.97% | **86.20%** |
| F1-score | 81.49% | 81.35% | **82.41%** |
| AUC-ROC | 88.32% | 88.11% | **89.59%** |

### Discussion
- The **Neural Network outperformed both classical models** on all metrics
- Random Forest had the highest Recall but lower Precision
- Logistic Regression performed competitively after StandardScaler was applied
- The Neural Network used early stopping, suggesting fast convergence on this dataset
- The performance gap between models is small (~1-2%), suggesting the features are informative regardless of model complexity

![Model Comparison](SRC/plots/model_comparison.png)

### Confusion Matrices
![Confusion Matrix - Random Forest](SRC/plots/confusion_matrix_Random_Forest.png)
![Confusion Matrix - Logistic Regression](SRC/plots/confusion_matrix_Logistic_Regression.png)
![Confusion Matrix - Neural Network](SRC/plots/confusion_matrix_Neural_Network.png)

### ROC Curves
![ROC Curve - Random Forest](SRC/plots/roc_curve_Random_Forest.png)
![ROC Curve - Logistic Regression](SRC/plots/roc_curve_Logistic_Regression.png)
![ROC Curve - Neural Network](SRC/plots/roc_curve_Neural_Network.png)

### Feature Importance (Random Forest)
![Feature Importance](SRC/plots/rf_feature_importance.png)

### Logistic Regression Coefficients
![LR Coefficients](SRC/plots/lr_coefficients.png)

### Neural Network Loss Curves
![NN Loss Curves](SRC/plots/nn_loss_curves.png)

---

## 7. Best Model Designation

**Best model: Neural Network** (`Models/best_model.pt`)

Justified by highest F1-score (82.41%) and AUC-ROC (89.59%) on the test set.

### Architecture
Input(84) → Linear(128) → ReLU → Dropout(0.3)

→ Linear(64)  → ReLU → Dropout(0.2)

→ Linear(32)  → ReLU

→ Linear(1)   → Sigmoid
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Binary Cross-Entropy
- **Early stopping:** patience=10

---

##  EDA Highlights

### Satisfaction Rate by Age Group
![Satisfaction by Age Group](plots/liked_by_age_group.png)

### Satisfaction Rate by Trip Purpose
![Satisfaction by Trip Purpose](plots/liked_by_trip_purpose.png)

### Satisfaction Rate by Process
![Satisfaction by Process](plots/liked_by_process.png)

### Top 10 Features Correlated with Satisfaction
![Top 10 Correlations](plots/top10_correlations.png)

### 10 Lowest Rated Services
![Lowest Rated Services](plots/lowest_rated_services.png)

### Business vs Leisure
![Business vs Leisure](plots/business_vs_leisure.png)

### Correlation Heatmap
![Correlation Heatmap](plots/heatmap_top_features.png)

---

##  Installation & Execution

```bash
# 1. Clone the repo
git clone https://github.com/antoniakatsouri1-math/airline-passenger-satisfaction.git
cd airline-passenger-satisfaction

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python main.py
```

Results are saved in:
- `plots/` — EDA visualizations
- `SRC/plots/` — Model & PCA plots
- `Models/` — Trained models & scaler