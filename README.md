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

### ROC Curves

![ROC Curve - Random Forest](SRC/plots/roc_curve_Random_Forest.png)

**Random Forest (AUC=0.883):** Strong discriminative ability. The curve rises steeply at low False Positive Rates, meaning the model correctly identifies most satisfied passengers before making many errors.

![ROC Curve - Logistic Regression](SRC/plots/roc_curve_Logistic_Regression.png)

**Logistic Regression (AUC=0.881):** Very similar to Random Forest. Slightly lower AUC but comparable shape, confirming that the relationship between features and satisfaction is largely linear.

![ROC Curve - Neural Network](SRC/plots/roc_curve_Neural_Network.png)

**Neural Network (AUC=0.895):** Highest AUC of all three models. The curve hugs the top-left corner more tightly, indicating better discrimination between satisfied and dissatisfied passengers across all thresholds.

---
### ROC Curves

![ROC Curve - Random Forest](SRC/plots/roc_curve_Random_Forest.png)

**Random Forest (AUC=0.883):** Strong discriminative ability. The curve rises steeply at low False Positive Rates, meaning the model correctly identifies most satisfied passengers before making many errors.

![ROC Curve - Logistic Regression](SRC/plots/roc_curve_Logistic_Regression.png)

**Logistic Regression (AUC=0.881):** Very similar to Random Forest. Slightly lower AUC but comparable shape, confirming that the relationship between features and satisfaction is largely linear.

![ROC Curve - Neural Network](SRC/plots/roc_curve_Neural_Network.png)

**Neural Network (AUC=0.895):** Highest AUC of all three models. The curve hugs the top-left corner more tightly, indicating better discrimination between satisfied and dissatisfied passengers across all thresholds.

---
### Feature Importance (Random Forest)

![Feature Importance](SRC/plots/rf_feature_importance.png)

The Random Forest identifies `location_and_movement` as the single most important feature (importance ≈ 0.18), followed by `boarding_lounge_comfort` and the engineered `comfort_score`. This confirms our EDA finding that physical comfort and navigation within the airport are the primary drivers of satisfaction. Notably, our engineered features (`comfort_score`, `cleanliness_score`) appear in the top 5, validating the feature engineering step.

---


### Logistic Regression Coefficients

![LR Coefficients](SRC/plots/lr_coefficients.png)

The Logistic Regression assigns the highest absolute coefficient to `location_and_movement` (≈ 0.9), consistent with the Random Forest. `baggage_claim_process` and `boarding_lounge_comfort` follow closely. The model also highlights `connection` type and `trips_last_12_months` as influential — frequent travelers and those with direct connections tend to be more satisfied. This linear model provides highly interpretable weights directly tied to satisfaction probability.

---

### Neural Network Loss Curves

![NN Loss Curves](SRC/plots/nn_loss_curves.png)

The training loss decreases steadily across all 19 epochs, while the validation loss plateaus around epoch 5-7 and then slightly increases — a classic sign of mild overfitting. Early stopping correctly triggered at epoch 19, restoring the best weights from the lowest validation loss. The gap between train and validation loss is small, suggesting the model generalizes well to unseen data. The fast convergence (only 19 epochs) reflects the high informativeness of the features.

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
---

## FastAPI Endpoint

The best model (Neural Network) is exposed as a REST API using FastAPI.

### Start the API

The API starts automatically after the pipeline completes. To start it manually:

```bash
python main.py
```

Then open: **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

### Endpoint

**POST** `/predict`

Accepts passenger data as JSON and returns a satisfaction prediction.

**Response format:**
```json
{
  "prediction": 1,
  "label": "Satisfied",
  "probability": 0.9942
}
```

---

### Example 1 — Highly Satisfied Passenger 

A young leisure traveler with excellent ratings across all services.

**Input:**
```json
{
  "process": "Boarding",
  "month": "June",
  "connection": "Boarded at the airport",
  "ticket_purchased_by": "By the passenger",
  "age_group": "18 to 25 years",
  "trip_purpose": "Leisure",
  "traveling_alone": "No",
  "boarding_lounge_comfort": 5,
  "overall_airport_cleanliness": 5,
  "restrooms": 5,
  "restroom_cleanliness": 5,
  "restroom_maintenance": 5,
  "parking": 5,
  "parking_facility_quality": 5,
  "parking_value_for_money": 5,
  "baggage_claim_process": 5,
  "checkin_process": 5,
  "checkin_queue_wait_time": 1,
  "security_screening_process": 5,
  "security_queue_wait_time": 1,
  "food_beverage_outlets": 5,
  "food_beverage_price_quality": 5,
  "retail_outlets": 5,
  "retail_price_quality": 5,
  "thermal_comfort": 5,
  "acoustic_comfort": 5,
  "airline_service": 5,
  "staff_courtesy": 5,
  "seat_availability": 5,
  "power_outlet_availability": 5,
  "airport_internet": 5,
  "signage": 5,
  "location_and_movement": 5
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Satisfied",
  "probability": 0.9942
}
```

**Interpretation:** The model is 99.42% confident this passenger is satisfied. High ratings across all key services (lounge comfort, cleanliness, baggage, food) combined with short queue wait times strongly predict satisfaction.

---

### Example 2 — Dissatisfied Passenger 

A middle-aged business traveler with poor ratings and long queues.

**Input:**
```json
{
  "process": "Disembarkation",
  "month": "January",
  "connection": "Connecting",
  "ticket_purchased_by": "By third parties",
  "age_group": "36 to 45 years",
  "trip_purpose": "Business",
  "traveling_alone": "Yes",
  "boarding_lounge_comfort": 1,
  "overall_airport_cleanliness": 1,
  "restrooms": 1,
  "restroom_cleanliness": 1,
  "restroom_maintenance": 1,
  "parking": 1,
  "parking_facility_quality": 1,
  "parking_value_for_money": 1,
  "baggage_claim_process": 1,
  "checkin_process": 1,
  "checkin_queue_wait_time": 5,
  "security_screening_process": 1,
  "security_queue_wait_time": 5,
  "food_beverage_outlets": 1,
  "food_beverage_price_quality": 1,
  "retail_outlets": 1,
  "retail_price_quality": 1,
  "thermal_comfort": 1,
  "acoustic_comfort": 1,
  "airline_service": 1,
  "staff_courtesy": 1,
  "seat_availability": 1,
  "power_outlet_availability": 1,
  "airport_internet": 1,
  "signage": 1,
  "location_and_movement": 1
}
```

**Response:**
```json
{
  "prediction": 0,
  "label": "Not Satisfied",
  "probability": 0.0
}
```

**Interpretation:** The model is certain this passenger is not satisfied. All services rated 1/5, long queue times (5/5), and the profile of a business traveler traveling alone in January (typically the most critical demographic) all contribute to a strong dissatisfaction prediction.

---

### Example 3 — Borderline Passenger 

A middle-aged leisure traveler with mixed ratings.

**Input:**
```json
{
  "process": "Boarding",
  "month": "March",
  "connection": "Boarded at the airport",
  "ticket_purchased_by": "By the passenger",
  "age_group": "46 to 55 years",
  "trip_purpose": "Leisure",
  "traveling_alone": "Yes",
  "boarding_lounge_comfort": 3,
  "overall_airport_cleanliness": 3,
  "restrooms": 3,
  "restroom_cleanliness": 3,
  "restroom_maintenance": 3,
  "parking": 3,
  "parking_facility_quality": 3,
  "parking_value_for_money": 2,
  "baggage_claim_process": 3,
  "checkin_process": 3,
  "checkin_queue_wait_time": 3,
  "security_screening_process": 3,
  "security_queue_wait_time": 3,
  "food_beverage_outlets": 3,
  "food_beverage_price_quality": 2,
  "retail_outlets": 3,
  "retail_price_quality": 2,
  "thermal_comfort": 3,
  "acoustic_comfort": 3,
  "airline_service": 3,
  "staff_courtesy": 3,
  "seat_availability": 3,
  "power_outlet_availability": 3,
  "airport_internet": 3,
  "signage": 3,
  "location_and_movement": 3
}
```

**Response:**
```json
{
  "prediction": 0,
  "label": "Not Satisfied",
  "probability": 0.42
}
```

**Interpretation:** With all ratings at 3/5 (average) and price-related services rated lower (2/5), the model predicts slight dissatisfaction. This aligns with our EDA finding that price quality is the lowest-rated service overall. The probability of 42% indicates uncertainty — this passenger is close to the decision boundary.