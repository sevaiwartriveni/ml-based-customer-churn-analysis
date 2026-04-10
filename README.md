# Telco Customer Churn Prediction

A machine learning pipeline that predicts whether a telecom customer will churn (leave the service) or not, using the IBM Telco Customer Churn dataset.

---

## How It Works

### 1. Load & Clean Data
The dataset is loaded from a CSV file. The `customerID` column is dropped since it's just an identifier and has no predictive value. `TotalCharges` comes in as a string in the raw data (some entries are blank), so it's converted to numeric and any missing values are filled with the median.

### 2. Label Encoding
All text columns like `gender`, `Contract`, and `PaymentMethod` are converted to numbers using `LabelEncoder` because machine learning models only work with numerical input. A separate encoder is stored for each column so new customer data can be transformed the same way later.

### 3. Feature Engineering
New features are created from the existing ones to give the models more useful signals:

- **AvgCharges** — average charge per month of tenure
- **CLV** — monthly charges multiplied by tenure, a rough customer lifetime value
- **TotalServices** — how many add-on services the customer subscribes to
- **IsNewCustomer** — flags customers with 6 or fewer months of tenure
- **ChargeRatio** — monthly charges relative to total charges; a spike here can indicate recent price increases
- **TenureBin** — tenure grouped into 4 bins to capture non-linear patterns
- **HighCharge** — binary flag for customers above the 75th percentile in monthly charges
- **ServicePerCharge** — services used per unit of charge, a value-for-money indicator
- **LoyaltyScore** — tenure multiplied by contract type; long-term contract customers score higher

### 4. Exploratory Data Analysis
Three sets of plots are generated before any modelling:

- **Churn distribution, Contract vs Churn, Monthly Charges vs Churn** — shows how imbalanced the classes are, confirms that month-to-month customers churn more, and shows churned customers tend to pay more monthly.
- **Churn rate by 6 categorical features** — bar charts showing the actual proportion of churners in each category, making it easy to spot which groups are high risk.
- **Correlation heatmap** — shows how all features relate to each other and to the target variable. Only the lower triangle is shown to avoid duplicating information.

### 5. Train/Test Split and SMOTE
The data is split 80/20 into training and test sets with `stratify=y` so both sets have the same churn ratio. Features are scaled using `StandardScaler` so that large-valued columns don't dominate distance-based models.

SMOTE (Synthetic Minority Over-Sampling Technique) is then applied **only to the training set** to balance the classes by generating synthetic churn examples. Applying it to the test set would be data leakage.

### 6. Hyperparameter Tuning
`GridSearchCV` with 5-fold cross-validation is used to find the best settings for XGBoost and Random Forest. The scoring metric is F1 (not accuracy) because the dataset is imbalanced and accuracy alone would be misleading. Logistic Regression, Decision Tree, and Gradient Boosting are trained with fixed settings.

### 7. Voting Ensemble
A soft voting classifier combines Logistic Regression, Random Forest, XGBoost, and Gradient Boosting. Soft voting averages the predicted probabilities across models rather than taking a majority vote, which tends to give better results.

### 8. Model Evaluation
All six models are evaluated on the test set using accuracy, F1 score, and ROC AUC. A results table is printed and the best model by F1 is automatically selected. A confusion matrix heatmap is plotted for the best model.

### 9. ROC and Precision-Recall Curves
Both curves are plotted for all six models together. The ROC curve shows the trade-off between catching real churners and raising false alarms. The Precision-Recall curve is plotted alongside it because it gives a more honest picture when classes are imbalanced.

### 10. Learning Curves
Learning curves are plotted for XGBoost, Random Forest, and Logistic Regression to check for overfitting or underfitting. If the gap between training and validation F1 is large, the model is overfitting. If both scores are low, it is underfitting.

### 11. Feature Importance
The top 15 most important features are plotted as horizontal bar charts for XGBoost, Random Forest, and Gradient Boosting. Comparing across three models makes the results more trustworthy — if a feature ranks high in all three, it genuinely matters.

### 12. Threshold Tuning
The default classification threshold is 0.5, but this is not always optimal. The code tests every threshold between 0.3 and 0.7 and picks the one that gives the highest F1. A line plot shows F1 vs threshold with the optimal point marked. The model is then re-evaluated at this threshold and compared against the default.

### 13. Cross-Validation
Stratified K-Fold cross-validation with 5 splits is run for all six models on the full dataset. Results are reported as mean ± standard deviation, giving a more reliable estimate of real-world performance than a single train/test split.

### 14. Risk Segmentation
Customers are bucketed into High Risk, Medium Risk, and Low Risk groups based on their tenure. Customers under 12 months are High Risk, 12–24 months are Medium Risk, and above 24 months are Low Risk.

### 15. Predict New Customer
A reusable `predict_churn()` function accepts a dictionary of customer attributes, applies all the same feature engineering and scaling steps, and returns a prediction, churn probability, and risk level. This makes the model easy to use for inference on new data without rerunning the full pipeline.

### 16. Save the Model
The best model, scaler, and encoders are saved to disk using `joblib` so they can be loaded later for predictions without retraining from scratch.

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib
python churn_prediction.py
```

Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same folder before running.
