# Credit Card Fraud Detection Model

This project develops an end-to-end machine learning pipeline to detect fraudulent credit card transactions. Because fraudulent transactions make up less than 1 percent of the dataset, the project focuses on handling class imbalance and maximizing recall using techniques such as SMOTE, model comparison, and threshold tuning.

## Project Description
The dataset contains anonymized credit card transaction data with severe class imbalance. This project enhances a basic Logistic Regression model by introducing SMOTE oversampling, feature scaling, Random Forest, and XGBoost models. It evaluates multiple algorithms and selects the model that best detects rare fraud cases while maintaining low false negatives.

## Key Steps
1. **Data Loading and Exploration**  
   - Loaded `creditcard.csv` into a DataFrame  
   - Reviewed class distribution, summary statistics, and missing values  

2. **Data Preprocessing**  
   - Verified dataset integrity  
   - Separated features (X) and target (Y)  
   - Applied SMOTE on the training set  
   - Standardized numerical features  

3. **Feature and Target Separation**  
   - X includes Time, Amount, and PCA-transformed V1–V28  
   - Y is the fraud class label (0 = legitimate, 1 = fraud)

4. **Train–Test Split**  
   - 80–20 split with stratification to preserve class ratio  

5. **Model Training**  
   - Logistic Regression  
   - Random Forest Classifier  
   - XGBoost Classifier (best performer)  

6. **Model Evaluation**  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  
   - ROC-AUC  
   - PR-AUC  
   - Confusion matrix  
   - Threshold tuning  

## Results
### Baseline Model
- Logistic Regression  
- Test Accuracy: 92.89 percent  
- Low recall, many fraud cases missed  

### Improved Final Model
Using SMOTE + StandardScaler + XGBoost/RandomForest:
- Test Accuracy: **99.94 percent**  
- Strong recall and F1-score  
- Higher ROC-AUC and PR-AUC  
- Significantly more reliable for fraud detection  

## Dataset
Source: Kaggle – Credit Card Fraud Detection  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

Download manually and place `creditcard.csv` in the project folder.

## Dependencies
Install required libraries:
```
pip install numpy pandas scikit-learn imbalanced-learn xgboost matplotlib seaborn joblib
```
## Project Structure

CreditCard_FraudDetection/
│
├── credit_card_fraud_detection.ipynb # Main notebook
├── creditcard.csv # Dataset (download separately, not stored in repo)
└── README.md # Project documentation

## Future Improvements
- Hyperparameter tuning  
- LightGBM/CatBoost models  
- Cost-sensitive learning  
- SHAP-based model explainability  
- Real-time fraud detection API  
- Deployment with Flask/FastAPI  


