import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv(r"c:/Users/T Ganesh/Desktop/gsg/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Prepare features and target
X = df.drop(['Churn', 'customerID'], axis=1)
le_target = LabelEncoder()
y = le_target.fit_transform(df['Churn'])

# Define preprocessors
num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns
numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features),        
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_s = preprocessor.fit_transform(X_train)
X_test_s = preprocessor.transform(X_test)

# Best model: RandomForest with tuning (simplified from notebook)
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
random_search = RandomizedSearchCV(rf, param_grid, n_iter=5, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
random_search.fit(X_train_s, y_train)

# Best model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_s)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Best params: {random_search.best_params_}")

# Save model and preprocessor
import os
os.makedirs('models', exist_ok=True)
with open('models/churn_prediction_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le_target, f)
print("Models saved to models/")
