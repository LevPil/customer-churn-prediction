
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)
import warnings
warnings.filterwarnings('ignore')

print("_" * 60)
print("1.Загрузка и анализ")
print("_" * 60)

df = pd.read_csv("Telco-Customer-Churn.csv")
print(f"Размер датасета: {df.shape}")
print(f"\nТипы данных:\n{df.dtypes}")
print(f"\nПропуски:\n{df.isnull().sum()}")


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"\nПропуски после конвертации TotalCharges: {df['TotalCharges'].isnull().sum()}")


print(f"\nРаспределение Churn:\n{df['Churn'].value_counts(normalize=True)}")


print("\n" + "_" * 60)
print("2.Краткий EDA")
print("_" * 60)


print("\nChurn rate по типу контракта:")
print(df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean()))

print("\n" + "_" * 60)
print("3.Препроцессинг")
print("_" * 60)

df = df.drop(columns=['customerID'])


y = (df['Churn'] == 'Yes').astype(int)
X = df.drop(columns=['Churn'])

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Числовые признаки: {numeric_features}")
print(f"Категориальные: {categorical_features}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

print("\n" + "_" * 60)
print("4.Обучение моделей")
print("_" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'ROC-AUC': auc
    }
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

print("\n" + "_" * 60)
print("5.СРАВНЕНИЕ МОДЕЛЕЙ")
print("_" * 60)
results_df = pd.DataFrame(results).T
print(results_df.round(4))

print("\n" + "_" * 60)
print("6. Важность признаков (Random Forest)")
print("_" * 60)

rf_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1))
])
rf_pipe.fit(X_train, y_train)

ohe = rf_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = ohe.get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_feature_names)

importances = rf_pipe.named_steps['classifier'].feature_importances_
feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feat_imp.head(15).to_string(index=False))


import joblib
joblib.dump(rf_pipe, 'churn_model.joblib')
print("\nМодель сохранена в churn_model.joblib")

print("\n" + "_" * 60)

