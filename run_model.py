import sys
sys.stdout = open("results.txt", "w", encoding="utf-8")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

print("Loading dataset...")
df = pd.read_csv('Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv')

print("\n--- Data Information ---")
print("Shape:", df.shape)
print("Null values:\n", df.isnull().sum())
print("\nSleep hours distribution:\n", df['sleep_hours'].value_counts())

le = LabelEncoder()
df['addiction_level'] = le.fit_transform(df['addiction_level'])

X = df[['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours']]
y = df['addiction_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Training RandomForestClassifier ---")
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 7]
}

random_search = RandomizedSearchCV(estimator=rf,
                                   param_distributions=param_dist,
                                   n_iter=10,
                                   cv=5,
                                   scoring='accuracy',
                                   n_jobs=-1,
                                   random_state=42)

random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)
