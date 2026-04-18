import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("Loading dataset...")
df = pd.read_csv('Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv')

# Drop any nulls in target
df = df.dropna(subset=['addiction_level'])

print("Encoding target variable...")
le = LabelEncoder()
df['addiction_level_encoded'] = le.fit_transform(df['addiction_level'])

X = df[['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours']]
y = df['addiction_level_encoded']

print("Training RandomForest model...")
rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1)
rf.fit(X, y)

print("Saving model and encoder...")
with open('addiction_model.pkl', 'wb') as f:
    pickle.dump({'model': rf, 'encoder': le}, f)

print("Training finished! Model saved to addiction_model.pkl")
