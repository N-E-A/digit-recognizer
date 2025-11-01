import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

print("✅ Loading dataset...")
data = pd.read_csv('../data/train.csv')

# Separate features and labels
X = data.drop('label', axis=1).values / 255.0
y = data['label'].values

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

print("✅ Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_val, y_val)
print(f"🎯 Validation Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, 'digit_model.pkl')
print("✅ Model trained and saved as 'digit_model.pkl'")
