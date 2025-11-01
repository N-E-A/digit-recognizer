import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, 'digit_model.pkl')
TRAIN_PATH = os.path.join(os.path.dirname(ROOT), 'data', 'train.csv')

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading sample from training data...")
data = pd.read_csv(TRAIN_PATH).sample(1000, random_state=42)
X = data.drop('label', axis=1).values / 255.0
y = data['label'].values

print("Predicting...")
y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
print(f"✅ Accuracy on 1000-sample training subset: {acc * 100:.2f}%")
