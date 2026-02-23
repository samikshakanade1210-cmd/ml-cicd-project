import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

print("Starting model training...")

#Create dummy dataset
X= np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

#Create model
model =LinearRegression()

#Train model
model.fit(X, y)

#Create folder to store model
os.makedirs("models", exist_ok=True)

#Save model
joblib.dump(model, "models/model.pkl")

print("Model trained and saved successfully!")