import pandas as pd
import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Bengaluru_House_Data.csv")

# Keep required columns
df = df[["location", "total_sqft", "bath", "balcony", "price"]]

# Convert total_sqft to numeric
def convert_sqft(value):
    try:
        return float(str(value).split("-")[0])
    except:
        return np.nan

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

# Remove rows with missing values
df.dropna(subset=["price"], inplace=True)

# Features and target
X = df[["location", "total_sqft", "bath", "balcony"]]
y = df["price"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            ["total_sqft", "bath", "balcony"]
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            ["location"]
        )
    ]
)

# Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train
model.fit(X, y)

# Save model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
