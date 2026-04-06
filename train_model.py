import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

#df = pd.read_csv("Bengaluru_House_Data.csv")
df = pd.read_csv(r"C:\Users\luvku\Downloads\archive (9)\Bengaluru_House_Data.csv")

df = df[["location", "total_sqft", "bath", "balcony", "price"]]

def convert_sqft(value):
    try:
        return float(str(value).split("-")[0])
    except:
        return np.nan

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

df = df.dropna(subset=["price"])

X = df[["location", "total_sqft", "bath", "balcony"]]
y = df["price"]

numerical_features = ["total_sqft", "bath", "balcony"]
categorical_features = ["location"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numerical_features),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X, y)

with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully")