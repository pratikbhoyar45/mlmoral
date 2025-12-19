import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("diabetes_powerbi_30000_plus.csv")

print("Columns in CSV:")
print(df.columns)

target = "Diabetes_binary"

X = df.drop(target, axis=1)
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler, X.columns.tolist()), f)

print("✅ Model trained successfully")
print("✅ model.pkl created")
