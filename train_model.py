import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 1. Load dataset
df = pd.read_csv("data/sme_loan_dataset.csv")

# ✅ 2. One-hot encode the 'sector' categorical variable
df = pd.get_dummies(df, columns=["sector"], drop_first=True)

# ✅ 3. Define features (X) and target (y)
X = df.drop("loan_approved", axis=1)
y = df["loan_approved"]

# ✅ 4. Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 5. Train RandomForest model
model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ✅ 6. Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ 7. Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- Top Features Influencing Loan Approval ---")
print(importances.head(10))

# ✅ Save feature importance plot
plt.figure(figsize=(8, 6))
sns.barplot(x=importances.head(10), y=importances.head(10).index)
plt.title("Top 10 Important Features for Loan Approval")
plt.xlabel("Feature Importance Score")
plt.tight_layout()
plt.savefig("data/feature_importance.png")
plt.show()

# ✅ 8. Save the trained model
joblib.dump(model, "model/sme_loan_model.pkl")
print("\n✅ Model saved as model/sme_loan_model.pkl")
