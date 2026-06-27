# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
df = pd.read_csv("../data/credit_card_fraud_synthetic.csv")
# %%
X = df.drop(columns=["Class"])
y = df["Class"]
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=63,
    stratify=y,
)
# %%
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%
model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)
# %%
y_pred = model.predict(X_test)
# %%
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# The initial baseline predicted only the majority class due to severe class imbalance. I addressed this by using class-weighted training.
