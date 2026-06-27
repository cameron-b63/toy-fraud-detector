# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("../data/credit_card_fraud_synthetic.csv")

# %%
df.head()

# %%
df.info()

# %%
df.describe()

# %%
df.columns

# %%
df.shape

# %%
df["Class"].value_counts(normalize=True)

# %%
df.isnull().sum()

# %%
plt.hist(df["V3"], bins=50)
plt.xlabel("V3")
plt.ylabel("Count")

# %%
corr = df.corr(numeric_only=True)
corr["Class"].sort_values()
# The dataset contains 1000 transactions with a severe class imbalance (1.8% fraud). There are no missing values, and the features are already numerical.
