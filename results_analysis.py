import pandas as pd

# Adjustable weights
alpha = 0.50
beta = 0.30
gamma = 0.20

df = pd.read_csv("evaluation_results.csv")

# Compute unified evaluation metric (UEM)
df["UEM"] = (
    alpha * df["relevance"] +
    beta * df["consistency"] +
    gamma * df["completeness"]
)

# Average UEM per method
method_scores = df.groupby("method")["UEM"].mean().sort_values(ascending=False)

print(method_scores)
