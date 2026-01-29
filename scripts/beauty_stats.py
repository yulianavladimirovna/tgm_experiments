import pandas as pd

PATH = "data/amazon/beauty_5core_tgm.csv"

# 1) load (only needed columns)
df = pd.read_csv(
    PATH,
    usecols=["from", "to", "timestamp", "value"],
    dtype={"from": "string", "to": "string"},
)

# 2) compute stats (assumes: from = users, to = items)
n_interactions = len(df)
n_users = df["from"].nunique()
n_items = df["to"].nunique()
avg_len = df.groupby("from").size().mean()  # avg interactions per user

computed = {
    "Dataset": "Beauty",
    "#Items": n_items,
    "#Users": n_users,
    "#Interactions": n_interactions,
    "Avg. length": avg_len,
}

reference = {
    "Dataset": "Beauty (ref)",
    "#Items": 163_098,
    "#Users": 146_845,
    "#Interactions": 783_047,
    "Avg. length": 5.332,
}

out = pd.DataFrame([computed, reference])

out_fmt = out.copy()
for c in ["#Items", "#Users", "#Interactions"]:
    out_fmt[c] = out_fmt[c].map(lambda x: f"{int(x):,}")
out_fmt["Avg. length"] = out_fmt["Avg. length"].map(lambda x: f"{float(x):.3f}")

print(out_fmt.to_string(index=False))
