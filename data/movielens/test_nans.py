import pandas as pd

df = pd.read_csv("ml-100k_ratings_tgm.csv")
print(df["value"].describe())
print(df.shape)