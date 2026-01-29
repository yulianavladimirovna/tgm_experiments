import gzip
import json
import pandas as pd

inp = "data/amazon/reviews_Beauty_5.json.gz"
out = "data/amazon/beauty_5core_tgm.csv"

rows = []
with gzip.open(inp, "rt", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        user = obj.get("reviewerID")
        item = obj.get("asin")
        rating = obj.get("overall")
        ts = obj.get("unixReviewTime")

        if user is None or item is None or rating is None or ts is None:
            continue

        rows.append({
            "from": f"u_{user}",
            "to": f"i_{item}",
            "timestamp": int(ts),
            "value": float(rating),
        })

df = pd.DataFrame(rows).sort_values("timestamp")
df.to_csv(out, index=False)
print("Saved:", out, "rows:", len(df))