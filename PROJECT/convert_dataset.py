import pandas as pd

labels = {}
tweets = {}

# Read labels
with open("label.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(":")
        label_text = parts[0]
        tweet_id = parts[1]

        if label_text == "non-rumor":
            label = 0
        else:
            label = 1

        labels[tweet_id] = label


# Read tweets
with open("source_tweets.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")

        if len(parts) < 2:
            continue

        tweet_id = parts[0]
        text = parts[1]

        tweets[tweet_id] = text


rows = []

for tweet_id in tweets:

    rows.append({
        "thread_id": tweet_id,
        "tweet_id": tweet_id,
        "parent_id": tweet_id,
        "text": tweets[tweet_id],
        "label": labels.get(tweet_id, 0)
    })


df = pd.DataFrame(rows)

import os
os.makedirs("dataset", exist_ok=True)

df.to_csv("dataset/rumours.csv", index=False)

print("Dataset created: dataset/rumours.csv")
print("Total samples:", len(df))

import sys
if "--pheme" in sys.argv:
    from data_loader import download_pheme_if_needed
    import os
    # force re-download by removing old CSV if it exists
    if os.path.exists("dataset/rumours.csv"):
        os.remove("dataset/rumours.csv")
        print("Removed old dataset/rumours.csv")
    download_pheme_if_needed()
    print("Done. Run python train.py to train on PHEME-9.")
