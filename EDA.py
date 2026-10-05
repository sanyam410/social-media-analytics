#Part 4: pull MongoDB → pandas → EDA + charts
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient

sns.set_style("whitegrid")
os.makedirs("reports", exist_ok=True)
db = MongoClient("mongodb://localhost:27017/")["social_media_analytics"]

# ---------- extract: posts enriched with author info (aggregation → DataFrame) ----------
posts = pd.DataFrame(db.posts.aggregate([
    {"$lookup": {"from": "users", "localField": "user_id", "foreignField": "_id", "as": "u"}},
    {"$unwind": "$u"},
    {"$project": {"_id": 0, "post_id": 1, "content_type": 1, "category": 1, "hashtags": 1,
                  "posted_at": 1, "views": 1, "shares": 1, "likes_count": 1,
                  "comments_count": 1, "engagement_rate": 1, "viral": 1,
                  "account_type": "$u.account_type", "followers": "$u.followers_count"}},
]))
posts["posted_at"] = pd.to_datetime(posts["posted_at"], utc=True)
posts["n_hashtags"] = posts["hashtags"].str.len().fillna(0)
posts = posts.drop(columns="hashtags")

comments = pd.DataFrame(db.comments.find({}, {"_id": 0, "post_id": 1, "sentiment": 1}))
print(posts.info(), "\n")
print(posts[["views", "likes_count", "comments_count", "shares", "engagement_rate"]].describe().round(2))

# ---------- 1. views distribution (heavily right-skewed) ----------
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(posts["views"], bins=60, log=True, color="#4c72b0")
ax.set_title("Views per post (log y) — long tail, viral posts are the outliers")
plt.tight_layout(); plt.savefig("reports/01_views_hist.png", dpi=120); plt.show()

# ---------- 2. content type: median views (means get hijacked by viral outliers) ----------
order = posts.groupby("content_type")["views"].median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=posts, x="content_type", y="views", order=order, showfliers=False, ax=ax)
ax.set_title("Median views by content type — expect reel > video > carousel > image > text")
plt.tight_layout(); plt.savefig("reports/02_content_box.png", dpi=120); plt.show()

# ---------- 3. posting hour ----------
fig, ax = plt.subplots(figsize=(9, 4))
posts.groupby(posts["posted_at"].dt.hour)["views"].median().plot(kind="bar", ax=ax, color="#dd8452")
ax.set_title("Median views by posting hour (UTC) — evening elevator face")
plt.tight_layout(); plt.savefig("reports/03_hour.png", dpi=120); plt.show()

# ---------- 4. monthly activity trend ----------
monthly = posts.set_index("posted_at").resample("MS").agg(
    n=("post_id", "count"), avg_views=("views", "mean"))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(monthly.index, monthly["n"], label="posts/month")
ax2 = ax.twinx()
ax2.plot(monthly.index, monthly["avg_views"], color="#c44e52", label="avg views")
ax.set_title("Monthly posts vs avg views"); fig.legend(loc="upper left")
plt.tight_layout(); plt.savefig("reports/04_monthly.png", dpi=120); plt.show()

# ---------- 5. followers vs views (proxy = final follower count) ----------
p = posts.copy(); p["followers_p1"] = p["followers"] + 1
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=p, x="followers_p1", y="views", hue="viral", alpha=0.5, ax=ax)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_title("Followers vs views (log-log) — viral posts break the trend line")
plt.tight_layout(); plt.savefig("reports/05_scatter.png", dpi=120); plt.show()
print("\nviews↔likes corr (expected near-perfect — likes follow views):",
      round(float(posts["views"].corr(posts["likes_count"])), 3))

# ---------- 6. correlation heatmap ----------
cols = ["views", "likes_count", "comments_count", "shares", "n_hashtags",
        "followers", "engagement_rate"]
correlation_matrix = posts[cols].corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlations — hashtags & ER should show ~0 vs views")
plt.tight_layout(); plt.savefig("reports/06_heatmap.png", dpi=120); plt.show()

# ---------- 7. replicate pipeline 6 in pandas (validation) ----------
c = comments.merge(posts[["post_id", "engagement_rate"]], on="post_id")
c["bucket"] = np.select(
    [c["engagement_rate"] <= 0.03, c["engagement_rate"] >= 0.075],
    ["cold", "hot"], default="warm")
res = c.groupby("bucket")["sentiment"].apply(lambda s: (s == "positive").mean() * 100).round(1)
print("\n% positive by bucket (MongoDB said: cold 45.7 / warm 59.2 / hot 63.7):")
print(res)
