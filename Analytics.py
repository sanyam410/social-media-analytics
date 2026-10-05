#Part 3: MongoDB aggregation pipelines
import time
from pymongo import MongoClient

db = MongoClient("mongodb://localhost:27017/")["social_media_analytics"]

def run(label, pipeline, coll="posts"):
    t = time.perf_counter()
    out = list(db[coll].aggregate(pipeline))
    ms = (time.perf_counter() - t) * 1000
    print(f"\n{'='*60}\n{label}  ({ms:.0f} ms)\n{'='*60}")
    for row in out:
        print(row)
    return out

# 1) Which content type performs best?
run("1. Performance by content type", [
    {"$group": {"_id": "$content_type", "n": {"$sum": 1},
                "avg_views": {"$avg": "$views"},
                "avg_engagement_rate": {"$avg": "$engagement_rate"}}},
    {"$project": {"_id": 1, "n": 1,
                  "avg_views": {"$round": ["$avg_views", 0]},
                  "avg_engagement_rate": {"$round": ["$avg_engagement_rate", 4]}}},
    {"$sort": {"avg_views": -1}},
])

# 2) When should you post? (hour-of-day)
run("2. Views by posting hour", [
    {"$group": {"_id": {"$hour": "$posted_at"}, "n": {"$sum": 1},
                "avg_views": {"$avg": "$views"}}},
    {"$project": {"hour": "$_id", "n": 1, "avg_views": {"$round": ["$avg_views", 0]}}},
    {"$sort": {"hour": 1}},
])

# 3) Hashtag reach (note: tags were randomly assigned — we're testing, not assuming)
run("3. Top 10 hashtags by associated views", [
    {"$unwind": "$hashtags"},
    {"$group": {"_id": "$hashtags", "posts": {"$sum": 1},
                "total_views": {"$sum": "$views"}}},
    {"$sort": {"total_views": -1}}, {"$limit": 10},
])

# 4) Viral vs normal: reach ≠ quality
run("4. Viral vs non-viral", [
    {"$group": {"_id": "$viral", "n": {"$sum": 1},
                "avg_views": {"$avg": "$views"},
                "avg_engagement_rate": {"$avg": "$engagement_rate"}}},
    {"$project": {"viral": "$_id", "n": 1,
                  "avg_views": {"$round": ["$avg_views", 0]},
                  "avg_engagement_rate": {"$round": ["$avg_engagement_rate", 4]}}},
])

# 5) Creator leaderboard (joins posts → users)
run("5. Top 10 creators by total engagement", [
    {"$group": {"_id": "$user_id",
                "total_eng": {"$sum": {"$add": ["$likes_count", "$comments_count", "$shares"]}},
                "posts": {"$sum": 1}}},
    {"$sort": {"total_eng": -1}}, {"$limit": 10},
    {"$lookup": {"from": "users", "localField": "_id", "foreignField": "_id", "as": "u"}},
    {"$unwind": "$u"},
    {"$project": {"_id": 0, "username": "$u.username", "type": "$u.account_type",
                  "followers": "$u.followers_count", "posts": 1, "total_eng": 1,
                  "eng_per_post": {"$round": [{"$divide": ["$total_eng", "$posts"]}, 1]}}},
])

# 6) Does sentiment track reception? (comments → posts join)
run("6. % positive comments by post reception", [
    {"$lookup": {"from": "posts", "localField": "post_id", "foreignField": "_id", "as": "p"}},
    {"$unwind": "$p"},
    {"$project": {
        "bucket": {"$cond": [{"$gte": ["$p.engagement_rate", 0.075]}, "hot",
                   {"$cond": [{"$lte": ["$p.engagement_rate", 0.03]}, "cold", "warm"]}]},
        "is_pos": {"$cond": [{"$eq": ["$sentiment", "positive"]}, 1, 0]}}},
    {"$group": {"_id": "$bucket", "comments": {"$sum": 1}, "pos_share": {"$avg": "$is_pos"}}},
    {"$project": {"_id": 0, "bucket": "$_id", "comments": 1,
                  "pos_pct": {"$round": [{"$multiply": ["$pos_share", 100]}, 1]}}},
    {"$sort": {"bucket": 1}},
],coll="comments")

# 7) $facet dashboard — same query, pretty-printed per section
facet_pipe = [
    {"$facet": {
        "kpis": [{"$group": {"_id": None, "posts": {"$sum": 1},
                             "total_views": {"$sum": "$views"},
                             "avg_er": {"$avg": "$engagement_rate"}}}],
        "by_content_type": [{"$group": {"_id": "$content_type", "n": {"$sum": 1},
                                        "avg_views": {"$avg": "$views"}}},
                            {"$sort": {"avg_views": -1}},
                            {"$project": {"_id": 1, "n": 1, "avg_views": {"$round": ["$avg_views", 0]}}}],
        "top_posts": [{"$sort": {"views": -1}}, {"$limit": 5},
                      {"$project": {"_id": 0, "post_id": 1, "content_type": 1, "views": 1, "viral": 1}}],
        "top_categories": [{"$group": {"_id": "$category", "avg_er": {"$avg": "$engagement_rate"}}},
                           {"$sort": {"avg_er": -1}}, {"$limit": 5}],
    }}
]

t = time.perf_counter()
facet_out = list(db.posts.aggregate(facet_pipe))[0]   # $facet → ONE document
ms = (time.perf_counter() - t) * 1000
print(f"\n{'='*60}\n7. $facet dashboard (one round-trip)  ({ms:.0f} ms)\n{'='*60}")

for section, rows in facet_out.items():               # print each section separately
    print(f"\n▸ {section}")
    for r in rows:
        print("  ", r)

# 8) % positive comments by author country × post category
#    Path: comments → posts (author + category) → users (author's country)
run("8. Sentiment by author country × category (homework)", [
    {"$lookup": {"from": "posts", "localField": "post_id", "foreignField": "_id", "as": "p"}},
    {"$unwind": "$p"},
    {"$project": {"author_id": "$p.user_id", "category": "$p.category",
                  "is_pos": {"$cond": [{"$eq": ["$sentiment", "positive"]}, 1, 0]}}},
    {"$lookup": {"from": "users", "localField": "author_id", "foreignField": "_id", "as": "u"}},
    {"$unwind": "$u"},
    {"$group": {"_id": {"country": "$u.country", "category": "$category"},
                "comments": {"$sum": 1}, "pos_share": {"$avg": "$is_pos"}}},
    {"$match": {"comments": {"$gte": 150}}},                    # drop tiny noisy cells
    {"$project": {"_id": 0, "country": "$_id.country", "category": "$_id.category",
                  "comments": 1, "pos_pct": {"$round": [{"$multiply": ["$pos_share", 100]}, 1]}}},
    {"$sort": {"pos_pct": -1}}, {"$limit": 10},
], coll="comments")                        # ← don't forget this (pipeline 6's lesson!)

