# Importing necessary libraries
from pymongo import MongoClient
from faker import Faker
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta

# reproducible dataset
random.seed(42)  
Faker.seed(42)
fake = Faker()

# Connecting to MongoDB
client =MongoClient("mongodb://localhost:27017/")

db = client["social_media_analytics"]

print("Connected to MongoDB!")
print("Database:", db.name)

# # ---------- config ----------
N_USERS = 1_000
N_FOLLOWS = 20_000
START = datetime(2024, 10, 1)
END = datetime(2026, 10, 1)
JOIN_WINDOW_DAYS = 548   # everyone joins in the first ~18 months

# (account_type, share of users, popularity weight = chance of being followed)
ACCOUNT_TYPES = [
    ("regular",    0.60,  1),
    ("creator",    0.25, 12),
    ("business",   0.10,  6),
    ("influencer", 0.05, 60),
]
pop_by_type = {t: w for t, _, w in ACCOUNT_TYPES}

COUNTRIES = [
    ("India", 0.22), ("USA", 0.18), ("UK", 0.08), ("Brazil", 0.07),
    ("Germany", 0.06), ("Nigeria", 0.05), ("Indonesia", 0.05),
    ("Philippines", 0.04), ("Canada", 0.04), ("Australia", 0.03),
    ("France", 0.03), ("Japan", 0.03), ("Mexico", 0.03), ("Spain", 0.03),
    ("Other", 0.10),
]

def weighted_choice(pairs):
    items, weights = zip(*pairs)
    return random.choices(items, weights=weights, k=1)[0]

# ---------- fresh start (safe to re-run) ----------
for name in ["users", "posts", "comments", "likes", "followers"]:
    db[name].drop()

users, usernames = [], set()
for i in range(1, N_USERS + 1):
    username = fake.user_name()
    while username in usernames:
        username = fake.user_name()
    usernames.add(username)

    users.append({
        "_id": i,                      # int business key — readable & consistent for $lookup
        "user_id": i,
        "username": username,
        "age": int(random.triangular(16, 60, 24)),   # skews young, like real platforms
        "gender": random.choice(["male", "female", "other"]),
        "country": weighted_choice(COUNTRIES),
        "account_type": weighted_choice([(name, share) for name, share, _ in ACCOUNT_TYPES]),
        "joined_at": fake.date_time_between(START, START + timedelta(days=JOIN_WINDOW_DAYS)),
    })

# ---------- 2. follower network (weighted by popularity) ----------
join_by_id = {u["user_id"]: u["joined_at"] for u in users}
followee_weights = [pop_by_type[u["account_type"]] for u in users]

seen, edges = set(), []
while len(edges) < N_FOLLOWS:
    f = random.randrange(1, N_USERS + 1)              # follower: uniform
    g = random.choices(range(1, N_USERS + 1), weights=followee_weights, k=1)[0]
    if f == g or (f, g) in seen:                      # no self-follows, no dupes
        continue
    earliest = max(join_by_id[f], join_by_id[g])      # edge can't predate either join
    seen.add((f, g))
    edges.append({
        "follower_id": f,
        "following_id": g,
        "followed_at": fake.date_time_between(earliest, END),
    })

# ---------- 3. derive counts FROM the network (never random) ----------
followers_count, following_count = defaultdict(int), defaultdict(int)
for e in edges:
    followers_count[e["following_id"]] += 1
    following_count[e["follower_id"]] += 1

for u in users:
    u["followers_count"] = followers_count[u["user_id"]]
    u["following_count"] = following_count[u["user_id"]]

# ---------- 4. insert + protect integrity with indexes ----------
db.users.insert_many(users)
db.followers.insert_many(edges)

db.users.create_index("username", unique=True)
db.followers.create_index([("following_id", 1), ("follower_id", 1)], unique=True)
db.followers.create_index("follower_id")

# ---------- 5. verify ----------
print("users:", db.users.count_documents({}))
print("followers:", db.followers.count_documents({}))
print("by account type:", dict(Counter(u["account_type"] for u in users)))
print("\nTop 5 most-followed:")
for u in sorted(users, key=lambda x: -x["followers_count"])[:5]:
    print(f'  @{u["username"]:<20} {u["followers_count"]:4d} followers  ({u["account_type"]})')


# ============================================================
# PART 2 — posts + engagement engine (views → likes → comments)
# ============================================================
import bisect
import math
from collections import Counter

N_POSTS, N_LIKES, N_COMMENTS = 10_000, 100_000, 30_000
POST_END = datetime(2026, 10, 1) - timedelta(days=14)   # buffer so late posts get interactions
SNAPSHOT = datetime(2026, 10, 1)

POST_RATES = {"regular": 6, "creator": 15, "business": 8, "influencer": 30}  # posts/year

CONTENT = [  # (type, share, reach multiplier)
    ("image", 0.30, 0.9), ("video", 0.20, 1.2), ("reel", 0.25, 1.5),
    ("carousel", 0.12, 1.0), ("text", 0.13, 0.6),
]
CT_WEIGHTS = [c[1] for c in CONTENT]
CT_MULT    = {c[0]: c[2] for c in CONTENT}

CATEGORIES = ["tech", "travel", "food", "fitness", "fashion", "gaming",
              "education", "music", "lifestyle", "news"]
HASHTAGS = ["python", "datascience", "mongodb", "ai", "travel", "foodie", "fitness",
            "fashion", "gaming", "music", "viral", "trending", "tips", "daily",
            "photooftheday", "reels", "tech", "lifestyle"]
HOUR_W = [4, 3, 2, 2, 2, 3, 4, 5, 6, 6, 6, 7, 8, 7, 7, 7, 8, 9, 10, 11, 12, 11, 9, 6]

CAPTIONS = ["Loving this vibe ✨", "Big things coming 🚀", "My latest work!", "Sunday reset ☀️",
            "Behind the scenes 🎬", "New post, new energy 💫", "Grateful for today 🙏", "Weekend mood 🎉",
            "Just posted", "Update from my day", "Some thoughts...", "A little throwback",
            "Midweek check-in", "Rough day...", "Disappointed honestly", "Not my best moment"]
COMMENTS = {
    "positive": ["Great post! 🔥", "Love this ❤️", "So useful, thanks!", "Amazing 🔥",
                 "This made my day!", "Well said 👏", "Inspiring ✨", "Exactly what I needed"],
    "neutral":  ["Interesting", "Nice", "Thanks for sharing", "Hmm okay", "Noted", "Cool"],
    "negative": ["Not convinced", "Could be better", "Dislike this", "Meh", "Why though?", "Overhyped"],
}

# ---------- normalize user info (adjust names if your Part 1 differs) ----------
UID    = lambda u: u.get("user_id", u["_id"])
UJOIN  = {UID(u): u["joined_at"] for u in users}
UTYPE  = {UID(u): u["account_type"] for u in users}
ALL_IDS = [UID(u) for u in users]
ids_by_join = sorted(ALL_IDS, key=lambda i: UJOIN[i])
join_times  = [UJOIN[i] for i in ids_by_join]

# ---------- 1. exactly N_POSTS, weighted by activity × window length ----------
post_weights = [max(POST_RATES[UTYPE[i]], 1) * max((POST_END - UJOIN[i]).days, 1) for i in ALL_IDS]
owners = random.choices(ALL_IDS, weights=post_weights, k=N_POSTS)

def draw_time(start, end):
    """Hour-of-day + weekend weighted timestamp (acceptance sampling)."""
    span = (end - start).total_seconds()
    while True:
        t = start + timedelta(seconds=random.uniform(0, span))
        w = HOUR_W[t.hour] * (1.15 if t.weekday() >= 5 else 1.0)
        if random.random() * 14 < w:
            return t

draft = sorted((draw_time(UJOIN[o], POST_END), o) for o in owners)

# ---------- 2. follower sweep: network AS OF each post ----------
timeline = sorted(edges, key=lambda e: e["followed_at"])
followers_of = {i: set() for i in ALL_IDS}
ei = 0

posts, viral_count, total_views = [], 0, 0
for n, (posted_at, owner) in enumerate(draft, start=1):
    while ei < len(timeline) and timeline[ei]["followed_at"] <= posted_at:
        e = timeline[ei]
        followers_of[e["following_id"]].add(e["follower_id"])
        ei += 1

    fcount = len(followers_of[owner])
    ctype  = random.choices([c[0] for c in CONTENT], weights=CT_WEIGHTS)[0]
    viral  = random.random() < 0.015
    reach  = random.lognormvariate(math.log(4), 0.7)          # median 4× followers, σ 0.7
    views  = (random.uniform(80, 180) + fcount * reach) * CT_MULT[ctype] \
             * (HOUR_W[posted_at.hour] / 7) \
             * (random.uniform(5, 15) if viral else 1)
    views  = max(int(views), 20)
    total_views += views
    viral_count += viral

    n_elig = bisect.bisect_right(join_times, posted_at)       # users joined by post time
    posts.append({
        "_id": n, "post_id": n, "user_id": owner,
        "content_type": ctype, "category": random.choice(CATEGORIES),
        "caption": random.choice(CAPTIONS),
        "hashtags": random.sample(HASHTAGS, random.randint(1, 5)),
        "posted_at": posted_at, "viral": viral, "views": views,
        "_elig": n_elig,                                       # temp: actor-pool size
    })

# ---------- 3. exact-total allocation (largest remainder + capped redistribution) ----------
def allocate(raw, target, caps):
    if sum(caps) < target:
        raise RuntimeError("total capacity < target — relax caps")
    scale = target / sum(raw)
    alloc = [min(int(r * scale), c) for r, c in zip(raw, caps)]
    rem = target - sum(alloc)
    order = sorted(range(len(raw)), key=lambda i: (raw[i] * scale) % 1, reverse=True)
    while rem > 0:                                             # soak up overflow
        moved = False
        for i in order:
            if rem == 0:
                break
            if alloc[i] < caps[i]:
                take = min(rem, caps[i] - alloc[i])
                alloc[i] += take
                rem -= take
                moved = True
        if not moved:
            raise RuntimeError("redistribution stalled")
    return alloc

like_raw, like_caps, comment_raw, comment_caps = [], [], [], []
for p in posts:
    lr = min(max(random.lognormvariate(math.log(0.05), 0.5), 0.005), 0.15)   # ~5% of views
    cr = min(max(random.lognormvariate(math.log(0.015), 0.6), 0.001), 0.06)  # ~1.5% of views
    like_raw.append(p["views"] * lr)
    comment_raw.append(p["views"] * cr)
    like_caps.append(min(int(0.25 * p["views"]) + 5, p["_elig"] - 1))        # ≤ pool size
    comment_caps.append(min(int(0.08 * p["views"]) + 3, p["_elig"] - 1))

alloc_likes    = allocate(like_raw, N_LIKES, like_caps)
alloc_comments = allocate(comment_raw, N_COMMENTS, comment_caps)

# ---------- 4. build interaction docs (actor pools + time decay + sentiment) ----------
def pick_sentiment(er, controversial):
    if controversial:
        w = (0.30, 0.30, 0.40)
    elif er >= 0.06:
        w = (0.76, 0.18, 0.06)
    elif er <= 0.02:
        w = (0.45, 0.40, 0.15)
    else:
        w = (0.62, 0.28, 0.10)
    return random.choices(("positive", "neutral", "negative"), weights=w)[0]

like_docs, comment_docs = [], []
for idx, p in enumerate(posts):
    author, t0 = p["user_id"], p["posted_at"]
    pool = [u for u in ids_by_join[:p["_elig"]] if u != author]
    fpool = list(followers_of[author])
    random.shuffle(fpool)

    # ----- likes: ~70% followers-as-of-post, rest random; unique (user, post) -----
    k = alloc_likes[idx]
    chosen = set()
    n_f = min(int(0.7 * k), len(fpool))
    for uid in fpool[:n_f]:
        chosen.add(uid)
        like_docs.append({"user_id": uid, "post_id": p["_id"],
                          "liked_at": min(t0 + timedelta(hours=random.expovariate(1 / 8)), SNAPSHOT)})
    rest = k - len(chosen)
    if rest > 0:
        cand = [u for u in pool if u not in chosen]
        for uid in random.sample(cand, min(rest, len(cand))):
            like_docs.append({"user_id": uid, "post_id": p["_id"],
                              "liked_at": min(t0 + timedelta(hours=random.expovariate(1 / 8)), SNAPSHOT)})

    # ----- comments: repeats allowed; sentiment tied to reception -----
    m = alloc_comments[idx]
    er = k / max(p["views"], 1)
    controversial = p["viral"] and random.random() < 0.10
    for _ in range(m):
        sent = pick_sentiment(er, controversial)
        at = t0 + timedelta(days=abs(random.gauss(1.5, 1.2)))
        comment_docs.append({"user_id": random.choice(pool), "post_id": p["_id"],
                             "text": random.choice(COMMENTS[sent]), "sentiment": sent,
                             "created_at": min(at, SNAPSHOT)})

    # ----- denormalized counts, derived not random -----
    p["shares"] = int(p["views"] * min(max(random.lognormvariate(math.log(0.005), 0.5), 0.0005), 0.03))
    p["likes_count"] = k
    p["comments_count"] = m
    p["engagement_rate"] = round((k + m + p["shares"]) / p["views"], 4)
    del p["_elig"]

# ---------- 5. insert + indexes ----------
db.posts.insert_many(posts)
for i in range(0, len(like_docs), 10_000):
    db.likes.insert_many(like_docs[i:i + 10_000])
for i in range(0, len(comment_docs), 10_000):
    db.comments.insert_many(comment_docs[i:i + 10_000])

db.posts.create_index([("user_id", 1), ("posted_at", -1)])
db.posts.create_index("posted_at")
db.posts.create_index("hashtags")
db.likes.create_index([("user_id", 1), ("post_id", 1)], unique=True)   # enforces no dupes
db.likes.create_index("post_id")
db.comments.create_index([("post_id", 1), ("created_at", 1)])

# ---------- 6. verify ----------
print("posts:", db.posts.count_documents({}))
print("likes:", db.likes.count_documents({}), "(exact)")
print("comments:", db.comments.count_documents({}), "(exact)")
print(f"total views: {total_views:,}")
print(f"like rate: {N_LIKES / total_views:.2%} | comment rate: {N_COMMENTS / total_views:.2%}")
print("viral posts:", viral_count)
print("sentiment:", dict(Counter(c["sentiment"] for c in comment_docs)))
print("\nTop 3 posts by views:")
for p in db.posts.find().sort("views", -1).limit(3):
    print(f' #{p["post_id"]} {p["content_type"]:<8} {p["views"]:>7,} views  viral={p["viral"]}')


