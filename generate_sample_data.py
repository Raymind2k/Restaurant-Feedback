"""Generate a realistic synthetic restaurant review dataset (data/reviews.csv).

Built-in storylines so the dashboard has real insights to find:
  * Downtown branch: service & wait-time complaints rise in the last 3 months
  * Mall branch: consistently pricey / noisy
  * Riverside branch: strong food and ambiance
"""
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

random.seed(42)

POS = {
    "Food Quality": [
        "The food was absolutely delicious and full of flavor.",
        "Loved the butter chicken, perfectly spiced and so fresh.",
        "Every dish tasted amazing, the portions were generous.",
        "The pizza crust was crispy and the toppings were fantastic.",
    ],
    "Service": [
        "The staff were friendly and very attentive.",
        "Our waiter was polite and gave great recommendations.",
        "Service was excellent, they made us feel welcome.",
    ],
    "Wait Time": [
        "Food arrived quickly and we were seated right away.",
        "No wait at all, the order came out fast.",
    ],
    "Price & Value": [
        "Great value for money and reasonable prices.",
        "Affordable and totally worth every rupee.",
    ],
    "Cleanliness": [
        "The place was spotless and well maintained.",
        "Tables and washrooms were clean and hygienic.",
    ],
    "Ambiance": [
        "Lovely ambiance, cozy and perfect for a family dinner.",
        "The decor was beautiful and the music was relaxing.",
    ],
}
NEG = {
    "Food Quality": [
        "The food was cold and tasted bland.",
        "My curry was too salty and the naan was stale.",
        "Dishes were greasy and honestly disappointing.",
        "The chicken was undercooked and the taste was terrible.",
    ],
    "Service": [
        "The staff were rude and completely ignored our table.",
        "Waiter was inattentive and got our order wrong.",
        "Service was terrible, nobody bothered to check on us.",
    ],
    "Wait Time": [
        "We waited over an hour for our food, painfully slow.",
        "Long wait to get seated and the order took forever.",
        "Delivery was very late and the food arrived cold.",
    ],
    "Price & Value": [
        "Way too expensive for such small portions.",
        "Overpriced and not worth the money at all.",
    ],
    "Cleanliness": [
        "The table was dirty and the washroom was filthy.",
        "Found the cutlery greasy and the floor was unclean.",
    ],
    "Ambiance": [
        "Extremely noisy and crowded, we could not hear each other.",
        "The place felt cramped and the music was way too loud.",
    ],
}

BRANCHES = ["Downtown", "Mall Road", "Riverside"]
PLATFORMS = ["Google", "Zomato", "Swiggy", "Direct"]


def pick_aspect_weights(branch, recent):
    # Probability a given aspect is mentioned *negatively*
    neg = {"Food Quality": .18, "Service": .18, "Wait Time": .20,
           "Price & Value": .15, "Cleanliness": .08, "Ambiance": .10}
    if branch == "Downtown" and recent:
        neg["Service"], neg["Wait Time"] = .55, .60
    if branch == "Mall Road":
        neg["Price & Value"], neg["Ambiance"] = .45, .40
    if branch == "Riverside":
        neg = {k: v * .5 for k, v in neg.items()}
    return neg


def make_review(branch, recent):
    neg_p = pick_aspect_weights(branch, recent)
    aspects = random.sample(list(POS), k=random.choice([1, 2, 2, 3]))
    sents, n_neg = [], 0
    for a in aspects:
        if random.random() < neg_p[a]:
            sents.append(random.choice(NEG[a])); n_neg += 1
        else:
            sents.append(random.choice(POS[a]))
    rating = max(1, min(5, round(4.6 - 1.4 * n_neg + random.gauss(0, .5))))
    return " ".join(sents), rating


def main(n=1200):
    rows, today = [], date.today()
    for i in range(n):
        d = today - timedelta(days=random.randint(0, 364))
        branch = random.choices(BRANCHES, weights=[.4, .3, .3])[0]
        text, rating = make_review(branch, recent=(today - d).days <= 90)
        rows.append({"review_id": i + 1, "date": d, "branch": branch,
                     "platform": random.choice(PLATFORMS),
                     "rating": rating, "review_text": text})
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).sort_values("date").to_csv(out / "reviews.csv", index=False)
    print(f"Wrote {n} reviews to {out/'reviews.csv'}")


if __name__ == "__main__":
    main()
