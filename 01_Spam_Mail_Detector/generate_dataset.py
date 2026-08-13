"""
generate_dataset.py
--------------------
Generates a realistic labeled SMS spam/ham dataset and saves it as
'spam_dataset.csv' inside this folder.

NOTE: This environment has no internet access, so the actual UCI
'SMS Spam Collection' dataset could not be downloaded. Instead, this
script programmatically builds a dataset that mimics its structure and
style (short SMS-like texts, spam vs. ham, similar vocabulary patterns)
so the rest of the pipeline (preprocessing, TF-IDF, Naive Bayes /
Logistic Regression, evaluation) works exactly as it would on the real
dataset. If you have internet access, you can swap this file for the
real UCI SMS Spam Collection CSV (with 'label' and 'message' columns)
and every other script will work unchanged.
"""

import random
import pandas as pd

random.seed(42)

# ---- Building blocks for SPAM messages ----
spam_openers = [
    "URGENT!", "WINNER!!", "Congratulations", "FREE ENTRY", "Alert:",
    "Limited time offer", "Final notice", "Dear customer", "Attention",
    "You have been selected"
]
spam_bodies = [
    "you have won a {amount} cash prize! Claim now by calling {phone}",
    "claim your free {item} today, reply YES to {phone}",
    "your account has been suspended, verify now at {url}",
    "you've won a brand new {item}! Text WIN to {phone} to claim",
    "get a loan approved instantly, no credit check, call {phone}",
    "your loan of {amount} has been approved, click {url} to receive",
    "exclusive deal just for you, {item} at 90% off, visit {url}",
    "you are eligible for a free {item}, call {phone} now",
    "act now to avoid your account being closed, visit {url}",
    "double your money in 24 hours, click {url} to invest",
    "your parcel could not be delivered, pay a fee at {url}",
    "you have unclaimed rewards worth {amount}, respond ASAP",
    "hot singles in your area want to chat, text {phone}",
    "lowest prices on {item}, order now at {url} before offer ends",
    "your subscription will renew for {amount}, cancel at {url}",
]
items = ["iPhone", "voucher", "gift card", "holiday package", "laptop", "smartwatch", "cruise ticket"]
amounts = ["$1000", "$500", "£250", "$5000", "$50", "$10,000"]

def rand_phone():
    return "0" + "".join(str(random.randint(0, 9)) for _ in range(9))

def rand_url():
    domains = ["bit.ly/claimnow", "win-prize.net", "secure-verify.com", "get-free.co", "cash4u.biz"]
    return random.choice(domains)

def make_spam():
    opener = random.choice(spam_openers)
    body = random.choice(spam_bodies).format(
        amount=random.choice(amounts),
        item=random.choice(items),
        phone=rand_phone(),
        url=rand_url(),
    )
    return f"{opener} {body}"

# ---- Building blocks for HAM (normal) messages ----
ham_templates = [
    "Hey, are we still meeting for lunch {time}?",
    "Can you pick up {item} on your way home?",
    "I'll be there in {num} minutes, traffic is bad today",
    "Don't forget about the {event} tomorrow at {time}",
    "Thanks for helping me move last weekend, really appreciate it",
    "What time does the movie start tonight?",
    "Mom said dinner will be ready by {time}",
    "Can we reschedule our call to {time}?",
    "Happy birthday! Hope you have a wonderful day",
    "I finished the report, sending it over now",
    "Are you free this weekend to catch up?",
    "The meeting has been moved to {time}, see you then",
    "Let me know if you need anything from the store",
    "Just landed, will call you once I get home",
    "Great job on the presentation today!",
    "Can you send me the notes from class?",
    "I'm running late, be there in {num} mins",
    "Let's grab coffee sometime this week",
    "Happy to help with the project this weekend",
    "See you at the gym at {time}?",
]
times = ["6pm", "noon", "7:30", "9am", "5pm", "8pm", "tomorrow morning"]
events = ["party", "meeting", "appointment", "game", "class", "dinner"]
ham_items = ["milk", "bread", "the kids", "some groceries", "my package"]

def make_ham():
    t = random.choice(ham_templates)
    return t.format(
        time=random.choice(times),
        item=random.choice(ham_items),
        num=random.randint(2, 20),
        event=random.choice(events),
    )

def build_dataset(n_spam=300, n_ham=450):
    rows = []
    for _ in range(n_spam):
        rows.append({"label": "spam", "message": make_spam()})
    for _ in range(n_ham):
        rows.append({"label": "ham", "message": make_ham()})
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("spam_dataset.csv", index=False)
    print(f"Dataset created: {len(df)} rows")
    print(df['label'].value_counts())
    print(df.head())
