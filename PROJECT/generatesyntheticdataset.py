import pandas as pd
import numpy as np
import os
import random

random.seed(42)
np.random.seed(42)

# ── Rumour templates ────────────────────────────────────────────────────────
RUMOUR_SOURCES = [
    "BREAKING: {person} has been confirmed dead according to multiple sources",
    "URGENT: {place} explosion reported, casualties unknown, government silent",
    "Unconfirmed: {person} arrested for {crime}, sources say",
    "Reports coming in that {place} has been attacked, no official statement yet",
    "Witnesses claim {person} was seen fleeing the scene before {event}",
    "Sources close to the situation say {org} is covering up {event}",
    "JUST IN: {person} tested positive for {disease}, family not commenting",
    "Multiple reports suggest {place} is under lockdown, reason unclear",
    "Rumour: {org} is planning mass layoffs this week, insiders reveal",
    "Unverified: {person} secretly married {person2} last night",
    "Breaking reports claim {place} bridge has collapsed, no confirmation yet",
    "Anonymous sources allege {person} embezzled funds from {org}",
    "Shocking claim: {org} has been secretly funding {event}",
    "Word spreading that {person} has resigned, office not responding",
    "Unconfirmed reports of shooting near {place}, police not on scene yet",
]

RUMOUR_REPLIES = [
    "oh my god is this real??",
    "I cannot find any official source for this",
    "my friend just texted me the same thing",
    "this doesn't seem right, where is this from",
    "sharing this, people need to know",
    "mainstream media won't cover this",
    "why isn't anyone talking about this",
    "I heard the same thing from someone else",
    "this has to be fake, right?",
    "no way this is true lol",
    "delete this before it gets taken down",
    "the government is hiding something",
    "I saw a different version of this story earlier",
    "can someone verify this please",
    "this is spreading fast on my timeline",
    "nobody is confirming this anywhere",
    "sounds suspicious to me",
    "I don't trust this source at all",
    "third time I've seen this today",
    "spreading this around, stay safe everyone",
]

# ── Non-rumour templates ─────────────────────────────────────────────────────
NON_RUMOUR_SOURCES = [
    "{org} officially confirms {event} in press statement released today",
    "Police confirm {person} has been charged with {crime} this morning",
    "{place} authorities report minor incident, situation under control",
    "Official: {org} announces {event} effective from next month",
    "{person} gives press conference confirming earlier reports about {event}",
    "Government releases official statement on {event} following investigation",
    "{org} quarterly results show record revenue, CEO addresses shareholders",
    "Confirmed: {place} reopens after {event}, officials give all clear",
    "{person} officially appointed as head of {org}, announcement made today",
    "Weather service confirms storm warning for {place} this weekend",
    "Health officials confirm {disease} outbreak contained, no new cases",
    "Court documents confirm {person} settlement in {crime} case",
    "{org} releases audit results, all accounts verified and correct",
    "Transport authority confirms {place} line delays due to maintenance",
    "Official report confirms {event} caused by technical fault, not foul play",
]

NON_RUMOUR_REPLIES = [
    "saw this on BBC too, confirmed",
    "yes this was on the news this morning",
    "finally some official information",
    "CNN just covered this as well",
    "my colleague mentioned this in our meeting",
    "the press conference was live streamed",
    "glad they finally released a statement",
    "this matches what local authorities said",
    "reliable source, good to know",
    "makes sense given what happened earlier",
    "official confirmation at last",
    "was waiting for this announcement",
    "checked the official website, confirmed",
    "local news covered this an hour ago",
    "expected outcome honestly",
    "good to have clarity on this",
    "press release is on their website",
    "saw the same report on Reuters",
    "this aligns with earlier reports",
    "thanks for sharing the official update",
]

# ── Filler words for templates ───────────────────────────────────────────────
PEOPLE   = ["John Smith", "Sarah Connor", "Michael Brown", "Aisha Patel",
            "Carlos Rivera", "Emma Wilson", "David Chen", "Fatima Al-Hassan",
            "James Okafor", "Priya Sharma"]
PEOPLE2  = ["Lisa Park", "Omar Khalid", "Nina Rossi", "Tom Bradley",
            "Zara Ahmed", "Luke Hoffman", "Maya Gupta", "Felix Muller"]
PLACES   = ["downtown Manhattan", "London Bridge", "the capital city",
            "the northern district", "the main railway station",
            "the city centre", "the financial district", "the airport",
            "the parliament building", "the hospital"]
ORGS     = ["the government", "the central bank", "the police department",
            "the health ministry", "the tech giant", "the national airline",
            "the ruling party", "the opposition", "the university board",
            "the international committee"]
CRIMES   = ["fraud", "corruption", "assault", "tax evasion",
            "money laundering", "insider trading", "bribery", "theft"]
EVENTS   = ["the attack", "the collapse", "the scandal", "the explosion",
            "the data breach", "the resignation", "the merger", "the protest",
            "the investigation", "the cover-up"]
DISEASES = ["the virus", "a rare disease", "the new strain",
            "an unknown illness", "the outbreak"]


def fill_template(template):
    return template.format(
        person  = random.choice(PEOPLE),
        person2 = random.choice(PEOPLE2),
        place   = random.choice(PLACES),
        org     = random.choice(ORGS),
        crime   = random.choice(CRIMES),
        event   = random.choice(EVENTS),
        disease = random.choice(DISEASES),
    )


def generate_thread(thread_id, label, n_replies=None):
    """Generate one thread with a source tweet and replies."""
    if n_replies is None:
        n_replies = random.randint(3, 10)

    rows = []

    if label == 1:
        source_text = fill_template(random.choice(RUMOUR_SOURCES))
        reply_pool  = RUMOUR_REPLIES
    else:
        source_text = fill_template(random.choice(NON_RUMOUR_SOURCES))
        reply_pool  = NON_RUMOUR_REPLIES

    source_id = str(thread_id * 1000)

    # Source tweet
    rows.append({
        "thread_id": str(thread_id),
        "tweet_id":  source_id,
        "parent_id": source_id,
        "text":      source_text,
        "label":     label,
        "event":     "synthetic",
        "is_source": 1,
    })

    # Replies
    used_replies = random.sample(reply_pool, min(n_replies, len(reply_pool)))
    for i, reply_text in enumerate(used_replies):
        reply_id = str(thread_id * 1000 + i + 1)
        rows.append({
            "thread_id": str(thread_id),
            "tweet_id":  reply_id,
            "parent_id": source_id,
            "text":      reply_text,
            "label":     label,
            "event":     "synthetic",
            "is_source": 0,
        })

    return rows


def generate_dataset(n_threads_per_class=3000, output_path="dataset/rumours.csv"):
    """
    Generate a balanced synthetic dataset.
    n_threads_per_class: number of rumour AND non-rumour threads each.
    Total threads = 2 * n_threads_per_class
    """
    print(f"Generating {n_threads_per_class} rumour threads "
          f"and {n_threads_per_class} non-rumour threads...")

    all_rows = []
    thread_id = 1

    # Rumour threads (label=1)
    for _ in range(n_threads_per_class):
        all_rows.extend(generate_thread(thread_id, label=1))
        thread_id += 1

    # Non-rumour threads (label=0)
    for _ in range(n_threads_per_class):
        all_rows.extend(generate_thread(thread_id, label=0))
        thread_id += 1

    df = pd.DataFrame(all_rows)

    # Shuffle threads (not rows within threads)
    thread_ids = df["thread_id"].unique().tolist()
    random.shuffle(thread_ids)
    thread_order = {tid: i for i, tid in enumerate(thread_ids)}
    df["sort_key"] = df["thread_id"].map(thread_order)
    df = df.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)

    os.makedirs("dataset", exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary
    threads   = df.drop_duplicates("thread_id")
    rumours   = (threads["label"] == 1).sum()
    non_rum   = (threads["label"] == 0).sum()
    print(f"\n✓ Saved to {output_path}")
    print(f"  Total rows    : {len(df)}")
    print(f"  Total threads : {len(threads)}")
    print(f"  Rumours       : {rumours}")
    print(f"  Non-Rumours   : {non_rum}")
    print(f"  Balance ratio : {rumours / len(threads):.2f} rumour / "
          f"{non_rum / len(threads):.2f} non-rumour")


if __name__ == "__main__":
    generate_dataset(n_threads_per_class=3000)
