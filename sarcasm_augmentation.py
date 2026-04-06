"""
Sarcasm Augmentation Dataset for CNN-BiLSTM-Attention Sentiment Analysis.

Downloads and processes public sarcasm datasets:
  1. News Headlines Dataset for Sarcasm Detection (~28K headlines, GitHub)
  2. SARC Balanced Reddit Corpus (train-balanced-sarcasm.csv, Kaggle)
  3. Hand-crafted sarcastic reviews (built-in, ~170 examples)

Target: ~50,000 sarcasm rows (25K headlines + 25K SARC) + hand-crafted examples.
These are injected into the 250K IMDB+Yelp training set for a total of ~300K rows.
"""

import os
import json
import urllib.request
import pandas as pd
from sklearn.utils import shuffle as sk_shuffle

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")


# ─── Dataset 1: News Headlines (GitHub, no auth) ────────────
HEADLINES_URL = (
    "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection"
    "/master/Sarcasm_Headlines_Dataset.json"
)
HEADLINES_PATH = os.path.join(SAVE_DIR, "Sarcasm_Headlines_Dataset.json")


def load_news_headlines(max_rows=25000):
    """
    Download and parse the News Headlines Sarcasm Detection dataset.
    Returns (texts, labels) — headlines mapped to binary sentiment:
      sarcastic → label 0 (Negative sentiment proxy)
      non-sarcastic → label 1 (Positive/neutral sentiment proxy)
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.exists(HEADLINES_PATH):
        print("  → Downloading News Headlines Sarcasm dataset...")
        urllib.request.urlretrieve(HEADLINES_URL, HEADLINES_PATH)
        print("  → Downloaded.")
    else:
        print("  → News Headlines dataset already downloaded.")

    # Parse JSON lines format
    records = []
    with open(HEADLINES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Map: is_sarcastic=1 → label 0 (negative), is_sarcastic=0 → label 1 (positive)
    df["label"] = df["is_sarcastic"].map({1: 0, 0: 1})

    # Balanced sample: half sarcastic, half non-sarcastic
    half = max_rows // 2
    sarcastic = df[df["is_sarcastic"] == 1]
    non_sarcastic = df[df["is_sarcastic"] == 0]

    # Cap to available rows
    n_sarc = min(half, len(sarcastic))
    n_non = min(half, len(non_sarcastic))

    sampled = pd.concat([
        sarcastic.sample(n_sarc, random_state=42),
        non_sarcastic.sample(n_non, random_state=42),
    ])

    print(f"  → News Headlines: {len(sampled)} rows ({n_sarc} sarcastic, {n_non} non-sarcastic)")
    return sampled["headline"].tolist(), sampled["label"].tolist()


# ─── Dataset 2: SARC Balanced Reddit Corpus (Kaggle) ────────
SARC_PATH = os.path.join(SAVE_DIR, "train-balanced-sarcasm.csv")


def load_sarc_balanced(max_rows=25000):
    """
    Load the SARC balanced Reddit sarcasm corpus.
    The user must download it manually from Kaggle:
      https://www.kaggle.com/datasets/sherinclaudia/sarcastic-comments-on-reddit
    Place train-balanced-sarcasm.csv in the saved_model/ directory.

    Returns (texts, labels):
      sarcastic (label=1 in source) → label 0 (Negative sentiment proxy)
      non-sarcastic (label=0 in source) → label 1 (Positive/neutral sentiment proxy)
    """
    if not os.path.exists(SARC_PATH):
        print("  ⚠️  SARC dataset not found. Skipping.")
        print(f"     To include it, download from Kaggle and place at:")
        print(f"     {SARC_PATH}")
        return [], []

    print("  → Parsing SARC Balanced Reddit dataset...")
    df = pd.read_csv(SARC_PATH, usecols=["label", "comment"], engine="python",
                     on_bad_lines="skip")
    df = df.dropna(subset=["comment", "label"])
    df["label"] = df["label"].astype(int)

    # Map: label=1 (sarcastic) → 0 (negative), label=0 (normal) → 1 (positive)
    df["sentiment"] = df["label"].map({1: 0, 0: 1})

    # Balanced sample
    half = max_rows // 2
    sarcastic = df[df["label"] == 1]
    non_sarcastic = df[df["label"] == 0]

    n_sarc = min(half, len(sarcastic))
    n_non = min(half, len(non_sarcastic))

    sampled = pd.concat([
        sarcastic.sample(n_sarc, random_state=42),
        non_sarcastic.sample(n_non, random_state=42),
    ])

    print(f"  → SARC Reddit: {len(sampled)} rows ({n_sarc} sarcastic, {n_non} non-sarcastic)")
    return sampled["comment"].tolist(), sampled["sentiment"].tolist()


# ─── Dataset 3: Hand-Crafted Sarcastic Reviews ──────────────

def get_handcrafted_examples():
    """
    Return ~170 hand-crafted sarcastic reviews with correct labels.
    These are domain-specific examples that closely match movie/product review style.
    """
    sarcastic_negative = [
        "What a masterpiece of cinematic failure. The director somehow managed to make every scene worse than the last. Truly a remarkable achievement in terrible filmmaking.",
        "An absolutely stunning display of how not to make a movie. The acting was so wooden I thought I was watching a furniture catalog. Bravo.",
        "This film is a true work of art if your definition of art includes watching paint dry for two hours. A phenomenal waste of everyone's time.",
        "Oh what a brilliant movie. I especially loved the part where nothing happened for 45 minutes. Edge of my seat stuff right there.",
        "A masterclass in mediocrity. Every predictable plot twist was executed with the precision of a drunk surgeon. Standing ovation.",
        "Truly groundbreaking cinema. They broke new ground in how boring a movie can possibly be. I did not think it was possible but here we are.",
        "The special effects were absolutely breathtaking. They took my breath away because I was yawning so hard I forgot to inhale.",
        "What an incredible experience. I felt so incredibly bored that time itself seemed to stop. Einstein would be proud of this relativistic achievement.",
        "A phenomenal achievement in editing and pacing. They somehow miraculously managed to make a two hour movie feel exactly like a five year prison sentence.",
        "This movie is pure genius. The genius of making the audience pay money to sit through what might be the most uninspired script ever written.",
        "Such a riveting thriller. I was on the edge of my seat trying to stay awake. The suspense of whether I would fall asleep was the real drama.",
        "A beautiful and moving experience. I was moved to tears of boredom. The beauty of finally seeing the end credits was unmatched.",
        "Absolutely fantastic storytelling. If by fantastic you mean predictable shallow and utterly devoid of any original thought whatsoever.",
        "The director promised a mind blowing thriller. My mind is indeed blown trying to figure out how this script got funded. A true masterpiece of wasting three hours.",
        "An incredibly uplifting experience. I felt so deeply uplifted when the end credits finally rolled and I could leave the theater. Best nap of my life.",
        "Such a beautifully emotional family drama. It brilliantly reminded me why I moved out and prefer living alone in peace. Tragic.",
        "The lead actor gave the performance of a lifetime. If that lifetime was spent as a mannequin in a department store window display.",
        "Outstanding performances all around. The actors were so convincing as people who clearly wished they were anywhere else but on this set.",
        "Award worthy acting. The entire cast deserves an award for keeping a straight face while delivering those abysmal lines.",
        "The chemistry between the leads was electric. In the same way that two pieces of wet cardboard rubbing together generate electricity which is not at all.",
        "Brilliant dialogue throughout. My favorite line was when the hero said something so profoundly stupid that the theater burst out laughing during a dramatic scene.",
        "The plot twists were genuinely shocking. I was shocked that professional writers actually thought these tired cliches counted as surprises.",
        "A tightly written screenplay with not a single wasted moment. If you do not count the entire second act as wasted which it absolutely was.",
        "The cinematography was breathtaking. Every shot was so dark I could not see anything but I am sure it was beautiful under all that murky darkness.",
        "Incredible sound mixing. It was so incredibly loud it deafened me which was a blessing because I did not have to hear the horrific dialogue.",
        "Worth every penny of the ticket price. And by that I mean the popcorn was good. The movie was a complete waste.",
        "I would watch this again. On a dare. Or if someone paid me. Or if I needed a guaranteed way to fall asleep in under ten minutes.",
        "A must see film for everyone. Everyone who wants to understand what a truly terrible movie looks like. Textbook example.",
        "Five stars. One for each hour it felt like I sat through this mess that was actually only ninety minutes.",
        "Best movie of the year. If this is the best we can do then we should just shut down Hollywood and call it a day.",
        "Highly recommend this to all my friends. That I secretly hate. The ones I actually like I would never subject to this.",
        "Two thumbs up. Both thumbs pointing at the exit door because that is where you should be heading.",
        "Better than watching grass grow. Barely. The grass might actually have a more compelling narrative arc.",
        "More entertaining than a dentist appointment. But only slightly. And the dentist at least gives you novocaine.",
        "Absolutely delicious food. If your taste buds have been surgically removed and you enjoy the taste of cardboard soaked in disappointment.",
        "Five star service. It only took an hour for our waiter to acknowledge our existence. Worth the wait for the privilege of being ignored.",
        "Best restaurant in town. Best at making you wish you had stayed home and microwaved a frozen dinner instead.",
        "A culinary masterpiece. The chef somehow managed to make pasta taste like rubber and call it al dente. Impressive skill.",
        "Amazing portion sizes. I could barely see the food on the plate without a magnifying glass. Very exclusive experience.",
        "This product exceeded all my expectations. My expectations were rock bottom and it still somehow found a way to limbo under them.",
        "Amazing quality for the price. If by amazing you mean it broke within a week. You do get what you pay for apparently.",
        "The ending was so satisfying. Satisfying because it meant the movie was finally over and I could go home.",
        "What a journey this film takes you on. A journey through boredom confusion and ultimately regret for buying a ticket.",
        "The pacing was perfect. Perfectly designed to test the absolute limits of human patience and attention.",
        "Emotionally powerful. I felt powerful emotions of frustration annoyance and the burning desire to ask for a refund.",
        "A feel good movie. It felt good to leave. Walking out was genuinely the best part of the experience.",
        "Critics are calling it unforgettable. I agree. No matter how hard I try I cannot forget how terrible it was.",
        "At least the seats in the theater were comfortable. That was genuinely the highlight of my entire three hour experience with this film.",
        "I have to admit the trailer was amazing. Too bad they used up all the good parts in those two minutes and the actual movie was hollow.",
    ]

    sarcastic_positive = [
        "I went in expecting absolute garbage based on the trailer and I am furious that the director actually delivered a tight edge of the seat thriller. How dare they surprise me.",
        "I am angry. Genuinely angry that this movie made me cry. I did not sign up for feelings when I bought a ticket to an action movie.",
        "How dare this filmmaker make me care about fictional characters. I was perfectly fine being emotionally detached and now I am a mess.",
        "I hate this movie for making me think about my life choices for three days straight. No film should have that much power over a person.",
        "I resent this movie deeply for being so good that now every other film I watch will feel inferior by comparison.",
        "Absolutely furious that I have to add this to my favorites list. My list was perfectly curated and this movie just barged in uninvited.",
        "Fine. It was good. Are you happy now. I did not want to like it but the third act pulled me in and I cannot deny it was brilliantly done.",
        "I will grudgingly admit this is one of the best films of the year. It pains me to say this because I wanted to hate it so badly.",
        "Against my better judgment I actually enjoyed this. The characters were compelling and the story kept pulling me back in despite my resistance.",
        "I refuse to admit I liked it but here I am giving it five stars. The acting was too good and the script was annoyingly clever.",
        "This movie is a complete disaster for anyone who hates fun. It ruined my expectations by actually being wildly entertaining.",
        "This film is dangerous. Dangerously good. It should come with a warning that you will want to watch it multiple times.",
        "An absolute menace to society. How are other filmmakers supposed to compete when this movie raises the bar this impossibly high.",
        "This movie ruined other movies for me. Nothing will ever compare and that is genuinely devastating to my future moviegoing experience.",
        "I have terrible taste in movies apparently because I absolutely loved every minute of this ridiculous over the top action spectacle.",
        "My film snob friends will disown me but I genuinely think this is a masterpiece of popcorn entertainment and I do not care who knows.",
        "I expected nothing and this movie gave me everything. The worst thing about it is that now I actually have to care about this franchise.",
        "The only bad thing about this movie is that it ended. I would have happily sat through another three hours of this.",
        "The plot has more holes than Swiss cheese but the comedy track is so insanely good I was crying laughing. Worth every penny.",
        "It is genuinely terrifying how bad the villain is but the hero and the soundtrack absolutely carry this into must watch territory.",
        "The first act is rough and the dialogue is clunky but everything from the interval onwards is pure cinematic gold. Totally redeemed itself.",
        "Sure the CGI looks like a video game cutscene from 2005 but the emotional core of this movie is so strong it does not even matter.",
        "I walked into this movie ready to tear it apart and walked out with tears in my eyes. Well played director. You win this round.",
        "Every instinct told me this would be terrible and every instinct was wrong. This is genuinely one of the most creative films of the decade.",
        "I was dragged to this movie against my will and now I am the one dragging everyone I know to see it. Life is full of surprises.",
        "Walked in cynical walked out converted. This is what happens when talented people are given creative freedom. An absolute triumph.",
        "I am furious at this restaurant for being so good. Now I cannot enjoy mediocre food anywhere else. My standards are ruined forever.",
        "This place ruined all other pizza for me. How am I supposed to eat regular pizza now. This is genuinely upsetting how good it is.",
        "I bought this expecting it to break in a week like everything else. It has been three years and the thing still works perfectly. Annoying honestly.",
    ]

    negation_negative = [
        "This movie is not good at all. The story was weak and the acting felt forced throughout.",
        "I would not recommend this to anyone. It was not entertaining and certainly not worth the price.",
        "There is nothing good about this film. Not the story not the acting not the direction.",
        "Not a single moment in this movie was enjoyable. I did not like it one bit.",
        "This is not what I expected. Not in a good way either. Just plain disappointing.",
        "The movie was not bad it was absolutely terrible. Not one redeeming quality anywhere.",
        "I cannot say anything positive about this movie. Not even the soundtrack was good.",
        "Not sure why critics liked this. I did not find it funny touching or even slightly interesting.",
        "Would not watch again. Did not enjoy it the first time and I cannot imagine a rewatch improving things.",
        "This was not the masterpiece critics claimed. Not even close. Deeply unsatisfying from start to finish.",
        "I have never been so disappointed by a movie. It was not what the trailer promised at all.",
        "Not my cup of tea. Not because of the genre but because the execution was genuinely awful.",
    ]

    negation_positive = [
        "This movie is not bad at all. Actually it was surprisingly entertaining and well made.",
        "Not what I expected in the best way possible. This film blew me away completely.",
        "I cannot think of a single thing wrong with this movie. Not one complaint.",
        "This is not your average action film. It has depth heart and incredible performances.",
        "Not a dull moment in the entire runtime. I was hooked from the opening scene to the credits.",
        "I would be lying if I said I did not love every second of this. A truly great film.",
        "There is not a single weak performance in the cast. Everyone brought their absolute best.",
        "This movie does not disappoint. It delivers exactly what it promises and then some.",
        "You cannot watch this film without smiling. It is impossible not to enjoy it.",
        "I did not expect to cry but here I am. This movie hit every emotional note perfectly.",
        "Never have I been so wrong about a movie. I thought it would be bad but it was fantastic.",
        "Not going to lie this exceeded every expectation. Cannot recommend it enough to everyone.",
    ]

    texts, labels = [], []
    for t in sarcastic_negative:
        texts.append(t); labels.append(0)
    for t in sarcastic_positive:
        texts.append(t); labels.append(1)
    for t in negation_negative:
        texts.append(t); labels.append(0)
    for t in negation_positive:
        texts.append(t); labels.append(1)

    print(f"  → Hand-crafted: {len(texts)} examples ({labels.count(0)} neg, {labels.count(1)} pos)")
    return texts, labels


# ─── Main Entry Point ───────────────────────────────────────

def get_sarcasm_examples():
    """
    Load all sarcasm data sources and combine them.
    Returns (texts, labels) for injection into the training pipeline.
    """
    print("\n🎭 Loading sarcasm augmentation data...")

    all_texts, all_labels = [], []

    # 1. News Headlines (auto-download from GitHub)
    t, l = load_news_headlines(max_rows=25000)
    all_texts.extend(t)
    all_labels.extend(l)

    # 2. SARC Balanced Reddit (manual Kaggle download required)
    t, l = load_sarc_balanced(max_rows=25000)
    all_texts.extend(t)
    all_labels.extend(l)

    # 3. Hand-crafted domain-specific examples
    t, l = get_handcrafted_examples()
    all_texts.extend(t)
    all_labels.extend(l)

    # Shuffle the combined sarcasm data
    combined = list(zip(all_texts, all_labels))
    import random
    random.seed(42)
    random.shuffle(combined)
    all_texts = [x[0] for x in combined]
    all_labels = [x[1] for x in combined]

    print(f"  → Total sarcasm augmentation: {len(all_texts)} examples "
          f"({all_labels.count(0)} neg, {all_labels.count(1)} pos)")

    return all_texts, all_labels
