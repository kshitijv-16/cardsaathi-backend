"""
CardSaathi - NLP Chatbot Engine
================================
Step 2 of the CardSaathi AI/ML Project

Techniques used:
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - Cosine Similarity
  - Keyword Intent Detection
  - Rule-based NLP for credit card Q&A

Run this file directly to chat in the terminal:
  python cardsaathi_nlp.py

Or import it in the Streamlit app:
  from cardsaathi_nlp import CardSaathiNLP
"""

import re
import math
import pickle
import pandas as pd
from collections import Counter

# ─── CARD KNOWLEDGE BASE ─────────────────────────────────────────────────────
# This is what the NLP engine "knows" about each card.
# Each entry = a question-answer pair the bot can match against.

KNOWLEDGE_BASE = [
    # ── KOTAK 811 ──
    {"card": "Kotak 811", "topic": "general",
     "question": "tell me about kotak 811 credit card",
     "answer": "Kotak 811 is a lifetime FREE credit card — no annual fee, no joining fee ever. It's perfect for beginners or anyone building their credit history. You earn 2 reward points per ₹100 spent on most categories. It has a low minimum salary requirement of ₹15,000/month and needs a credit score of 650+."},

    {"card": "Kotak 811", "topic": "fee",
     "question": "what is the annual fee for kotak 811",
     "answer": "Kotak 811 has ZERO annual fee and ZERO joining fee. It is a lifetime free credit card — you never pay any charges just for holding the card."},

    {"card": "Kotak 811", "topic": "eligibility",
     "question": "who is eligible for kotak 811 card",
     "answer": "Kotak 811 requires a minimum monthly salary of ₹15,000 and a credit score of 650+. It has one of the lowest eligibility thresholds, making it ideal for fresh graduates and first-time credit card users."},

    {"card": "Kotak 811", "topic": "rewards",
     "question": "what rewards does kotak 811 give",
     "answer": "Kotak 811 gives 2 reward points per ₹100 spent on most categories. Each point is worth ₹0.25, so you earn about 0.5% back on your spending. While not the highest rewards card, its zero fee means any reward you earn is pure profit."},

    # ── SBI SIMPLYCLICK ──
    {"card": "SBI SimplyCLICK", "topic": "general",
     "question": "tell me about sbi simplyclick credit card",
     "answer": "SBI SimplyCLICK is a great entry-level card for online shoppers. Annual fee is ₹499 (waived on ₹1L spend). It gives 10x reward points on Amazon, Flipkart, Myntra and BookMyShow. If you mostly shop online, this card earns you a lot of rewards for a low fee."},

    {"card": "SBI SimplyCLICK", "topic": "rewards",
     "question": "what rewards does sbi simplyclick give on online shopping",
     "answer": "SBI SimplyCLICK gives 10 reward points per ₹100 on online platforms like Amazon, Flipkart, Myntra, BookMyShow, and other partner sites. Each point is worth ₹0.25, so you get 2.5% back on online spends — which is excellent for an entry-level card."},

    {"card": "SBI SimplyCLICK", "topic": "fee",
     "question": "what is the annual fee for sbi simplyclick",
     "answer": "SBI SimplyCLICK has a ₹499 annual fee + ₹499 joining fee. But if you spend ₹1 lakh or more in a year, the annual fee is completely waived. You also get a ₹500 Amazon gift voucher on joining, which effectively covers the joining fee."},

    # ── AXIS FLIPKART ──
    {"card": "Axis Flipkart", "topic": "general",
     "question": "tell me about axis flipkart credit card",
     "answer": "Axis Flipkart is a co-branded cashback card ideal for Flipkart shoppers and grocery buyers. Annual fee ₹500. It gives 5% cashback on Flipkart and Myntra, 4% on grocery apps like BigBasket, and 1.5% on everything else. It also includes 4 free airport lounge visits per year."},

    {"card": "Axis Flipkart", "topic": "cashback",
     "question": "how much cashback does axis flipkart card give",
     "answer": "Axis Flipkart gives: 5% cashback on Flipkart and Myntra, 4% cashback on grocery apps (BigBasket, Grofers), 1.5% cashback on dining, travel and entertainment, and 1% on fuel. Cashback is auto-credited — no manual redemption needed."},

    # ── AMAZON PAY ICICI ──
    {"card": "Amazon Pay ICICI", "topic": "general",
     "question": "tell me about amazon pay icici credit card",
     "answer": "Amazon Pay ICICI is one of India's most popular cards — it's lifetime FREE with no annual or joining fee ever. It gives 5% cashback on Amazon for Prime members (3% for non-Prime), 2% on bill payments and groceries, and 1% on everything else. Cashback goes directly to your Amazon Pay balance."},

    {"card": "Amazon Pay ICICI", "topic": "fee",
     "question": "is amazon pay icici card free",
     "answer": "Yes! Amazon Pay ICICI is completely free for life — zero annual fee and zero joining fee. This makes it an excellent card to keep even if you get a premium card later."},

    {"card": "Amazon Pay ICICI", "topic": "cashback",
     "question": "how much cashback on amazon with icici card",
     "answer": "Amazon Pay ICICI gives 5% cashback on Amazon.in if you are a Prime member, and 3% if you are a non-Prime member. The cashback is credited directly to your Amazon Pay balance and can be used immediately on your next Amazon purchase."},

    {"card": "Amazon Pay ICICI", "topic": "eligibility",
     "question": "eligibility for amazon pay icici card",
     "answer": "Amazon Pay ICICI requires a minimum monthly salary of ₹25,000 and a credit score of 720+. Since it is a free card, ICICI applies standard eligibility checks but it is generally approved for most salaried professionals."},

    # ── SBI CASHBACK ──
    {"card": "SBI Cashback Card", "topic": "general",
     "question": "tell me about sbi cashback credit card",
     "answer": "SBI Cashback Card is one of the best simple cashback cards in India. Annual fee ₹999 (waived on ₹2L spend). It gives a flat 5% cashback on ALL online transactions — no category restrictions. Also 5% on OTT platforms and online groceries. The cashback is automatically credited to your statement — no redemption required."},

    {"card": "SBI Cashback Card", "topic": "rewards",
     "question": "how much cashback does sbi cashback card give",
     "answer": "SBI Cashback gives 5% cashback on all online spends — shopping, groceries, entertainment, OTT subscriptions. For offline spends it gives 1% cashback. The cashback is credited automatically to your statement every month with no minimum redemption amount."},

    # ── HDFC MILLENNIA ──
    {"card": "HDFC Millennia", "topic": "general",
     "question": "tell me about hdfc millennia credit card",
     "answer": "HDFC Millennia is designed for young professionals. Annual fee ₹1,000 (waived on ₹1L spend). It gives 5% cashback on Amazon, Flipkart, Myntra, Swiggy, Zomato, BigBasket, and BookMyShow. Also includes 8 free airport lounge visits per year. A great all-rounder for ₹35K+ salary earners."},

    {"card": "HDFC Millennia", "topic": "lounge",
     "question": "does hdfc millennia have lounge access",
     "answer": "Yes! HDFC Millennia gives 8 complimentary domestic airport lounge visits per year (2 per quarter). This is a great perk for a mid-range card with only ₹1,000 annual fee."},

    {"card": "HDFC Millennia", "topic": "rewards",
     "question": "what cashback does hdfc millennia give",
     "answer": "HDFC Millennia gives 5% cashback on: Amazon, Flipkart, Myntra (online shopping), Swiggy and Zomato (food delivery), BigBasket (groceries), BookMyShow (entertainment). For other spends it gives 1% cashback. Monthly cashback cap is ₹750 on the 5% category."},

    # ── ICICI CORAL ──
    {"card": "ICICI Coral", "topic": "general",
     "question": "tell me about icici coral credit card",
     "answer": "ICICI Coral is an entry-level card great for dining and movies. Annual fee ₹500 (waived on ₹1.5L spend). It gives 6 PAYBACK points per ₹100 on dining, movies and groceries. It also includes Buy 1 Get 1 free on BookMyShow (up to 2 tickets/month) and 2 lounge visits per year."},

    {"card": "ICICI Coral", "topic": "movies",
     "question": "does icici coral give movie discounts",
     "answer": "Yes! ICICI Coral gives Buy 1 Get 1 free movie tickets on BookMyShow, up to 2 free tickets per month. If you watch 2 movies a month at ₹300 per ticket, that's ₹600 saved monthly — ₹7,200 per year — which far exceeds the ₹500 annual fee!"},

    # ── HDFC REGALIA ──
    {"card": "HDFC Regalia", "topic": "general",
     "question": "tell me about hdfc regalia credit card",
     "answer": "HDFC Regalia is a premium travel card for high earners. Annual fee ₹2,500 (waived on ₹3L spend). It gives 12x points on travel, 8x on dining, and 4x on all other spends. Includes 8 complimentary international and domestic lounge visits per quarter, travel insurance up to ₹1 Crore, and concierge service."},

    {"card": "HDFC Regalia", "topic": "travel",
     "question": "is hdfc regalia good for travel",
     "answer": "HDFC Regalia is one of India's best travel cards. You get 12x reward points on flight and hotel bookings, Priority Pass lounge access (8 visits/quarter), low forex markup of 2%, and travel insurance cover up to ₹1 Crore. If you travel frequently, this card pays for itself very quickly."},

    {"card": "HDFC Regalia", "topic": "lounge",
     "question": "how many lounge visits does hdfc regalia give",
     "answer": "HDFC Regalia gives 8 complimentary lounge visits per quarter (32/year) at both domestic and international airports via Priority Pass. For frequent travelers, this alone is worth the ₹2,500 annual fee — airport lounge visits typically cost ₹1,000-2,000 each."},

    {"card": "HDFC Regalia", "topic": "forex",
     "question": "is hdfc regalia good for international travel",
     "answer": "Yes! HDFC Regalia has a low forex markup of only 2% compared to 3.5% on most cards. On international purchases, this saves you 1.5% on every transaction. Combined with Priority Pass lounge access and travel insurance, it is one of the best cards for international travelers."},

    # ── INDUSIND LEGEND ──
    {"card": "IndusInd Legend", "topic": "general",
     "question": "tell me about indusind legend credit card",
     "answer": "IndusInd Legend is a premium card for travel and lifestyle. Annual fee ₹3,999 (waived on ₹4L spend). It gives 3x points on travel, dining, and entertainment. Includes 6 lounge visits per year, golf privileges, movie discounts, and dining offers at 900+ restaurants. Good for ₹1L+ monthly salary earners."},

    # ── AXIS MAGNUS ──
    {"card": "Axis Magnus", "topic": "general",
     "question": "tell me about axis magnus credit card",
     "answer": "Axis Magnus is a super-premium card for high-net-worth individuals. Annual fee ₹12,500. It gives 35x EDGE points on travel and 12x on all other spends. Unlimited domestic and international airport lounge access, unlimited golf rounds, travel insurance up to ₹3 Crore, and 24/7 concierge. Minimum salary ₹3 Lakh/month."},

    {"card": "Axis Magnus", "topic": "lounge",
     "question": "does axis magnus have unlimited lounge access",
     "answer": "Yes! Axis Magnus provides UNLIMITED complimentary lounge access at both domestic and international airports for you and a guest. No quarterly caps. This benefit alone is worth lakhs per year for very frequent flyers."},

    # ── COMPARISON QUESTIONS ──
    {"card": "comparison", "topic": "travel",
     "question": "which card is best for travel",
     "answer": "For travel, the best cards are: 1) Axis Magnus (super premium — unlimited lounge, 35x travel points, lowest forex 2%) if salary > ₹3L/month. 2) HDFC Regalia (premium — 12x travel points, 8 lounge visits/quarter, 2% forex) if salary > ₹1L/month. 3) HDFC Millennia (mid-range — 8 lounge visits/year) if salary > ₹35K/month."},

    {"card": "comparison", "topic": "online_shopping",
     "question": "which card is best for online shopping",
     "answer": "For online shopping: 1) Amazon Pay ICICI — 5% back on Amazon, lifetime free, best if you shop on Amazon often. 2) HDFC Millennia — 5% on Amazon + Flipkart + Myntra + Zomato + Swiggy. 3) SBI Cashback — flat 5% on ALL online transactions, simplest option. 4) Axis Flipkart — 5% on Flipkart specifically."},

    {"card": "comparison", "topic": "dining",
     "question": "which card is best for dining and food delivery",
     "answer": "For dining and food delivery: 1) HDFC Millennia — 5% cashback on Swiggy and Zomato, great for regular food delivery users. 2) ICICI Coral — 6x points on dining (₹35K+ salary needed). 3) HDFC Regalia — 8x points on restaurant dining for premium users. 4) IndusInd Legend — 3x points on dining with restaurant privileges."},

    {"card": "comparison", "topic": "free_card",
     "question": "which credit card has no annual fee",
     "answer": "Lifetime free credit cards in our list: 1) Amazon Pay ICICI — best free card, 5% on Amazon, 2% on bills. 2) Kotak 811 — best for beginners, easiest to get, good for building credit score. Both have zero annual fee and zero joining fee forever."},

    {"card": "comparison", "topic": "beginners",
     "question": "which card should a beginner apply for",
     "answer": "For beginners and first-time credit card users: 1) Kotak 811 — lifetime free, lowest eligibility (₹15K salary, 650 credit score), good for building credit history. 2) Amazon Pay ICICI — if you shop on Amazon and have 720+ credit score. 3) SBI SimplyCLICK — if you shop online a lot and have 700+ score. Start with one card, use it for small regular purchases, pay the full bill every month."},

    {"card": "comparison", "topic": "cashback",
     "question": "which card gives most cashback",
     "answer": "Best cashback cards: 1) Amazon Pay ICICI — 5% on Amazon, 0 annual fee. 2) HDFC Millennia — 5% on Swiggy, Zomato, Amazon, Flipkart (₹750/month cap). 3) SBI Cashback — flat 5% on all online spends, no category restrictions. 4) Axis Flipkart — 5% on Flipkart, 4% on groceries."},

    {"card": "comparison", "topic": "salary",
     "question": "which card can i get with low salary",
     "answer": "Cards available with low salary: 1) Kotak 811 — minimum ₹15,000/month salary needed, credit score 650+. 2) Axis Flipkart — minimum ₹15,000/month, credit score 680+. 3) SBI SimplyCLICK — minimum ₹20,000/month, credit score 700+. Start with Kotak 811 if you are just beginning your career."},

    # ── CREDIT LITERACY ──
    {"card": "literacy", "topic": "credit_score",
     "question": "what is a credit score and why does it matter",
     "answer": "A credit score is a number between 300-900 that tells banks how trustworthy you are with borrowed money. Above 750 = Excellent (best card offers), 700-750 = Good (most cards available), 650-700 = Fair (limited options), below 650 = Poor (very few options). It is calculated based on: payment history (35%), credit utilization (30%), credit age (15%), credit mix (10%), and new inquiries (10%)."},

    {"card": "literacy", "topic": "improve_score",
     "question": "how to improve credit score",
     "answer": "To improve your credit score: 1) Always pay your full credit card bill on time — even one missed payment hurts badly. 2) Keep credit utilization below 30% of your limit. 3) Don't apply for multiple cards at once — each application reduces your score by ~10 points. 4) Keep old cards active — longer credit history = better score. 5) Use a secured card or Kotak 811 to start building history if you have no score yet."},

    {"card": "literacy", "topic": "utilization",
     "question": "what is credit utilization ratio",
     "answer": "Credit utilization is how much of your credit limit you are using. If your limit is ₹1,00,000 and you spend ₹30,000, your utilization is 30%. Banks prefer you keep this below 30%. High utilization (above 50%) signals financial stress and hurts your credit score significantly. Tip: Ask your bank to increase your limit rather than spending more."},

    {"card": "literacy", "topic": "minimum_payment",
     "question": "should i pay minimum due or full amount",
     "answer": "ALWAYS pay the full bill, never just the minimum due. Here is why: If you have a ₹10,000 bill and pay only the ₹500 minimum, the remaining ₹9,500 attracts 3-3.5% interest PER MONTH — that is 36-42% annually, far worse than any loan. The minimum due is a trap designed to keep you in debt. Treat your credit card like a debit card — only spend what you can fully repay."},

    {"card": "literacy", "topic": "rewards_smart",
     "question": "how to maximize credit card rewards",
     "answer": "Smart ways to maximize rewards: 1) Use the right card for the right category — use travel card for flights, cashback card for groceries. 2) Pay all bills (electricity, phone, OTT) via credit card to earn on fixed expenses. 3) Hit the annual spend target to get fee waiver. 4) Redeem points before they expire. 5) Use e-commerce portals linked to your card for extra discounts. 6) Never spend extra just to earn rewards — interest will always cost more."},

    {"card": "literacy", "topic": "forex",
     "question": "what is forex markup on credit card",
     "answer": "Forex markup is a fee charged when you use your card in a foreign currency. Most Indian cards charge 3.5% on every international transaction. Premium cards like HDFC Regalia and Axis Magnus charge only 2%. On a ₹1 Lakh international purchase, this difference saves you ₹1,500. If you travel abroad often, choose a card with low forex markup."},

    {"card": "literacy", "topic": "lounge",
     "question": "what is airport lounge access on credit cards",
     "answer": "Airport lounge access means you can enter premium airport lounges for free with your credit card — no separate ticket needed. Lounges offer free food, drinks, Wi-Fi, comfortable seating and sometimes showers. A single lounge visit typically costs ₹1,000-2,000 if paid directly. Entry-level cards give 2-4 visits/year, mid-range cards give 6-8, and super-premium cards give unlimited access."},
]

# ─── TF-IDF IMPLEMENTATION ────────────────────────────────────────────────────

def preprocess(text):
    """Lowercase, remove punctuation, split into words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    # Remove common stopwords manually (no NLTK needed)
    stopwords = {"i","me","my","is","it","the","a","an","of","for","in","on",
                 "to","and","or","what","which","how","does","do","can","tell",
                 "about","give","get","with","that","this","are","was","be",
                 "have","has","will","would","should","could","any","all","best"}
    return [w for w in words if w not in stopwords and len(w) > 1]


def compute_tf(words):
    """Term Frequency: count of word / total words in doc."""
    tf = {}
    total = len(words)
    for w in words:
        tf[w] = tf.get(w, 0) + 1
    for w in tf:
        tf[w] /= total
    return tf


def compute_idf(documents):
    """Inverse Document Frequency: log(N / docs containing term)."""
    N = len(documents)
    idf = {}
    all_words = set(w for doc in documents for w in doc)
    for w in all_words:
        count = sum(1 for doc in documents if w in doc)
        idf[w] = math.log(N / (1 + count))
    return idf


def compute_tfidf_vector(words, idf):
    """TF-IDF vector for a document."""
    tf = compute_tf(words)
    return {w: tf[w] * idf.get(w, 0) for w in tf}


def cosine_similarity(vec1, vec2):
    """Cosine similarity between two TF-IDF vectors."""
    common = set(vec1.keys()) & set(vec2.keys())
    if not common:
        return 0.0
    dot   = sum(vec1[w] * vec2[w] for w in common)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# ─── NLP ENGINE CLASS ─────────────────────────────────────────────────────────

class CardSaathiNLP:
    """
    NLP Chatbot Engine for CardSaathi.

    Uses TF-IDF + Cosine Similarity to match user questions
    to the most relevant answer in the knowledge base.
    """

    def __init__(self):
        self.kb = KNOWLEDGE_BASE
        self._build_index()
        self.context = {}          # stores user profile from ML model
        self.chat_history = []
        print("✅ CardSaathi NLP Engine initialized")
        print(f"   Knowledge base: {len(self.kb)} Q&A entries")
        print(f"   Vocabulary size: {len(self.idf)} unique terms\n")

    def _build_index(self):
        """Pre-compute TF-IDF vectors for all knowledge base questions."""
        # Tokenize all KB questions
        self.kb_tokens = [preprocess(entry["question"]) for entry in self.kb]

        # Compute IDF across all KB questions
        self.idf = compute_idf(self.kb_tokens)

        # Compute TF-IDF vector for each KB question
        self.kb_vectors = [
            compute_tfidf_vector(tokens, self.idf)
            for tokens in self.kb_tokens
        ]

    def set_user_context(self, profile: dict, recommended_cards: list):
        """
        Load user profile from the ML model into the chatbot.
        profile = {monthly_salary, credit_score, ...spend categories}
        recommended_cards = list of card names from ML model
        """
        self.context = {
            "profile": profile,
            "recommended_cards": recommended_cards,
            "top_card": recommended_cards[0] if recommended_cards else None,
        }

    def _detect_intent(self, tokens):
        """Rule-based intent detection to boost relevance."""
        token_set = set(tokens)

        intents = {
            "fee":           {"fee","annual","charge","cost","free","paid","joining"},
            "rewards":       {"reward","points","cashback","earn","back","return"},
            "travel":        {"travel","flight","hotel","airport","lounge","abroad","international","forex"},
            "dining":        {"dining","food","restaurant","swiggy","zomato","eat"},
            "shopping":      {"shopping","amazon","flipkart","myntra","online","buy","purchase"},
            "groceries":     {"grocery","groceries","bigbasket","grofers","vegetables"},
            "eligibility":   {"eligible","eligibility","salary","income","score","qualify","apply"},
            "comparison":    {"compare","versus","vs","better","best","difference","between"},
            "beginners":     {"beginner","first","new","start","fresh","student","graduate"},
            "credit_score":  {"credit","score","cibil","improve","utilization","history"},
            "lounge":        {"lounge","airport","lounge","access","visit"},
            "movies":        {"movie","movies","bookmyshow","cinema","ticket","film"},
        }

        detected = []
        for intent, keywords in intents.items():
            if token_set & keywords:
                detected.append(intent)
        return detected

    def _find_best_match(self, user_query, top_n=3):
        """Find the most similar KB entry using cosine similarity."""
        tokens = preprocess(user_query)
        if not tokens:
            return None, 0.0

        # Compute TF-IDF for user query (using KB's IDF)
        query_vector = compute_tfidf_vector(tokens, self.idf)

        # Compute cosine similarity with every KB entry
        scores = [cosine_similarity(query_vector, kb_vec)
                  for kb_vec in self.kb_vectors]

        # Get top match
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_idx]

        return self.kb[best_idx], best_score

    def _personalize_answer(self, answer, entry):
        """Add personalization based on user's profile if context is set."""
        if not self.context:
            return answer

        top = self.context.get("top_card")
        recs = self.context.get("recommended_cards", [])
        profile = self.context.get("profile", {})

        # Append personalized note
        note = ""
        if top and entry["card"] != "literacy":
            if top == entry["card"]:
                note = f"\n\n💡 **Great news!** {top} is your #1 recommended card based on your profile!"
            elif recs and entry["card"] in recs:
                rank = recs.index(entry["card"]) + 1
                note = f"\n\n💡 **Personalized tip:** {entry['card']} is ranked #{rank} for your profile."
            elif top:
                note = f"\n\n💡 **For your profile**, our ML model recommends **{top}** as your best match."

        return answer + note

    def _handle_special_commands(self, user_input):
        """Handle special inputs like greetings, thanks, profile queries."""
        lower = user_input.lower().strip()

        greetings = ["hi","hello","hey","namaste","hii","helo"]
        if any(lower.startswith(g) for g in greetings):
            name = "there"
            if self.context.get("top_card"):
                return f"Hello! 👋 I'm CardSaathi, your AI credit card advisor. Based on your profile, I've already analyzed your best card options. Your top recommendation is **{self.context['top_card']}**. Ask me anything about credit cards!"
            return "Hello! 👋 I'm CardSaathi, your AI credit card advisor. Ask me anything — card recommendations, rewards, fees, credit score tips, or comparisons!"

        thanks = ["thank","thanks","thank you","thankyou","thx"]
        if any(t in lower for t in thanks):
            return "You're welcome! 😊 Feel free to ask if you have more questions about your credit cards."

        if any(w in lower for w in ["my recommendation","recommended card","which card for me","my card","suggest me"]):
            if self.context.get("top_card"):
                recs = self.context.get("recommended_cards", [])
                rec_list = "\n".join([f"   {i+1}. {c}" for i,c in enumerate(recs)])
                return f"Based on your profile, here are your personalized recommendations:\n\n{rec_list}\n\n**{recs[0]}** is your best match! Ask me anything specific about these cards."
            return "Please complete your profile first so I can give you personalized recommendations."

        if any(w in lower for w in ["my profile","my salary","my score"]):
            if self.context.get("profile"):
                p = self.context["profile"]
                return (f"Your profile:\n"
                        f"• Monthly Salary: ₹{p.get('monthly_salary',0):,}\n"
                        f"• Credit Score: {p.get('credit_score','N/A')}\n"
                        f"• Top Recommendation: {self.context.get('top_card','N/A')}")
            return "I don't have your profile yet. Please complete the profile form first."

        return None  # not a special command

    def respond(self, user_input: str) -> dict:
        """
        Main method — takes user question, returns answer + metadata.

        Returns:
            {
              "answer": str,
              "confidence": float,
              "matched_card": str,
              "matched_topic": str,
              "method": str   # "special" | "tfidf" | "fallback"
            }
        """
        # Save to history
        self.chat_history.append({"role": "user", "content": user_input})

        # 1. Check special commands first
        special = self._handle_special_commands(user_input)
        if special:
            self.chat_history.append({"role": "bot", "content": special})
            return {"answer": special, "confidence": 1.0,
                    "matched_card": "system", "matched_topic": "special",
                    "method": "special"}

        # 2. TF-IDF cosine similarity search
        best_entry, score = self._find_best_match(user_input)

        # 3. If confidence is good enough, return the answer
        if best_entry and score > 0.05:
            answer = self._personalize_answer(best_entry["answer"], best_entry)
            self.chat_history.append({"role": "bot", "content": answer})
            return {
                "answer": answer,
                "confidence": round(score, 3),
                "matched_card": best_entry["card"],
                "matched_topic": best_entry["topic"],
                "method": "tfidf"
            }

        # 4. Fallback response
        fallback = ("I'm not sure about that specific question. Try asking about:\n"
                    "• A specific card (e.g. 'tell me about HDFC Regalia')\n"
                    "• Comparisons (e.g. 'which card is best for travel?')\n"
                    "• Rewards/fees (e.g. 'what cashback does Axis Flipkart give?')\n"
                    "• Credit tips (e.g. 'how to improve my credit score?')")
        self.chat_history.append({"role": "bot", "content": fallback})
        return {"answer": fallback, "confidence": 0.0,
                "matched_card": "none", "matched_topic": "fallback",
                "method": "fallback"}

    def explain_tfidf(self, query: str):
        """
        Educational method — shows how TF-IDF works on a given query.
        Great for project demonstrations!
        """
        tokens = preprocess(query)
        tf = compute_tf(tokens)
        vector = compute_tfidf_vector(tokens, self.idf)

        print(f"\n{'='*55}")
        print(f"  TF-IDF Explanation for: '{query}'")
        print(f"{'='*55}")
        print(f"\n1. After preprocessing: {tokens}")
        print(f"\n2. Term Frequency (TF):")
        for w, v in sorted(tf.items(), key=lambda x: -x[1])[:5]:
            print(f"   '{w}': {v:.3f}")
        print(f"\n3. TF-IDF Scores (higher = more important):")
        top_tfidf = sorted(vector.items(), key=lambda x: -x[1])[:5]
        for w, v in top_tfidf:
            bar = "█" * int(v * 100)
            print(f"   '{w}': {v:.4f}  {bar}")
        print(f"\n4. These scores are compared to all KB entries")
        print(f"   using Cosine Similarity to find the best answer.")
        print(f"{'='*55}\n")


# ─── TERMINAL CHATBOT ────────────────────────────────────────────────────────

def run_terminal_chatbot():
    """Run the chatbot as an interactive terminal session."""
    print("\n" + "="*55)
    print("   CardSaathi — AI Credit Card Advisor (NLP Demo)")
    print("="*55)

    bot = CardSaathiNLP()

    # Demo: set a sample user context
    bot.set_user_context(
        profile={"monthly_salary": 45000, "credit_score": 730,
                 "travel_spend": 1000, "dining_spend": 3000,
                 "shopping_spend": 4000, "grocery_spend": 2000,
                 "entertainment_spend": 1500, "fuel_spend": 500,
                 "utilities_spend": 1000},
        recommended_cards=["HDFC Millennia", "SBI Cashback Card",
                           "Amazon Pay ICICI", "Axis Flipkart"]
    )

    print("Type your question below. Type 'explain <question>' to see")
    print("how TF-IDF works. Type 'quit' to exit.\n")

    SAMPLE_QUESTIONS = [
        "which card is best for online shopping?",
        "does hdfc millennia have lounge access?",
        "which card has no annual fee?",
        "how to improve my credit score?",
        "what is credit utilization?",
    ]
    print("💡 Sample questions to try:")
    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"   {i}. {q}")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("CardSaathi: Goodbye! Happy spending wisely! 💳")
            break

        # Educational mode
        if user_input.lower().startswith("explain "):
            query = user_input[8:]
            bot.explain_tfidf(query)
            result = bot.respond(query)
            print(f"\nCardSaathi: {result['answer']}")
            print(f"[Confidence: {result['confidence']:.3f} | "
                  f"Method: {result['method']} | "
                  f"Matched: {result['matched_card']} / {result['matched_topic']}]\n")
        else:
            result = bot.respond(user_input)
            print(f"\nCardSaathi: {result['answer']}\n")


if __name__ == "__main__":
    run_terminal_chatbot()
