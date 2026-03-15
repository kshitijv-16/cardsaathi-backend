"""
CardSaathi — Flask API Backend
================================
Connects the React UI to the Python ML model + NLP engine

Install requirements:
    pip install flask flask-cors scikit-learn pandas numpy

Run with:
    python app.py

API Endpoints:
    POST /recommend  → runs ML model, returns card recommendations
    POST /chat       → runs NLP engine, returns chatbot response
    GET  /health     → checks if server is running
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import re
import math

app = Flask(__name__)
CORS(app)  # Allow React (localhost:5173) to call this API

# ─── CARD DATA ────────────────────────────────────────────────────────────────
CARD_LABELS = [
    "Kotak 811", "SBI SimplyCLICK", "Axis Flipkart", "Amazon Pay ICICI",
    "SBI Cashback Card", "HDFC Millennia", "ICICI Coral",
    "HDFC Regalia", "IndusInd Legend", "Axis Magnus"
]

CARD_DATA = {
    "Kotak 811":         {"bank":"Kotak Bank",   "fee":0,     "minSal":15000,  "minScore":650, "cashback":False, "lounge":False, "loungeN":0,   "forex":3.5, "best":["Online Shopping","General"],       "gradient":["#EE3124","#FF6B35"], "network":"VISA"},
    "SBI SimplyCLICK":   {"bank":"SBI Card",     "fee":499,   "minSal":20000,  "minScore":700, "cashback":False, "lounge":False, "loungeN":0,   "forex":3.5, "best":["Online Shopping","Entertainment"],  "gradient":["#22409A","#4169C8"], "network":"VISA"},
    "Axis Flipkart":     {"bank":"Axis Bank",    "fee":500,   "minSal":15000,  "minScore":680, "cashback":True,  "lounge":True,  "loungeN":4,   "forex":3.5, "best":["Shopping","Groceries"],             "gradient":["#97144D","#C9184A"], "network":"MC"},
    "Amazon Pay ICICI":  {"bank":"ICICI Bank",   "fee":0,     "minSal":25000,  "minScore":720, "cashback":True,  "lounge":False, "loungeN":0,   "forex":3.5, "best":["Amazon","Utilities"],               "gradient":["#FF9900","#FF6600"], "network":"VISA"},
    "SBI Cashback Card": {"bank":"SBI Card",     "fee":999,   "minSal":30000,  "minScore":700, "cashback":True,  "lounge":False, "loungeN":0,   "forex":3.5, "best":["Online Shopping","General"],        "gradient":["#1565C0","#1976D2"], "network":"VISA"},
    "HDFC Millennia":    {"bank":"HDFC Bank",    "fee":1000,  "minSal":35000,  "minScore":720, "cashback":True,  "lounge":True,  "loungeN":8,   "forex":3.5, "best":["Dining","Shopping"],                "gradient":["#004C97","#0D47A1"], "network":"MC"},
    "ICICI Coral":       {"bank":"ICICI Bank",   "fee":500,   "minSal":20000,  "minScore":680, "cashback":False, "lounge":True,  "loungeN":2,   "forex":3.5, "best":["Dining","Movies"],                  "gradient":["#B02A30","#E53935"], "network":"VISA"},
    "HDFC Regalia":      {"bank":"HDFC Bank",    "fee":2500,  "minSal":100000, "minScore":750, "cashback":False, "lounge":True,  "loungeN":32,  "forex":2.0, "best":["Travel","Dining"],                  "gradient":["#1A237E","#283593"], "network":"VISA"},
    "IndusInd Legend":   {"bank":"IndusInd",     "fee":3999,  "minSal":100000, "minScore":750, "cashback":False, "lounge":True,  "loungeN":6,   "forex":2.5, "best":["Travel","Dining"],                  "gradient":["#4A148C","#6A1B9A"], "network":"MC"},
    "Axis Magnus":       {"bank":"Axis Bank",    "fee":12500, "minSal":300000, "minScore":780, "cashback":False, "lounge":True,  "loungeN":999, "forex":2.0, "best":["Travel","Luxury"],                  "gradient":["#212121","#37474F"], "network":"MC"},
}

FEATURES = [
    "monthly_salary", "credit_score", "travel_spend", "dining_spend",
    "shopping_spend", "grocery_spend", "entertainment_spend",
    "fuel_spend", "utilities_spend"
]

# ─── LOAD OR TRAIN ML MODEL ───────────────────────────────────────────────────
def load_or_train_model():
    if os.path.exists("rf_model.pkl"):
        print("✅ Loaded existing ML model (rf_model.pkl)")
        with open("rf_model.pkl", "rb") as f:
            return pickle.load(f)

    print("🌲 Training new Random Forest model...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    np.random.seed(42)

    def assign_card(row):
        s, cs = row["monthly_salary"], row["credit_score"]
        tr, dn = row["travel_spend"], row["dining_spend"]
        sh, gr = row["shopping_spend"], row["grocery_spend"]
        en = row["entertainment_spend"]
        if s >= 300000 and cs >= 780 and tr > 8000: return 9
        if s >= 100000 and cs >= 750:
            return 7 if tr > 5000 else 8 if dn > 4000 else 7
        if s >= 35000 and cs >= 720:
            return 5 if (dn > 3000 or en > 2000) else 3
        if s >= 30000 and cs >= 700:
            return 4 if (sh > 3000 or gr > 2000) else 6
        if s >= 25000 and cs >= 720: return 3
        if s >= 20000 and cs >= 700: return 1
        if s >= 15000 and cs >= 680: return 2
        return 0

    rows = []
    for _ in range(2000):
        salary = int(np.random.choice(
            [np.random.randint(10000,25000), np.random.randint(25000,60000),
             np.random.randint(60000,150000), np.random.randint(150000,500000)],
            p=[0.25, 0.35, 0.25, 0.15]))
        cs = int(np.clip(np.random.normal(720, 60), 500, 900))
        base = min(salary * 0.4, 30000)
        row = {
            "monthly_salary":       salary, "credit_score": cs,
            "travel_spend":         int(max(0, np.random.exponential(base*0.15))),
            "dining_spend":         int(max(0, np.random.exponential(base*0.20))),
            "shopping_spend":       int(max(0, np.random.exponential(base*0.25))),
            "grocery_spend":        int(max(0, np.random.exponential(base*0.20))),
            "entertainment_spend":  int(max(0, np.random.exponential(base*0.10))),
            "fuel_spend":           int(max(0, np.random.exponential(base*0.08))),
            "utilities_spend":      int(max(0, np.random.exponential(base*0.10))),
        }
        row["card_recommended"] = assign_card(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    X = df[FEATURES]; y = df["card_recommended"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    with open("rf_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Model trained and saved!")
    return model

MODEL = load_or_train_model()

# ─── NLP KNOWLEDGE BASE ───────────────────────────────────────────────────────
KB = [
    {"card":"Kotak 811","q":"kotak 811 free beginner annual fee zero lifetime",
     "a":"💳 **Kotak 811** is **lifetime FREE** — zero annual fee ever! Perfect for beginners. Needs only ₹15K/month salary and 650+ credit score. Earn 2 points per ₹100 spent."},
    {"card":"Amazon Pay ICICI","q":"amazon icici cashback free prime lifetime",
     "a":"💳 **Amazon Pay ICICI** is **lifetime FREE**. Gives **5% cashback on Amazon** for Prime members (3% non-Prime), 2% on bills and groceries. Cashback auto-credited to Amazon Pay balance."},
    {"card":"HDFC Millennia","q":"hdfc millennia cashback swiggy zomato dining lounge shopping",
     "a":"💳 **HDFC Millennia** gives **5% cashback** on Swiggy, Zomato, Amazon, Flipkart & BigBasket. Annual fee ₹1,000 (waived on ₹1L spend). **8 free lounge visits/year**. Best for ₹35K+ earners."},
    {"card":"HDFC Regalia","q":"hdfc regalia travel lounge forex premium international points",
     "a":"💳 **HDFC Regalia** — premium travel card. **32 lounge visits/year** via Priority Pass. 12x points on travel, **2% forex** for international use. Travel insurance ₹1 Crore. Annual fee ₹2,500."},
    {"card":"SBI SimplyCLICK","q":"sbi simplyclick online shopping flipkart amazon rewards points",
     "a":"💳 **SBI SimplyCLICK** gives **10x reward points** on Amazon, Flipkart, Myntra and BookMyShow. Annual fee ₹499 (waived on ₹1L). Effectively 2.5% back on online shopping."},
    {"card":"Axis Flipkart","q":"axis flipkart cashback grocery shopping lounge",
     "a":"💳 **Axis Flipkart** gives **5% on Flipkart & Myntra**, **4% on grocery apps**. Annual fee ₹500. Includes 4 lounge visits/year."},
    {"card":"SBI Cashback Card","q":"sbi cashback flat online all transactions auto credit",
     "a":"💳 **SBI Cashback Card** gives **flat 5% on ALL online transactions** — no restrictions! Annual fee ₹999 (waived on ₹2L). Cashback auto-credited monthly."},
    {"card":"ICICI Coral","q":"icici coral movies bookmyshow dining buy one get one entertainment",
     "a":"💳 **ICICI Coral** gives **Buy 1 Get 1 free movie tickets** on BookMyShow (2/month)! 6x points on dining. Annual fee ₹500. Saves ₹7,200/year on movies alone."},
    {"card":"Axis Magnus","q":"axis magnus unlimited lounge luxury super premium travel golf",
     "a":"💳 **Axis Magnus** — super premium. **Unlimited lounge access**, 35x travel points, 2% forex. Annual fee ₹12,500. Needs ₹3L+/month salary."},
    {"card":"IndusInd Legend","q":"indusind legend travel dining golf lounge premium",
     "a":"💳 **IndusInd Legend** gives 3x points on travel, dining & entertainment. Annual fee ₹3,999. 6 lounge visits/year, golf privileges, dining at 900+ restaurants."},
    {"card":"comparison","q":"best card travel international airport lounge forex abroad",
     "a":"✈️ **Best for travel:**\n1. **Axis Magnus** — unlimited lounge, 35x points, 2% forex (₹3L+ salary)\n2. **HDFC Regalia** — 32 lounge visits/yr, 2% forex (₹1L+ salary)\n3. **HDFC Millennia** — 8 lounge visits/yr (₹35K+ salary)"},
    {"card":"comparison","q":"best card online shopping amazon flipkart myntra cashback rewards",
     "a":"🛍️ **Best for online shopping:**\n1. **Amazon Pay ICICI** — 5% on Amazon, lifetime free\n2. **HDFC Millennia** — 5% on Amazon+Flipkart+Zomato+Swiggy\n3. **SBI Cashback** — flat 5% on ALL online transactions"},
    {"card":"comparison","q":"best card dining food delivery swiggy zomato restaurant",
     "a":"🍽️ **Best for dining:**\n1. **HDFC Millennia** — 5% on Swiggy & Zomato\n2. **ICICI Coral** — 6x points on dining + free movies\n3. **HDFC Regalia** — 8x points on restaurants"},
    {"card":"comparison","q":"best card no annual fee lifetime free zero joining",
     "a":"🆓 **Lifetime FREE cards:**\n1. **Amazon Pay ICICI** — 5% Amazon cashback, best free card\n2. **Kotak 811** — easiest to get, perfect for beginners (₹15K salary, 650 score)"},
    {"card":"comparison","q":"best card beginner first time student low salary",
     "a":"👶 **Best for beginners:**\n1. **Kotak 811** — free, ₹15K+ salary, 650+ score\n2. **Amazon Pay ICICI** — free, great cashback, 720+ score\nAlways pay the full bill every month!"},
    {"card":"comparison","q":"best card most cashback rewards earn money back",
     "a":"💰 **Best cashback cards:**\n1. **Amazon Pay ICICI** — 5% Amazon, zero fee\n2. **HDFC Millennia** — 5% on Swiggy+Amazon+Flipkart\n3. **SBI Cashback** — flat 5% all online\n4. **Axis Flipkart** — 5% Flipkart, 4% groceries"},
    {"card":"comparison","q":"card low salary minimum income credit score poor",
     "a":"💼 **Cards for lower salary/score:**\n1. **Kotak 811** — ₹15K/month, score 650+\n2. **Axis Flipkart** — ₹15K/month, score 680+\n3. **SBI SimplyCLICK** — ₹20K/month, score 700+"},
    {"card":"literacy","q":"credit score cibil what is why important number",
     "a":"📊 **Credit Score (300-900):**\n🟢 750+ = Excellent\n🟡 700-750 = Good\n🟠 650-700 = Fair\n🔴 Below 650 = Poor\n\nBased on: payment history (35%), utilization (30%), credit age (15%), mix (10%), inquiries (10%)."},
    {"card":"literacy","q":"how to improve credit score tips increase build",
     "a":"📈 **How to improve your score:**\n1. Always pay **full bill on time**\n2. Keep utilization **below 30%**\n3. Don't apply for multiple cards at once\n4. Keep old cards active\n5. Use **Kotak 811** to start building history"},
    {"card":"literacy","q":"credit utilization ratio what how much limit spending",
     "a":"📉 **Credit Utilization** = Spent ÷ Limit × 100\n\nExample: Limit ₹1,00,000, spent ₹30,000 = **30% utilization**\nKeep it **below 30%**. Above 50% hurts your score badly."},
    {"card":"literacy","q":"minimum due pay full bill interest trap debt",
     "a":"⚠️ **ALWAYS pay the full bill!**\n\nMinimum due is a trap — remaining balance attracts **3-3.5% per MONTH (36-42% annually)**.\n\nExample: ₹10,000 bill → pay ₹500 min → ₹9,500 costs ₹332/month in interest!"},
    {"card":"literacy","q":"maximize rewards points cashback tips smart use card",
     "a":"💡 **Smart tips:**\n1. Use right card per category\n2. Pay all bills via credit card\n3. Hit annual spend target for fee waiver\n4. Redeem points before expiry\n5. Never overspend just for rewards"},
    {"card":"literacy","q":"forex markup international travel abroad foreign currency transaction",
     "a":"🌍 **Forex Markup** = fee on foreign currency transactions.\nMost cards: **3.5%**. Premium cards: **2%**.\nOn ₹1 Lakh abroad, 2% vs 3.5% saves you ₹1,500!"},
    {"card":"literacy","q":"airport lounge access what is free credit card visit",
     "a":"🛫 **Airport Lounge Access** = enter premium lounges FREE with your card.\nFree food, drinks, Wi-Fi, comfortable seating. Worth ₹1,000-2,000 per visit.\nEntry: 2-4/yr · Mid-range: 6-8/yr · Super-premium: Unlimited"},
]

# ─── TF-IDF + COSINE SIMILARITY ──────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    stopwords = {"i","me","my","is","it","the","a","an","of","for","in","on","to",
                 "and","or","what","which","how","does","do","can","tell","about",
                 "give","get","with","that","this","are","was","be","have","has"}
    return [w for w in text.split() if w not in stopwords and len(w) > 2]

def tfidf_cosine_match(query):
    query_words = preprocess(query)
    if not query_words:
        return -1, 0.0

    all_docs = [preprocess(e["q"]) for e in KB]
    N = len(all_docs)

    def idf(w):
        count = sum(1 for doc in all_docs if w in doc)
        return math.log(N / (1 + count))

    def vec(words):
        tf = {}
        for w in words: tf[w] = tf.get(w, 0) + 1
        total = len(words) or 1
        return {w: (c/total)*idf(w) for w, c in tf.items()}

    def cosine(v1, v2):
        common = set(v1) & set(v2)
        if not common: return 0.0
        dot = sum(v1[w]*v2[w] for w in common)
        n1 = math.sqrt(sum(x**2 for x in v1.values()))
        n2 = math.sqrt(sum(x**2 for x in v2.values()))
        return dot/(n1*n2) if n1 and n2 else 0.0

    qvec = vec(query_words)
    scores = [cosine(qvec, vec(doc)) for doc in all_docs]
    best = int(np.argmax(scores))
    return best, scores[best]

# ─── API ROUTES ───────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — React calls this to verify backend is running."""
    return jsonify({"status": "ok", "message": "CardSaathi API is running! 🚀"})


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Runs the ML model and returns card recommendations.

    Request body (JSON):
    {
        "salary": 50000,
        "score": 720,
        "travel": 2000,
        "dining": 2000,
        "shopping": 3000,
        "grocery": 2000,
        "entertainment": 1000,
        "fuel": 1000,
        "utilities": 1000,
        "wantsLounge": false,
        "wantsFree": false,
        "wantsCashback": false
    }
    """
    try:
        data = request.get_json()

        # Build feature row for ML model
        row = {
            "monthly_salary":       data.get("salary", 50000),
            "credit_score":         data.get("score", 720),
            "travel_spend":         data.get("travel", 0),
            "dining_spend":         data.get("dining", 0),
            "shopping_spend":       data.get("shopping", 0),
            "grocery_spend":        data.get("grocery", 0),
            "entertainment_spend":  data.get("entertainment", 0),
            "fuel_spend":           data.get("fuel", 0),
            "utilities_spend":      data.get("utilities", 0),
        }

        df_input = pd.DataFrame([row])[FEATURES]

        # Get ML model probabilities
        probas  = MODEL.predict_proba(df_input)[0]
        classes = MODEL.classes_

        # Rank by probability
        ranked = sorted(zip(classes, probas), key=lambda x: -x[1])

        # Filter eligible + build response
        results = []
        salary = row["monthly_salary"]
        score  = row["credit_score"]

        for cls, prob in ranked:
            name = CARD_LABELS[cls]
            card = CARD_DATA[name]

            # Eligibility check
            if salary < card["minSal"] or score < card["minScore"]:
                continue

            # Preference bonus
            pref_score = 0
            if data.get("wantsLounge") and card["lounge"]:     pref_score += 20
            if data.get("wantsFree")   and card["fee"] == 0:   pref_score += 25
            if data.get("wantsCashback") and card["cashback"]: pref_score += 15

            match_score = min(int(prob * 100) + pref_score, 99)

            results.append({
                "id":         name.lower().replace(" ","_"),
                "name":       name,
                "bank":       card["bank"],
                "fee":        card["fee"],
                "minSal":     card["minSal"],
                "minScore":   card["minScore"],
                "cashback":   card["cashback"],
                "lounge":     card["lounge"],
                "loungeN":    card["loungeN"],
                "forex":      card["forex"],
                "best":       card["best"],
                "gradient":   card["gradient"],
                "network":    card["network"],
                "matchScore": match_score,
                "mlProb":     round(float(prob) * 100, 1),
            })

            if len(results) == 5:
                break

        return jsonify({
            "success":  True,
            "results":  results,
            "topCard":  results[0]["name"] if results else None,
            "eligible": len(results),
            "total":    len(CARD_LABELS),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Runs the NLP engine and returns a chatbot response.

    Request body (JSON):
    {
        "message": "which card is best for travel?",
        "topCard": "HDFC Regalia",
        "recommendedCards": ["HDFC Regalia", "HDFC Millennia", "Axis Flipkart"]
    }
    """
    try:
        data    = request.get_json()
        message = data.get("message", "").strip()
        top     = data.get("topCard", "")
        recs    = data.get("recommendedCards", [])
        lower   = message.lower()

        # Special responses
        if re.match(r"^(hi|hello|hey|namaste)", lower):
            reply = (f"👋 Hey! I'm **CardSaathi AI**. Your top recommended card is **{top}**. Ask me anything!"
                     if top else "👋 Hey! I'm **CardSaathi AI**. Ask me anything about credit cards!")
            return jsonify({"success": True, "reply": reply, "method": "greeting"})

        if "thank" in lower:
            return jsonify({"success": True, "reply": "😊 You're welcome! Happy to help anytime.", "method": "thanks"})

        if any(w in lower for w in ["my recommendation","best for me","suggest me","which card for me"]):
            if recs:
                lines = "\n".join([f"{i+1}. **{c}**" for i,c in enumerate(recs[:3])])
                reply = f"🎯 Your top picks:\n{lines}\n\n**{recs[0]}** is your best match!"
            else:
                reply = "Please complete your profile first to get personalized recommendations."
            return jsonify({"success": True, "reply": reply, "method": "recs"})

        # TF-IDF + Cosine Similarity
        best_idx, score = tfidf_cosine_match(message)

        if score > 0.1:
            entry  = KB[best_idx]
            answer = entry["a"]

            # Personalize answer
            if top and entry["card"] not in ["comparison","literacy"]:
                if entry["card"] == top:
                    answer += f"\n\n💡 Great news — **{top}** is your #1 recommended card!"
                elif entry["card"] in recs:
                    rank = recs.index(entry["card"]) + 1
                    answer += f"\n\n💡 **{entry['card']}** is ranked **#{rank}** in your recommendations."
                elif top:
                    answer += f"\n\n💡 Based on your profile, your top pick is **{top}**."

            return jsonify({
                "success":      True,
                "reply":        answer,
                "method":       "tfidf",
                "confidence":   round(score, 3),
                "matchedCard":  entry["card"],
            })

        # Fallback
        fallback = ("🤔 Try asking:\n"
                    "- *Which card is best for travel?*\n"
                    "- *Which card has no annual fee?*\n"
                    "- *How to improve my credit score?*\n"
                    "- *What is credit utilization?*")
        return jsonify({"success": True, "reply": fallback, "method": "fallback"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─── RUN SERVER ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  CardSaathi Flask API")
    print("="*50)
    print("  ML Model:  Random Forest (rf_model.pkl)")
    print("  NLP:       TF-IDF + Cosine Similarity")
    print("  Endpoints: /health  /recommend  /chat")
    print("  Running at: http://localhost:5000")
    print("="*50 + "\n")
app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
