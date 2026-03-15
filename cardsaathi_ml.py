

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─── CARD LABELS ─────────────────────────────────────────────────────────────
CARD_LABELS = [
    "Kotak 811",           # 0 — beginners, low salary
    "SBI SimplyCLICK",     # 1 — entry, online shopping
    "Axis Flipkart",       # 2 — entry, shopping + groceries
    "Amazon Pay ICICI",    # 3 — mid, online + utilities
    "SBI Cashback Card",   # 4 — mid, general cashback
    "HDFC Millennia",      # 5 — mid, dining + entertainment
    "ICICI Coral",         # 6 — mid, dining + movies
    "HDFC Regalia",        # 7 — premium, travel
    "IndusInd Legend",     # 8 — premium, travel + dining
    "Axis Magnus",         # 9 — super premium, travel + luxury
]

# ─── SYNTHETIC DATA GENERATION ───────────────────────────────────────────────
def assign_card(row):
    s  = row["monthly_salary"]
    cs = row["credit_score"]
    tr = row["travel_spend"]
    dn = row["dining_spend"]
    sh = row["shopping_spend"]
    gr = row["grocery_spend"]
    en = row["entertainment_spend"]
    fu = row["fuel_spend"]
    ut = row["utilities_spend"]

    total = tr+dn+sh+gr+en+fu+ut

    # Super premium
    if s >= 300000 and cs >= 780:
        if tr > 8000:  return 9   # Axis Magnus

    # Premium
    if s >= 100000 and cs >= 750:
        if tr > 5000:  return 7   # HDFC Regalia
        if dn > 4000:  return 8   # IndusInd Legend
        return 7

    # Mid range — dining/entertainment
    if s >= 35000 and cs >= 720:
        if dn > 3000 or en > 2000: return 5   # HDFC Millennia
        if sh > 4000:              return 3   # Amazon Pay ICICI

    # Mid range — cashback
    if s >= 30000 and cs >= 700:
        if sh > 3000 or gr > 2000: return 4   # SBI Cashback
        if dn > 2000:              return 6   # ICICI Coral

    # Entry — shopping
    if s >= 25000 and cs >= 720:
        if sh > 2000 or ut > 1000: return 3   # Amazon Pay ICICI

    if s >= 20000 and cs >= 700:
        if sh > 2000 or en > 1000: return 1   # SBI SimplyCLICK

    if s >= 15000 and cs >= 680:
        if sh > 1500 or gr > 1500: return 2   # Axis Flipkart

    # Beginner / anyone
    return 0   # Kotak 811

def generate_dataset(n=2000):
    rows = []
    for _ in range(n):
        salary = int(np.random.choice([
            np.random.randint(10000, 25000),
            np.random.randint(25000, 60000),
            np.random.randint(60000, 150000),
            np.random.randint(150000, 500000),
        ], p=[0.25, 0.35, 0.25, 0.15]))

        credit_score = int(np.clip(np.random.normal(720, 60), 500, 900))

        # Spend correlates loosely with salary
        base = min(salary * 0.4, 30000)
        row = {
            "monthly_salary":    salary,
            "credit_score":      credit_score,
            "travel_spend":      int(max(0, np.random.exponential(base * 0.15))),
            "dining_spend":      int(max(0, np.random.exponential(base * 0.20))),
            "shopping_spend":    int(max(0, np.random.exponential(base * 0.25))),
            "grocery_spend":     int(max(0, np.random.exponential(base * 0.20))),
            "entertainment_spend": int(max(0, np.random.exponential(base * 0.10))),
            "fuel_spend":        int(max(0, np.random.exponential(base * 0.08))),
            "utilities_spend":   int(max(0, np.random.exponential(base * 0.10))),
        }
        row["card_recommended"] = assign_card(row)
        rows.append(row)
    return pd.DataFrame(rows)

print("=" * 55)
print("   CardSaathi — ML Credit Card Recommendation Model")
print("=" * 55)

print("\n📦 Step 1: Generating training dataset...")
df = generate_dataset(2000)
df["card_name"] = df["card_recommended"].map(dict(enumerate(CARD_LABELS)))
print(f"   ✅ {len(df)} samples generated")
print(f"\n   Card distribution:")
dist = df["card_name"].value_counts()
for name, count in dist.items():
    bar = "█" * (count // 20)
    print(f"   {name:<25} {bar} {count}")

# ─── FEATURES & LABELS ───────────────────────────────────────────────────────
FEATURES = ["monthly_salary","credit_score","travel_spend","dining_spend",
            "shopping_spend","grocery_spend","entertainment_spend",
            "fuel_spend","utilities_spend"]

X = df[FEATURES]
y = df["card_recommended"]

# ─── TRAIN / TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Step 2: Train/Test Split")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing samples  : {len(X_test)}")

# ─── TRAIN RANDOM FOREST ─────────────────────────────────────────────────────
print("\n🌲 Step 3: Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced"
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"   ✅ Random Forest Accuracy: {rf_acc*100:.1f}%")

# ─── TRAIN DECISION TREE (for visualization) ─────────────────────────────────
print("\n🌳 Step 4: Training Decision Tree (for visualization)...")
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced")
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)
print(f"   ✅ Decision Tree Accuracy : {dt_acc*100:.1f}%")

print("\n📋 Classification Report (Random Forest):")
print(classification_report(y_test, rf_preds,
      target_names=[CARD_LABELS[i] for i in sorted(y.unique())],
      zero_division=0))

# ─── VISUALIZATIONS ──────────────────────────────────────────────────────────
plt.style.use("dark_background")
PURPLE = "#a78bfa"
TEAL   = "#2dd4bf"
AMBER  = "#fbbf24"
PINK   = "#f472b6"
BG     = "#0a0f1e"
CARD   = "#111827"

PALETTE = ["#a78bfa","#34d399","#60a5fa","#f472b6","#fbbf24",
           "#f87171","#818cf8","#2dd4bf","#fb923c","#a3e635"]

print("\n🎨 Step 5: Generating visualizations...")

# ── VIZ 1: Dataset Overview ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
fig.suptitle("CardSaathi — Dataset Overview", fontsize=16, fontweight="bold",
             color="white", y=0.98)

# 1a: Card distribution
ax = axes[0, 0]
ax.set_facecolor(CARD)
counts = df["card_name"].value_counts()
bars = ax.barh(counts.index, counts.values, color=PALETTE[:len(counts)], height=0.6)
ax.set_title("Training Data — Card Distribution", color="white", pad=8)
ax.set_xlabel("Number of samples", color="#94a3b8")
ax.tick_params(colors="white", labelsize=8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2,
            str(val), va="center", color="white", fontsize=8)
ax.spines[:].set_visible(False)
ax.grid(axis="x", alpha=0.1)

# 1b: Salary distribution
ax = axes[0, 1]
ax.set_facecolor(CARD)
ax.hist(df["monthly_salary"]/1000, bins=30, color=PURPLE, alpha=0.8, edgecolor="none")
ax.set_title("Monthly Salary Distribution", color="white", pad=8)
ax.set_xlabel("Salary (₹ thousands)", color="#94a3b8")
ax.set_ylabel("Count", color="#94a3b8")
ax.tick_params(colors="white")
ax.spines[:].set_visible(False)
ax.grid(axis="y", alpha=0.1)

# 1c: Credit score distribution
ax = axes[1, 0]
ax.set_facecolor(CARD)
ax.hist(df["credit_score"], bins=30, color=TEAL, alpha=0.8, edgecolor="none")
ax.axvline(650, color=AMBER, linestyle="--", alpha=0.7, label="Fair (650)")
ax.axvline(700, color=PINK,  linestyle="--", alpha=0.7, label="Good (700)")
ax.axvline(750, color="#4ade80", linestyle="--", alpha=0.7, label="Excellent (750)")
ax.set_title("Credit Score Distribution", color="white", pad=8)
ax.set_xlabel("Credit Score", color="#94a3b8")
ax.set_ylabel("Count", color="#94a3b8")
ax.tick_params(colors="white")
ax.legend(fontsize=8, labelcolor="white", facecolor=BG, edgecolor="none")
ax.spines[:].set_visible(False)
ax.grid(axis="y", alpha=0.1)

# 1d: Average spend per category
ax = axes[1, 1]
ax.set_facecolor(CARD)
spend_cols = ["travel_spend","dining_spend","shopping_spend","grocery_spend",
              "entertainment_spend","fuel_spend","utilities_spend"]
labels = ["Travel","Dining","Shopping","Grocery","Entertain","Fuel","Utilities"]
means = [df[c].mean() for c in spend_cols]
bars = ax.bar(labels, means, color=PALETTE[:7], width=0.6)
ax.set_title("Avg Monthly Spend per Category", color="white", pad=8)
ax.set_ylabel("₹ Amount", color="#94a3b8")
ax.tick_params(colors="white", labelsize=8)
for bar, val in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
            f"₹{int(val):,}", ha="center", color="white", fontsize=7)
ax.spines[:].set_visible(False)
ax.grid(axis="y", alpha=0.1)

plt.tight_layout()
plt.savefig("/home/claude/viz1_dataset_overview.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
plt.close()
print("   ✅ viz1_dataset_overview.png")

# ── VIZ 2: Confusion Matrix ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
fig.suptitle("CardSaathi — Model Evaluation", fontsize=16,
             fontweight="bold", color="white", y=1.01)

for ax, preds, title, acc in [
    (axes[0], rf_preds, f"Random Forest  (Accuracy: {rf_acc*100:.1f}%)", rf_acc),
    (axes[1], dt_preds, f"Decision Tree  (Accuracy: {dt_acc*100:.1f}%)", dt_acc),
]:
    ax.set_facecolor(CARD)
    cm = confusion_matrix(y_test, preds)
    present = sorted(y_test.unique())
    tick_labels = [CARD_LABELS[i] for i in present]

    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                cmap="Purples", linewidths=0.4,
                xticklabels=tick_labels, yticklabels=tick_labels,
                annot_kws={"size": 8})
    ax.set_title(title, color="white", pad=10, fontsize=11)
    ax.set_xlabel("Predicted", color="#94a3b8")
    ax.set_ylabel("Actual", color="#94a3b8")
    ax.tick_params(colors="white", labelsize=7)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig("/home/claude/viz2_confusion_matrix.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✅ viz2_confusion_matrix.png")

# ── VIZ 3: Feature Importance ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
ax.set_facecolor(CARD)

importances = rf_model.feature_importances_
feat_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=True)

nice_names = {
    "monthly_salary":"Monthly Salary", "credit_score":"Credit Score",
    "travel_spend":"Travel Spend", "dining_spend":"Dining Spend",
    "shopping_spend":"Shopping Spend", "grocery_spend":"Grocery Spend",
    "entertainment_spend":"Entertainment Spend", "fuel_spend":"Fuel Spend",
    "utilities_spend":"Utilities Spend"
}
feat_df["Feature"] = feat_df["Feature"].map(nice_names)

colors = [PURPLE if i >= len(feat_df)-3 else TEAL for i in range(len(feat_df))]
bars = ax.barh(feat_df["Feature"], feat_df["Importance"]*100,
               color=colors, height=0.55)

for bar, val in zip(bars, feat_df["Importance"]*100):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
            f"{val:.1f}%", va="center", color="white", fontsize=9)

ax.set_title("Feature Importance — What Drives the Recommendation?",
             color="white", fontsize=13, pad=12)
ax.set_xlabel("Importance (%)", color="#94a3b8")
ax.tick_params(colors="white", labelsize=10)
ax.spines[:].set_visible(False)
ax.grid(axis="x", alpha=0.1)

top3 = mpatches.Patch(color=PURPLE, label="Top 3 most important features")
rest = mpatches.Patch(color=TEAL,   label="Other features")
ax.legend(handles=[top3, rest], fontsize=9, labelcolor="white",
          facecolor=BG, edgecolor="none", loc="lower right")

plt.tight_layout()
plt.savefig("/home/claude/viz3_feature_importance.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✅ viz3_feature_importance.png")

# ── VIZ 4: Decision Tree ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 10), facecolor=BG)
ax.set_facecolor(BG)

present_classes = sorted(y_train.unique())
class_names_present = [CARD_LABELS[i] for i in present_classes]

plot_tree(dt_model, feature_names=FEATURES,
          class_names=class_names_present,
          filled=True, rounded=True, fontsize=7,
          ax=ax, impurity=False, proportion=False,
          max_depth=3)

ax.set_title("Decision Tree — How the Model Thinks (max depth 3)",
             color="white", fontsize=14, pad=12)

plt.tight_layout()
plt.savefig("/home/claude/viz4_decision_tree.png", dpi=130,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✅ viz4_decision_tree.png")

# ── VIZ 5: Model Comparison + Live Prediction Demo ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.suptitle("CardSaathi — Model Comparison & Live Predictions",
             fontsize=14, color="white", fontweight="bold")

# 5a: Accuracy comparison bar
ax = axes[0]
ax.set_facecolor(CARD)
models = ["Decision Tree", "Random Forest"]
accs   = [dt_acc*100, rf_acc*100]
bars   = ax.bar(models, accs, color=[TEAL, PURPLE], width=0.4)
ax.set_ylim(0, 110)
ax.set_title("Model Accuracy Comparison", color="white", pad=8)
ax.set_ylabel("Accuracy (%)", color="#94a3b8")
ax.tick_params(colors="white")
ax.spines[:].set_visible(False)
ax.grid(axis="y", alpha=0.1)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f"{val:.1f}%", ha="center", color="white", fontsize=12, fontweight="bold")

# 5b: Live prediction demo — 5 sample users
ax = axes[1]
ax.set_facecolor(CARD)
ax.set_title("Live Predictions — Sample User Profiles", color="white", pad=8)
ax.axis("off")

sample_users = [
    {"monthly_salary":18000,"credit_score":660,"travel_spend":500,"dining_spend":1000,"shopping_spend":2000,"grocery_spend":1500,"entertainment_spend":500,"fuel_spend":300,"utilities_spend":500},
    {"monthly_salary":40000,"credit_score":730,"travel_spend":1000,"dining_spend":3000,"shopping_spend":4000,"grocery_spend":2000,"entertainment_spend":1500,"fuel_spend":500,"utilities_spend":1000},
    {"monthly_salary":70000,"credit_score":760,"travel_spend":5000,"dining_spend":2000,"shopping_spend":3000,"grocery_spend":2000,"entertainment_spend":1000,"fuel_spend":1000,"utilities_spend":1500},
    {"monthly_salary":150000,"credit_score":800,"travel_spend":10000,"dining_spend":4000,"shopping_spend":5000,"grocery_spend":3000,"entertainment_spend":2000,"fuel_spend":1000,"utilities_spend":2000},
    {"monthly_salary":350000,"credit_score":820,"travel_spend":20000,"dining_spend":8000,"shopping_spend":10000,"grocery_spend":4000,"entertainment_spend":3000,"fuel_spend":2000,"utilities_spend":3000},
]

table_data = []
for u in sample_users:
    pred = rf_model.predict(pd.DataFrame([u]))[0]
    table_data.append([
        f"₹{u['monthly_salary']//1000}K",
        str(u["credit_score"]),
        CARD_LABELS[pred]
    ])

table = ax.table(
    cellText=table_data,
    colLabels=["Salary/mo", "Score", "Recommended Card"],
    loc="center", cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 2.2)

for (r, c), cell in table.get_celld().items():
    cell.set_facecolor(BG if r == 0 else CARD)
    cell.set_text_props(color="white" if r > 0 else AMBER)
    cell.set_edgecolor("#1e293b")
    if r > 0 and c == 2:
        cell.set_facecolor("#1a1040")
        cell.set_text_props(color=PURPLE)

plt.tight_layout()
plt.savefig("/home/claude/viz5_model_comparison.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✅ viz5_model_comparison.png")

# ── SAVE MODEL + DATA ─────────────────────────────────────────────────────────
import pickle, os
with open("/home/claude/rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
with open("/home/claude/dt_model.pkl", "wb") as f:
    pickle.dump(dt_model, f)
df.to_csv("/home/claude/cardsaathi_dataset.csv", index=False)
print("\n💾 Step 6: Saved model files")
print("   ✅ rf_model.pkl  (Random Forest)")
print("   ✅ dt_model.pkl  (Decision Tree)")
print("   ✅ cardsaathi_dataset.csv")

print(f"""
╔══════════════════════════════════════════════════╗
║         CardSaathi ML Model — Summary            ║
╠══════════════════════════════════════════════════╣
║  Training samples   : {len(X_train):<5}                      ║
║  Testing samples    : {len(X_test):<5}                      ║
║  Features used      : {len(FEATURES):<5}                      ║
║  Classes (cards)    : {len(CARD_LABELS):<5}                      ║
║  Decision Tree Acc  : {dt_acc*100:<5.1f}%                     ║
║  Random Forest Acc  : {rf_acc*100:<5.1f}%                     ║
╚══════════════════════════════════════════════════╝
""")
