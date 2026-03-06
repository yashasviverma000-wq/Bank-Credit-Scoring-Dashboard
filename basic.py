import pandas as pd
import numpy as np

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- avoid tkinter backend issues
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from io import BytesIO
import base64

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# =====================================================
# NLTK (download only if missing)
# =====================================================
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# =====================================================
# LOAD CREDIT DATA (EXACT FILE NAME)
# =====================================================
df = pd.read_excel("Final data converted from num to cat.xlsx")

# =====================================================
# HANDLE MISSING VALUES
# =====================================================
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# =====================================================
# TARGET COLUMN
# =====================================================
target = "bad"

# =====================================================
# ENCODE CATEGORICAL VARIABLES
# =====================================================
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =====================================================
# SCALING (FOR LR & SVM)
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# MODELS (ALL 6)
# =====================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True, class_weight="balanced"),  # fixes SVM minority issue
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42),
}

roc_data = {}
comparison = []

for name, model in models.items():
    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    roc_data[name] = (fpr, tpr, roc_auc)

    rep = classification_report(y_test, preds, output_dict=True)
    comparison.append({
        "Model": name,
        "AUC": round(roc_auc, 3),
        "Accuracy": round(rep["accuracy"], 3),
        "Precision": round(rep["1"]["precision"], 3),
        "Recall": round(rep["1"]["recall"], 3),
        "F1-score": round(rep["1"]["f1-score"], 3),
    })

comparison_df = pd.DataFrame(comparison)

# =====================================================
# SENTIMENT + WORDCLOUD (EXACTLY FROM TXT FILE)
# =====================================================
with open("NPS Survey.txt", "r", encoding="utf-8") as f:
    reviews = [line.strip() for line in f.readlines() if line.strip()]

sia = SentimentIntensityAnalyzer()
sentiment_df = pd.DataFrame({"Feedback": reviews})
sentiment_df["Score"] = sentiment_df["Feedback"].apply(lambda x: sia.polarity_scores(x)["compound"])
sentiment_df["Sentiment"] = sentiment_df["Score"].apply(
    lambda s: "Positive" if s >= 0.05 else ("Negative" if s <= -0.05 else "Neutral")
)

# build wordcloud ONCE (not from Excel!)
wc_text = " ".join(sentiment_df["Feedback"].astype(str))
wc = WordCloud(width=1000, height=500, background_color="white").generate(wc_text)

buf = BytesIO()
plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig(buf, format="png", bbox_inches="tight")
plt.close()

buf.seek(0)
wc_base64 = base64.b64encode(buf.read()).decode()

# =====================================================
# DASH APP
# =====================================================
app = dash.Dash(__name__)
app.title = "Bank Credit Scoring Dashboard"

app.layout = html.Div(style={"backgroundColor": "white", "padding": "20px"}, children=[
    html.H1("BANK CREDIT SCORING DASHBOARD", style={"textAlign": "center"}),

    dcc.Tabs([

        # ================= TAB 1 : EDA =================
        dcc.Tab(label="EDA", children=[
            html.Br(),
            dcc.Dropdown(
                id="eda-choice",
                options=[
                    {"label": "Target Distribution (bad)", "value": "target"},
                    {"label": "Debt-to-Income vs bad (boxplot)", "value": "debtinc"},
                    {"label": "Credit Debt vs bad (boxplot)", "value": "creddebt"},
                    {"label": "Other Debt vs bad (boxplot)", "value": "othdebt"},
                    {"label": "Age Distribution", "value": "age"},
                ],
                value="target"
            ),
            dcc.Graph(id="eda-graph")
        ]),

        # ================= TAB 2 : MODEL ROC =================
        dcc.Tab(label="MODELS", children=[
            html.Br(),
            dcc.Dropdown(
                id="model-choice",
                options=[{"label": m, "value": m} for m in models],
                value="Logistic Regression"
            ),
            dcc.Graph(id="roc-graph")
        ]),

        # ================= TAB 3 : MODEL COMPARISON =================
        dcc.Tab(label="MODEL COMPARISON", children=[
            html.Br(),
            dcc.Graph(
                figure=px.bar(
                    comparison_df,
                    x="Model",
                    y=["AUC", "Accuracy", "Precision", "Recall", "F1-score"],
                    barmode="group",
                    title="Model Performance Comparison"
                )
            )
        ]),

        # ================= TAB 4 : WORDCLOUD & SENTIMENT =================
        dcc.Tab(label="WORDCLOUD & SENTIMENT", children=[
            html.Br(),
            html.H3("WordCloud (from NPS Survey.txt)", style={"textAlign": "center"}),
            html.Div(
                html.Img(src="data:image/png;base64," + wc_base64, style={"width": "90%"}),
                style={"textAlign": "center"}
            ),
            html.Br(),
            dcc.Graph(
                figure=px.histogram(
                    sentiment_df,
                    x="Sentiment",
                    color="Sentiment",
                    title="Sentiment Distribution (VADER)"
                )
            )
        ]),
    ])
])

# =====================================================
# CALLBACKS
# =====================================================
@app.callback(Output("eda-graph", "figure"),
              Input("eda-choice", "value"))
def update_eda(choice):
    if choice == "target":
        return px.histogram(df, x=target, title="Target Distribution (bad)")
    if choice in ["debtinc", "creddebt", "othdebt"]:
        return px.box(df, x=target, y=choice, title=f"{choice} vs bad")
    if choice == "age":
        return px.histogram(df, x="age", title="Age Distribution")
    return px.histogram(df, x=target, title="Target Distribution (bad)")


@app.callback(Output("roc-graph", "figure"),
              Input("model-choice", "value"))
def update_roc(model_name):
    fpr, tpr, roc_auc = roc_data[model_name]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash")))
    fig.update_layout(
        title=f"ROC Curve - {model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )
    return fig


# =====================================================
# RUN LOCALHOST
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
