import pandas as pd
import numpy as np

import dash
from dash import dcc, html, Input, Output, dash_table
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
# LOAD CREDIT DATA
# =====================================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_excel(os.path.join(BASE_DIR, "Final data converted from num to cat.xlsx"))

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
# MODELS (ALL 6)  ✅ removed use_label_encoder
# =====================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
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

comparison_df = pd.DataFrame(comparison).sort_values("AUC", ascending=False)

# =====================================================
# SENTIMENT + WORDCLOUD (FROM TXT)
# =====================================================
with open(os.path.join(BASE_DIR, "NPS Survey.txt"), "r", encoding="utf-8") as f:
    reviews = [line.strip() for line in f.readlines() if line.strip()]

sia = SentimentIntensityAnalyzer()
sentiment_df = pd.DataFrame({"Feedback": reviews})
sentiment_df["Score"] = sentiment_df["Feedback"].apply(lambda x: sia.polarity_scores(x)["compound"])
sentiment_df["Sentiment"] = sentiment_df["Score"].apply(
    lambda s: "Positive" if s >= 0.05 else ("Negative" if s <= -0.05 else "Neutral")
)

wc_text = " ".join(sentiment_df["Feedback"].astype(str))
wc = WordCloud(width=1200, height=550, background_color="white").generate(wc_text)

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
# FIGURE STYLING
# =====================================================
def style_fig(fig, title=None):
    fig.update_layout(
        template="plotly_white",
        title=title,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial", size=13),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, zeroline=False)
    return fig

def kpi_card(label, value, sub=None):
    return html.Div(
        className="kpi",
        children=[
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(sub or "", className="kpi-sub"),
        ],
    )

# KPI values
n_rows = len(df)
try:
    pos_rate = float(df[target].mean())
except Exception:
    pos_rate = float(df_encoded[target].mean())

best_model = comparison_df.iloc[0]["Model"]
best_auc = float(comparison_df.iloc[0]["AUC"])
avg_auc = float(comparison_df["AUC"].mean())

# =====================================================
# DASH APP
# =====================================================
external_stylesheets = ["https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Bank Credit Scoring Dashboard"

app.layout = html.Div(
    className="page",
    children=[
        # HEADER
        html.Div(
            className="topbar",
            children=[
                html.Div(
                    children=[
                        html.H1("Bank Credit Scoring Dashboard", className="title"),
                        html.Div("EDA • Model ROC • Model comparison • Customer feedback sentiment", className="subtitle"),
                    ]
                ),
                html.Div(
                    className="tag",
                    children=f"Dataset rows: {n_rows:,}  |  Target positive rate: {pos_rate:.2%}",
                ),
            ],
        ),

        # KPI ROW
        html.Div(
            className="kpi-row",
            children=[
                kpi_card("Best model (AUC)", f"{best_auc:.3f}", best_model),
                kpi_card("Average AUC (all models)", f"{avg_auc:.3f}", "Higher is better"),
                kpi_card("Models evaluated", f"{len(models)}", "LR, NB, SVM, DT, RF, XGB"),
                kpi_card("NPS feedback lines", f"{len(sentiment_df):,}", "VADER sentiment"),
            ],
        ),

        # MAIN CARD (TABS)
        html.Div(
            className="card",
            children=[
                dcc.Tabs(
                    className="dash-tabs",
                    children=[
                        # TAB 1: EDA
                        dcc.Tab(
                            label="EDA",
                            children=[
                                html.Div(style={"height": "10px"}),
                                html.Div(
                                    className="controls",
                                    children=[
                                        html.Div("Choose chart", className="control-label"),
                                        dcc.Dropdown(
                                            id="eda-choice",
                                            className="dd",
                                            clearable=False,
                                            options=[
                                                {"label": "Target Distribution (bad)", "value": "target"},
                                                {"label": "Debt-to-Income vs bad (boxplot)", "value": "debtinc"},
                                                {"label": "Credit Debt vs bad (boxplot)", "value": "creddebt"},
                                                {"label": "Other Debt vs bad (boxplot)", "value": "othdebt"},
                                                {"label": "Age Distribution", "value": "age"},
                                            ],
                                            value="target",
                                        ),
                                    ],
                                ),
                                dcc.Graph(id="eda-graph", config={"displayModeBar": False}),
                            ],
                        ),

                        # TAB 2: ROC
                        dcc.Tab(
                            label="MODELS (ROC)",
                            children=[
                                html.Div(style={"height": "10px"}),
                                html.Div(
                                    className="controls",
                                    children=[
                                        html.Div("Select model", className="control-label"),
                                        dcc.Dropdown(
                                            id="model-choice",
                                            className="dd",
                                            clearable=False,
                                            options=[{"label": m, "value": m} for m in models],
                                            value="Logistic Regression",
                                        ),
                                    ],
                                ),
                                dcc.Graph(id="roc-graph", config={"displayModeBar": False}),
                            ],
                        ),

                        # TAB 3: COMPARISON
                        dcc.Tab(
                            label="MODEL COMPARISON",
                            children=[
                                html.Div(style={"height": "10px"}),
                                dcc.Graph(
                                    id="compare-graph",
                                    config={"displayModeBar": False},
                                    figure=style_fig(
                                        px.bar(
                                            comparison_df,
                                            x="Model",
                                            y=["AUC", "Accuracy", "Precision", "Recall", "F1-score"],
                                            barmode="group",
                                            title="Model Performance Comparison",
                                        ),
                                        title="Model Performance Comparison",
                                    ),
                                ),
                                html.Div(style={"height": "8px"}),

                                html.Div(
                                    className="inner-card",
                                    children=[
                                        html.Div("Performance Table", className="inner-title"),
                                        dash_table.DataTable(
                                            data=comparison_df.to_dict("records"),
                                            columns=[{"name": c, "id": c} for c in comparison_df.columns],
                                            sort_action="native",
                                            page_size=10,
                                            style_table={"overflowX": "auto"},
                                            style_header={
                                                "backgroundColor": "#f8fafc",
                                                "border": "1px solid #e5e7eb",
                                                "fontWeight": "700",
                                                "color": "#0f172a",
                                            },
                                            style_cell={
                                                "border": "1px solid #e5e7eb",
                                                "padding": "10px",
                                                "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial",
                                                "fontSize": "13px",
                                                "color": "#0f172a",
                                            },
                                            style_data_conditional=[
                                                {"if": {"row_index": "odd"}, "backgroundColor": "#fcfcfd"},
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # TAB 4: WORDCLOUD
                        dcc.Tab(
                            label="WORDCLOUD & SENTIMENT",
                            children=[
                                html.Div(style={"height": "10px"}),
                                html.Div(
                                    className="inner-card",
                                    children=[
                                        html.Div("WordCloud (from NPS Survey.txt)", className="inner-title"),
                                        html.Img(
                                            src="data:image/png;base64," + wc_base64,
                                            style={
                                                "width": "100%",
                                                "borderRadius": "14px",
                                                "border": "1px solid #e5e7eb",
                                            },
                                        ),
                                    ],
                                ),
                                html.Div(style={"height": "14px"}),
                                dcc.Graph(
                                    id="sent-graph",
                                    config={"displayModeBar": False},
                                    figure=style_fig(
                                        px.histogram(
                                            sentiment_df,
                                            x="Sentiment",
                                            color="Sentiment",
                                            title="Sentiment Distribution (VADER)",
                                        ),
                                        title="Sentiment Distribution (VADER)",
                                    ),
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),

        html.Div(
            "Tip: Sort the performance table by clicking column headers.",
            style={"color": "#64748b", "fontSize": "12px", "marginTop": "10px"},
        ),
    ],
)

# =====================================================
# CALLBACKS
# =====================================================
@app.callback(Output("eda-graph", "figure"), Input("eda-choice", "value"))
def update_eda(choice):
    if choice == "target":
        fig = px.histogram(df, x=target, title="Target Distribution (bad)")
    elif choice in ["debtinc", "creddebt", "othdebt"]:
        fig = px.box(df, x=target, y=choice, title=f"{choice} vs bad")
    elif choice == "age":
        fig = px.histogram(df, x="age", title="Age Distribution")
    else:
        fig = px.histogram(df, x=target, title="Target Distribution (bad)")

    return style_fig(fig, title=fig.layout.title.text)

@app.callback(Output("roc-graph", "figure"), Input("model-choice", "value"))
def update_roc(model_name):
    fpr, tpr, roc_auc = roc_data[model_name]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{model_name} (AUC = {roc_auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))

    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return style_fig(fig, title="ROC Curve")

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
