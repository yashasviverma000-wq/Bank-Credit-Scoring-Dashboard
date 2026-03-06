import os
import base64
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State, dash_table, ALL
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wordcloud import WordCloud

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, "Final data converted from num to cat.xlsx")
TXT_PATH = os.path.join(BASE_DIR, "NPS Survey.txt")
CACHE_PATH = os.path.join(BASE_DIR, "model_artifacts.joblib")

TARGET = "bad"


# =====================================================
# NLTK
# =====================================================
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")


# =====================================================
# HELPERS
# =====================================================
def make_wordcloud_base64(text: str) -> str:
    wc = WordCloud(width=1200, height=550, background_color="white").generate(text)
    buf = BytesIO()
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_decision_tree_base64(model, feature_names, class_names=("0", "1")) -> str:
    buf = BytesIO()
    plt.figure(figsize=(24, 14))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=list(class_names),
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=3
    )
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def style_fig(fig, title=None):
    fig.update_layout(
        template="plotly_white",
        title=title,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial",
            size=13
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
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


def detect_numeric_series(series: pd.Series, threshold: float = 0.7) -> bool:
    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().mean() >= threshold


# =====================================================
# DATA + MODEL PREP
# =====================================================
def fit_and_prepare():
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    df.columns = df.columns.str.strip()

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")

    original_df = df.copy()

    feature_types = {}
    categorical_options = {}
    numeric_defaults = {}

    # ---------------- CLEAN / MISSING ----------------
    for col in df.columns:
        df[col] = df[col].replace(["nan", "None", ""], np.nan)

        if col == TARGET:
            continue

        if detect_numeric_series(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
            feature_types[col] = "numeric"
            numeric_defaults[col] = float(df[col].median())
        else:
            df[col] = df[col].astype(str).replace("nan", np.nan)
            mode_val = df[col].mode()
            fill_value = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)
            df[col] = df[col].astype(str).str.strip()
            feature_types[col] = "categorical"
            categorical_options[col] = sorted(df[col].dropna().astype(str).unique().tolist())

    # ---------------- TARGET ----------------
    if detect_numeric_series(df[TARGET], threshold=0.95):
        y_raw = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
        target_encoder = None
    else:
        target_encoder = LabelEncoder()
        y_raw = target_encoder.fit_transform(df[TARGET].astype(str).str.strip())

    df_processed = df.copy()
    encoders = {}

    # ---------------- ENCODE FEATURES ----------------
    for col in df_processed.columns:
        if col == TARGET:
            continue
        if feature_types[col] == "categorical":
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            encoders[col] = le
        else:
            df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce").fillna(
                numeric_defaults[col]
            )

    X = df_processed.drop(columns=[TARGET], errors="ignore").copy()
    y = pd.Series(y_raw, name=TARGET)

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------- MODELS ----------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }

    roc_data = {}
    comparison = {}
    fitted_models = {}

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

        rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
        positive_class = "1" if "1" in rep else [k for k in rep.keys() if k not in ["accuracy", "macro avg", "weighted avg"]][0]

        comparison[name] = {
            "Model": name,
            "AUC": round(roc_auc, 3),
            "Accuracy": round(rep["accuracy"], 3),
            "Precision": round(rep[positive_class]["precision"], 3),
            "Recall": round(rep[positive_class]["recall"], 3),
            "F1-score": round(rep[positive_class]["f1-score"], 3),
        }

        roc_data[name] = (fpr, tpr, roc_auc)
        fitted_models[name] = model

    comparison_df = pd.DataFrame(list(comparison.values())).sort_values("AUC", ascending=False)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]
    decision_tree_model = fitted_models["Decision Tree"]

    # ---------------- FEATURE IMPORTANCE ----------------
    if hasattr(best_model, "feature_importances_"):
        fi = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": fi
        }).sort_values("Importance", ascending=False)
    elif hasattr(best_model, "coef_"):
        fi = np.abs(best_model.coef_[0])
        feature_importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": fi
        }).sort_values("Importance", ascending=False)
    else:
        feature_importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": np.zeros(len(feature_cols))
        }).sort_values("Importance", ascending=False)

    # ---------------- DECISION TREE IMAGE ----------------
    dtree_base64 = make_decision_tree_base64(
        decision_tree_model,
        feature_names=feature_cols,
        class_names=("0", "1")
    )

    # ---------------- SENTIMENT ----------------
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        reviews = [line.strip() for line in f.readlines() if line.strip()]

    sia = SentimentIntensityAnalyzer()
    sentiment_df = pd.DataFrame({"Feedback": reviews})
    sentiment_df["Score"] = sentiment_df["Feedback"].apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )
    sentiment_df["Sentiment"] = sentiment_df["Score"].apply(
        lambda s: "Positive" if s >= 0.05 else ("Negative" if s <= -0.05 else "Neutral")
    )

    wc_text = " ".join(sentiment_df["Feedback"].astype(str))
    wc_base64 = make_wordcloud_base64(wc_text)

    artifacts = {
        "df": df,
        "original_df": original_df,
        "feature_cols": feature_cols,
        "feature_types": feature_types,
        "categorical_options": categorical_options,
        "numeric_defaults": numeric_defaults,
        "encoders": encoders,
        "target_encoder": target_encoder,
        "scaler": scaler,
        "models": fitted_models,
        "best_model_name": best_model_name,
        "comparison_df": comparison_df,
        "roc_data": roc_data,
        "feature_importance_df": feature_importance_df,
        "sentiment_df": sentiment_df,
        "wc_base64": wc_base64,
        "dtree_base64": dtree_base64,
    }

    joblib.dump(artifacts, CACHE_PATH)
    return artifacts


def load_artifacts():
    if os.path.exists(CACHE_PATH):
        try:
            art = joblib.load(CACHE_PATH)
            # if old cache missing new key, rebuild
            required_keys = ["dtree_base64", "feature_importance_df", "roc_data", "comparison_df"]
            if all(k in art for k in required_keys):
                return art
        except Exception:
            pass
    return fit_and_prepare()


# =====================================================
# LOAD ARTIFACTS
# =====================================================
art = load_artifacts()

df = art["df"]
original_df = art["original_df"]
feature_cols = art["feature_cols"]
feature_types = art["feature_types"]
categorical_options = art["categorical_options"]
numeric_defaults = art["numeric_defaults"]
encoders = art["encoders"]
target_encoder = art["target_encoder"]
scaler = art["scaler"]
models = art["models"]
best_model_name = art["best_model_name"]
best_model = models[best_model_name]
comparison_df = art["comparison_df"]
roc_data = art["roc_data"]
feature_importance_df = art["feature_importance_df"]
sentiment_df = art["sentiment_df"]
wc_base64 = art["wc_base64"]
dtree_base64 = art.get("dtree_base64", "")

# KPI values
n_rows = len(df)
pos_rate = float(pd.to_numeric(df[TARGET], errors="coerce").fillna(0).mean())
best_auc = float(comparison_df.iloc[0]["AUC"])
avg_auc = float(comparison_df["AUC"].mean())

all_columns = [TARGET] + feature_cols


# =====================================================
# PREDICTION INPUTS
# =====================================================
def make_prediction_inputs():
    components = []
    for col in feature_cols:
        label = html.Label(col, className="control-label", style={"marginTop": "10px"})

        if feature_types[col] == "numeric":
            comp = dcc.Input(
                id={"type": "pred-input", "index": col},
                type="number",
                value=round(numeric_defaults.get(col, 0), 3),
                className="dd",
                style={"width": "100%"}
            )
        else:
            opts = categorical_options.get(col, ["Unknown"])
            comp = dcc.Dropdown(
                id={"type": "pred-input", "index": col},
                options=[{"label": x, "value": x} for x in opts],
                value=opts[0] if opts else None,
                clearable=False,
                className="dd"
            )

        components.append(html.Div([label, comp], style={"marginBottom": "12px"}))
    return components


# =====================================================
# DASH APP
# =====================================================
external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = "Bank Credit Scoring Dashboard"

app.layout = html.Div(
    className="page",
    children=[
        html.Div(
            className="topbar",
            children=[
                html.Div(
                    children=[
                        html.H1("Bank Credit Scoring Dashboard", className="title"),
                        html.Div(
                            "EDA • ROC • Comparison • Feature Importance • Prediction • Sentiment",
                            className="subtitle",
                        ),
                    ]
                ),
                html.Div(
                    className="tag",
                    children=f"Dataset rows: {n_rows:,} | Positive rate: {pos_rate:.2%}",
                ),
            ],
        ),

        html.Div(
            className="kpi-row",
            children=[
                kpi_card("Best model (AUC)", f"{best_auc:.3f}", best_model_name),
                kpi_card("Average AUC", f"{avg_auc:.3f}", "Across all models"),
                kpi_card("Models evaluated", f"{len(models)}", "LR, NB, SVM, DT, RF, XGB"),
                kpi_card("Feedback lines", f"{len(sentiment_df):,}", "VADER sentiment"),
            ],
        ),

        html.Div(
            className="card",
            children=[
                dcc.Tabs(
                    className="dash-tabs",
                    children=[
                        # =====================================================
                        # TAB 1: EDA
                        # =====================================================
                        dcc.Tab(
                            label="EDA",
                            children=[
                                html.Div(style={"height": "10px"}),

                                html.Div(
                                    className="controls",
                                    children=[
                                        html.Div("EDA Graph Type", className="control-label"),
                                        dcc.Dropdown(
                                            id="eda-graph-type",
                                            className="dd",
                                            clearable=False,
                                            options=[
                                                {"label": "Correlation Heatmap", "value": "corr"},
                                                {"label": "Relationship Between Two Variables", "value": "relation"},
                                                {"label": "Single Variable Distribution", "value": "single"},
                                            ],
                                            value="corr",
                                        ),
                                    ],
                                ),

                                html.Div(style={"height": "10px"}),

                                html.Div(
                                    className="row",
                                    children=[
                                        html.Div(
                                            className="col-md-6",
                                            children=[
                                                html.Div("X Variable", className="control-label"),
                                                dcc.Dropdown(
                                                    id="eda-x-col",
                                                    className="dd",
                                                    clearable=False,
                                                    options=[{"label": c, "value": c} for c in all_columns],
                                                    value=feature_cols[0] if feature_cols else TARGET,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-md-6",
                                            children=[
                                                html.Div("Y Variable", className="control-label"),
                                                dcc.Dropdown(
                                                    id="eda-y-col",
                                                    className="dd",
                                                    clearable=False,
                                                    options=[{"label": c, "value": c} for c in all_columns],
                                                    value=feature_cols[1] if len(feature_cols) > 1 else TARGET,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),

                                html.Div(style={"height": "10px"}),
                                dcc.Graph(id="eda-graph", config={"displayModeBar": False}),
                            ],
                        ),

                        # =====================================================
                        # TAB 2: MODELS (ROC)
                        # =====================================================
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
                                            value=best_model_name,
                                        ),
                                    ],
                                ),
                                dcc.Graph(id="roc-graph", config={"displayModeBar": False}),
                                html.Div(
                                    id="decision-tree-container",
                                    className="inner-card",
                                    style={"marginTop": "14px", "display": "none"},
                                    children=[
                                        html.Div("Decision Tree Visualization", className="inner-title"),
                                        html.Img(
                                            id="decision-tree-image",
                                            src="data:image/png;base64," + dtree_base64,
                                            style={
                                                "width": "100%",
                                                "borderRadius": "14px",
                                                "border": "1px solid #e5e7eb",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # =====================================================
                        # TAB 3: MODEL COMPARISON
                        # =====================================================
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
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # =====================================================
                        # TAB 4: FEATURE IMPORTANCE
                        # =====================================================
                        dcc.Tab(
                            label="FEATURE IMPORTANCE",
                            children=[
                                html.Div(style={"height": "10px"}),
                                dcc.Graph(
                                    id="fi-graph",
                                    config={"displayModeBar": False},
                                    figure=style_fig(
                                        px.bar(
                                            feature_importance_df.head(15),
                                            x="Importance",
                                            y="Feature",
                                            orientation="h",
                                            title=f"Feature Importance ({best_model_name})",
                                        ),
                                        title=f"Feature Importance ({best_model_name})",
                                    ),
                                ),
                            ],
                        ),

                        # =====================================================
                        # TAB 5: LOAN RISK PREDICTION
                        # =====================================================
                        dcc.Tab(
                            label="LOAN RISK PREDICTION",
                            children=[
                                html.Div(style={"height": "10px"}),
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Div(
                                            className="col-md-6",
                                            children=[
                                                html.Div("Enter borrower details", className="inner-title"),
                                                html.Div(make_prediction_inputs(), className="inner-card"),
                                                html.Button(
                                                    "Predict Default Risk",
                                                    id="predict-btn",
                                                    n_clicks=0,
                                                    className="btn btn-primary",
                                                    style={"marginTop": "10px"}
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-md-6",
                                            children=[
                                                html.Div("Prediction Output", className="inner-title"),
                                                html.Div(id="prediction-output", className="inner-card"),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # =====================================================
                        # TAB 6: WORDCLOUD & SENTIMENT
                        # =====================================================
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
    ],
)


# =====================================================
# CALLBACKS
# =====================================================
@app.callback(
    Output("eda-graph", "figure"),
    Input("eda-graph-type", "value"),
    Input("eda-x-col", "value"),
    Input("eda-y-col", "value"),
)
def update_eda(graph_type, x_col, y_col):
    if graph_type == "corr":
        numeric_df = df.copy()
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

        corr_df = numeric_df.select_dtypes(include=[np.number])

        if corr_df.shape[1] < 2:
            fig = px.imshow(
                pd.DataFrame([[1]], columns=["No numeric columns"], index=["No numeric columns"]),
                text_auto=True,
                title="Correlation Heatmap"
            )
        else:
            corr = corr_df.corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap"
            )
        return style_fig(fig, "Correlation Heatmap")

    if graph_type == "single":
        if x_col not in df.columns:
            fig = px.histogram(df, x=TARGET, title="Distribution")
            return style_fig(fig, "Distribution")

        if feature_types.get(x_col) == "numeric" or x_col == TARGET:
            fig = px.histogram(
                df,
                x=x_col,
                color=TARGET if x_col != TARGET else None,
                title=f"Distribution of {x_col}"
            )
        else:
            counts = df[x_col].value_counts().reset_index()
            counts.columns = [x_col, "Count"]
            fig = px.bar(counts, x=x_col, y="Count", title=f"Count of {x_col}")

        return style_fig(fig, fig.layout.title.text)

    # relationship graph
    if x_col not in df.columns or y_col not in df.columns:
        fig = px.histogram(df, x=TARGET, title="Relationship Plot")
        return style_fig(fig, "Relationship Plot")

    x_numeric = (feature_types.get(x_col) == "numeric") or (x_col == TARGET)
    y_numeric = (feature_types.get(y_col) == "numeric") or (y_col == TARGET)

    if x_numeric and y_numeric:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=TARGET if TARGET in df.columns else None,
            title=f"{x_col} vs {y_col}"
        )
    elif (not x_numeric) and y_numeric:
        fig = px.box(
            df,
            x=x_col,
            y=y_col,
            color=x_col,
            title=f"{y_col} by {x_col}"
        )
    elif x_numeric and (not y_numeric):
        fig = px.box(
            df,
            x=y_col,
            y=x_col,
            color=y_col,
            title=f"{x_col} by {y_col}"
        )
    else:
        cross = pd.crosstab(df[x_col], df[y_col]).reset_index()
        fig = px.bar(
            cross,
            x=x_col,
            y=cross.columns[1:],
            barmode="group",
            title=f"{x_col} vs {y_col}"
        )

    return style_fig(fig, fig.layout.title.text)


@app.callback(
    Output("roc-graph", "figure"),
    Output("decision-tree-container", "style"),
    Input("model-choice", "value")
)
def update_roc(model_name):
    fpr, tpr, roc_auc = roc_data[model_name]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"{model_name} (AUC = {roc_auc:.3f})"
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(dash="dash")
    ))

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )
    fig = style_fig(fig, title="ROC Curve")

    if model_name == "Decision Tree":
        tree_style = {"marginTop": "14px", "display": "block"}
    else:
        tree_style = {"marginTop": "14px", "display": "none"}

    return fig, tree_style


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State({"type": "pred-input", "index": ALL}, "value"),
    State({"type": "pred-input", "index": ALL}, "id"),
)
def predict_default(n_clicks, values, ids):
    if not n_clicks:
        return html.Div("Click the button to predict default probability.")

    if not values or not ids:
        return html.Div("No input values found.")

    row = {}
    for val, item_id in zip(values, ids):
        col = item_id["index"]
        row[col] = val

    input_df = pd.DataFrame([row])

    for col in feature_cols:
        if feature_types[col] == "numeric":
            try:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
                input_df[col] = input_df[col].fillna(numeric_defaults.get(col, 0))
            except Exception:
                input_df[col] = numeric_defaults.get(col, 0)
        else:
            input_df[col] = input_df[col].astype(str).fillna("Unknown")
            le = encoders[col]
            val = input_df[col].iloc[0]
            if val not in le.classes_:
                val = le.classes_[0]
            input_df[col] = le.transform([val])

    input_df = input_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    if best_model_name in ["Logistic Regression", "SVM"]:
        input_scaled = scaler.transform(input_df)
        prob = float(best_model.predict_proba(input_scaled)[0][1])
        pred = int(best_model.predict(input_scaled)[0])
    else:
        prob = float(best_model.predict_proba(input_df)[0][1])
        pred = int(best_model.predict(input_df)[0])

    if prob >= 0.70:
        risk = "High Risk"
    elif prob >= 0.40:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    result_cards = html.Div(
        children=[
            kpi_card("Probability of Default", f"{prob:.2%}", best_model_name),
            kpi_card("Predicted Class", str(pred), "1 means likely default"),
            kpi_card("Risk Category", risk, "Threshold-based"),
        ]
    )

    top_factors = feature_importance_df.head(5)[["Feature", "Importance"]].copy()
    top_table = dash_table.DataTable(
        data=top_factors.to_dict("records"),
        columns=[{"name": c, "id": c} for c in top_factors.columns],
        style_table={"overflowX": "auto"},
        style_header={"fontWeight": "bold"},
        style_cell={"padding": "8px", "fontSize": "13px"},
    )

    return html.Div([
        result_cards,
        html.Br(),
        html.H5("Top global risk factors"),
        top_table
    ])


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=False)