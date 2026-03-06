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

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# =====================================================
# LOAD DATA (EXACT FILE NAME)
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
# LABEL ENCODING
# =====================================================
df_encoded = df.copy()
encoders = {}

for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =====================================================
# SCALING (REQUIRED FOR SVM & LOGISTIC)
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# MODELS (ALL 6 FROM NOTEBOOK)
# =====================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False)
}

roc_data = {}
comparison = []

# =====================================================
# TRAIN MODELS & METRICS
# =====================================================
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

    report = classification_report(y_test, preds, output_dict=True)

    comparison.append({
        "Model": name,
        "Accuracy": report["accuracy"],
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1-score": report["1"]["f1-score"]
    })

comparison_df = pd.DataFrame(comparison)

# =====================================================
# DASH APP
# =====================================================
app = dash.Dash(__name__)
app.title = "Bank Credit Scoring Dashboard"

app.layout = html.Div(style={"backgroundColor": "white", "padding": "20px"}, children=[

    html.H1("Bank Credit Scoring Dashboard", style={"textAlign": "center"}),

    dcc.Tabs([

        # ================= TAB 1 : EDA =================
        dcc.Tab(label="EDA", children=[
            html.Br(),
            dcc.Dropdown(
                id="eda-choice",
                options=[
                    {"label": "Age Distribution", "value": "age"},
                    {"label": "Debt Income Ratio", "value": "debtinc"},
                    {"label": "Credit vs Other Debt", "value": "debt"},
                    {"label": "Target Distribution", "value": "target"}
                ],
                value="age"
            ),
            dcc.Graph(id="eda-graph")
        ]),

        # ================= TAB 2 : ROC =================
        dcc.Tab(label="Model ROC", children=[
            html.Br(),
            dcc.Dropdown(
                id="model-choice",
                options=[{"label": m, "value": m} for m in models],
                value="Logistic Regression"
            ),
            dcc.Graph(id="roc-graph")
        ]),

        # ================= TAB 3 : COMPARISON =================
        dcc.Tab(label="Model Comparison", children=[
            html.Br(),
            dcc.Graph(
                figure=px.bar(
                    comparison_df,
                    x="Model",
                    y=["Accuracy", "Precision", "Recall", "F1-score"],
                    barmode="group",
                    title="Model Performance Comparison"
                )
            )
        ]),

        # ================= TAB 4 : WORDCLOUD =================
        dcc.Tab(label="WordCloud", children=[
            html.Br(),
            html.Div(id="wordcloud-area")
        ])

    ])
])

# =====================================================
# CALLBACKS
# =====================================================
@app.callback(Output("eda-graph", "figure"),
              Input("eda-choice", "value"))
def update_eda(choice):
    if choice == "age":
        return px.histogram(df, x="age", title="Age Distribution")
    elif choice == "debtinc":
        return px.box(df, y="debtinc", title="Debt Income Ratio")
    elif choice == "debt":
        return px.scatter(df, x="creddebt", y="othdebt",
                          title="Credit Debt vs Other Debt")
    else:
        return px.histogram(df, x="bad", title="Target Variable Distribution")


@app.callback(Output("roc-graph", "figure"),
              Input("model-choice", "value"))
def update_roc(model_name):
    fpr, tpr, roc_auc = roc_data[model_name]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                             name=f"AUC = {roc_auc:.2f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             line=dict(dash="dash")))

    fig.update_layout(
        title=f"ROC Curve - {model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )
    return fig


@app.callback(Output("wordcloud-area", "children"),
              Input("model-choice", "value"))
def wordcloud_callback(_):

    text_cols = df.select_dtypes(include="object").columns

    if len(text_cols) == 0:
        return html.H4("No text column available for WordCloud.")

    text = " ".join(df[text_cols[0]].astype(str))

    wc = WordCloud(width=800, height=400,
                   background_color="white").generate(text)

    buf = BytesIO()
    plt.imshow(wc)
    plt.axis("off")
    plt.savefig(buf, format="png")
    plt.close()

    encoded = base64.b64encode(buf.getvalue()).decode()

    return html.Img(src="data:image/png;base64," + encoded)

# =====================================================
# RUN LOCALHOST
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
