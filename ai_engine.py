import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


# ============================================================
# EXPENSE CATEGORIES
# ============================================================

CATEGORIES = [
    "Food",
    "Transport",
    "Shopping",
    "Bills",
    "Entertainment",
    "Healthcare",
    "Education",
    "Rent",
    "Travel",
    "Other",
]


# ============================================================
# TRAINING DATA FOR CATEGORY PREDICTION
# ============================================================

texts = [
    "restaurant food lunch cafe groceries",
    "uber ola taxi bus train metro petrol",
    "amazon clothes shoes shopping mall cosmetics",
    "electricity water internet mobile bill recharge",
    "movie netflix spotify games concert",
    "doctor medicine pharmacy hospital health",
    "course college books tuition class",
    "rent apartment house landlord",
    "flight hotel vacation trip travel",
    "miscellaneous other expense",
]


# ============================================================
# TF-IDF + LOGISTIC REGRESSION CATEGORY MODEL
# ============================================================

V = TfidfVectorizer(ngram_range=(1, 2))

M = LogisticRegression(
    max_iter=1000,
    random_state=42
).fit(
    V.fit_transform(texts),
    CATEGORIES
)


def predict_categories(texts):
    """
    Predict expense categories from descriptions.
    """
    if not texts:
        return []

    cleaned_texts = [
        str(text) if text is not None else "Expense"
        for text in texts
    ]

    return M.predict(
        V.transform(cleaned_texts)
    ).tolist()


# ============================================================
# PREPARE DATAFRAME
# ============================================================

def prepare_df(df):
    """
    Standardizes uploaded/manual expense data.

    Expected columns:
        date
        description
        amount
        category

    Missing columns are automatically created.
    """

    # Make sure input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df = df.copy()

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )
    else:
        df["date"] = pd.Timestamp.today()

    # Replace invalid dates
    df["date"] = df["date"].fillna(
        pd.Timestamp.today()
    )

    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    if "description" in df.columns:
        df["description"] = (
            df["description"]
            .fillna("Expense")
            .astype(str)
        )
    else:
        df["description"] = "Expense"

    # --------------------------------------------------------
    # AMOUNT
    # --------------------------------------------------------

    if "amount" in df.columns:

        df["amount"] = pd.to_numeric(
            df["amount"],
            errors="coerce"
        )

        df["amount"] = df["amount"].fillna(0.0)

    else:
        df["amount"] = 0.0

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    if "category" in df.columns:

        df["category"] = (
            df["category"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    else:

        df["category"] = pd.Series(
            "",
            index=df.index,
            dtype="object"
        )

    # --------------------------------------------------------
    # PREDICT MISSING CATEGORIES
    # --------------------------------------------------------

    missing = (
        df["category"].isna()
        | df["category"].eq("")
    )

    if missing.any():

        predictions = predict_categories(
            df.loc[
                missing,
                "description"
            ].tolist()
        )

        df.loc[
            missing,
            "category"
        ] = predictions

    # --------------------------------------------------------
    # MONTH
    # --------------------------------------------------------

    df["month"] = (
        df["date"]
        .dt.to_period("M")
        .astype(str)
    )

    # --------------------------------------------------------
    # SORT DATA
    # --------------------------------------------------------

    return (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )


# ============================================================
# BEHAVIOR PROFILE
# ============================================================

def behavior_profile(df):

    if df.empty or len(df) < 2:

        return pd.DataFrame([
            {
                "profile": "Getting Started",
                "description":
                    "Add more data for a stronger profile."
            }
        ])

    m = (
        df.groupby("month")["amount"]
        .agg(["sum", "count"])
        .reset_index()
    )

    k = min(3, len(m))

    # KMeans requires at least 1 cluster
    if k <= 1:

        m["profile"] = "Moderate Spender"

        return m

    labs = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    ).fit_predict(
        m[["sum", "count"]]
    )

    # Sort cluster labels according to spending
    cluster_spending = (
        m.assign(cluster=labs)
        .groupby("cluster")["sum"]
        .mean()
        .sort_values()
    )

    profile_map = {}

    names = [
        "Low Spender",
        "Moderate Spender",
        "High Spender"
    ]

    for position, cluster_id in enumerate(
        cluster_spending.index
    ):

        profile_map[
            cluster_id
        ] = names[
            min(position, 2)
        ]

    m["profile"] = [
        profile_map[x]
        for x in labs
    ]

    return m


# ============================================================
# ANOMALY DETECTION
# ============================================================

def detect_anomalies(df):

    x = df.copy()

    x["anomaly"] = False

    if len(x) >= 5:

        model = IsolationForest(
            contamination="auto",
            random_state=42
        )

        predictions = model.fit_predict(
            x[["amount"]]
        )

        x["anomaly"] = predictions == -1

    return x


# ============================================================
# NEXT MONTH FORECAST
# ============================================================

def forecast_next_month(df):

    if df.empty:
        return 0.0

    m = (
        df.groupby("month")["amount"]
        .sum()
        .reset_index()
    )

    if len(m) == 0:
        return 0.0

    if len(m) == 1:
        return float(m["amount"].iloc[0])

    X = np.arange(
        len(m)
    ).reshape(-1, 1)

    y = m["amount"].values

    model = LinearRegression()

    model.fit(X, y)

    prediction = model.predict(
        [[len(m)]]
    )[0]

    return max(
        0.0,
        float(prediction)
    )


# ============================================================
# SMART BUDGET
# ============================================================

def smart_budget(df, salary=0):

    if df.empty:

        return pd.DataFrame(
            columns=[
                "category",
                "recommended_budget"
            ]
        )

    b = (
        df.groupby("category")["amount"]
        .mean()
        * 1.05
    )

    # Keep total recommended budget below 90% salary
    if salary and salary > 0:

        if b.sum() > salary * 0.9:

            b *= (
                salary * 0.9
            ) / b.sum()

    return (
        b
        .reset_index(
            name="recommended_budget"
        )
    )


# ============================================================
# RECURRING MERCHANTS
# ============================================================

def recurring_merchants(df):

    if df.empty:

        return pd.DataFrame(
            columns=[
                "description",
                "occurrences",
                "average_amount"
            ]
        )

    x = (
        df.groupby("description")
        .agg(
            occurrences=("amount", "size"),
            average_amount=("amount", "mean")
        )
        .reset_index()
    )

    return (
        x[
            x["occurrences"] >= 2
        ]
        .sort_values(
            "occurrences",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# DUPLICATE EXPENSE DETECTION
# ============================================================

def find_duplicate_candidates(df):

    if df.empty:
        return df.copy()

    required_columns = [
        "date",
        "description",
        "amount",
        "category"
    ]

    # Make sure required columns exist
    for column in required_columns:

        if column not in df.columns:

            df[column] = ""

    return df[
        df.duplicated(
            required_columns,
            keep=False
        )
    ].copy()


# ============================================================
# AI INSIGHTS
# ============================================================

def generate_insights(df, salary):

    if df.empty:

        return [
            "Add expenses to generate insights."
        ]

    total = float(
        df["amount"].sum()
    )

    # Avoid division by zero
    if total <= 0:

        return [
            "Add valid expense amounts to generate insights."
        ]

    top = (
        df.groupby("category")["amount"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    largest_category = top.index[0]

    largest_percentage = (
        top.iloc[0] / total
    ) * 100

    balance = float(
        salary - total
    )

    out = [

        f"Largest category: "
        f"{largest_category} "
        f"({largest_percentage:.1f}% of total).",

        f"Estimated balance: "
        f"₹{balance:,.2f}."
    ]

    if balance < 0:

        out.append(
            "Expenses exceed entered salary; "
            "review flexible categories."
        )

    elif salary > 0 and total > salary * 0.8:

        out.append(
            "You have used more than 80% "
            "of your entered salary."
        )

    elif salary > 0:

        savings = salary - total

        out.append(
            f"Potential remaining savings: "
            f"₹{savings:,.2f}."
        )

    return out