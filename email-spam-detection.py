import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main-title {
    background: linear-gradient(135deg, #7f1d1d, #ef4444);
    padding: 45px;
    border-radius: 20px;
    color: white;
    margin-bottom: 30px;
}

.main-title h1 {
    font-size: 52px;
    margin-bottom: 15px;
}

.main-title p {
    font-size: 20px;
}

.metric-box {
    padding: 20px;
    border-radius: 15px;
    background: #1f2937;
    text-align: center;
}

.result-spam {
    background: #7f1d1d;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;
}

.result-safe {
    background: #166534;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("completeSpamAssassin.csv")

    # Remove unnecessary column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    # Remove missing email bodies
    df = df.dropna(subset=["Body"])

    # Convert email body to string
    df["Body"] = df["Body"].astype(str)

    return df


df = load_data()


# ==========================================================
# PREPARE DATA
# ==========================================================

X = df["Body"]
y = df["Label"]

# Convert email text into numerical features
vectorizer = CountVectorizer(
    stop_words="english"
)

X_vectorized = vectorizer.fit_transform(X)


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# TRAIN MODEL
# ==========================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)


# ==========================================================
# MODEL EVALUATION
# ==========================================================

prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    prediction
)

report = classification_report(
    y_test,
    prediction,
    output_dict=True
)

matrix = confusion_matrix(
    y_test,
    prediction
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📧 Email Spam")

st.sidebar.markdown(
    "### Machine Learning Classification System"
)

st.sidebar.divider()

st.sidebar.subheader("Model")

st.sidebar.write("Logistic Regression")

st.sidebar.subheader("Dataset")

st.sidebar.write(
    f"{len(df):,} email records"
)

st.sidebar.subheader("NLP Features")

st.sidebar.write(
    f"{X_vectorized.shape[1]:,} text features"
)

st.sidebar.divider()

st.sidebar.success(
    "Model Status: Ready"
)


# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<div class="main-title">

<h1>📧 Email Spam Detection</h1>

<p>
Detect whether an email is Spam or Not Spam
using Natural Language Processing and Logistic Regression.
</p>

</div>
""", unsafe_allow_html=True)


# ==========================================================
# MODEL INFORMATION
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Dataset Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Input Features",
        f"{X_vectorized.shape[1]:,}"
    )

with col3:
    st.metric(
        "Model",
        "Logistic Regression"
    )

with col4:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


st.divider()


# ==========================================================
# EMAIL INPUT
# ==========================================================

st.subheader("📨 Check an Email")

st.write(
    "Enter or paste an email message below to classify it."
)

email_text = st.text_area(
    "Email Message",
    height=220,
    placeholder="Enter an email message to classify..."
)


# ==========================================================
# PREDICTION
# ==========================================================

if st.button(
    "🔍 Check Email",
    use_container_width=True
):

    if not email_text.strip():

        st.warning(
            "Please enter an email message first."
        )

    else:

        # Convert email into numerical features
        email_vector = vectorizer.transform(
            [email_text]
        )

        # Prediction
        result = model.predict(
            email_vector
        )[0]

        # Prediction probability
        probability = model.predict_proba(
            email_vector
        )[0]

        confidence = max(probability) * 100


        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        if result == 1:

            st.markdown(
                f"""
                <div class="result-spam">

                <h1>🚨 SPAM EMAIL</h1>

                <h3>
                The model classified this email as Spam.
                </h3>

                <p>
                Prediction Confidence: {confidence:.2f}%
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-safe">

                <h1>✅ NOT SPAM</h1>

                <h3>
                The model classified this email as Not Spam.
                </h3>

                <p>
                Prediction Confidence: {confidence:.2f}%
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

st.divider()

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Training Emails",
        f"{len(y_train):,}"
    )

with col3:
    st.metric(
        "Testing Emails",
        f"{len(y_test):,}"
    )


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

st.subheader("🔢 Confusion Matrix")

cm_df = pd.DataFrame(
    matrix,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

st.dataframe(
    cm_df,
    use_container_width=True
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

st.subheader("📋 Classification Report")

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.round(3),
    use_container_width=True
)


# ==========================================================
# DATASET PREVIEW
# ==========================================================

st.divider()

st.subheader("📄 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ==========================================================
# PROJECT INFORMATION
# ==========================================================

st.divider()

st.subheader("ℹ️ About the Project")

st.write("""
This project uses Natural Language Processing to classify
emails as Spam or Not Spam.

The email body is converted into numerical features using
CountVectorizer. A Logistic Regression model is then trained
on the processed email dataset.

This version does not use Gmail API, Gmail login, OAuth,
or access to any user's inbox. The user manually enters
an email message for classification.
""")

st.caption(
    "Email Spam Detection • Python • Streamlit • "
    "Scikit-learn • Logistic Regression"
)