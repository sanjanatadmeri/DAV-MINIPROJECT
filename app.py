import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load project files
model = joblib.load("churn_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
df = pd.read_csv("cleaned_churn_data.csv")

# Dashboard title
st.title("Customer Churn Prediction Dashboard")
st.write("Telecommunication Customer Retention Analysis")

# Calculate statistics
total = len(df)
churned = (df["Churn"] == "Yes").sum()
retained = (df["Churn"] == "No").sum()

churn_rate = churned / total * 100
retention_rate = retained / total * 100

# KPI cards
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Customers", total)
c2.metric("Churned Customers", churned)
c3.metric("Churn Rate", f"{churn_rate:.2f}%")
c4.metric("Retention Rate", f"{retention_rate:.2f}%")

st.divider()

# Churn distribution
st.subheader("Customer Churn Distribution")

fig, ax = plt.subplots()
df["Churn"].value_counts().plot(
    kind="bar", ax=ax
)
ax.set_xlabel("Churn Status")
ax.set_ylabel("Number of Customers")
st.pyplot(fig)
plt.close(fig)

# Contract analysis
st.subheader("Churn by Contract Type")

contract_data = pd.crosstab(
    df["Contract"], df["Churn"]
)
st.bar_chart(contract_data)

# Payment method analysis
st.subheader("Churn by Payment Method")

payment_data = pd.crosstab(
    df["PaymentMethod"], df["Churn"]
)
st.bar_chart(payment_data)

# Monthly charges
st.subheader("Monthly Charges vs Churn")

fig, ax = plt.subplots()
sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges",
    ax=ax
)
st.pyplot(fig)
plt.close(fig)

# Individual prediction
st.divider()
st.header("Individual Customer Churn Prediction")

with st.form("prediction_form"):

    tenure = st.slider("Tenure (Months)", 0, 72, 12)

    monthly = st.number_input(
        "Monthly Charges", min_value=0.0, value=70.0
    )

    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=840.0
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    phone = st.selectbox("Phone Service", ["Yes", "No"])

    security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    multiple = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    paperless = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )

    submitted = st.form_submit_button("Predict Churn")

    if submitted:

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges
        }])

        encoded = pd.get_dummies(
            customer, drop_first=True, dtype=int
        )

        encoded = encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = model.predict(encoded)[0]
        probability = model.predict_proba(encoded)[0][1]

        if prediction == 1:
            st.error("Customer is likely to churn!")
        else:
            st.success("Customer is likely to stay!")

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

st.caption("Data Analytics Mini Project - Experiment 12")