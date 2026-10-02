import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# Sample Training Dataset
# -----------------------------

data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 5, 7, 9, 4, 6],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90, 52, 62, 72, 85, 95, 68, 78],
    "Previous_Score": [35, 40, 45, 50, 55, 60, 65, 80, 38, 48, 58, 70, 90, 52, 68],
    "Result": [
        "Fail", "Fail", "Fail", "Fail", "Pass",
        "Pass", "Pass", "Pass", "Fail", "Fail",
        "Pass", "Pass", "Pass", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# Train ML Model
# -----------------------------

X = df[["Study_Hours", "Attendance", "Previous_Score"]]
y = df["Result"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# -----------------------------
# Website Design
# -----------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction")
st.write(
    "Enter the student's details below to predict their academic performance."
)

st.divider()

# -----------------------------
# User Inputs
# -----------------------------

study_hours = st.number_input(
    "📚 Study Hours per Day",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.5
)

attendance = st.number_input(
    "📅 Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

previous_score = st.number_input(
    "📝 Previous Exam Score (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=1.0
)

st.divider()

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict Performance"):

    input_data = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Score": [previous_score]
    })

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == "Pass":
        st.success("✅ Student is likely to PASS")
    else:
        st.error("❌ Student may NEED IMPROVEMENT")

    # Prediction probability
    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities) * 100

    st.info(f"Model Confidence: {confidence:.2f}%")

    # Show entered information
    st.subheader("Student Details")

    result_table = pd.DataFrame({
        "Parameter": [
            "Study Hours",
            "Attendance",
            "Previous Score"
        ],
        "Value": [
            f"{study_hours} hours/day",
            f"{attendance}%",
            f"{previous_score}%"
        ]
    })

    st.table(result_table)

# -----------------------------
# About Project
# -----------------------------

st.divider()

st.subheader("📌 About This Project")

st.write(
    """
    This project uses Machine Learning to predict student performance.
    A Decision Tree Classification algorithm is used.

    The prediction is based on:
    - Study Hours
    - Attendance
    - Previous Exam Score
    """
)