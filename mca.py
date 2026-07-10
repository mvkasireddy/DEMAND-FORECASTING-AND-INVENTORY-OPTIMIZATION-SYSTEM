import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import pickle
import requests
import urllib.parse
import matplotlib.pyplot as plt

# RAG IMPORT
from rag_pdf_pipeline import create_vector_db, retrieve_context


# LOAD MODEL + ENCODERS

@st.cache_resource
def load_artifacts():
    with open("xgb_demand_forecasting_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    return model, encoders

model, encoders = load_artifacts()


# LOAD RAG

@st.cache_resource
def load_rag():
    return create_vector_db()

vector_db = load_rag()

# SESSION STATE

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "report" not in st.session_state:
    st.session_state.report = ""

# UI

st.title(" Demand Forecasting System")
st.divider()

st.header("Input Features")

price = st.number_input("Price", value=500.0)
discount = st.number_input("Discount (%)", value=10)
inventory_level = st.number_input("Inventory Level", value=100)
promotion = st.selectbox("Promotion", [0, 1])
competitor_pricing = st.number_input("Competitor Pricing", value=450.0)

category_col_name = "Category" if "Category" in encoders else "category"

category = st.selectbox(
    "Category",
    encoders[category_col_name].classes_.tolist()
)

# INPUT DATA

input_data = pd.DataFrame({
    "Price": [price],
    "Discount": [discount],
    "Inventory Level": [inventory_level],
    "Promotion": [promotion],
    "Competitor Pricing": [competitor_pricing],
    "Category": [category]
})

for col, encoder in encoders.items():
    if col in input_data.columns:
        input_data[col] = encoder.transform(input_data[col])

st.divider()

# PREDICT BUTTON

if st.button(" Predict Demand"):
    prediction = model.predict(input_data)[0]

    st.session_state.prediction_done = True
    st.session_state.last_prediction = int(max(0, prediction))
    st.session_state.chat_history = []
    st.session_state.report = ""

# MAIN OUTPUT SECTION

if st.session_state.prediction_done:

    # Prediction
    st.success(f" Predicted Demand: {st.session_state.last_prediction} units")

    #  GRAPH (FIXED SIZE)

    st.subheader(" Demand Insights")

    fig, ax = plt.subplots(figsize=(5, 2.5))  # smaller graph
    ax.bar(
        ["Demand", "Inventory", "Competitor"],
        [
            st.session_state.last_prediction,
            inventory_level,
            competitor_pricing
        ]
    )
    ax.set_title("Demand Comparison", fontsize=10)

    st.pyplot(fig)

    #  REPORT SECTION
    st.subheader(" Report & Actions")

    col1, col2, col3 = st.columns(3)

    # Generate Report
    with col1:
        if st.button(" Generate"):
            d = st.session_state.last_prediction

            demand_level = "High" if d > 150 else "Moderate" if d > 50 else "Low"
            price_status = "Higher" if price > competitor_pricing else "Competitive"

            st.session_state.report = f"""
Demand Forecast Report

Category: {category}
Predicted Demand: {d}

Price: {price}
Competitor Price: {competitor_pricing}
Discount: {discount}%
Inventory: {inventory_level}
Promotion: {promotion}

Analysis:
- Demand Level: {demand_level}
- Price Status: {price_status}

Recommendations:
- Adjust pricing
- Optimize discount
- Manage inventory
"""

    # WhatsApp
    with col2:
        phone = st.text_input("📱 Number", key="phone")

        if st.button("📤 Send"):
            if st.session_state.report and phone:
                url = f"https://api.whatsapp.com/send?phone={phone}&text={urllib.parse.quote(st.session_state.report)}"
                st.markdown(f"[👉 Send Message]({url})")
            else:
                st.warning("Generate report first")

    # Download
    with col3:
        if st.session_state.report:
            st.download_button(
                "📥 Download",
                st.session_state.report,
                file_name="report.txt"
            )

    # Show report
    if st.session_state.report:
        st.text_area(" Report", st.session_state.report, height=150)

    # AI FUNCTION
    def local_llm_response(user_query):
        context = retrieve_context(vector_db, user_query)

        prompt = f"""
Retail expert assistant.

Demand: {st.session_state.last_prediction}
Price: {price}
Inventory: {inventory_level}

Context:
{context}

Question: {user_query}

Answer:
- Reason:
- Action:
"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=300
            )

            if response.status_code == 200:
                return response.json()["response"]
            else:
                return response.text

        except Exception as e:
            return str(e)

    # AI CHAT (LAST)
    st.subheader(" AI Assistant")

    user_q = st.text_input("Ask question")

    if st.button("Ask AI"):
        if user_q:
            res = local_llm_response(user_q)
            st.session_state.chat_history.append(f" {user_q}")
            st.session_state.chat_history.append(f" {res}")

    for msg in st.session_state.chat_history:
        st.write(msg)