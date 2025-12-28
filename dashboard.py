import streamlit as st
import requests
import json
import plotly.graph_objects as go
import pandas as pd

# --- CONFIGURATION ---
API_URL = "https://customer-api-7pqx.onrender.com/predict_segment"
st.set_page_config(page_title="Customer Segmentation Analysis", layout="wide")

# --- HEADER ---
st.title("Customer Segmentation Analysis Dashboard")
st.markdown("Use this tool to predict customer behavior and get product recommendations.")
st.divider()

# --- SIDEBAR: INPUTS ---
st.sidebar.header("User Profile")

# 1. Basic Info
gender = st.sidebar.radio("Gender", ["Male", "Female"])
orders = st.sidebar.slider("Past Orders", min_value=0, max_value=20, value=5)

# 2. Brand Preferences (Top 10 Brands for Demo)
st.sidebar.header("Brand Purchases")
st.sidebar.caption("How many times did they buy/search these?")

brands_list = [
    "Jordan", "Gatorade", "Samsung", "Asus", "Udis", 
    "Monters", "Huawei", "Smucker", "Walmart", "Cymbal"
]

selected_brands = {}
for brand in brands_list:
    count = st.sidebar.number_input(f"{brand}", min_value=0, max_value=10, value=0, key=brand)
    if count > 0:
        selected_brands[brand] = count

# --- MAIN ACTION ---
if st.sidebar.button("🔍 Predict Segment", type="primary"):
    
    # 1. Prepare Payload for API
    payload = {
        "Gender": "M" if gender == "Male" else "F",
        "Orders": orders,
        "Brands": selected_brands
    }
    
    # 2. Call the API
    try:
        with st.spinner("Analyzing Customer Pattern..."):
            response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # --- DISPLAY RESULTS ---
            
            # Section A: Key Metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"### Segment: {result['segment_name']}")
                st.metric(label="Cluster ID", value=result['cluster_id'])
                
            with col2:
                st.info("### 💡 Marketing Action")
                if result['cluster_id'] == 0:
                    st.write("👉 Send emails about **Gadgets & Tech**.")
                elif result['cluster_id'] == 1:
                    st.write("👉 Offer discounts on **Fashion & Apparel**.")
                else:
                    st.write("👉 Promote **Home & Kitchen** items.")

            st.divider()

            # Section B: Visual Analysis (Radar Chart)
            st.subheader("📊 Profile Comparison: You vs Segment Average")
            
            # 1. take out data
            user_brands = selected_brands
            cluster_profile = result['cluster_profile'] # data come from API
            
            # only show the comparison of user's choosen brand
            if user_brands:
                categories = list(user_brands.keys())
                
                # User Values
                user_values = list(user_brands.values())
                
                # Cluster Average Values (for same brand)
                cluster_values = [cluster_profile.get(brand, 0) for brand in categories]
                
                # to close the Radar Chart (Start point = End point)
                categories += [categories[0]]
                user_values += [user_values[0]]
                cluster_values += [cluster_values[0]]
                
                fig = go.Figure()

                # Trace 1: The User
                fig.add_trace(go.Scatterpolar(
                    r=user_values,
                    theta=categories,
                    fill='toself',
                    name='This Customer',
                    line_color='blue'
                ))

                # Trace 2: The Segment Average (Real Data)
                fig.add_trace(go.Scatterpolar(
                    r=cluster_values,
                    theta=categories,
                    fill='toself',
                    name=f'Segment Avg',
                    line_color='red',
                    opacity=0.5
                ))

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, max(max(user_values), max(cluster_values)) + 1])),
                    showlegend=True,
                    title="Behavior Comparison"
                )
                
                st.plotly_chart(fig, selection_mode="points") # Default behavior
                st.plotly_chart(fig, width="stretch")
                
                st.caption("🔴 Red Area: What typical people in this group do.")
                st.caption("🔵 Blue Area: What THIS customer is doing.")
            else:
                st.info("Select brands in the sidebar to see the comparison chart.")

        else:
            st.error(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection Error: Is the FastAPI backend running?")
        st.code("uvicorn main:app --reload", language="bash")

# --- FOOTER ---
st.markdown("---")
st.caption("Built with FastAPI & Streamlit • Machine Learning Model v1.0 • Made with ❤️ by Alok Ranjan ")
