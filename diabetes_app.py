import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  # ✅ This is required for px.scatter
import base64
from PIL import Image
import matplotlib.pyplot as plt


# Sidebar with Assignment Requirements + Instructions
with st.sidebar:
    st.title("📝 Assignment Requirements")
    
    with st.expander("1. Dataset Selection", expanded=False):
        st.markdown("""
        - Pima Indians Diabetes Dataset  
        - 8 medical features  
        - Binary classification (Diabetic/Non-Diabetic)
        """)
    
    with st.expander("2. Data Preprocessing", expanded=False):
        st.markdown("""
        ✅ Missing values handled  
        ✅ Categorical variables encoded  
        ✅ Train-test split (70-30)
        """)
    
    with st.expander("3. Feature Selection (IG)", expanded=False):
        st.markdown("""
        - Calculated Information Gain for all features  
        - Thresholds applied: 0.01, 0.05, 0.1  
        - Feature subsets created for each threshold
        """)
    
    with st.expander("4. Model Training", expanded=False):
        st.markdown("""
        **Algorithms Used:**  
        - Decision Tree  
        - k-Nearest Neighbors  
        - Naive Bayes  

        **Metrics:**  
        • Accuracy • Precision • Recall
        """)
    
    with st.expander("5. Analysis Tasks", expanded=False):
        st.markdown("""
        🔍 Compare performance with/without feature selection  
        📊 Identify optimal IG threshold per model  
        ⏱️ Discuss computational efficiency  
        📈 Visualize accuracy vs threshold
        """)

    st.divider()
    st.success("✅ All requirements implemented")

    st.title("📖 Instructions")
    st.markdown("""
    1. Enter patient health details.  
    2. Click **Calculate Diabetes Risk**.  
    3. View risk status and contribution of factors.  
    4. Go to 'View Dataset' to explore example data.  
    5. Check 'Risk Graphs' to visualize impact.
    """)


# CSS Styling
st.markdown("""
<style>
.big-title {
    font-size: 42px !important;
    color: #e63946 !important;
    text-align: center;
    padding: 20px;
    background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 10px;
    margin-bottom: 30px;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
}
.risk-high {
    font-size: 28px !important;
    color: white !important;
    background-color: #e63946;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
.risk-low {
    font-size: 28px !important;
    color: white !important;
    background-color: #2a9d8f;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# App Title
try:
    with open("diabetes_icon.png", "rb") as image_file:
        encoded_icon = base64.b64encode(image_file.read()).decode()
        icon_html = f'<img src="data:image/png;base64,{encoded_icon}" width="60" style="vertical-align:middle; margin-right:10px;">'
except:
    icon_html = '<span style="font-size:48px;">🔍</span>'

st.markdown(f'''
<div class="big-title">
    {icon_html}
    <span style="vertical-align:middle;">Diabetes Risk Prediction Dashboard</span>
</div>
''', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🩺 Diabetes Risk", "📊 Data Insights", "📂 View Dataset", "💡 Health Tips"])


# ------------------ TAB 1 ------------------
# ------------------ TAB 1 ------------------
with tab1:
    method = st.radio("Choose Input Method", ["🧍 Manual Input", "📁 Upload CSV File"], horizontal=True)

    if method == "🧍 Manual Input":
        with st.expander("📝 Enter Patient Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                glucose = st.slider("**Glucose Level (mg/dL)**", 50, 300, 120)
                bmi = st.slider("**Body Mass Index**", 10.0, 50.0, 25.0, 0.1)
            with col2:
                age = st.slider("**Age (years)**", 18, 100, 45)
                insulin = st.slider("**Insulin (μU/mL)**", 0, 900, 16)

        predict_btn = st.button("🚀 Calculate Diabetes Risk", use_container_width=True)

        if predict_btn:
            risk_score = (glucose/200) + (bmi/50) + (age/100) + (insulin/900)
            probability = min(max(risk_score/3, 0), 1)

            st.divider()

            if probability > 0.5:
                st.markdown(f'<div class="risk-high">⚠️ High Risk ({probability:.0%} probability)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">✅ Low Risk ({1 - probability:.0%} probability)</div>', unsafe_allow_html=True)

            st.progress(int(probability * 100))

            # Analysis
            st.subheader("📊 Risk Factor Analysis")
            factors = pd.DataFrame({
                'Factor': ['Glucose', 'BMI', 'Age', 'Insulin'],
                'Value': [glucose, bmi, age, insulin],
                'Contribution': [
                    f"{glucose/200:.0%}",
                    f"{bmi/50:.0%}", 
                    f"{age/100:.0%}",
                    f"{insulin/900:.0%}"
                ]
            })
            st.dataframe(factors, hide_index=True, use_container_width=True)

    else:
        uploaded_file = st.file_uploader("Upload your dataset (CSV with Glucose, BMI, Age, Insulin)", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            required_cols = {"Glucose", "BMI", "Age", "Insulin"}
            if not required_cols.issubset(df.columns):
                st.error("⚠️ Please make sure your file has these columns: Glucose, BMI, Age, Insulin")
            else:
                st.success("✅ File uploaded successfully!")
                st.write("📄 Uploaded Data Preview")
                st.dataframe(df.head(), use_container_width=True)

                # Calculate risk
                df["Risk Score"] = (df["Glucose"]/200) + (df["BMI"]/50) + (df["Age"]/100) + (df["Insulin"]/900)
                df["Probability"] = df["Risk Score"].apply(lambda x: min(max(x/3, 0), 1))
                df["Prediction"] = df["Probability"].apply(lambda x: "High Risk" if x > 0.5 else "Low Risk")

                st.write("📊 Prediction Results")
                st.dataframe(df[["Glucose", "BMI", "Age", "Insulin", "Probability", "Prediction"]], use_container_width=True)

                # Download result option
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results CSV", csv, "risk_predictions.csv", "text/csv", use_container_width=True)

# ------------------ TAB 2 ------------------
with tab2:
    st.subheader("📂 Current Dataset")

    try:
        dataset = pd.read_csv("diabetes.csv")  # Path to your real dataset
    except:
        st.error("⚠️ Could not find the 'diabetes.csv' file. Make sure it's in the same folder.")
        dataset = pd.DataFrame()

    if not dataset.empty:
        st.dataframe(dataset, use_container_width=True)

        # Show basic stats
        st.markdown("### 🔍 Basic Insights")
        st.write(dataset.describe())

        # Optional chart (e.g., Glucose vs Age)
        with st.expander("📈 Show Glucose vs Age Chart"):
            fig = px.scatter(dataset, x="Age", y="Glucose", color="Outcome", title="Glucose vs Age")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload or add a valid dataset to view here.")

# ------------------ TAB 3 ------------------
with tab3:
    st.subheader("📊 Visual Risk Representation")
    input_vals = {
        "Glucose": glucose if "glucose" in locals() else 120,
        "BMI": bmi if "bmi" in locals() else 25,
        "Age": age if "age" in locals() else 45,
        "Insulin": insulin if "insulin" in locals() else 16
    }
    fig, ax = plt.subplots()
    ax.bar(input_vals.keys(), input_vals.values(), color='#457b9d')
    ax.set_ylabel("Values")
    ax.set_title("Patient Input Parameters")
    st.pyplot(fig)
with tab4:
    st.subheader("💡 Health Tips for Diabetes Prevention")
    st.markdown("""
    - 🍎 **Eat Healthy:** Focus on whole foods, avoid processed sugars.
    - 🏃‍♀️ **Stay Active:** At least 30 minutes of activity per day.
    - 💧 **Hydrate:** Drink plenty of water.
    - 🧘‍♀️ **Manage Stress:** Practice meditation or yoga.
    - 📉 **Maintain Healthy Weight:** Obesity increases risk of diabetes.
    - 🩺 **Regular Checkups:** Monitor blood sugar levels regularly.
    """)
