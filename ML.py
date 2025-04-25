import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="🌲 TPO Clustering Dashboard 🪵", layout="wide")
st.markdown(
    "<h2 style='color:#006600;'>🌲 Timber Product Output Data Clustering Dashboard 🪵</h2>",
    unsafe_allow_html=True
)

st.markdown("---")

# Load data and model
df = pd.read_csv("cleaned_data.csv")
clustering_data = df[['GREEN_TONS', 'MCFVOL']].dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)
gmm = joblib.load("gmm.pkl")

# Predict clusters
result = df.copy()
result['Cluster'] = gmm.predict(scaled_data)


# Sidebar - User input for prediction
with st.sidebar.form("predict_form"):
    st.header("Predict Cluster")
    green_tons_input = st.number_input("GREEN_TONS - Green Timber Weight in tonne", min_value=0.01, value=10.0)
    mcfvol_input = st.number_input("MCFVOL - Million Cubic Feet Volume", min_value=0.01, value=50.0)
    
    # Predict cluster for user input
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #87cefa;    
            color: navy;                  
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5em 1em;
            border: none;
        }
        div.stButton > button:first-child:hover {
            background-color: #DABFFF;   
            color: #665687;
        }
        </style>
    """, unsafe_allow_html=True)
    submitted = st.form_submit_button("Predict")
    if submitted:
        user_input_scaled = scaler.transform([[green_tons_input, mcfvol_input]])
        predicted_cluster = gmm.predict(user_input_scaled)[0]
        st.success(f"Predicted Cluster: {predicted_cluster}")


# Cluster summary
cluster_counts = result['Cluster'].value_counts().sort_index()

# Mean and Std Dev by Cluster
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
numeric_columns.append('Cluster')
mean_result = result[numeric_columns].groupby('Cluster').mean().T
var_result = result[numeric_columns].groupby('Cluster').std().T

col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Cluster Counts")
    st.dataframe(cluster_counts)
with col2:
    st.subheader("📉 Cluster Means")
    st.dataframe(mean_result.style.background_gradient(cmap="Blues").format("{:.2f}"))

st.subheader("📊 Cluster Standard Deviations")
st.dataframe(var_result.style.background_gradient(cmap="Reds").format("{:.2f}"))

# Custom color palette
color_palette = sns.color_palette(["#00B4D8", "#90E0EF", "#CAF0F8", "#0077B6", "#023E8A"])

# Plotting
st.subheader("Visualizations")

# Import color palettes from seaborn
color_palette = sns.color_palette("crest")  

# Product Type Distribution
with st.expander("📊 Product Type Distribution by Cluster"):
    st.write("### Product Type Distribution by Cluster")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.countplot(data=result, x='PRODCD_MEANING', hue='Cluster', ax=ax1, palette=color_palette)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.set_xlabel("Product Type")  
    st.pyplot(fig1)

# Species Group Distribution
with st.expander("📊 Species Group Distribution by Cluster"):
    st.write("### Species Group Distribution by Cluster")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.countplot(data=result, x='SPGRP_NAME', hue='Cluster', ax=ax2, palette=color_palette)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=80)
    ax2.set_xlabel("Species Group")  
    st.pyplot(fig2)

# Source Group Distribution
with st.expander("📊 Source Group Distribution by Cluster"):
    st.write("### Source Group Distribution by Cluster")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.countplot(data=result, x='SOURCECD_MEANING', hue='Cluster', ax=ax3, palette=color_palette)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    ax3.set_xlabel("Source Group")  
    st.pyplot(fig3)

# Owner Group Distribution
with st.expander("📊 Owner Group Distribution by Cluster"):
    st.write("### Owner Group Distribution by Cluster")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.countplot(data=result, x='OWNER_MEANING', hue='Cluster', ax=ax4, palette=color_palette)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    ax4.set_xlabel("Owner Group")  
    st.pyplot(fig4)

# State Distribution by Cluster
with st.expander("📍 State Distribution by Cluster"):
    st.write("### State Distribution by Cluster")
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    sns.countplot(data=result, x='STATE_NAME', hue='Cluster', ax=ax5, palette=color_palette)
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
    ax5.set_xlabel("State")  
    st.pyplot(fig5)

# Boxplots
st.subheader("📦 Boxplots by Cluster")
def boxplot_column(col_name, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=result, x='Cluster', y=col_name, ax=ax, palette=color_palette)
    ax.set_title(title)
    st.pyplot(fig)

boxplot_column('GREEN_TONS', "Green Tons by Cluster")
boxplot_column('RPA_STD_AMOUNT', "Standard Amount by Cluster")
boxplot_column('MCFVOL', "MCF Volume by Cluster")

# Dominant characteristics
st.subheader("🏷️ Dominant Characteristics by Cluster")
st.dataframe(
    result.groupby("Cluster").agg({
        "PRODCD_MEANING": lambda x: x.value_counts().index[0],
        "SPGRP_NAME": lambda x: x.value_counts().index[0],
        "SOURCECD_MEANING": lambda x: x.value_counts().index[0],
        "MCFVOL": "mean",
        "GREEN_TONS": "mean",
        "RPA_STD_AMOUNT": "mean"
    })
)

# Value per ton
result["value_per_ton"] = result["RPA_STD_AMOUNT"] / result["GREEN_TONS"]
st.subheader("🔢 Value per Ton by Cluster")
st.dataframe(result.groupby("Cluster")["value_per_ton"].describe())

# Footer
st.markdown("---")
st.markdown("© 2025 Forestry Analytics Team @ TARUMT RDS2S3 G2-G6")
