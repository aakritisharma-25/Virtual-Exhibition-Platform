import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("Clustered_Exhibition_Data.csv")

# Encode categorical columns
encoders = {}
for col in ['Nationality', 'Gender', 'ExhibitionRole']:
    encoders[col] = LabelEncoder()
    df[col + '_enc'] = encoders[col].fit_transform(df[col].astype(str))

features = df[['Nationality_enc', 'Gender_enc', 'ExhibitionRole_enc']]
knn_model = NearestNeighbors(n_neighbors=5)
knn_model.fit(features)

# Sidebar navigation
tab = st.sidebar.radio("Navigation", [
    "Recommendation System", "Sample User Output", "How it Works", "Evaluation Metrics", "Workflow"
])

# Helper to calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return precision, recall, f1

# Cluster Visualization
def show_pca_plot():
    cluster_features = [
        'ExhibitionTitle', 'ExhibitionBeginDate', 'ExhibitionEndDate',
        'ExhibitionRole', 'ConstituentType', 'DisplayName',
        'Nationality', 'ConstituentBeginDate', 'ConstituentEndDate', 'Gender'
    ]
    df_viz = df[cluster_features].fillna("Unknown").astype(str)
    for col in df_viz.columns:
        df_viz[col] = LabelEncoder().fit_transform(df_viz[col])
    X_scaled = StandardScaler().fit_transform(df_viz)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='Set2')
    plt.title("Cluster Visualization (PCA)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter)
    st.pyplot(plt)

# --- Main Tab ---
if tab == "Recommendation System":
    st.title("🎨 Exhibition Recommendation System")
    st.markdown("Choose your preferences from available data:")

    nationalities = sorted(df['Nationality'].dropna().unique())
    genders = sorted(df['Gender'].dropna().unique())
    roles = sorted(df['ExhibitionRole'].dropna().unique())

    nationality = st.selectbox("Select Nationality", nationalities)
    gender = st.selectbox("Select Gender", genders)
    role = st.selectbox("Select Role", roles)
    k_value = st.slider("Select number of recommendations (k)", 1, 10, 5)

    if st.button("Get Recommendations"):
        user_vec = [
            encoders['Nationality'].transform([nationality])[0],
            encoders['Gender'].transform([gender])[0],
            encoders['ExhibitionRole'].transform([role])[0]
        ]
        knn_model.set_params(n_neighbors=k_value)
        distances, indices = knn_model.kneighbors([user_vec])
        recommended = df.iloc[indices[0]]

        st.subheader("🔗 Recommendations:")
        for _, row in recommended.iterrows():
            link = f"https://www.moma.org/calendar/exhibitions/{str(row['ExhibitionURL']).split('/')[-1]}"
            st.markdown(f"- [{row['ExhibitionTitle']}]({link})")

# --- Sample User Output Tab ---
elif tab == "Sample User Output":
    st.title("👤 Sample User Output")
    sample_user = df[['Nationality', 'Gender', 'ExhibitionRole']].drop_duplicates().sample(1).iloc[0]

    st.write("Sample User Traits:")
    st.write(f"**Nationality:** {sample_user['Nationality']}")
    st.write(f"**Gender:** {sample_user['Gender']}")
    st.write(f"**Role:** {sample_user['ExhibitionRole']}")

    user_vec = [
        encoders['Nationality'].transform([sample_user['Nationality']])[0],
        encoders['Gender'].transform([sample_user['Gender']])[0],
        encoders['ExhibitionRole'].transform([sample_user['ExhibitionRole']])[0]
    ]
    distances, indices = knn_model.kneighbors([user_vec])
    recommended = df.iloc[indices[0]]

    st.subheader("🔗 Recommendations:")
    for _, row in recommended.iterrows():
        link = f"https://www.moma.org/calendar/exhibitions/{str(row['ExhibitionURL']).split('/')[-1]}"
        st.markdown(f"- [{row['ExhibitionTitle']}]({link})")

# --- How it Works Tab ---
elif tab == "How it Works":
    st.title("📊 How it Works")
    st.write("The following PCA graph shows how exhibitions are grouped into clusters.")
    show_pca_plot()
    st.markdown("""
    **Cluster Info (Example):**
    - Cluster 0: Modern Art, International Artists
    - Cluster 1: Local Artists, Gender-Diverse
    - Cluster 2: Historical Retrospectives
    """)

# --- Evaluation Metrics Tab ---
elif tab == "Evaluation Metrics":
    st.title("📈 Evaluation Metrics")

    # Sample simulated predictions
    y_true = [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
    y_pred = [1, 0, 1, 1, 1, 0, 0, 1, 0, 1]

    precision, recall, f1 = calculate_metrics(y_true, y_pred)
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")

    conf_matrix = confusion_matrix(y_true, y_pred)
    st.write("**Confusion Matrix:**")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=["Not Relevant", "Relevant"],
                yticklabels=["Not Relevant", "Relevant"])
    st.pyplot(fig)

# --- Workflow Tab ---
elif tab == "Workflow":
    st.title("📌 Project Workflow")
    st.markdown("Details coming soon...")
