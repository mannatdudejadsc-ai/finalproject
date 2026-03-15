import streamlit as st
import torch
import torch.nn.functional as F
from model import HybridRumourModel
import plotly.graph_objects as go
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
import json
import os

# Load MiniLM once
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Page Config
st.set_page_config(page_title="Rumour Detection", page_icon="📰", layout="wide")

# Styling
st.markdown("""
<style>

.main {
    background-color: #0b132b;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    background-color: white;
    color: black;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    margin-top: 20px;
}

.result-box h2 {
    color: #0d47a1;
}

.result-box p {
    color: black;
}

.stButton > button {
    border-radius: 10px;
    background-color: #0d47a1;
    color: white;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("📰 Rumour Detection: Hybrid GGNN + GAT")
st.markdown("### Early detection of rumours on social media using Graph Neural Networks")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This application uses a hybrid Deep Learning model combining:
- GGNN (Gated Graph Neural Networks): For temporal propagation.
- GAT (Graph Attention Networks): For identifying influential nodes.
""")

# Load Model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridRumourModel(num_features=384, hidden_dim=64).to(device)

    try:
        model.load_state_dict(torch.load("hybrid_rumour_model.pth", map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("Model file not found. Please run training first.")
        return None, None


model, device = load_model()

# User Input
input_text = st.text_area(
    "Enter News Text / Tweet Content",
    height=100,
    placeholder="Type the content here to analyze..."
)

# Analyze Button
if st.button("Analyze"):

    if not input_text.strip():
        st.warning("Please enter some text to analyze.")

    elif model:

        with st.spinner("Analyzing propagation patterns..."):

            # Convert text → MiniLM embedding
            embedding = embedder.encode(input_text)

            x = torch.tensor(embedding, dtype=torch.float).unsqueeze(0).to(device)

            edge_index = torch.empty((2, 0), dtype=torch.long).to(device)

            batch = torch.tensor([0]).to(device)

            data = Data(x=x, edge_index=edge_index, batch=batch)

            # Prediction
            with torch.no_grad():
                out = model(data)
                probs = torch.exp(out)
                pred_class = out.argmax(dim=1).item()
                confidence = probs[0][pred_class].item()

        labels = ["Non-Rumour", "Rumour"]
        prediction = labels[pred_class]

        # Display Results
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="result-box">
                <h3>Prediction Result</h3>
                <h2>{prediction}</h2>
                <p>Confidence Score: <strong>{confidence:.4f}</strong></p>
            </div>
            """, unsafe_allow_html=True)

        with col2:

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={'text': "Confidence (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#1a237e"},
                    'steps': [
                        {'range': [0, 50], 'color': "#e8eaf6"},
                        {'range': [50, 80], 'color': "#c5cae9"},
                        {'range': [80, 100], 'color': "#9fa8da"}
                    ]
                }
            ))

            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Model Comparison Chart
        st.markdown("### Model Architecture Performance")

        if os.path.exists("model_metrics.json"):

            with open("model_metrics.json") as f:
                metrics = json.load(f)

            architectures = list(metrics.keys())
            accuracies = list(metrics.values())

        else:

            architectures = ['GAT (Baseline)', 'GGNN (Baseline)', 'Hybrid (Ours)']
            accuracies = [0, 0, 0]
        
        colors = ['#bdbdbd', '#bdbdbd', '#1565c0']

        fig_bar = go.Figure(data=[go.Bar(
            x=architectures,
            y=accuracies,
            marker_color=colors,
            text=accuracies,
            textposition='auto'
        )])

        fig_bar.update_layout(
            title="Validation Accuracy Comparison",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1]),
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_bar, use_container_width=True)