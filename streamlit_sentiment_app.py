import streamlit as st
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import time
import os
import zipfile
import requests
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def download_model_from_release():
    """Download model from GitHub release if not present"""
    model_dir = Path("model")
    
    # Check if model already exists
    if model_dir.exists() and (model_dir / "config.json").exists():
        return True
    
    try:
        st.info("📥 Model not found. Downloading from GitHub release...")
        
        # UPDATE THIS URL with your actual GitHub release URL
        # Format: https://github.com/USERNAME/REPO/releases/download/TAG/FILE.zip
        release_url = "https://github.com/muhammadhoud/Sentiment-Analysis-App/releases/tag/v1.0.0/model.zip"
        
        # Create model directory
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        response = requests.get(release_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        if response.status_code == 404:
            st.error("❌ Model file not found in GitHub releases. Please upload the model.")
            st.info("""
            **To fix this:**
            1. Go to your GitHub repository
            2. Click "Releases" → "Create a new release"
            3. Upload `sentiment_bert_model.zip`
            4. Update the `release_url` in the code with your actual URL
            """)
            return False
        
        zip_path = "model_temp.zip"
        progress_bar = st.progress(0)
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress_bar.progress(downloaded / total_size)
        
        progress_bar.empty()
        st.info("📦 Extracting model files...")
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        
        # Clean up
        os.remove(zip_path)
        
        st.success("✅ Model downloaded successfully!")
        time.sleep(2)
        st.rerun()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error downloading model: {str(e)}")
        st.info("""
        **Alternative options:**
        1. Add model files directly to your GitHub repository in `model/` folder
        2. Use Hugging Face Hub (change model_path to 'username/model-name')
        3. Check if the release URL is correct
        """)
        return False

class SentimentPredictor:
    """Handles predictions using trained sentiment analysis models."""
    
    def __init__(self, model_path: str = "model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.class_names = ['Negative', 'Neutral', 'Positive']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @st.cache_resource
    def load_model(_self):
        """Load model and tokenizer (cached)"""
        try:
            _self.model = AutoModelForSequenceClassification.from_pretrained(_self.model_path)
            _self.tokenizer = AutoTokenizer.from_pretrained(_self.model_path)
            _self.model.to(_self.device)
            _self.model.eval()
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_single(self, text: str) -> Dict:
        """Predict sentiment for a single text"""
        if self.model is None or self.tokenizer is None:
            if not self.load_model():
                return None
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
        
        confidence = probabilities[0][predicted_class].item()
        
        class_probabilities = {
            self.class_names[i]: probabilities[0][i].item()
            for i in range(len(self.class_names))
        }
        
        return {
            'text': text,
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'class_probabilities': class_probabilities
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment for multiple texts"""
        results = []
        progress_bar = st.progress(0)
        
        for i, text in enumerate(texts):
            result = self.predict_single(text)
            if result:
                results.append(result)
            progress_bar.progress((i + 1) / len(texts))
        
        progress_bar.empty()
        return results

def plot_confidence_gauge(confidence: float, sentiment: str):
    """Create a gauge chart for confidence score"""
    color = "#28a745" if sentiment == "Positive" else "#dc3545" if sentiment == "Negative" else "#ffc107"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff9c4'},
                {'range': [75, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_probability_distribution(class_probabilities: Dict):
    """Create a bar chart for probability distribution"""
    sentiments = list(class_probabilities.keys())
    probabilities = [class_probabilities[s] * 100 for s in sentiments]
    colors = ['#dc3545', '#ffc107', '#28a745']
    
    fig = go.Figure(data=[
        go.Bar(
            x=sentiments,
            y=probabilities,
            marker_color=colors,
            text=[f'{p:.1f}%' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Probability Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def display_result(result: Dict):
    """Display prediction result with styling"""
    sentiment = result['predicted_class']
    confidence = result['confidence']
    
    # Sentiment emoji
    emoji = "😊" if sentiment == "Positive" else "😐" if sentiment == "Neutral" else "😞"
    
    # Sentiment box with styling
    if sentiment == "Positive":
        st.markdown(f'<div class="sentiment-positive">', unsafe_allow_html=True)
    elif sentiment == "Negative":
        st.markdown(f'<div class="sentiment-negative">', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="sentiment-neutral">', unsafe_allow_html=True)
    
    st.markdown(f"## {emoji} Sentiment: **{sentiment}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_gauge = plot_confidence_gauge(confidence, sentiment)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        fig_prob = plot_probability_distribution(result['class_probabilities'])
        st.plotly_chart(fig_prob, use_container_width=True)

def main():
    # Check if model exists, if not download it
    model_dir = Path("model")
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        st.warning("⚠️ Model files not found locally.")
        
        if st.button("📥 Download Model from GitHub Release"):
            download_model_from_release()
            return
        
        st.info("""
        **Setup Instructions:**
        1. Click the button above to download the model from GitHub release, OR
        2. Add model files directly to `model/` folder in your repository, OR
        3. Use Hugging Face Hub (update `model_path` to 'username/model-name')
        
        **Model files needed:**
        - config.json
        - pytorch_model.bin
        - tokenizer_config.json
        - vocab.txt
        - special_tokens_map.json
        """)
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">🎭 Sentiment Analysis App</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.info(
            "This app uses a fine-tuned BERT model to analyze sentiment in text. "
            "It classifies text into three categories: Positive, Neutral, or Negative."
        )
        
        st.header("🔧 Model Info")
        st.write("**Base Model:** BERT (bert-base-uncased)")
        st.write("**Classes:** Negative, Neutral, Positive")
        st.write("**Max Length:** 128 tokens")
        
        st.header("💡 Tips")
        st.write("- Enter clear and complete sentences")
        st.write("- Longer texts provide better context")
        st.write("- Try different types of feedback")
    
    # Initialize predictor
    predictor = SentimentPredictor()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Text", "📄 Batch Analysis", "ℹ️ Examples"])
    
    with tab1:
        st.header("Analyze Single Text")
        
        # Text input
        user_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here..."
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if analyze_button:
            if user_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    result = predictor.predict_single(user_input)
                    
                    if result:
                        st.success("Analysis complete!")
                        display_result(result)
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    with tab2:
        st.header("Batch Analysis")
        st.write("Analyze multiple texts at once")
        
        # Text area for multiple inputs
        batch_input = st.text_area(
            "Enter multiple texts (one per line):",
            height=200,
            placeholder="Text 1\nText 2\nText 3\n..."
        )
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Or upload a CSV file with a 'text' column",
            type=['csv']
        )
        
        analyze_batch = st.button("🔍 Analyze Batch", type="primary")
        
        if analyze_batch:
            texts = []
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    texts = df['text'].astype(str).tolist()
                else:
                    st.error("CSV must have a 'text' column")
            elif batch_input.strip():
                texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
            
            if texts:
                with st.spinner(f"Analyzing {len(texts)} texts..."):
                    results = predictor.predict_batch(texts)
                
                if results:
                    st.success(f"✅ Analyzed {len(results)} texts!")
                    
                    # Create results dataframe
                    results_df = pd.DataFrame([
                        {
                            'Text': r['text'][:100] + '...' if len(r['text']) > 100 else r['text'],
                            'Sentiment': r['predicted_class'],
                            'Confidence': f"{r['confidence']:.2%}",
                            'Negative': f"{r['class_probabilities']['Negative']:.2%}",
                            'Neutral': f"{r['class_probabilities']['Neutral']:.2%}",
                            'Positive': f"{r['class_probabilities']['Positive']:.2%}"
                        }
                        for r in results
                    ])
                    
                    # Display results table
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("📊 Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    sentiment_counts = results_df['Sentiment'].value_counts()
                    
                    with col1:
                        st.metric("😞 Negative", sentiment_counts.get('Negative', 0))
                    with col2:
                        st.metric("😐 Neutral", sentiment_counts.get('Neutral', 0))
                    with col3:
                        st.metric("😊 Positive", sentiment_counts.get('Positive', 0))
                    
                    # Pie chart
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'Negative': '#dc3545',
                            'Neutral': '#ffc107',
                            'Positive': '#28a745'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("⚠️ Please enter texts or upload a file.")
    
    with tab3:
        st.header("Example Texts")
        st.write("Click on any example to analyze it:")
        
        examples = {
            "Positive Examples": [
                "I absolutely love this product! It's amazing and works perfectly.",
                "Excellent customer service! They were very helpful and responsive.",
                "Outstanding! Exceeded all my expectations!"
            ],
            "Negative Examples": [
                "This is terrible. I'm very disappointed with the quality.",
                "Poor quality materials, not worth the money at all.",
                "Not happy with the purchase, will not recommend."
            ],
            "Neutral Examples": [
                "The product is okay, nothing special but it works.",
                "It's decent for the price, but could be better.",
                "Average experience, met basic expectations."
            ]
        }
        
        for category, example_list in examples.items():
            st.subheader(category)
            for i, example in enumerate(example_list):
                if st.button(f"📝 {example}", key=f"example_{category}_{i}"):
                    with st.spinner("Analyzing..."):
                        result = predictor.predict_single(example)
                        if result:
                            display_result(result)

if __name__ == "__main__":
    main()
