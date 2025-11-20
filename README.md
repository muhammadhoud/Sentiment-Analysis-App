# Customer Feedback Sentiment Analysis with BERT

![Sentiment Analysis](https://img.shields.io/badge/Task-Sentiment%20Analysis-blue)
![BERT](https://img.shields.io/badge/Model-BERT-yellow)
![HuggingFace](https://img.shields.io/badge/Framework-HuggingFace-orange)

A comprehensive solution for classifying customer feedback sentiment using fine-tuned BERT model. This project demonstrates how to build, train, and deploy a state-of-the-art sentiment analysis system that can accurately classify customer feedback into positive, negative, or neutral categories.

## Project Overview

**Problem Statement**: Perform automated sentiment classification on customer feedback to determine whether the sentiment is positive, negative, or neutral.

**Solution**: Fine-tune a BERT-based model for sequence classification with comprehensive preprocessing, training, and evaluation pipeline.

### Key Features
- ✅ **Fine-tuned BERT Model** for accurate sentiment classification
- ✅ **Comprehensive Preprocessing** with text cleaning and normalization
- ✅ **Advanced Visualization** for data analysis and model performance
- ✅ **Real-time Predictions** with confidence scores
- ✅ **Production-ready Pipeline** with modular architecture

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/customer-sentiment-analysis.git
cd customer-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Install specific packages
pip install torch transformers datasets pandas numpy matplotlib seaborn scikit-learn
```

### Basic Usage

```python
from sentiment_analysis import main_sentiment_analysis

# Run the complete pipeline
main_sentiment_analysis()
```

### Quick Prediction

```python
from sentiment_predictor import SentimentPredictor

# Initialize predictor
predictor = SentimentPredictor()

# Single prediction
feedback = "I absolutely love this product! It's amazing and works perfectly."
result = predictor.predict_single(feedback)

print(f"Sentiment: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['class_probabilities']}")
```


## Dataset

### Source
- **Dataset**: Customer Feedback Dataset from Kaggle
- **URL**: [Kaggle Dataset](https://www.kaggle.com/datasets/vishweshsalodkar/customer-feedback-dataset?select=sentiment-analysis.csv)

### Statistics
- **Total Samples**: 85
- **Positive**: 42 samples (49.4%)
- **Negative**: 33 samples (38.8%)
- **Neutral**: 10 samples (11.8%)

### Sample Data
| Text | Sentiment |
|------|-----------|
| "I love this product! It's amazing." | Positive |
| "Terrible quality, very disappointed." | Negative |
| "The product is okay, nothing special." | Neutral |

## 🏗️ Model Architecture

### Base Model
- **Model**: `bert-base-uncased`
- **Parameters**: 110 million
- **Task**: Sequence Classification
- **Output Classes**: 3 (Negative, Neutral, Positive)

### Training Configuration
```python
training_args = {
    'num_train_epochs': 5,
    'per_device_train_batch_size': 16,
    'learning_rate': 2e-5,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'max_length': 128
}
```

## 📈 Performance Metrics

### Overall Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | 88.89% |
| **Precision** | 58.33% |
| **Recall** | 66.67% |
| **F1-Score (Macro)** | 61.90% |
| **F1-Score (Weighted)** | 84.13% |

### Class-wise Performance
- **Negative**: Precision 75.0%, Recall 100.0%, F1 85.7%
- **Neutral**: Precision 0.0%, Recall 0.0%, F1 0.0%
- **Positive**: Precision 100.0%, Recall 100.0%, F1 100.0%

*Note: Neutral class performance is lower due to class imbalance*

## 🔧 Usage

### 1. Data Preprocessing

```python
from src.preprocessor import SentimentPreprocessor

preprocessor = SentimentPreprocessor(model_name="bert-base-uncased")
df = preprocessor.load_and_preprocess_data("data/sentiment-analysis.csv")
datasets = preprocessor.create_datasets(df)
```

### 2. Model Training

```python
from src.trainer import SentimentTrainer

trainer = SentimentTrainer()
trainer.load_model()
train_result = trainer.train(
    train_dataset=datasets['train'],
    val_dataset=datasets['validation'],
    num_epochs=5
)
```

### 3. Model Evaluation

```python
test_metrics = trainer.evaluate(datasets['test'])
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
```

### 4. Making Predictions

```python
from src.predictor import SentimentPredictor

predictor = SentimentPredictor("models/sentiment_bert")

# Single prediction
result = predictor.predict_single("Excellent product quality!")

# Batch predictions
feedbacks = [
    "Great service and fast delivery!",
    "Poor customer support experience.",
    "The product meets expectations."
]
results = predictor.predict_batch(feedbacks)
```

## 🎯 Example Predictions

### High Confidence Examples
1. **"I absolutely love this product! It's amazing and works perfectly."**
   - ✅ **Predicted**: POSITIVE (79.18% confidence)
   - Probabilities: Negative 4.35%, Neutral 16.47%, Positive 79.18%

2. **"This is terrible. I'm very disappointed with the quality."**
   - ✅ **Predicted**: NEGATIVE (74.84% confidence)
   - Probabilities: Negative 74.84%, Neutral 11.56%, Positive 13.60%

3. **"The product is okay, nothing special but it works."**
   - ✅ **Predicted**: NEUTRAL (52.67% confidence)
   - Probabilities: Negative 9.59%, Neutral 52.67%, Positive 37.74%

## 📊 Visualizations

The project includes comprehensive visualizations:

- **Sentiment Distribution**: Bar chart and pie chart of class distribution
- **Confusion Matrix**: Model performance across classes
- **Metrics Comparison**: Bar chart of accuracy, precision, recall, F1 scores
- **Prediction Confidence**: Confidence scores for example predictions

## 🚀 Deployment

### Option 1: Local API
```python
from fastapi import FastAPI
from src.predictor import SentimentPredictor

app = FastAPI()
predictor = SentimentPredictor()

@app.post("/predict")
async def predict_sentiment(text: str):
    result = predictor.predict_single(text)
    return result
```

### Option 2: Streamlit App
```python
import streamlit as st
from src.predictor import SentimentPredictor

st.title("Customer Sentiment Analysis")
predictor = SentimentPredictor()

text_input = st.text_area("Enter customer feedback:")
if st.button("Analyze Sentiment"):
    result = predictor.predict_single(text_input)
    st.write(f"Sentiment: **{result['predicted_class']}**")
    st.write(f"Confidence: {result['confidence']:.2%}")
```

## 🔧 Configuration

### Model Parameters
```python
# config.py
MODEL_CONFIG = {
    'model_name': 'bert-base-uncased',
    'num_labels': 3,
    'max_length': 128,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 5
}
```

### Data Preprocessing
- Text cleaning and normalization
- Label encoding (Negative→0, Neutral→1, Positive→2)
- Dataset splitting (80% train, 10% validation, 10% test)
- BERT tokenization with truncation and padding

## 🤝 Contributing

We welcome contributions! 

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/customer-sentiment-analysis.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
```

## 📄 License

This project is licensed under the MIT License

## 🙏 Acknowledgments

- **Hugging Face** for the transformers library and BERT model
- **Kaggle** for the customer feedback dataset
- **Google Colab** for computational resources
- **BERT authors** for the foundational model


---

**⭐ If this project helped you, please give it a star on GitHub!**

---

