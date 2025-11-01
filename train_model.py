"""
Sentiment Analysis Model Training Script
This script can be run independently to train a BERT model for sentiment analysis
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
import argparse
from typing import Dict, List

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

class SentimentPreprocessor:
    """Handles preprocessing and tokenization of customer feedback data"""

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and clean the dataset"""
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df['text'] = df['text'].astype(str).str.replace('"', '').str.strip()
        df['sentiment'] = df['sentiment'].astype(str).str.replace('"', '').str.strip()
        df['label'] = df['sentiment'].map(self.label_map)

        df = df.dropna(subset=['text', 'label'])
        df['label'] = df['label'].astype(int)
        df = df[df['text'].str.len() > 0]
        df = df.drop_duplicates(subset=['text'])

        print(f"✓ Loaded {len(df)} samples")
        print(f"✓ Sentiment distribution:\n{df['sentiment'].value_counts()}")

        return df[['text', 'sentiment', 'label']]

    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize the input texts"""
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'input_ids': tokenized['input_ids'].tolist(),
            'attention_mask': tokenized['attention_mask'].tolist(),
            'labels': examples['label']
        }

    def create_datasets(self, df: pd.DataFrame, train_ratio: float = 0.8,
                       val_ratio: float = 0.1) -> DatasetDict:
        """Split and create train/val/test datasets"""
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        n_samples = len(df)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]

        print(f"✓ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        train_tokenized = train_dataset.map(
            self.tokenize_function, batched=True, batch_size=16,
            remove_columns=train_dataset.column_names
        )
        val_tokenized = val_dataset.map(
            self.tokenize_function, batched=True, batch_size=16,
            remove_columns=val_dataset.column_names
        )
        test_tokenized = test_dataset.map(
            self.tokenize_function, batched=True, batch_size=16,
            remove_columns=test_dataset.column_names
        )

        train_tokenized.set_format(type='torch')
        val_tokenized.set_format(type='torch')
        test_tokenized.set_format(type='torch')

        return DatasetDict({
            'train': train_tokenized,
            'validation': val_tokenized,
            'test': test_tokenized
        })


class SentimentTrainer:
    """Handles training of sentiment analysis models"""

    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model(self):
        """Load pre-trained model and tokenizer"""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels,
            problem_type="single_label_classification"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"✓ Loaded model: {self.model_name}")

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='macro', zero_division=0),
            'recall': recall_score(labels, predictions, average='macro', zero_division=0),
            'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
            'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0)
        }

    def train(self, train_dataset, val_dataset, output_dir: str = "./model",
              num_epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5,
              early_stopping_patience: int = 2):
        """Train the model"""
        if self.model is None:
            self.load_model()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=2,
            seed=42,
            report_to="none"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience)]
        )

        print("\n🚀 Starting training...")
        train_result = self.trainer.train()

        # Save model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(str(output_path))

        # Save metrics
        with open(output_path / "training_metrics.json", "w") as f:
            json.dump(train_result.metrics, f, indent=2)

        print(f"✓ Model saved to {output_path}")
        return train_result

    def evaluate(self, test_dataset) -> Dict:
        """Evaluate the model"""
        print("\n📊 Evaluating on test set...")
        predictions = self.trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids

        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels, average='macro', zero_division=0),
            'recall': recall_score(true_labels, pred_labels, average='macro', zero_division=0),
            'f1_macro': f1_score(true_labels, pred_labels, average='macro', zero_division=0),
            'f1_weighted': f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
        }

        class_names = ['Negative', 'Neutral', 'Positive']
        report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)
        cm = confusion_matrix(true_labels, pred_labels)

        metrics['classification_report'] = report
        metrics['confusion_matrix'] = cm.tolist()

        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1-Macro:    {metrics['f1_macro']:.4f}")
        print(f"F1-Weighted: {metrics['f1_weighted']:.4f}")
        print("="*50)

        return metrics


def main():
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--model', type=str, default='bert-base-uncased', help='Model name')
    parser.add_argument('--output', type=str, default='./model', help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=128, help='Max sequence length')
    
    args = parser.parse_args()

    print("\n" + "="*50)
    print("SENTIMENT ANALYSIS MODEL TRAINING")
    print("="*50)

    # Preprocessing
    print("\n📋 Loading and preprocessing data...")
    preprocessor = SentimentPreprocessor(model_name=args.model, max_length=args.max_length)
    df = preprocessor.load_and_preprocess_data(args.data)
    datasets = preprocessor.create_datasets(df)

    # Training
    print("\n🎯 Initializing model...")
    trainer = SentimentTrainer(model_name=args.model)
    trainer.train(
        train_dataset=datasets['train'],
        val_dataset=datasets['validation'],
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    # Evaluation
    metrics = trainer.evaluate(datasets['test'])

    # Save final metrics
    output_path = Path(args.output)
    with open(output_path / "test_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k not in ['classification_report', 'confusion_matrix']}, f, indent=2)

    print("\n✅ Training completed successfully!")
    print(f"📁 Model and metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
