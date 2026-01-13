"""Zero-shot text classification module with modern features."""

import logging
from typing import Dict, List, Optional, Union, Tuple
import json
from pathlib import Path

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZeroShotClassifier:
    """Modern zero-shot text classifier with enhanced features.
    
    This class provides zero-shot text classification capabilities using
    state-of-the-art transformer models with additional features like
    batch processing, confidence scoring, and evaluation metrics.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: Optional[str] = None,
        use_fp16: bool = False
    ):
        """Initialize the zero-shot classifier.
        
        Args:
            model_name: Hugging Face model name for zero-shot classification
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            use_fp16: Whether to use half precision for faster inference
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16
        
        logger.info(f"Initializing classifier with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Initialize the pipeline
        self._initialize_pipeline()
        
    def _initialize_pipeline(self) -> None:
        """Initialize the zero-shot classification pipeline."""
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=self.device,
                torch_dtype=torch.float16 if self.use_fp16 else torch.float32
            )
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def classify(
        self,
        text: Union[str, List[str]],
        candidate_labels: List[str],
        multi_label: bool = False,
        hypothesis_template: str = "This text is about {}."
    ) -> Union[Dict, List[Dict]]:
        """Classify text using zero-shot classification.
        
        Args:
            text: Text(s) to classify
            candidate_labels: List of possible labels
            multi_label: Whether to allow multiple labels per text
            hypothesis_template: Template for hypothesis generation
            
        Returns:
            Classification results with labels and scores
        """
        try:
            if isinstance(text, str):
                result = self.classifier(
                    text,
                    candidate_labels,
                    multi_label=multi_label,
                    hypothesis_template=hypothesis_template
                )
                return result
            else:
                # Batch processing
                results = []
                for t in text:
                    result = self.classifier(
                        t,
                        candidate_labels,
                        multi_label=multi_label,
                        hypothesis_template=hypothesis_template
                    )
                    results.append(result)
                return results
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise
    
    def classify_with_confidence(
        self,
        text: Union[str, List[str]],
        candidate_labels: List[str],
        confidence_threshold: float = 0.5,
        **kwargs
    ) -> Union[Dict, List[Dict]]:
        """Classify text with confidence filtering.
        
        Args:
            text: Text(s) to classify
            candidate_labels: List of possible labels
            confidence_threshold: Minimum confidence score for predictions
            **kwargs: Additional arguments for classification
            
        Returns:
            Classification results filtered by confidence threshold
        """
        results = self.classify(text, candidate_labels, **kwargs)
        
        if isinstance(results, dict):
            return self._filter_by_confidence(results, confidence_threshold)
        else:
            return [self._filter_by_confidence(r, confidence_threshold) for r in results]
    
    def _filter_by_confidence(self, result: Dict, threshold: float) -> Dict:
        """Filter results by confidence threshold."""
        filtered_labels = []
        filtered_scores = []
        
        for label, score in zip(result['labels'], result['scores']):
            if score >= threshold:
                filtered_labels.append(label)
                filtered_scores.append(score)
        
        return {
            'sequence': result['sequence'],
            'labels': filtered_labels,
            'scores': filtered_scores
        }
    
    def evaluate_on_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        label_column: str = "label",
        candidate_labels: Optional[List[str]] = None
    ) -> Dict:
        """Evaluate the classifier on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            text_column: Name of the text column
            label_column: Name of the label column
            candidate_labels: Labels to use for classification
            
        Returns:
            Evaluation metrics and results
        """
        if candidate_labels is None:
            candidate_labels = list(set(dataset[label_column]))
        
        predictions = []
        true_labels = []
        
        logger.info(f"Evaluating on {len(dataset)} samples")
        
        for i, sample in enumerate(dataset):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(dataset)} samples")
            
            result = self.classify(sample[text_column], candidate_labels)
            predicted_label = result['labels'][0]
            
            predictions.append(predicted_label)
            true_labels.append(sample[label_column])
        
        # Calculate metrics
        report = classification_report(true_labels, predictions, output_dict=True)
        
        return {
            'classification_report': report,
            'predictions': predictions,
            'true_labels': true_labels,
            'accuracy': report['accuracy']
        }
    
    def save_results(self, results: Dict, filepath: str) -> None:
        """Save classification results to file.
        
        Args:
            results: Results dictionary to save
            filepath: Path to save the results
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def plot_confusion_matrix(
        self,
        true_labels: List[str],
        predictions: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix for evaluation results.
        
        Args:
            true_labels: True labels
            predictions: Predicted labels
            save_path: Optional path to save the plot
        """
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=sorted(set(true_labels)),
            yticklabels=sorted(set(true_labels))
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()


class ZeroShotClassifierConfig:
    """Configuration class for zero-shot classifier."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.default_config = {
            "model_name": "facebook/bart-large-mnli",
            "device": "auto",
            "use_fp16": False,
            "confidence_threshold": 0.5,
            "hypothesis_template": "This text is about {}.",
            "batch_size": 32,
            "max_length": 512
        }
        
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
        else:
            self.config = self.default_config.copy()
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Merge with defaults for missing keys
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def save_config(self, config_path: str) -> None:
        """Save configuration to file."""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value) -> None:
        """Set configuration value."""
        self.config[key] = value
