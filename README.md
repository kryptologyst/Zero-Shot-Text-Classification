# Zero-Shot Text Classification

A production-ready zero-shot text classification system using state-of-the-art transformer models from Hugging Face. This project allows you to classify text into categories without requiring any training data for those categories.

## Features

- **Zero-Shot Classification**: Classify text into user-defined categories without training data
- **Multiple Interfaces**: CLI, Streamlit web app, and Python API
- **State-of-the-Art Models**: Support for BART, RoBERTa, and other transformer models
- **Batch Processing**: Process multiple texts efficiently
- **Confidence Scoring**: Filter results by confidence thresholds
- **Evaluation Tools**: Comprehensive evaluation metrics and visualization
- **Synthetic Data**: Built-in dataset generator for testing and benchmarking
- **Modern Architecture**: Type hints, logging, configuration management, and clean code structure

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Zero-Shot-Text-Classification.git
cd Zero-Shot-Text-Classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install development dependencies:
```bash
pip install pytest black flake8 mypy
```

## Quick Start

### Web Interface (Recommended)

Launch the Streamlit web app:
```bash
streamlit run web_app/app.py
```

Open your browser to `http://localhost:8501` and start classifying text!

### Command Line Interface

Classify a single text:
```bash
python cli.py classify "The stock market is growing" --labels economy finance politics
```

Interactive mode:
```bash
python cli.py interactive
```

### Python API

```python
from src.classifier import ZeroShotClassifier

# Initialize classifier
classifier = ZeroShotClassifier()

# Classify text
result = classifier.classify(
    "The stock market continues to show strong growth despite inflation concerns.",
    ["economy", "finance", "politics", "sports", "technology"]
)

print(f"Predicted: {result['labels'][0]} (Score: {result['scores'][0]:.3f})")
```

## Usage Examples

### Basic Classification

```python
from src.classifier import ZeroShotClassifier

classifier = ZeroShotClassifier()

# Single text classification
text = "Artificial intelligence is revolutionizing healthcare diagnostics."
labels = ["technology", "healthcare", "science", "business"]

result = classifier.classify(text, labels)
print(f"Top prediction: {result['labels'][0]} ({result['scores'][0]:.3f})")
```

### Batch Processing

```python
texts = [
    "The Federal Reserve announced new monetary policy changes.",
    "Machine learning algorithms improve recommendation systems.",
    "The championship game ended in overtime with a dramatic finish."
]

results = classifier.classify(texts, ["economy", "technology", "sports"])
for text, result in zip(texts, results):
    print(f"{text[:50]}... -> {result['labels'][0]}")
```

### Confidence Filtering

```python
# Only return predictions above 0.7 confidence
result = classifier.classify_with_confidence(
    text, 
    labels, 
    confidence_threshold=0.7
)
```

### Multi-label Classification

```python
# Allow multiple labels per text
result = classifier.classify(
    text, 
    labels, 
    multi_label=True
)
```

## 🔧 Configuration

### YAML Configuration

Create or modify `config/config.yaml`:

```yaml
model_name: "facebook/bart-large-mnli"
device: "auto"
use_fp16: false
confidence_threshold: 0.5
hypothesis_template: "This text is about {}."
batch_size: 32
max_length: 512
logging_level: "INFO"
```

### Programmatic Configuration

```python
from src.classifier import ZeroShotClassifierConfig

config = ZeroShotClassifierConfig("config/config.yaml")
classifier = ZeroShotClassifier(
    model_name=config.get("model_name"),
    device=config.get("device"),
    use_fp16=config.get("use_fp16")
)
```

## Evaluation and Benchmarking

### Generate Synthetic Dataset

```python
from src.dataset_generator import SyntheticDatasetGenerator

generator = SyntheticDatasetGenerator()
dataset = generator.generate_dataset(num_samples=200, balanced=True)
```

### Evaluate Model Performance

```python
# Evaluate on synthetic dataset
results = classifier.evaluate_on_dataset(dataset, candidate_labels=categories)

print(f"Accuracy: {results['accuracy']:.3f}")
print("Classification Report:")
print(results['classification_report'])
```

### CLI Evaluation

```bash
# Generate dataset
python cli.py generate-dataset --output data/synthetic.json --samples 200

# Evaluate model
python cli.py evaluate --dataset data/synthetic.json --labels economy finance politics
```

## Web Interface Features

The Streamlit web app provides:

- **Single Text Classification**: Interactive text input with real-time results
- **Batch Processing**: Upload CSV files for bulk classification
- **Model Evaluation**: Generate synthetic datasets and evaluate performance
- **Interactive Demo**: Try different examples with various categories
- **Visualization**: Charts and graphs for results analysis

## Project Structure

```
zero-shot-text-classification/
├── src/                          # Source code
│   ├── __init__.py
│   ├── classifier.py            # Main classification module
│   └── dataset_generator.py     # Synthetic dataset generation
├── web_app/                      # Streamlit web application
│   └── app.py
├── config/                       # Configuration files
│   └── config.yaml
├── data/                         # Data directory
├── models/                       # Model storage
├── tests/                        # Test files
├── cli.py                        # Command-line interface
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Testing

Run tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Performance Tips

1. **GPU Acceleration**: Use CUDA for faster inference
2. **Batch Processing**: Process multiple texts together
3. **Model Selection**: Choose appropriate models for your use case
4. **Confidence Thresholding**: Filter low-confidence predictions
5. **Caching**: Enable model caching for repeated use

## Supported Models

- `facebook/bart-large-mnli` (default, recommended)
- `roberta-large-mnli`
- `microsoft/DialoGPT-medium`
- Any compatible Hugging Face model

## API Reference

### ZeroShotClassifier

#### `__init__(model_name, device, use_fp16)`
Initialize the classifier with specified model and settings.

#### `classify(text, candidate_labels, multi_label, hypothesis_template)`
Classify text using zero-shot classification.

#### `classify_with_confidence(text, candidate_labels, confidence_threshold)`
Classify text with confidence filtering.

#### `evaluate_on_dataset(dataset, text_column, label_column, candidate_labels)`
Evaluate classifier performance on a dataset.

### SyntheticDatasetGenerator

#### `generate_dataset(num_samples, categories, balanced)`
Generate synthetic dataset for testing.

#### `create_benchmark_dataset()`
Create benchmark dataset with all categories.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the transformers library
- Facebook AI for the BART model
- The open-source community for various dependencies

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation
- Review existing issues and discussions
# Zero-Shot-Text-Classification
