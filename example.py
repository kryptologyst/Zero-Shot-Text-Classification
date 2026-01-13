#!/usr/bin/env python3
"""Example script demonstrating zero-shot text classification."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from classifier import ZeroShotClassifier
from dataset_generator import SyntheticDatasetGenerator


def main():
    """Main example function."""
    print("🤖 Zero-Shot Text Classification Example")
    print("=" * 50)
    
    # Initialize classifier
    print("Initializing classifier...")
    classifier = ZeroShotClassifier()
    
    # Example 1: Single text classification
    print("\n📝 Example 1: Single Text Classification")
    text = "The stock market continues to show strong growth despite inflation concerns."
    labels = ["economy", "finance", "politics", "sports", "technology"]
    
    result = classifier.classify(text, labels)
    
    print(f"Text: {text}")
    print(f"Labels: {labels}")
    print("Results:")
    for label, score in zip(result['labels'], result['scores']):
        print(f"  {label}: {score:.3f}")
    
    # Example 2: Batch processing
    print("\n📊 Example 2: Batch Processing")
    texts = [
        "Artificial intelligence is revolutionizing healthcare diagnostics.",
        "The Federal Reserve announced new monetary policy changes.",
        "The championship game ended in overtime with a dramatic finish.",
        "Researchers discover new species in deep ocean exploration."
    ]
    
    results = classifier.classify(texts, labels)
    
    print("Batch Results:")
    for text, result in zip(texts, results):
        print(f"  {text[:50]}... -> {result['labels'][0]} ({result['scores'][0]:.3f})")
    
    # Example 3: Confidence filtering
    print("\n🎯 Example 3: Confidence Filtering")
    result = classifier.classify_with_confidence(
        text, 
        labels, 
        confidence_threshold=0.3
    )
    
    print(f"Text: {text}")
    print(f"Confidence threshold: 0.3")
    print("Filtered Results:")
    for label, score in zip(result['labels'], result['scores']):
        print(f"  {label}: {score:.3f}")
    
    # Example 4: Synthetic dataset generation
    print("\n🔬 Example 4: Synthetic Dataset Generation")
    generator = SyntheticDatasetGenerator()
    dataset = generator.generate_dataset(num_samples=20, balanced=True)
    
    print(f"Generated dataset with {len(dataset)} samples")
    print("Sample data:")
    for i in range(3):
        print(f"  {dataset['text'][i][:60]}... -> {dataset['label'][i]}")
    
    # Example 5: Model evaluation
    print("\n📈 Example 5: Model Evaluation")
    try:
        eval_results = classifier.evaluate_on_dataset(
            dataset,
            candidate_labels=list(set(dataset['label']))
        )
        
        print(f"Accuracy: {eval_results['accuracy']:.3f}")
        print("Top 3 predictions:")
        for i in range(3):
            pred = eval_results['predictions'][i]
            true = eval_results['true_labels'][i]
            print(f"  Predicted: {pred}, True: {true}")
    
    except Exception as e:
        print(f"Evaluation failed (this is expected in example mode): {e}")
    
    print("\n✅ Example completed successfully!")
    print("\nTo run the full application:")
    print("  - Web interface: streamlit run web_app/app.py")
    print("  - CLI: python cli.py interactive")


if __name__ == "__main__":
    main()
