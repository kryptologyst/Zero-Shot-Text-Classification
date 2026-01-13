"""Command-line interface for zero-shot text classification."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

from src.classifier import ZeroShotClassifier, ZeroShotClassifierConfig
from src.dataset_generator import SyntheticDatasetGenerator


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Zero-Shot Text Classification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a single text
  python cli.py classify "The stock market is growing" --labels economy finance politics

  # Classify from file
  python cli.py classify --input texts.txt --labels economy finance politics

  # Generate synthetic dataset
  python cli.py generate-dataset --output data/synthetic.json --samples 200

  # Evaluate model
  python cli.py evaluate --dataset data/synthetic.json --labels economy finance politics

  # Interactive mode
  python cli.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Classify command
    classify_parser = subparsers.add_parser('classify', help='Classify text')
    classify_parser.add_argument('text', nargs='?', help='Text to classify')
    classify_parser.add_argument('--input', '-i', help='Input file with texts (one per line)')
    classify_parser.add_argument('--labels', '-l', nargs='+', required=True, help='Candidate labels')
    classify_parser.add_argument('--output', '-o', help='Output file for results')
    classify_parser.add_argument('--model', '-m', default='facebook/bart-large-mnli', help='Model name')
    classify_parser.add_argument('--device', '-d', default='auto', help='Device (cpu/cuda/auto)')
    classify_parser.add_argument('--multi-label', action='store_true', help='Enable multi-label classification')
    classify_parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold')
    
    # Generate dataset command
    generate_parser = subparsers.add_parser('generate-dataset', help='Generate synthetic dataset')
    generate_parser.add_argument('--output', '-o', required=True, help='Output file path')
    generate_parser.add_argument('--samples', '-s', type=int, default=200, help='Number of samples')
    generate_parser.add_argument('--categories', '-cat', nargs='+', help='Categories to include')
    generate_parser.add_argument('--balanced', action='store_true', help='Balance samples across categories')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on dataset')
    eval_parser.add_argument('--dataset', '-d', required=True, help='Dataset file path')
    eval_parser.add_argument('--labels', '-l', nargs='+', help='Candidate labels')
    eval_parser.add_argument('--model', '-m', default='facebook/bart-large-mnli', help='Model name')
    eval_parser.add_argument('--output', '-o', help='Output file for evaluation results')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive classification mode')
    interactive_parser.add_argument('--model', '-m', default='facebook/bart-large-mnli', help='Model name')
    interactive_parser.add_argument('--device', '-d', default='auto', help='Device (cpu/cuda/auto)')
    
    return parser


def classify_text(args) -> None:
    """Handle text classification."""
    # Initialize classifier
    classifier = ZeroShotClassifier(
        model_name=args.model,
        device=args.device
    )
    
    # Get input texts
    if args.text:
        texts = [args.text]
    elif args.input:
        with open(args.input, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Please provide either text or input file")
        sys.exit(1)
    
    # Classify texts
    results = classifier.classify_with_confidence(
        texts,
        args.labels,
        confidence_threshold=args.confidence,
        multi_label=args.multi_label
    )
    
    # Handle single vs batch results
    if len(texts) == 1:
        results = [results]
    
    # Display results
    for i, (text, result) in enumerate(zip(texts, results)):
        print(f"\nText {i+1}: {text}")
        print("Classification Results:")
        
        if result['labels']:
            for label, score in zip(result['labels'], result['scores']):
                print(f"  {label}: {score:.3f}")
        else:
            print("  No labels above confidence threshold")
    
    # Save results if output file specified
    if args.output:
        output_data = []
        for text, result in zip(texts, results):
            output_data.append({
                'text': text,
                'labels': result['labels'],
                'scores': result['scores']
            })
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {args.output}")


def generate_dataset(args) -> None:
    """Handle dataset generation."""
    generator = SyntheticDatasetGenerator()
    
    dataset = generator.generate_dataset(
        num_samples=args.samples,
        categories=args.categories,
        balanced=args.balanced
    )
    
    generator.save_dataset(dataset, args.output)
    print(f"Generated dataset with {len(dataset)} samples saved to {args.output}")


def evaluate_model(args) -> None:
    """Handle model evaluation."""
    # Initialize classifier
    classifier = ZeroShotClassifier(
        model_name=args.model
    )
    
    # Load dataset
    generator = SyntheticDatasetGenerator()
    dataset = generator.load_dataset(args.dataset)
    
    # Get labels
    if args.labels:
        candidate_labels = args.labels
    else:
        candidate_labels = list(set(dataset['label']))
    
    print(f"Evaluating model on {len(dataset)} samples...")
    print(f"Candidate labels: {candidate_labels}")
    
    # Evaluate
    results = classifier.evaluate_on_dataset(
        dataset,
        candidate_labels=candidate_labels
    )
    
    # Display results
    print(f"\nAccuracy: {results['accuracy']:.3f}")
    print("\nClassification Report:")
    print(json.dumps(results['classification_report'], indent=2))
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nEvaluation results saved to {args.output}")


def interactive_mode(args) -> None:
    """Handle interactive classification mode."""
    # Initialize classifier
    print(f"Loading model: {args.model}")
    classifier = ZeroShotClassifier(
        model_name=args.model,
        device=args.device
    )
    
    print("Interactive mode started. Type 'quit' to exit.")
    print("Enter labels separated by commas, then press Enter.")
    
    # Get labels
    labels_input = input("Labels: ").strip()
    if not labels_input:
        print("No labels provided. Exiting.")
        return
    
    labels = [label.strip() for label in labels_input.split(',') if label.strip()]
    print(f"Using labels: {labels}")
    
    # Interactive loop
    while True:
        text = input("\nEnter text to classify: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            continue
        
        try:
            result = classifier.classify_with_confidence(text, labels)
            
            print("\nClassification Results:")
            if result['labels']:
                for label, score in zip(result['labels'], result['scores']):
                    print(f"  {label}: {score:.3f}")
            else:
                print("  No labels above confidence threshold")
        
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main CLI function."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'classify':
            classify_text(args)
        elif args.command == 'generate-dataset':
            generate_dataset(args)
        elif args.command == 'evaluate':
            evaluate_model(args)
        elif args.command == 'interactive':
            interactive_mode(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
