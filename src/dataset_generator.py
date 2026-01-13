"""Synthetic dataset generator for zero-shot text classification."""

import json
import random
from typing import List, Dict, Tuple
from pathlib import Path
from datasets import Dataset


class SyntheticDatasetGenerator:
    """Generate synthetic datasets for zero-shot text classification testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize the dataset generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        
        # Define sample texts for different categories
        self.sample_texts = {
            "economy": [
                "The stock market continues to show strong growth despite inflation concerns.",
                "Unemployment rates have dropped significantly this quarter.",
                "The Federal Reserve announced new monetary policy changes.",
                "GDP growth exceeded expectations in the latest economic report.",
                "Consumer spending patterns shifted dramatically during the pandemic.",
                "Trade negotiations between countries are progressing well.",
                "The housing market shows signs of recovery.",
                "Corporate earnings reports exceeded analyst predictions."
            ],
            "technology": [
                "Artificial intelligence is revolutionizing healthcare diagnostics.",
                "New quantum computing breakthrough promises faster processing.",
                "Cybersecurity threats are becoming more sophisticated.",
                "Cloud computing adoption continues to accelerate globally.",
                "Machine learning algorithms improve recommendation systems.",
                "Blockchain technology finds applications beyond cryptocurrency.",
                "5G networks enable new mobile applications.",
                "Virtual reality transforms entertainment and education."
            ],
            "politics": [
                "The new legislation aims to address climate change concerns.",
                "International diplomatic relations show signs of improvement.",
                "Voter turnout reached record levels in the recent election.",
                "Policy changes affect healthcare accessibility nationwide.",
                "Government spending priorities shift toward infrastructure.",
                "Political parties debate tax reform proposals.",
                "Foreign policy decisions impact global trade relationships.",
                "Constitutional amendments require bipartisan support."
            ],
            "sports": [
                "The championship game ended in overtime with a dramatic finish.",
                "Athletes break world records at the international competition.",
                "Team chemistry proved crucial for the successful season.",
                "Training methods evolve with sports science advances.",
                "Fan attendance reaches pre-pandemic levels.",
                "Player transfers reshape team rosters significantly.",
                "Olympic preparations continue despite logistical challenges.",
                "Youth sports programs promote physical fitness and teamwork."
            ],
            "science": [
                "Researchers discover new species in deep ocean exploration.",
                "Climate change studies reveal concerning temperature trends.",
                "Medical breakthrough offers hope for rare disease treatment.",
                "Space exploration missions gather valuable planetary data.",
                "Renewable energy technologies become more efficient.",
                "Genetic research advances understanding of human evolution.",
                "Environmental conservation efforts show positive results.",
                "Scientific collaboration accelerates discovery processes."
            ],
            "entertainment": [
                "The new movie breaks box office records worldwide.",
                "Streaming platforms compete for exclusive content rights.",
                "Music festivals return with enhanced safety protocols.",
                "Celebrity endorsements influence consumer purchasing decisions.",
                "Gaming industry revenue surpasses traditional entertainment.",
                "Social media platforms launch new creative features.",
                "Award shows celebrate diverse artistic achievements.",
                "Live performances adapt to hybrid virtual formats."
            ]
        }
    
    def generate_dataset(
        self,
        num_samples: int = 100,
        categories: Optional[List[str]] = None,
        balanced: bool = True
    ) -> Dataset:
        """Generate a synthetic dataset for classification.
        
        Args:
            num_samples: Total number of samples to generate
            categories: List of categories to include (default: all)
            balanced: Whether to balance samples across categories
            
        Returns:
            Generated dataset
        """
        if categories is None:
            categories = list(self.sample_texts.keys())
        
        texts = []
        labels = []
        
        if balanced:
            samples_per_category = num_samples // len(categories)
            remainder = num_samples % len(categories)
            
            for i, category in enumerate(categories):
                category_samples = samples_per_category + (1 if i < remainder else 0)
                
                for _ in range(category_samples):
                    text = random.choice(self.sample_texts[category])
                    texts.append(text)
                    labels.append(category)
        else:
            for _ in range(num_samples):
                category = random.choice(categories)
                text = random.choice(self.sample_texts[category])
                texts.append(text)
                labels.append(category)
        
        # Shuffle the dataset
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        
        return Dataset.from_dict({
            "text": list(texts),
            "label": list(labels)
        })
    
    def generate_test_cases(self) -> List[Dict[str, str]]:
        """Generate test cases for zero-shot classification.
        
        Returns:
            List of test cases with text and expected category
        """
        test_cases = []
        
        for category, texts in self.sample_texts.items():
            for text in texts[:2]:  # Take first 2 texts from each category
                test_cases.append({
                    "text": text,
                    "expected_category": category
                })
        
        random.shuffle(test_cases)
        return test_cases
    
    def save_dataset(self, dataset: Dataset, filepath: str) -> None:
        """Save dataset to file.
        
        Args:
            dataset: Dataset to save
            filepath: Path to save the dataset
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON format
        data = {
            "texts": dataset["text"],
            "labels": dataset["label"]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_dataset(self, filepath: str) -> Dataset:
        """Load dataset from file.
        
        Args:
            filepath: Path to load the dataset from
            
        Returns:
            Loaded dataset
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return Dataset.from_dict({
            "text": data["texts"],
            "label": data["labels"]
        })
    
    def create_benchmark_dataset(self) -> Tuple[Dataset, List[str]]:
        """Create a benchmark dataset for evaluation.
        
        Returns:
            Tuple of (dataset, candidate_labels)
        """
        categories = list(self.sample_texts.keys())
        dataset = self.generate_dataset(num_samples=200, categories=categories)
        
        return dataset, categories
