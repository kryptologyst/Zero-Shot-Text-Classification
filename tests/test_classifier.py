"""Test suite for zero-shot text classification."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.classifier import ZeroShotClassifier, ZeroShotClassifierConfig
from src.dataset_generator import SyntheticDatasetGenerator


class TestZeroShotClassifier:
    """Test cases for ZeroShotClassifier."""
    
    @pytest.fixture
    def mock_classifier(self):
        """Create a mock classifier for testing."""
        with patch('src.classifier.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            classifier = ZeroShotClassifier()
            return classifier
    
    def test_classifier_initialization(self, mock_classifier):
        """Test classifier initialization."""
        assert mock_classifier.model_name == "facebook/bart-large-mnli"
        assert mock_classifier.device in ["cpu", "cuda"]
    
    def test_classify_single_text(self, mock_classifier):
        """Test single text classification."""
        # Mock the classifier response
        mock_result = {
            'sequence': 'Test text',
            'labels': ['technology', 'science'],
            'scores': [0.8, 0.2]
        }
        mock_classifier.classifier.return_value = mock_result
        
        result = mock_classifier.classify(
            "Test text",
            ["technology", "science"]
        )
        
        assert result['labels'] == ['technology', 'science']
        assert result['scores'] == [0.8, 0.2]
    
    def test_classify_batch_texts(self, mock_classifier):
        """Test batch text classification."""
        mock_result = {
            'sequence': 'Test text',
            'labels': ['technology'],
            'scores': [0.8]
        }
        mock_classifier.classifier.return_value = mock_result
        
        texts = ["Test text 1", "Test text 2"]
        results = mock_classifier.classify(texts, ["technology"])
        
        assert len(results) == 2
        assert all(result['labels'] == ['technology'] for result in results)
    
    def test_classify_with_confidence(self, mock_classifier):
        """Test classification with confidence filtering."""
        mock_result = {
            'sequence': 'Test text',
            'labels': ['technology', 'science'],
            'scores': [0.8, 0.3]
        }
        mock_classifier.classifier.return_value = mock_result
        
        result = mock_classifier.classify_with_confidence(
            "Test text",
            ["technology", "science"],
            confidence_threshold=0.5
        )
        
        assert len(result['labels']) == 1
        assert result['labels'][0] == 'technology'
        assert result['scores'][0] == 0.8
    
    def test_filter_by_confidence(self, mock_classifier):
        """Test confidence filtering."""
        result = {
            'sequence': 'Test text',
            'labels': ['technology', 'science', 'business'],
            'scores': [0.8, 0.3, 0.2]
        }
        
        filtered = mock_classifier._filter_by_confidence(result, 0.5)
        
        assert filtered['labels'] == ['technology']
        assert filtered['scores'] == [0.8]


class TestSyntheticDatasetGenerator:
    """Test cases for SyntheticDatasetGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create a dataset generator for testing."""
        return SyntheticDatasetGenerator(seed=42)
    
    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert len(generator.sample_texts) > 0
        assert 'economy' in generator.sample_texts
        assert 'technology' in generator.sample_texts
    
    def test_generate_dataset(self, generator):
        """Test dataset generation."""
        dataset = generator.generate_dataset(num_samples=10, balanced=True)
        
        assert len(dataset) == 10
        assert 'text' in dataset.column_names
        assert 'label' in dataset.column_names
    
    def test_generate_test_cases(self, generator):
        """Test test case generation."""
        test_cases = generator.generate_test_cases()
        
        assert len(test_cases) > 0
        assert all('text' in case for case in test_cases)
        assert all('expected_category' in case for case in test_cases)
    
    def test_save_and_load_dataset(self, generator):
        """Test dataset saving and loading."""
        dataset = generator.generate_dataset(num_samples=5)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            generator.save_dataset(dataset, temp_path)
            loaded_dataset = generator.load_dataset(temp_path)
            
            assert len(loaded_dataset) == len(dataset)
            assert loaded_dataset['text'] == dataset['text']
            assert loaded_dataset['label'] == dataset['label']
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_create_benchmark_dataset(self, generator):
        """Test benchmark dataset creation."""
        dataset, categories = generator.create_benchmark_dataset()
        
        assert len(dataset) > 0
        assert len(categories) > 0
        assert all(cat in generator.sample_texts for cat in categories)


class TestZeroShotClassifierConfig:
    """Test cases for ZeroShotClassifierConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ZeroShotClassifierConfig()
        
        assert config.get('model_name') == "facebook/bart-large-mnli"
        assert config.get('device') == "auto"
        assert config.get('use_fp16') is False
    
    def test_config_get_set(self):
        """Test configuration get/set methods."""
        config = ZeroShotClassifierConfig()
        
        config.set('test_key', 'test_value')
        assert config.get('test_key') == 'test_value'
        assert config.get('nonexistent_key', 'default') == 'default'
    
    def test_save_and_load_config(self):
        """Test configuration saving and loading."""
        config = ZeroShotClassifierConfig()
        config.set('test_key', 'test_value')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save_config(temp_path)
            loaded_config = ZeroShotClassifierConfig(temp_path)
            
            assert loaded_config.get('test_key') == 'test_value'
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_classification(self):
        """Test end-to-end classification workflow."""
        with patch('src.classifier.pipeline') as mock_pipeline:
            mock_classifier_instance = Mock()
            mock_classifier_instance.return_value = {
                'sequence': 'Test text',
                'labels': ['technology'],
                'scores': [0.8]
            }
            mock_pipeline.return_value = mock_classifier_instance
            
            classifier = ZeroShotClassifier()
            generator = SyntheticDatasetGenerator()
            
            # Generate test data
            dataset = generator.generate_dataset(num_samples=5)
            
            # Classify
            result = classifier.classify(
                dataset['text'][0],
                ['technology', 'science']
            )
            
            assert 'labels' in result
            assert 'scores' in result


if __name__ == "__main__":
    pytest.main([__file__])
