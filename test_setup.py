#!/usr/bin/env python3
"""Simple test script to verify the installation."""

def test_basic_imports():
    """Test basic imports."""
    try:
        import transformers
        print("✅ transformers imported successfully")
        
        import datasets
        print("✅ datasets imported successfully")
        
        import torch
        print("✅ torch imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_simple_classification():
    """Test simple classification without our custom modules."""
    try:
        from transformers import pipeline
        
        # Create a simple classifier
        classifier = pipeline("zero-shot-classification")
        
        # Test classification
        result = classifier(
            "The stock market is growing",
            ["economy", "technology", "politics"]
        )
        
        print("✅ Simple classification test passed")
        print(f"   Result: {result['labels'][0]} ({result['scores'][0]:.3f})")
        return True
        
    except Exception as e:
        print(f"❌ Classification test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Zero-Shot Text Classification Setup")
    print("=" * 50)
    
    # Test imports
    if not test_basic_imports():
        print("\n❌ Basic imports failed. Please check your installation.")
        return
    
    # Test simple classification
    if not test_simple_classification():
        print("\n❌ Simple classification failed.")
        return
    
    print("\n✅ All tests passed! The basic setup is working.")
    print("\nNext steps:")
    print("1. Run: python3 example.py")
    print("2. Run: streamlit run web_app/app.py")
    print("3. Run: python3 cli.py interactive")

if __name__ == "__main__":
    main()
