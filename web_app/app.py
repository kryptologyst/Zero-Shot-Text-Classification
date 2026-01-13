"""Streamlit web interface for zero-shot text classification."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import json
from pathlib import Path

from classifier import ZeroShotClassifier, ZeroShotClassifierConfig
from dataset_generator import SyntheticDatasetGenerator


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'config' not in st.session_state:
        st.session_state.config = ZeroShotClassifierConfig()
    if 'dataset_generator' not in st.session_state:
        st.session_state.dataset_generator = SyntheticDatasetGenerator()


def load_classifier(model_name: str, device: str, use_fp16: bool):
    """Load the zero-shot classifier."""
    try:
        with st.spinner(f"Loading {model_name}..."):
            classifier = ZeroShotClassifier(
                model_name=model_name,
                device=device,
                use_fp16=use_fp16
            )
            st.session_state.classifier = classifier
            st.success("Classifier loaded successfully!")
            return True
    except Exception as e:
        st.error(f"Failed to load classifier: {str(e)}")
        return False


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Zero-Shot Text Classification",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Zero-Shot Text Classification")
    st.markdown("Classify text into categories without training data using state-of-the-art transformer models.")
    
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_name = st.selectbox(
            "Model",
            [
                "facebook/bart-large-mnli",
                "facebook/bart-large-mnli",
                "microsoft/DialoGPT-medium",
                "roberta-large-mnli"
            ],
            help="Select the pre-trained model for zero-shot classification"
        )
        
        # Device selection
        device = st.selectbox(
            "Device",
            ["auto", "cpu", "cuda"],
            help="Select the device for inference"
        )
        
        # Precision
        use_fp16 = st.checkbox(
            "Use FP16",
            value=False,
            help="Use half precision for faster inference (requires GPU)"
        )
        
        # Load classifier button
        if st.button("Load Classifier", type="primary"):
            load_classifier(model_name, device, use_fp16)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Text", "📊 Batch Processing", "📈 Evaluation", "🎯 Demo"])
    
    with tab1:
        st.header("Single Text Classification")
        
        if st.session_state.classifier is None:
            st.warning("Please load a classifier first using the sidebar.")
        else:
            # Text input
            text = st.text_area(
                "Enter text to classify:",
                value="The stock market continues to show strong growth despite inflation concerns.",
                height=100
            )
            
            # Candidate labels
            st.subheader("Candidate Labels")
            col1, col2 = st.columns(2)
            
            with col1:
                labels_text = st.text_area(
                    "Enter labels (one per line):",
                    value="economy\nfinance\npolitics\nsports\ntechnology",
                    height=150
                )
            
            with col2:
                st.markdown("**Quick presets:**")
                if st.button("News Categories"):
                    labels_text = "economy\npolitics\nsports\ntechnology\nentertainment"
                if st.button("Sentiment"):
                    labels_text = "positive\nnegative\nneutral"
                if st.button("Topics"):
                    labels_text = "science\nhealth\neducation\nbusiness\nlifestyle"
            
            candidate_labels = [label.strip() for label in labels_text.split('\n') if label.strip()]
            
            # Classification options
            col1, col2 = st.columns(2)
            with col1:
                multi_label = st.checkbox("Multi-label classification", value=False)
            with col2:
                confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.1)
            
            # Classify button
            if st.button("Classify", type="primary"):
                if text and candidate_labels:
                    try:
                        result = st.session_state.classifier.classify_with_confidence(
                            text,
                            candidate_labels,
                            confidence_threshold=confidence_threshold,
                            multi_label=multi_label
                        )
                        
                        # Display results
                        st.subheader("Classification Results")
                        
                        # Create results dataframe
                        results_df = pd.DataFrame({
                            'Label': result['labels'],
                            'Score': result['scores']
                        })
                        
                        # Display as table
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Display as bar chart
                        if result['labels']:
                            fig = px.bar(
                                results_df,
                                x='Score',
                                y='Label',
                                orientation='h',
                                title="Classification Scores",
                                color='Score',
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show top prediction
                        if result['labels']:
                            top_label = result['labels'][0]
                            top_score = result['scores'][0]
                            st.success(f"**Top Prediction:** {top_label} (Score: {top_score:.3f})")
                        
                    except Exception as e:
                        st.error(f"Classification failed: {str(e)}")
                else:
                    st.error("Please enter text and at least one label.")
    
    with tab2:
        st.header("Batch Processing")
        
        if st.session_state.classifier is None:
            st.warning("Please load a classifier first using the sidebar.")
        else:
            # File upload
            uploaded_file = st.file_uploader(
                "Upload CSV file with 'text' column",
                type=['csv'],
                help="CSV file should have a 'text' column containing texts to classify"
            )
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.subheader("Preview of uploaded data")
                st.dataframe(df.head())
                
                if 'text' in df.columns:
                    # Candidate labels
                    labels_text = st.text_area(
                        "Enter labels (one per line):",
                        value="economy\nfinance\npolitics\nsports\ntechnology",
                        height=100
                    )
                    candidate_labels = [label.strip() for label in labels_text.split('\n') if label.strip()]
                    
                    if st.button("Process Batch", type="primary"):
                        if candidate_labels:
                            try:
                                texts = df['text'].tolist()
                                results = st.session_state.classifier.classify(texts, candidate_labels)
                                
                                # Process results
                                predictions = []
                                scores = []
                                
                                for result in results:
                                    predictions.append(result['labels'][0])
                                    scores.append(result['scores'][0])
                                
                                # Add results to dataframe
                                df['predicted_label'] = predictions
                                df['confidence_score'] = scores
                                
                                st.subheader("Results")
                                st.dataframe(df, use_container_width=True)
                                
                                # Download results
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="Download Results",
                                    data=csv,
                                    file_name="classification_results.csv",
                                    mime="text/csv"
                                )
                                
                            except Exception as e:
                                st.error(f"Batch processing failed: {str(e)}")
                        else:
                            st.error("Please enter at least one label.")
                else:
                    st.error("CSV file must contain a 'text' column.")
    
    with tab3:
        st.header("Model Evaluation")
        
        if st.session_state.classifier is None:
            st.warning("Please load a classifier first using the sidebar.")
        else:
            st.subheader("Generate Synthetic Dataset")
            
            col1, col2 = st.columns(2)
            with col1:
                num_samples = st.slider("Number of samples", 50, 500, 200)
            with col2:
                balanced = st.checkbox("Balanced dataset", value=True)
            
            if st.button("Generate Dataset", type="primary"):
                try:
                    dataset, categories = st.session_state.dataset_generator.create_benchmark_dataset()
                    
                    st.success(f"Generated dataset with {len(dataset)} samples")
                    st.write(f"Categories: {', '.join(categories)}")
                    
                    # Show sample data
                    sample_df = pd.DataFrame({
                        'text': dataset['text'][:5],
                        'label': dataset['label'][:5]
                    })
                    st.subheader("Sample Data")
                    st.dataframe(sample_df)
                    
                    # Evaluate model
                    if st.button("Evaluate Model", type="primary"):
                        with st.spinner("Evaluating model..."):
                            results = st.session_state.classifier.evaluate_on_dataset(
                                dataset,
                                candidate_labels=categories
                            )
                        
                        st.subheader("Evaluation Results")
                        st.metric("Accuracy", f"{results['accuracy']:.3f}")
                        
                        # Classification report
                        report_df = pd.DataFrame(results['classification_report']).transpose()
                        st.subheader("Detailed Classification Report")
                        st.dataframe(report_df)
                        
                        # Confusion matrix
                        st.subheader("Confusion Matrix")
                        fig = go.Figure(data=go.Heatmap(
                            z=results['classification_report'],
                            x=categories,
                            y=categories,
                            colorscale='Blues'
                        ))
                        fig.update_layout(
                            title="Confusion Matrix",
                            xaxis_title="Predicted",
                            yaxis_title="True"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
    
    with tab4:
        st.header("Interactive Demo")
        
        if st.session_state.classifier is None:
            st.warning("Please load a classifier first using the sidebar.")
        else:
            st.subheader("Try Different Examples")
            
            # Example texts
            examples = {
                "Economy": "The Federal Reserve announced new monetary policy changes to address inflation concerns.",
                "Technology": "Artificial intelligence is revolutionizing healthcare diagnostics with unprecedented accuracy.",
                "Politics": "The new legislation aims to address climate change through comprehensive environmental policies.",
                "Sports": "The championship game ended in overtime with a dramatic finish that thrilled fans worldwide.",
                "Science": "Researchers discover new species in deep ocean exploration using advanced robotic technology."
            }
            
            selected_example = st.selectbox("Choose an example:", list(examples.keys()))
            example_text = examples[selected_example]
            
            st.text_area("Example text:", value=example_text, height=100, disabled=True)
            
            # Quick classify
            if st.button("Quick Classify", type="primary"):
                candidate_labels = list(examples.keys())
                
                try:
                    result = st.session_state.classifier.classify(example_text, candidate_labels)
                    
                    # Display results
                    results_df = pd.DataFrame({
                        'Label': result['labels'],
                        'Score': result['scores']
                    })
                    
                    st.subheader("Classification Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Visualize results
                    fig = px.bar(
                        results_df,
                        x='Score',
                        y='Label',
                        orientation='h',
                        title="Classification Scores",
                        color='Score',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Classification failed: {str(e)}")


if __name__ == "__main__":
    main()
