"""
Professional utility functions for the Wildlife Classification System
Comprehensive utilities with enhanced functionality and professional visualization
Updated with pipeline validation and debugging capabilities
"""

import os
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union, Any
import joblib
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                           precision_recall_fscore_support, roc_curve, auc)
from config import (Config, DATA_CONFIG, MODEL_CONFIG, FEATURE_CONFIG, UI_CONFIG,
                   DEEP_LEARNING_CONFIG, PROCESSING_CONFIG, VISUALIZATION_CONFIG,
                   get_data_path, get_class_path, ensure_directories)

# Set visualization style
plt.style.use(VISUALIZATION_CONFIG.style)
sns.set_palette(VISUALIZATION_CONFIG.color_palette)


# Pipeline Validation and Debugging Utilities
def validate_feature_pipeline(feature_extractor, model_name, ml_results):
    """
    Validate that the feature pipeline is correctly configured for prediction
    Returns (is_valid, error_message, expected_features)
    """

    # Check if feature extractor exists and is fitted
    if feature_extractor is None:
        return False, "Feature extractor not found", None

    if not hasattr(feature_extractor, 'is_fitted') or not feature_extractor.is_fitted:
        return False, "Feature extractor not fitted. Please train models first.", None

    # Check if model exists
    model_key = model_name.replace("ML_", "")
    if model_key not in ml_results:
        return False, f"Model '{model_key}' not found in results", None

    model_info = ml_results[model_key]
    if model_info['model'] is None:
        return False, f"Model '{model_key}' is None", None

    model = model_info['model']

    # Get expected features from the trained model
    if hasattr(model, 'n_features_in_'):
        model_expected_features = model.n_features_in_
    elif hasattr(model, 'coef_'):
        model_expected_features = model.coef_.shape[1] if len(model.coef_.shape) > 1 else len(model.coef_)
    else:
        model_expected_features = "unknown"

    # Get expected features from pipeline
    pipeline_expected_features = feature_extractor.get_expected_feature_count()

    # Check if they match
    if isinstance(model_expected_features, int) and isinstance(pipeline_expected_features, int):
        if model_expected_features != pipeline_expected_features:
            return False, f"Feature count mismatch: model expects {model_expected_features}, pipeline produces {pipeline_expected_features}", model_expected_features

    return True, "Pipeline validation successful", model_expected_features

def debug_feature_transformation(feature_extractor, raw_features):
    """
    Debug the feature transformation pipeline step by step
    """
    st.subheader("Feature Pipeline Debug")

    if raw_features is None:
        st.error("Raw features are None")
        return None

    st.write(f"**Step 0 - Raw features:** {raw_features.shape}")

    # Ensure 2D
    if len(raw_features.shape) == 1:
        features = raw_features.reshape(1, -1)
        st.write(f"**Step 1 - Reshaped to 2D:** {features.shape}")
    else:
        features = raw_features
        st.write(f"**Step 1 - Already 2D:** {features.shape}")

    # Scaling
    try:
        if feature_extractor.scaler is not None:
            features_scaled = feature_extractor.scaler.transform(features)
            st.write(f"**Step 2 - After scaling:** {features_scaled.shape}")
        else:
            features_scaled = features
            st.write(f"**Step 2 - No scaling applied:** {features_scaled.shape}")
    except Exception as e:
        st.error(f"Error in scaling step: {e}")
        return None

    # Feature selection
    try:
        if feature_extractor.feature_selector is not None:
            features_selected = feature_extractor.feature_selector.transform(features_scaled)
            st.write(f"**Step 3 - After feature selection:** {features_selected.shape}")
        else:
            features_selected = features_scaled
            st.write(f"**Step 3 - No feature selection:** {features_selected.shape}")
    except Exception as e:
        st.error(f"Error in feature selection step: {e}")
        return None

    # PCA
    try:
        if feature_extractor.pca_transformer is not None:
            features_final = feature_extractor.pca_transformer.transform(features_selected)
            st.write(f"**Step 4 - After PCA:** {features_final.shape}")
        else:
            features_final = features_selected
            st.write(f"**Step 4 - No PCA applied:** {features_final.shape}")
    except Exception as e:
        st.error(f"Error in PCA step: {e}")
        return None

    st.success(f"**Final transformed features:** {features_final.shape}")
    return features_final

def display_pipeline_status(feature_extractor):
    """Display the current pipeline configuration"""
    st.subheader("Pipeline Status")

    if feature_extractor is None:
        st.error("Feature extractor not found")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Pipeline Components:**")
        if hasattr(feature_extractor, 'pipeline_info'):
            info = feature_extractor.pipeline_info
            st.write(f"- Original features: {info.get('original_features', 'Unknown')}")
            st.write(f"- Scaling: {'✅' if info.get('scaling_applied', False) else '❌'}")
            st.write(f"- Feature selection: {'✅' if info.get('feature_selection_applied', False) else '❌'}")
            st.write(f"- PCA: {'✅' if info.get('pca_applied', False) else '❌'}")
        else:
            st.write("Pipeline info not available")

    with col2:
        st.write("**Expected Output:**")
        expected_features = feature_extractor.get_expected_feature_count()
        st.write(f"- Final feature count: {expected_features}")

        if hasattr(feature_extractor, 'pipeline_info'):
            st.write(f"- Pipeline summary: {feature_extractor.get_pipeline_summary()}")

def safe_make_prediction_with_debug(image, model_name, feature_extractor, ml_results, debug_mode=False):
    """
    Make prediction with comprehensive debugging and validation
    """

    # Step 1: Validate pipeline
    is_valid, error_msg, expected_features = validate_feature_pipeline(feature_extractor, model_name, ml_results)

    if not is_valid:
        st.error(f"Pipeline validation failed: {error_msg}")
        if debug_mode:
            display_pipeline_status(feature_extractor)
        return None, None, error_msg

    # Step 2: Extract features
    temp_path = "temp_pred_image.jpg"
    image.save(temp_path)

    try:
        raw_features = feature_extractor.extract_single_image_features(temp_path)

        if raw_features is None:
            return None, None, "Could not extract features from image"

        if debug_mode:
            st.write(f"Raw features extracted: {raw_features.shape}")

        # Step 3: Transform features with optional debugging
        if debug_mode:
            features_transformed = debug_feature_transformation(feature_extractor, raw_features)
        else:
            features_transformed = feature_extractor.transform_features(raw_features)

        if features_transformed is None:
            return None, None, "Feature transformation failed"

        # Step 4: Final validation before prediction
        model_key = model_name.replace("ML_", "")
        model = ml_results[model_key]['model']

        if debug_mode:
            st.write(f"Model expects: {expected_features} features")
            st.write(f"Providing: {features_transformed.shape[1]} features")

        # Step 5: Make prediction
        prediction = model.predict(features_transformed)[0]

        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_transformed)[0]

        return prediction, probabilities, "Success"

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        if debug_mode:
            st.error(error_msg)
            st.exception(e)
        return None, None, error_msg

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


class ImageProcessor:
    """Professional image processing utilities"""

    @staticmethod
    def load_image_safely(image_path: str) -> Optional[np.ndarray]:
        """Safely load an image file with comprehensive error handling"""
        try:
            if not os.path.exists(image_path):
                st.warning(f"Image file not found: {image_path}")
                return None

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                st.warning(f"Could not decode image: {image_path}")
                return None

            return img

        except Exception as e:
            st.error(f"Error loading image {image_path}: {str(e)}")
            return None

    @staticmethod
    def load_color_image_safely(image_path: str) -> Optional[np.ndarray]:
        """Load color image safely"""
        try:
            if not os.path.exists(image_path):
                return None

            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img

        except Exception as e:
            st.error(f"Error loading color image {image_path}: {str(e)}")
            return None

    @staticmethod
    def validate_uploaded_image(uploaded_file) -> bool:
        """Validate uploaded image file with comprehensive checks"""
        if uploaded_file is None:
            return False

        # Check file type
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in DATA_CONFIG.valid_extensions:
            st.error(f"Invalid file type. Supported types: {DATA_CONFIG.valid_extensions}")
            return False

        # Check file size (limit to 10MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File size too large. Please upload an image smaller than 10MB.")
            return False

        # Try to open and validate image
        try:
            image = Image.open(uploaded_file)

            # Check image mode
            if image.mode not in ['RGB', 'L', 'RGBA']:
                st.warning("Image format may not be optimal. Converting to RGB.")

            # Check minimum dimensions
            if image.width < 32 or image.height < 32:
                st.error("Image too small. Minimum size: 32x32 pixels")
                return False

            return True

        except Exception as e:
            st.error(f"Invalid image file: {str(e)}")
            return False


class DatasetAnalyzer:
    """Comprehensive dataset analysis utilities"""

    @staticmethod
    def get_image_files(directory: str) -> List[str]:
        """Get all valid image files from a directory with sorting"""
        if not os.path.exists(directory):
            return []

        files = []
        for file in sorted(os.listdir(directory)):
            if os.path.splitext(file)[1].lower() in DATA_CONFIG.valid_extensions:
                files.append(file)
        return files

    @staticmethod
    def analyze_image_properties(image_paths: List[str], max_analyze: int = 100) -> Dict:
        """Analyze properties of images with progress tracking"""
        properties = {
            'widths': [],
            'heights': [],
            'sizes': [],
            'aspect_ratios': [],
            'color_modes': [],
            'file_formats': []
        }

        # Limit analysis for performance
        paths_to_analyze = image_paths[:max_analyze]

        progress_bar = st.progress(0)

        for i, img_path in enumerate(paths_to_analyze):
            try:
                # Get image dimensions and properties
                img = Image.open(img_path)
                properties['widths'].append(img.width)
                properties['heights'].append(img.height)
                properties['aspect_ratios'].append(img.width / img.height)
                properties['color_modes'].append(img.mode)
                properties['file_formats'].append(img.format)

                # Get file size in KB
                size_kb = os.path.getsize(img_path) / 1024
                properties['sizes'].append(size_kb)

            except Exception as e:
                continue

            # Update progress
            progress_bar.progress((i + 1) / len(paths_to_analyze))

        progress_bar.empty()

        # Add summary statistics
        if properties['widths']:
            properties['stats'] = {
                'total_analyzed': len(properties['widths']),
                'avg_width': np.mean(properties['widths']),
                'avg_height': np.mean(properties['heights']),
                'avg_size_kb': np.mean(properties['sizes']),
                'avg_aspect_ratio': np.mean(properties['aspect_ratios']),
                'most_common_format': max(set(properties['file_formats']),
                                        key=properties['file_formats'].count) if properties['file_formats'] else 'Unknown',
                'most_common_mode': max(set(properties['color_modes']),
                                      key=properties['color_modes'].count) if properties['color_modes'] else 'Unknown'
            }

        return properties

    @staticmethod
    def calculate_data_quality_metrics(class_counts: Dict) -> Dict:
        """Calculate comprehensive data quality metrics"""
        if not class_counts or not any(class_counts.values()):
            return {}

        counts = list(class_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        total_count = sum(counts)

        metrics = {
            'total_images': total_count,
            'num_classes': len(class_counts),
            'max_class_size': max_count,
            'min_class_size': min_count,
            'imbalance_ratio': max_count / min_count if min_count > 0 else float('inf'),
            'balance_score': 1 - (max_count - min_count) / max_count if max_count > 0 else 0,
            'avg_class_size': total_count / len(class_counts),
            'std_class_size': np.std(counts),
            'cv_class_size': np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0,
            'class_distribution': class_counts,
            'gini_coefficient': DatasetAnalyzer._calculate_gini(counts)
        }

        return metrics

    @staticmethod
    def _calculate_gini(values):
        """Calculate Gini coefficient for class distribution"""
        if not values:
            return 0

        values = sorted(values)
        n = len(values)
        cumulative = np.cumsum(values)

        return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n

    @staticmethod
    def generate_quality_recommendations(metrics: Dict) -> List[str]:
        """Generate comprehensive data quality recommendations"""
        recommendations = []

        # Class balance analysis
        imbalance_ratio = metrics.get('imbalance_ratio', 1)
        if imbalance_ratio > 3:
            recommendations.append("Warning: Severe class imbalance detected! Strongly consider data augmentation or resampling.")
        elif imbalance_ratio > 1.5:
            recommendations.append("Warning: Moderate class imbalance detected. Consider data augmentation for smaller classes.")
        else:
            recommendations.append("Good: Classes are reasonably balanced.")

        # Dataset size analysis
        total_images = metrics.get('total_images', 0)
        if total_images < 500:
            recommendations.append("Warning: Very small dataset. Consider data augmentation and transfer learning.")
        elif total_images < 1000:
            recommendations.append("Warning: Small dataset size. Consider data augmentation techniques.")
        elif total_images > 10000:
            recommendations.append("Good: Large dataset - excellent for deep learning approaches.")
        else:
            recommendations.append("Good: Dataset size appears adequate for training.")

        # Per-class analysis
        min_class_size = metrics.get('min_class_size', 0)
        if min_class_size < 50:
            recommendations.append("Warning: Some classes have very few samples. This may significantly affect model performance.")
        elif min_class_size < 100:
            recommendations.append("Warning: Some classes have few samples. Consider focused data collection or augmentation.")

        # Gini coefficient analysis
        gini = metrics.get('gini_coefficient', 0)
        if gini > 0.3:
            recommendations.append("Info: High class distribution inequality. Consider stratified sampling strategies.")

        return recommendations


class ModelEvaluator:
    """Professional model evaluation and comparison utilities"""

    @staticmethod
    def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray,
                                   class_names: List[str], title: str = "Confusion Matrix",
                                   normalize: bool = False) -> go.Figure:
        """Create professional interactive confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            text_template = '%{z:.2%}'
        else:
            text_template = '%{z:d}'

        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title=title,
            labels=dict(x="Predicted Class", y="True Class"),
            x=class_names,
            y=class_names,
            color_continuous_scale=VISUALIZATION_CONFIG.confusion_matrix_cmap
        )

        fig.update_traces(texttemplate=text_template)
        fig.update_layout(
            height=VISUALIZATION_CONFIG.dashboard_height // 2,
            font=dict(size=12)
        )

        return fig

    @staticmethod
    def create_comprehensive_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                                            class_names: List[str], model_name: str) -> plt.Figure:
        """Create comprehensive confusion matrix with matplotlib"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title(f'Confusion Matrix - {model_name}')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')

        # Normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax2, cbar_kws={'label': 'Proportion'})
        ax2.set_title(f'Normalized Confusion Matrix - {model_name}')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')

        # Per-class metrics
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

        classes = class_names
        precision = [report[cls]['precision'] for cls in classes]
        recall = [report[cls]['recall'] for cls in classes]
        f1_score = [report[cls]['f1-score'] for cls in classes]

        x = np.arange(len(classes))
        width = 0.25

        ax3.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax3.bar(x, recall, width, label='Recall', alpha=0.8)
        ax3.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)

        ax3.set_xlabel('Classes')
        ax3.set_ylabel('Score')
        ax3.set_title('Per-Class Performance Metrics')
        ax3.set_xticks(x)
        ax3.set_xticklabels(classes, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)

        ax4.bar(classes, per_class_acc, color='lightgreen', alpha=0.7)
        ax4.set_xlabel('Classes')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Per-Class Accuracy')
        ax4.set_xticklabels(classes, rotation=45)

        # Add value labels
        for i, v in enumerate(per_class_acc):
            ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def create_metrics_comparison_plot(results: Dict, metric: str = 'accuracy') -> go.Figure:
        """Create enhanced model comparison plot"""
        models = list(results.keys())
        values = []

        for model in models:
            if isinstance(results[model], dict):
                values.append(results[model].get(metric, 0))
            else:
                values.append(results[model])

        # Sort by performance
        sorted_data = sorted(zip(models, values), key=lambda x: x[1], reverse=True)
        sorted_models, sorted_values = zip(*sorted_data)

        fig = px.bar(
            x=list(sorted_models),
            y=list(sorted_values),
            title=f"Model Performance Comparison - {metric.title()}",
            labels={'x': 'Model', 'y': metric.title()},
            color=list(sorted_values),
            color_continuous_scale=VISUALIZATION_CONFIG.performance_cmap,
            text=[f'{v:.3f}' for v in sorted_values]
        )

        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            showlegend=False
        )

        return fig

    @staticmethod
    def format_metrics_table(results: Dict) -> pd.DataFrame:
        """Format model results into comprehensive metrics table"""
        metrics_data = []

        for name, result in results.items():
            if isinstance(result, dict) and 'y_test' in result and 'predictions' in result:
                y_test = result['y_test']
                predictions = result['predictions']

                if predictions is not None:
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_test, predictions, average='weighted'
                    )

                    # Calculate per-class metrics
                    cm = confusion_matrix(y_test, predictions)
                    per_class_acc = cm.diagonal() / cm.sum(axis=1)

                    metrics_data.append({
                        'Model': name,
                        'Accuracy': f"{result['accuracy']:.4f}",
                        'Precision': f"{precision:.4f}",
                        'Recall': f"{recall:.4f}",
                        'F1-Score': f"{f1:.4f}",
                        'Best Class Acc': f"{np.max(per_class_acc):.4f}",
                        'Worst Class Acc': f"{np.min(per_class_acc):.4f}",
                        'Training Time': f"{result.get('training_time', 0):.2f}s"
                    })
                else:
                    metrics_data.append({
                        'Model': name,
                        'Accuracy': f"{result['accuracy']:.4f}",
                        'Precision': 'N/A',
                        'Recall': 'N/A',
                        'F1-Score': 'N/A',
                        'Best Class Acc': 'N/A',
                        'Worst Class Acc': 'N/A',
                        'Training Time': f"{result.get('training_time', 0):.2f}s"
                    })
            else:
                # Handle simple accuracy results
                accuracy = result if isinstance(result, (int, float)) else result.get('accuracy', 0)
                metrics_data.append({
                    'Model': name,
                    'Accuracy': f"{accuracy:.4f}",
                    'Precision': 'N/A',
                    'Recall': 'N/A',
                    'F1-Score': 'N/A',
                    'Best Class Acc': 'N/A',
                    'Worst Class Acc': 'N/A',
                    'Training Time': 'N/A'
                })

        df = pd.DataFrame(metrics_data)
        return df.sort_values('Accuracy', ascending=False)


class VisualizationEngine:
    """Professional visualization engine for comprehensive analysis"""

    @staticmethod
    def create_class_distribution_plot(class_counts: Dict) -> go.Figure:
        """Create enhanced class distribution visualization"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Class Distribution (Count)', 'Class Distribution (%)'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )

        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        total = sum(counts)
        percentages = [count/total * 100 for count in counts]

        # Bar chart
        fig.add_trace(
            go.Bar(
                x=classes,
                y=counts,
                name="Count",
                marker_color=px.colors.qualitative.Set3[:len(classes)],
                text=counts,
                textposition='outside'
            ),
            row=1, col=1
        )

        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=classes,
                values=counts,
                name="Distribution",
                textinfo='label+percent'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=500,
            showlegend=False,
            title_text="Dataset Class Analysis"
        )

        return fig

    @staticmethod
    def create_feature_distribution_plot(features: np.ndarray,
                                       title: str = "Feature Distribution") -> go.Figure:
        """Create feature distribution analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Histogram', 'Box Plot', 'Q-Q Plot', 'Statistics'),
        )

        # Histogram
        fig.add_trace(
            go.Histogram(x=features, nbinsx=50, name="Distribution"),
            row=1, col=1
        )

        # Box plot
        fig.add_trace(
            go.Box(y=features, name="Box Plot"),
            row=1, col=2
        )

        # Statistics summary
        stats = {
            'Mean': np.mean(features),
            'Std': np.std(features),
            'Min': np.min(features),
            'Max': np.max(features),
            'Median': np.median(features),
            'Skewness': float(pd.Series(features).skew()),
            'Kurtosis': float(pd.Series(features).kurtosis())
        }

        fig.add_trace(
            go.Table(
                header=dict(values=['Statistic', 'Value']),
                cells=dict(values=[list(stats.keys()),
                                 [f'{v:.4f}' for v in stats.values()]])
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=VISUALIZATION_CONFIG.dashboard_height,
            title_text=title,
            showlegend=False
        )

        return fig

    @staticmethod
    def create_image_stats_dashboard(properties: Dict, title: str = "Image Analysis Dashboard") -> go.Figure:
        """Create comprehensive image statistics dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Width Distribution', 'Height Distribution',
                          'File Size Distribution', 'Aspect Ratio Distribution',
                          'Dimension Scatter', 'Summary Statistics'),
        )

        # Width distribution
        fig.add_trace(
            go.Histogram(x=properties['widths'], name="Width", nbinsx=30),
            row=1, col=1
        )

        # Height distribution
        fig.add_trace(
            go.Histogram(x=properties['heights'], name="Height", nbinsx=30),
            row=1, col=2
        )

        # File size distribution
        fig.add_trace(
            go.Histogram(x=properties['sizes'], name="Size (KB)", nbinsx=30),
            row=2, col=1
        )

        # Aspect ratio distribution
        fig.add_trace(
            go.Histogram(x=properties['aspect_ratios'], name="Aspect Ratio", nbinsx=30),
            row=2, col=2
        )

        # Dimension scatter plot
        fig.add_trace(
            go.Scatter(
                x=properties['widths'],
                y=properties['heights'],
                mode='markers',
                name="Dimensions",
                opacity=0.6
            ),
            row=3, col=1
        )

        # Summary statistics
        if 'stats' in properties:
            stats = properties['stats']
            fig.add_trace(
                go.Table(
                    header=dict(values=['Property', 'Value']),
                    cells=dict(values=[
                        ['Images Analyzed', 'Avg Width', 'Avg Height', 'Avg Size (KB)',
                         'Avg Aspect Ratio', 'Most Common Format', 'Most Common Mode'],
                        [stats['total_analyzed'], f"{stats['avg_width']:.0f}px",
                         f"{stats['avg_height']:.0f}px", f"{stats['avg_size_kb']:.1f}KB",
                         f"{stats['avg_aspect_ratio']:.2f}", stats['most_common_format'],
                         stats['most_common_mode']]
                    ])
                ),
                row=3, col=2
            )

        fig.update_layout(
            height=900,
            showlegend=False,
            title_text=title
        )

        return fig


class ModelPersistence:
    """Professional model saving and loading utilities"""

    @staticmethod
    def save_model_components(scaler, selector, models: Dict,
                            save_dir: str = "saved_models",
                            include_metadata: bool = True) -> bool:
        """Save trained model components with metadata"""
        ensure_directories()
        os.makedirs(save_dir, exist_ok=True)

        try:
            # Save scaler and selector
            if scaler is not None:
                joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
            if selector is not None:
                joblib.dump(selector, os.path.join(save_dir, "selector.joblib"))

            # Save individual models
            saved_models = {}
            for name, model_info in models.items():
                if model_info is not None and 'model' in model_info:
                    model_filename = f"{name.lower().replace(' ', '_')}_model.joblib"
                    model_path = os.path.join(save_dir, model_filename)
                    joblib.dump(model_info['model'], model_path)

                    saved_models[name] = {
                        'filename': model_filename,
                        'accuracy': model_info.get('accuracy', 0),
                        'training_time': model_info.get('training_time', 0)
                    }

            # Save metadata
            if include_metadata:
                metadata = {
                    'save_timestamp': datetime.now().isoformat(),
                    'config_snapshot': {
                        'classes': DATA_CONFIG.classes,
                        'random_state': MODEL_CONFIG.random_state,
                        'feature_config': {
                            'target_size': FEATURE_CONFIG.target_size,
                            'hog_orientations': FEATURE_CONFIG.hog_orientations
                        }
                    },
                    'saved_models': saved_models,
                    'total_models': len(saved_models)
                }

                with open(os.path.join(save_dir, "metadata.json"), 'w') as f:
                    json.dump(metadata, f, indent=2)

            return True

        except Exception as e:
            st.error(f"Error saving models: {str(e)}")
            return False

    @staticmethod
    def load_model_components(save_dir: str = "saved_models"):
        """Load saved model components with validation"""
        try:
            # Load scaler and selector
            scaler_path = os.path.join(save_dir, "scaler.joblib")
            selector_path = os.path.join(save_dir, "selector.joblib")

            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            selector = joblib.load(selector_path) if os.path.exists(selector_path) else None

            # Load available models
            models = {}
            model_files = [f for f in os.listdir(save_dir) if f.endswith('_model.joblib')]

            for model_file in model_files:
                model_name = model_file.replace('_model.joblib', '').replace('_', ' ').title()
                model_path = os.path.join(save_dir, model_file)
                model = joblib.load(model_path)

                models[model_name] = {
                    'model': model,
                    'loaded_from': model_file
                }

            # Load metadata if available
            metadata_path = os.path.join(save_dir, "metadata.json")
            metadata = None
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            return scaler, selector, models, metadata

        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return None, None, {}, None


class PredictionInterface:
    """Professional prediction interface utilities"""

    @staticmethod
    def display_prediction_results(prediction: int, probabilities: Optional[np.ndarray],
                                 class_names: List[str], true_class: Optional[str] = None,
                                 model_name: str = "Model"):
        """Display prediction results with enhanced visualization"""
        predicted_class = class_names[prediction]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Prediction Result")

            # Create metrics display
            col1a, col1b = st.columns(2)

            with col1a:
                st.metric("Model", model_name)
                st.metric("Predicted Class", predicted_class)

            with col1b:
                if probabilities is not None:
                    confidence = np.max(probabilities)
                    st.metric("Confidence", f"{confidence:.1%}")

                    # Confidence level indicator
                    if confidence > 0.8:
                        st.success("High Confidence")
                    elif confidence > 0.6:
                        st.warning("Medium Confidence")
                    else:
                        st.error("Low Confidence")

            if true_class:
                is_correct = predicted_class.lower() == true_class.lower()
                result_text = "Correct" if is_correct else "Incorrect"
                result_color = "success" if is_correct else "error"

                if is_correct:
                    st.success(f"**Result:** {result_text}")
                else:
                    st.error(f"**Result:** {result_text}")
                    st.info(f"**True Class:** {true_class.title()}")

        with col2:
            if probabilities is not None:
                st.subheader("Class Probabilities")

                # Create probability DataFrame
                prob_df = pd.DataFrame({
                    'Class': [cls.title() for cls in class_names],
                    'Probability': probabilities,
                    'Percentage': probabilities * 100
                }).sort_values('Probability', ascending=False)

                # Probability bar chart
                fig = px.bar(
                    prob_df,
                    x='Class',
                    y='Percentage',
                    title=f"Prediction Confidence - {model_name}",
                    color='Percentage',
                    color_continuous_scale='viridis',
                    text=[f'{p:.1f}%' for p in prob_df['Percentage']]
                )

                fig.update_traces(textposition='outside')
                fig.update_layout(
                    xaxis_title="Animal Class",
                    yaxis_title="Confidence (%)",
                    showlegend=False,
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Probability table
                st.subheader("Detailed Probabilities")
                st.dataframe(
                    prob_df.style.format({'Probability': '{:.4f}', 'Percentage': '{:.2f}%'}),
                    use_container_width=True
                )


class StyleManager:
    """Professional styling and CSS management"""

    @staticmethod
    def create_custom_css():
        """Create comprehensive custom CSS for the application"""
        css = f"""
        <style>
            /* Main Layout Styling */
            .main-header {{
                font-size: 3rem;
                color: {UI_CONFIG.primary_color};
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
                font-weight: 700;
                background: linear-gradient(135deg, {UI_CONFIG.primary_color}, {UI_CONFIG.secondary_color});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            
            /* Card Styling */
            .metric-container {{
                background: linear-gradient(135deg, {UI_CONFIG.secondary_color} 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 15px;
                margin: 0.5rem 0;
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                color: white;
                text-align: center;
            }}
            
            .status-card {{
                background: white;
                padding: 1.5rem;
                border-radius: 15px;
                border: 1px solid #e0e0e0;
                margin: 1rem 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.08);
                transition: all 0.3s ease;
            }}
            
            .status-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.15);
            }}
            
            /* Message Boxes */
            .success-message {{
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                border-left: 5px solid {UI_CONFIG.success_color};
                padding: 1.5rem;
                margin: 1rem 0;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                color: #155724;
            }}
            
            .warning-message {{
                background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                border-left: 5px solid {UI_CONFIG.warning_color};
                padding: 1.5rem;
                margin: 1rem 0;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                color: #856404;
            }}
            
            .error-message {{
                background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
                border-left: 5px solid {UI_CONFIG.danger_color};
                padding: 1.5rem;
                margin: 1rem 0;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                color: #721c24;
            }}
            
            .info-message {{
                background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
                border-left: 5px solid {UI_CONFIG.info_color};
                padding: 1.5rem;
                margin: 1rem 0;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                color: #0c5460;
            }}
            
            /* Progress Bar */
            .stProgress > div > div > div > div {{
                background: linear-gradient(135deg, {UI_CONFIG.primary_color}, {UI_CONFIG.secondary_color});
            }}
        </style>
        """
        return css

    @staticmethod
    def display_success_message(message: str, title: str = "Success"):
        """Display styled success message"""
        st.markdown(f"""
        <div class="success-message">
            <h4>{title}</h4>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def display_warning_message(message: str, title: str = "Warning"):
        """Display styled warning message"""
        st.markdown(f"""
        <div class="warning-message">
            <h4>{title}</h4>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)


class ProgressTracker:
    """Professional progress tracking with enhanced features"""

    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.description = description
        self.start_time = datetime.now()
        self.step_times = []

    def update(self, step_description: str = "", increment: int = 1):
        """Update progress with timing information"""
        self.current_step += increment
        progress = min(self.current_step / self.total_steps, 1.0)
        self.progress_bar.progress(progress)

        # Calculate timing
        current_time = datetime.now()
        self.step_times.append(current_time)

        elapsed = (current_time - self.start_time).total_seconds()

        if self.current_step > 0:
            avg_time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining = remaining_steps * avg_time_per_step

            if step_description:
                status = f"{self.description}: {step_description} ({self.current_step}/{self.total_steps}) - ETA: {estimated_remaining:.1f}s"
            else:
                status = f"{self.description}: {self.current_step}/{self.total_steps} - ETA: {estimated_remaining:.1f}s"
        else:
            status = f"{self.description}: {step_description}" if step_description else self.description

        self.status_text.text(status)

    def finish(self, success_message: str = "Complete"):
        """Finish progress tracking with summary"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        self.progress_bar.empty()
        self.status_text.success(f"{success_message} - Total time: {total_time:.2f}s")


class ReportGenerator:
    """Professional report generation utilities"""

    @staticmethod
    def generate_model_performance_report(all_results: Dict,
                                        dataset_metrics: Dict = None,
                                        save_path: str = None) -> str:
        """Generate comprehensive model performance report"""

        # Initialize report
        report = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report.append("=" * 80)
        report.append("AFRICAN WILDLIFE CLASSIFICATION - MODEL PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {timestamp}")
        report.append(f"System Configuration: {DEEP_LEARNING_CONFIG.device}")
        report.append("")

        # Dataset summary
        if dataset_metrics:
            report.append("DATASET SUMMARY")
            report.append("-" * 40)
            report.append(f"Total Images: {dataset_metrics.get('total_images', 'N/A'):,}")
            report.append(f"Number of Classes: {dataset_metrics.get('num_classes', 'N/A')}")
            report.append(f"Class Balance Ratio: {dataset_metrics.get('imbalance_ratio', 'N/A'):.2f}")
            report.append(f"Gini Coefficient: {dataset_metrics.get('gini_coefficient', 'N/A'):.3f}")
            report.append("")

        # Model performance ranking
        if all_results:
            sorted_results = sorted(all_results.items(),
                                  key=lambda x: x[1].get('accuracy', 0), reverse=True)

            report.append("MODEL PERFORMANCE RANKING")
            report.append("-" * 40)

            for i, (name, result) in enumerate(sorted_results, 1):
                accuracy = result.get('accuracy', 0)
                training_time = result.get('training_time', 0)
                report.append(f"{i:2d}. {name:<20} : {accuracy:.4f} ({training_time:.2f}s)")

            report.append("")

            # Best model analysis
            best_name, best_result = sorted_results[0]
            report.append("BEST MODEL ANALYSIS")
            report.append("-" * 40)
            report.append(f"Model: {best_name}")
            report.append(f"Accuracy: {best_result.get('accuracy', 0):.4f} ({best_result.get('accuracy', 0)*100:.2f}%)")
            report.append(f"Training Time: {best_result.get('training_time', 0):.2f} seconds")
            report.append("")

        report.append("=" * 80)

        # Join report and optionally save
        report_text = "\n".join(report)

        if save_path:
            ensure_directories()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_text)

        return report_text


# Convenience functions for backward compatibility
def create_progress_tracker(total_steps: int, description: str = "Processing") -> ProgressTracker:
    """Create a progress tracking instance - legacy function"""
    return ProgressTracker(total_steps, description)