"""
Complete Streamlit Application for African Wildlife Classification System
Fixed version with stable visualizations and proper state management
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import hashlib

# Import custom modules
try:
    from config import Config
    from session_manager import SessionManager
    from data_manager import DataManager
    from feature_engineering import FeatureExtractor, FeatureOptimizer
    from ml_models import MLModelTrainer
    from deep_learning_models import (
        DeepLearningTrainer, DeepLearningVisualizer,
        create_model_ensemble, analyze_model_complexity
    )
    from visualization_utils import ProfessionalVisualizer
    from utils import (
        validate_feature_pipeline, debug_feature_transformation,
        display_pipeline_status, safe_make_prediction_with_debug,
        StyleManager
    )
except ImportError as e:
    st.error(f"Required modules not found: {e}")
    st.error("Please ensure all custom modules are properly installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Wildlife Classification System",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS with visualization fixes
st.markdown("""
<style>
    /* Force plot refresh and visibility */
    .js-plotly-plot .plotly .modebar {
        display: block !important;
    }
    
    .js-plotly-plot .plotly .modebar-container {
        opacity: 1 !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        overflow: visible !important;
    }
    
    .plotly-graph-div {
        height: auto !important;
        width: 100% !important;
    }
    
    /* Force re-render of matplotlib plots */
    .element-container .stPlotlyChart > div {
        width: 100% !important;
    }
    
    /* Main styling */
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        font-weight: 700;
        background: linear-gradient(135deg, #2E8B57, #667eea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        color: white;
        text-align: center;
    }
    
    .success-message {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: #155724;
    }
    
    .info-message {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 5px solid #17a2b8;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)


class WildlifeClassificationApp:
    """Main application class with enhanced deep learning visualizations and fixed plots"""

    def __init__(self):
        self.session_manager = SessionManager()
        self.data_manager = DataManager()
        self.visualizer = ProfessionalVisualizer()

        # Initialize session state
        self.session_manager.initialize_session_state()

        # Load previous session if available
        if st.sidebar.button("📄 Load Previous Session"):
            if self.session_manager.load_session():
                st.success("Previous session loaded successfully!")
                st.experimental_rerun()

        # Auto-save session periodically
        if hasattr(st.session_state, 'last_autosave'):
            if time.time() - st.session_state.last_autosave > 300:  # Every 5 minutes
                self.session_manager.save_session()
                st.session_state.last_autosave = time.time()
        else:
            st.session_state.last_autosave = time.time()

    def display_header(self):
        """Display professional header"""
        st.markdown('<h1 class="main-header">🦁 African Wildlife Classification System</h1>',
                   unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #666; margin-bottom: 2rem;">Advanced ML & Deep Learning with Fixed Visualizations</h3>',
                   unsafe_allow_html=True)

        # Display validation strategy info
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>{Config.CV_FOLDS}-Fold CV</h3>
                <p>Traditional ML</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h3>70/15/15</h3>
                <p>Deep Learning</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            device_info = "GPU" if hasattr(Config, 'DEVICE') and Config.DEVICE.type == "cuda" else "CPU"
            st.markdown(f"""
            <div class="metric-container">
                <h3>{device_info}</h3>
                <p>Device</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            n_jobs = getattr(Config, 'N_JOBS', 'Auto')
            st.markdown(f"""
            <div class="metric-container">
                <h3>{n_jobs}</h3>
                <p>CPU Cores</p>
            </div>
            """, unsafe_allow_html=True)

    def display_sidebar_status(self):
        """Display comprehensive sidebar status with visualization controls"""
        st.sidebar.markdown("### 📊 System Status")

        status = self.session_manager.get_training_status()

        for task, completed in status.items():
            if completed:
                st.sidebar.markdown(f"✅ {task}")
            else:
                st.sidebar.markdown(f"⏳ {task}")

        # Visualization Controls
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎨 Visualization Controls")

        if st.sidebar.button("🔄 Refresh All Plots"):
            self.clear_visualization_cache()
            st.sidebar.success("All plots refreshed!")
            st.experimental_rerun()

        if st.sidebar.button("🧹 Clear Plot Cache"):
            self.clear_plot_cache()
            st.sidebar.success("Plot cache cleared!")

        # Validation strategy info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔍 Validation Strategy")

        st.sidebar.info(f"Traditional ML: {Config.CV_FOLDS}-fold CV")
        st.sidebar.info("Deep Learning: 70/15/15 split")

        # Enhanced features info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ✨ Enhanced Features")
        st.sidebar.success("🎨 Fixed Visualizations")
        st.sidebar.success("🧠 Model Interpretability")
        st.sidebar.success("📈 Real-time Training Analytics")
        st.sidebar.success("🎯 Ensemble Predictions")

        # Quick actions
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚡ Quick Actions")

        if st.sidebar.button("💾 Save Session"):
            self.session_manager.save_session()
            st.sidebar.success("Session saved!")

        if st.sidebar.button("🗑️ Clear All Data"):
            if st.sidebar.checkbox("Confirm deletion"):
                self.session_manager.clear_session()
                self.clear_visualization_cache()
                st.sidebar.success("All data cleared!")
                st.experimental_rerun()

        # Dataset info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📁 Dataset Info")

        dataset_exists, message, class_counts = self.data_manager.check_dataset_existence()
        if dataset_exists:
            total_images = sum([info['count'] if isinstance(info, dict) else info
                              for info in class_counts.values()])
            st.sidebar.success(f"Dataset: {total_images} images")

            for cls, info in class_counts.items():
                count = info['count'] if isinstance(info, dict) else info
                st.sidebar.text(f"{cls.title()}: {count}")
        else:
            st.sidebar.error("Dataset not found")

    def _display_existing_feature_engineering_results(self):
        """Display existing feature engineering results if available"""

        # Check if we have feature extraction results
        if 'feature_extractor' in st.session_state and st.session_state.feature_extractor is not None:
            feature_extractor = st.session_state.feature_extractor

            # Display feature extraction summary
            with st.expander("📊 Feature Extraction Results", expanded=False):
                st.subheader("🔍 Feature Extraction Summary")

                if hasattr(feature_extractor, 'pipeline_info') and feature_extractor.pipeline_info:
                    info = feature_extractor.pipeline_info

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Original Features", info.get('original_features', 'N/A'))
                    with col2:
                        st.metric("Total Samples", info.get('total_samples', 'N/A'))
                    with col3:
                        scaling_status = "✅ Applied" if info.get('scaling_applied', False) else "❌ Not Applied"
                        st.metric("Scaling", scaling_status)
                    with col4:
                        pipeline_summary = feature_extractor.get_pipeline_summary()
                        st.text_area("Pipeline Summary", pipeline_summary, height=100)

                else:
                    st.info("Feature extractor exists but detailed info not available")

        # Check if we have feature optimization results
        if 'feature_optimizer' in st.session_state and st.session_state.feature_optimizer is not None:
            optimizer = st.session_state.feature_optimizer

            with st.expander("🎯 Feature Selection & PCA Results", expanded=True):
                st.subheader("🎯 Advanced Feature Selection Analysis")

                # Re-display feature selection results if available
                if hasattr(optimizer, 'results') and 'feature_selection' in optimizer.results:
                    performance_results = optimizer.results['feature_selection']

                    # Get original features count from feature extractor
                    original_features = st.session_state.get('X', np.array([])).shape[1] if 'X' in st.session_state else 8186

                    # Use the fixed display method from feature engineering
                    try:
                        optimizer._display_feature_selection_results_fixed(performance_results, original_features)
                    except Exception as e:
                        st.error(f"Error displaying feature selection results: {e}")

                        # Fallback: show basic table
                        results_data = []
                        for method, results in performance_results.items():
                            results_data.append({
                                'Method': method,
                                'Features': results['n_features'],
                                'CV Accuracy': f"{results['accuracy']:.4f}",
                                'Time (s)': f"{results['time']:.2f}"
                            })

                        if results_data:
                            df = pd.DataFrame(results_data)
                            st.dataframe(df, use_container_width=True)

                        if st.button("🔄 Refresh Feature Selection Display"):
                            st.experimental_rerun()

                # Display PCA results if available
                if hasattr(optimizer, 'pca_transformer') and optimizer.pca_transformer is not None:
                    st.subheader("📉 PCA Dimensionality Reduction")

                    # Show PCA summary
                    n_components = optimizer.pca_transformer.n_components_
                    explained_variance = optimizer.pca_transformer.explained_variance_ratio_.sum()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("PCA Components", n_components)
                    with col2:
                        st.metric("Explained Variance", f"{explained_variance:.3f}")
                    with col3:
                        st.metric("Variance %", f"{explained_variance*100:.1f}%")

                    # Try to recreate PCA visualization
                    if st.button("🔄 Show PCA Analysis", key="show_pca_analysis"):
                        try:
                            # Create simplified PCA visualization
                            thresholds = Config.PCA_VARIANCE_THRESHOLDS
                            components_data = []

                            for threshold in thresholds:
                                if threshold <= explained_variance:
                                    components_data.append({
                                        'Threshold': f"{threshold*100:.0f}%",
                                        'Components': n_components,
                                        'Explained_Variance': explained_variance
                                    })

                            if components_data:
                                df_pca = pd.DataFrame(components_data)
                                st.dataframe(df_pca, use_container_width=True)

                        except Exception as e:
                            st.error(f"Error creating PCA visualization: {e}")

                else:
                    st.info("PCA not applied in current pipeline")

        # Show current pipeline status
        if ('feature_extractor' in st.session_state and
            st.session_state.feature_extractor is not None and
            hasattr(st.session_state.feature_extractor, 'get_expected_feature_count')):

            st.subheader("🔧 Current Pipeline Status")

            expected_features = st.session_state.feature_extractor.get_expected_feature_count()
            pipeline_summary = st.session_state.feature_extractor.get_pipeline_summary()

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Pipeline:** {pipeline_summary}")
            with col2:
                st.info(f"**Expected Features:** {expected_features}")

    def clear_visualization_cache(self):
        """Clear all visualization-related session state"""
        keys_to_remove = []
        for key in st.session_state.keys():
            if any(viz_type in key for viz_type in [
                'feature_selection_', 'pca_analysis_', 'plot_', 'feature_',
                'pca_', 'viz_', 'chart_', 'graph_'
            ]):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del st.session_state[key]

    def clear_plot_cache(self):
        """Clear only plot cache"""
        keys_to_remove = []
        for key in st.session_state.keys():
            if any(plot_type in key for plot_type in ['plot_', 'chart_', 'graph_']):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del st.session_state[key]

    def show_system_overview(self):
        """Display comprehensive system overview"""
        st.header("🏠 System Overview")

        # Dataset check
        dataset_exists, message, class_counts = self.data_manager.check_dataset_existence()

        if not dataset_exists:
            st.error("Dataset not found! Please ensure the 'african-wildlife' directory exists with the required structure.")
            st.info("Expected structure: african-wildlife/{buffalo,elephant,rhino,zebra}/")
            return

        st.success(message)

        # Validation strategy explanation
        st.subheader("🔍 Mixed Validation Strategy")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="info-message">
            <strong>Traditional ML: Cross-Validation</strong><br>
            • {Config.CV_FOLDS}-fold cross-validation for model selection<br>
            • Uses 100% of data for training<br>
            • No separate test set holdout<br>
            • Robust performance estimates<br>
            • Ideal for smaller datasets
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="success-message">
            <strong>Deep Learning: Train/Val/Test Split</strong><br>
            • Stratified 70/15/15 split<br>
            • Separate validation for early stopping<br>
            • Independent test set for final evaluation<br>
            • Prevents overfitting in complex models<br>
            • Standard practice for deep learning
            </div>
            """, unsafe_allow_html=True)

        # Enhanced features showcase
        st.subheader("✨ Enhanced Features")

        feature_cols = st.columns(4)

        with feature_cols[0]:
            st.markdown("""
            <div class="metric-container">
                <h4>🎨 Fixed Visualizations</h4>
                <p>Stable plots that persist across tab switches</p>
            </div>
            """, unsafe_allow_html=True)

        with feature_cols[1]:
            st.markdown("""
            <div class="metric-container">
                <h4>🧠 Model Interpretability</h4>
                <p>Attention maps and prediction explanations</p>
            </div>
            """, unsafe_allow_html=True)

        with feature_cols[2]:
            st.markdown("""
            <div class="metric-container">
                <h4>📈 Real-time Analytics</h4>
                <p>Live training progress, overfitting detection</p>
            </div>
            """, unsafe_allow_html=True)

        with feature_cols[3]:
            st.markdown("""
            <div class="metric-container">
                <h4>🎯 Ensemble Methods</h4>
                <p>Multi-model predictions and analysis</p>
            </div>
            """, unsafe_allow_html=True)

        # EDA section
        st.subheader("📊 Exploratory Data Analysis")

        if st.session_state.get('eda_completed', False):
            st.success("✅ EDA completed successfully!")

            if 'eda_results' in st.session_state:
                eda_results = st.session_state.eda_results
                dataset_info = eda_results.get('dataset_info', {})
                total_images = dataset_info.get('total_images', 0)
                quality_report = eda_results.get('quality_report', {})

                # Quick metrics overview
                col3, col4, col5, col6 = st.columns(4)
                with col3:
                    st.metric("Total Images", f"{total_images:,}")
                with col4:
                    classes = getattr(Config, 'CLASSES', ['buffalo', 'elephant', 'rhino', 'zebra'])
                    st.metric("Classes", len(classes))
                with col5:
                    if class_counts:
                        balance_ratio = max([info['count'] if isinstance(info, dict) else info
                                           for info in class_counts.values()]) / \
                                      min([info['count'] if isinstance(info, dict) else info
                                          for info in class_counts.values()])
                        st.metric("Balance Ratio", f"{balance_ratio:.1f}")
                    else:
                        st.metric("Balance Ratio", "N/A")
                with col6:
                    quality_score = quality_report.get('quality_score', 0)
                    st.metric("Quality Score", f"{quality_score}%")

                recommendations = eda_results.get('recommendations', [])
                if recommendations:
                    with st.expander("💡 Dataset Recommendations"):
                        for rec in recommendations:
                            st.write(f"• {rec}")

            if st.button("🔄 Re-run EDA"):
                st.session_state.eda_completed = False
                st.session_state.eda_results = None

                with st.spinner("Performing comprehensive EDA..."):
                    eda_results = self.data_manager.perform_comprehensive_eda()
                    if eda_results:
                        self.session_manager.save_session()
                        st.success("EDA re-run completed!")
                    else:
                        st.error("EDA failed. Please check your dataset structure.")
        else:
            st.info("EDA not completed yet")
            if st.button("🚀 Start EDA", type="primary"):
                with st.spinner("Performing comprehensive EDA..."):
                    eda_results = self.data_manager.perform_comprehensive_eda()
                    if eda_results:
                        self.session_manager.save_session()
                        st.success("EDA completed!")
                    else:
                        st.error("EDA failed. Please check your dataset structure.")

    def show_traditional_ml(self):
        """Display traditional ML training interface with fixed visualizations"""
        st.header("🔧 Traditional Machine Learning")

        # Visualization refresh button at the top
        col_refresh, col_space = st.columns([1, 4])
        with col_refresh:
            if st.button("🔄 Refresh ML Visualizations", key="refresh_ml_main"):
                self.clear_visualization_cache()
                st.success("ML visualizations refreshed!")
                st.experimental_rerun()

        if not self.data_manager.check_dataset_existence()[0]:
            st.error("Dataset not found! Please check the System Overview.")
            return

        st.info(f"Using {Config.CV_FOLDS}-fold cross-validation for model evaluation")

        # Always show feature engineering results if they exist
        self._display_existing_feature_engineering_results()

        # Training configuration
        st.subheader("⚙️ Training Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            use_optimized = st.checkbox(
                "Use Optimized Hyperparameters",
                value=True,
                help="Use pre-tuned hyperparameters for better performance"
            )

        with col2:
            perform_feature_selection = st.checkbox(
                "Advanced Feature Selection",
                value=True,
                help="Compare multiple feature selection methods using cross-validation"
            )

        with col3:
            train_ensembles = st.checkbox(
                "Train Ensemble Methods",
                value=True,
                help="Train comprehensive ensemble models"
            )

        if st.session_state.get('ml_models_trained', False):
            st.success("✅ Traditional ML models already trained!")

            if st.button("🔄 Retrain Models"):
                st.session_state.ml_models_trained = False
                st.session_state.ml_results = None
                self.clear_visualization_cache()
                st.experimental_rerun()

            if 'ml_results' in st.session_state and st.session_state.ml_results:
                self.display_ml_results_fixed(st.session_state.ml_results)

            return

        if st.button("🚀 Start Traditional ML Training", type="primary"):
            self.train_traditional_ml(use_optimized, perform_feature_selection, train_ensembles)

    def train_traditional_ml(self, use_optimized, perform_feature_selection, train_ensembles):
        """Train traditional ML models with fixed visualizations"""

        start_time = time.time()

        try:
            # Feature extraction
            st.header("🔍 Feature Extraction")

            if 'feature_extractor' not in st.session_state or st.session_state.feature_extractor is None:
                feature_extractor = FeatureExtractor()

                with st.spinner("Extracting features for cross-validation..."):
                    X, y = feature_extractor.prepare_dataset()

                if X is None:
                    st.error("Feature extraction failed!")
                    return

                st.success(f"✅ Features extracted! Total samples: {X.shape[0]}")
                st.info(f"Original feature dimension: {X.shape[1]}")

                st.session_state.feature_extractor = feature_extractor
                st.session_state.X = X
                st.session_state.y = y
            else:
                feature_extractor = st.session_state.feature_extractor
                X = st.session_state.X
                y = st.session_state.y
                st.info("✅ Using cached features")

            # Feature optimization with fixed visualizations
            X_final = X

            if perform_feature_selection:
                st.header("🎯 Advanced Feature Selection")

                # Clear any existing feature selection visualizations
                self.clear_feature_selection_cache()

                if 'feature_optimizer' not in st.session_state:
                    optimizer = FeatureOptimizer()

                    # Feature selection with fixed visualization
                    with st.spinner("Performing feature selection analysis..."):
                        X_selected, best_selector = optimizer.compare_feature_selection_methods(X, y)

                    # PCA analysis with fixed visualization
                    with st.spinner("Performing PCA analysis..."):
                        X_final, final_components = optimizer.apply_pca_analysis(X_selected, y)

                    feature_extractor.set_feature_pipeline(
                        feature_selector=best_selector,
                        pca_transformer=optimizer.pca_transformer
                    )

                    st.session_state.feature_optimizer = optimizer
                    st.session_state.X_final = X_final
                else:
                    X_final = st.session_state.X_final
                    st.info("✅ Using cached feature optimization results")

            # Model training
            st.header("🤖 Model Training")

            trainer = MLModelTrainer()

            with st.spinner("Training individual models with cross-validation..."):
                individual_results = trainer.train_individual_models(X_final, y, use_optimized)

            if train_ensembles:
                st.subheader("🎯 Ensemble Methods")
                with st.spinner("Training ensemble methods..."):
                    ensemble_results = trainer.train_ensemble_methods(X_final, y, individual_results)
                    all_results = {**individual_results, **ensemble_results}
            else:
                all_results = individual_results

            st.session_state.ml_results = all_results
            st.session_state.ml_models_trained = True

            training_time = time.time() - start_time
            best_cv_acc = max([r['cv_mean'] for r in all_results.values() if r['model'] is not None])
            self.session_manager.add_training_event(
                'Traditional ML Training',
                f"{len(all_results)} models",
                best_cv_acc,
                training_time
            )

            self.display_ml_results_fixed(all_results)
            self.session_manager.save_session()

        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.exception(e)

    def clear_feature_selection_cache(self):
        """Clear feature selection visualization cache"""
        keys_to_remove = []
        for key in st.session_state.keys():
            if any(fs_type in key for fs_type in ['feature_selection_', 'pca_analysis_']):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del st.session_state[key]

    def display_ml_results_fixed(self, all_results):
        """Display ML results with fixed visualizations"""

        st.header("📊 Training Results")

        # Add refresh button for results
        if st.button("🔄 Refresh Results Visualization", key="refresh_ml_results"):
            self.clear_visualization_cache()
            st.experimental_rerun()

        trainer = MLModelTrainer()
        best_model_info = trainer.display_results(all_results)

        if best_model_info:
            st.subheader("📈 Performance Visualizations")

            # Create stable visualization container
            with st.container():
                try:
                    self.visualizer.create_model_comparison_dashboard(all_results)
                except Exception as e:
                    st.error(f"Visualization error: {e}")
                    st.info("Click 'Refresh Results Visualization' to reload the charts.")

        # Performance summary
        summary_data = []
        for name, result in all_results.items():
            if result['model'] is not None and result['accuracy'] > 0:
                row = {
                    'Model': name,
                    'CV Mean': f"{result['cv_mean']:.4f}",
                    'CV Std': f"{result['cv_std']:.4f}",
                    'CV Accuracy (%)': f"{result['cv_mean']*100:.2f}%",
                    'Training Time': f"{result.get('training_time', 0):.2f}s",
                }
                summary_data.append(row)

        if summary_data:
            df = pd.DataFrame(summary_data)
            df = df.sort_values('CV Mean', ascending=False)
            st.dataframe(df, use_container_width=True)

    def show_enhanced_deep_learning(self):
        """Enhanced deep learning training interface"""
        st.header("🧠 Enhanced Deep Learning Models")

        if not self.data_manager.check_dataset_existence()[0]:
            st.error("Dataset not found! Please check the System Overview.")
            return

        st.info("🎨 Enhanced deep learning with comprehensive visualizations")

        # Configuration
        st.subheader("⚙️ Advanced Training Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            model_choice = st.selectbox(
                "Select Model Architecture",
                ["Custom CNN", "ResNet-18 Transfer"],
                help="Choose between custom CNN or transfer learning"
            )

        with col2:
            epochs = st.slider("Training Epochs", 5, 30, 15)
            show_augmentation = st.checkbox("Show Data Augmentation", value=True)

        with col3:
            enable_visualizations = st.checkbox("Enable Advanced Visualizations", value=True)
            save_checkpoints = st.checkbox("Save Model Checkpoints", value=False)

        # Check training status
        model_key = 'cnn_trained' if model_choice == "Custom CNN" else 'resnet_trained'
        result_key = 'cnn_results' if model_choice == "Custom CNN" else 'resnet_results'

        if st.session_state.get(model_key, False):
            st.success(f"✅ {model_choice} already trained!")

            if result_key in st.session_state and st.session_state[result_key]:
                self.display_enhanced_deep_learning_results(st.session_state[result_key], enable_visualizations)

            if st.button(f"🔄 Retrain {model_choice}"):
                st.session_state[model_key] = False
                st.session_state[result_key] = None
                st.experimental_rerun()

            return

        # Training button
        if st.button(f"🚀 Start Enhanced {model_choice} Training", type="primary"):
            self.train_enhanced_deep_learning_model(model_choice, epochs, enable_visualizations, show_augmentation)

    def train_enhanced_deep_learning_model(self, model_choice, epochs, enable_visualizations=True, show_augmentation=True):
        """Enhanced deep learning training"""

        start_time = time.time()

        try:
            trainer = DeepLearningTrainer()

            # Training with visualizations
            if model_choice == "Custom CNN":
                with st.spinner("Training Custom CNN with enhanced analytics..."):
                    results = trainer.train_custom_cnn(epochs, enable_visualizations)

                if results:
                    st.session_state.cnn_results = results
                    st.session_state.cnn_trained = True

                    training_time = time.time() - start_time
                    self.session_manager.add_training_event(
                        'Enhanced Deep Learning Training',
                        'Custom CNN',
                        results['best_accuracy'],
                        training_time
                    )

                    self.display_enhanced_deep_learning_results(results, enable_visualizations)

            else:  # ResNet-18
                with st.spinner("Training ResNet-18 with enhanced analytics..."):
                    results = trainer.train_resnet_transfer(epochs, enable_visualizations)

                if results:
                    st.session_state.resnet_results = results
                    st.session_state.resnet_trained = True

                    training_time = time.time() - start_time
                    self.session_manager.add_training_event(
                        'Enhanced Deep Learning Training',
                        'ResNet-18',
                        results['best_accuracy'],
                        training_time
                    )

                    self.display_enhanced_deep_learning_results(results, enable_visualizations)

            self.session_manager.save_session()

        except Exception as e:
            st.error(f"Enhanced deep learning training failed: {str(e)}")
            st.exception(e)

    def display_enhanced_deep_learning_results(self, results, enable_visualizations=True):
        """Display enhanced deep learning results"""

        st.header("📊 Enhanced Deep Learning Results")

        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Test Accuracy", f"{results['best_accuracy']:.4f}")

        with col2:
            val_accuracy = results.get('val_accuracy', 0)
            st.metric("Validation Accuracy", f"{val_accuracy:.4f}")

        with col3:
            st.metric("Test Accuracy %", f"{results['best_accuracy']*100:.2f}%")

        with col4:
            final_epoch = results.get('final_epoch', 0)
            st.metric("Final Epoch", str(final_epoch))

        # Comprehensive analysis with visualizations
        if enable_visualizations and 'history' in results:
            trainer = DeepLearningTrainer()

            st.subheader("📈 Comprehensive Analysis")

            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["Training Dashboard", "Model Architecture", "Performance Analysis"])

            with tab1:
                # Training dashboard with unique key
                dashboard_key = f"training_dashboard_{hashlib.md5(str(results.get('history', {})).encode()).hexdigest()[:8]}"
                try:
                    trainer.visualizer.create_training_dashboard(results['history'], dashboard_key)
                except Exception as e:
                    st.error(f"Dashboard visualization error: {e}")
                    if st.button("Refresh Training Dashboard", key=f"refresh_{dashboard_key}"):
                        st.experimental_rerun()

            with tab2:
                # Model architecture with unique key
                arch_key = f"model_arch_{hashlib.md5(str(results.get('model_name', '')).encode()).hexdigest()[:8]}"
                try:
                    trainer.visualizer.create_model_architecture_visualization(
                        results['model'], results['model_name'], arch_key
                    )
                except Exception as e:
                    st.error(f"Architecture visualization error: {e}")
                    if st.button("Refresh Architecture View", key=f"refresh_arch_{arch_key}"):
                        st.experimental_rerun()

                # Model complexity analysis
                if results.get('model'):
                    try:
                        complexity = analyze_model_complexity(results['model'])

                        st.subheader("Model Complexity Analysis")
                        col5, col6, col7 = st.columns(3)
                        with col5:
                            st.metric("Parameters", f"{complexity['total_params']:,}")
                        with col6:
                            st.metric("Model Size (MB)", f"{complexity['model_size_mb']:.1f}")
                        with col7:
                            st.metric("Est. FLOPs", f"{complexity['estimated_flops']:,}")
                    except Exception as e:
                        st.error(f"Complexity analysis error: {e}")

            with tab3:
                # Performance analysis with error handling
                try:
                    self.create_performance_analysis(results)
                except Exception as e:
                    st.error(f"Performance analysis error: {e}")
                    if st.button("Refresh Performance Analysis"):
                        st.experimental_rerun()

        # Performance summary
        st.subheader("📋 Performance Summary")

        summary_data = {
            'Metric': ['Test Accuracy', 'Validation Accuracy', 'Final Epoch', 'Model Size'],
            'Value': [
                f"{results['best_accuracy']:.4f} ({results['best_accuracy']*100:.2f}%)",
                f"{val_accuracy:.4f} ({val_accuracy*100:.2f}%)",
                str(final_epoch),
                f"{sum(p.numel() for p in results['model'].parameters()):,} params" if results.get('model') else "N/A"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    def create_performance_analysis(self, results):
        """Create detailed performance analysis with error handling"""

        st.subheader("Model Performance Analysis")

        try:
            # Standard evaluation
            trainer = DeepLearningTrainer()
            evaluation_results = trainer.evaluate_model(results, results['model_name'])

            if evaluation_results:
                st.success("Model evaluation completed")
        except Exception as e:
            st.error(f"Performance analysis failed: {e}")
            st.info("Training completed successfully, but detailed analysis is unavailable.")

    def show_ensemble_predictions(self):
        """Ensemble predictions interface"""

        st.header("🎯 Ensemble Model Predictions")

        # Check available models
        available_models = {}

        if st.session_state.get('cnn_trained', False) and 'cnn_results' in st.session_state:
            available_models['Custom CNN'] = st.session_state.cnn_results['model']

        if st.session_state.get('resnet_trained', False) and 'resnet_results' in st.session_state:
            available_models['ResNet-18'] = st.session_state.resnet_results['model']

        if len(available_models) < 2:
            st.info("Train at least 2 deep learning models to enable ensemble predictions")
            return

        st.success(f"Ensemble available with {len(available_models)} models: {', '.join(available_models.keys())}")

        # Create ensemble
        ensemble_model = create_model_ensemble(
            {k: {'model': v} for k, v in available_models.items()},
            Config.DEVICE
        )

        if ensemble_model is None:
            st.error("Failed to create ensemble model")
            return

        # Prediction interface
        uploaded_file = st.file_uploader("Upload image for ensemble prediction", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("🎯 Make Ensemble Prediction"):
                self.make_ensemble_prediction(ensemble_model, image, list(available_models.keys()))

    def make_ensemble_prediction(self, ensemble_model, image, model_names):
        """Make ensemble prediction with error handling"""

        try:
            # Prepare image
            transform = transforms.Compose([
                transforms.Resize((Config.DEEP_IMG_SIZE, Config.DEEP_IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            input_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)

            # Make prediction
            ensemble_model.eval()
            with torch.no_grad():
                ensemble_output = ensemble_model(input_tensor)
                probabilities = ensemble_output.squeeze().cpu().numpy()
                prediction = probabilities.argmax()

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                predicted_class = Config.CLASSES[prediction]
                confidence = probabilities[prediction]

                st.metric("Predicted Class", predicted_class.title())
                st.metric("Ensemble Confidence", f"{confidence:.4f}")
                st.metric("Models Used", len(model_names))

                if confidence > 0.8:
                    st.success("High Confidence")
                elif confidence > 0.6:
                    st.warning("Medium Confidence")
                else:
                    st.error("Low Confidence")

            with col2:
                # Probability chart
                class_names = [cls.title() for cls in Config.CLASSES]
                prob_data = pd.DataFrame({
                    'Class': class_names,
                    'Probability': probabilities
                })

                import plotly.express as px
                fig = px.bar(prob_data, x='Class', y='Probability',
                            title="Ensemble Prediction Probabilities",
                            color='Probability', color_continuous_scale='viridis')
                fig.update_traces(texttemplate='%{y:.3f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Ensemble prediction failed: {str(e)}")

    def show_model_comparison(self):
        """Display comprehensive model comparison"""
        st.header("📊 Comprehensive Model Comparison")

        st.info("Traditional ML shows cross-validated performance. Deep learning shows test set performance.")

        # Collect all results
        all_results = {}

        # Traditional ML results
        if st.session_state.get('ml_models_trained', False) and 'ml_results' in st.session_state:
            ml_results = st.session_state.ml_results
            for name, result in ml_results.items():
                if result['model'] is not None and result['cv_mean'] > 0:
                    all_results[f"ML_{name}"] = {
                        'accuracy': result['cv_mean'],
                        'model_type': 'Traditional ML'
                    }

        # Deep learning results
        if st.session_state.get('cnn_trained', False) and 'cnn_results' in st.session_state:
            cnn_result = st.session_state.cnn_results
            all_results["Deep_Custom_CNN"] = {
                'accuracy': cnn_result['best_accuracy'],
                'model_type': 'Deep Learning'
            }

        if st.session_state.get('resnet_trained', False) and 'resnet_results' in st.session_state:
            resnet_result = st.session_state.resnet_results
            all_results["Deep_ResNet18"] = {
                'accuracy': resnet_result['best_accuracy'],
                'model_type': 'Deep Learning'
            }

        if not all_results:
            st.warning("No trained models found. Please train some models first!")
            return

        # Create comparison dashboard with error handling
        try:
            self.visualizer.create_model_comparison_dashboard(all_results)
        except Exception as e:
            st.error(f"Comparison visualization error: {e}")
            if st.button("Refresh Comparison Dashboard"):
                self.clear_visualization_cache()
                st.experimental_rerun()

        # Ranking table
        st.subheader("Model Performance Ranking")

        ranking_data = []
        for name, result in sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            clean_name = name.replace('ML_', '').replace('Deep_', '').replace('_', ' ')
            validation_method = "CV" if result['model_type'] == 'Traditional ML' else "Test"
            ranking_data.append({
                'Rank': len(ranking_data) + 1,
                'Model': clean_name,
                'Type': result['model_type'],
                'Validation': validation_method,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Accuracy (%)': f"{result['accuracy']*100:.2f}%"
            })

        df = pd.DataFrame(ranking_data)
        st.dataframe(df, use_container_width=True)

        # Performance insights
        best_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
        ml_models = {k: v for k, v in all_results.items() if v['model_type'] == 'Traditional ML'}
        dl_models = {k: v for k, v in all_results.items() if v['model_type'] == 'Deep Learning'}

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="success-message">
            <strong>Best Overall Performance</strong><br>
            Model: {best_model[0].replace('ML_', '').replace('Deep_', '').replace('_', ' ')}<br>
            Accuracy: {best_model[1]['accuracy']:.4f} ({best_model[1]['accuracy']*100:.2f}%)
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if ml_models and dl_models:
                best_ml = max(ml_models.values(), key=lambda x: x['accuracy'])['accuracy']
                best_dl = max(dl_models.values(), key=lambda x: x['accuracy'])['accuracy']

                if best_dl > best_ml:
                    advantage = "Deep Learning"
                    diff = (best_dl - best_ml) * 100
                else:
                    advantage = "Traditional ML"
                    diff = (best_ml - best_dl) * 100

                st.markdown(f"""
                <div class="info-message">
                <strong>Method Comparison</strong><br>
                {advantage} performs better<br>
                Advantage: {diff:.2f} percentage points
                </div>
                """, unsafe_allow_html=True)

    def show_prediction_interface(self):
        """Display enhanced prediction interface"""
        st.header("🎯 Model Prediction Interface")

        # Get available models
        available_models = self._get_organized_models()

        if not available_models:
            st.warning("No trained models available. Please train some models first!")
            return

        # Display model categories
        st.subheader("Available Models")

        total_models = sum(len(models) for models in available_models.values())
        st.info(f"Total available models: {total_models}")

        # Model selection
        model_categories = list(available_models.keys())
        selected_category = st.selectbox("Select Model Category", model_categories)

        if selected_category in available_models and available_models[selected_category]:
            models_in_category = available_models[selected_category]

            if selected_category == "Top 5 Traditional ML":
                st.write("**Top 5 performing traditional ML models based on CV accuracy:**")
                for i, (model_name, cv_score) in enumerate(models_in_category, 1):
                    st.write(f"{i}. {model_name}: {cv_score:.4f}")
            elif selected_category == "Ensemble Methods":
                st.write("**Individual ensemble methods:**")
                for model_name, cv_score in models_in_category:
                    st.write(f"• {model_name}: {cv_score:.4f}")
            elif selected_category == "Deep Learning":
                st.write("**Trained deep learning models:**")
                for model_name, test_score in models_in_category:
                    st.write(f"• {model_name}: {test_score:.4f} (test accuracy)")

            model_options = [f"{name} (Acc: {score:.4f})" for name, score in models_in_category]
            model_names = [name for name, score in models_in_category]

            selected_display = st.selectbox("Select Specific Model", model_options)
            selected_model = model_names[model_options.index(selected_display)]
        else:
            st.warning(f"No models available in {selected_category} category")
            return

        # Input method selection
        st.subheader("Input Method")
        input_method = st.radio("Choose Input Method", ["Upload Image", "Select from Dataset"])

        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                if st.button("Make Prediction"):
                    self.make_prediction(image, selected_model)

        else:
            # Select from dataset
            classes = getattr(Config, 'CLASSES', ['buffalo', 'elephant', 'rhino', 'zebra'])
            selected_class = st.selectbox("Select class", classes)
            data_dir = getattr(Config, 'DATA_DIR', 'african-wildlife')
            cls_path = os.path.join(data_dir, selected_class)

            if os.path.exists(cls_path):
                valid_exts = getattr(Config, 'VALID_EXTS', ['.jpg', '.jpeg', '.png'])
                files = [f for f in os.listdir(cls_path)
                        if os.path.splitext(f)[1].lower() in valid_exts]

                if files:
                    selected_image = st.selectbox("Select image", files[:20])
                    img_path = os.path.join(cls_path, selected_image)

                    image = Image.open(img_path)
                    st.image(image, caption=f"Selected from {selected_class}", use_column_width=True)

                    if st.button("Make Prediction"):
                        self.make_prediction(image, selected_model, true_class=selected_class)

    def _get_organized_models(self):
        """Get available models organized by category"""
        organized_models = {}

        # Traditional ML models
        if st.session_state.get('ml_models_trained', False) and 'ml_results' in st.session_state:
            ml_results = st.session_state.ml_results

            traditional_models = []
            ensemble_models = []

            for name, result in ml_results.items():
                if result['model'] is not None and 'cv_mean' in result:
                    cv_score = result['cv_mean']

                    if any(ensemble_name in name.lower() for ensemble_name in
                           ['voting', 'bagging', 'gradient', 'ada', 'boost']):
                        ensemble_models.append((name, cv_score))
                    else:
                        traditional_models.append((name, cv_score))

            traditional_models.sort(key=lambda x: x[1], reverse=True)
            top_5_traditional = traditional_models[:5]

            if top_5_traditional:
                organized_models["Top 5 Traditional ML"] = top_5_traditional

            if ensemble_models:
                ensemble_models.sort(key=lambda x: x[1], reverse=True)
                organized_models["Ensemble Methods"] = ensemble_models

        # Deep learning models
        deep_learning_models = []

        if st.session_state.get('cnn_trained', False) and 'cnn_results' in st.session_state:
            cnn_result = st.session_state.cnn_results
            test_score = cnn_result.get('best_accuracy', 0)
            deep_learning_models.append(("Custom_CNN", test_score))

        if st.session_state.get('resnet_trained', False) and 'resnet_results' in st.session_state:
            resnet_result = st.session_state.resnet_results
            test_score = resnet_result.get('best_accuracy', 0)
            deep_learning_models.append(("ResNet18", test_score))

        if deep_learning_models:
            deep_learning_models.sort(key=lambda x: x[1], reverse=True)
            organized_models["Deep Learning"] = deep_learning_models

        return organized_models

    def make_prediction(self, image, model_name, true_class=None):
        """Make prediction with comprehensive validation and error handling"""

        try:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Prediction Result")

                # Handle traditional ML models
                if model_name in ['SVM', 'Random Forest', 'Logistic Regression', 'KNN', 'Naive Bayes'] or \
                   any(ensemble_name in model_name for ensemble_name in
                       ['Voting', 'Bagging', 'Gradient', 'AdaBoost']):

                    if 'feature_extractor' not in st.session_state:
                        st.error("Feature extractor not available")
                        return

                    if 'ml_results' not in st.session_state:
                        st.error("ML results not available")
                        return

                    feature_extractor = st.session_state.feature_extractor
                    ml_results = st.session_state.ml_results

                    debug_mode = st.checkbox("Enable Debug Mode")

                    ml_model_name = f"ML_{model_name}"
                    prediction, probabilities, result_message = safe_make_prediction_with_debug(
                        image, ml_model_name, feature_extractor, ml_results, debug_mode
                    )

                    if prediction is None:
                        st.error(f"Prediction failed: {result_message}")
                        return

                elif model_name in ["Custom_CNN", "ResNet18"]:
                    # Deep learning prediction
                    if model_name == "Custom_CNN":
                        if 'cnn_results' not in st.session_state:
                            st.error("Custom CNN model not available")
                            return
                        model = st.session_state.cnn_results['model']
                    else:  # ResNet18
                        if 'resnet_results' not in st.session_state:
                            st.error("ResNet-18 model not available")
                            return
                        model = st.session_state.resnet_results['model']

                    # Prepare image
                    img_size = getattr(Config, 'DEEP_IMG_SIZE', 224)
                    device = getattr(Config, 'DEVICE', 'cpu')

                    val_transform = transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

                    input_tensor = val_transform(image).unsqueeze(0).to(device)

                    # Predict
                    model.eval()
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
                        prediction = outputs.argmax(1).item()

                else:
                    st.error(f"Unknown model type: {model_name}")
                    return

                # Display results
                classes = getattr(Config, 'CLASSES', ['buffalo', 'elephant', 'rhino', 'zebra'])
                predicted_class = classes[prediction]
                st.metric("Predicted Class", predicted_class.title())

                if true_class:
                    is_correct = predicted_class == true_class
                    st.metric("Result", "Correct" if is_correct else "Incorrect")

            with col2:
                if probabilities is not None:
                    st.subheader("Class Probabilities")

                    classes = getattr(Config, 'CLASSES', ['buffalo', 'elephant', 'rhino', 'zebra'])
                    prob_df = pd.DataFrame({
                        'Class': [cls.title() for cls in classes],
                        'Probability': probabilities
                    })

                    import plotly.express as px
                    fig = px.bar(prob_df, x='Class', y='Probability',
                                title="Prediction Confidence",
                                color='Probability', color_continuous_scale='viridis')
                    fig.update_traces(texttemplate='%{y:.3f}', textposition='outside')

                    # Use unique key for prediction plot
                    prediction_key = f"prediction_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
                    st.plotly_chart(fig, use_container_width=True, key=prediction_key)

                    max_prob = np.max(probabilities)
                    st.metric("Max Probability", f"{max_prob:.3f}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

    def run(self):
        """Main application runner"""

        # Display header
        self.display_header()

        # Display sidebar
        self.display_sidebar_status()

        # Navigation
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Navigation")

        page = st.sidebar.selectbox(
            "Choose a section",
            [
                "System Overview",
                "Traditional ML",
                "Enhanced Deep Learning",
                "Model Comparison",
                "Prediction Interface",
                "Ensemble Predictions"
            ],
            key="navigation"
        )

        # Route to appropriate page
        if page == "System Overview":
            self.show_system_overview()
        elif page == "Traditional ML":
            self.show_traditional_ml()
        elif page == "Enhanced Deep Learning":
            self.show_enhanced_deep_learning()
        elif page == "Model Comparison":
            self.show_model_comparison()
        elif page == "Prediction Interface":
            self.show_prediction_interface()
        elif page == "Ensemble Predictions":
            self.show_ensemble_predictions()

        # Footer
        st.sidebar.markdown("---")
        st.sidebar.info(f"Device: {Config.DEVICE}")


# Main execution
def main():
    """Main function to run the enhanced application"""

    try:
        app = WildlifeClassificationApp()
        app.run()

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)

        # Recovery options
        st.markdown("---")
        st.subheader("Recovery Options")

        if st.button("Restart Application"):
            st.experimental_rerun()

        if st.button("Clear All Data"):
            session_manager = SessionManager()
            session_manager.clear_session()
            st.success("All data cleared. Please refresh the page.")


if __name__ == "__main__":
    main()