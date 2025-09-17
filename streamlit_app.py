"""
Main Streamlit Application for African Wildlife Classification System
Updated with proper train/validation/test splitting and comprehensive pipeline management
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

# Import custom modules
try:
    from config import Config
    from session_manager import SessionManager
    from data_manager import DataManager
    from feature_engineering import FeatureExtractor, FeatureOptimizer
    from ml_models import MLModelTrainer
    from deep_learning_models import DeepLearningTrainer
    from visualization_utils import ProfessionalVisualizer
    from utils import (
        validate_feature_pipeline, debug_feature_transformation,
        display_pipeline_status, safe_make_prediction_with_debug,
        StyleManager
    )
except ImportError:
    st.error("Required modules not found. Please ensure all custom modules are properly installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Wildlife Classification System",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(StyleManager.create_custom_css(), unsafe_allow_html=True)


class WildlifeClassificationApp:
    """Main application class with proper validation strategy and pipeline management"""

    def __init__(self):
        self.session_manager = SessionManager()
        self.data_manager = DataManager()
        self.visualizer = ProfessionalVisualizer()

        # Initialize session state
        self.session_manager.initialize_session_state()

        # Load previous session if available
        if st.sidebar.button("🔄 Load Previous Session"):
            if self.session_manager.load_session():
                st.success("Previous session loaded successfully!")
                st.rerun()

        # Auto-save session periodically
        if hasattr(st.session_state, 'last_autosave'):
            if time.time() - st.session_state.last_autosave > 300:  # Every 5 minutes
                self.session_manager.save_session()
                st.session_state.last_autosave = time.time()
        else:
            st.session_state.last_autosave = time.time()

    def display_header(self):
        """Display professional header with validation strategy info"""
        st.markdown('<h1 class="main-header">🦁 African Wildlife Classification System</h1>',
                   unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #666; margin-bottom: 2rem;">Machine Learning with Proper Validation</h3>',
                   unsafe_allow_html=True)

        # Display validation strategy info
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
                validation_type = f"{Config.CV_FOLDS}-Fold CV"
            else:
                validation_type = "Train/Val/Test"
            st.markdown(f"""
            <div class="metric-container">
                <h3>{validation_type}</h3>
                <p>Validation</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            device_info = "GPU" if hasattr(Config, 'DEVICE') and Config.DEVICE.type == "cuda" else "CPU"
            st.markdown(f"""
            <div class="metric-container">
                <h3>{device_info}</h3>
                <p>Device</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            n_jobs = getattr(Config, 'N_JOBS', 'Auto')
            st.markdown(f"""
            <div class="metric-container">
                <h3>{n_jobs}</h3>
                <p>CPU Cores</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            # Show data split percentages
            if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
                split_info = f"{(1-Config.TEST_SIZE)*100:.0f}/{Config.TEST_SIZE*100:.0f}"
            else:
                train_size = getattr(Config, 'TRAIN_SIZE', 0.7)
                val_size = getattr(Config, 'VALIDATION_SIZE', 0.15)
                test_size = getattr(Config, 'TEST_SIZE', 0.15)
                split_info = f"{train_size*100:.0f}/{val_size*100:.0f}/{test_size*100:.0f}"
            st.markdown(f"""
            <div class="metric-container">
                <h3>{split_info}</h3>
                <p>Data Split %</p>
            </div>
            """, unsafe_allow_html=True)

    def display_sidebar_status(self):
        """Display comprehensive sidebar status with validation info"""
        st.sidebar.markdown("### 📊 System Status")

        status = self.session_manager.get_training_status()

        for task, completed in status.items():
            if completed:
                st.sidebar.markdown(f"✅ {task}")
            else:
                st.sidebar.markdown(f"⏳ {task}")

        # Validation strategy info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔍 Validation Strategy")

        if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
            st.sidebar.info(f"Using {Config.CV_FOLDS}-fold cross-validation")
            st.sidebar.text(f"Train+Val: {(1-Config.TEST_SIZE)*100:.0f}%")
            st.sidebar.text(f"Test: {Config.TEST_SIZE*100:.0f}%")
        else:
            st.sidebar.info("Using train/validation/test split")
            train_size = getattr(Config, 'TRAIN_SIZE', 0.7)
            val_size = getattr(Config, 'VALIDATION_SIZE', 0.15)
            test_size = getattr(Config, 'TEST_SIZE', 0.15)
            st.sidebar.text(f"Train: {train_size*100:.0f}%")
            st.sidebar.text(f"Validation: {val_size*100:.0f}%")
            st.sidebar.text(f"Test: {test_size*100:.0f}%")

        # Quick actions
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚡ Quick Actions")

        if st.sidebar.button("💾 Save Session"):
            self.session_manager.save_session()
            st.sidebar.success("Session saved!")

        if st.sidebar.button("🗑️ Clear All Data"):
            if st.sidebar.checkbox("Confirm deletion"):
                self.session_manager.clear_session()
                st.sidebar.success("All data cleared!")
                st.rerun()

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

    def show_system_overview(self):
        """Display comprehensive system overview with validation explanation"""
        st.header("🏠 System Overview")

        # Dataset check
        dataset_exists, message, class_counts = self.data_manager.check_dataset_existence()

        if not dataset_exists:
            st.error("Dataset not found! Please ensure the 'african-wildlife' directory exists with the required structure.")
            st.info("Expected structure: african-wildlife/{buffalo,elephant,rhino,zebra}/")
            return

        st.success(message)

        # Validation strategy explanation
        st.subheader("🔍 Validation Strategy")

        col1, col2 = st.columns(2)

        with col1:
            if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
                st.markdown(f"""
                <div class="info-message">
                <strong>Cross-Validation Approach</strong><br>
                • {Config.CV_FOLDS}-fold cross-validation for model selection<br>
                • Separate test set ({Config.TEST_SIZE*100:.0f}%) for final evaluation<br>
                • No data leakage - test set never used for decisions<br>
                • More robust performance estimates
                </div>
                """, unsafe_allow_html=True)
            else:
                train_size = getattr(Config, 'TRAIN_SIZE', 0.7)
                val_size = getattr(Config, 'VALIDATION_SIZE', 0.15)
                test_size = getattr(Config, 'TEST_SIZE', 0.15)
                st.markdown(f"""
                <div class="info-message">
                <strong>Train/Validation/Test Split</strong><br>
                • Training set ({train_size*100:.0f}%) for model training<br>
                • Validation set ({val_size*100:.0f}%) for model selection<br>
                • Test set ({test_size*100:.0f}%) for final evaluation<br>
                • No data leakage - clear separation of concerns
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="success-message">
            <strong>Why This Matters</strong><br>
            • Prevents overfitting to test data<br>
            • Provides honest performance estimates<br>
            • Enables proper model selection<br>
            • Follows ML best practices<br>
            • Results are scientifically valid
            </div>
            """, unsafe_allow_html=True)

        # EDA section
        st.subheader("📊 Exploratory Data Analysis")

        if st.session_state.get('eda_completed', False):
            st.success("✅ EDA completed successfully!")

            # Display EDA results if available
            if 'eda_results' in st.session_state:
                eda_results = st.session_state.eda_results

                # Show quick statistics from EDA results
                dataset_info = eda_results.get('dataset_info', {})
                total_images = dataset_info.get('total_images', 0)
                image_stats = eda_results.get('image_stats', {})
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

                # Display recommendations if available
                recommendations = eda_results.get('recommendations', [])
                if recommendations:
                    with st.expander("💡 Dataset Recommendations"):
                        for rec in recommendations:
                            st.write(f"• {rec}")

                # Note: The detailed EDA visualizations are already displayed
                # by the data_manager.perform_comprehensive_eda() method
                st.info("All detailed visualizations, sample images, and statistics are displayed above during EDA execution.")

            # Re-run EDA option
            if st.button("🔄 Re-run EDA"):
                # Clear previous EDA state
                st.session_state.eda_completed = False
                st.session_state.eda_results = None

                with st.spinner("Performing comprehensive EDA..."):
                    eda_results = self.data_manager.perform_comprehensive_eda()
                    if eda_results:
                        self.session_manager.save_session()
                        st.success("EDA re-run completed! Updated visualizations are displayed above.")
                    else:
                        st.error("EDA failed. Please check your dataset structure.")
                # Note: No st.rerun() here - let user see the new results
        else:
            st.info("EDA not completed yet")
            if st.button("🚀 Start EDA", type="primary"):
                with st.spinner("Performing comprehensive EDA..."):
                    # The perform_comprehensive_eda method now handles all visualizations internally
                    eda_results = self.data_manager.perform_comprehensive_eda()
                    if eda_results:
                        self.session_manager.save_session()
                        st.success("EDA completed! All visualizations are displayed above.")
                        # Don't rerun immediately - let user see the results
                    else:
                        st.error("EDA failed. Please check your dataset structure.")

    def show_traditional_ml(self):
        """Display traditional ML training interface with proper validation"""
        st.header("🔧 Traditional Machine Learning")

        # Check prerequisites
        if not self.data_manager.check_dataset_existence()[0]:
            st.error("Dataset not found! Please check the System Overview.")
            return

        # Validation strategy reminder
        if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
            st.info(f"Using {Config.CV_FOLDS}-fold cross-validation for model selection and hyperparameter tuning")
        else:
            st.info("Using separate validation set for model selection - test set reserved for final evaluation")

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
                help="Compare multiple feature selection methods using validation data"
            )

        with col3:
            train_ensembles = st.checkbox(
                "Train Ensemble Methods",
                value=True,
                help="Train comprehensive ensemble models"
            )

        # Display current status
        if st.session_state.get('ml_models_trained', False):
            st.success("✅ Traditional ML models already trained with proper validation!")

            if st.button("🔄 Retrain Models"):
                st.session_state.ml_models_trained = False
                st.session_state.ml_results = None
                st.rerun()

            # Display existing results
            if 'ml_results' in st.session_state and st.session_state.ml_results:
                self.display_ml_results(st.session_state.ml_results)

            return

        # Training button
        if st.button("🚀 Start Traditional ML Training", type="primary"):
            self.train_traditional_ml(use_optimized, perform_feature_selection, train_ensembles)

    def train_traditional_ml(self, use_optimized, perform_feature_selection, train_ensembles):
        """Train traditional ML models with proper train/validation/test splits and pipeline management"""

        start_time = time.time()

        try:
            # Feature extraction
            st.header("🔍 Feature Extraction")

            if 'feature_extractor' not in st.session_state or st.session_state.feature_extractor is None:
                feature_extractor = FeatureExtractor()

                with st.spinner("Extracting features with proper data splitting..."):
                    X_train, X_val, X_test, y_train, y_val, y_test = feature_extractor.prepare_dataset()

                if X_train is None:
                    st.error("Feature extraction failed!")
                    return

                if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
                    st.success(f"✅ Features extracted! Training+CV: {X_train.shape[0]}, Test: {X_test.shape[0]}")
                else:
                    st.success(f"✅ Features extracted! Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

                st.info(f"Original feature dimension: {X_train.shape[1]}")

                st.session_state.feature_extractor = feature_extractor
                st.session_state.X_train = X_train
                st.session_state.X_val = X_val
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_val = y_val
                st.session_state.y_test = y_test
            else:
                feature_extractor = st.session_state.feature_extractor
                X_train = st.session_state.X_train
                X_val = st.session_state.X_val
                X_test = st.session_state.X_test
                y_train = st.session_state.y_train
                y_val = st.session_state.y_val
                y_test = st.session_state.y_test
                st.info("✅ Using cached features")

            # Feature optimization with proper pipeline management
            X_train_final, X_val_final, X_test_final = X_train, X_val, X_test
            feature_selector = None
            pca_transformer = None

            if perform_feature_selection:
                st.header("🎯 Advanced Feature Selection")

                if 'feature_optimizer' not in st.session_state:
                    optimizer = FeatureOptimizer()

                    # Feature selection
                    X_train_selected, X_val_selected, X_test_selected, best_selector = optimizer.compare_feature_selection_methods(
                        X_train, X_val, X_test, y_train, y_val, y_test
                    )

                    # PCA analysis
                    X_train_final, X_val_final, X_test_final, final_components = optimizer.apply_pca_analysis(
                        X_train_selected, X_val_selected, X_test_selected, y_train, y_val, y_test
                    )

                    feature_selector = best_selector
                    pca_transformer = optimizer.pca_transformer

                    # Configure the feature pipeline in the extractor
                    feature_extractor.set_feature_pipeline(
                        feature_selector=feature_selector,
                        pca_transformer=pca_transformer
                    )

                    # Store optimization results
                    original_features = X_train.shape[1]
                    final_features = X_train_final.shape[1]
                    reduction_pct = (1 - final_features / original_features) * 100
                    st.session_state.feature_reduction_info = reduction_pct

                    st.info(f"✅ Feature optimization complete: {original_features} → {final_features} features ({reduction_pct:.1f}% reduction)")

                    st.session_state.feature_optimizer = optimizer
                    st.session_state.X_train_final = X_train_final
                    st.session_state.X_val_final = X_val_final
                    st.session_state.X_test_final = X_test_final
                else:
                    optimizer = st.session_state.feature_optimizer
                    X_train_final = st.session_state.X_train_final
                    X_val_final = st.session_state.X_val_final
                    X_test_final = st.session_state.X_test_final

                    # Reconfigure the pipeline from stored optimizer
                    feature_extractor.set_feature_pipeline(
                        feature_selector=optimizer.best_selector,
                        pca_transformer=optimizer.pca_transformer
                    )

                    st.info("✅ Using cached feature optimization")
            else:
                # Even without feature selection, we need to configure the pipeline
                feature_extractor.set_feature_pipeline(
                    feature_selector=None,
                    pca_transformer=None
                )

            # Display pipeline summary
            st.info(f"Feature pipeline: {feature_extractor.get_pipeline_summary()}")
            st.info(f"Expected features for prediction: {feature_extractor.get_expected_feature_count()}")

            # Model training with proper validation
            st.header("🤖 Model Training")

            trainer = MLModelTrainer()

            with st.spinner("Training individual models with proper validation..."):
                individual_results = trainer.train_individual_models(
                    X_train_final, X_val_final, X_test_final, y_train, y_val, y_test, use_optimized
                )

            # Ensemble training
            if train_ensembles:
                st.subheader("🎯 Ensemble Methods")
                with st.spinner("Training ensemble methods..."):
                    ensemble_results = trainer.train_ensemble_methods(
                        X_train_final, X_val_final, X_test_final, y_train, y_val, y_test, individual_results
                    )
                    # Combine individual and ensemble results
                    all_results = {**individual_results, **ensemble_results}
            else:
                all_results = individual_results

            # Store results with pipeline information
            st.session_state.ml_results = all_results
            st.session_state.ml_models_trained = True
            st.session_state.feature_extractor = feature_extractor  # Make sure the configured extractor is saved

            # Add training event
            training_time = time.time() - start_time
            best_test_acc = max([r['accuracy'] for r in all_results.values() if r['model'] is not None])
            self.session_manager.add_training_event(
                'Traditional ML Training',
                f"{len(all_results)} models",
                best_test_acc,
                training_time
            )

            # Display results
            self.display_ml_results(all_results)

            # Save session
            self.session_manager.save_session()

        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.exception(e)

    def display_ml_results(self, all_results):
        """Display comprehensive ML results with validation metrics"""

        st.header("📊 Training Results")

        # Initialize trainer for result display
        trainer = MLModelTrainer()

        # Display results with proper validation metrics
        best_model_info = trainer.display_results(all_results)

        if best_model_info:
            best_name, best_result = best_model_info

            # Display confusion matrix (uses test set)
            trainer.display_confusion_matrix(best_result, best_name)

            # Create comprehensive visualizations
            st.subheader("📈 Performance Visualizations")
            self.visualizer.create_model_comparison_dashboard(all_results)

        # Performance summary table
        st.subheader("📋 Detailed Performance Table")

        summary_data = []
        for name, result in all_results.items():
            if result['model'] is not None and result['accuracy'] > 0:
                row = {
                    'Model': name,
                    'Test Accuracy': f"{result['accuracy']:.4f}",
                    'Test Accuracy (%)': f"{result['accuracy']*100:.2f}%",
                    'Training Time': f"{result.get('training_time', 0):.2f}s",
                }

                # Add validation metrics if available
                if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
                    if 'cv_mean' in result:
                        row['CV Mean'] = f"{result['cv_mean']:.4f}"
                        row['CV Std'] = f"{result['cv_std']:.4f}"
                else:
                    if 'val_accuracy' in result:
                        row['Val Accuracy'] = f"{result['val_accuracy']:.4f}"

                summary_data.append(row)

        if summary_data:
            df = pd.DataFrame(summary_data)
            df = df.sort_values('Test Accuracy', ascending=False)
            st.dataframe(df, use_container_width=True)

        # Validation explanation
        st.info("Note: Test accuracy represents final unbiased performance. Validation metrics were used for model selection only.")

    def show_deep_learning(self):
        """Display deep learning training interface with proper validation"""
        st.header("🧠 Deep Learning Models")

        if not self.data_manager.check_dataset_existence()[0]:
            st.error("Dataset not found! Please check the System Overview.")
            return

        # Validation strategy reminder
        if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
            st.info("Deep learning uses validation split from training data (simplified CV)")
        else:
            st.info("Using proper train/validation/test split for deep learning")

        # Configuration
        st.subheader("⚙️ Training Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            model_choice = st.selectbox(
                "Select Model",
                ["Custom CNN", "ResNet-18 Transfer"],
                help="Choose between custom CNN or transfer learning"
            )

        with col2:
            epochs = st.slider("Training Epochs", 5, 30, 15)

        with col3:
            device = getattr(Config, 'DEVICE', 'cpu')
            batch_size = getattr(Config, 'DEEP_BATCH_SIZE', 32)
            img_size = getattr(Config, 'DEEP_IMG_SIZE', 224)
            patience = getattr(Config, 'PATIENCE', 5)

            st.markdown(f"""
            <div class="status-card">
            <strong>Device:</strong> {device}<br>
            <strong>Batch Size:</strong> {batch_size}<br>
            <strong>Image Size:</strong> {img_size}px<br>
            <strong>Early Stopping:</strong> {patience} epochs
            </div>
            """, unsafe_allow_html=True)

        # Check training status
        model_key = 'cnn_trained' if model_choice == "Custom CNN" else 'resnet_trained'
        result_key = 'cnn_results' if model_choice == "Custom CNN" else 'resnet_results'

        if st.session_state.get(model_key, False):
            st.success(f"✅ {model_choice} already trained with proper validation!")

            if st.button(f"🔄 Retrain {model_choice}"):
                st.session_state[model_key] = False
                st.session_state[result_key] = None
                st.rerun()

            # Display existing results
            if result_key in st.session_state and st.session_state[result_key]:
                self.display_deep_learning_results(st.session_state[result_key])

            return

        # Training button
        if st.button(f"🚀 Start {model_choice} Training", type="primary"):
            self.train_deep_learning_model(model_choice, epochs)

    def train_deep_learning_model(self, model_choice, epochs):
        """Train deep learning model with proper validation"""

        start_time = time.time()

        try:
            trainer = DeepLearningTrainer()

            if model_choice == "Custom CNN":
                with st.spinner("Training Custom CNN with proper validation..."):
                    results = trainer.train_custom_cnn(epochs)

                if results:
                    st.session_state.cnn_results = results
                    st.session_state.cnn_trained = True

                    # Add training event
                    training_time = time.time() - start_time
                    self.session_manager.add_training_event(
                        'Deep Learning Training',
                        'Custom CNN',
                        results['best_accuracy'],
                        training_time
                    )

                    self.display_deep_learning_results(results)

            else:  # ResNet-18
                with st.spinner("Training ResNet-18 with proper validation..."):
                    results = trainer.train_resnet_transfer(epochs)

                if results:
                    st.session_state.resnet_results = results
                    st.session_state.resnet_trained = True

                    # Add training event
                    training_time = time.time() - start_time
                    self.session_manager.add_training_event(
                        'Deep Learning Training',
                        'ResNet-18',
                        results['best_accuracy'],
                        training_time
                    )

                    self.display_deep_learning_results(results)

            # Save session
            self.session_manager.save_session()

        except Exception as e:
            st.error(f"Deep learning training failed: {str(e)}")
            st.exception(e)

    def display_deep_learning_results(self, results):
        """Display deep learning training results with validation info"""

        st.header("📊 Deep Learning Results")

        # Training summary with validation info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Test Accuracy", f"{results['best_accuracy']:.4f}")

        with col2:
            st.metric("Test Accuracy %", f"{results['best_accuracy']*100:.2f}%")

        with col3:
            st.metric("Training Epochs", results['final_epoch'])

        # Show validation accuracy if available
        if 'val_accuracy' in results:
            col4, col5 = st.columns(2)
            with col4:
                st.metric("Validation Accuracy", f"{results['val_accuracy']:.4f}")
            with col5:
                st.metric("Val-Test Gap", f"{abs(results['val_accuracy'] - results['best_accuracy']):.4f}")

        # Training curves
        st.subheader("📈 Training Progress")
        self.visualizer.create_training_progress_dashboard(results)

        # Model evaluation
        if 'model' in results:
            trainer = DeepLearningTrainer()
            eval_results = trainer.evaluate_model(results['model'], results['model_name'])

        # Validation explanation
        st.info("Note: Best accuracy represents final test set performance. Validation was used for early stopping and model selection.")

    def show_model_comparison(self):
        """Display comprehensive model comparison with proper validation context"""
        st.header("📊 Comprehensive Model Comparison")

        # Validation context
        if hasattr(Config, 'USE_CROSS_VALIDATION') and Config.USE_CROSS_VALIDATION:
            st.info("All results show final test set performance. Cross-validation was used for model selection.")
        else:
            st.info("All results show final test set performance. Validation sets were used for model selection.")

        # Collect all results
        all_results = {}

        # Traditional ML results
        if st.session_state.get('ml_models_trained', False) and 'ml_results' in st.session_state:
            ml_results = st.session_state.ml_results
            for name, result in ml_results.items():
                if result['model'] is not None and result['accuracy'] > 0:
                    all_results[f"ML_{name}"] = {
                        'accuracy': result['accuracy'],
                        'training_time': result.get('training_time', 0),
                        'model_type': 'Traditional ML'
                    }

        # Deep learning results
        if st.session_state.get('cnn_trained', False) and 'cnn_results' in st.session_state:
            cnn_result = st.session_state.cnn_results
            all_results["Deep_Custom_CNN"] = {
                'accuracy': cnn_result['best_accuracy'],
                'training_time': 0,
                'model_type': 'Deep Learning'
            }

        if st.session_state.get('resnet_trained', False) and 'resnet_results' in st.session_state:
            resnet_result = st.session_state.resnet_results
            all_results["Deep_ResNet18"] = {
                'accuracy': resnet_result['best_accuracy'],
                'training_time': 0,
                'model_type': 'Deep Learning'
            }

        if not all_results:
            st.warning("No trained models found. Please train some models first!")
            return

        # Create comprehensive comparison dashboard
        self.visualizer.create_model_comparison_dashboard(all_results)

        # Ranking table
        st.subheader("🏆 Model Performance Ranking")

        ranking_data = []
        for name, result in sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            clean_name = name.replace('ML_', '').replace('Deep_', '').replace('_', ' ')
            ranking_data.append({
                'Rank': len(ranking_data) + 1,
                'Model': clean_name,
                'Type': result['model_type'],
                'Test Accuracy': f"{result['accuracy']:.4f}",
                'Test Accuracy (%)': f"{result['accuracy']*100:.2f}%"
            })

        df = pd.DataFrame(ranking_data)
        st.dataframe(df, use_container_width=True)

        # Performance insights
        st.subheader("💡 Performance Insights")

        best_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
        ml_models = {k: v for k, v in all_results.items() if v['model_type'] == 'Traditional ML'}
        dl_models = {k: v for k, v in all_results.items() if v['model_type'] == 'Deep Learning'}

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="success-message">
            <strong>🏆 Best Overall Performance</strong><br>
            Model: {best_model[0].replace('ML_', '').replace('Deep_', '').replace('_', ' ')}<br>
            Test Accuracy: {best_model[1]['accuracy']:.4f} ({best_model[1]['accuracy']*100:.2f}%)
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
                <strong>📈 Method Comparison</strong><br>
                {advantage} performs better<br>
                Advantage: {diff:.2f} percentage points
                </div>
                """, unsafe_allow_html=True)

        # Generate report
        if st.button("📄 Generate Performance Report"):
            report_fig = self.visualizer.create_performance_report(all_results)

    def show_prediction_interface(self):
        """Display prediction interface with comprehensive validation"""
        st.header("🔮 Model Prediction Interface")

        # Check available models
        available_models = []

        if st.session_state.get('ml_models_trained', False):
            available_models.extend(['ML_SVM', 'ML_Random Forest', 'ML_Logistic Regression'])

        if st.session_state.get('cnn_trained', False):
            available_models.append('Custom_CNN')

        if st.session_state.get('resnet_trained', False):
            available_models.append('ResNet18')

        if not available_models:
            st.warning("No trained models available. Please train some models first!")
            return

        # Model selection
        selected_model = st.selectbox("Select Model for Prediction", available_models)

        # Input method
        input_method = st.radio("Choose Input Method", ["Upload Image", "Select from Dataset"])

        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

                if st.button("🎯 Make Prediction"):
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
                    st.image(image, caption=f"Selected from {selected_class}", use_container_width=True)

                    if st.button("🎯 Make Prediction"):
                        self.make_prediction(image, selected_model, true_class=selected_class)

    def make_prediction(self, image, model_name, true_class=None):
        """Make prediction on a single image with comprehensive validation and debugging"""

        try:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🎯 Prediction Result")

                if model_name.startswith("ML_"):
                    # Traditional ML prediction with comprehensive validation
                    if 'feature_extractor' not in st.session_state:
                        st.error("Feature extractor not available")
                        return

                    if 'ml_results' not in st.session_state:
                        st.error("ML results not available")
                        return

                    feature_extractor = st.session_state.feature_extractor
                    ml_results = st.session_state.ml_results

                    # Add debug toggle
                    debug_mode = st.checkbox("Enable Debug Mode", help="Show detailed pipeline information")

                    # Use the safe prediction function with validation
                    prediction, probabilities, result_message = safe_make_prediction_with_debug(
                        image, model_name, feature_extractor, ml_results, debug_mode
                    )

                    if prediction is None:
                        st.error(f"Prediction failed: {result_message}")
                        return

                else:
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

                # Display results
                classes = getattr(Config, 'CLASSES', ['buffalo', 'elephant', 'rhino', 'zebra'])
                predicted_class = classes[prediction]
                st.metric("Predicted Class", predicted_class.title())

                if true_class:
                    is_correct = predicted_class == true_class
                    st.metric("Result", "✅ Correct" if is_correct else "❌ Incorrect")

            with col2:
                if probabilities is not None:
                    st.subheader("📊 Class Probabilities")

                    classes = getattr(Config, 'CLASSES', ['buffalo', 'elephant', 'rhino', 'zebra'])
                    prob_df = pd.DataFrame({
                        'Class': [cls.title() for cls in classes],
                        'Probability': probabilities
                    })

                    # Create probability chart
                    import plotly.express as px
                    fig = px.bar(
                        prob_df,
                        x='Class',
                        y='Probability',
                        title="Prediction Confidence",
                        color='Probability',
                        color_continuous_scale='viridis'
                    )
                    fig.update_traces(texttemplate='%{y:.3f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                    # Confidence assessment
                    max_prob = np.max(probabilities)
                    if max_prob > 0.8:
                        st.success("🟢 High Confidence Prediction")
                    elif max_prob > 0.6:
                        st.info("🟡 Medium Confidence Prediction")
                    else:
                        st.warning("🔴 Low Confidence Prediction")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)

            # Additional debugging information
            if st.checkbox("Show Debug Information"):
                st.subheader("🔧 Debug Information")

                if 'feature_extractor' in st.session_state:
                    display_pipeline_status(st.session_state.feature_extractor)

                if model_name.startswith("ML_") and 'ml_results' in st.session_state:
                    model_key = model_name.replace("ML_", "")
                    if model_key in st.session_state.ml_results:
                        model_info = st.session_state.ml_results[model_key]
                        st.write(f"Model info available: {model_info is not None}")
                        if model_info and 'model' in model_info:
                            st.write(f"Model object: {model_info['model'] is not None}")
                            if hasattr(model_info['model'], 'n_features_in_'):
                                st.write(f"Model expects {model_info['model'].n_features_in_} features")

                st.write("Session state keys:")
                for key in st.session_state.keys():
                    if not key.startswith('_'):
                        st.write(f"- {key}")

    def run(self):
        """Main application runner"""

        # Display header
        self.display_header()

        # Display sidebar
        self.display_sidebar_status()

        # Navigation
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Navigation")

        page = st.sidebar.selectbox(
            "Choose a section",
            ["🏠 System Overview", "🔧 Traditional ML", "🧠 Deep Learning",
             "📊 Model Comparison", "🔮 Prediction Interface"],
            key="navigation"
        )

        # Route to appropriate page
        if page == "🏠 System Overview":
            self.show_system_overview()
        elif page == "🔧 Traditional ML":
            self.show_traditional_ml()
        elif page == "🧠 Deep Learning":
            self.show_deep_learning()
        elif page == "📊 Model Comparison":
            self.show_model_comparison()
        elif page == "🔮 Prediction Interface":
            self.show_prediction_interface()


# Main execution
def main():
    """Main function to run the application"""

    try:
        app = WildlifeClassificationApp()
        app.run()

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)

        # Provide recovery options
        st.markdown("---")
        st.subheader("🔧 Recovery Options")

        if st.button("🔄 Restart Application"):
            st.rerun()

        if st.button("🗑️ Clear All Data"):
            session_manager = SessionManager()
            session_manager.clear_session()
            st.success("All data cleared. Please refresh the page.")

if __name__ == "__main__":
    main()